"""Generic LLM-batch chunk builder (runs on a fleet runner).

Turns a *skeleton* shard (text-only requests + media URLs) into a full Gemini Batch input file
(images inlined as base64), uploads it to the Files API, and creates the batch job. Client-agnostic:
all prompt/schema logic lives with the client that wrote the skeleton.

Skeleton line (JSONL, one per request):
  {"key": "<id>", "prompt": "<full text prompt>", "images": ["<blob url>", ...],
   "generation_config": {...gemini generationConfig...}}

Job discovery: the batch job's display_name is "<client>/<dataset>/<shard>", so the submitting side
can reconstruct its ledger with client.batches.list() - no shared state store needed.

Quarters directive (18 Oct 2026, Kwality re-read v2): an image entry "quarters:<blob url>" with label "<n>|<label>"
expands into 5 images -- the whole photo and its 4 quarters -- from ONE fetch of the original, with exactly the labels
the v1 mirror pipeline produced ("Image n: <label> (whole photo)[ (brightened)]", "Image n, top-left quarter[ (brightened)]", ...).
Dark photos (rounded mean luminance < 0.36) are brightened first, as media_mirror.py enhance=auto. A photo that cannot be
fetched or decoded is skipped with no label at all (as v1). Other entries build exactly as before.

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS (query string or full container SAS URL)
usage: python tools/llm_batch_build.py --skeleton-url <blob url of shard jsonl> --client flora --dataset riyadh
         --shard shard_003 --model gemini-3.7-flash [--send-dim 1536] [--workers 16]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few blob cards are truncated JPEGs; decode what is there


def sas_query() -> str:
    sas = os.environ.get("MEDIA_BLOB_READ_SAS", "")
    if not sas:
        raise SystemExit("MEDIA_BLOB_READ_SAS missing")
    return sas.split("?", 1)[1] if "?" in sas else sas.lstrip("?")


BLOB_HOST = ".blob.core.windows.net"


def signed(url: str) -> str:
    """SAS only belongs on our blob URLs.

    Skeletons can now mix blob URLs (screenshots) with public CDN URLs (Google photos
    and Street View), because photos for some clients were never downloaded to blob.
    Appending a SAS to those is wrong twice over: it is a pointless query on lh3, and
    Street View URLs already carry "?panoid=", so a second "?" makes them malformed and
    every photo silently fails -- leaving a vision run that saw only screenshots.
    """
    if BLOB_HOST not in url:
        return url
    return url if "sig=" in url else f"{url}?{sas_query()}"


# Street View thumbnails 403 the requests default User-Agent; a browser UA is enough.
# Without it every streetviewpixels image is dropped by _fetch_or_none and outlets whose
# only imagery is Street View reach the model with no images at all.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
FETCH_HEADERS = {"User-Agent": UA, "Referer": "https://www.google.com/maps/"}


SAS_RE = re.compile(r"sig=[^&\s'\"]+")
_SECRETS: list[str] = []   # API key value(s) once known: masked too (genai SDK error text is the one place a key could surface)


def _mask(s: str) -> str:
    """never let a SAS signature or an API key reach a log"""
    s = SAS_RE.sub("sig=<redacted>", s)
    for k in _SECRETS:
        s = s.replace(k, "<key>")
    return s


_TL = threading.local()
RETRY_STATUS = {429, 500, 502, 503, 504}


def _http(method: str, url: str, tries: int = 5, **kw) -> requests.Response:
    """Every blob request goes through here: keep-alive Session per thread; transport errors and 429/5xx (Azure "server
    busy") retried with exponential backoff + jitter (Retry-After honoured). Returns the last response; raises only on
    transport failure, as a RuntimeError with the SAS masked and no exception chaining (no signed URL in a traceback)."""
    s = getattr(_TL, "s", None)
    if s is None:
        s = _TL.s = requests.Session()
    for attempt in range(tries):
        wait = min(30, 2 ** attempt) * random.uniform(0.5, 1.5)
        try:
            r = s.request(method, url, **kw)
            if r.status_code not in RETRY_STATUS or attempt == tries - 1:
                return r
            ra = r.headers.get("Retry-After", "")
            wait = max(wait, float(ra)) if ra.isdigit() else wait
        except requests.RequestException as exc:
            if attempt == tries - 1:
                raise RuntimeError(_mask(str(exc))[:160]) from None
        time.sleep(wait)
    raise AssertionError("unreachable")


def _http_get(url: str, tries: int = 5, **kw) -> requests.Response:
    return _http("GET", url, tries, **kw)


def fetch_raw(url: str) -> bytes:
    r = _http_get(signed(url), timeout=60, headers=FETCH_HEADERS)
    if r.status_code != 200:
        # A 4xx will not become a 200 on retry (screenshot URLs are derived, ~55% legitimately 404).
        raise RuntimeError(f"fetch failed {url}: HTTP {r.status_code}")
    return r.content


COLLAGE = "collage:"
QUARTERS = "quarters:"
# v1 quarter names (runs/2026-10-16_cooling_recheck/full_run.py QN) and crop boxes (parlour_prep.py BOX)
QN = ["top-left", "top-right", "bottom-left", "bottom-right"]
BOX = [[0, 0, .5, .5], [.5, 0, 1, .5], [0, .5, .5, 1], [.5, .5, 1, 1]]
AUTO_DARK = 0.36


# _brighten / _lum: copied verbatim from media_mirror.py (enhance=auto) so this file stays standalone.
# KEEP IDENTICAL to media_mirror.py -- Kwality v2/fleet/test_quarters.py checks the outputs match.
def _brighten(im):
    from PIL import ImageFilter, ImageOps, ImageStat
    if ImageStat.Stat(im.convert("L")).mean[0] / 255 > 0.45:
        return Image.blend(im, ImageOps.autocontrast(im, cutoff=0.5, preserve_tone=True), 0.25)
    im = ImageOps.autocontrast(im, cutoff=1, preserve_tone=True)
    mean = min(max(ImageStat.Stat(im.convert("L")).mean[0] / 255, 0.01), 0.99)
    gamma = min(1.0, max(0.45, math.log(0.47) / math.log(mean)))   # mean ** gamma ~= 0.47, clamped
    im = im.point([round(255 * (i / 255) ** gamma) for i in range(256)] * 3)
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))


def _lum(im) -> float:
    from PIL import ImageStat
    return ImageStat.Stat(im.convert("L")).mean[0] / 255


def _jpeg(im, dim: int) -> bytes:
    im.thumbnail((dim, dim))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


_RENDER = threading.BoundedSemaphore(8)   # full-size decodes at once (memory: ~8 x a few hundred MB worst case)


def render_views(raw: bytes, dim: int) -> tuple[str, list[bytes]]:
    """One original -> (" (brightened)" or "", [whole, top-left, top-right, bottom-left, bottom-right] JPEGs).
    Brightness decision as media_mirror.py enhance=auto (measured on the whole photo); quarters are cut from the
    full-size (brightened) photo; every view is resized to `dim` and JPEG-encoded once (q90)."""
    with _RENDER:
        im = Image.open(BytesIO(raw)).convert("RGB")
        br = ""
        if round(_lum(im), 3) < AUTO_DARK:          # media_mirror compares the ROUNDED value
            im, br = _brighten(im), " (brightened)"
        w, h = im.size
        quarters = [_jpeg(im.crop((round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h))), dim) for b in BOX]
        return br, [_jpeg(im, dim)] + quarters      # whole last: thumbnail works in place


def view_labels(n: int, label: str, br: str) -> list[str]:
    """exactly the v1 strings (parlour_gemini.rows())"""
    return [f"Image {n}: {label} (whole photo){br}"] + [f"Image {n}, {q} quarter{br}" for q in QN]


# (url, dim) -> Future of render_views(). Shard-wide, never evicted (an evicting cache refetched URLs and could drop
# in-flight futures). ponytail: ~0.5-0.75 GB at 500 photos x 5 views; add refcount eviction if shards grow past that.
_VIEWS: dict = {}
_VLOCK = threading.Lock()


def views(url: str, dim: int) -> tuple[str, list[bytes]]:
    """Per-process cache: a URL is fetched and rendered once, also when several threads ask for it at the same time."""
    k = (url, dim)
    with _VLOCK:
        f, mine = _VIEWS.get(k), False
        if f is None:
            f, mine = Future(), True
            _VIEWS[k] = f
    if mine:
        try:
            f.set_result(render_views(fetch_raw(url), dim))
        except BaseException as exc:  # noqa: BLE001  (never leave other waiters on an unset future)
            f.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
    return f.result()


def quarters_or_empty(url: str, n: int, label: str, dim: int) -> tuple:
    """((label, jpeg) x 5) or () -- never fails the shard: a photo that cannot be fetched or decoded is skipped entirely
    (no label), as v1 did. The 40% DROPPED gate in main() still catches a systematic failure (e.g. a bad SAS)."""
    try:
        br, imgs = views(url, dim)
    except Exception as exc:  # noqa: BLE001
        print(_mask(f"  photo dropped (quarters): {url.split('?', 1)[0].rsplit('/', 1)[-1][:52]} :: {exc}")[:200])
        DROPPED.append(url)
        return ()
    return tuple(zip(view_labels(n, label, br), imgs))


def _collage(spec: str, dim: int) -> bytes:
    """A 2x2 grid of up to 4 images, composed here so nothing has to be stored.

    Image entry "collage:<urlsafe-base64 JSON {"tiles": [url, ...], "labels": [str, ...]}>". Used for Street View:
    8 cuts of one panorama sent as 2 collages instead of 8 images (pilot 2026-09-26: boards read as well as the
    full ring, at about a fifth of the added tokens). Each tile is 768 px, labelled bottom-left (the image
    number stamp goes top-left). A tile that fails becomes a grey "unavailable" square; all four failing
    raises, so the caller drops the image like any dead CDN link.
    """
    from PIL import ImageDraw, ImageFont
    d = json.loads(base64.urlsafe_b64decode(spec.encode("ascii")).decode("utf-8"))
    tiles, labels = d.get("tiles") or [], d.get("labels") or []
    font = ImageFont.load_default(size=30)
    canvas, ok = Image.new("RGB", (1536, 1536), (60, 60, 60)), 0
    for i, url in enumerate(tiles[:4]):
        im = None
        for attempt in range(3):
            try:
                r = _http_get(signed(url), tries=1, timeout=60, headers=FETCH_HEADERS)
                if r.status_code == 200:
                    im = Image.open(BytesIO(r.content)).convert("RGB").resize((768, 768))
                    break
                if 400 <= r.status_code < 500:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2 * (attempt + 1))
        label = labels[i] if i < len(labels) else ""
        if im is None:
            im = Image.new("RGB", (768, 768), (90, 90, 90))
            label = (label + " -- unavailable").strip(" -")
        else:
            ok += 1
        if label:
            dr = ImageDraw.Draw(im)
            w = dr.textlength(label, font=font)
            dr.rectangle([0, 768 - 46, w + 20, 768], fill="black")
            dr.text((10, 768 - 42), label, fill="yellow", font=font)
        canvas.paste(im, ((i % 2) * 768, (i // 2) * 768))
    if not ok:
        raise RuntimeError("collage: every tile failed")
    canvas.thumbnail((dim, dim))
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def fetch_resized(url: str, dim: int) -> bytes:
    if url.startswith(COLLAGE):
        return _collage(url[len(COLLAGE):], dim)
    raw = fetch_raw(url)
    try:
        return _jpeg(Image.open(BytesIO(raw)).convert("RGB"), dim)
    except Exception as exc:  # noqa: BLE001  (undecodable -> dropped by _fetch_or_none)
        raise RuntimeError(f"fetch failed {url}: cannot identify / decoder: {str(exc)[:80]}") from None


DROPPED: list[str] = []   # every image we gave up on, for the sanity gate below


def _fetch_or_none(url: str, dim: int, optional: bool) -> bytes | None:
    """Never let one bad image sink a whole shard: missing optional images and undecodable images are dropped
    (logged); only repeated transport failures on a required image still raise."""
    try:
        return fetch_resized(url, dim)
    except RuntimeError as exc:
        msg = str(exc)
        # A 4xx from a public CDN is a dead photo, not a broken run: Google image URLs
        # expire (the gps-proxy ones 403 within days of a scrape), and one stale URL in
        # a 100-outlet shard must not fail every outlet in it.
        # A 4xx on OUR blob is different -- that means a bad or expired SAS, a config
        # error we want loud rather than silently missing screenshots.
        # A public CDN image is NEVER fatal. It can 404 (expired gps-proxy link), 403
        # (rate limit) or 500 (Google server error) -- one 500 out of 9,217 images killed
        # a 1,705-outlet shard, and across 3.78M fetches that would kill every shard.
        # Only OUR blob still raises: a failure there is a bad SAS, i.e. systematic.
        cdn = BLOB_HOST not in url
        if (optional or cdn or "truncated" in msg or "cannot identify" in msg
                or "decoder" in msg.lower() or "HTTP 404" in msg):
            why = "optional" if optional else "cdn unavailable" if cdn else "undecodable"
            print(_mask(f"  image dropped ({why}): {url.split('?', 1)[0].rsplit('/', 1)[-1][:52]} :: {msg}")[:200])
            DROPPED.append(url)
            return None
        raise


PROMPT_PLACEHOLDER = (
    'name: "<outlet name>"\n'
    'address: "<address>"\n'
    'google_category: "<google category>"\n'
    'rating: "<rating>"   reviews: "<review count>"   status: "<status>"')


def shard_prompt(skeleton_url: str) -> str | None:
    """Fetch <skeleton_base>/prompt.txt, the prompt stored ONCE per shard.

    Repeating a 16 KB prompt on every row makes the skeleton for 716k outlets 11.5 GB
    to upload from a laptop; stored once it is ~158 MB. The model still receives the
    full prompt -- this is upload bandwidth, not tokens.
    """
    base = skeleton_url.rsplit("/", 1)[0]
    r = _http_get(signed(f"{base}/prompt.txt"), timeout=60)
    return r.text if r.status_code == 200 else None


def shard_system(skeleton_url: str) -> str | None:
    """<skeleton_base>/system_prompt.txt -> CACHED-prompt mode (added 2026-09-17, KW Batch 2/3).

    The fixed instructions go into ONE explicit Gemini cache per project (client/dataset) and every
    request carries cachedContent + its own images + its metadata as the LAST text part. Measured: implicit
    caching gave 0 cached tokens on KW Batch 1 (per-outlet metadata sat inside the prompt); explicit caching
    inside Batch API jobs cached 4,329-4,883 tokens per request ($0.02/M instead of $0.15/M).
    """
    base = skeleton_url.rsplit("/", 1)[0]
    r = _http_get(signed(f"{base}/system_prompt.txt"), timeout=60)
    return r.text if r.status_code == 200 else None


def project_cache(client, types, model: str, display: str, system: str) -> str:
    """Reuse this project's live cache for client/dataset if it outlives a batch (>26h left), else create one.
    One cache per project, not per shard: storage bills per token-hour."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    for c in client.caches.list():
        if c.display_name == display and c.expire_time and c.expire_time - now > dt.timedelta(hours=26):
            print(f"  reusing cache {c.name} (expires {c.expire_time})")
            return c.name
    c = client.caches.create(model=model, config=types.CreateCachedContentConfig(
        system_instruction=system, display_name=display, ttl="172800s"))
    print(f"  created cache {c.name} ({getattr(c.usage_metadata, 'total_token_count', '?')} tokens, 48h)")
    # Parallel legs can race list-then-create. Converge on the OLDEST live cache; delete only our own if it lost
    # (never another leg's: it may already be referenced by a built file). The list can lag, so ours is added by hand.
    try:
        live = {x.name: x for x in client.caches.list()
                if x.display_name == display and x.expire_time and x.expire_time - now > dt.timedelta(hours=26)}
    except Exception as exc:  # noqa: BLE001
        print(f"  caches.list after create failed ({_mask(str(exc))[:80]}); keeping {c.name}")
        return c.name
    live[c.name] = c
    oldest = min(live.values(), key=lambda x: (str(getattr(x, "create_time", "") or ""), x.name))
    if oldest.name != c.name:
        print(f"  another leg created {oldest.name} first; using it and deleting ours")
        try:
            client.caches.delete(name=c.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  caches.delete({c.name}) failed: {_mask(str(exc))[:80]} (expires in 48h)")
    return oldest.name


def row_prompt(sk: dict, template: str | None) -> str:
    """Row carries either a full prompt (old shards) or just its metadata block."""
    if sk.get("prompt"):
        return sk["prompt"]
    if not template:
        raise SystemExit(f"row {sk.get(chr(39)+chr(107)+chr(101)+chr(121)+chr(39))} has no prompt and no shard prompt.txt")
    return template.replace(PROMPT_PLACEHOLDER, sk.get("meta") or "", 1)


def stamp_number(data: bytes, k: int) -> bytes:
    """Burn the image's number into its top-left corner. With 20+ images per request, labels in the
    prompt alone let per-image answers drift off by one or more (L'Oreal pilot: 20 of 24 entries
    returned, misaligned from image 10); stamped numbers brought 4 of 4 long sets back complete."""
    from PIL import ImageDraw, ImageFont
    im = Image.open(BytesIO(data)).convert("RGB")
    s = max(28, min(im.size) // 9)
    dr = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=s)
    w = dr.textlength(str(k), font=font)
    dr.rectangle([0, 0, w + s * 0.6, s * 1.4], fill="black")
    dr.text((s * 0.3, s * 0.15), str(k), fill="yellow", font=font)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def build_line(sk: dict, dim: int, pool: ThreadPoolExecutor,
               template: str | None = None, cache: str | None = None) -> tuple[bytes, int]:
    # cached mode: no prompt text up front -- images first, the row's metadata LAST (see shard_system)
    parts = [] if cache else [{"text": row_prompt(sk, template)}]
    urls = sk.get("images") or []
    optional = set(sk.get("optional_images") or [])   # e.g. a listing screenshot: skip if gone, never fail the shard
    # Optional, per skeleton line (older skeletons carry neither and build exactly as before):
    #   labels: one text part sent immediately BEFORE each image ("Image 7: photo, posted 2026-03")
    #   stamp:  burn the 1-based image number onto the pixels
    labels = sk.get("labels") or []
    stamp = bool(sk.get("stamp"))

    def one(i: int):
        u = urls[i]
        if not u.startswith(QUARTERS):
            return _fetch_or_none(u, dim, u in optional)
        if stamp or i >= len(labels) or "|" not in labels[i]:
            raise SystemExit(f"row {sk.get('key')}: a quarters: image needs a '<n>|<label>' label and stamp=false")
        n, lab = labels[i].split("|", 1)
        return quarters_or_empty(u[len(QUARTERS):], int(n), lab, dim)

    fetched = list(pool.map(one, range(len(urls))))
    sent = 0
    for i, b in enumerate(fetched):
        if isinstance(b, tuple):      # quarters directive: 5 labelled views, or nothing at all
            for lab, data in b:
                parts.append({"text": lab})
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}})
                sent += 1
            continue
        label = labels[i] if i < len(labels) else ""
        if b is None:
            # keep the numbering: a dropped image must not shift every later label and stamp
            if label:
                parts.append({"text": f"{label} -- unavailable, not sent"})
            continue
        if stamp:
            b = stamp_number(b, i + 1)
        if label:
            parts.append({"text": label})
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(b).decode("ascii")}})
        sent += 1
    if cache:
        parts.append({"text": sk.get("meta") or ""})
    if not parts:
        parts = [{"text": sk.get("meta") or ""}]
    req = {"contents": [{"role": "user", "parts": parts}], "generationConfig": sk.get("generation_config") or {}}
    if cache:
        req["cachedContent"] = cache
    return (json.dumps({"key": sk["key"], "request": req}, ensure_ascii=False) + "\n").encode("utf-8"), sent


def normalise_thinking(model: str, gen: dict, key: str) -> dict:
    """Make one model-agnostic skeleton work on any Gemini model.

    The thinking control is not portable and the mismatch is silent until collection:
    a whole batch comes back as {"code": 3, "invalid argument"} with nothing usable.
      gemini-3.1-flash-lite: thinkingBudget=0 -> 0 thoughts; thinkingLevel=low -> 116
      gemini-3.5-flash-lite: thinkingBudget=0 -> HTTP 400;   thinkingLevel=low -> 0
    So probe once per shard with a trivial request and swap the field if rejected,
    rather than hardcoding a model list that the next release invalidates.
    """
    tc = (gen or {}).get("thinkingConfig")
    if not tc:
        return gen
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    probe = {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": gen}
    r = requests.post(url, headers={"x-goog-api-key": key}, json=probe, timeout=60)
    if r.status_code == 200:
        return gen
    alt = dict(gen)
    if "thinkingBudget" in tc:
        alt["thinkingConfig"] = {"thinkingLevel": "low"}
    elif "thinkingLevel" in tc:
        alt["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        raise SystemExit(f"{model} rejected generationConfig and no thinking swap applies: {r.text[:200]}")
    probe["generationConfig"] = alt
    r2 = requests.post(url, headers={"x-goog-api-key": key}, json=probe, timeout=60)
    if r2.status_code != 200:
        raise SystemExit(f"{model} rejects both thinking forms: {r2.text[:200]}")
    print(f"  thinkingConfig {tc} rejected by {model}; using {alt['thinkingConfig']}")
    return alt


CONTAINER = "https://micromarket.blob.core.windows.net/scraper-media"   # = llm_batch_collect.py
_MK: dict = {}   # submit-marker state for this shard: base url, write SAS, run id, shard, settled


def _wq() -> str:
    for v in ("MEDIA_BLOB_WRITE_SAS", "PHASE2_SHOT_BLOB_SAS"):
        sas = os.environ.get(v, "")
        if "sig=" in sas:
            return sas.split("?", 1)[1] if "?" in sas else sas.lstrip("?")
    raise SystemExit("MEDIA_BLOB_WRITE_SAS missing (needed for the submit markers)")


def marker(kind: str, if_absent: bool = False, **fields):
    """CONTRACT_SUBMIT.md: write llm-batch/<client>/<dataset>/submit/<shard>.<kind>.json (retried). True once stored.
    if_absent (final review, 18 Oct): atomic claim with If-None-Match: * - returns "exists" when ANOTHER attempt already
    holds the marker (a retry of our own successful write is recognised by run_id + file and counts as True)."""
    import datetime as dt
    at = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")
    body = json.dumps({"run_id": _MK["run_id"], **fields, "at": at}).encode("utf-8")
    url = f"{_MK['base']}.{kind}.json?{_MK['wq']}"
    hdr = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json", **({"If-None-Match": "*"} if if_absent else {})}
    why = "?"
    for attempt in range(4):
        try:
            r = _http("PUT", url, data=body, timeout=60, headers=hdr)
            if r.status_code in (200, 201):
                print(f"  marker {kind}: {json.dumps(fields)}")
                return True
            if if_absent and r.status_code in (409, 412):   # already exists: ours (lost response) or another attempt's
                try:
                    g = _http("GET", url, timeout=60)
                    cur = json.loads(g.text) if g.status_code == 200 else {}
                except Exception:  # noqa: BLE001
                    cur = {}
                if str(cur.get("run_id")) == str(_MK["run_id"]) and all(cur.get(k) == v for k, v in fields.items()):
                    print(f"  marker {kind}: already ours (earlier write succeeded)")
                    return True
                print(f"  marker {kind} held by another attempt (run {cur.get('run_id', '?')})")
                return "exists"
            why = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001  (_http already masks)
            why = _mask(str(exc))[:120]
        time.sleep(2 * (attempt + 1))
    print(f"  !! marker {kind} NOT written after 4 tries ({why})")
    return False


def already_submitted() -> str:
    """Final review (18 Oct): the builder itself refuses a shard that may already have a batch (manual "re-run failed jobs",
    a repeated shard name, a driver bug). Anything but a clean 404 on results / intent / receipt counts as "may exist"."""
    ds = _MK["base"].rsplit("/submit/", 1)[0]
    for what, url in (("results", f"{ds}/results/{_MK['shard']}.jsonl"), ("intent", f"{_MK['base']}.intent.json"),
                      ("receipt", f"{_MK['base']}.receipt.json")):
        try:
            r = _http("HEAD", f"{url}?{_MK['wq']}", timeout=60)
            code = r.status_code
        except Exception as exc:  # noqa: BLE001  (_http already masks)
            code = type(exc).__name__
        if code != 404:
            return f"{what} -> {code}"
    return ""


def prefail(reason: str, transient: bool, msg: str):
    """Pre-submit failure: nothing was created. Marker, then a non-zero exit with no URL in it."""
    marker("prefail", reason=reason, transient=transient)
    _MK["settled"] = True
    raise SystemExit(_mask(f"{reason} {_MK['shard']}: {msg}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton-url", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--send-dim", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="batch_out")
    ap.add_argument("--key-index", default="", help="1..N -> LLM_BATCH_GEMINI_KEY_<n>")
    ap.add_argument("--max-minutes", type=float, default=105,
                    help="deadline from process start; the Files-quota wait gives up before it (build job timeout is 120)")
    a = ap.parse_args()
    t0 = time.time()
    _MK.update(base=f"{CONTAINER}/llm-batch/{a.client}/{a.dataset}/submit/{a.shard}", wq=_wq(),
               run_id=os.environ.get("GITHUB_RUN_ID", "local"), shard=a.shard, settled=False)
    seen = already_submitted()
    if seen:   # no prefail marker: it would overwrite the evidence of the earlier attempt
        _MK["settled"] = True
        raise SystemExit(f"ALREADY_SUBMITTED {a.shard}: {seen} - refusing to build (no upload, no batch)")

    # One project per key: at Tier 3 the binding limit is 20 GB of Files API storage
    # per project (~95k outlets), so a 716k run needs 8. --key-index selects which.
    key = (os.environ.get(f"LLM_BATCH_GEMINI_KEY_{a.key_index}")
           if a.key_index else None) or os.environ.get("LLM_BATCH_GEMINI_KEY")
    if not key:
        prefail("NO_KEY", False, f"set LLM_BATCH_GEMINI_KEY_{a.key_index or 1} or LLM_BATCH_GEMINI_KEY")
    _SECRETS.append(key)
    print(f"  using key slot {a.key_index or chr(39)+chr(39)} (len {len(key)})")
    from google import genai
    from google.genai import types

    try:
        r = _http_get(signed(a.skeleton_url), timeout=120)
    except RuntimeError as exc:
        prefail("SKELETON_FETCH", True, str(exc))
    if r.status_code != 200:
        prefail("SKELETON_FETCH", True, f"skeleton HTTP {r.status_code}")
    skel = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    system = shard_system(a.skeleton_url)
    template = None if (system or all(x.get("prompt") for x in skel)) else shard_prompt(a.skeleton_url)
    print(f"skeleton {a.shard}: {len(skel)} requests, {sum(len(s.get('images') or []) for s in skel)} images"
          f"{' [cached-prompt mode]' if system else ''}")
    client = genai.Client(api_key=key)
    cache = project_cache(client, types, a.model, f"{a.client}/{a.dataset}/system", system) if system else None
    # Shard-level generationConfig (<base>/generation_config.json) for rows that carry none: a response
    # schema repeated on every row would add ~5 KB x rows to the skeleton upload (2026-09-17).
    rg = _http_get(signed(a.skeleton_url.rsplit("/", 1)[0] + "/generation_config.json"), timeout=60)
    if rg.status_code == 200:
        shard_gen = rg.json()
        for s in skel:
            s.setdefault("generation_config", shard_gen)
        print(f"  shard generation_config: {sorted(shard_gen)}")
    gen0 = next((s.get("generation_config") for s in skel if s.get("generation_config")), None)
    if gen0:
        fixed = normalise_thinking(a.model, gen0, key)
        if fixed is not gen0:
            for s in skel:
                if s.get("generation_config"):
                    s["generation_config"] = fixed

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.client}_{a.dataset}_{a.shard}.jsonl"
    n_img = 0
    max_bytes = int(float(os.environ.get("LLM_BATCH_MAX_FILE_BYTES", "1.8e9")))   # Batch input max is 2 GB per file
    # Blob latency, not CPU, bounds the build: fan out across outlets (order preserved) and across each
    # outlet's images. Groups of `workers` outlets keep memory bounded (~workers x outlet bytes).
    with path.open("wb") as fh, ThreadPoolExecutor(max_workers=a.workers) as outlets, \
            ThreadPoolExecutor(max_workers=a.workers * 2) as images:
        for g in range(0, len(skel), a.workers):
            group = skel[g:g + a.workers]
            for data, k in outlets.map(lambda sk: build_line(sk, a.send_dim, images, template, cache), group):
                fh.write(data)
                n_img += k
            i = min(g + a.workers, len(skel))
            print(f"  built {i}/{len(skel)} ({n_img} images, {fh.tell() / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
            # The built file IS the upload (base64 already inlined), so its size is exact, checked as it grows (fh.tell():
            # stat() lagged by the unflushed write buffer).
            # Not transient: the same rows rebuild to about the same size (only dropped photos vary) -> split the shard.
            if fh.tell() > max_bytes:
                prefail("SHARD_TOO_BIG", False, f"{fh.tell() / 1e9:.2f} GB after {i}/{len(skel)} rows "
                        f"> {max_bytes / 1e9:.2f} GB ceiling - split this shard (nothing uploaded or submitted)")
    # Tolerating single failures must not become tolerating all of them: a shard that
    # lost most of its imagery would still "succeed" and be judged on nothing.
    want = sum(len(x.get("images") or []) for x in skel)
    if want and len(DROPPED) / want > 0.40:
        prefail("TOO_MANY_DROPPED", False, f"{len(DROPPED)}/{want} images unfetchable "
                f"({100*len(DROPPED)/want:.0f}%) -- systematic, not transient")
    print(f"images: {n_img} sent, {len(DROPPED)} dropped of {want} "
          f"({100*len(DROPPED)/max(want,1):.1f}%)")
    size = path.stat().st_size
    print(f"built {path.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f}s")

    # Files-API budget (20 GB/project): wait for the collector to free inputs of finished jobs before uploading,
    # so ALL shards can be dispatched at once and drain wave by wave unattended.
    # Concurrent runners all read the same "active" total, so the pre-check alone can overshoot Google's 20 GB cap;
    # keep headroom AND treat a 429 on upload as "wait for the collector, then retry" rather than failing the shard.
    budget = int(float(os.environ.get("LLM_BATCH_FILES_BUDGET", "17e9")))
    waited = 0
    up = None
    while up is None:
        try:
            active = sum(int(getattr(f, "size_bytes", 0) or 0) for f in client.files.list())
        except Exception as exc:  # noqa: BLE001
            print(f"files.list failed ({_mask(str(exc))[:80]}); assuming budget ok")
            active = 0
        if time.time() + 600 > t0 + a.max_minutes * 60:   # final review: never start an upload this close to the deadline
            prefail("QUOTA_TIMEOUT", True, f"less than 10 min left of --max-minutes={a.max_minutes:g} before upload "
                    f"(waited {waited // 60} min); nothing uploaded or submitted - safe to re-dispatch")
        if active + size <= budget:
            t1 = time.time()
            try:
                up = client.files.upload(file=str(path), config=types.UploadFileConfig(display_name=path.name, mime_type="jsonl"))
                break
            except Exception as exc:  # noqa: BLE001
                if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                    prefail("UPLOAD_FAILED", True, f"{type(exc).__name__}: {str(exc)[:200]}")
                print(f"upload 429 (Files quota) at active={active / 1e9:.1f} GB - waiting for collector to free inputs")
        elif waited == 0:
            print(f"Files budget: {active / 1e9:.1f} GB active + {size / 1e9:.2f} GB > {budget / 1e9:.1f} GB - waiting for collector")
        # Bounded by the job deadline (the old fixed 5h outlived the 120-min job timeout): fail cleanly while
        # nothing is uploaded or submitted, so the driver can simply re-dispatch this shard.
        if time.time() + 180 > t0 + a.max_minutes * 60:
            prefail("QUOTA_TIMEOUT", True, f"Files quota not freed within --max-minutes={a.max_minutes:g} "
                    f"(waited {waited // 60} min); nothing uploaded or submitted - safe to re-dispatch")
        time.sleep(180)
        waited += 180
    if waited:
        print(f"budget freed after {waited // 60} min")
    print(f"uploaded {up.name} in {time.time() - t1:.0f}s")

    def drop_upload():
        try:
            client.files.delete(name=up.name)
            print(f"  deleted uploaded file {up.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! files.delete({up.name}) failed: {_mask(str(exc))[:120]} (expires in 48h)")

    # CONTRACT_SUBMIT: the intent marker is what stops a re-send. No intent stored -> no create.
    seen = already_submitted()   # second check: another attempt may have started while this one built/uploaded
    if seen:
        drop_upload()
        _MK["settled"] = True
        raise SystemExit(f"ALREADY_SUBMITTED {a.shard}: {seen} (found after upload) - upload deleted, no batch created")
    claim = marker("intent", if_absent=True, file=up.name)
    if claim == "exists":   # another attempt claimed this shard between the checks and now: never create a second batch
        drop_upload()
        _MK["settled"] = True
        raise SystemExit(f"ALREADY_SUBMITTED {a.shard}: intent claimed by another attempt - upload deleted, no batch created")
    if claim is not True:
        drop_upload()
        prefail("INTENT_WRITE_FAILED", True, "intent marker could not be written; upload deleted, no batch created")
    _MK["settled"] = True
    display = f"{a.client}/{a.dataset}/{a.shard}"
    try:
        job = client.batches.create(model=a.model, src=up.name, config=types.CreateBatchJobConfig(display_name=display))
    except BaseException as exc:  # noqa: BLE001  (incl. a cancel mid-call: Google may or may not have created it)
        drop_upload()
        marker("prefail", reason="CREATE_FAILED", transient=False)
        raise SystemExit(_mask(f"CREATE_FAILED {a.shard}: {type(exc).__name__}: {exc}")[:400]) from None
    print(f"batch created {job.name} display_name={display} state={job.state}")
    if not marker("receipt", batch=job.name, file=up.name):
        print(f"!! RECEIPT NOT WRITTEN for {display} batch {job.name} - the intent marker still blocks a re-send")
    stub = {"job": job.name, "display_name": display, "src_file": up.name, "input_bytes": size, "count": len(skel),
            "keys": [s["key"] for s in skel], "images": n_img, "built_s": round(t1 - t0), "model": a.model,
            "cache": cache}
    (out / f"{a.shard}.job.json").write_text(json.dumps(stub), encoding="utf-8")
    print(json.dumps({k: v for k, v in stub.items() if k != "keys"}))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except BaseException as exc:  # noqa: BLE001  (no traceback: it could carry a signed URL)
        code = exc.code if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        if code in (None, 0):
            raise
        if _MK.get("base") and not _MK.get("settled"):   # unexpected pre-submit failure
            marker("prefail", reason="BUILD_FAILED", transient=True)
        sys.exit(_mask(str(code))[:400])
