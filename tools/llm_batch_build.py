"""Generic LLM-batch chunk builder (runs on a fleet runner).

Turns a *skeleton* shard (text-only requests + media URLs) into a full Gemini Batch input file
(images inlined as base64), uploads it to the Files API, and creates the batch job. Client-agnostic:
all prompt/schema logic lives with the client that wrote the skeleton.

Skeleton line (JSONL, one per request):
  {"key": "<id>", "prompt": "<full text prompt>", "images": ["<blob url>", ...],
   "generation_config": {...gemini generationConfig...}}

Job discovery: the batch job's display_name is "<client>/<dataset>/<shard>", so the submitting side
can reconstruct its ledger with client.batches.list() - no shared state store needed.

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS (query string or full container SAS URL)
usage: python tools/llm_batch_build.py --skeleton-url <blob url of shard jsonl> --client flora --dataset riyadh
         --shard shard_003 --model gemini-3.7-flash [--send-dim 1536] [--workers 16]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


def fetch_resized(url: str, dim: int) -> bytes:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(signed(url), timeout=60, headers=FETCH_HEADERS)
            if r.status_code == 200:
                im = Image.open(BytesIO(r.content)).convert("RGB")
                im.thumbnail((dim, dim))
                buf = BytesIO()
                im.save(buf, "JPEG", quality=90)
                return buf.getvalue()
            last = f"HTTP {r.status_code}"
            # A 4xx will not become a 200 on retry. Screenshot URLs are now derived
            # rather than looked up, so ~55% of them legitimately 404; retrying each
            # one four times with backoff would add ~20s per missing screenshot and
            # dominate the whole build.
            if 400 <= r.status_code < 500:
                break
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:120]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


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
            print(f"  image dropped ({why}): {url.rsplit('/', 1)[-1][:52]} :: {msg[-52:]}")
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
    r = requests.get(signed(f"{base}/prompt.txt"), timeout=60)
    return r.text if r.status_code == 200 else None


def row_prompt(sk: dict, template: str | None) -> str:
    """Row carries either a full prompt (old shards) or just its metadata block."""
    if sk.get("prompt"):
        return sk["prompt"]
    if not template:
        raise SystemExit(f"row {sk.get(chr(39)+chr(107)+chr(101)+chr(121)+chr(39))} has no prompt and no shard prompt.txt")
    return template.replace(PROMPT_PLACEHOLDER, sk.get("meta") or "", 1)


def build_line(sk: dict, dim: int, pool: ThreadPoolExecutor,
               template: str | None = None) -> tuple[bytes, int]:
    parts = [{"text": row_prompt(sk, template)}]
    urls = sk.get("images") or []
    optional = set(sk.get("optional_images") or [])   # e.g. a listing screenshot: skip if gone, never fail the shard
    imgs = [b for b in pool.map(lambda u: _fetch_or_none(u, dim, u in optional), urls) if b is not None]
    for b in imgs:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(b).decode("ascii")}})
    req = {"contents": [{"parts": parts}], "generationConfig": sk.get("generation_config") or {}}
    return (json.dumps({"key": sk["key"], "request": req}, ensure_ascii=False) + "\n").encode("utf-8"), len(imgs)


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
    a = ap.parse_args()

    # One project per key: at Tier 3 the binding limit is 20 GB of Files API storage
    # per project (~95k outlets), so a 716k run needs 8. --key-index selects which.
    key = (os.environ.get(f"LLM_BATCH_GEMINI_KEY_{a.key_index}")
           if a.key_index else None) or os.environ.get("LLM_BATCH_GEMINI_KEY")
    if not key:
        raise SystemExit(f"no key: set LLM_BATCH_GEMINI_KEY_{a.key_index or 1} "
                         f"or LLM_BATCH_GEMINI_KEY")
    print(f"  using key slot {a.key_index or chr(39)+chr(39)} (len {len(key)})")
    from google import genai
    from google.genai import types

    t0 = time.time()
    r = requests.get(signed(a.skeleton_url), timeout=120)
    r.raise_for_status()
    skel = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    template = None if all(x.get("prompt") for x in skel) else shard_prompt(a.skeleton_url)
    print(f"skeleton {a.shard}: {len(skel)} requests, {sum(len(s.get('images') or []) for s in skel)} images")
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
    # Blob latency, not CPU, bounds the build: fan out across outlets (order preserved) and across each
    # outlet's images. Groups of `workers` outlets keep memory bounded (~workers x outlet bytes).
    with path.open("wb") as fh, ThreadPoolExecutor(max_workers=a.workers) as outlets,             ThreadPoolExecutor(max_workers=a.workers * 2) as images:
        for g in range(0, len(skel), a.workers):
            group = skel[g:g + a.workers]
            for data, k in outlets.map(lambda sk: build_line(sk, a.send_dim, images, template), group):
                fh.write(data)
                n_img += k
            i = min(g + a.workers, len(skel))
            print(f"  built {i}/{len(skel)} ({n_img} images, {path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    # Tolerating single failures must not become tolerating all of them: a shard that
    # lost most of its imagery would still "succeed" and be judged on nothing.
    want = sum(len(x.get("images") or []) for x in skel)
    if want and len(DROPPED) / want > 0.40:
        raise SystemExit(f"aborting {a.shard}: {len(DROPPED)}/{want} images unfetchable "
                         f"({100*len(DROPPED)/want:.0f}%) -- systematic, not transient")
    print(f"images: {n_img} sent, {len(DROPPED)} dropped of {want} "
          f"({100*len(DROPPED)/max(want,1):.1f}%)")
    size = path.stat().st_size
    print(f"built {path.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f}s")

    client = genai.Client(api_key=key)
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
            print(f"files.list failed ({str(exc)[:80]}); assuming budget ok")
            active = 0
        if active + size <= budget:
            t1 = time.time()
            try:
                up = client.files.upload(file=str(path), config=types.UploadFileConfig(display_name=path.name, mime_type="jsonl"))
                break
            except Exception as exc:  # noqa: BLE001
                if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                    raise
                print(f"upload 429 (Files quota) at active={active / 1e9:.1f} GB - waiting for collector to free inputs")
        elif waited == 0:
            print(f"Files budget: {active / 1e9:.1f} GB active + {size / 1e9:.2f} GB > {budget / 1e9:.1f} GB - waiting for collector")
        if waited > 5 * 3600:
            raise SystemExit("gave up waiting for Files-API budget after 5h")
        time.sleep(180)
        waited += 180
    if waited:
        print(f"budget freed after {waited // 60} min")
    print(f"uploaded {up.name} in {time.time() - t1:.0f}s")
    display = f"{a.client}/{a.dataset}/{a.shard}"
    job = client.batches.create(model=a.model, src=up.name, config=types.CreateBatchJobConfig(display_name=display))
    print(f"batch created {job.name} display_name={display} state={job.state}")
    stub = {"job": job.name, "display_name": display, "src_file": up.name, "input_bytes": size, "count": len(skel),
            "keys": [s["key"] for s in skel], "images": n_img, "built_s": round(t1 - t0), "model": a.model}
    (out / f"{a.shard}.job.json").write_text(json.dumps(stub), encoding="utf-8")
    print(json.dumps({k: v for k, v in stub.items() if k != "keys"}))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
