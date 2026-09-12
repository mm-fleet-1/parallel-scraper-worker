"""Mirror expiring media URLs to our blob container (runs on a fleet runner).

Google photo URLs are NOT durable - the gps-proxy links 403 within days of a scrape
(see the comments in llm_batch_build.py). Any client deliverable that shows a photo has
to point at our own copy, so this walks a shard of (key, images) and PUTs each image to
    <dest-base>/<key>/<name>
Client-agnostic: whoever writes the manifest decides the keys, the URLs and the names.

Manifest shard line (JSONL, one per outlet, on blob):
  {"key": "<id>", "images": [{"url": "<any http url>", "name": "01_photo.jpg"}, ...]}

Result line, written back to <dest-base>/_mirrored/<shard>.jsonl:
  {"key": "<id>", "images": [{"src": "...", "url": "<durable blob url, no SAS>", "ok": true}, ...]}

The result file is the contract: a CDN URL that has already expired cannot be mirrored,
and the caller must be told which ones so it ships no dead <img>. Do NOT derive the
durable URL locally and assume it exists.

Idempotent: an image whose destination blob already exists is skipped (HEAD with the read
SAS), so a re-dispatch after a timeout costs only the HEADs. Safe to re-run.

env: MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (falls back to PHASE2_SHOT_BLOB_SAS)
usage: python tools/media_mirror.py --manifest-url <blob url of shard jsonl> \
         --dest-base <blob url prefix> --shard shard_001 [--dim 1600] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests

BLOB_HOST = ".blob.core.windows.net"
# Street View thumbnails 403 the requests default User-Agent (same fix as llm_batch_build).
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
FETCH_HEADERS = {"User-Agent": UA, "Referer": "https://www.google.com/maps/"}
CT = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _q(var: str, *fallbacks: str) -> str:
    for v in (var, *fallbacks):
        s = os.environ.get(v, "")
        if s:
            return s.split("?", 1)[1] if "?" in s else s.lstrip("?")
    raise SystemExit(f"{var} missing")


def signed(url: str, q: str) -> str:
    """SAS belongs only on our own blob URLs - appending one to a Street View URL (which
    already carries '?panoid=') makes it malformed and the image silently disappears."""
    if BLOB_HOST not in url:
        return url
    return url if "sig=" in url else f"{url}?{q}"


def fetch(url: str, rq: str, dim: int) -> bytes:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(signed(url, rq), timeout=60, headers=FETCH_HEADERS)
            if r.status_code == 200:
                return _resize(r.content, dim) if dim else r.content
            last = f"HTTP {r.status_code}"
            if 400 <= r.status_code < 500:
                break          # an expired CDN link will not become a 200 on retry
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:120]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown")


def _resize(raw: bytes, dim: int) -> bytes:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im = Image.open(BytesIO(raw)).convert("RGB")
    im.thumbnail((dim, dim))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def exists(url: str, rq: str) -> bool:
    try:
        return requests.head(f"{url}?{rq}", timeout=30).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def put(url: str, data: bytes, wq: str, ct: str) -> None:
    r = requests.put(f"{url}?{wq}", data=data,
                     headers={"x-ms-blob-type": "BlockBlob", "Content-Type": ct}, timeout=300)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"PUT HTTP {r.status_code} {r.text[:120]}")


def mirror_one(img: dict, key: str, dest_base: str, rq: str, wq: str, dim: int) -> dict:
    name = img["name"]
    dest = f"{dest_base}/{key}/{name}"
    out = {"src": img["url"], "url": dest}
    if exists(dest, rq):
        return {**out, "ok": True, "skipped": True}
    try:
        data = fetch(img["url"], rq, dim)
        put(dest, data, wq, CT.get(name.rsplit(".", 1)[-1].lower(), "application/octet-stream"))
        return {**out, "ok": True, "bytes": len(data)}
    except Exception as exc:  # noqa: BLE001
        # One dead photo must never fail the shard: expired CDN links are expected and the
        # caller drops them from the deliverable on the strength of ok=false.
        return {**out, "ok": False, "error": str(exc)[:160]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-url", required=True, help="blob url of the shard's manifest jsonl")
    ap.add_argument("--dest-base", required=True, help="blob url prefix to write <key>/<name> under")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--dim", type=int, default=0, help="longest side; 0 (default) stores bytes as fetched")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rq, wq = _q("MEDIA_BLOB_READ_SAS"), _q("MEDIA_BLOB_WRITE_SAS", "PHASE2_SHOT_BLOB_SAS")

    r = requests.get(signed(a.manifest_url, rq), timeout=120)
    if r.status_code != 200:
        sys.exit(f"manifest {a.manifest_url}: HTTP {r.status_code}")
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    jobs = [(row["key"], img) for row in rows for img in row.get("images", [])]
    print(f"{a.shard}: {len(rows):,} keys, {len(jobs):,} images", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        done = list(ex.map(lambda j: (j[0], mirror_one(j[1], j[0], a.dest_base, rq, wq, a.dim)), jobs))

    by_key: dict[str, list[dict]] = {row["key"]: [] for row in rows}
    for key, res in done:
        by_key[key].append(res)
    body = "\n".join(json.dumps({"key": k, "images": v}, ensure_ascii=False)
                     for k, v in by_key.items()).encode()
    put(f"{a.dest_base}/_mirrored/{a.shard}.jsonl", body, wq, "application/x-ndjson")

    ok = sum(1 for _, x in done if x["ok"])
    skipped = sum(1 for _, x in done if x.get("skipped"))
    print(f"{a.shard}: mirrored {ok - skipped:,} | already there {skipped:,} | failed {len(done) - ok:,}")
    for _, x in done:
        if not x["ok"]:
            print(f"  FAIL {x['src'][:90]} -> {x['error']}")
    # A shard where EVERY image failed is a config error (bad SAS, wrong dest), not dead
    # CDN links - fail loudly rather than writing an all-false result file and moving on.
    if done and ok == 0:
        sys.exit(f"{a.shard}: every image failed - check MEDIA_BLOB_WRITE_SAS and --dest-base")


if __name__ == "__main__":
    main()
