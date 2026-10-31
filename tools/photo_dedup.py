"""Content-based duplicate check between an outlet's stored September photos and its current
photos (runs on a fleet runner; no model calls). Paths are blob paths in scraper-media.

Manifest line (JSONL on blob): {"key": "<place_id>", "sep": [path, ...], "cur": [path, ...]}
Result line, written to <out-base>/<shard>.jsonl:
  {"key", "dims": {path: [w, h]}, "fail": [path, ...],
   "pairs": [{"s", "c", "dist", "ar", "mad", "corr", "bmax"}, ...]}
pairs = every (sep, cur) pair with dhash distance <= 24, plus each sep photo's lowest-mad pair,
so the caller can tune thresholds without re-downloading.
  dist  dhash64 Hamming (grayscale 9x8 LANCZOS, same as gallery_select.dhash64)
  ar    relative aspect-ratio difference
  mad   mean |diff| of both images as 64x64 grayscale (0-255)
  corr  Pearson correlation of the same 64x64 arrays
  bmax  worst 8x8-block mean |diff| (catches a local change a global mean hides)
  mad256 / bmax256 / p99_256  same at 256x256 (16px blocks), only when dist <= 12
Idempotent: a shard whose result already exists is skipped.
env: MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
CONT = "https://micromarket.blob.core.windows.net/scraper-media"


def _q(var: str) -> str:
    s = os.environ.get(var, "")
    if not s:
        raise SystemExit(f"{var} missing")
    return s.split("?", 1)[1] if "?" in s else s.lstrip("?")


_TL = threading.local()


def _session() -> requests.Session:
    """18 Oct (codex review): one keep-alive session per worker thread instead of a new connection per GET."""
    s = getattr(_TL, "s", None)
    if s is None:
        s = _TL.s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4))
    return s


def get(url: str, rq: str) -> bytes | None:
    for attempt in range(6):
        try:
            r = _session().get(f"{url}?{rq}", timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:  # noqa: BLE001 - never echo the exception: it carries the signed URL
            pass
        time.sleep(min(30, 2 * 2 ** attempt) * (0.5 + random.random()))   # 18 Oct: backoff + jitter (503 under load)
    return None


def features(raw: bytes):
    im = Image.open(BytesIO(raw))
    w, h = im.size
    g = im.convert("L")
    px = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (px[:, :-1] > px[:, 1:]).flatten()
    dh = int("".join("1" if b else "0" for b in bits), 2)
    a64 = np.asarray(g.resize((64, 64), Image.LANCZOS), dtype=np.float32)
    a256 = np.asarray(g.resize((256, 256), Image.LANCZOS), dtype=np.uint8)
    return w, h, dh, a64, a256


def compare(fs, fc) -> dict:
    ws, hs, ds, a, A = fs
    wc, hc, dc, b, B = fc
    ra, rb = ws / hs, wc / hc
    d = np.abs(a - b)
    sa, sb = a.std(), b.std()
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.0
    return {"dist": (ds ^ dc).bit_count(), "ar": round(abs(ra - rb) / max(ra, rb), 4),
            "mad": round(float(d.mean()), 3), "corr": round(corr, 5),
            "bmax": round(float(d.reshape(8, 8, 8, 8).mean(axis=(1, 3)).max()), 3),
            **_fine(A, B, ds ^ dc)}


def _fine(A, B, x):
    """256x256 check (only for dhash candidates): mean |diff|, worst 16x16-px block, 99th pct pixel."""
    if x.bit_count() > 12:
        return {}
    D = np.abs(A.astype(np.int16) - B.astype(np.int16)).astype(np.float32)
    return {"mad256": round(float(D.mean()), 3), "bmax256": round(float(D.reshape(16, 16, 16, 16).mean(axis=(1, 3)).max()), 3),
            "p99_256": float(np.percentile(D, 99))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-url", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    rq, wq = _q("MEDIA_BLOB_READ_SAS"), _q("MEDIA_BLOB_WRITE_SAS")
    out_url = f"{a.out_base}/{a.shard}.jsonl"
    try:
        if requests.head(f"{out_url}?{rq}", timeout=30).status_code == 200:
            print(f"{a.shard}: result exists, skipped"); return
    except Exception:  # noqa: BLE001
        pass
    r = requests.get(f"{a.manifest_url}?{rq}", timeout=120)
    if r.status_code != 200:
        sys.exit(f"{a.shard}: manifest HTTP {r.status_code}")
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    paths = sorted({p for row in rows for p in row["sep"] + row["cur"]})
    print(f"{a.shard}: {len(rows)} outlets, {len(paths)} images", flush=True)

    def one(p):
        raw = get(f"{CONT}/{p}", rq)
        if raw is None:
            return p, None
        try:
            return p, features(raw)
        except Exception:  # noqa: BLE001
            return p, None

    t = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        F = dict(ex.map(one, paths))
    print(f"{a.shard}: fetched in {time.time() - t:.0f}s, failed {sum(v is None for v in F.values())}", flush=True)

    out = []
    for row in rows:
        pairs = []
        for s in row["sep"]:
            if F.get(s) is None:
                continue
            ps = [{"s": s, "c": c, **compare(F[s], F[c])} for c in row["cur"] if F.get(c) is not None]
            if not ps:
                continue
            best = min(ps, key=lambda x: x["mad"])
            pairs += [x for x in ps if x["dist"] <= 24 or x is best]
        out.append({"key": row["key"],
                    "dims": {p: list(F[p][:2]) for p in row["sep"] + row["cur"] if F.get(p) is not None},
                    "fail": [p for p in row["sep"] + row["cur"] if F.get(p) is None], "pairs": pairs})
    # Before the PUT (Codex 1 Nov): an all-failed result must never be stored, or the rerun after the
    # credential fix finds it ("result exists, skipped") and the shard is never compared.
    if paths and all(v is None for v in F.values()):
        sys.exit(f"{a.shard}: every image failed - check MEDIA_BLOB_READ_SAS")
    body = ("\n".join(json.dumps(o) for o in out) + "\n").encode()
    code = "none"
    for attempt in range(10):   # 18 Oct: Azure 503 (server busy) at 860 parallel jobs - retry with backoff + jitter
        try:
            pr = _session().put(f"{out_url}?{wq}", data=body, timeout=300,
                                headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "application/x-ndjson"})
            code = pr.status_code
            if code in (200, 201):
                break
        except Exception as e:  # noqa: BLE001 - never echo the exception: it carries the signed URL
            code = type(e).__name__
        time.sleep(min(60, 3 * 2 ** attempt) * (0.5 + random.random()))
    else:
        sys.exit(f"{a.shard}: result PUT failed after retries ({code})")
    print(f"{a.shard}: wrote {len(out)} outlets")


if __name__ == "__main__":
    main()
