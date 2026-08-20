"""Inline-request LLM batch builder (runs on a fleet runner).

Sibling of tools/llm_batch_build.py. Same skeleton input, same image materialisation,
but creates batch jobs with INLINE requests (src=[...]) instead of uploading an input
file to the Files API. Differences that matter:

  * no Files API  -> no 20 GB per-project budget, no upload step, no collector needed
                     to free storage
  * 20 MB cap per job -> ~4 outlets at 1536px menu cards, so --per-job is small
  * results come back attached to the job as dest.inlined_responses, NOT as an output
    file, so the normal collector finds nothing. This script collects them itself and
    writes results.jsonl, which the workflow uploads as an artifact.

Why it exists: many small jobs appear to get more concurrent request slots than a few
large ones (measured 60 jobs -> all running at once, vs 34 large jobs -> 17 concurrent
requests against a 100 ceiling). This is the machinery to test that at fleet scale.

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS
usage: python tools/llm_batch_inline.py --skeleton-url <blob url> --client flora \
         --dataset prod_p1 --shard shard_001 --model gemini-3.7-flash \
         [--per-job 1] [--limit 20] [--wait-minutes 40]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

INLINE_CAP = 18_000_000  # keep under the 20 MB inline ceiling


def sas_query() -> str:
    sas = os.environ.get("MEDIA_BLOB_READ_SAS", "")
    if not sas:
        raise SystemExit("MEDIA_BLOB_READ_SAS missing")
    return sas.split("?", 1)[1] if "?" in sas else sas.lstrip("?")


def signed(url: str) -> str:
    return url if "sig=" in url else f"{url}?{sas_query()}"


def fetch_resized(url: str, dim: int) -> bytes:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(signed(url), timeout=60)
            if r.status_code == 200:
                im = Image.open(BytesIO(r.content)).convert("RGB")
                im.thumbnail((dim, dim))
                buf = BytesIO()
                im.save(buf, "JPEG", quality=90)
                return buf.getvalue()
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:120]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def _fetch_or_none(url: str, dim: int, optional: bool) -> bytes | None:
    try:
        return fetch_resized(url, dim)
    except RuntimeError as exc:
        msg = str(exc)
        if optional or "truncated" in msg or "cannot identify" in msg or "HTTP 404" in msg:
            print(f"  image dropped: {url.rsplit('/', 1)[-1]} :: {msg[-70:]}")
            return None
        raise


def build_request(sk: dict, dim: int, pool: ThreadPoolExecutor) -> dict:
    parts = [{"text": sk["prompt"]}]
    urls = sk.get("images") or []
    optional = set(sk.get("optional_images") or [])
    for b in pool.map(lambda u: _fetch_or_none(u, dim, u in optional), urls):
        if b is not None:
            parts.append({"inlineData": {"mimeType": "image/jpeg",
                                         "data": base64.b64encode(b).decode("ascii")}})
    gc = sk.get("generation_config") or {}
    cfg = {"temperature": gc.get("temperature", 0),
           "response_mime_type": gc.get("responseMimeType", "application/json")}
    if gc.get("mediaResolution"):
        cfg["media_resolution"] = gc["mediaResolution"]
    if gc.get("maxOutputTokens"):
        cfg["max_output_tokens"] = gc["maxOutputTokens"]
    if gc.get("thinkingConfig", {}).get("thinkingLevel"):
        cfg["thinking_config"] = {"thinking_level": gc["thinkingConfig"]["thinkingLevel"]}
    return {"contents": [{"parts": parts}], "config": cfg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton-url", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--send-dim", type=int, default=1536)
    ap.add_argument("--per-job", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = whole shard")
    ap.add_argument("--wait-minutes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="batch_out")
    a = ap.parse_args()

    key = os.environ.get("LLM_BATCH_GEMINI_KEY")
    if not key:
        raise SystemExit("LLM_BATCH_GEMINI_KEY missing")
    from google import genai

    t0 = time.time()
    r = requests.get(signed(a.skeleton_url), timeout=120)
    r.raise_for_status()
    skel = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    if a.limit:
        skel = skel[:a.limit]
    print(f"skeleton {a.shard}: {len(skel)} requests, per_job={a.per_job}", flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=key)
    jobs = []  # (job_name, [keys])
    with ThreadPoolExecutor(max_workers=a.workers * 2) as images:
        for i in range(0, len(skel), a.per_job):
            group = skel[i:i + a.per_job]
            reqs = [build_request(sk, a.send_dim, images) for sk in group]
            size = len(json.dumps(reqs).encode())
            if size > INLINE_CAP:
                # Never silently drop outlets: split the group and submit the halves.
                print(f"  group {i} is {size/1e6:.1f} MB > inline cap, splitting", flush=True)
                halves = [reqs[:len(reqs) // 2 or 1], reqs[len(reqs) // 2 or 1:]]
                keysets = [[s["key"] for s in group[:len(reqs) // 2 or 1]],
                           [s["key"] for s in group[len(reqs) // 2 or 1:]]]
            else:
                halves, keysets = [reqs], [[s["key"] for s in group]]
            for part, (rq, ks) in enumerate(zip(halves, keysets)):
                if not rq:
                    continue
                j = client.batches.create(
                    model=a.model, src=rq,
                    config={"display_name": f"{a.client}/{a.dataset}/{a.shard}-{i:04d}-{part}"})
                jobs.append((j.name, ks))
            if (i // max(1, a.per_job)) % 10 == 0:
                print(f"  created {len(jobs)} jobs ({time.time() - t0:.0f}s)", flush=True)
    print(f"created {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    (out / f"{a.client}_{a.dataset}_{a.shard}.jobs.json").write_text(
        json.dumps([{"job": n, "keys": k} for n, k in jobs], indent=1), encoding="utf-8")

    # ---- collect: inline results live on the job, so poll and write them out here
    res_path = out / f"{a.client}_{a.dataset}_{a.shard}.results.jsonl"
    deadline = time.time() + a.wait_minutes * 60
    pending = dict(jobs)
    done = 0
    with res_path.open("w", encoding="utf-8") as fh:
        while pending and time.time() < deadline:
            for name in list(pending):
                b = client.batches.get(name=name)
                st = b.state.name.replace("JOB_STATE_", "")
                if st not in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"):
                    continue
                keys = pending.pop(name)
                for n, resp in enumerate(getattr(b.dest, "inlined_responses", None) or []):
                    k = keys[n] if n < len(keys) else f"{name}#{n}"
                    rec = {"key": k, "job": name, "state": st}
                    if getattr(resp, "error", None) or not getattr(resp, "response", None):
                        rec["error"] = str(getattr(resp, "error", "no response"))[:300]
                    else:
                        cand = (resp.response.candidates or [None])[0]
                        rec["raw"] = ("".join(p.text or "" for p in cand.content.parts)
                                      if cand else "")
                        u = resp.response.usage_metadata
                        rec["in_tokens"] = getattr(u, "prompt_token_count", 0) or 0
                        rec["out_tokens"] = ((getattr(u, "candidates_token_count", 0) or 0)
                                             + (getattr(u, "thoughts_token_count", 0) or 0))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done += 1
            if pending:
                print(f"  {done}/{len(jobs)} jobs done, {len(pending)} pending "
                      f"({time.time() - t0:.0f}s)", flush=True)
                time.sleep(30)
    print(f"collected {done}/{len(jobs)} jobs in {time.time() - t0:.0f}s -> {res_path.name}")
    if pending:
        print(f"WARNING {len(pending)} jobs still running at timeout; their results are NOT "
              f"in the artifact. Re-collect with batches.list(display_name prefix "
              f"{a.client}/{a.dataset}/{a.shard}).")


if __name__ == "__main__":
    main()
