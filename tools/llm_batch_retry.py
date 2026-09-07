"""Re-create batch jobs that FAILED, reusing the input file already in the Files API.

Google fails batches for its own reasons -- we saw code 13 "Failed to open database:
/span/global/labs-genai-api-spanner:prod within deadline", with all 1,705 requests still
pending. Nothing about the input was wrong. Rebuilding it would mean re-fetching ~8,700
images and re-uploading 613 MB for a fault that was never ours.

llm_batch_collect now keeps the input file for TERMINAL_BAD jobs (input_retained +
src_file in the status blob), so a retry is just batches.create against the same file.

usage:
  python tools/llm_batch_retry.py --client kwality --dataset prod_v23 [--model M] [--dry-run]

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

BLOB = "https://micromarket.blob.core.windows.net/scraper-media"
CONTAINER = BLOB


def q(name):
    v = os.environ.get(name, "")
    if not v:
        raise SystemExit(f"{name} missing")
    return v.split("?", 1)[1] if "?" in v else v.lstrip("?")


def list_status(client, dataset, rq):
    prefix = f"llm-batch/{client}/{dataset}/status/"
    out, marker = [], ""
    while True:
        r = requests.get(f"{CONTAINER}?restype=container&comp=list&prefix={prefix}"
                         f"&maxresults=1000{marker}&{rq}", timeout=180)
        if r.status_code != 200:
            return out
        out += [n for n in re.findall(r"<Name>([^<]+)</Name>", r.text) if n.endswith(".json")]
        m = re.search(r"<NextMarker>([^<]*)</NextMarker>", r.text)
        if not (m and m.group(1)):
            return out
        marker = "&marker=" + m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("LLM_BATCH_GEMINI_KEY")
    if not key:
        raise SystemExit("LLM_BATCH_GEMINI_KEY missing")
    rq, wq = q("MEDIA_BLOB_READ_SAS"), q("MEDIA_BLOB_WRITE_SAS")
    from google import genai
    client = genai.Client(api_key=key)

    names = list_status(client, a.dataset, rq)
    print(f"{len(names)} status blobs for {a.client}/{a.dataset}")
    retried = skipped = lost = 0
    for name in sorted(names):
        url = f"{BLOB}/{name}"
        r = requests.get(f"{url}?{rq}", timeout=120)
        if r.status_code != 200:
            continue
        st = r.json()
        if st.get("state") not in ("JOB_STATE_FAILED", "JOB_STATE_EXPIRED",
                                   "JOB_STATE_CANCELLED"):
            skipped += 1
            continue
        src = st.get("src_file")
        if not src:
            # collected before the keep-inputs fix, or the file really is gone: this one
            # has to go back through the builder.
            print(f"  {st.get('shard')}: input gone -- needs a full rebuild")
            lost += 1
            continue
        # confirm the file is still there before claiming a retry
        try:
            client.files.get(name=src)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {st.get('shard')}: {src} not retrievable ({str(exc)[:70]}) -- rebuild")
            lost += 1
            continue
        if a.dry_run:
            print(f"  {st.get('shard')}: would re-create from {src}")
            retried += 1
            continue
        job = client.batches.create(
            model=a.model or st.get("model", "").replace("models/", ""), src=src,
            config={"display_name": st.get("display_name")})
        st.update({"job": job.name, "state": "JOB_STATE_PENDING", "error": None,
                   "retried_from": st.get("job"), "results_url": None,
                   "collected_at": None})
        requests.put(f"{url}?{wq}", data=json.dumps(st).encode(), timeout=120,
                     headers={"x-ms-blob-type": "BlockBlob",
                              "Content-Type": "application/json"})
        print(f"  {st.get('shard')}: re-created as {job.name}")
        retried += 1
    print(f"\nretried {retried}, still-fine {skipped}, need rebuild {lost}")


if __name__ == "__main__":
    main()
