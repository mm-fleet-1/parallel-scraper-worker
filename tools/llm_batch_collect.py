"""Generic LLM-batch collector (runs on a fleet runner, on a cron).

For every Gemini batch job in this project whose display_name looks like "<client>/<dataset>/<shard>":
  * write/refresh a status stub   -> blob llm-batch/<client>/<dataset>/status/<shard>.json
  * if SUCCEEDED and not yet collected: download results -> blob .../results/<shard>.jsonl,
    then DELETE the Files-API input (frees the 20 GB per-project budget for the next wave)
  * if FAILED / EXPIRED / CANCELLED: stub carries the error; input file deleted too
Idempotent: "collected" == results blob exists (HEAD with the read SAS). Safe to run every few minutes.

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (falls back to PHASE2_SHOT_BLOB_SAS)
usage: python tools/llm_batch_collect.py [--client flora] [--dataset riyadh_p1] [--keep-inputs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

CONTAINER = "https://micromarket.blob.core.windows.net/scraper-media"
TERMINAL_BAD = {"JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"}


def _q(var: str, *fallbacks: str) -> str:
    for v in (var, *fallbacks):
        sas = os.environ.get(v, "")
        if "sig=" in sas:
            return sas.split("?", 1)[1] if "?" in sas else sas.lstrip("?")
    raise SystemExit(f"{var} missing")


def blob_exists(url: str, rq: str) -> bool:
    return requests.head(f"{url}?{rq}", timeout=30).status_code == 200


def blob_put(url: str, data: bytes, wq: str, ct: str = "application/json") -> None:
    r = requests.put(f"{url}?{wq}", data=data, headers={"x-ms-blob-type": "BlockBlob", "Content-Type": ct}, timeout=600)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"PUT {url.rsplit('/', 1)[-1]}: HTTP {r.status_code} {r.text[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client")
    ap.add_argument("--dataset")
    ap.add_argument("--wait-minutes", type=int, default=0,
                    help="poll until every matching job is terminal (0 = single sweep, the cron behaviour). "
                         "Used when chained onto llm-batch-build, where jobs are still RUNNING at first sweep.")
    ap.add_argument("--keep-inputs", action="store_true", help="do not delete Files-API inputs after collect")
    a = ap.parse_args()
    key = os.environ.get("LLM_BATCH_GEMINI_KEY") or sys.exit("LLM_BATCH_GEMINI_KEY missing")
    rq, wq = _q("MEDIA_BLOB_READ_SAS"), _q("MEDIA_BLOB_WRITE_SAS", "PHASE2_SHOT_BLOB_SAS")
    from google import genai
    client = genai.Client(api_key=key)

    deadline = time.time() + a.wait_minutes * 60
    while True:
        r = sweep(a, client, rq, wq)
        if not a.wait_minutes or r["unfinished"] == 0:
            break
        if time.time() >= deadline:
            print(f"WARNING {r['unfinished']} job(s) still running at --wait-minutes={a.wait_minutes}. "
                  f"Their results are NOT collected; re-dispatch this workflow to finish.")
            break
        print(f"  {r['unfinished']} job(s) still running; re-checking in 60s", flush=True)
        time.sleep(60)


def sweep(a, client, rq, wq) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # BatchJob does not expose its source file; the builder uploads it as "<client>_<dataset>_<shard>.jsonl",
    # so map display_name -> file name once here.
    files_by_display: dict[str, str] = {}
    try:
        for f in client.files.list():
            if getattr(f, "display_name", None):
                files_by_display[f.display_name] = f.name
    except Exception as exc:  # noqa: BLE001
        print(f"files.list failed: {str(exc)[:120]}")
    seen = collected = failed = unfinished = 0
    # A shard can be re-dispatched (new job, same display_name); only the NEWEST job per name is authoritative.
    by_name: dict[str, list] = {}
    for job in client.batches.list():
        dn = getattr(job, "display_name", "") or ""
        parts = dn.split("/")
        if len(parts) != 3:
            continue
        cl, ds, shard = parts
        if (a.client and cl != a.client) or (a.dataset and ds != a.dataset):
            continue
        by_name.setdefault(dn, []).append(job)
    for dn, jobs in by_name.items():
        jobs.sort(key=lambda j: str(getattr(j, "create_time", "")))
        job = jobs[-1]
        cl, ds, shard = dn.split("/")
        seen += 1
        try:  # list() omits src/dest/batch_stats; the full object has them
            job = client.batches.get(name=job.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  batches.get({job.name}) failed: {str(exc)[:120]}")
        base = f"{CONTAINER}/llm-batch/{cl}/{ds}"
        results_url = f"{base}/results/{shard}.jsonl"
        status_url = f"{base}/status/{shard}.json"
        state = getattr(job.state, "name", str(job.state))
        src = files_by_display.get(f"{cl}_{ds}_{shard}.jsonl")  # None once deleted (=> stub input_deleted stays True below)
        stats = getattr(job, "batch_stats", None)
        # A cancelled job reports SUCCEEDED with every request still pending and an empty responses file.
        # Treat it as CANCELLED: no results written, keys go back to the pool on the next sync.
        if state == "JOB_STATE_SUCCEEDED" and stats is not None:
            rc = int(getattr(stats, "request_count", 0) or 0)
            pc = int(getattr(stats, "pending_request_count", 0) or 0)
            if rc and pc >= rc:
                state = "JOB_STATE_CANCELLED"
        stub = {"client": cl, "dataset": ds, "shard": shard, "job": job.name, "display_name": dn, "state": state,
                "model": getattr(job, "model", None), "src_file": src,
                "create_time": str(getattr(job, "create_time", "") or ""),
                "update_time": str(getattr(job, "update_time", "") or ""),
                "request_count": getattr(stats, "request_count", None) if stats else None,
                "pending_request_count": getattr(stats, "pending_request_count", None) if stats else None,
                "checked_at": now, "results_url": None, "collected_at": None, "input_deleted": False, "error": None}
        already = blob_exists(results_url, rq)
        if state == "JOB_STATE_SUCCEEDED":
            if not already:
                dest = job.dest
                fname = getattr(dest, "file_name", None)
                if fname:
                    data = client.files.download(file=fname)
                    text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                else:
                    # INLINE job: dest.inlined_responses carry no key, so anything written
                    # here is unattributable and can never be ingested. llm_batch_inline.py
                    # publishes these itself, keyed from the per-job keymap it records at
                    # submit time -- so skip rather than write junk. (158 keyless blobs
                    # were produced this way before it was caught, 2026-08-21.)
                    print(f"  skip {dn}: inline job, published by llm_batch_inline.py")
                    continue
                blob_put(results_url, text.encode("utf-8"), wq, "application/x-ndjson")
                collected += 1
                print(f"collected {dn} -> {results_url} ({len(text) / 1e6:.1f} MB)")
            stub["results_url"] = results_url
            stub["collected_at"] = now
        elif state in TERMINAL_BAD:
            err = getattr(job, "error", None)
            stub["error"] = str(err)[:500] if err else state
            failed += 1
        if (state == "JOB_STATE_SUCCEEDED" or state in TERMINAL_BAD) and not src:
            stub["input_deleted"] = True  # no file with that name left in the project
        if (state == "JOB_STATE_SUCCEEDED" or state in TERMINAL_BAD) and src and not a.keep_inputs:
            try:
                client.files.delete(name=src)
                stub["input_deleted"] = True
            except Exception as exc:  # noqa: BLE001
                gone = "404" in str(exc) or "not found" in str(exc).lower()
                stub["input_deleted"] = gone
                if not gone:
                    stub["input_delete_error"] = str(exc)[:300]
                    print(f"  files.delete({src}) failed: {str(exc)[:200]}")
        if state != "JOB_STATE_SUCCEEDED" and state not in TERMINAL_BAD:
            unfinished += 1
        blob_put(status_url, json.dumps(stub).encode("utf-8"), wq)
        print(f"{dn:48} {state:24} pending={stub['pending_request_count']} collected={'yes' if stub['results_url'] else 'no'} input_deleted={stub['input_deleted']}")
    out = {"jobs_seen": seen, "newly_collected": collected, "failed": failed,
           "unfinished": unfinished, "checked_at": now}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
