"""Generic LLM-batch collector (runs on a fleet runner, on a cron).

For every Gemini batch job in this project whose display_name looks like "<client>/<dataset>/<shard>":
  * write/refresh a status stub   -> blob llm-batch/<client>/<dataset>/status/<shard>.json
  * if SUCCEEDED and not yet collected: download results -> blob .../results/<shard>.jsonl,
    then DELETE the Files-API input (frees the 20 GB per-project budget for the next wave)
  * if FAILED / EXPIRED / CANCELLED: stub carries the error and the input file is
    KEPT (src_file + input_retained) so the job can be re-created without rebuilding
Idempotent: "collected" == results blob exists (HEAD with the read SAS). Safe to run every few minutes.

env: LLM_BATCH_GEMINI_KEY, MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (falls back to PHASE2_SHOT_BLOB_SAS)
usage: python tools/llm_batch_collect.py [--client flora] [--dataset riyadh_p1] [--keep-inputs]
       [--shards shard_0001,shard_0002 --wait-minutes 200]
--shards (18 Oct 2026): only those shards; with --wait-minutes, keep polling until EACH listed shard is collected or
terminally failed -- including shards whose batch does not exist yet (still building on another run), so a collect can
be dispatched next to its builds instead of after them. No dataset clean-up (jobs / cache) in this mode: other shards of
the dataset may still be building against the cache. Run one plain --client --dataset sweep at the very end for that.
--chained (v2 round 2, CONTRACT_SUBMIT.md): the collect job chained after a build run's matrix. A listed shard with no batch
is waited for only while llm-batch/<client>/<dataset>/submit/<shard>.intent.json may exist (any answer but a clean 404);
a shard whose build failed before submitting (no intent) is not waited for. Exits once nothing is left to wait for.
Stranded results: anything still running at --wait-minutes stays uncollected here; the driver dispatches a dataset-wide
sweep (llm-batch-collect.yml with client + dataset, no shards) at the end, which picks those up.
--key-index N: use LLM_BATCH_GEMINI_KEY_<N> (the project the build ran in), as llm_batch_build.py does.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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


SAS_RE = re.compile(r"sig=[^&\s'\"]+")


_SECRETS: list[str] = []   # the API key once known


def _mask(s: str) -> str:
    s = SAS_RE.sub("sig=<redacted>", s)
    for k in _SECRETS:
        s = s.replace(k, "<key>")
    return s


def _http(method: str, url: str, tries: int = 4, **kw) -> requests.Response:
    """All blob calls: transport errors / 429 / 5xx retried; a transport failure raises with the SAS masked, unchained."""
    for attempt in range(tries):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                return r
        except requests.RequestException as exc:
            if attempt == tries - 1:
                raise RuntimeError(f"{method} {url.split('?', 1)[0].rsplit('/', 1)[-1]}: {_mask(str(exc))[:160]}") from None
        time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def blob_status(url: str, rq: str) -> int | None:
    """HTTP status of a HEAD, None when the request itself failed."""
    try:
        return _http("HEAD", f"{url}?{rq}", timeout=30).status_code
    except RuntimeError:
        return None


def blob_exists(url: str, rq: str) -> bool:
    return blob_status(url, rq) == 200


def blob_put(url: str, data: bytes, wq: str, ct: str = "application/json", if_absent: bool = False) -> bool:
    """False only when if_absent and the blob already exists (final review: two collectors never overwrite results)."""
    hdr = {"x-ms-blob-type": "BlockBlob", "Content-Type": ct, **({"If-None-Match": "*"} if if_absent else {})}
    r = _http("PUT", f"{url}?{wq}", data=data, headers=hdr, timeout=600)
    if if_absent and r.status_code in (409, 412):
        return False
    if r.status_code not in (200, 201):
        raise RuntimeError(_mask(f"PUT {url.rsplit('/', 1)[-1]}: HTTP {r.status_code} {r.text[:160]}"))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client")
    ap.add_argument("--dataset")
    ap.add_argument("--wait-minutes", type=int, default=0,
                    help="poll until every matching job is terminal (0 = single sweep, the cron behaviour). "
                         "Used when chained onto llm-batch-build, where jobs are still RUNNING at first sweep.")
    ap.add_argument("--keep-inputs", action="store_true", help="do not delete Files-API inputs after collect")
    ap.add_argument("--shards", default="", help="comma-separated shard names (needs --client and --dataset)")
    ap.add_argument("--chained", action="store_true",
                    help="after a build run: do not wait for listed shards that never wrote an intent marker")
    ap.add_argument("--key-index", default="", help="1..N -> LLM_BATCH_GEMINI_KEY_<n> (same slot as the build)")
    a = ap.parse_args()
    a.shards = {x.strip() for x in a.shards.split(",") if x.strip()}
    if a.shards and not (a.client and a.dataset):
        sys.exit("--shards needs --client and --dataset")
    key = ((os.environ.get(f"LLM_BATCH_GEMINI_KEY_{a.key_index}") if a.key_index else None)
           or os.environ.get("LLM_BATCH_GEMINI_KEY") or sys.exit(f"LLM_BATCH_GEMINI_KEY_{a.key_index or ''} missing"))
    _SECRETS.append(key)
    print(f"  using key slot {a.key_index or chr(39) * 2}")
    rq, wq = _q("MEDIA_BLOB_READ_SAS"), _q("MEDIA_BLOB_WRITE_SAS", "PHASE2_SHOT_BLOB_SAS")
    from google import genai
    client = genai.Client(api_key=key)

    deadline = time.time() + a.wait_minutes * 60
    while True:
        r = sweep(a, client, rq, wq)
        if not a.wait_minutes or (r["unfinished"] == 0 and not r["not_submitted"]):
            break
        if time.time() >= deadline:
            print(f"WARNING {r['unfinished']} job(s) still running, {len(r['not_submitted'])} shard(s) not visible as a batch "
                  f"{sorted(r['not_submitted'])[:20]} at --wait-minutes={a.wait_minutes}. "
                  f"Their results are NOT collected here; the driver's final dataset sweep (llm-batch-collect.yml, "
                  f"client + dataset, no shards) collects them.")
            break
        pause = 30 if a.shards else 60
        print(f"  {r['unfinished']} job(s) still running, {len(r['not_submitted'])} not submitted yet; re-checking in {pause}s", flush=True)
        time.sleep(pause)


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
        print(f"files.list failed: {_mask(str(exc))[:120]}")
    seen = collected = failed = unfinished = 0
    fully_done: list[str] = []
    datasets: set = set()
    # A shard can be re-dispatched (new job, same display_name); only the NEWEST job per name is authoritative.
    by_name: dict[str, list] = {}
    for job in client.batches.list():
        dn = getattr(job, "display_name", "") or ""
        parts = dn.split("/")
        if len(parts) != 3:
            continue
        cl, ds, shard = parts
        if (a.client and cl != a.client) or (a.dataset and ds != a.dataset) or (a.shards and shard not in a.shards):
            continue
        by_name.setdefault(dn, []).append(job)
    for dn, jobs in by_name.items():
        jobs.sort(key=lambda j: str(getattr(j, "create_time", "")))
        job = jobs[-1]
        if len(jobs) > 1:   # final review: should never happen (builder refuses ALREADY_SUBMITTED) - make it loud
            print(f"!! DUPLICATE BATCHES for {dn}: {[j.name for j in jobs]} - collecting the newest only; the older ones "
                  f"are billed and need a manual collect / cancel")
        cl, ds, shard = dn.split("/")
        seen += 1
        try:  # list() omits src/dest/batch_stats; the full object has them
            job = client.batches.get(name=job.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  batches.get({job.name}) failed: {_mask(str(exc))[:120]}")
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
                if not text.strip():   # final review: never publish an empty results file
                    print(f"  !! {dn}: batch output is empty - not published, left for a later collect")
                    continue
                if blob_put(results_url, text.encode("utf-8"), wq, "application/x-ndjson", if_absent=True):
                    collected += 1
                    print(f"collected {dn} -> {results_url} ({len(text) / 1e6:.1f} MB)")
                else:
                    print(f"  {dn}: results already published by another collector - kept as is")
            stub["results_url"] = results_url
            stub["collected_at"] = now
        elif state in TERMINAL_BAD:
            err = getattr(job, "error", None)
            stub["error"] = _mask(str(err))[:500] if err else state
            failed += 1
        # Only a SUCCEEDED job is finished with its input. A FAILED job is a retry
        # candidate: Google fails batches for its own reasons (we saw
        # "Failed to open database ... within deadline", an internal code 13), and
        # deleting the input turns a free re-create into re-fetching thousands of
        # images and rebuilding a 613 MB file. Keep inputs for TERMINAL_BAD so a retry
        # can resubmit the file that already exists.
        reusable = state in TERMINAL_BAD
        if state == "JOB_STATE_SUCCEEDED" and not src:
            stub["input_deleted"] = True  # no file with that name left in the project
        if reusable and src:
            stub["input_retained"] = True
            stub["src_file"] = src
        if state == "JOB_STATE_SUCCEEDED" and src and not a.keep_inputs:
            try:
                client.files.delete(name=src)
                stub["input_deleted"] = True
            except Exception as exc:  # noqa: BLE001
                gone = "404" in _mask(str(exc)) or "not found" in _mask(str(exc)).lower()
                stub["input_deleted"] = gone
                if not gone:
                    stub["input_delete_error"] = _mask(str(exc))[:300]
                    print(f"  files.delete({src}) failed: {_mask(str(exc))[:200]}")
        if state != "JOB_STATE_SUCCEEDED" and state not in TERMINAL_BAD:
            unfinished += 1
        blob_put(status_url, json.dumps(stub).encode("utf-8"), wq)
        if state == "JOB_STATE_SUCCEEDED" and stub.get("results_url") and stub.get("input_deleted"):
            fully_done.append(job.name)
        datasets.add((cl, ds))
        print(f"{dn:48} {state:24} pending={stub['pending_request_count']} collected={'yes' if stub['results_url'] else 'no'} input_deleted={stub['input_deleted']}")
    # Cleanup (2026-09-17): once EVERY job of this client/dataset in this project is collected, delete the
    # finished batch jobs (their results + status stubs are on blob) and the cached-prompt-mode cache, which
    # bills storage per hour. Failed jobs are kept (retry candidates), so nothing is deleted while any exist.
    cleaned = {"jobs_deleted": 0, "caches_deleted": 0}
    if a.client and a.dataset and unfinished == 0 and failed == 0 and seen and len(fully_done) == seen \
            and not a.keep_inputs and not a.shards:
        for name in fully_done:
            try:
                client.batches.delete(name=name)
                cleaned["jobs_deleted"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  batches.delete({name}) failed: {_mask(str(exc))[:120]}")
        try:
            for c in client.caches.list():
                if c.display_name == f"{a.client}/{a.dataset}/system":
                    client.caches.delete(name=c.name)
                    cleaned["caches_deleted"] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  cache cleanup failed: {_mask(str(exc))[:120]}")
    not_submitted = a.shards - {dn.split("/")[2] for dn in by_name}
    if a.chained:   # wait only for shards that may still have a batch: results absent AND intent not a clean 404
        base = f"{CONTAINER}/llm-batch/{a.client}/{a.dataset}"
        not_submitted = {s for s in not_submitted if not blob_exists(f"{base}/results/{s}.jsonl", rq)
                         and blob_status(f"{base}/submit/{s}.intent.json", rq) != 404}
    out = {"jobs_seen": seen, "newly_collected": collected, "failed": failed,
           "unfinished": unfinished, "not_submitted": sorted(not_submitted), "checked_at": now, **cleaned}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except Exception as exc:  # noqa: BLE001  (no traceback: it could carry a signed URL)
        sys.exit(_mask(f"{type(exc).__name__}: {exc}")[:400])
