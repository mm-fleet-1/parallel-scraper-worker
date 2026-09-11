"""Phase 2 worker — API mode (for the PUBLIC worker repo).

Same scrape loop as cloud/phase2_worker.py, but the state backend is the lease/submit
API (cloud/http_state.HttpState) instead of direct Azure SQL. It carries each pid's
`attempts` (the lease version) from /lease through to /results so the server can reject
stale submissions. Holds no DB/Drive creds — only WORKER_API_BASE + WORKER_TOKEN.

    python -m cloud.phase2_worker_api --run-id kwality_jaipur --shard 0 --max-minutes 300
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

from parallel_scraper.logging_setup import phase_var, run_id_var, setup_logging

from cloud.http_state import HttpState
from cloud.phase2_common import P2Result, _default_session_factory, _split_image_urls

logger = logging.getLogger(__name__)

# Screenshot sink, in priority order:
#  1. PHASE2_SHOT_BLOB_SAS set → PUT each PNG to the Azure Blob staging container
#     (durable queue the private drainer empties CONTINUOUSLY, even mid-run).
#  2. PHASE2_SHOT_DIR set → copy PNGs + manifest line; the workflow uploads the dir
#     as a per-shard build artifact (drained later by the desktop drainer). Also the
#     fallback when a Blob PUT fails, so screenshots are never dropped.
#  3. neither → POST bytes to the API (uploads to Drive inline; legacy slow path).
# All of this stays OFF the scrape hot path relative to Drive: Drive upload happens
# in the drainer, never on the runner.
_SHOT_DIR = os.environ.get("PHASE2_SHOT_DIR") or None
_SHOT_MANIFEST_LOCK = threading.Lock()


def _sink_shots(state, run_id, pid, attempts, shots):
    """Persist a place's screenshots — Blob staging, artifact dir, or the API,
    in that order (see module comment). Best-effort; callers wrap in try/except."""
    if not shots:
        return
    from cloud import blob_shots
    if blob_shots.blob_sas_url():
        failed = {}
        for kind, src in shots.items():
            try:
                blob_shots.put_shot(run_id, pid, kind, attempts, src)
            except Exception:
                logger.warning("phase2.shot_blob_put_failed run_id=%s pid=%s kind=%s",
                               run_id, pid, kind, exc_info=True)
                failed[kind] = src
        if not failed:
            return
        shots = failed  # fall through: stage the failures via dir/API instead
    if _SHOT_DIR:
        Path(_SHOT_DIR).mkdir(parents=True, exist_ok=True)
        manifest = Path(_SHOT_DIR) / "manifest.jsonl"
        for kind, src in shots.items():
            fname = f"{pid}_{kind}.png"
            shutil.copyfile(src, Path(_SHOT_DIR) / fname)
            with _SHOT_MANIFEST_LOCK:
                with open(manifest, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"run_id": run_id, "place_id": pid,
                                        "kind": kind, "attempts": attempts,
                                        "file": fname}) + "\n")
    else:
        state.upload_screenshots(run_id, pid, attempts, shots)


def _sink_photos(run_id, pid, image_urls):
    """Mirror place photos (CDN URLs) into Blob under <run_id>/<pid>/. Best-effort:
    never fails the scrape; no artifact fallback (the CDN URLs stay in the metadata,
    so photos can be backfilled later by re-fetching them)."""
    from cloud import blob_shots
    if not blob_shots.photos_enabled():
        return
    try:
        blob_shots.upload_place_photos(run_id, pid, image_urls)
    except Exception:
        logger.warning("phase2.photo_sink_failed run_id=%s pid=%s", run_id, pid, exc_info=True)


def _media_one(session, run_id, pid, row):
    """Dated gallery walk on the still-open panel — EVERY place: row["photo_dates"]
    = {place_photos[{url,date,kind}], imagery, errors}, all gallery items. With
    PHASE2_HORECA=1 also row["horeca"] (menu cards, mirrored to Blob). Rides the
    data JSON into place_data — no API change. Best-effort: never fails the place."""
    if not hasattr(session, "media_capture"):
        return
    try:
        m = session.media_capture(pid)
        if not m:
            return
        row["photo_dates"] = m["photo_dates"]
        hc = m.get("horeca")
        if not hc:
            return
        row["horeca"] = hc
        from cloud import blob_shots
        if blob_shots.photos_enabled() and hc.get("menu_photos"):
            try:
                blob_shots.upload_menu_photos(run_id, pid, hc["menu_photos"])
            except Exception:
                logger.warning("phase2.menu_sink_failed run_id=%s pid=%s",
                               run_id, pid, exc_info=True)
    except Exception:
        logger.warning("phase2.media_failed run_id=%s pid=%s", run_id, pid, exc_info=True)


def _panel_extras_one(session, run_id, pid):
    """Overview-pane capture (PHASE2_PANEL_EXTRAS=1). Returns a row for
    /phase2/attributes, or None when the toggle is off or nothing usable came
    back. Runs AFTER _media_one: it re-navigates to the place URL, and the
    gallery walk cannot survive that. Best-effort — these are additive fields and
    must never fail a place that already scraped fine."""
    if not hasattr(session, "panel_extras"):
        return None
    try:
        ex = session.panel_extras(pid)
    except Exception:
        logger.warning("phase2.panel_extras_failed run_id=%s pid=%s", run_id, pid,
                       exc_info=True)
        return None
    if not ex:
        return None
    row = {k: ex.get(k) for k in
           ("share_link", "price_range_text", "price_min", "price_max",
            "price_currency", "price_reported_by", "has_menu_tab",
            "popular_times", "hours", "links", "menu_items", "amenities")}
    row["place_id"] = pid
    row["menu_item_count"] = len(ex.get("menu_items") or [])
    row["has_popular_times"] = bool(ex.get("popular_times"))
    row["has_web_results"] = bool(ex.get("links"))
    row["hours_captured"] = bool(ex.get("hours"))
    return row


def _flush_attrs(state, run_id, pend_attrs):
    """Submit buffered Overview rows. Never raises: losing additive fields must
    not lose the metadata flush that rides beside it."""
    if not pend_attrs:
        return
    try:
        out = state.submit_attributes(run_id, list(pend_attrs))
        logger.info("phase2.attrs_flush rows=%d %s", len(pend_attrs), out)
    except Exception:
        logger.warning("phase2.attrs_flush_failed run_id=%s rows=%d", run_id,
                       len(pend_attrs), exc_info=True)
    pend_attrs.clear()


def _lease_with_retry(state, run_id, shard, batch, attempts: int = 5,
                      deadline: float | None = None):
    """Claim a batch, retrying on empty results. Empty lease != empty queue:
    with concurrent claimers (multiple API replicas / fleets) the READPAST
    claim can skip locked rows and return nothing while thousands remain
    queued. Backs off 15/30/45/60s (+shard jitter) before concluding dry.

    `deadline` is a time.monotonic() stamp past which backing off is pointless —
    the shard is out of budget and will exit on the next loop anyway. Without it
    a drained shard blocks ~150s at the end of every run, and any caller with no
    time left (max_minutes=0) waits out the full ladder before returning.
    """
    for i in range(attempts):
        leased = state.claim_place_ids(run_id, str(shard), int(batch))
        if leased:
            return leased
        if i + 1 >= attempts:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(min(60.0, 15.0 * (i + 1)) + (hash(str(shard)) % 7))
    return []


def _resolve_concurrency(concurrency) -> int:
    """Resolve K from the arg or PHASE2_CONCURRENCY (default 1), clamped to [1, 5]."""
    if concurrency is None:
        concurrency = os.environ.get("PHASE2_CONCURRENCY", "1")
    try:
        k = int(concurrency)
    except (TypeError, ValueError):
        k = 1
    return max(1, min(5, k))


def run_worker(run_id, shard, max_minutes=300, batch=20, state=None,
               session_factory=None, flush_interval_s=None, max_places=None,
               concurrency=None) -> P2Result:
    state = state if state is not None else HttpState()
    flush_interval_s = (
        float(os.environ.get("PHASE2_FLUSH_INTERVAL_S", "20"))
        if flush_interval_s is None else flush_interval_s
    )
    phase_var.set("phase2")
    run_id_var.set(run_id)
    if state.get_run(run_id) is None:
        raise RuntimeError(f"run_id not found / not allowed: {run_id}")

    k = _resolve_concurrency(concurrency)
    if k > 1:
        return _run_worker_concurrent(
            run_id, shard, k, max_minutes=max_minutes, batch=batch, state=state,
            session_factory=session_factory, flush_interval_s=flush_interval_s,
            max_places=max_places)

    if str(shard) == "0":
        try:
            logger.info("phase2.reclaim run_id=%s reclaimed=%d", run_id, state.reclaim_place_ids(run_id, 30))
        except Exception:
            logger.warning("phase2.reclaim_failed", exc_info=True)

    work_dir = tempfile.mkdtemp(prefix="p2api_")
    session = (session_factory or (lambda **k: _default_session_factory(work_dir)))(work_dir=work_dir)

    result = P2Result()
    started = time.monotonic()
    last_flush = started
    stop_after = max(0.0, float(max_minutes) * 60.0 - 300.0)
    recycle_every = int(os.environ.get("PHASE2_RECYCLE_EVERY", "50"))
    capture_shots = os.environ.get("PHASE2_CAPTURE_SCREENSHOTS", "1") != "0"
    pend_done: list[dict] = []
    pend_failed: list[dict] = []
    pend_attrs: list[dict] = []

    def flush():
        nonlocal last_flush
        if pend_done or pend_failed:
            out = state.submit(run_id, list(pend_done), list(pend_failed))
            acc = int(out.get("accepted", 0))
            stale = int(out.get("stale", 0))
            # accepted spans both done+failed; count locally for logging only
            result.done += len([d for d in pend_done])
            result.failed += len([f for f in pend_failed])
            logger.info("phase2.flush submitted=%d accepted=%d stale=%d done_total=%d failed_total=%d rps=%.3f",
                        len(pend_done) + len(pend_failed), acc, stale, result.done, result.failed,
                        (result.done + result.failed) / max(0.001, time.monotonic() - started))
        pend_done.clear()
        pend_failed.clear()
        _flush_attrs(state, run_id, pend_attrs)
        last_flush = time.monotonic()

    n_since_recycle = 0
    try:
        while True:
            if stop_after and (time.monotonic() - started) > stop_after:
                logger.info("phase2.time_budget_exit run_id=%s shard=%s", run_id, shard)
                break
            if max_places is not None and (result.done + result.failed + len(pend_done) + len(pend_failed)) >= max_places:
                break
            leased = _lease_with_retry(state, run_id, shard, batch,
                                       deadline=started + stop_after)
            if not leased:
                break
            for rec in leased:
                pid = str(rec["place_id"])
                attempts = int(rec["attempts"])
                try:
                    row = session.scrape(pid)
                    if not row or (row.get("name") or "") in ("", "FAILED"):
                        raise RuntimeError("blank-after-warmup / no data")
                    _media_one(session, run_id, pid, row)   # before serializing
                    urls = _split_image_urls(row.get("image_urls"))
                    pend_done.append({"place_id": pid, "attempts": attempts,
                                      "data": json.dumps(row, default=str), "image_urls": urls})
                    _ex = _panel_extras_one(session, run_id, pid)   # last: it re-navigates
                    if _ex:
                        pend_attrs.append(_ex)
                    _sink_photos(run_id, pid, urls)
                    # Page screenshots ride a separate multipart endpoint (bytes -> API ->
                    # Drive), guarded by the same lease attempts. Best-effort: a failed
                    # upload must never fail the scrape or block the metadata submit.
                    if capture_shots:
                        try:
                            shots = (session.screenshot_paths(pid)
                                     if hasattr(session, "screenshot_paths") else {})
                            _sink_shots(state, run_id, pid, attempts, shots)
                        except Exception:
                            logger.warning("phase2.shot_sink_failed run_id=%s pid=%s",
                                           run_id, pid, exc_info=True)
                except Exception as exc:
                    logger.warning("phase2.place_failed run_id=%s pid=%s", run_id, pid, exc_info=True)
                    pend_failed.append({"place_id": pid, "attempts": attempts, "error": str(exc)})
                n_since_recycle += 1
                if n_since_recycle >= recycle_every:
                    try:
                        session.recycle()
                    except Exception:
                        logger.debug("recycle_failed", exc_info=True)
                    n_since_recycle = 0
                if (time.monotonic() - last_flush) >= flush_interval_s:
                    flush()
        flush()
    finally:
        try:
            session.close()
        except Exception:
            pass
    logger.info("phase2.complete run_id=%s shard=%s done=%d failed=%d", run_id, shard, result.done, result.failed)
    return result


def _scrape_one(session, run_id, pid, attempts, capture_shots, shot_state):
    """Scrape a single pid on `session`.
    Return (done_dict_or_None, failed_dict_or_None, extras_row_or_None).

    Screenshot upload uses `shot_state` (this thread's own HttpState) and is
    best-effort: a failed upload must never fail the scrape or block the submit.
    """
    try:
        row = session.scrape(pid)
        if not row or (row.get("name") or "") in ("", "FAILED"):
            raise RuntimeError("blank-after-warmup / no data")
        _media_one(session, run_id, pid, row)   # before serializing
        urls = _split_image_urls(row.get("image_urls"))
        done = {"place_id": pid, "attempts": attempts,
                "data": json.dumps(row, default=str), "image_urls": urls}
        _sink_photos(run_id, pid, urls)
        if capture_shots:
            try:
                shots = (session.screenshot_paths(pid)
                         if hasattr(session, "screenshot_paths") else {})
                _sink_shots(shot_state, run_id, pid, attempts, shots)
            except Exception:
                logger.warning("phase2.shot_sink_failed run_id=%s pid=%s",
                               run_id, pid, exc_info=True)
        # Last, because it re-navigates the page away from the scraped panel.
        return done, None, _panel_extras_one(session, run_id, pid)
    except Exception as exc:
        logger.warning("phase2.place_failed run_id=%s pid=%s", run_id, pid, exc_info=True)
        return None, {"place_id": pid, "attempts": attempts, "error": str(exc)}, None


def _run_worker_concurrent(run_id, shard, k, max_minutes, batch, state,
                           session_factory, flush_interval_s, max_places) -> P2Result:
    """K>1 path: K worker threads, each with its own PlaywrightSession + HttpState.

    Feeder (this thread) leases pids onto a queue; workers scrape; this thread
    also flushes submitted results every flush_interval_s. Each leased pid is
    submitted exactly once (done XOR failed). K==1 never reaches here.
    """
    if str(shard) == "0":
        try:
            logger.info("phase2.reclaim run_id=%s reclaimed=%d", run_id, state.reclaim_place_ids(run_id, 30))
        except Exception:
            logger.warning("phase2.reclaim_failed", exc_info=True)

    result = P2Result()
    started = time.monotonic()
    stop_after = max(0.0, float(max_minutes) * 60.0 - 300.0)
    recycle_every = int(os.environ.get("PHASE2_RECYCLE_EVERY", "50"))
    capture_shots = os.environ.get("PHASE2_CAPTURE_SCREENSHOTS", "1") != "0"
    factory = session_factory or (lambda **kw: _default_session_factory(kw["work_dir"]))

    pend_done: list[dict] = []
    pend_failed: list[dict] = []
    pend_attrs: list[dict] = []
    lock = threading.Lock()
    work_q: "queue.Queue" = queue.Queue()  # unbounded: items are tiny tuples
    _SENTINEL = object()
    sessions: list = []
    sessions_lock = threading.Lock()
    leased_total = 0  # feeder-only

    logger.info("phase2.concurrent_start run_id=%s shard=%s k=%d", run_id, shard, k)

    def worker(idx: int):
        try:
            work_dir = tempfile.mkdtemp(prefix=f"p2api_{idx}_")
            session = factory(work_dir=work_dir)
            shot_state = HttpState() if state.__class__ is HttpState else state.__class__()
        except Exception:
            # Session/state setup failed: drain the queue marking pids failed so
            # nothing is lost, until a sentinel arrives. Never block the feeder.
            logger.warning("phase2.worker_setup_failed idx=%d", idx, exc_info=True)
            while True:
                item = work_q.get()
                if item is _SENTINEL:
                    return
                pid, attempts = item
                with lock:
                    pend_failed.append({"place_id": pid, "attempts": attempts,
                                        "error": "worker setup failed"})
            return
        with sessions_lock:
            sessions.append(session)
        n_since_recycle = 0
        while True:
            item = work_q.get()
            if item is _SENTINEL:
                return
            pid, attempts = item
            done, failed, extras = _scrape_one(session, run_id, pid, attempts,
                                               capture_shots, shot_state)
            with lock:
                if done is not None:
                    pend_done.append(done)
                else:
                    pend_failed.append(failed)
                if extras:
                    pend_attrs.append(extras)
            n_since_recycle += 1
            if n_since_recycle >= recycle_every:
                try:
                    session.recycle()
                except Exception:
                    logger.debug("recycle_failed", exc_info=True)
                n_since_recycle = 0

    def flush():
        with lock:
            done = pend_done[:]
            failed = pend_failed[:]
            attrs = pend_attrs[:]
            pend_done.clear()
            pend_failed.clear()
            pend_attrs.clear()
        _flush_attrs(state, run_id, attrs)
        if done or failed:
            try:
                out = state.submit(run_id, done, failed)
            except Exception:
                # A failed submit must not kill the feeder or drop the batch: put
                # the rows back and try again next flush. (Before: the exception
                # propagated, the feeder died into join(), and the batch — already
                # popped from pend_* — was silently lost until reclaim.)
                logger.warning("phase2.flush_failed rows=%d — requeued", len(done) + len(failed),
                               exc_info=True)
                with lock:
                    pend_done[:0] = done
                    pend_failed[:0] = failed
                return
            acc = int(out.get("accepted", 0))
            stale = int(out.get("stale", 0))
            result.done += len(done)
            result.failed += len(failed)
            logger.info("phase2.flush submitted=%d accepted=%d stale=%d done_total=%d failed_total=%d rps=%.3f",
                        len(done) + len(failed), acc, stale, result.done, result.failed,
                        (result.done + result.failed) / max(0.001, time.monotonic() - started))

    threads = [threading.Thread(target=worker, args=(i,), name=f"p2-worker-{i}", daemon=True)
               for i in range(k)]
    for t in threads:
        t.start()

    # Feeder: lease pids until empty, the time budget, or max_places; then sentinels.
    # BACKPRESSURE: keep at most a small buffer queued (~one batch). Without this the
    # feeder leases batch-after-batch into the unbounded queue and a fast-starting shard
    # marks thousands of pids 'scraping' ahead of its K workers — starving the other 19
    # shards and ballooning reclaim if it dies. We only lease another batch once the
    # queue has drained below the high-water mark.
    high_water = max(int(batch), 2 * k)
    last_flush = started
    try:
        while True:
            now = time.monotonic()
            if stop_after and (now - started) > stop_after:
                logger.info("phase2.time_budget_exit run_id=%s shard=%s", run_id, shard)
                break
            if max_places is not None and leased_total >= max_places:
                break
            # Hold off leasing while the workers still have a buffer; flush on interval.
            if work_q.qsize() >= high_water:
                if flush_interval_s and (now - last_flush) >= flush_interval_s:
                    flush()
                    last_flush = time.monotonic()
                time.sleep(0.2)
                continue
            leased = _lease_with_retry(state, run_id, shard, batch,
                                       deadline=started + stop_after)
            if not leased:
                break
            for rec in leased:
                pid = str(rec["place_id"])
                attempts = int(rec["attempts"])
                work_q.put((pid, attempts))
                leased_total += 1
            # Mid-run flush so results land while workers keep scraping.
            if flush_interval_s and (time.monotonic() - last_flush) >= flush_interval_s:
                flush()
                last_flush = time.monotonic()
    finally:
        for _ in range(k):
            work_q.put(_SENTINEL)
        # Keep flushing while the workers drain their buffer: a shard holds up to
        # high_water + k results here (~28 at k=3), which took ~10 min to appear
        # in /phase2/progress and would be lost to a runner kill.
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
            if flush_interval_s and (time.monotonic() - last_flush) >= flush_interval_s:
                flush()
                last_flush = time.monotonic()
        flush()  # final flush after every pid is accounted for
        with sessions_lock:
            for s in sessions:
                try:
                    s.close()
                except Exception:
                    pass

    logger.info("phase2.complete run_id=%s shard=%s done=%d failed=%d k=%d",
                run_id, shard, result.done, result.failed, k)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-places", type=int)
    ap.add_argument("--concurrency", type=int, default=None,
                    help="placeIds scraped at once per shard (1-5; env PHASE2_CONCURRENCY is the fallback)")
    args = ap.parse_args(argv)
    setup_logging(None, level="INFO")
    run_worker(args.run_id, args.shard, max_minutes=args.max_minutes,
               batch=args.batch, max_places=args.max_places,
               concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
