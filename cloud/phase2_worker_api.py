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
from datetime import datetime, timezone
from pathlib import Path

from parallel_scraper.logging_setup import phase_var, run_id_var, setup_logging
from parallel_scraper.normalise import null_if_sentinel

from cloud import delisted, staging
from cloud.http_state import HttpState
from cloud.phase2_common import P2Result, _default_session_factory, _split_image_urls
_RECLAIM_MIN = int(os.environ.get("PHASE2_RECLAIM_MINUTES", "30") or "30")

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
    """LEGACY (PHASE2_LEGACY_PANEL_PHOTOS=1 only; the gallery flow is _photos_one).
    Mirror place photos (CDN URLs) into Blob under <run_id>/<pid>/. Best-effort:
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


_TL = threading.local()


def _http_get(url: str, timeout: float) -> bytes | None:
    """GET a CDN image with one requests.Session per thread (§9.1). None unless 200 + body."""
    import requests
    s = getattr(_TL, "http", None)
    if s is None:
        s = _TL.http = requests.Session()
    r = s.get(url, timeout=timeout)
    return r.content if r.status_code == 200 and r.content else None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ms(a: float, b: float) -> int:
    return int((b - a) * 1000)


# Per-place phase ms, in the TIMING line and place_data $.timing; 0 = phase did not run.
# goto/panel_ready/extract/reviews come from session.scrape (row["timing"]); walk_* and
# degraded_retry from walk_meta.timings. submit_wait: 0, not measured ($.timing is
# serialized before the POST it would time).
TIMING_KEYS = ("goto", "panel_ready", "extract", "reviews", "degraded_retry", "walk_owner",
               "walk_latest", "walk_all", "thumbs", "mirror", "submit_wait", "total")


def _place_timing(scrape_tm: dict | None, walk_tm: dict | None, tm: dict, total: int) -> dict:
    t = dict.fromkeys(TIMING_KEYS, 0)
    t.update({k: int(v) for k, v in (scrape_tm or {}).items() if k in t})
    for chip, ms in (walk_tm or {}).items():
        c = chip.lower()
        k = ("degraded_retry" if c == "degraded_retry" else "walk_owner" if c.startswith("by owner")
             else "walk_latest" if c.startswith("latest") else "walk_all")   # All, Menu, current
        t[k] += int(ms)
    t.update(thumbs=int(tm["thumbs"]), mirror=int(tm["mirror"]), total=total)
    return t


def _timing_line(pid, t: dict) -> str:
    """Legacy keys first, contiguous and in their old order (p200 a1.py regex), then the rest."""
    legacy = {"panel": t["goto"] + t["panel_ready"] + t["extract"] + t["reviews"],
              "reviews": t["reviews"],
              "walk": t["walk_owner"] + t["walk_latest"] + t["walk_all"],
              "thumbs": t["thumbs"], "mirror": t["mirror"], "total": t["total"]}
    rest = {k: t[k] for k in TIMING_KEYS if k not in legacy}
    return f"TIMING pid={pid} " + " ".join(f"{k}={v}" for k, v in {**legacy, **rest}.items())


def _full_url(it: dict) -> str | None:
    """Google CDN URL of a walked item: the viewer's own image URL (`!6s`, see
    horeca_capture), else the lh3 path of an AF1Qip photo id from the viewer key
    (`!1s`). mirror_selected rewrites either to =s1600."""
    key = it.get("viewer_key") or ""
    if not it.get("url") and key.startswith("AF1Qip"):
        return f"https://lh3.googleusercontent.com/p/{key}"
    return it.get("url")


def _photos_one(run_id, pid, row, panel_urls) -> tuple[list[dict], dict]:
    """Design §1 steps 2-4: select_photos -> mirror_selected -> place_media rows, and
    row["photo_dates"]["summary"] (§4.3). Returns (media, {thumbs, mirror} ms).
    Never raises: an error here is recorded as a failed walk, so the repair pass
    (media_partial=1) picks the place up again."""
    from cloud import blob_shots
    from parallel_scraper import gallery_select as gs
    tm = {"thumbs": 0, "mirror": 0}
    pd = row.get("photo_dates") or {}
    items = pd.get("items") or []
    wm = pd.get("walk_meta") or {"walk_ok": False, "reason": "no_capture"}
    selected, outcomes, media, cands, sv = [], None, [], [], None
    try:
        # Identity = viewer key, not URL: one photo can arrive under different URL
        # paths. A repeat is merged, not dropped (gallery_select §2 step 4 rules):
        # union of sources (owner wins as primary), newest caption, first ordinal.
        by_key, cands = {}, []
        for it in sorted(items, key=lambda i: i.get("ordinal", 0)):
            k = it.get("viewer_key")
            c = by_key.get(k) if k else None
            if c is None:
                c = {**it, "url": _full_url(it), "sources": [it.get("source") or "gallery"]}
                cands.append(c)
                if k:
                    by_key[k] = c
                continue
            src = it.get("source") or "gallery"
            if src not in c["sources"]:
                c["sources"].append(src)
            if "owner" in c["sources"]:
                c["source"] = "owner"
            new_m = gs.parse_photo_month(it.get("caption"))
            if new_m and new_m > (gs.parse_photo_month(c.get("caption")) or ""):
                c["caption"] = it.get("caption")
        sv = next((c for c in cands if c.get("kind") == "street_view" and c.get("url")), None) \
            or next(({"url": u} for u in panel_urls if "streetviewpixels" in u), None)
        if not wm.get("walk_ok"):   # panel strip: fallback source only (§1)
            cands = [{"kind": "photo", "url": u, "source": "panel", "ordinal": i}
                     for i, u in enumerate(panel_urls)]
        key_of = {c["url"]: c.get("viewer_key") or c["url"] for c in cands if c.get("url")}
        thumbs: dict = {}

        def fetch_thumb(url):   # keyed by viewer key: each photo fetched once (§9.1)
            k = key_of.get(url, url)
            if k not in thumbs:
                thumbs[k] = _http_get(gs.thumb_url(url), 3)
            return thumbs[k]

        t0 = time.monotonic()
        selected = gs.select_photos(
            cands, now=datetime.now(timezone.utc),
            cap=int(os.environ.get("PHASE2_PHOTOS_MAX") or 20),
            thresh=int(os.environ.get("PHASE2_DHASH_THRESHOLD") or 8),
            owner_count_known=bool(wm.get("owner_count_known")),
            owner_count=wm.get("owner_count"), fetch_thumb=fetch_thumb)
        tm["thumbs"] = _ms(t0, time.monotonic())
        if sv:   # §7: its own row, ordinal n, outside the keep cap
            selected.append(gs.Selected(
                url=blob_shots._upscale_url(sv["url"]), sources={"street_view"},
                source="street_view", photo_month=gs.parse_photo_month(sv.get("caption")),
                photo_date_raw=sv.get("caption"), ordinal=len(selected)))
        sas = blob_shots.blob_sas_url() if blob_shots.photos_enabled() else None
        t0 = time.monotonic()
        # No SAS (or PHASE2_PHOTOS_TO_BLOB=0): nothing is mirrored. Skip the CDN fetch; rows stay 'failed'.
        outcomes = blob_shots.mirror_selected(
            run_id, pid, selected, fetch=(lambda u: _http_get(u, 20)) if sas else (lambda u: None))
        tm["mirror"] = _ms(t0, time.monotonic())
        base = blob_shots._split_sas(sas)[0] if sas else ""
        fetched_at = _utc_iso()
        for sel, o in zip(selected, outcomes):
            is_sv = sel.source == "street_view"
            media.append({
                "ordinal": o["ordinal"], "kind": "street_view" if is_sv else "photo",
                "source_url": blob_shots._source_url(sel),
                "blob_url": f"{base}/{o['blob_name']}" if o["blob_name"] else None,
                "posted_month": sel.photo_month, "caption_raw": sel.photo_date_raw,
                "source": sel.source, "status": o["status"], "sha1": o["sha1"],
                "width": o["width"], "height": o["height"], "bytes": o["bytes"],
                "dhash": gs.signed64(sel.dhash), "stale": sel.stale, "fetched_at": fetched_at,
                "pano_url": sv.get("pano_url") if is_sv else None,
                "google_photo_id": _photo_id(sel.gid)})
    except Exception:
        logger.warning("phase2.photos_failed run_id=%s pid=%s", run_id, pid, exc_info=True)
        wm = {**wm, "walk_ok": False, "reason": "photo_pipeline_error"}
        selected, outcomes, media = [], None, []
    summ = gs.summary(items, selected, wm, outcomes)
    # §8.3: a failed walk stores only the summary, never a partial photo_dates.
    if not summ["walk_ok"]:
        row["photo_dates"] = {"summary": summ}
    elif os.environ.get("PHASE2_PHOTO_DATES_SLIM", "0") == "1":
        row["photo_dates"] = _slim_photo_dates(pd, summ, cands, selected, sv)
    else:
        row["photo_dates"] = {**pd, "summary": summ}
    return media, tm


def _photo_id(key) -> str | None:   # one definition: gallery_select.photo_id
    from parallel_scraper import gallery_select as _gs
    return _gs.photo_id(key)


def _slim_photo_dates(pd, summ, cands, selected, sv) -> dict:
    """PHASE2_PHOTO_DATES_SLIM=1: cut place_data log bytes. Selection already ran and
    place_media holds the kept photos, so the walk lists are audit-only: drop `items`,
    shrink `place_photos` to {url_sha16, posted_month, kind, selected, band, ordinal}
    per candidate. selected = has a place_media row. Other keys pass through unchanged.
    url_sha16 = first 16 hex of sha1(url as place_media.source_url stores it: =s1600
    for photos, upscaled for Street View) — the audit join key candidate -> media row.
    place_media.sha1 is the hash of the image BYTES, not of the URL: never join on it."""
    import hashlib
    from cloud import blob_shots
    from parallel_scraper import gallery_select as gs
    band = {s.url: s.band for s in selected if s.source != "street_view"}
    sv_sel = any(s.source == "street_view" for s in selected)

    def sha16(c):
        u = c.get("url")
        if not u:
            return None
        u = (blob_shots._upscale_url(u) if gs.item_kind(c) == "street_view"
             else u.split("=")[0] + "=s1600")
        return hashlib.sha1(u.encode()).hexdigest()[:16]

    out = {k: v for k, v in pd.items() if k != "items"}
    out["place_photos"] = [
        {"url_sha16": sha16(c), "posted_month": gs.parse_photo_month(c.get("caption")),
         "kind": gs.item_kind(c), "selected": c.get("url") in band or (c is sv and sv_sel),
         "band": band.get(c.get("url")) or None, "ordinal": c.get("ordinal")}
        for c in cands]
    out["summary"] = {**summ, "slim": 1}
    return out


def _page_url(session) -> str:
    return session.page_state()[0] if hasattr(session, "page_state") else ""


# ponytail: guessed settle time for Maps' URL rewrite of a gone place (p200: 4 of 6 logged failures
# still showed ?q=place_id:); no timing data, raise it if blanks with the old URL persist.
_BLANK_RECHECK_S = 5.0


def _delisted_or_raise(session, state, run_id, pid, attempts) -> None:
    """Blank/FAILED scrape (§8.4, §9). /sorry/ -> BOT_PAGE, raise (retried). Removal evidence
    (text or the /maps/place//@ redirect) -> POST /phase2/delisted and return. No evidence ->
    re-read once after _BLANK_RECHECK_S (the redirect lands late), then raise as before."""
    def _check():
        url, text = session.page_state(True) if hasattr(session, "page_state") else ("", "")
        if "/sorry/" in url:
            logger.warning("BOT_PAGE url=%s pid=%s", url, pid)
            raise RuntimeError(f"bot page: {url}")
        return url, delisted.detect_delisted(text, url)
    url, ev = _check()
    if not ev:
        time.sleep(_BLANK_RECHECK_S)
        url, ev = _check()
    if not ev:
        raise RuntimeError("blank-after-warmup / no data")
    out = state.mark_delisted(run_id, pid, attempts, _utc_iso(), ev, url) or {}
    logger.info("phase2.delisted run_id=%s pid=%s accepted=%s url=%s",
                run_id, pid, out.get("accepted"), url)


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


_ATTR_CHUNK = 200   # cloud/api.py WORKER_MAX_RESULTS: a kept-and-grown buffer must not hit 413


def _flush_attrs(state, run_id, pend_attrs) -> None:
    """Submit buffered Overview rows, in place: sent rows leave pend_attrs, unsent rows
    stay for the next flush (a 503, any non-2xx, any error). Never raises: losing
    additive fields must not lose the metadata flush that rides beside it."""
    while pend_attrs:
        chunk = pend_attrs[:_ATTR_CHUNK]
        try:
            out = state.submit_attributes(run_id, chunk)
        except Exception:
            logger.warning("phase2.attrs_flush_failed run_id=%s rows=%d kept", run_id,
                           len(pend_attrs), exc_info=True)
            return
        del pend_attrs[:len(chunk)]
        logger.info("phase2.attrs_flush rows=%d %s", len(chunk), out)


def _withhold_for_unsent_attrs(done: list[dict], attrs_left: list[dict]) -> tuple[list, list]:
    """Final flush only: (done rows to submit, done rows to keep). A place whose attribute
    rows could not be sent is NOT submitted as done, so it stays 'scraping', is reclaimed
    after the reclaim window and re-scraped with its attributes (reclaim selects only
    'scraping'; a done place would never get them back). The gate counts the kept rows
    as unflushed -> exit 3."""
    hold = {str(a.get("place_id")) for a in attrs_left}
    send = [d for d in done if str(d["place_id"]) not in hold]
    keep = [d for d in done if str(d["place_id"]) in hold]
    if keep:
        logger.warning("phase2.final_flush_withheld rows=%d (attributes unsent; left for reclaim)", len(keep))
    return send, keep


def _lease_with_retry(state, run_id, shard, batch, attempts: int = 5,
                      deadline: float | None = None, flush=None):
    """Claim a batch, retrying on empty results. Empty lease != empty queue.

    The API (design §9.2) answers an empty lease with `pending`: true while the run
    still has queued or scraping rows (held in another API pod's buffer, READPAST-
    skipped, or in flight on live shards that may release them). Then sleep its
    `retry_after_s` and retry, up to PHASE2_EMPTY_LEASE_MAX_WAIT_S (180) in total;
    `pending` false means drained, so return at once. Pilot p200: a shard whose
    lease hit a pod with nothing buffered exited after the old ladder while 52 rows
    were still 'scraping' in another pod's buffer.

    A response without `pending` (an API not yet redeployed, or a state backend that
    is not HTTP) keeps the old ladder: back off 15/30/45/60s (+shard jitter).

    `deadline` is a time.monotonic() stamp past which waiting is pointless — the
    shard is out of budget and will exit on the next loop anyway.

    `flush` (the caller's result flush, a no-op on an empty buffer) runs before every
    pending sleep. w2: each shard held its last 13-18 results through the whole wait,
    and those unflushed rows were the 'scraping' that kept every shard's lease
    pending (8-10 min tail, 15% of wall clock). Flushed, the run drains to
    pending=false as soon as the last place is done. Default 180 s: after the flush
    only rows un-served in another API pod's buffer can still come back, and those
    are released at PHASE2_LEASE_MAX_AGE_S (600) — a later top-up run picks them up.
    """
    # 300 s: covers the API's tail release (PHASE2_LEASE_TAIL_IDLE_S 60 s + a 15 s claimer tick).
    max_wait = float(os.environ.get("PHASE2_EMPTY_LEASE_MAX_WAIT_S") or 300)
    waited, i = 0.0, 0
    while True:
        leased = state.claim_place_ids(run_id, str(shard), int(batch))
        if leased:
            return leased
        if deadline is not None and time.monotonic() >= deadline:
            return []
        info = getattr(state, "last_lease", None)
        pending = info.get("pending") if isinstance(info, dict) else None
        if pending is None:                     # old API / non-HTTP backend
            i += 1
            if i >= attempts:
                return []
            time.sleep(min(60.0, 15.0 * i) + (hash(str(shard)) % 7))
            continue
        if info.get("hold"):
            # §13c ingest pause: the API holds every lease. Flush, wait, ask again; a hold
            # does not count toward the empty-lease limit.
            if flush is not None:
                flush()
            time.sleep(max(1.0, float(info.get("retry_after_s") or 60)))
            continue
        if not pending or waited >= max_wait:
            if pending:
                logger.info("phase2.lease_wait_exhausted run_id=%s shard=%s waited=%.0fs",
                            run_id, shard, waited)
            return []
        pause = max(1.0, float(info.get("retry_after_s") or 15))
        if flush is not None:
            flush()
        time.sleep(pause)
        waited += pause
        fresh = getattr(state, "fresh_connection", None)
        if fresh is not None:   # next lease over a new connection: maybe another API pod
            fresh()


def _resolve_concurrency(concurrency) -> int:
    """Resolve K from the arg or PHASE2_CONCURRENCY (default 1), clamped to [1, 5]."""
    if concurrency is None:
        concurrency = os.environ.get("PHASE2_CONCURRENCY", "1")
    try:
        k = int(concurrency)
    except (TypeError, ValueError):
        k = 1
    return max(1, min(5, k))


def _make_stager(run_id, shard):
    """§13c: a Stager for PHASE2_STAGING=shadow|1, None for 0 (today's path)."""
    m = staging.mode()
    if m == "0":
        return None
    logger.info("phase2.staging mode=%s run_id=%s shard=%s", m, run_id, shard)
    return staging.Stager(run_id, m, worker=os.environ.get("GITHUB_REPOSITORY", ""), shard=str(shard))


def _stage_one(stager, result, done, failed, extras, repair_gen) -> bool:
    """Write one outlet's object to the spool (mode shadow|1). Returns True when this place's
    result must NOT also go to /results (mode 1). Delisted places (both None) have no object."""
    if stager is None or (done is None and failed is None):
        return False
    try:
        if done is not None:
            stager.stage(done, "done", repair_gen=repair_gen, attributes=extras)
        else:
            stager.stage(failed, "failed", repair_gen=repair_gen)
    except Exception:
        # Spool write failed (disk): in mode 1 the place is leased but never reported, so it
        # counts as lost work (exit 3) and comes back through reclaim.
        logger.error("phase2.stage_failed pid=%s", (done or failed)["place_id"], exc_info=True)
    if stager.mode != "1":
        return False
    if done is not None:
        result.done += 1
    else:
        result.failed += 1
    return True


def _spool_wait(stager, flush, deadline: float | None = None) -> bool:
    """Spool at its cap: lease nothing until it drains (keep reporting meanwhile). Returns
    False when leasing must stop: 403 (the SAS expired), or the time budget ran out while
    waiting (the shard then drains and exits; leftovers are exit 3)."""
    while stager is not None and not stager.forbidden and stager.full():
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning("phase2.spool_full_at_budget spooled=%d", stager.spooled())
            return False
        flush()
        time.sleep(0.5)
    if stager is not None and stager.forbidden:
        logger.error("phase2.stage_forbidden: result SAS refused (403); leasing stopped")
        return False
    return True


def _finish_stager(stager, result, final_flush) -> None:
    """Shard end: drain the spool (<= 120 s), run the final flush (which reports every
    acknowledged ref in mode 1), then record what is left."""
    if stager is not None:
        stager.drain(staging.DRAIN_S)
    final_flush()
    if stager is None:
        return
    stager.close()
    result.spool_left = stager.spooled()
    result.refs_unreported = len(stager.refs)
    logger.info("phase2.staging_end mode=%s uploaded=%d spool_left=%d refs_unreported=%d "
                "accepted=%d superseded=%d rejected=%d put_errors=%d", stager.mode, stager.uploaded,
                result.spool_left, result.refs_unreported, stager.accepted, stager.superseded,
                stager.rejected, stager.put_errors)


def _startup_reclaim(state, run_id) -> None:
    """Shard 0 reclaims stale leases at start (manual dispatch has no sequencer). The API
    answers {"reclaimed": 0, "skipped": "cooldown"|"busy"} when this pod reclaimed the run
    recently or is reclaiming another: that is normal, logged at DEBUG only."""
    try:
        n = state.reclaim_place_ids(run_id, _RECLAIM_MIN)
    except Exception:
        logger.warning("phase2.reclaim_failed", exc_info=True)
        return
    skipped = (getattr(state, "last_reclaim", None) or {}).get("skipped")
    if skipped:
        logger.debug("phase2.reclaim_skipped run_id=%s reason=%s", run_id, skipped)
    else:
        logger.info("phase2.reclaim run_id=%s reclaimed=%d", run_id, n)


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
        _startup_reclaim(state, run_id)

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
    stager = _make_stager(run_id, shard)

    def flush(final: bool = False):
        nonlocal last_flush
        if stager is not None and stager.mode == "1":
            stager.report(state)                  # refs of acknowledged objects -> /phase2/staged
        _flush_attrs(state, run_id, pend_attrs)   # first: a raising results submit must not skip it
        send, keep = (_withhold_for_unsent_attrs(pend_done, pend_attrs) if final
                      else (list(pend_done), []))
        if send or pend_failed:
            out = state.submit(run_id, send, list(pend_failed))
            acc = int(out.get("accepted", 0))
            stale = int(out.get("stale", 0))
            # accepted spans both done+failed; count locally for logging only
            result.done += len(send)
            result.failed += len(pend_failed)
            logger.info("phase2.flush submitted=%d accepted=%d stale=%d done_total=%d failed_total=%d rps=%.3f",
                        len(send) + len(pend_failed), acc, stale, result.done, result.failed,
                        (result.done + result.failed) / max(0.001, time.monotonic() - started))
        pend_done[:] = keep
        pend_failed.clear()
        last_flush = time.monotonic()

    n_since_recycle = 0
    try:
        while True:
            if stop_after and (time.monotonic() - started) > stop_after:
                logger.info("phase2.time_budget_exit run_id=%s shard=%s", run_id, shard)
                break
            if max_places is not None and (result.done + result.failed + len(pend_done) + len(pend_failed)) >= max_places:
                break
            if not _spool_wait(stager, flush, started + stop_after if stop_after else None):
                break
            n = batch if stager is None else min(int(batch), stager.lease_room(0))
            if n <= 0:     # cannot happen after _spool_wait; never lease 0
                continue
            leased = _lease_with_retry(state, run_id, shard, n,
                                       deadline=started + stop_after, flush=flush)
            if not leased:
                break
            for rec in leased:
                pid = str(rec["place_id"])
                attempts = int(rec["attempts"])
                done, failed, ex = _scrape_one(session, run_id, pid, attempts,
                                               capture_shots, state, state)
                staged = _stage_one(stager, result, done, failed, ex, int(rec.get("repair_gen") or 0))
                if staged:
                    pass                 # mode 1: the object carries result + attributes
                elif done is not None:
                    pend_done.append(done)
                elif failed is not None:
                    pend_failed.append(failed)
                else:
                    result.done += 1     # delisted: tombstoned, counts as done (§9)
                if ex and not staged:
                    pend_attrs.append(ex)
                n_since_recycle += 1
                if n_since_recycle >= recycle_every:
                    try:
                        session.recycle()
                    except Exception:
                        logger.debug("recycle_failed", exc_info=True)
                    n_since_recycle = 0
                since = time.monotonic() - last_flush
                if since >= flush_interval_s or (stager is not None and stager.report_due(since)):
                    flush()
        _finish_stager(stager, result, lambda: flush(final=True))
    finally:
        try:
            session.close()
        except Exception:
            pass
    result.attrs_unsent = len(pend_attrs)
    logger.info("phase2.complete run_id=%s shard=%s done=%d failed=%d", run_id, shard, result.done, result.failed)
    return result


def _na_to_none(row: dict) -> dict:
    """§8.1: top-level 'N/A' / '' strings -> None before serialization; the nulled
    keys go in row['raw_sentinels'] for audit."""
    nulled = [k for k, v in row.items() if isinstance(v, str) and null_if_sentinel(v) is None]
    for k in nulled:
        row[k] = None
    row["raw_sentinels"] = nulled
    return row


def _scrape_one(session, run_id, pid, attempts, capture_shots, shot_state, state=None):
    """Scrape a single pid on `session` (design §1 order: panel, gallery walks,
    select + mirror, then serialize). Return (done, failed, extras); all three None
    when the place was tombstoned as delisted (§8.4) through `state` (default
    shot_state; the gated state in main()).

    Screenshot upload uses `shot_state` (this thread's own HttpState) and is
    best-effort: a failed upload must never fail the scrape or block the submit.
    """
    state = state if state is not None else shot_state
    t0 = time.monotonic()
    try:
        row = session.scrape(pid)
        if not row or (row.get("name") or "") in ("", "FAILED"):
            _delisted_or_raise(session, state, run_id, pid, attempts)
            return None, None, None
        _media_one(session, run_id, pid, row)
        # read before _photos_one: a failed walk keeps only the summary in photo_dates
        walk_tm = ((row.get("photo_dates") or {}).get("walk_meta") or {}).get("timings")
        urls = _split_image_urls(row.get("image_urls"))
        media, tm = (_photos_one(run_id, pid, row, urls) if hasattr(session, "media_capture")
                     else ([], {"thumbs": 0, "mirror": 0}))
        if os.environ.get("PHASE2_LEGACY_PANEL_PHOTOS", "0") == "1":
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
        extras = _panel_extras_one(session, run_id, pid)
        row["timing"] = _place_timing(row.get("timing"), walk_tm, tm, _ms(t0, time.monotonic()))
        done = {"place_id": pid, "attempts": attempts,
                "data": json.dumps(_na_to_none(row), default=str), "image_urls": urls, "media": media}
        timing = _timing_line(pid, row["timing"])
        print(timing, flush=True)   # p10: unconditional, whatever the log handlers do
        logger.info(timing)
        return done, None, extras
    except Exception as exc:
        logger.warning("phase2.place_failed run_id=%s pid=%s url=%s", run_id, pid,
                       _page_url(session), exc_info=True)
        return None, {"place_id": pid, "attempts": attempts, "error": str(exc)}, None


def _run_worker_concurrent(run_id, shard, k, max_minutes, batch, state,
                           session_factory, flush_interval_s, max_places) -> P2Result:
    """K>1 path: K worker threads, each with its own PlaywrightSession + HttpState.

    Feeder (this thread) leases pids onto a queue; workers scrape; this thread
    also flushes submitted results every flush_interval_s. Each leased pid is
    submitted exactly once (done XOR failed). K==1 never reaches here.
    """
    if str(shard) == "0":
        _startup_reclaim(state, run_id)

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
    processed = [0]   # places the workers finished (staged, buffered or failed); under `lock`
    stager = _make_stager(run_id, shard)

    logger.info("phase2.concurrent_start run_id=%s shard=%s k=%d", run_id, shard, k)

    def worker(idx: int):
        try:
            work_dir = tempfile.mkdtemp(prefix=f"p2api_{idx}_")
            session = factory(work_dir=work_dir)
            inner = getattr(state, "_inner", state)   # main() wraps it in _GateState
            shot_state = HttpState() if inner.__class__ is HttpState else inner.__class__()
        except Exception:
            # Session/state setup failed: drain the queue marking pids failed so
            # nothing is lost, until a sentinel arrives. Never block the feeder.
            logger.warning("phase2.worker_setup_failed idx=%d", idx, exc_info=True)
            while True:
                item = work_q.get()
                if item is _SENTINEL:
                    return
                pid, attempts, gen = item
                failed = {"place_id": pid, "attempts": attempts, "error": "worker setup failed"}
                with lock:
                    if not _stage_one(stager, result, None, failed, None, gen):
                        pend_failed.append(failed)
                    processed[0] += 1
            return
        with sessions_lock:
            sessions.append(session)
        n_since_recycle = 0
        while True:
            item = work_q.get()
            if item is _SENTINEL:
                return
            pid, attempts, gen = item
            done, failed, extras = _scrape_one(session, run_id, pid, attempts,
                                               capture_shots, shot_state, state)
            with lock:
                staged = _stage_one(stager, result, done, failed, extras, gen)
                if staged:
                    pass                 # mode 1: the object carries result + attributes
                elif done is not None:
                    pend_done.append(done)
                elif failed is not None:
                    pend_failed.append(failed)
                else:
                    result.done += 1     # delisted: tombstoned, counts as done (§9)
                if extras and not staged:
                    pend_attrs.append(extras)
                processed[0] += 1
            n_since_recycle += 1
            if n_since_recycle >= recycle_every:
                try:
                    session.recycle()
                except Exception:
                    logger.debug("recycle_failed", exc_info=True)
                n_since_recycle = 0

    def flush(final: bool = False):
        if stager is not None and stager.mode == "1":
            stager.report(state)                  # refs of acknowledged objects -> /phase2/staged
        with lock:
            done = pend_done[:]
            failed = pend_failed[:]
            attrs = pend_attrs[:]
            pend_done.clear()
            pend_failed.clear()
            pend_attrs.clear()
        _flush_attrs(state, run_id, attrs)
        if attrs:   # not sent: back to the front, next flush (final flush included) retries
            with lock:
                pend_attrs[:0] = attrs
        if final:
            done, keep = _withhold_for_unsent_attrs(done, attrs)
            with lock:
                pend_done[:0] = keep
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

    def wait_flush():
        nonlocal last_flush
        flush()
        last_flush = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if stop_after and (now - started) > stop_after:
                logger.info("phase2.time_budget_exit run_id=%s shard=%s", run_id, shard)
                break
            if max_places is not None and leased_total >= max_places:
                break
            # Hold off leasing while the workers still have a buffer, or while the staging
            # spool is at its cap (§13c); flush on interval.
            if stager is not None and stager.forbidden:
                logger.error("phase2.stage_forbidden: result SAS refused (403); leasing stopped")
                break
            with lock:
                held = leased_total - processed[0]
            room = int(batch) if stager is None else min(int(batch), stager.lease_room(held))
            if work_q.qsize() >= high_water or room <= 0:
                if (flush_interval_s and (now - last_flush) >= flush_interval_s) or (
                        stager is not None and stager.report_due(now - last_flush)):
                    flush()
                    last_flush = time.monotonic()
                time.sleep(0.2)
                continue
            leased = _lease_with_retry(state, run_id, shard, room,
                                       deadline=started + stop_after, flush=wait_flush)
            if not leased:
                break
            for rec in leased:
                pid = str(rec["place_id"])
                attempts = int(rec["attempts"])
                work_q.put((pid, attempts, int(rec.get("repair_gen") or 0)))
                leased_total += 1
            # Mid-run flush so results land while workers keep scraping.
            since = time.monotonic() - last_flush
            if (flush_interval_s and since >= flush_interval_s) or (
                    stager is not None and stager.report_due(since)):
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
            since = time.monotonic() - last_flush
            if (flush_interval_s and since >= flush_interval_s) or (
                    stager is not None and stager.report_due(since)):
                flush()
                last_flush = time.monotonic()
        # final flush after every pid is accounted for (and the spool drained)
        _finish_stager(stager, result, lambda: flush(final=True))
        with sessions_lock:
            for s in sessions:
                try:
                    s.close()
                except Exception:
                    pass

    result.attrs_unsent = len(pend_attrs)
    logger.info("phase2.complete run_id=%s shard=%s done=%d failed=%d k=%d",
                run_id, shard, result.done, result.failed, k)
    return result


class _GateState:
    """Wraps the state backend for main()'s shard gate (design §9, level 1).

    Counts what was leased and what the API accepted, so main() can tell lost work
    without reaching into the scrape loop. Everything else passes straight through."""

    def __init__(self, inner):
        self._inner = inner
        self.leased = 0          # pids handed to this shard
        self.submitted = 0       # rows in submits that returned (accepted or stale)
        self.acc_done = 0
        self.acc_failed = 0
        self.superseded = 0      # §13c: refs answered 'superseded' (not lost work)
        self._warned = False
        self._first_lease = self._first_done = False
        self._lock = threading.Lock()   # mark_delisted comes from the K worker threads

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def claim_place_ids(self, run_id, worker, n):
        out = self._inner.claim_place_ids(run_id, worker, n)
        if out:
            self.leased += len(out)
            if not self._first_lease:
                self._first_lease = True
                print(f"STARTUP t_first_lease={int(time.time())}", flush=True)
        return out

    def submit(self, run_id, done, failed):
        out = self._inner.submit(run_id, done, failed) or {}
        with self._lock:
            # rows the API refused even alone (413) are not submitted: lost work, exit 3
            self.submitted += len(done) + len(failed) - len(out.get("rejected") or [])
            if "accepted_done" in out and "accepted_failed" in out:
                self.acc_done += int(out["accepted_done"])
                self.acc_failed += int(out["accepted_failed"])
            else:
                if not self._warned:
                    self._warned = True
                    logger.warning("phase2.gate no accepted_done/accepted_failed in submit response; "
                                   "falling back to submitted counts (stale rows count as accepted)")
                self.acc_done += len(done)
                self.acc_failed += len(failed)
        if self.acc_done and not self._first_done:
            # ponytail: measured at first accepted submit, so includes up to one flush interval
            self._first_done = True
            print(f"STARTUP t_first_done={int(time.time())}", flush=True)
        return out


    def submit_staged(self, run_id, refs):
        """§13c mode 1: accepted and superseded refs are submitted work; rejected refs are not."""
        out = self._inner.submit_staged(run_id, refs) or {}
        with self._lock:
            self.submitted += int(out.get("accepted", 0)) + int(out.get("superseded", 0))
            self.acc_done += int(out.get("accepted_done", 0))
            self.acc_failed += int(out.get("accepted_failed", 0))
            self.superseded += int(out.get("superseded", 0))
        if self.acc_done and not self._first_done:
            self._first_done = True
            print(f"STARTUP t_first_done={int(time.time())}", flush=True)
        return out

    def mark_delisted(self, *args, **kwargs):
        """An accepted tombstone counts as done (§9); a stale one as submitted only."""
        out = self._inner.mark_delisted(*args, **kwargs) or {}
        with self._lock:
            self.submitted += 1
            self.acc_done += int(out.get("accepted", 0))
        return out


def shard_exit_code(gate: _GateState, max_fail: float, attrs_unsent: int = 0,
                    spool_left: int = 0, refs_unreported: int = 0) -> int:
    """0 ok, 3 = this shard lost work (§9 level 1). Results are already flushed.
    attrs_unsent: /attributes rows the final flush could not send; lost work like results.
    §13c mode 1: spool_left = objects never acknowledged by Blob (lost work, exit 3);
    refs_unreported = durable objects whose /phase2/staged report failed: not lost (reconcile
    stages them), so they are not 'unflushed'; superseded refs are not lost either."""
    lost = gate.leased - gate.submitted - refs_unreported
    den = gate.acc_done + gate.acc_failed
    rate = gate.acc_failed / den if den else 0.0
    superseded = int(getattr(gate, "superseded", 0) or 0)
    logger.info("phase2.gate leased=%d submitted=%d accepted_done=%d accepted_failed=%d "
                "unflushed=%d fail_rate=%.3f max=%.3f api_503=%d attrs_unsent=%d "
                "spool_left=%d refs_unreported=%d superseded=%d",
                gate.leased, gate.submitted, gate.acc_done, gate.acc_failed, lost, rate, max_fail,
                int(getattr(gate, "api_503", 0) or 0), attrs_unsent, spool_left, refs_unreported,
                superseded)
    reasons = [r for r, n in (("unflushed_buffer", lost), ("attrs_unsent", attrs_unsent),
                              ("spool_unsent", spool_left)) if n > 0]
    if reasons:   # all named when several lost work: no reason hides another
        logger.error("phase2.gate exit=3 reason=%s unflushed=%d attrs_unsent=%d spool_left=%d",
                     "+".join(reasons), lost, attrs_unsent, spool_left)
        return 3
    if refs_unreported:
        logger.error("phase2.gate refs_unreported=%d: objects are durable in Blob but their "
                     "/phase2/staged report failed; not lost work, reconcile stages them",
                     refs_unreported)
    if gate.leased and not (den + superseded + refs_unreported):
        logger.error("phase2.gate exit=3 reason=no_accepted_rows leased=%d", gate.leased)
        return 3
    # ponytail: ratio gate needs a floor, else 3 bad listings in a 20-place shard trip it (p200)
    min_failed = int(os.environ.get("PHASE2_SHARD_MIN_FAILED", "5") or "5")
    if rate > max_fail and gate.acc_failed >= min_failed:
        logger.error("phase2.gate exit=3 reason=fail_rate %.3f > %.3f (failed=%d >= %d)", rate, max_fail, gate.acc_failed, min_failed)
        return 3
    return 0


def main(argv=None, state=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-places", type=int)
    ap.add_argument("--concurrency", type=int, default=None,
                    help="placeIds scraped at once per shard (1-5; env PHASE2_CONCURRENCY is the fallback)")
    ap.add_argument("--max-fail-pct", type=float,
                    default=100 * float(os.environ.get("PHASE2_SHARD_MAX_FAIL") or 0.10),
                    help="exit 3 when accepted failed/(done+failed) exceeds this %% "
                         "(env PHASE2_SHARD_MAX_FAIL as a fraction, default 0.10)")
    args = ap.parse_args(argv)
    # p10: the stderr handler defaulted to WARNING, so no INFO line reached the runner log
    level = (os.environ.get("PHASE2_LOG_LEVEL") or "INFO").upper()
    os.environ.setdefault("PARALLEL_SCRAPER_STDERR_LEVEL", level)
    setup_logging(None, level=level)
    gate = _GateState(state if state is not None else HttpState())
    res = run_worker(args.run_id, args.shard, max_minutes=args.max_minutes,
                     batch=args.batch, max_places=args.max_places,
                     concurrency=args.concurrency, state=gate)
    if staging.mode() != "1":   # shadow: results went to /results; a left-over shadow object is not lost work
        if res.spool_left:
            logger.warning("phase2.shadow_spool_left=%d (not counted as lost work)", res.spool_left)
        res.spool_left = res.refs_unreported = 0
    return shard_exit_code(gate, args.max_fail_pct / 100.0, attrs_unsent=res.attrs_unsent,
                           spool_left=res.spool_left, refs_unreported=res.refs_unreported)


if __name__ == "__main__":
    raise SystemExit(main())
