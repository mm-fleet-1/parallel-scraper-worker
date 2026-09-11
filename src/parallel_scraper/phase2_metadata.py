"""Per-placeId Playwright deep-scrape using the vendored extractor."""
from __future__ import annotations

import asyncio
import csv
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Append-only sidecar that records, per place, that its image-download job ran
# to completion. Used to recover image downloads lost when the process is hard-
# killed after a place is marked `done` but before its (async) images finish.
IMAGE_STATE_FIELDS = ("place_id", "expected", "saved", "completed_at")

try:
    import dataclasses as _dc
    from parallel_scraper.vendor.pipeline_essentials import run_production_pipeline as _rpp
    from parallel_scraper.vendor.pipeline_essentials.run_production_pipeline import (
        run_phase1, _download_one_image, _upscale_image_url, _process_one_place,
    )
    from parallel_scraper.vendor.pipeline_essentials.poc_crawlee.models import PlaceData, ScrapeConfig
    # The vendored extractor's _process_one_place trims each result to the module
    # global COLUMN_ORDER (a 14-column list) before handing it to result_cb. That
    # silently drops fields the extractor *did* populate — photo_count,
    # total_photos, latest_review_date, menu_link, price_level, plus_code, etc.
    # Widen it to every PlaceData field so nothing is dropped; the parallel
    # scraper's own schema-locked CSV writer does the final trim to the
    # operator-selected columns.
    _rpp.COLUMN_ORDER = [f.name for f in _dc.fields(PlaceData)]
    _IMPORT_OK = True
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    _IMPORT_OK = False
    _IMPORT_ERROR = e
    run_phase1 = None  # type: ignore
    ScrapeConfig = None  # type: ignore
    _download_one_image = None  # type: ignore
    _upscale_image_url = None   # type: ignore
    _process_one_place = None   # type: ignore


def build_place_url(place_id: str) -> str:
    """Build a Google Maps place URL that round-trips through extract_place_id_from_url."""
    return f"https://www.google.com/maps/place/?q=place_id:{place_id}"


def _safe_place_id_dirname(place_id: str) -> str:
    """Strip characters unsafe for a directory name. Google placeIds are usually
    alphanumeric (ChIJ…), but defensively replace anything outside [A-Za-z0-9_-]."""
    safe = []
    for ch in (place_id or "").strip():
        safe.append(ch if (ch.isalnum() or ch in "_-") else "_")
    return "".join(safe) or "unknown_place"


def _shot_paths(shots_dir: str, place_id: str) -> dict:
    """Map each panel kind to its on-disk PNG path, for kinds that actually exist.

    Mirrors the names written by ``_capture_panels`` (``{safe}_overview.png`` /
    ``{safe}_reviews.png``). Returns ``{}`` when no screenshots were captured.
    """
    safe = _safe_place_id_dirname(place_id)
    out = {}
    for kind in ("overview", "reviews"):
        p = Path(shots_dir) / f"{safe}_{kind}.png"
        if p.exists():
            out[kind] = str(p)
    return out


def _dedupe_image_urls(image_urls_str: str, max_images: int = 50) -> list[str]:
    """Parse the ';'-joined URL string into a deduped, capped list. Dedupe is
    by URL base (size/qs suffix stripped) so a thumb + full-size of the same
    image count once. Shared by the downloader and the completion-marker logic
    (so `expected` matches what the downloader actually attempts)."""
    if not image_urls_str:
        return []
    text = str(image_urls_str).strip()
    if not text or text == "N/A":
        return []
    urls = [u.strip() for u in text.split(";") if u.strip()]
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        base = u.split("=")[0] if "=" in u else u
        if base in seen:
            continue
        seen.add(base)
        unique.append(u)
    return unique[:max_images]


def load_completed_image_ids(path) -> set[str]:
    """Return the set of place_ids whose image job has run to completion,
    read from the append-only images_state.csv sidecar. Empty if absent."""
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pid = (row.get("place_id") or "").strip()
                if pid:
                    out.add(pid)
    except Exception:
        logger.warning("load_completed_image_ids failed path=%s", p, exc_info=True)
    return out


def download_place_images(place_id: str, image_urls_str: str, images_root: Path,
                          max_threads: int = 8, max_images: int = 50) -> int:
    """Download a place's photo URLs into images_root/<place_id>/place_id_N.<ext>.

    Loose files, one folder per place. Returns the number of images saved. URLs
    are deduped by their base (size-suffix stripped) and upscaled to high-res
    via the vendored helpers. Failures are silent — image downloads are
    best-effort and never fail a scrape.
    """
    if not _IMPORT_OK:
        return 0
    unique = _dedupe_image_urls(image_urls_str, max_images)
    if not unique:
        return 0
    place_dir = Path(images_root) / _safe_place_id_dirname(place_id)
    place_dir.mkdir(parents=True, exist_ok=True)
    # Parallel fetch — _download_one_image accepts (idx, url) and returns
    # (idx, ext, bytes) or None.
    results: dict[int, tuple] = {}
    upscaled = [(i, _upscale_image_url(u)) for i, u in enumerate(unique)]
    with ThreadPoolExecutor(max_workers=min(max_threads, len(upscaled))) as pool:
        futs = {pool.submit(_download_one_image, args): args[0] for args in upscaled}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                idx, ext, data = r
                results[idx] = (ext, data)
    if not results:
        return 0
    saved = 0
    for idx in sorted(results):
        ext, data = results[idx]
        out_path = place_dir / f"{_safe_place_id_dirname(place_id)}_{saved + 1}.{ext}"
        try:
            out_path.write_bytes(data)
            saved += 1
        except Exception:
            logger.debug("download_place_images.write_failed path=%s", out_path, exc_info=True)
    return saved


class ImageDownloadPool:
    """Background pool that downloads place images off the scrape critical path.

    Phase-2 consumers submit `(place_id, image_urls)` and return immediately;
    a small set of worker threads call `download_place_images`. The backlog is
    bounded: when the queue is full, `submit` returns False and the caller
    falls back to an inline download, so images are never dropped and the
    queue cannot grow without limit.

    Note: a place is marked `done` as soon as its row is persisted, before its
    images finish downloading. `drain_and_close` flushes the backlog on a
    graceful stop; a hard kill (taskkill) loses whatever is still in flight,
    which is acceptable since image downloads are best-effort.
    """

    _SENTINEL = object()

    def __init__(self, images_root, workers: int = 6, maxsize: int = 512,
                 marker_writer=None) -> None:
        self._images_root = Path(images_root)
        # Append-only completion-marker writer (BufferedCsvWriter or None).
        self._marker = marker_writer
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(maxsize)))
        self._threads = [
            threading.Thread(target=self._worker, name=f"img-{i}", daemon=True)
            for i in range(max(1, int(workers)))
        ]
        for t in self._threads:
            t.start()

    def submit(self, place_id: str, image_urls: str) -> bool:
        """Non-blocking enqueue. Returns False if the backlog is full so the
        caller can fall back to an inline download."""
        try:
            self._q.put_nowait((place_id, image_urls))
            return True
        except queue.Full:
            return False

    def download_inline(self, place_id: str, image_urls: str) -> int:
        """Download synchronously on the calling thread (backlog-full fallback)
        and record the completion marker, so inline-downloaded places aren't
        re-attempted on resume."""
        return self._download_and_mark(place_id, image_urls)

    def _download_and_mark(self, place_id: str, image_urls: str) -> int:
        saved = download_place_images(place_id, image_urls, self._images_root)
        # Marker = "the image job ran to completion" (NOT "all files present").
        # Written even when saved < expected (dead URLs) so we don't re-chase
        # permanently-dead URLs every resume; absence of the marker means the
        # job never finished (hard kill), which is what we recover.
        if self._marker is not None:
            try:
                expected = len(_dedupe_image_urls(image_urls))
                self._marker.append({
                    "place_id": place_id,
                    "expected": str(expected),
                    "saved": str(saved),
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            except Exception:
                logger.warning("image_pool.marker_write_failed place_id=%s",
                               place_id, exc_info=True)
        return saved

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._SENTINEL:
                    return
                place_id, image_urls = item
                try:
                    self._download_and_mark(place_id, image_urls)
                except Exception:
                    logger.warning("image_pool.download_failed place_id=%s",
                                   place_id, exc_info=True)
            finally:
                self._q.task_done()

    def drain_and_close(self, timeout: float = 60.0) -> None:
        """Let queued downloads finish (up to `timeout`), then stop workers."""
        for _ in self._threads:
            try:
                self._q.put(self._SENTINEL, timeout=1.0)
            except queue.Full:
                pass
        deadline = time.monotonic() + max(0.0, timeout)
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = sum(1 for t in self._threads if t.is_alive())
        if alive:
            logger.warning("image_pool.drain_timeout still_alive=%d", alive)


def scrape_one(
    place_id: str,
    csv_path: str,
    failures_path: str,
    delay_ms: int = 500,
) -> Optional[dict]:
    """Scrape a single placeId via Playwright. Returns the captured row dict or None.

    Runs in its own asyncio event loop so it's safe to call from a thread.
    """
    if not _IMPORT_OK:
        raise RuntimeError(
            f"vendored pipeline extractor import failed: {_IMPORT_ERROR}"
        )

    url = build_place_url(place_id)
    captured: list[dict] = []

    def on_row(row: dict) -> None:
        captured.append(row)

    config = ScrapeConfig(extract_share_link=False)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_phase1(
            rows=[{"url": url, "place_id": place_id, "label": ""}],
            csv_path=str(csv_path),
            failures_path=str(failures_path),
            config=config,
            batch_size=1,
            delay_ms=delay_ms,
            process_batch_size=0,
            browsers=1,
            contexts_per_browser=1,
            resume=False,
            lean=False,
            lean_enrich=False,
            photo_urls=False,
            result_cb=on_row,
        ))
    finally:
        try:
            loop.close()
        finally:
            asyncio.set_event_loop(None)

    if not captured:
        return None
    return captured[0]


class PlaywrightSession:
    """Long-lived Playwright browser + page held by a Phase-2 consumer thread.

    Replaces the per-place `scrape_one` pattern (which launched and tore down a
    full Chromium for every place — ~1-2 s/place wasted). One session per
    consumer thread, reused across many places, periodically recycled by the
    consumer to bound memory creep.

    Auto-recovers if the page or browser is closed between scrapes (e.g. a
    transient crash) by relaunching transparently.
    """

    def __init__(self, csv_path: str, failures_path: str, delay_ms: int = 500,
                 full_relaunch_every: int = 10, capture_screenshots: bool = False) -> None:
        if not _IMPORT_OK:
            raise RuntimeError(f"vendored pipeline extractor import failed: {_IMPORT_ERROR}")
        self._csv_path = str(csv_path)
        self._failures_path = str(failures_path)
        self._delay_ms = int(delay_ms)
        self._full_relaunch_every = int(full_relaunch_every)
        self._capture_screenshots = bool(capture_screenshots)
        self._shots_dir = Path(csv_path).parent / "screenshots"
        if self._capture_screenshots:
            self._shots_dir.mkdir(parents=True, exist_ok=True)
        self._recycle_count = 0
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        # asyncio.Lock must be created inside the loop it runs on
        self._csv_lock = self._loop.run_until_complete(self._make_lock())
        self.places_scraped = 0
        self._open_browser()

    @staticmethod
    async def _make_lock() -> asyncio.Lock:
        return asyncio.Lock()

    # ── browser lifecycle ──────────────────────────────────────

    def _open_browser(self) -> None:
        async def _setup() -> None:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--disable-gpu"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                extra_http_headers={"accept-language": "en-IN,en;q=0.9"},
            )
            self._page = await self._context.new_page()
            # Warm the session: visiting maps.google.com once before scraping
            # individual place URLs causes Google to plant the cookies
            # (CONSENT/NID/SOCS) that unlock the full place panel. Without
            # this, headless Chromium gets a degraded UI: F7nice ships only
            # the rating ("4.7") with no review count, and there is no
            # "Reviews for X" tab. Cost: ~1.5s once per browser.
            try:
                await self._page.goto(
                    "https://www.google.com/maps",
                    wait_until="domcontentloaded",
                    timeout=15000,
                )
                await self._page.wait_for_timeout(1500)
            except Exception as e:
                logger.debug("phase2_session warm-up failed (continuing): %s", e)
        self._loop.run_until_complete(_setup())

    def _close_browser(self) -> None:
        async def _cleanup() -> None:
            try:
                if self._context is not None:
                    await self._context.close()
            except Exception:
                pass
            try:
                if self._browser is not None:
                    await self._browser.close()
            except Exception:
                pass
            try:
                if self._pw is not None:
                    await self._pw.stop()
            except Exception:
                pass
        try:
            self._loop.run_until_complete(_cleanup())
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._pw = None

    def _recycle_page(self) -> None:
        """Swap in a fresh page on the SAME context. Context-scoped warm-up
        cookies (CONSENT/NID/SOCS) persist, so no re-warm-up is needed
        (~50ms vs ~1.5-2s for a full relaunch). Falls back to a full relaunch
        if the context is gone or the swap fails."""
        if self._context is None:
            self._close_browser()
            self._open_browser()
            return

        async def _swap() -> None:
            try:
                if self._page is not None and not self._page.is_closed():
                    await self._page.close()
            except Exception:
                pass
            self._page = await self._context.new_page()

        try:
            self._loop.run_until_complete(_swap())
        except Exception:
            logger.warning("phase2_session.page_recycle_failed — full relaunch",
                           exc_info=True)
            self._close_browser()
            self._open_browser()

    def recycle(self) -> None:
        """Recycle between scrape batches. Page-only by default (fast, keeps
        the warm context); every `full_relaunch_every` recycles, do a full
        Chromium relaunch to reclaim RSS growth. Resets places_scraped."""
        self._recycle_count += 1
        if (self._full_relaunch_every > 0
                and self._recycle_count % self._full_relaunch_every == 0):
            self._close_browser()
            self._open_browser()
        else:
            self._recycle_page()
        self.places_scraped = 0

    def close(self) -> None:
        """Final teardown — close everything and shut down the event loop."""
        self._close_browser()
        try:
            self._loop.close()
        except Exception:
            pass

    # ── per-place scrape ───────────────────────────────────────

    def _page_alive(self) -> bool:
        try:
            return self._page is not None and not self._page.is_closed()
        except Exception:
            return False

    def scrape(self, place_id: str) -> Optional[dict]:
        """Scrape one placeId against the held browser/page. Returns the row
        dict (or a FAILED dict on a per-place error)."""
        if not self._page_alive():
            logger.info("phase2_session.reopen reason=page_closed")
            self.recycle()
        url = build_place_url(place_id)
        captured: list[dict] = []

        def on_row(row: dict) -> None:
            captured.append(row)

        async def _do() -> None:
            await _process_one_place(
                page=self._page,
                row={"url": url, "place_id": place_id, "label": ""},
                csv_path=self._csv_path,
                failures_path=self._failures_path,
                delay_ms=self._delay_ms,
                csv_lock=self._csv_lock,
                progress_cb=None,
                lean=False,
                lean_enrich=False,
                photo_urls=False,
                result_cb=on_row,
            )
            # Best-effort panel screenshots for LLM review. The page is left on
            # the Reviews tab (Sort=Newest) by the extractor, so we shoot that
            # first, then flip to Overview. Never fails the scrape.
            if self._capture_screenshots and captured:
                name = (captured[0].get("name") or "").strip()
                if name and name != "FAILED":
                    try:
                        await self._capture_panels(place_id)
                    except Exception:
                        logger.debug("phase2_session.capture_panels_failed place_id=%s",
                                     place_id, exc_info=True)

        self._loop.run_until_complete(_do())
        self.places_scraped += 1
        return captured[0] if captured else None

    def panel_extras(self, place_id: str) -> Optional[dict]:
        """Overview-pane capture (price band, hours, popular times, links, share
        link, structured menu). Gated by env PHASE2_PANEL_EXTRAS=1.

        Navigates BACK to the place URL first: scrape() leaves the page on the
        reviews pane, which replaces Overview, and every module below lives on
        Overview. Run this AFTER horeca_capture — the gallery walk needs the
        post-scrape panel state and a re-goto does not reproduce it.
        Never raises; returns None when off or the page is dead.
        """
        import os as _os
        if _os.environ.get("PHASE2_PANEL_EXTRAS", "0") != "1":
            return None
        if not self._page_alive():
            return None
        from parallel_scraper.panel_extras import capture_panel_extras
        want_menu = _os.environ.get("PHASE2_MENU_TAB", "1") != "0"
        want_about = _os.environ.get("PHASE2_ABOUT_TAB", "1") != "0"

        async def _do() -> dict:
            await self._page.goto(build_place_url(place_id),
                                  wait_until="domcontentloaded", timeout=60_000)
            return await capture_panel_extras(self._page, want_menu=want_menu,
                                              want_about=want_about)

        try:
            ex = self._loop.run_until_complete(_do())
            if ex.get("degraded"):
                # Session-scoped degraded panel (no tab strip): a fresh browser gets
                # the full panel back at once. Relaunch (~2s) and retry once.
                logger.info("phase2_session.panel_degraded place_id=%s — relaunching", place_id)
                self._close_browser()
                self._open_browser()
                ex = self._loop.run_until_complete(_do())
            return ex
        except Exception:
            logger.warning("phase2_session.panel_extras_failed place_id=%s",
                           place_id, exc_info=True)
            return None

    def media_capture(self, place_id: str) -> Optional[dict]:
        """Dated gallery walk on the CURRENT page — call right after
        scrape(place_id) while the panel is still open. ALWAYS on: every place
        gets {"photo_dates": {place_photos, imagery, errors}}. With env
        PHASE2_HORECA=1 the Menu tab is walked first and {"horeca": {...menu}}
        is added. PHASE2_PHOTO_DATES_CAP (default 0 = all) / PHASE2_HORECA_PHOTO_CAP
        (menu, default 12). Returns None only when the page is dead; never raises."""
        import os as _os
        if not self._page_alive():
            return None
        from parallel_scraper.horeca_capture import (PHOTO_CAP_DEFAULT,
                                                     PLACE_PHOTO_CAP_DEFAULT, capture_horeca)
        with_menu = _os.environ.get("PHASE2_HORECA", "0") == "1"
        menu_cap = int(_os.environ.get("PHASE2_HORECA_PHOTO_CAP", PHOTO_CAP_DEFAULT))
        place_cap = int(_os.environ.get("PHASE2_PHOTO_DATES_CAP", PLACE_PHOTO_CAP_DEFAULT))

        async def _do() -> dict:
            return await capture_horeca(self._page, photo_cap=menu_cap, with_menu=with_menu,
                                        place_photo_cap=place_cap)

        try:
            hc = self._loop.run_until_complete(_do())
        except Exception:
            logger.warning("phase2_session.media_capture_failed place_id=%s",
                           place_id, exc_info=True)
            return None
        out = {"photo_dates": {k: hc.get(k) for k in ("place_photos", "imagery")}}
        out["photo_dates"]["errors"] = [e for e in hc["errors"] if e.startswith("place_photos")]
        if with_menu:
            out["horeca"] = {k: hc.get(k) for k in ("menu_link", "website", "menu_photos")}
            out["horeca"]["errors"] = [e for e in hc["errors"]
                                       if not e.startswith("place_photos")]
        return out

    def horeca_capture(self, place_id: str, photo_cap: int | None = None) -> Optional[dict]:
        """Old combined shape for the local tools (rescrape/probe/smoke test)."""
        m = self.media_capture(place_id)
        if not m:
            return None
        return {**(m.get("horeca") or {}), **m["photo_dates"],
                "errors": (m.get("horeca") or {}).get("errors", []) + m["photo_dates"]["errors"]}

    def screenshot_paths(self, place_id: str) -> dict:
        """Return {kind: png_path} for the panel screenshots captured for this place
        (empty if capture is off or nothing was written). Used by the API-mode worker
        to know which files to upload."""
        if not self._capture_screenshots:
            return {}
        return _shot_paths(str(self._shots_dir), place_id)

    # ── screenshots (optional, for LLM review) ─────────────────

    async def _shot_panel(self, page, out_path: "Path") -> None:
        """Screenshot the left place-panel element only (excludes the map),
        clipped to the visible viewport so a long reviews feed isn't captured
        in full. Best-effort."""
        try:
            panel = await page.query_selector('div[role="main"]')
            if panel is None:
                return
            box = await panel.bounding_box()
            if not box:
                return
            vp = page.viewport_size or {"width": 1920, "height": 1080}
            x, y = max(0.0, box["x"]), max(0.0, box["y"])
            clip = {"x": x, "y": y,
                    "width": min(box["width"], vp["width"] - x),
                    "height": min(box["height"], vp["height"] - y)}
            if clip["width"] <= 1 or clip["height"] <= 1:
                return
            await page.screenshot(path=str(out_path), clip=clip)
        except Exception:
            logger.debug("phase2_session.shot_panel_failed path=%s", out_path, exc_info=True)

    async def _click_overview(self, page) -> bool:
        for sel in ('button[aria-label*="Overview" i][role="tab"]',
                    'button[role="tab"][data-tab-index="0"]',
                    'button[aria-label*="Overview" i]'):
            try:
                tab = await page.query_selector(sel)
                if tab:
                    await tab.click(timeout=2000)
                    return True
            except Exception:
                continue
        # Fallbacks by accessible role / visible label (handles tabs whose
        # aria-label is localized or "Overview of <place>").
        for getter in (lambda: page.get_by_role("tab", name="Overview"),
                       lambda: page.get_by_text("Overview", exact=True)):
            try:
                await getter().first.click(timeout=2000)
                return True
            except Exception:
                continue
        return False

    async def _capture_panels(self, place_id: str) -> None:
        page = self._page
        safe = _safe_place_id_dirname(place_id)
        # A Reviews tab only exists when the place has reviews. If so, the page
        # is currently on it (extractor sorted by Newest) — shoot the histogram +
        # reviews, then flip to Overview. If there are NO reviews, Maps shows a
        # single panel (the overview content) with no tab bar — capture it once,
        # correctly named _overview.
        reviews_tab = await page.query_selector(
            'button[role="tab"][aria-label*="Reviews" i], '
            'button[aria-label^="Reviews for" i]'
        )
        if reviews_tab is not None:
            await self._shot_panel(page, self._shots_dir / f"{safe}_reviews.png")
            if await self._click_overview(page):
                await page.wait_for_timeout(500)
                await self._shot_panel(page, self._shots_dir / f"{safe}_overview.png")
        else:
            await self._shot_panel(page, self._shots_dir / f"{safe}_overview.png")
