"""HoReCa capture: menu photos + place photos with posted dates + imagery class.

Runs on the PlaywrightSession's live page AFTER the metadata scrape (the page is
already on the place panel with warm cookies). Ported from tools/menu_probe.py,
which was validated on 151 Riyadh outlets (see menu_batch_riyadh.jsonl).

Hard-won behaviors baked in (do not "simplify" these away):
- The gallery viewer's main image is NOT a queryable <img> (canvas-rendered):
  full-res URLs are captured by sniffing lh3.googleusercontent responses with a
  large =w/=s size param.
- Advancing slides works reliably via the visible "Next" button; ArrowRight
  does not always reach the viewer.
- Caption in the top-left header ("Photo - Jun 2026" / "Video - Jul 2026") is
  the item's date; the bottom "Image capture:" bar is Street View attribution
  (also the date for street-view items). Both are month precision only.
- Videos load no new big image, so their sniffed URL is a neighbour's — null it.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

PHOTO_CAP_DEFAULT = 12        # menu cards (horeca=1 only)
# place-photo date walk cap. The dates feed the Phase 3 vision pass (pick recent
# storefront/shelf shots), so a pool of 20 is enough; ~85% of target outlets have <=20
# photos and are dated in full. Also keeps every walk far inside the 30-min lease (an
# uncapped 13k-photo hypermarket took ~2.5 h and was marked failed). NB gallery order is
# Google's, not newest-first. 0 = no cap; env PHASE2_PHOTO_DATES_CAP overrides.
PLACE_PHOTO_CAP_DEFAULT = 20
_WALK_GUARD = 5000            # loop guard for the uncapped walk, not a data cap

_EVAL_CURRENT_ITEM = """() => {
    let best = null, bestA = 0;
    document.querySelectorAll('img[src*="googleusercontent"], img[src*="streetviewpixels"]').forEach(im => {
        const r = im.getBoundingClientRect();
        if (r.width < 250 || r.height < 250) return;
        const a = r.width * r.height;
        if (a > bestA) { bestA = a; best = im.src; }
    });
    let vid = null;
    document.querySelectorAll('video').forEach(v => {
        const r = v.getBoundingClientRect();
        if (r.width >= 250 && r.left >= 0 && r.left < innerWidth) vid = v.currentSrc || 'video';
    });
    let cap = null, attr = null;
    for (const el of document.querySelectorAll('div,span')) {
        const t = (el.textContent || '').trim();
        if (el.children.length !== 0 || t.length > 45) continue;
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        if (/^(Photo|Video)\\s*[-\\u2013]/.test(t) && r.top < 200 && !cap) cap = t;
        if (/^Image capture:/i.test(t) && !attr) attr = t;
    }
    return {img: best, vid, cap, attr};
}"""

_EVAL_STRIP_SRCS = """() => {
    const srcs = [];
    document.querySelectorAll('img[src*="googleusercontent"]').forEach(im => {
        const r = im.getBoundingClientRect();
        if (r.width >= 60 && r.width <= 200) srcs.push(im.src);
    });
    return [...new Set(srcs)];
}"""


class _Skip(Exception):
    """Section switched off — not an error, nothing recorded."""


async def _click_next(page) -> None:
    nxt = page.locator('button[aria-label*="Next" i]').first
    try:
        if await nxt.count():
            await nxt.click(timeout=3_000)
            return
    except Exception:
        pass
    await page.keyboard.press("ArrowRight")


_NEXT_DISABLED = """() => {
    const b = document.querySelector('button[aria-label*="Next" i]');
    return !!(b && (b.disabled || b.getAttribute('aria-disabled') === 'true'));
}"""


# Bring the LAST loaded thumbnail into view: the grid lazy-loads on intersection, so
# setting scrollTop on containers did nothing, while scrollIntoView loaded the next
# batch (20 -> 22) with the viewer still open. Escape is not an option — it tears the
# whole gallery down (0 thumbnails left).
_SCROLL_GRID = """() => {
    const t = [...document.querySelectorAll('[data-photo-index]')];
    if (!t.length) return 0;
    t[t.length - 1].scrollIntoView({block: 'end'});
    return t.length;
}"""


async def _load_more(page) -> bool:
    """The 'See photos' grid lazy-loads ~10 thumbnails at a time and the viewer can
    only page through loaded ones, so Next turns off at item 10 until the grid is
    scrolled (fleet: 546 listings stopped at exactly 10 of 22-40). Scroll it and wait
    for Next to come back; False = the real end of the gallery."""
    if not await page.evaluate(_SCROLL_GRID):
        return False
    for _ in range(8):
        await page.wait_for_timeout(500)
        if not await page.evaluate(_NEXT_DISABLED):
            return True
    return False


def _photo_key(href: str) -> str:
    """The viewer URL carries the current item's id (`!1s<id>`); it changes on
    every Next whether or not the image comes over the network."""
    m = re.search(r"!1s([^!]+)", href or "")
    return m.group(1) if m else (href or "")


async def _walk_viewer(page, sniffed: list[str], cap_items: int) -> list[dict]:
    """Walk the open gallery viewer, one dict per item: {url, date, kind}.
    cap_items=0 walks to the end: Next disabled is the end of the gallery.

    Item identity is the viewer URL, not the sniffed image URL. Sniff-based
    identity stalled when images arrived late (or not at all): two same-month
    captions then looked identical and the walk stopped at ~10 of 29."""
    items: list[dict] = []
    seen: set[str] = set()
    stale = 0
    # The first item can render late (a Street View canvas draws its "Image
    # capture" caption after the viewer opens). Reading too early saw nothing,
    # pressed Next into an empty panorama and gave up with zero items.
    for _ in range(8):
        got = await page.evaluate(_EVAL_CURRENT_ITEM)
        if got.get("img") or got.get("cap") or got.get("attr") or sniffed:
            break
        await page.wait_for_timeout(500)
    n_sniff_used = 0
    for _ in range(cap_items * 2 if cap_items else _WALK_GUARD):
        got = await page.evaluate(_EVAL_CURRENT_ITEM)
        key = _photo_key(page.url)
        src, cap, attr = got.get("img"), got.get("cap"), got.get("attr")
        if not src and len(sniffed) > n_sniff_used:
            src = sniffed[-1]           # only an image that arrived for THIS step
        n_sniff_used = len(sniffed)
        is_video = bool(cap and cap.lower().startswith("video"))
        if is_video:
            src = None
        sig = f"{key}|{cap}|{got.get('vid')}"
        if sig in seen:
            stale += 1
            if stale >= 3:
                break
        else:
            seen.add(sig)
            stale = 0
            if is_video:
                items.append({"url": None, "date": cap, "kind": "video"})
            elif cap:
                is_sv = bool(src and "streetviewpixels" in src)
                items.append({"url": src, "date": cap,
                              "kind": "street_view" if is_sv else "photo"})
            elif src:
                is_sv = "streetviewpixels" in src or bool(attr)
                items.append({"url": src, "date": attr,
                              "kind": "street_view" if is_sv else "photo"})
            elif attr:
                # Street View item with no <img>: either the cover of a listing with
                # no photo gallery (no Next -> stop below) or the last gallery item.
                items.append({"url": None, "date": attr, "kind": "street_view"})
        if cap_items and len(items) >= cap_items:
            break
        if await page.evaluate(_NEXT_DISABLED) and not await _load_more(page):
            break
        if not await page.locator('button[aria-label*="Next" i]').count():
            break    # a lone Street View panorama: arrow keys move the camera
        n_before, key_before = len(sniffed), key
        await _click_next(page)
        # Wait up to ~8 s for the viewer to move. On slow fleet runners 3 s was not
        # enough: the unchanged URL read as a repeat, 3 repeats ended the walk early
        # (Forever Living: 26 in the grid, fleet walked 9) and re-clicking Next while
        # the last slide was still loading skipped items. Fast loads still break at once.
        for _ in range(26):
            await page.wait_for_timeout(300)
            if _photo_key(page.url) != key_before or len(sniffed) > n_before:
                break
    return items


async def capture_horeca(page, photo_cap: int = PHOTO_CAP_DEFAULT, with_menu: bool = True,
                         place_photo_cap: int = PLACE_PHOTO_CAP_DEFAULT) -> dict:
    """Capture menu/photo/date/imagery data from the place panel currently on
    `page`. Never raises — every section degrades to empty + an errors entry.

    with_menu=False skips links + the Menu tab and only walks the place-photo
    gallery (the always-on photo-date capture). Order is unchanged when on:
    menu first, then the gallery re-entered through the menu thumbs."""
    out: dict = {"menu_link": None, "website": None, "menu_photos": [],
                 "place_photos": [], "imagery": None, "errors": []}
    menu_walked = False

    sniffed: list[str] = []

    def _sniff(resp):
        u = resp.url
        if "googleusercontent.com" in u:
            m = re.search(r"=w(\d+)|=s(\d+)", u)
            if m and int(m.group(1) or m.group(2)) >= 400:
                sniffed.append(u)

    page.on("response", _sniff)
    try:
        # links from the overview panel
        try:
            if not with_menu:
                raise _Skip
            el = page.locator('a[data-item-id="menu"]').first
            if await el.count():
                out["menu_link"] = await el.get_attribute("href")
            ws = page.locator('a[data-item-id="authority"]').first
            if await ws.count():
                out["website"] = await ws.get_attribute("href")
        except _Skip:
            pass
        except Exception as e:
            out["errors"].append(f"links: {e}")

        # menu tab: filmstrip srcs + dated walk
        try:
            tab = page.get_by_role("tab", name=re.compile("^Menu", re.I)).first
            if with_menu and await tab.count():
                await tab.click()
                await page.wait_for_timeout(2_500)
                thumbs = page.locator('button[aria-label^="Photo "]')
                if await thumbs.count():
                    aria0 = await thumbs.first.get_attribute("aria-label") or ""
                    m = re.search(r"of (\d+)", aria0)
                    total = min(int(m.group(1)) if m else await thumbs.count(), photo_cap)
                    await thumbs.first.click()
                    menu_walked = True
                    await page.wait_for_timeout(2_000)
                    strip = await page.evaluate(_EVAL_STRIP_SRCS)
                    dates = []
                    for _ in range(max(total, min(len(strip), photo_cap))):
                        got = await page.evaluate(_EVAL_CURRENT_ITEM)
                        dates.append(got.get("cap") or got.get("attr"))
                        await _click_next(page)
                        await page.wait_for_timeout(1_200)
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(800)
                    out["menu_photos"] = [
                        {"url": s, "date": dates[i] if i < len(dates) else None}
                        for i, s in enumerate(strip[:photo_cap])]
        except Exception as e:
            out["errors"].append(f"menu_tab: {e}")

        # place photos: re-enter gallery (menu thumbs if present, else cover)
        try:
            entered = False
            # Menu thumbs only exist after a menu walk; without one, a "Photo "
            # button on the Reviews pane is a reviewer photo, so go via the cover.
            thumbs2 = page.locator('button[aria-label^="Photo "]')
            if menu_walked and await thumbs2.count():
                await thumbs2.first.click()
                await page.wait_for_timeout(2_000)
                entered = True
                out["entry"] = "menu_thumbs"
            else:
                ov = page.get_by_role("tab", name=re.compile("^Overview", re.I)).first
                # Screenshot capture already leaves the page on Overview, and on the
                # fleets a pane overlay can intercept the click for the full 30s.
                if await ov.count() and await ov.get_attribute("aria-selected") != "true":
                    # In-page click: a pointer click lands on the overlay instead.
                    await ov.evaluate("e => e.click()")
                    await page.wait_for_timeout(2_000)
                # "See photos" opens the FULL gallery (viewer on item 1, Next walks
                # every photo). The hero image can open a ~10-item preview carousel
                # instead — Akansha Medical: hero 10, See photos 29. Hero is only the
                # fallback. Never a bare "Photo of ...": that matches reviewer avatars.
                see = page.locator("button:has-text('See photos')").first
                cover = page.locator('button[jsaction*="heroHeaderImage"]').first
                for name, entry, wait_ms in (("see_photos", see, 3_000), ("hero", cover, 2_500)):
                    if await entry.count():
                        await entry.evaluate("e => e.click()")   # overlay-proof, see above
                        await page.wait_for_timeout(wait_ms)
                        entered = True
                        out["entry"] = name   # hero = possible ~10-item preview
                        break
            if entered:
                allchip = page.get_by_role("tab", name=re.compile("^All$", re.I)).first
                if await allchip.count():
                    await allchip.click()
                    await page.wait_for_timeout(2_000)
                out["place_photos"] = await _walk_viewer(page, sniffed, place_photo_cap)
                await page.keyboard.press("Escape")
            real = sum(1 for p in out["place_photos"] if p["kind"] in ("photo", "video"))
            sv = sum(1 for p in out["place_photos"] if p["kind"] == "street_view")
            out["imagery"] = ("photos" if real else "street_view_only") \
                if (real or sv) else "none"
        except Exception as e:
            out["errors"].append(f"place_photos: {e}")
            out["imagery"] = out["imagery"] or "unknown"
    finally:
        try:
            page.remove_listener("response", _sniff)
        except Exception:
            pass
    return out
