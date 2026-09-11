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
PLACE_PHOTO_CAP_DEFAULT = 0   # place-photo date walk: 0 = every item in the gallery
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


async def _walk_viewer(page, sniffed: list[str], cap_items: int) -> list[dict]:
    """Walk the open gallery viewer, one dict per item: {url, date, kind}.
    cap_items=0 walks to the end (3 repeated slides = end of gallery)."""
    items: list[dict] = []
    seen: set[str] = set()
    stale = 0
    for _ in range(cap_items * 2 if cap_items else _WALK_GUARD):
        got = await page.evaluate(_EVAL_CURRENT_ITEM)
        src, cap, attr = got.get("img"), got.get("cap"), got.get("attr")
        if not src and sniffed:
            src = sniffed[-1]
        is_video = bool(cap and cap.lower().startswith("video"))
        if is_video:
            src = None
        sig = f"{cap}|{got.get('vid')}|{(src or '').split('=')[0]}"
        if sig in seen:
            stale += 1
            if stale >= 3:
                break
        else:
            seen.add(sig)
            stale = 0
            if is_video:
                items.append({"url": None, "date": cap, "kind": "video"})
            elif src:
                is_sv = "streetviewpixels" in src or (not cap and bool(attr))
                items.append({"url": src, "date": cap or attr,
                              "kind": "street_view" if is_sv else "photo"})
            elif attr:
                # Cover is a Street View panorama (canvas, no <img>, no Next): the
                # listing has no photo gallery. Keep the capture date and stop —
                # arrow keys here move the camera, not the gallery.
                items.append({"url": None, "date": attr, "kind": "street_view"})
                break
        if cap_items and len(items) >= cap_items:
            break
        n_before = len(sniffed)
        await _click_next(page)
        for _ in range(8):
            await page.wait_for_timeout(500)
            if len(sniffed) > n_before:
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
            else:
                ov = page.get_by_role("tab", name=re.compile("^Overview", re.I)).first
                # Screenshot capture already leaves the page on Overview, and on the
                # fleets a pane overlay can intercept the click for the full 30s.
                if await ov.count() and await ov.get_attribute("aria-selected") != "true":
                    # In-page click: a pointer click lands on the overlay instead.
                    await ov.evaluate("e => e.click()")
                    await page.wait_for_timeout(2_000)
                # Hero image only. A bare "Photo of ..." also matches reviewer
                # avatars, which open a profile, not the gallery.
                cover = page.locator('button[jsaction*="heroHeaderImage"]').first
                if await cover.count():
                    await cover.evaluate("e => e.click()")   # overlay-proof, see above
                    await page.wait_for_timeout(2_500)
                    entered = True
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
