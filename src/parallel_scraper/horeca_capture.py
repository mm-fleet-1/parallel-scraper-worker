"""HoReCa capture: menu photos + place photos with posted dates + imagery class.

Runs on the PlaywrightSession's live page AFTER the metadata scrape (the page is
already on the place panel with warm cookies). Ported from tools/menu_probe.py,
which was validated on 151 Riyadh outlets (see menu_batch_riyadh.jsonl).

Hard-won behaviors baked in (do not "simplify" these away):
- The gallery viewer's main image is a <canvas>, not a queryable <img>. The item's
  own image URL is in the viewer URL (`!6s<url>`) next to its id (`!1s<id>`), so
  the key and the image come from ONE read of location.href (checked live, Oct 2026).
- Advancing slides works reliably via the visible "Next" button; ArrowRight
  does not always reach the viewer.
- Caption in the top-left title card ("Photo - Jun 2026" / "Video - Jul 2026") is
  the item's date; the bottom "Image capture:" bar is attribution (the date for
  street-view items). Both are month precision only. The title card is ONE node
  for all slides; it re-renders ~250 ms BEFORE the URL key changes (measured), so
  "caption belongs to the new key" = the title card mutated after the click AND
  the key changed (design §9.1 item rule).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from playwright.async_api import TimeoutError as PWTimeout

logger = logging.getLogger(__name__)

PHOTO_CAP_DEFAULT = 12        # menu cards (horeca=1 only)
# Walk limits per chip walk (design §5). One default each; env overrides are read
# by phase2_metadata.media_capture. Photo cap 0 = no photo cap.
WALK_PHOTO_CAP_DEFAULT = 40
WALK_MAX_ITEMS_DEFAULT = 120
WALK_MAX_S_DEFAULT = 90
OWNER_PHOTO_CAP = 21          # the 21st photo tells "20 or fewer" from "more" (§5)
PLACE_PHOTO_CAP_DEFAULT = WALK_PHOTO_CAP_DEFAULT   # old name, kept for importers

# Next -> item ready, 267 items on 12 slow p200 outlets (Oct 2026): p50 0.29 s, p95 1.0 s,
# p99 1.2 s, max 3.9 s; so a 2 s fast wait, then the same wait extended up to the old
# 8 s ceiling WITHOUT re-clicking (a delayed first click plus a second one skips a
# photo) before an item is unknown; ITEM_STUCK unknowns in a row end that chip walk
# (p200: an unmatched item kind cost 8 s each until max_s). ITEM_RETRY=1 re-clicks Next
# when the key did not move (experiment only).
ITEM_TIMEOUT_MS = int(os.environ.get("PHASE2_ITEM_TIMEOUT_MS") or 2_000)
ITEM_MAX_MS = 8_000           # an item's whole wait from the Next click
TILE0_TIMEOUT_MS = 8_000      # a chip's first tile opened by click (not re-measured)
ITEM_RETRY = int(os.environ.get("PHASE2_ITEM_RETRY") or 0)
ITEM_STUCK = int(os.environ.get("PHASE2_ITEM_STUCK") or 3)
CHIP_TIMEOUT_MS = 2_000       # today's chip sleep, now a ceiling
LAZY_TIMEOUT_MS = 4_000       # today's lazy-load ceiling (8 x 500 ms)
LAZY_TRIES = 3                # scroll + wait attempts inside LAZY_TIMEOUT_MS
CHIP_FIRST_MS = 1_500         # chip's first item by key change, before the tile-0 click
MENU_CHIP_CAP = 21            # Menu-chip photos walked only to tag them (never mirrored)

# ONE read of the current viewer item (§9.1 item rule). Also installs, once per
# page, a MutationObserver that counts title-card re-renders (tc_gen).
_SNAP = r"""() => {
    const W = window, TC = '[jsaction*="titlecard"]';
    if (!W.__psTc) {
        W.__psTc = {gen: 0};
        try {
            new MutationObserver(ms => {
                for (const m of ms) {
                    const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
                    let hit = !!(t && t.closest && t.closest(TC));
                    if (!hit) for (const n of m.addedNodes)
                        if (n.nodeType === 1 && (n.matches(TC) || n.querySelector(TC))) { hit = true; break; }
                    if (hit) { W.__psTc.gen++; break; }
                }
            }).observe(document.body, {subtree: true, childList: true, characterData: true});
        } catch (e) {}
    }
    const dec = s => { try { return decodeURIComponent(s); } catch (e) { return s; } };
    const href = location.href, data = href.split('/data=')[1] || '';
    const km = /!1s([^!?]+)/.exec(data), sm = /!6s([^!?]+)/.exec(data);
    const key = km ? dec(km[1]) : href;
    let thumb = sm ? dec(sm[1]) : null;
    if (thumb && thumb.startsWith('//')) thumb = 'https:' + thumb;
    // The item's image is the one carrying the current key's !6s id; a neighbour that is
    // loaded (or larger) must never pass for it. No !6s (a lone Street View pano): the
    // streetviewpixels image only.
    const sid = thumb ? ((/panoid=([^&]+)/.exec(thumb) || [])[1] || thumb.split('=')[0].split('/').pop()) : null;
    let im = null;
    for (const i of document.querySelectorAll('img[src*="googleusercontent"], img[src*="streetviewpixels"]')) {
        const r = i.getBoundingClientRect();
        if (r.width < 250 || r.height < 250) continue;
        if (sid ? i.src.includes(sid) : i.src.includes('streetviewpixels')) { im = i; break; }
    }
    let src = im ? im.src : thumb, complete = false, nw = 0;
    if (im) { complete = im.complete; nw = im.naturalWidth; }
    else if (src) {
        if (!W.__psPre || Object.keys(W.__psPre).length > 300) W.__psPre = {};
        let p = W.__psPre[src];
        if (!p) { p = new Image(); p.src = src; W.__psPre[src] = p; }
        complete = p.complete; nw = p.naturalWidth;
    }
    const leaf = (root, re, maxTop) => {
        for (const el of root.querySelectorAll('div,span')) {
            const t = (el.textContent || '').trim();
            if (el.children.length !== 0 || t.length > 45 || !re.test(t)) continue;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0 || r.top >= maxTop) continue;
            return [el, t];
        }
        return [null, null];
    };
    const tc = document.querySelector(TC);
    const [capEl, cap] = leaf(tc || document, /^(Photo|Video)\s*[-–]/, 200);
    const [, attr] = leaf(document, /^Image capture:/i, 1e9);
    return {viewer_url_key: key, img_src: src, caption: cap, attr,
            slide_key: capEl && tc && tc.contains(capEl) ? key : null,
            tc_gen: W.__psTc.gen, img_complete: complete, natural_width: nw, href};
}"""

# Ready predicate for wait_for_function: returns the snapshot when ready.
# a = the snapshot taken just before the click (viewer_url_key, img_src, tc_gen).
_ITEM_READY = "(a) => { const s = (" + _SNAP + r""")();
    // First item of a new chip can be the photo already shown (owner #1 == All #1,
    // seen live): the key never changes and the title card need not re-render. Only
    // after the new grid's first tile was clicked (a.same_ok) is an unchanged key that
    // item. Grid thumb URLs are no proof: one photo comes under different URL paths.
    if (a.same_ok && s.viewer_url_key === a.viewer_url_key) {
        if (s.caption && s.slide_key === s.viewer_url_key
            && (/^video/i.test(s.caption) || (s.img_complete && s.natural_width > 0))) return s;
        return (!s.caption && s.attr && s.img_complete) ? s : false;
    }
    if (s.viewer_url_key === a.viewer_url_key) return false;
    // A key an earlier chip walk already recorded: the caller drops it (owner/menu wins),
    // and its title card often does not re-render on a revisit (live: a Menu item met
    // again in All timed out with key, image and caption all ready).
    if (a.known && a.known.includes(s.viewer_url_key)) return s;
    const slideOk = s.caption && s.slide_key === s.viewer_url_key;
    const capOk = slideOk && s.tc_gen > a.tc_gen;
    if (s.caption && /^video/i.test(s.caption)) return capOk ? s : false;   // §6: no video wait
    if (!s.img_src || s.img_src === a.img_src || !s.img_complete || !(s.natural_width > 0)) return false;
    // a.same_cap (extended wait only): the item before had the same caption text, so the
    // title card need not re-render (live: 3 "Photo - Apr 2024" in a row ended a walk).
    // Accepted only with the new key AND a new, loaded image, from this one snapshot.
    if (s.caption) return (capOk || (a.same_cap && slideOk && s.caption === a.caption)) ? s : false;
    if (s.attr) return s;           // Street View: only the "Image capture" bar
    // Partner photo (lh3 /gps-proxy/): never a caption nor an attribution bar. Live
    // (Oct 2026): every 8 s item timeout on the 90 s p200 walks was one of these.
    return /\/gps-proxy\//.test(s.img_src) ? s : false;
}"""

# Why an item wait timed out: which part of the item rule still fails against `a`.
_TIMEOUT_DIAG = "(a) => { const s = (" + _SNAP + r""")();
    const b = document.querySelector('button[aria-label*="Next" i]');
    return {key_changed: s.viewer_url_key !== a.viewer_url_key, src_changed: s.img_src !== a.img_src,
            has_src: !!s.img_src, complete: s.img_complete, nw: s.natural_width, cap: s.caption,
            slide_ok: s.slide_key === s.viewer_url_key, tc_delta: s.tc_gen - a.tc_gen, attr: !!s.attr,
            next: b ? (b.disabled || b.getAttribute('aria-disabled') === 'true' ? 'off' : 'on') : 'none',
            src: (s.img_src || '').slice(0, 60)};
}"""

_GRID = r"""() => {
    const t = [...document.querySelectorAll('[data-photo-index]')];
    const k = el => {
        const b = el.querySelector('[style*="background-image"]') || el;
        const m = /url\(["']?([^"')]+)/.exec(b.getAttribute('style') || '');
        if (m) return m[1].split('=')[0];
        const im = el.querySelector('img');
        return im ? im.src.split('=')[0] : null;
    };
    return {count: t.length, keys: t.slice(0, 3).map(k)};
}"""

# Chip readiness (§9.1): first 3 grid keys differ from a.prev AND count stable 500 ms.
_CHIP_READY = "(a) => { const g = (" + _GRID + r""")(), ks = JSON.stringify(g.keys);
    if (a.need_change && ks === a.prev) { window.__psChip = null; return false; }
    const now = performance.now(), st = window.__psChip;
    if (!st || st.n !== g.count || st.k !== ks) { window.__psChip = {n: g.count, k: ks, t: now}; return false; }
    return now - st.t >= 500 ? g : false;
}"""

_TABS = r"""() => [...document.querySelectorAll('[role=tab]')]
    .filter(t => { const r = t.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
    .map(t => [(t.getAttribute('aria-label') || t.textContent || '').trim(),
               t.getAttribute('aria-selected') === 'true'])"""

_EVAL_STRIP_SRCS = """() => {
    const srcs = [];
    document.querySelectorAll('img[src*="googleusercontent"]').forEach(im => {
        const r = im.getBoundingClientRect();
        if (r.width >= 60 && r.width <= 200) srcs.push(im.src);
    });
    return [...new Set(srcs)];
}"""

_NEXT_DISABLED = """() => {
    const b = document.querySelector('button[aria-label*="Next" i]');
    return !!(b && (b.disabled || b.getAttribute('aria-disabled') === 'true'));
}"""

# Gallery chips render after the first item: wait for a tab not in `before`.
_NEW_TABS = r"""(before) => [...document.querySelectorAll('[role=tab]')].some(t => {
    const r = t.getBoundingClientRect(), l = (t.getAttribute('aria-label') || t.textContent || '').trim();
    return r.width > 0 && r.height > 0 && l && !before.includes(l); })"""

_VIEWER_CLOSED ="""() => !document.querySelector('button[aria-label*="Next" i]')"""

_ENTRY_PRESENT = """() => !!document.querySelector('button[jsaction*="heroHeaderImage"]')
    || [...document.querySelectorAll('button')].some(b => (b.textContent || '').includes('See photos'))"""

# Bring the LAST loaded thumbnail into view and push its scroll container to the
# bottom: the grid lazy-loads on intersection (20 -> 22 seen with the viewer open).
# Escape is not an option — it tears the whole gallery down (0 thumbnails left).
_SCROLL_GRID = """() => {
    const t = [...document.querySelectorAll('[data-photo-index]')];
    if (!t.length) return 0;
    const last = t[t.length - 1];
    last.scrollIntoView({block: 'end'});
    for (let p = last.parentElement; p; p = p.parentElement)
        if (p.scrollHeight > p.clientHeight + 1 && /(auto|scroll)/.test(getComputedStyle(p).overflowY)) {
            p.scrollTop = p.scrollHeight; break; }
    return t.length;
}"""

_GRID_COUNT = "() => document.querySelectorAll('[data-photo-index]').length"

# Google's logged-out "limited view" (live Oct 2026, ~100% of sessions by late Oct):
# a fixed 10-tile gallery, no chips, and a "See more photos" button that opens a
# sign-in dialog. Server-side: no relaunch, cookie seed, UA, hl/gl or URL form lifts it.
_LIMITED_VIEW = r"""() => !!document.querySelector('button[aria-label="See more photos"]')
    || /limited view of Google Maps/i.test(document.body.innerText)"""

# The panel's photo total ("13 Photos" button on Overview): what the walk could not reach.
_PHOTO_TOTAL = r"""() => { for (const b of document.querySelectorAll('button')) {
    const m = /^\s*([\d,]+)\s+Photos?\s*$/i.exec(b.textContent || '');
    if (m) return parseInt(m[1].replace(/,/g, ''), 10); } return null; }"""

_CLICK_TILE0 = """() => { const t = document.querySelector('[data-photo-index="0"]');
    if (t) t.click(); return !!t; }"""

_GROWN = """(n) => {
    if (document.querySelectorAll('[data-photo-index]').length > n) return true;
    const b = document.querySelector('button[aria-label*="Next" i]');
    return !!(b && !b.disabled && b.getAttribute('aria-disabled') !== 'true');
}"""


class _Skip(Exception):
    """Section switched off — not an error, nothing recorded."""


def _ms_left(deadline: float | None, cap_ms: int) -> int:
    if deadline is None:
        return cap_ms
    return max(1, min(cap_ms, int((deadline - time.monotonic()) * 1000)))


async def _snap(page) -> dict:
    return await page.evaluate(_SNAP)


async def _wait_item(page, prev: dict, timeout_ms: int) -> dict | None:
    """Wait (50 ms poll) until the item rule holds against `prev`; None = timeout."""
    try:
        h = await page.wait_for_function(_ITEM_READY, arg=prev, polling=50,
                                         timeout=max(1, timeout_ms))
        return await h.json_value()
    except PWTimeout:
        return None


async def _wait_js(page, expr: str, arg=None, timeout_ms: int = 1_000) -> bool:
    try:
        await page.wait_for_function(expr, arg=arg, polling=50, timeout=max(1, timeout_ms))
        return True
    except PWTimeout:
        return False


async def _click_next(page, timeout_ms: int = 3_000) -> None:
    nxt = page.locator('button[aria-label*="Next" i]').first
    try:
        if await nxt.count():
            # In-page click: a pane overlay ate pointer clicks for the full timeout
            # (3 s per item, seen live Oct 2026), same as the entry clicks below.
            await nxt.evaluate("e => e.click()", timeout=timeout_ms)
            return
    except Exception:
        pass
    await page.keyboard.press("ArrowRight")


async def _load_more(page, timeout_ms: int = LAZY_TIMEOUT_MS) -> bool:
    """The grid lazy-loads thumbnails in batches and the viewer can only page through
    loaded ones, so Next turns off at the last loaded tile until the grid is scrolled.
    Scroll it and wait for more tiles or Next back on, LAZY_TRIES times inside one
    ceiling; False = no growth (the end, or a stall the caller judges with _stalled).
    Live (Oct 2026): the DEGRADED render is Google's logged-out "limited view"
    (_LIMITED_VIEW): a fixed 10-tile preview that never grows, whatever is scrolled.
    Early Oct a relaunch got the full gallery back ~25% of the time; by late Oct the
    gate covers ~every session and only a signed-in session shows more."""
    budget = min(LAZY_TIMEOUT_MS, timeout_ms)
    for _ in range(LAZY_TRIES):
        n = await page.evaluate(_SCROLL_GRID)
        if not n:
            return False
        if await _wait_js(page, _GROWN, n, max(1, budget // LAZY_TRIES)):
            return True
    return False


def _classify(s: dict) -> dict:
    """Ready snapshot -> item fields. Video: caption prefix. 360: Photo caption with
    '360' or a streetviewpixels image. Street View: no Photo/Video caption."""
    cap, src = s.get("caption"), s.get("img_src")
    sv_img = bool(src and "streetviewpixels" in src)
    it = {"url": src, "caption": cap, "viewer_key": s.get("viewer_url_key"), "pano_url": None}
    if cap and cap.lower().startswith("video"):
        it.update(kind="video", url=None)
    elif cap and ("360" in cap or sv_img):
        it["kind"] = "360"
    elif not cap and src and "/gps-proxy/" in src:
        it["kind"] = "photo"          # partner photo: undated, never Street View
    elif not cap:
        it.update(kind="street_view", caption=s.get("attr"), pano_url=s.get("href"))
    else:
        it["kind"] = "photo"
    return it


_UNKNOWN = {"kind": "unknown", "url": None, "caption": None, "viewer_key": None, "pano_url": None}


async def _walk_viewer(page, cur: dict | None, *, photo_cap: int, max_items: int,
                       deadline: float, known=()) -> tuple[list[dict], dict]:
    """Walk the open viewer from `cur` (the ready snapshot of the shown item, None =
    it timed out). Returns items and {limits_hit, unknown, photos}. Ends on: the real
    end of the gallery, a wrap to a seen key, or a limit (§5)."""
    items: list[dict] = []
    seen: set[str] = set()
    hit: list[str] = []
    photos = unknown = 0
    item_ms: list[int] = []   # per Next: ms to ready, negative = timed out
    stuck = 0                 # unknowns in a row
    why = "limit"
    end_tiles = None     # grid tile count when the walk ended on Next off + no growth
    while True:
        if cur is None:
            items.append(dict(_UNKNOWN))
            unknown += 1
        else:
            if cur["viewer_url_key"] in seen:
                why = "wrap"
                break
            seen.add(cur["viewer_url_key"])
            it = _classify(cur)
            items.append(it)
            photos += it["kind"] == "photo"
        if not await page.locator('button[aria-label*="Next" i]').count():
            why = "no_next"
            break    # a lone Street View panorama: arrow keys move the camera
        if await page.evaluate(_NEXT_DISABLED) and \
                not await _load_more(page, _ms_left(deadline, LAZY_TIMEOUT_MS)):
            if time.monotonic() >= deadline:
                hit.append("max_s")      # the lazy load ran out of walk time
            else:
                end_tiles = await page.evaluate(_GRID_COUNT)
                why = f"end tiles={end_tiles}"
            break
        if photo_cap and photos >= photo_cap:
            hit.append("photo_cap")
            break
        if len(items) >= max_items:
            hit.append("max_items")
            break
        if time.monotonic() >= deadline:
            hit.append("max_s")
            break
        prev = {**await _snap(page), "known": list(known)}
        t_i = time.monotonic()
        await _click_next(page, _ms_left(deadline, 3_000))
        cur = await _wait_item(page, prev, _ms_left(deadline, ITEM_TIMEOUT_MS))
        if cur is None and ITEM_RETRY and time.monotonic() < deadline:
            d = await page.evaluate(_TIMEOUT_DIAG, prev) or {}
            if not d.get("key_changed"):   # experiment: assume the click was lost
                await _click_next(page, _ms_left(deadline, 3_000))
        rest = ITEM_MAX_MS - int((time.monotonic() - t_i) * 1000)
        if cur is None and rest > 0 and time.monotonic() < deadline:
            cur = await _wait_item(page, {**prev, "same_cap": True}, _ms_left(deadline, rest))
        if cur is None:
            d = await page.evaluate(_TIMEOUT_DIAG, prev)
            logger.info("walk.item_timeout n=%d ms=%d diag=%s", len(items),
                        int((time.monotonic() - t_i) * 1000), json.dumps(d))
        item_ms.append(int((time.monotonic() - t_i) * 1000) * (1 if cur else -1))
        stuck = stuck + 1 if cur is None else 0
        if stuck >= ITEM_STUCK > 0:
            items.append(dict(_UNKNOWN))
            unknown += 1
            hit.append("item_stuck")
            break
        if cur is None and time.monotonic() >= deadline:
            items.append(dict(_UNKNOWN))     # the failed step still counts (item_timeout)
            unknown += 1
            hit.append("max_s")
            break
    logger.info("walk.end reason=%s items=%d unknown=%d hit=%s", why, len(items), unknown, hit)
    return items, {"limits_hit": hit, "unknown": unknown, "photos": photos,
                   "end_tiles": end_tiles, "item_ms": item_ms}


def _stalled(n_items: int, end_tiles: int | None, chips: bool) -> bool:
    """lazy_load_stall: the walk stopped before the grid's tile count, or (no chips)
    on a full 10-tile preview, which is the degraded render (see _load_more)."""
    if end_tiles is None:
        return False
    # ponytail: a real 10+ item gallery shown without chips is flagged too and the
    # repair pass re-walks it; use the page's photo total once the page shows one.
    return n_items < end_tiles or (not chips and end_tiles >= 10)


_PANEL_TAB = re.compile(r"^(Overview|Reviews|About)\b")


async def _list_gallery_chips(page, before: set[str] = frozenset()) -> list[str]:
    """Visible gallery chip labels. Panel tabs are hidden while the gallery is open;
    excluding by the label set seen `before` also dropped the gallery's "Menu" chip
    (the panel has a "Menu" tab too; p10: 0 Menu chips seen), so exclude by pattern."""
    return [lbl for lbl, _ in await page.evaluate(_TABS) if lbl and not _PANEL_TAB.match(lbl)]


async def _walk_tab(page, label: str | None, photo_cap: int, max_items: int, max_s: int,
                    first_click: bool = False, entry: dict | None = None, known=()
                    ) -> tuple[list[dict], dict] | None:
    """Select chip `label` (None = walk the stream already shown), wait for chip
    readiness (§9.1), walk it. None = the chip click failed or chip readiness timed
    out (walk skipped). `entry` = the ready snapshot of the item shown on gallery
    entry (None = the entry wait timed out); it is the first item when no chip is
    clicked, so a failed entry wait gives an unknown first item, never stale content."""
    t0 = time.monotonic()
    deadline = t0 + max_s
    prev = {**await _snap(page), "known": list(known)}
    need_change = False
    if label is not None:
        g0 = await page.evaluate(_GRID)
        tabs = dict(await page.evaluate(_TABS))
        # The first chip may already be the open stream (All on entry): its keys can
        # not change, so only grid stability is checked then (never aria alone).
        need_change = not (first_click and tabs.get(label))
        await page.evaluate("() => { window.__psChip = null; }")
        try:
            if need_change:   # intercepted / timed-out click = chip unavailable, not a crash
                await page.get_by_role("tab", name=label, exact=True).first.click(
                    timeout=_ms_left(deadline, 3_000))
            h = await page.wait_for_function(
                _CHIP_READY, arg={"prev": json.dumps(g0["keys"]), "need_change": need_change},
                polling=50, timeout=_ms_left(deadline, CHIP_TIMEOUT_MS))
            grid = await h.json_value()
        except Exception:
            return None
        if not grid["count"]:       # ok-empty: the chip really has no items
            return [], {"limits_hit": [], "unknown": 0, "photos": 0,
                        "ms": int((time.monotonic() - t0) * 1000)}
    cur = entry
    if need_change:
        # The viewer jumps to the chip's first item (~0.4 s live). If its key did not
        # change, the first item may be the photo already shown: open the new grid's
        # first tile, then an unchanged key IS that item (§9.1 finding 4).
        cur = await _wait_item(page, prev, _ms_left(deadline, CHIP_FIRST_MS))
        if cur is None and await page.evaluate(_CLICK_TILE0):
            cur = await _wait_item(page, {**prev, "same_ok": True},
                                   _ms_left(deadline, TILE0_TIMEOUT_MS))
    items, meta = await _walk_viewer(page, cur, photo_cap=photo_cap, max_items=max_items,
                                     deadline=deadline, known=known)
    meta["ms"] = int((time.monotonic() - t0) * 1000)
    return items, meta


def _new_walk_meta(entry: str | None) -> dict:
    return {"walk_ok": False, "reason": None, "owner_count": 0, "owner_count_known": False,
            "truncated": 0, "limits_hit": [], "truncated_reasons": [], "chip_unavailable": 0,
            "latest_ordered": None, "video_count": 0, "pano_count": 0, "unknown_count": 0,
            "chips_seen": [], "timings": {}, "entry": entry, "menu_count": 0,
            "degraded_gallery": 0, "limited_view": 0, "page_photo_total": None}


async def _walk_gallery(page, chips: list[str], wm: dict, photo_cap: int, max_items: int,
                        max_s: int, entry: dict | None = None) -> list[dict]:
    """Owner walk, then Latest (else All) walk (§2, §2.2). Fills `wm`; returns the
    merged items. Items are keyed by viewer key: an item seen in the owner walk is
    not added again (owner wins, §2), so a later thumb fetch happens once."""
    def pick(rx):
        return next((c for c in chips if re.match(rx, c, re.I)), None)

    owner, latest, allc = pick(r"^By owner"), pick(r"^Latest"), pick(r"^All$")
    menu = pick(r"^Menu$")
    items: list[dict] = []
    keys: set[str] = set()

    def add(walked, source, chip, meta):
        wm["timings"][chip or "current"] = meta["ms"]
        wm.setdefault("item_ms", {})[chip or "current"] = meta.get("item_ms", [])
        for h in meta["limits_hit"]:
            if h not in wm["limits_hit"]:
                wm["limits_hit"].append(h)
        if meta["unknown"] and "item_timeout" not in wm["truncated_reasons"]:
            wm["truncated_reasons"].append("item_timeout")
        if _stalled(len(walked), meta.get("end_tiles"), bool(chips)):
            if "lazy_load_stall" not in wm["truncated_reasons"]:
                wm["truncated_reasons"].append("lazy_load_stall")
            if not chips:
                wm["degraded_gallery"] = 1
        for it in walked:
            k = it["viewer_key"]
            if k and k in keys:
                continue
            if k:
                keys.add(k)
            items.append({**it, "source": source, "chip": chip, "ordinal": len(items)})

    # ONE budget (time, items, photos) for every chip walk of this gallery, from the
    # first one: the per-place worst case is a single walk's. Menu-chip photos are
    # tagged, never mirrored, so they spend time and items but not the photo cap.
    end, left = time.monotonic() + max_s, {"photos": photo_cap, "items": max_items}

    def budget(cap: int, photos: bool = True):
        """(photo cap, item cap, seconds) for the next chip walk; None = spent (recorded)."""
        s_left = end - time.monotonic()
        out = ("photo_cap" if photos and photo_cap and left["photos"] <= 0 else
               "max_items" if left["items"] <= 0 else "max_s" if s_left <= 0 else None)
        if out:
            if out not in wm["limits_hit"]:
                wm["limits_hit"].append(out)
            return None
        if photos and photo_cap:
            cap = min(cap, left["photos"]) if cap else left["photos"]
        return cap, left["items"], s_left

    def spend(res, photos: bool = True):
        left["items"] -= len(res[0])
        if photos:
            left["photos"] -= res[1]["photos"]

    clicked = False
    b = budget(OWNER_PHOTO_CAP) if owner else None
    if b:
        res = await _walk_tab(page, owner, *b, first_click=True, entry=entry)
        clicked = True
        if res is None:
            wm["chip_unavailable"] = 1
        else:
            spend(res)
            add(res[0], "owner", owner, res[1])
            wm["owner_count"] = res[1]["photos"]
            wm["owner_count_known"] = not res[1]["limits_hit"] and not res[1]["unknown"]
    elif not owner:
        # Chips listed without "By owner": no owner photos. No chips at all (the
        # gallery sometimes renders without them, seen live): owner count unknown.
        wm["owner_count_known"] = bool(chips)
    b = budget(MENU_CHIP_CAP, photos=False) if menu else None
    if b:
        # Menu chip: walked before the gallery so its photos keep chip="Menu" (a key
        # is added once) and gallery_select skips them. Not a gallery walk.
        res = await _walk_tab(page, menu, *b, first_click=not clicked, entry=entry, known=keys)
        clicked = True
        if res is not None:
            spend(res, photos=False)
            add(res[0], "menu", menu, res[1])
            wm["menu_count"] = res[1]["photos"]
    ran = False
    for chip in (c for c in (latest, allc) if c):
        b = budget(photo_cap)
        if not b:
            break
        res = await _walk_tab(page, chip, *b, first_click=not clicked, entry=entry, known=keys)
        clicked = True
        if res is None:
            wm["chip_unavailable"] = 1        # skipped Latest falls back to All
            continue
        spend(res)
        add(res[0], "gallery", chip, res[1])
        ran = True
        if chip is latest and set(res[1]["limits_hit"]) <= {"item_stuck"}:
            # Latest is only the recent items (live: 1 tile on a 100+ photo cafe, p200
            # kept 1 photo): walked to its end (or stuck), All follows for the rest
            # (keys dedup) within what is left of the budget.
            continue
        break
    if not (owner or latest or allc):
        # No chips at all (seen live, Oct 2026): the open stream is the gallery.
        res = await _walk_tab(page, None, photo_cap, max_items, max_s, entry=entry)
        add(res[0], "gallery", None, res[1])
        ran = True
    wm["walk_ok"] = ran
    if not ran:
        spent = [h for h in ("photo_cap", "max_items", "max_s") if h in wm["limits_hit"]]
        wm["reason"] = "budget_spent" if spent and (latest or allc) else "chip_unavailable"
    for h in wm["limits_hit"]:
        wm["truncated_reasons"].append(h)
    wm["truncated"] = int(bool(wm["truncated_reasons"]))
    wm["video_count"] = sum(i["kind"] == "video" for i in items)
    wm["pano_count"] = sum(i["kind"] == "360" for i in items)
    wm["unknown_count"] = sum(i["kind"] == "unknown" for i in items)
    return items


async def capture_horeca(page, photo_cap: int = PHOTO_CAP_DEFAULT, with_menu: bool = True,
                         place_photo_cap: int = WALK_PHOTO_CAP_DEFAULT,
                         walk_max_items: int = WALK_MAX_ITEMS_DEFAULT,
                         walk_max_s: int = WALK_MAX_S_DEFAULT) -> dict:
    """Capture menu/photo/date/imagery data from the place panel currently on
    `page`. Never raises — every section degrades to empty + an errors entry.

    with_menu=False skips links + the Menu tab and only walks the place-photo
    gallery (the always-on photo-date capture). Order is unchanged when on:
    menu first, then the gallery re-entered through the menu thumbs.
    place_photo_cap / walk_max_items / walk_max_s: limits per chip walk (§5).
    Adds `items` (owner + gallery walks, merged by viewer key) and `walk_meta`."""
    out: dict = {"menu_link": None, "website": None, "menu_photos": [],
                 "place_photos": [], "imagery": None, "errors": [], "items": [],
                 "walk_meta": _new_walk_meta(None)}
    menu_walked = False

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
            thumbs = page.locator('button[aria-label^="Photo "]')
            try:
                await thumbs.first.wait_for(state="visible", timeout=2_500)
            except PWTimeout:
                pass
            if await thumbs.count():
                aria0 = await thumbs.first.get_attribute("aria-label") or ""
                m = re.search(r"of (\d+)", aria0)
                total = min(int(m.group(1)) if m else await thumbs.count(), photo_cap)
                prev = await _snap(page)
                await thumbs.first.click()
                menu_walked = True
                cur = await _wait_item(page, prev, 2_000)
                strip = await page.evaluate(_EVAL_STRIP_SRCS)
                dates = []
                n = max(total, min(len(strip), photo_cap))
                for i in range(n):
                    dates.append((cur.get("caption") or cur.get("attr")) if cur else None)
                    if i + 1 < n:
                        prev = await _snap(page)
                        await _click_next(page)
                        cur = await _wait_item(page, prev, 1_200)
                await page.keyboard.press("Escape")
                await _wait_js(page, _VIEWER_CLOSED, timeout_ms=800)
                out["menu_photos"] = [
                    {"url": s, "date": dates[i] if i < len(dates) else None}
                    for i, s in enumerate(strip[:photo_cap])]
    except Exception as e:
        out["errors"].append(f"menu_tab: {e}")

    # place photos: enter the gallery (menu thumbs if present, else See photos / hero)
    wm = out["walk_meta"]
    try:
        before = {lbl for lbl, _ in await page.evaluate(_TABS)}
        pre = await _snap(page)          # key before the viewer opens (§9.1)
        wm["page_photo_total"] = await page.evaluate(_PHOTO_TOTAL)   # any pane (menu path too)
        entered, ceiling = None, 0
        # Menu thumbs only exist after a menu walk; without one, a "Photo "
        # button on the Reviews pane is a reviewer photo, so go via the cover.
        thumbs2 = page.locator('button[aria-label^="Photo "]')
        if menu_walked and await thumbs2.count():
            await thumbs2.first.click()
            entered, ceiling = "menu_thumbs", 2_000
        else:
            ov = page.get_by_role("tab", name=re.compile("^Overview", re.I)).first
            # Screenshot capture already leaves the page on Overview, and on the
            # fleets a pane overlay can intercept the click for the full 30s.
            if await ov.count() and await ov.get_attribute("aria-selected") != "true":
                # In-page click: a pointer click lands on the overlay instead.
                await ov.evaluate("e => e.click()")
                await _wait_js(page, _ENTRY_PRESENT, timeout_ms=2_000)
            wm["page_photo_total"] = (await page.evaluate(_PHOTO_TOTAL)) or wm["page_photo_total"]
            # "See photos" opens the FULL gallery (viewer on item 1, Next walks
            # every photo). The hero image can open a ~10-item preview carousel
            # instead — Akansha Medical: hero 10, See photos 29. Hero is only the
            # fallback. Never a bare "Photo of ...": that matches reviewer avatars.
            see = page.locator("button:has-text('See photos')").first
            cover = page.locator('button[jsaction*="heroHeaderImage"]').first
            for name, entry, wait_ms in (("see_photos", see, 3_000), ("hero", cover, 2_500)):
                if await entry.count():
                    await entry.evaluate("e => e.click()")   # overlay-proof, see above
                    entered, ceiling = name, wait_ms   # hero = possible ~10-item preview
                    break
        out["entry"] = wm["entry"] = entered
        if entered:
            t_entry = time.monotonic() + ceiling / 1000
            entry = await _wait_item(page, pre, ceiling)   # first item ready; None = unknown
            await _wait_js(page, _NEW_TABS, sorted(before), _ms_left(t_entry, ceiling))
            chips = await _list_gallery_chips(page, before)
            wm["chips_seen"] = chips
            wm["limited_view"] = int(await page.evaluate(_LIMITED_VIEW))
            out["items"] = await _walk_gallery(page, chips, wm, place_photo_cap,
                                               walk_max_items, walk_max_s, entry)
            if wm["limited_view"]:
                wm["truncated_reasons"].append("limited_view")
                wm["truncated"] = 1
            await page.keyboard.press("Escape")
        else:
            wm["reason"] = "no_entry"
        out["place_photos"] = [{"url": i["url"], "date": i["caption"], "kind": i["kind"]}
                               for i in out["items"]]
        real = sum(1 for p in out["place_photos"] if p["kind"] in ("photo", "video"))
        sv = sum(1 for p in out["place_photos"] if p["kind"] == "street_view")
        out["imagery"] = ("photos" if real else "street_view_only") \
            if (real or sv) else "none"
    except Exception as e:
        out["errors"].append(f"place_photos: {e}")
        out["imagery"] = out["imagery"] or "unknown"
        wm["walk_ok"], wm["reason"] = False, "exception"
    return out


if __name__ == "__main__":   # self-check of the pure parts: python -m parallel_scraper.horeca_capture
    base = {"viewer_url_key": "k", "img_src": "https://lh3.googleusercontent.com/p/x=w1",
            "attr": "Image capture: Jan 2024", "href": "https://maps/x"}
    assert _classify({**base, "caption": "Photo - Jan 2024"})["kind"] == "photo"
    assert _classify({**base, "caption": "Video - Jan 2024"}) == {
        "url": None, "caption": "Video - Jan 2024", "viewer_key": "k", "pano_url": None,
        "kind": "video"}
    assert _classify({**base, "caption": "Photo - 360 - Jan 2024"})["kind"] == "360"
    sv = _classify({**base, "caption": None})
    assert sv["kind"] == "street_view" and sv["pano_url"] == "https://maps/x" \
        and sv["caption"] == "Image capture: Jan 2024"
    assert _ms_left(time.monotonic() + 1, 8_000) <= 1_000 and _ms_left(None, 50) == 50
    print("ok")
