"""Overview-pane capture: price band, hours, popular times, outbound links,
share link and the structured Menu tab.

MUST run on a FRESH place panel. `PlaywrightSession.scrape()` expands the reviews
pane, which replaces Overview — every module below lives on Overview and is simply
absent afterwards. The caller navigates back to the place URL first; measured cost
of that re-goto is 0.3-0.5s. This is not an anti-bot gate: a logged-out headless
session with the stock UA sees everything once it is on the right pane.

Hard-won behaviours (do not "simplify" these away):
- Read the share link BEFORE touching any tab. The Share button lives on Overview;
  after a Menu-tab click it times out and returns nothing.
- Menu items sit behind a category chip. Clicking the Menu tab alone yields zero
  items — one chip has to be opened.
- Popular-times bars expose their value only in aria-label ("Tuesdays 6 AM 20%
  busy."), never in text.
- Google wraps outbound links as /url?q=<real>&fbclid=..&ved=..&usg=.. — store the
  unwrapped URL, the wrapper rots and carries tracking.
"""
from __future__ import annotations

import logging
import re
import urllib.parse

logger = logging.getLogger(__name__)

SETTLE_MS_DEFAULT = 5_000        # 2s proved flaky right after a gallery walk
MENU_SETTLE_MS = 3_500
CHIP_SETTLE_MS = 3_000
ABOUT_SETTLE_MS = 2_500
MAX_CHIP_TRIES = 3        # photo-only Menu tabs yield nothing; cap the hunt
SHARE_CLICK_MS = 15_000   # 6s lost 12% of links under fleet contention
SHARE_POLL_TRIES = 100    # x100ms = 10s for the dialog to render
SHARE_ATTEMPTS = 2

_DOWS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_TRACKING = {"fbclid", "opi", "sa", "ved", "usg", "gclid", "_aem", "ltsid", "igsh", "igshid",
             "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}
_LINK_KINDS = (
    ("instagram.com", "instagram"), ("facebook.com", "facebook"),
    ("tiktok.com", "tiktok"), ("snapchat.com", "snapchat"),
    ("twitter.com", "x"), ("x.com", "x"), ("t.me", "telegram"),
    ("youtube.com", "youtube"), ("linktr.ee", "linktree"),
    ("wa.me", "whatsapp"), ("api.whatsapp.com", "whatsapp"),
)
# 'Tuesdays 6 AM 20% busy.' / '0% busy at 6 AM.'
_BUSY_RE = re.compile(r"(\d{1,3})\s*%\s*busy", re.I)
_HOUR_RE = re.compile(r"(\d{1,2})\s*(AM|PM)", re.I)
_DOW_RE = re.compile("|".join(_DOWS), re.I)
# 'SAR 20–40 per person' / 'SAR 1-20' — en dash and hyphen both occur.
_PRICE_RE = re.compile(r"([A-Z]{3}|[^\W\d_]{1,4})\s*([\d,]+)\s*[–\-—]\s*([\d,]+)")
_PRICE_ONE = re.compile(r"([A-Z]{3})\s*([\d,]+)")
# Which panel lines the fallback may treat as a price band AT ALL. _PRICE_ONE alone
# matches any 3 capitals followed by digits, so it swallowed plus-code addresses
# ('MPH6+GJ Al Wizarat' -> currency MPH, price 6), hotel names ('OYO 165 ...') and
# codes like 'TOP100' — 5,987 bad rows in the Riyadh run. A real band is the WHOLE
# line: currency, a space, digits, optionally a range / '+' / truncation ellipsis.
_PRICE_BAND_LINE = re.compile(r"^[A-Z]{3}\s[\d,]+(\s*[–\-—]\s*[\d,]+)?\s*\+?\s*…?$")
# ...and the 3 letters must be a real currency. Business names shaped exactly like a
# band ("BUS 253", "FOX 16", "NSK 2022") slip through the pattern otherwise.
# Extend when a run moves to a new country.
_CURRENCIES = {
    "SAR", "AED", "QAR", "KWD", "BHD", "OMR", "JOD", "EGP", "TRY", "ILS",
    "INR", "PKR", "BDT", "LKR", "NPR",
    "USD", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD",
    "IDR", "PHP", "VND", "MYR", "SGD", "THB", "CNY", "JPY", "KRW", "HKD",
    "ZAR", "KES", "NGN", "MAD", "TND", "DZD",
}
_REPORTED_RE = re.compile(r"Reported by\s+([\d,]+)", re.I)

_JS_PANEL = """() => {
  const root = document.querySelector('div[role="main"]') || document.body;
  const lines = (root.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
  const aria = [...document.querySelectorAll('[aria-label]')]
    .map(e => e.getAttribute('aria-label') || '').filter(a => a && a.length < 160);
  const links = [...document.querySelectorAll('a[href]')]
    .map(a => a.href).filter(h => /^https?:/.test(h));
  // Popular times: ALL 7 days are in the DOM, one container per day (Sunday
  // first), only the selected day visible. Bar labels ('37% busy at 4 am.')
  // carry no day name, so the day is positional: read bars per container.
  let popular = null;
  const region = document.querySelector('div[role="region"][aria-label^="Popular times"]');
  if (region) {
    const bar0 = region.querySelector('[aria-label*="busy"]');
    let car = bar0;
    while (car && car !== region && car.children.length !== 7) car = car.parentElement;
    if (car && car !== region) {
      const kids = [...car.children];
      const days = kids.map(k => [...k.querySelectorAll('[aria-label*="busy"]')]
                                   .map(b => b.getAttribute('aria-label')));
      const visible = kids.findIndex(k => [...k.querySelectorAll('[aria-label*="busy"]')]
                                           .some(b => b.offsetWidth || b.offsetHeight));
      const sel = [...region.querySelectorAll('*')].map(e => (e.childElementCount ? '' : (e.innerText || '')).trim())
                    .find(t => /^(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?$/i.test(t)) || null;
      popular = {days, visible, selected: sel};
    }
  }
  return {
    lines, aria, links, popular,
    tabs: [...document.querySelectorAll('button[role="tab"]')].map(b => b.innerText.trim()),
  };
}"""

_JS_SHARE = """() => {
  const i = document.querySelector('input[value*="goo.gl"], input[value*="google.com/maps"]');
  return i ? i.value : null;
}"""

# Category chips are the siblings of the non-tab 'Overview' chip. Fall back to
# every panel button when that row is not found (MAX_CHIP_TRIES bounds the cost).
_JS_CHIPS = """() => {
  const all = [...document.querySelectorAll('div[role="main"] button')];
  const ov = all.find(b => b.getAttribute('role') !== 'tab' && (b.innerText || '').trim() === 'Overview');
  const row = ov ? [...ov.parentElement.querySelectorAll('button')] : all;
  return row.map(b => (b.innerText || '').trim());
}"""

# About tab: h2 = section ('Accessibility', 'Service options', 'Payments', ...),
# li[aria-label] = item ('Has wheelchair-accessible entrance', 'Does not have ...').
_JS_ABOUT = """() => {
  const root = document.querySelector('div[role="main"]') || document.body;
  const out = [];
  root.querySelectorAll('h2').forEach(h => {
    const section = (h.innerText || '').trim();
    if (!section) return;
    h.parentElement.querySelectorAll('li').forEach(li => {
      const label = li.getAttribute('aria-label') || (li.innerText || '').trim();
      if (label) out.push({section, label: label.replace(/\\s+/g, ' ').trim()});
    });
  });
  return out;
}"""

# Menu items are read from the DOM, not from flattened panel text: innerText loses
# the name/description boundary (a short description is indistinguishable from a
# category header sitting above a name). Here the price element's own container is
# walked, so the fields stay separated by structure.
_JS_MENU_ITEMS = """() => {
  const priceRe = /^[A-Z]{3}\\s*[\\d,]+(\\.\\d+)?$/;
  const root = document.querySelector('div[role="main"]') || document.body;
  const out = [];
  root.querySelectorAll('div,span,li').forEach(el => {
    if (el.children.length) return;
    const t = (el.innerText || '').replace(/\\u00a0/g, ' ').trim();
    if (!priceRe.test(t)) return;
    // climb to the smallest ancestor holding both the price and >=2 text lines
    let box = el, hops = 0;
    while (box.parentElement && hops < 5) {
      box = box.parentElement; hops++;
      const lines = (box.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
      if (lines.length >= 2) {
        const rest = lines.filter(l => !priceRe.test(l.replace(/\\u00a0/g, ' ')));
        if (rest.length) {
          out.push({price: t, name: rest[0], description: rest[1] || null});
          return;
        }
      }
    }
  });
  const seen = new Set();
  return out.filter(o => !seen.has(o.name + o.price) && seen.add(o.name + o.price));
}"""


def unwrap_url(raw: str | None) -> str | None:
    """'/url?q=https%3A//x.com/y&fbclid=..' -> 'https://x.com/y' (tracking stripped)."""
    if not raw:
        return None
    u = raw.strip()
    if u.startswith("/url?") or "google.com/url?" in u:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("q")
        if not q:
            return None
        u = q[0]
    parts = urllib.parse.urlsplit(u)
    if not parts.scheme.startswith("http"):
        return None
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k not in _TRACKING]
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(kept), ""))


def classify_link(url: str) -> str:
    host = (urllib.parse.urlsplit(url).netloc or "").lower()
    for needle, kind in _LINK_KINDS:
        if host == needle or host.endswith("." + needle):   # not substring: 't.me' in 'meta.com'
            return kind
    return "website"


def parse_price(lines: list[str]) -> dict:
    """Pull the price band out of the panel lines.

    'SAR 20–40 per person' + 'Reported by 267 people' -> min/max/currency/count.
    innerText truncates the inline copy with an ellipsis ('SAR 1–20…'), so the
    'per person' row is the reliable source.
    """
    out: dict = {"price_range_text": None, "price_min": None, "price_max": None,
                 "price_currency": None, "price_reported_by": None}
    for ln in lines:
        if "per person" in ln.lower() and any(ch.isdigit() for ch in ln):
            clean = ln.replace("\xa0", " ").strip()
            # 'per person' is Google's own price row, so this branch is trustworthy --
            # but a run in a country whose currency is not yet in _CURRENCIES should
            # drop the band rather than store a bogus currency code.
            if clean[:3].isupper() and clean[:3].isalpha() and clean[:3] not in _CURRENCIES:
                continue
            out["price_range_text"] = clean
            break
    if not out["price_range_text"]:
        for ln in lines:
            clean = ln.replace("\xa0", " ").strip()
            if _PRICE_BAND_LINE.match(clean) and clean[:3] in _CURRENCIES:
                out["price_range_text"] = clean
                break
    txt = (out["price_range_text"] or "").replace("\xa0", " ")
    m = _PRICE_RE.search(txt)
    if m:
        out["price_currency"] = m.group(1)[:3].upper()
        out["price_min"] = float(m.group(2).replace(",", ""))
        out["price_max"] = float(m.group(3).replace(",", ""))
    else:
        m1 = _PRICE_ONE.search(txt)
        if m1:
            out["price_currency"] = m1.group(1)[:3].upper()
            out["price_min"] = out["price_max"] = float(m1.group(2).replace(",", ""))
    for ln in lines:
        r = _REPORTED_RE.search(ln)
        if r:
            out["price_reported_by"] = int(r.group(1).replace(",", ""))
            break
    return out


def parse_popular_times(aria: list[str], selected_day: str | None = None) -> list[dict]:
    """aria-labels -> [{dow, hour_24, busy_pct}].

    A bar's label is 'Tuesdays 6 AM 20% busy.' — the day, when present, is IN the
    bar's own label. The day-selector dropdown also contributes labels that are
    bare day names ('Mondays'), and an earlier version carried the last day name
    forward across bars: every bar in a 1,000-place run came back stamped
    'Monday', including runs made on a Tuesday. So the day is only ever taken
    from a label that is actually a bar (one carrying a busy value).

    Google renders ONE day at a time (~24 bars), so `selected_day` — read from the
    day selector — is the fallback when bar labels omit it. Unresolvable days are
    left None for the caller to record honestly rather than guessed at.
    """
    out: list[dict] = []
    for a in aria:
        p = _parse_bar(a)
        if not p:
            continue                      # not a bar: a dropdown option or other control
        d = _DOW_RE.search(a)             # day must come from THIS bar's label
        dow = d.group(0).capitalize() if d else (selected_day or None)
        out.append({"dow": dow, "hour_24": p[0], "busy_pct": p[1]})
    return out


def _parse_bar(a: str) -> tuple[int | None, int] | None:
    """'37% busy at 4 am.' -> (4, 37); 'Currently 80% busy, usually 43% busy.' -> (h, 43).
    None when the label is not a bar."""
    m = re.search(r"usually\s+(\d{1,3})\s*%\s*busy", a, re.I) or _BUSY_RE.search(a)
    if not m:
        return None
    h = _HOUR_RE.search(a)
    hour = int(h.group(1)) % 12 + (12 if h.group(2).upper() == "PM" else 0) if h else None
    return hour, int(m.group(1))


def parse_popular_days(popular: dict | None) -> tuple[list[dict], str | None]:
    """Structured popular-times block -> (bars for all 7 days, selected day).

    `popular` = {days: [[bar labels] x7], visible: idx, selected: 'Tuesdays'} as
    read by _JS_PANEL. Google renders the seven day-containers Sunday-first, but
    rather than trust that, the day of the visible container is anchored to the
    selector text and the rest are counted from it. Falls back to Sunday-first
    when the selector is unreadable.
    """
    if not popular or not popular.get("days"):
        return [], None
    days = popular["days"]
    order = ("Sunday",) + _DOWS[:-1]                       # Sunday-first
    sel = None
    m = _DOW_RE.search(popular.get("selected") or "")
    if m:
        sel = m.group(0).capitalize()
    vis = popular.get("visible")
    offset = 0
    if sel and isinstance(vis, int) and vis >= 0:
        offset = (order.index(sel) - vis) % 7
    out: list[dict] = []
    for i, labels in enumerate(days[:7]):
        dow = order[(i + offset) % 7]
        prev_hour = None
        for a in labels or []:
            p = _parse_bar(a or "")
            if not p:
                continue
            hour = p[0]
            if hour is None and prev_hour is not None:
                hour = (prev_hour + 1) % 24       # live bar 'Currently 100% busy, usually 93% busy.'
            prev_hour = hour                       # has no hour; bars are positional
            out.append({"dow": dow, "hour_24": hour, "busy_pct": p[1]})
    return out, sel


def parse_selected_day(lines: list[str], aria: list[str]) -> str | None:
    """The day the popular-times chart is currently showing.

    The selector renders as a bare day name ('Tuesdays'), unlike a bar label which
    always carries a busy value. Return the singular form.
    """
    for src in (aria, lines):
        for t in src:
            t = (t or "").strip()
            if len(t) > 12 or _BUSY_RE.search(t):
                continue
            m = _DOW_RE.fullmatch(t.rstrip("s"))
            if m:
                return m.group(0).capitalize()
    return None


def parse_hours(aria: list[str]) -> list[dict]:
    """aria-labels carrying a weekday -> [{dow, text}]. Kept verbatim; splitting
    'Closed · Opens 9 AM' into open/close times is a downstream concern."""
    seen, out = set(), []
    for a in aria:
        d = _DOW_RE.search(a)
        if not d:
            continue
        dow = d.group(0).capitalize()
        if dow in seen or _BUSY_RE.search(a):
            continue
        seen.add(dow)
        out.append({"dow": dow, "text": _clean_hours_text(a)})
    return out


def _clean_hours_text(a: str) -> str:
    """'Tuesday, 6 am to 1 am, Copy open hours' -> 'Tuesday, 6 am to 1 am'.

    The aria-label ends with the button's own affordance text, and Google uses
    narrow no-break spaces around the times.
    """
    t = a.replace(" ", " ").replace(" ", " ")
    t = re.sub(r",?\s*Copy open hours\.?$", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" ,.")


async def capture_panel_extras(page, want_menu: bool = True,
                               settle_ms: int = SETTLE_MS_DEFAULT,
                               want_about: bool = True) -> dict:
    """Capture the Overview modules from the panel currently on `page`.

    `page` must already be on a fresh place URL. Never raises — each section
    degrades to empty plus an `errors` entry, matching horeca_capture.
    """
    out: dict = {
        "share_link": None, "price_range_text": None, "price_min": None,
        "price_max": None, "price_currency": None, "price_reported_by": None,
        "popular_times": [], "popular_times_day": None, "hours": [], "links": [],
        "has_menu_tab": False, "menu_items": [], "amenities": [], "errors": [],
    }
    try:
        await page.wait_for_timeout(settle_ms)
        panel = await page.evaluate(_JS_PANEL)
    except Exception as exc:
        out["errors"].append(f"panel: {exc}")
        return out

    lines, aria = panel.get("lines") or [], panel.get("aria") or []
    if not panel.get("tabs"):
        # Every real place panel has a tab strip (Overview/Reviews at least). Its
        # absence is the DEGRADED panel Google serves intermittently per session
        # (no popular times, no price, no Share) — a fresh browser gets the full
        # panel straight back. Flag it so the caller can relaunch and retry.
        out["errors"].append("panel: degraded (no tabs)")
        out["degraded"] = True
        return out
    try:
        out.update(parse_price(lines))
        bars, day = parse_popular_days(panel.get("popular"))
        if not bars:                          # structure not found: old flat path
            day = parse_selected_day(lines, aria)
            bars = parse_popular_times(aria, selected_day=day)
        out["popular_times"] = bars
        out["popular_times_day"] = day
        out["hours"] = parse_hours(aria)
    except Exception as exc:
        out["errors"].append(f"parse: {exc}")

    try:
        seen = set()
        for raw in panel.get("links") or []:
            u = unwrap_url(raw)
            if not u or "google.com/maps" in u or "/maps/" in u:
                continue
            host = urllib.parse.urlsplit(u).netloc.lower()
            if host.endswith("google.com") or host.endswith("google.co.in"):
                continue
            if u in seen:
                continue
            seen.add(u)
            out["links"].append({"kind": classify_link(u), "url": u[:2000]})
    except Exception as exc:
        out["errors"].append(f"links: {exc}")

    # Share BEFORE any tab switch — the button is not on the Menu pane.
    out["share_link"] = await _capture_share(page, out)

    tabs = panel.get("tabs") or []
    out["has_menu_tab"] = any("menu" in t.lower() for t in tabs)
    if want_menu and out["has_menu_tab"]:
        try:
            out["menu_items"] = await _capture_menu(page)
        except Exception as exc:
            out["errors"].append(f"menu: {exc}")
    if want_about and any("about" in t.lower() for t in tabs):
        try:
            out["amenities"] = await _capture_about(page)
        except Exception as exc:
            out["errors"].append(f"about: {exc}")
    return out


async def _capture_about(page) -> list[dict]:
    """Open the About tab and read every section/item. ~2.6s. Replaces the
    engine's `amenities` (0% captured) and `about` (3.9%, wrong element)."""
    await page.locator('button[role="tab"]').filter(has_text="About").first.click(timeout=6_000)
    await page.wait_for_timeout(ABOUT_SETTLE_MS)
    return normalize_amenities(await page.evaluate(_JS_ABOUT))


_NEGATIVE = re.compile(r"^(Does not|Doesn't|No |Not |Isn't|Is not)", re.I)
_PUA_RE = re.compile(r"^[\ue000-\uf8ff\s]+")


def normalize_amenities(rows) -> list[dict]:
    """[{section, label}] -> [{section, item, present}], deduped.

    Google phrases absence in the label itself ('Does not have wheelchair-
    accessible car park', 'No delivery'); the item text is kept verbatim so
    nothing is lost, `present` is the parsed polarity.
    """
    out, seen = [], set()
    for r in rows or []:
        section = (r.get("section") or "").strip()[:60]
        raw = (r.get("label") or "").strip()
        # Without an aria-label the row text starts with Google's icon-font glyph
        # (private-use codepoints): U+E5CA 'check' = present, U+E5CD 'close' = absent,
        # others are item icons (U+E033 hearing loop). Strip them, keep the polarity.
        glyph_absent = raw.startswith("\ue5cd")
        item = _PUA_RE.sub("", raw).strip()[:120]
        if not section or not item or (section, item) in seen:
            continue
        seen.add((section, item))
        out.append({"section": section, "item": item,
                    "present": not (glyph_absent or bool(_NEGATIVE.match(item)))})
    return out


async def _capture_share(page, out: dict) -> str | None:
    """Click Share and read the maps.app.goo.gl link.

    Every listing has a Share button — a miss is always contention, never the
    listing. A 1,000-place fleet run lost 12% of share links at 60 concurrent
    workers, yet every sampled failure captured the link first try in 0.5-0.8s
    when re-run alone. Three browsers per runner share one CPU, so the click and
    the dialog render both stretch. Hence generous timeouts and one retry: the
    happy path still costs ~0.6s.
    """
    for attempt in range(SHARE_ATTEMPTS):
        try:
            await page.locator('button[aria-label^="Share"], button:has-text("Share")'
                               ).first.click(timeout=SHARE_CLICK_MS)
            for _ in range(SHARE_POLL_TRIES):
                link = await page.evaluate(_JS_SHARE)
                if link:
                    await page.keyboard.press("Escape")
                    return link
                await page.wait_for_timeout(100)
            await page.keyboard.press("Escape")   # dialog open but no link yet
        except Exception as exc:
            if attempt + 1 >= SHARE_ATTEMPTS:
                out["errors"].append(f"share: {exc}")
                return None
        if attempt + 1 < SHARE_ATTEMPTS:
            await page.wait_for_timeout(500)
    out["errors"].append("share: no link after retries")
    return None


async def _capture_menu(page) -> list[dict]:
    """Open the Menu tab and walk EVERY category chip, reading item rows per chip.

    The Menu tab lands on an 'Overview' chip (menu photos + highlights, no priced
    rows); items sit behind the category chips that follow it — Juices, Burgers,
    Pies And Pizza, ... Stopping at the first chip that yielded items stored 62 of
    131 items (Juices only) on a sampled listing. Each chip costs ~2.5s; a chip
    that yields nothing (photo-only) does not stop the walk, but MAX_CHIP_TRIES
    consecutive empty chips do.

    Rows render as name / description / price stacked as separate lines, with the
    price line starting with the currency code.
    """
    await page.locator('button[role="tab"]').filter(has_text="Menu").first.click(timeout=6_000)
    await page.wait_for_timeout(MENU_SETTLE_MS)
    items = normalize_menu_items(await page.evaluate(_JS_MENU_ITEMS))
    seen = {(i["name"], i["price_amount"]) for i in items}
    empty_run = 0
    for chip in await page.evaluate(_JS_CHIPS):
        if not chip or len(chip) > 40 or chip in ("Overview", "Menu", "Reviews", "About"):
            continue
        if empty_run >= MAX_CHIP_TRIES:
            break                                  # photo-only menu: stop paying per chip
        try:
            await page.locator('div[role="main"] button').filter(has_text=chip
                                                                 ).first.click(timeout=5_000)
        except Exception:
            empty_run += 1
            continue
        fresh: list[dict] = []
        for _ in range(CHIP_SETTLE_MS // 300):     # poll: most chips render in <1s
            await page.wait_for_timeout(300)
            got = normalize_menu_items(await page.evaluate(_JS_MENU_ITEMS))
            fresh = [i for i in got if (i["name"], i["price_amount"]) not in seen]
            if fresh:                              # one more read: catch late rows
                await page.wait_for_timeout(400)
                got = normalize_menu_items(await page.evaluate(_JS_MENU_ITEMS))
                fresh = [i for i in got if (i["name"], i["price_amount"]) not in seen]
                break
        if not fresh:
            empty_run += 1
            continue
        empty_run = 0
        for i in fresh:
            i["category"] = chip[:100]
            seen.add((i["name"], i["price_amount"]))
        items.extend(fresh)
    return items


def normalize_menu_items(rows) -> list[dict]:
    """DOM rows [{price, name, description}] -> typed items.

    Rows whose price will not parse are dropped rather than stored as 0.0, so a
    selector drift shows up as missing items instead of a menu priced at nothing.
    """
    out: list[dict] = []
    for r in rows or []:
        m = _PRICE_ONE.match((r.get("price") or "").replace(" ", " ").strip())
        name = (r.get("name") or "").strip()
        if not m or not name:
            continue
        desc = (r.get("description") or "").strip() or None
        out.append({"name": name[:200], "description": desc[:500] if desc else None,
                    "currency": m.group(1).upper(),
                    "price_amount": float(m.group(2).replace(",", ""))})
    return out


def _selfcheck_price() -> None:
    """parse_price must accept real bands and reject addresses / names / codes."""
    good = {
        "SAR 20–40 per person": (20.0, 40.0),
        "SAR 1-20 per person": (1.0, 20.0),
        "SAR 20–40": (20.0, 40.0),
        "SAR 120–140…": (120.0, 140.0),
        "SAR 200+…": (200.0, 200.0),
    }
    for line, (lo, hi) in good.items():
        got = parse_price([line])
        assert got["price_min"] == lo and got["price_max"] == hi, (line, got)
    bad = ["BUS 253", "FOX 16", "NSK 2022", "MBS 1",
           "MPH6+GJ Al Wizarat, Riyadh Saudi Arabia",
           "HQV4+3X9, Al Aziziyah, Riyadh 14512, Saudi Arabia",
           "OYO 165 Orchida Al Hamra", "TOP100", "BDR222",
           "JHR4+RGQ, Dhahrat Laban, Riyadh 13782, Saudi Arabia"]
    for line in bad:
        got = parse_price([line])
        assert got["price_range_text"] is None, (line, got)
    print("parse_price selfcheck OK")


if __name__ == "__main__":
    _selfcheck_price()
