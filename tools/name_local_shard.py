"""Name-only pass: read <h1> and the secondary <h2> for a shard of place_ids.

Deliberately NOT a re-scrape. A normal phase2 run rewrites place_data.data wholesale,
which fires scraper.tr_place_data_core and overwrites EVERY place_core column from the
new scrape -- so a weaker scrape would degrade good data across 47k outlets just to
obtain one field. This reads two DOM nodes and writes nothing but its own results file.

Maps renders the secondary/local name as an <h2> sibling of the <h1> inside div.lMbq3e:

    <h1 class="DUwDvf">The taste of tea</h1>
    <h2 class="bwoZTb">مذاق الشاي</h2>

We send accept-language=en, so the <h1> is the ENGLISH name when the business registered
one and falls back to the local name when it did not. Only the <h1> was ever captured.

Runner has no DB driver (same as phase2-scrape), so results go to blob and are ingested
by clients/Flora/pipeline/ingest_name_local.py, which does a targeted UPDATE of
place_core.name_local only.

env: MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (or PHASE2_SHOT_BLOB_SAS)
usage:
  python tools/name_local_shard.py --shard-url <blob>/shards/shard_001.json \
      --out-base <blob>/results --shard shard_001 [--concurrency 8]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from pathlib import Path

import requests

EXTRACT_JS = """
() => {
    const nameEl = document.querySelector('div[class*="fontHeadlineLarge"] h1')
                || document.querySelector('h1.DUwDvf')
                || document.querySelector('h1');
    if (!nameEl) return null;
    const name = nameEl.textContent.trim();
    let local = '';
    // Scoped to the h1's own container ON PURPOSE: a listing page carries ~12 <h2>s;
    // an unscoped querySelector('h2') returns an unrelated heading.
    const box = nameEl.closest('div.lMbq3e') || (nameEl.parentElement && nameEl.parentElement.parentElement);
    const h2 = box ? (box.querySelector('h2.bwoZTb') || box.querySelector('h2')) : null;
    const t = h2 ? h2.textContent.trim() : '';
    if (t && t !== name) local = t;
    return {name: name, local: local};
}
"""


def _q(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "")
        if "sig=" in v:
            return v.split("?", 1)[1] if "?" in v else v.lstrip("?")
    raise SystemExit(f"missing SAS: {names}")


def blob_get(url: str) -> str:
    r = requests.get(f"{url}?{_q('MEDIA_BLOB_READ_SAS')}", timeout=120)
    r.raise_for_status()
    return r.text


def blob_put(url: str, data: bytes) -> None:
    r = requests.put(f"{url}?{_q('MEDIA_BLOB_WRITE_SAS', 'PHASE2_SHOT_BLOB_SAS')}", data=data,
                     headers={"x-ms-blob-type": "BlockBlob",
                             "Content-Type": "application/x-ndjson"}, timeout=600)
    if r.status_code not in (200, 201):
        raise SystemExit(f"PUT {url.rsplit('/', 1)[-1]}: HTTP {r.status_code} {r.text[:160]}")


async def worker(ctx, queue: asyncio.Queue, out: list, stats: dict) -> None:
    page = await ctx.new_page()
    # Only the header DOM is needed; blocking media is most of the speed.
    await page.route("**/*", lambda r: asyncio.ensure_future(
        r.abort() if r.request.resource_type in ("image", "media", "font", "stylesheet")
        else r.continue_()))
    try:
        while True:
            try:
                pid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            res, last = None, ""
            # Retry in-run: a timeout is usually transient (throttle or a slow render on
            # a 2-core runner). Discarding it stranded ~1,259 of 2,338 per shard on the
            # first fleet pass, all of which were perfectly scrapeable.
            for attempt in range(3):
                try:
                    await page.goto(f"https://www.google.com/maps/place/?q=place_id:{pid}&hl=en",
                                    wait_until="domcontentloaded", timeout=45000)
                    await page.wait_for_selector("h1", timeout=20000)
                    res = await page.evaluate(EXTRACT_JS)
                    if res:
                        break
                    stats["nodata"] += 1
                    break
                except Exception as exc:  # noqa: BLE001
                    last = str(exc)[:80]
                    # DIAGNOSE rather than assume: a throttle/consent interstitial has no
                    # <h1> and redirects to /sorry/ or consent.google.com; CPU starvation
                    # leaves us on the right URL with a normal title. These need different
                    # fixes, and "wait_for_selector timed out" cannot tell them apart.
                    try:
                        url, title = page.url, (await page.title())[:60]
                    except Exception:  # noqa: BLE001
                        url, title = "?", "?"
                    blocked = ("/sorry/" in url or "consent." in url
                               or "unusual traffic" in title.lower() or "before you continue" in title.lower())
                    if blocked:
                        stats["blocked"] += 1
                    if stats["err"] + stats["blocked"] <= 6:
                        print(f"  ! {pid} try{attempt + 1}: {last} | url={url[:70]} title={title!r} "
                              f"blocked={blocked}", flush=True)
                    # CONFIRMED 2026-08-24: the failures are google.com/sorry/ bot-detection,
                    # not slow renders. Retrying a block is pure waste -- all 3 attempts hit
                    # the same interstitial seconds apart. Give the outlet up immediately and
                    # let the next pass collect it; ingest only re-stages NULLs, so nothing
                    # is lost and the run finishes ~3x sooner on the blocked portion.
                    if blocked:
                        break
                    if attempt < 2:
                        await asyncio.sleep(3 * (attempt + 1))
            if res:
                # '' means checked-and-none; distinct from "never checked".
                out.append({"place_id": pid, "name": res["name"], "name_local": res["local"]})
                stats["ok"] += 1
                stats["found"] += 1 if res["local"] else 0
            elif last:
                stats["err"] += 1
            queue.task_done()
    finally:
        await page.close()


async def main_async(a) -> None:
    from playwright.async_api import async_playwright
    if a.shard_file:
        pids = json.loads(Path(a.shard_file).read_text(encoding="utf-8"))
    elif a.shard_url:
        pids = json.loads(blob_get(a.shard_url))
    else:
        raise SystemExit("need --shard-file or --shard-url")
    if a.limit:
        pids = pids[:a.limit]
    print(f"shard {a.shard}: {len(pids)} place_ids, concurrency={a.concurrency}", flush=True)
    queue: asyncio.Queue = asyncio.Queue()
    for p in pids:
        queue.put_nowait(p)
    out: list = []
    stats = {"ok": 0, "found": 0, "err": 0, "nodata": 0, "blocked": 0}
    t0 = time.time()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = await browser.new_context(locale="en-US",
                                        extra_http_headers={"accept-language": "en-IN,en;q=0.9"},
                                        viewport={"width": 1280, "height": 900})
        tasks = [asyncio.create_task(worker(ctx, queue, out, stats)) for _ in range(a.concurrency)]
        while any(not t.done() for t in tasks):
            await asyncio.sleep(20)
            el = time.time() - t0
            print(f"  [{stats['ok']}/{len(pids)}] found={stats['found']} err={stats['err']} "
                  f"{60 * stats['ok'] / max(el, 1):.0f}/min", flush=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n"
    blob_put(f"{a.out_base}/{a.shard}.jsonl", body.encode("utf-8"))
    el = time.time() - t0
    done = stats["ok"] + stats["nodata"]
    pct = 100 * done / max(len(pids), 1)
    print(f"done ok={stats['ok']} with_local={stats['found']} err={stats['err']} "
          f"blocked={stats['blocked']} nodata={stats['nodata']} coverage={pct:.1f}% "
          f"wall={el / 60:.1f} min -> {a.out_base}/{a.shard}.jsonl")
    # Results are already published above, so a non-zero exit loses nothing -- the point
    # is that a shard which dropped half its work must NOT report success. The first
    # fleet pass exited 0 on every runner while collectively stranding 54% of the work,
    # which is exactly the silent-success failure this guards against.
    if pct < a.min_coverage:
        raise SystemExit(f"FAILING: coverage {pct:.1f}% < --min-coverage {a.min_coverage}%. "
                         f"{stats['err']} errors ({stats['blocked']} look like blocking). "
                         f"Re-stage and re-dispatch; ingest is resume-safe.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-url", help="blob URL of the shard (needs MEDIA_BLOB_READ_SAS)")
    ap.add_argument("--shard-file", help="local path to the shard JSON, committed in this repo. "
                                         "Preferred: place_ids are public identifiers, so shipping "
                                         "them in the repo avoids handing a delete-capable SAS to "
                                         "19 public fleet repos just to read a list.")
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="pages per runner. 8 collapsed to 16/min with ~50%% errors on a "
                         "2-core runner; 20 fleets x 3 is still ~60 parallel pages.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=90.0,
                    help="exit non-zero below this %% of the shard, so a half-finished "
                         "run cannot show green")
    a = ap.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
