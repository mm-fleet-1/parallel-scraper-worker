"""Delisted places (design 2026-10-phase2-gallery-refresh §8.4).

Worker side: `detect_delisted` needs positive evidence (a known Google removal text, or Maps' redirect of an
unresolvable place_id to an empty place, never a /sorry/ page).
API side: `mark_delisted` writes the tombstone in one lease-gated transaction on a caller-supplied connection,
so this module never imports the DB layer (the public worker imports it too).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Known Google removal / not-found panel texts. The pilot extends the list; extra entries via
# PHASE2_DELISTED_TEXTS (separated by '|').
DELISTED_TEXTS = ["This place has been removed", "no longer exists"] + [
    t.strip() for t in os.environ.get("PHASE2_DELISTED_TEXTS", "").split("|") if t.strip()]
# p200: Maps rewrites ?q=place_id:<gone id> to /maps/place//@<viewer's area> (empty place segment, no
# name, no notice text). Reproduced across shards, IPs and a day apart. The ?q= form is /place/?q= (one slash).
DELISTED_URL_MARK = "/maps/place//"


def detect_delisted(page_text: str | None, page_url: str | None) -> str | None:
    """Evidence (a removal text, or 'redirect:/maps/place//') unless the URL is a /sorry/ page, else None.
    Only called once the panel has no name, so the redirect is never confused with a loaded place."""
    url = page_url or ""
    if "/sorry/" in url:
        return None
    if DELISTED_URL_MARK in url:
        logger.info("delisted evidence redirect at %s", url)
        return "redirect:" + DELISTED_URL_MARK
    if not page_text:
        return None
    low = page_text.lower()
    for t in DELISTED_TEXTS:
        if t.lower() in low:
            logger.info("delisted evidence %r at %s", t, page_url)
            return t
    return None


_TOMBSTONE_SQL = """
SET NOCOUNT ON;
DECLARE @run nvarchar(128) = ?, @pid nvarchar(256) = ?, @att int = ?, @obs datetime2(3) = ?, @err nvarchar(3900) = ?;
UPDATE scraper.place_ids
   SET phase2_status = 'delisted', last_error = @err, phase2_worker = NULL
 WHERE run_id = @run AND place_id = @pid AND phase2_status = 'scraping' AND attempts = @att;
IF @@ROWCOUNT = 0 BEGIN SELECT 0 AS accepted; RETURN; END;

IF EXISTS (SELECT 1 FROM scraper.place_data WHERE run_id = @run AND place_id = @pid)
BEGIN
    -- repair pass: clear this run's gallery and mark its walk failed; the place_data update fires the core
    -- trigger, so the place_core tombstone update must come after it
    DELETE FROM scraper.place_media
     WHERE run_id = @run AND place_id = @pid AND is_menu = 0 AND evidence_run_id IS NULL;
    -- §13d.3 delisting, statement (1): the summary columns, unconditionally (they are the source of
    -- truth; a mode-1 row has data NULL). Dynamic SQL: the columns exist only after the §13d DDL.
    IF COL_LENGTH('scraper.place_data', 'walk_ok') IS NOT NULL
        EXEC sp_executesql N'UPDATE scraper.place_data
           SET walk_ok = 0, walk_reason = N''delisted'', photos_selected = 0, photos_mirrored = 0
         WHERE run_id = @run AND place_id = @pid;', N'@run nvarchar(128), @pid nvarchar(256)', @run, @pid;
    -- statement (2): the same four values into the JSON, only where the row has JSON. Only these
    -- keys change (never a whole-summary replace), so photos_walked / truncated stay in the JSON
    -- as they stay in the columns. A summary (or photo_dates) that is missing gets just these keys.
    UPDATE scraper.place_data
       SET data = CASE WHEN JSON_QUERY(data, '$.photo_dates') IS NULL
                       THEN JSON_MODIFY(data, '$.photo_dates', JSON_QUERY(
                            '{"summary":{"walk_ok":false,"reason":"delisted","photos_selected":0,"photos_mirrored":0}}'))
                       WHEN JSON_QUERY(data, '$.photo_dates.summary') IS NULL
                       THEN JSON_MODIFY(data, '$.photo_dates.summary', JSON_QUERY(
                            '{"walk_ok":false,"reason":"delisted","photos_selected":0,"photos_mirrored":0}'))
                       ELSE JSON_MODIFY(JSON_MODIFY(JSON_MODIFY(JSON_MODIFY(data,
                            '$.photo_dates.summary.walk_ok', CAST(0 AS bit)),
                            '$.photo_dates.summary.reason', N'delisted'),
                            '$.photo_dates.summary.photos_selected', 0),
                            '$.photo_dates.summary.photos_mirrored', 0) END
     WHERE run_id = @run AND place_id = @pid AND data IS NOT NULL AND ISJSON(data) = 1;
END;

IF EXISTS (SELECT 1 FROM scraper.place_core WHERE run_id = @run AND place_id = @pid)
    UPDATE scraper.place_core
       SET status = 'Delisted', delisted_at = @obs, is_tombstone = 1,
           scraped_at = @obs   -- the tombstone is the newest observation (§8.6 order)
     WHERE run_id = @run AND place_id = @pid;
ELSE
    INSERT INTO scraper.place_core (run_id, place_id, status, delisted_at, scraped_at, is_tombstone)
    VALUES (@run, @pid, 'Delisted', @obs, @obs, 1);
SELECT 1 AS accepted;
"""


def mark_delisted(conn, run_id: str, place_id: str, attempts: int, observed_at: datetime,
                  evidence_text: str | None = None, page_url: str | None = None) -> dict:
    """Lease-gated tombstone in one transaction. observed_at: naive UTC datetime.
    Returns {accepted: 1, stale: 0} or {accepted: 0, stale: 1} (lease no longer current; nothing written)."""
    err = f"delisted: {evidence_text or ''} @ {page_url or ''}"[:3900]
    cur = conn.cursor()
    try:
        cur.execute(_TOMBSTONE_SQL, str(run_id), str(place_id), int(attempts), observed_at, err)
        row = cur.fetchone()
        accepted = int(row[0]) if row else 0
        if accepted:
            conn.commit()
        else:
            conn.rollback()
    except Exception:
        conn.rollback()
        raise
    logger.info("delisted run=%s place=%s accepted=%d url=%s", run_id, place_id, accepted, page_url)
    return {"accepted": accepted, "stale": 1 - accepted}
