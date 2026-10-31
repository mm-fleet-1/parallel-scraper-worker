"""Durable result staging, phase 1: the worker side (docs/design/2026-10-phase2-gallery-refresh.md §13c).

PHASE2_STAGING (env, set per run by the fleet workflow; default 0):
  0       today: results go to /phase2/results + /phase2/attributes. This module is not used.
  shadow  each completed outlet is ALSO written as one object under p2shadow/<run_id>/...;
          /results + /attributes as today; /phase2/staged is never called (§13c.6 step 2).
  1       each completed outlet is one object under p2results/<run_id>/...; after the PUT
          returned 201 (or 409 BlobAlreadyExists) its ref goes to POST /phase2/staged.
          No /results, no /attributes: attributes ride in the object.

Object name: <prefix>/<run_id>/<place_id>/a<attempts>-g<repair_gen>-v1-<sha256_16>.json.gz
Body (gzip of compact ASCII JSON): {schema_version, run_id, place_id, attempts, repair_gen,
worker, shard, scraped_at, outcome: done|failed, data, image_urls, media, attributes, error}.
sha256 is over the uncompressed body; the full value is also x-ms-meta-sha256.

Spool: every object is written to the spool dir ($PHASE2_SPOOL_DIR, else
$RUNNER_TEMP/p2spool) before its PUT and removed after it is acknowledged. One uploader
thread retries failed PUTs with backoff 1..60 s, never dropping an object. At SPOOL_CAP (50)
spooled objects the worker leases nothing more until the spool is empty, and it never holds
more than SPOOL_CAP + HELD_MAX (70) spooled objects plus leased, unfinished places; a 403
(SAS expired) stops leasing at once. Shadow mode has none of this: a shadow object whose PUT
fails is logged and dropped (/results still carries the data), and shadow never blocks
leasing or the time budget. Shard end: drain up to 120 s; what is left is lost work (exit 3) and
goes up as a workflow artifact; `python -m cloud.staging --replay-spool DIR` uploads and
reports it later.

The name, rank and identity helpers are shared with the API (rule F) and the ingester.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PREFIX = {"1": "p2results", "shadow": "p2shadow"}
MODES = ("0", "shadow", "1")
SPOOL_CAP = 50              # hard cap (§13c.3 worker rules) ...
HELD_MAX = 20               # ... plus at most this many leased, unfinished places: spooled + held <= 70
DRAIN_S = 120.0             # shard end: retry the spool this long
REPORT_MAX = 500            # refs per /phase2/staged call (API cap)
REPORT_EVERY_S = 30.0       # /phase2/staged cadence ...
REPORT_EVERY_N = 100        # ... or this many refs, whichever first

_NAME_RE = re.compile(
    r"^(?P<prefix>p2results|p2shadow)/(?P<run_id>[^/]+)/(?P<place_id>[^/]+)/"
    r"a(?P<attempts>\d+)-g(?P<repair_gen>\d+)-v(?P<v>\d+)-(?P<sha16>[0-9a-f]{16})\.json\.gz$")


def mode() -> str:
    """PHASE2_STAGING, validated: '0' | 'shadow' | '1'. A typo fails the shard at start."""
    m = (os.environ.get("PHASE2_STAGING") or "0").strip().lower()
    if m not in MODES:
        raise ValueError(f"PHASE2_STAGING must be one of {MODES}, got {m!r}")
    return m


def staging_sas() -> str | None:
    """Container SAS (sp=cw) for result objects. §13c.2: PHASE2_SHOT_BLOB_SAS (what the
    phase2 fleet workflow sets today), falling back to MEDIA_BLOB_WRITE_SAS."""
    return os.environ.get("PHASE2_SHOT_BLOB_SAS") or os.environ.get("MEDIA_BLOB_WRITE_SAS") or None


def blob_name(prefix: str, run_id: str, place_id: str, attempts: int, repair_gen: int, sha256: str) -> str:
    return (f"{prefix}/{run_id}/{place_id}/a{int(attempts)}-g{int(repair_gen)}"
            f"-v{SCHEMA_VERSION}-{sha256[:16]}.json.gz")


def parse_blob_name(name: str) -> dict | None:
    m = _NAME_RE.match(name or "")
    if not m:
        return None
    d = m.groupdict()
    return {**d, "attempts": int(d["attempts"]), "repair_gen": int(d["repair_gen"]), "v": int(d["v"])}


def ref_identity_ok(run_id: str, ref: dict) -> bool:
    """Rule F row 1: the ref's run_id, place_id, attempts and repair_gen agree with the
    production object name (and its sha256 with the name's sha256_16). Otherwise the ref is
    not a valid reference: rejected, no receipt."""
    n = parse_blob_name(str(ref.get("blob") or ""))
    sha = str(ref.get("sha256") or "")
    return bool(n and n["prefix"] == PREFIX["1"] and n["run_id"] == str(run_id)
                and n["place_id"] == str(ref.get("place_id"))
                and n["attempts"] == int(ref.get("attempts"))
                and n["repair_gen"] == int(ref.get("repair_gen"))
                and re.fullmatch(r"[0-9a-f]{64}", sha) and sha[:16] == n["sha16"])


# §13d.4 summary semantics: what TRY_CAST(JSON_VALUE(data, '$.photo_dates.summary.<key>') AS ...)
# returns today, computed in Python. Order = the place_data summary columns.
SUMMARY_COLS = ("walk_ok", "walk_reason", "photos_walked", "photos_selected", "photos_mirrored", "truncated")
_NO_SUMMARY = (None,) * len(SUMMARY_COLS)
_INT_RE = re.compile(r"[+-]?\d+")
_MAX_INT_CHARS = 20   # longer integer strings overflow every SQL integer type: TRY_CAST -> NULL


def _int_text(v: str):
    """An integer string as SQL's string->int cast reads it (blanks around it allowed), else None.
    Bounded first: Python's int() raises on very long digit strings (Codex R2 #1)."""
    t = v.strip(" ")
    if len(t) > _MAX_INT_CHARS or not _INT_RE.fullmatch(t):
        return None
    return int(t)


def _no_const(name):
    raise ValueError(name)   # NaN / Infinity: ISJSON() = 0, so every JSON_VALUE is NULL


def _sql_int(v):
    """TRY_CAST(JSON_VALUE(..) AS int): a JSON integer or an integer string (blanks around it
    allowed) in int range; true/false, floats ('3.0'), other strings, objects -> NULL."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        n = v
    elif isinstance(v, str) and _int_text(v) is not None:
        n = _int_text(v)
    else:
        return None
    return n if -2**31 <= n < 2**31 else None


def _sql_bit(v):
    """TRY_CAST(JSON_VALUE(..) AS bit): true / 'true' (any case) / non-zero number or integer
    string -> 1; false / 'false' / 0 / '0' -> 0; missing, null, other strings ('yes') -> NULL."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 0 if v == 0 else 1
    if isinstance(v, str):
        t = v.strip(" ")
        if t.lower() in ("true", "false"):
            return int(t.lower() == "true")
        n = _int_text(t)
        if n is not None:
            return 0 if n == 0 else 1
    return None


def _sql_text(v, width: int = 200):
    """JSON_VALUE of a string (NULL over 4000 UTF-16 units, lax mode), cut to nvarchar(width)."""
    if not isinstance(v, str):
        return None
    u = v.encode("utf-16-le", "surrogatepass")
    if len(u) > 8000:
        return None
    return u[:2 * width].decode("utf-16-le", "replace")


def summary_cols(data) -> tuple:
    """The ONE definition of the place_data summary columns (§13d.4, W-12 #5), used by
    sql_state's writers, rank_of, the gate and result_fetch's overlay. data = the result JSON
    (str, or an already parsed dict). Returns (walk_ok, walk_reason, photos_walked,
    photos_selected, photos_mirrored, truncated), each what today's SQL reads from
    $.photo_dates.summary: walk_ok as TRY_CAST(.. AS bit), the counts and truncated as
    TRY_CAST(.. AS int), reason as JSON_VALUE cut to 200. Not JSON / no summary -> all None.
    Lives here, not in sql_state, because rank_of needs it and this file is in the public
    worker set (sql_state is not); sql_state re-exports it."""
    try:
        if isinstance(data, (str, bytes)):
            data = json.loads(data, parse_constant=_no_const, object_pairs_hook=lambda p: dict(reversed(p)))
        pd = data.get("photo_dates") if isinstance(data, dict) else None
        s = pd.get("summary") if isinstance(pd, dict) else None
        if not isinstance(s, dict):
            return _NO_SUMMARY
        return (_sql_bit(s.get("walk_ok")), _sql_text(s.get("reason")), _sql_int(s.get("photos_walked")),
                _sql_int(s.get("photos_selected")), _sql_int(s.get("photos_mirrored")),
                _sql_int(s.get("truncated")))
    except Exception:  # noqa: BLE001 - never raise: a value SQL cannot read is NULL (JSON_VALUE)
        return _NO_SUMMARY


def pointer_flag() -> bool:
    """PHASE2_RESULT_POINTER=1 (§13d.4). Off (default, and on every worker and the API): today's
    exact rank and media_partial semantics. On (the mode-1 ingester): summary_cols semantics."""
    return (os.environ.get("PHASE2_RESULT_POINTER") or "0").strip() == "1"


def rank_of(body: dict) -> int:
    """§13c.3 precedence: 3 = done and complete, 2 = done and partial, 1 = failed. The worker
    supplies it; the ingester recomputes it and never trusts the supplied value.
    Flag off (pointer_flag): today's rule, walk_ok is True, truncated falsy, every media row
    mirrored. Flag on: walk_ok = 1 and truncated 0 or NULL from summary_cols (§13d.4, J4). The two
    agree on what workers write (walk_ok bool, truncated 0/1); only odd values (strings, a bool
    truncated) differ, and those then quarantine on a rank mismatch, never apply wrongly."""
    if body.get("outcome") != "done":
        return 1
    media = body.get("media") or []
    mirrored = sum(1 for m in media if (m or {}).get("status") in ("written", "exists"))
    if pointer_flag():
        walk_ok, _r, _w, _s, _m, truncated = summary_cols(body.get("data") or "{}")
        complete = walk_ok == 1 and truncated in (0, None) and mirrored == len(media)
        return 3 if complete else 2
    try:
        data = json.loads(body.get("data") or "{}")
    except (TypeError, ValueError):
        return 2
    s = ((data or {}).get("photo_dates") or {}).get("summary") or {}
    complete = s.get("walk_ok") is True and not s.get("truncated") and mirrored == len(media)
    return 3 if complete else 2


def build_object(run_id: str, result: dict, outcome: str, *, repair_gen: int = 0,
                 attributes: dict | None = None, worker: str = "", shard: str = "",
                 prefix: str = PREFIX["1"]) -> tuple[str, bytes, dict]:
    """(blob name, gzip bytes, ref) for one outlet. result = the exact dict /phase2/results
    would get (done: place_id, attempts, data, image_urls, media; failed: place_id,
    attempts, error)."""
    body = {"schema_version": SCHEMA_VERSION, "run_id": str(run_id),
            "place_id": str(result["place_id"]), "attempts": int(result["attempts"]),
            "repair_gen": int(repair_gen), "worker": str(worker), "shard": str(shard),
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                                  .replace("+00:00", "Z"),
            "outcome": outcome, "data": result.get("data"),
            "image_urls": list(result.get("image_urls") or []),
            "media": list(result.get("media") or []),
            "screenshots": list(result.get("screenshots") or []),
            "attributes": attributes or None, "error": result.get("error")}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                     default=str).encode("ascii")
    sha = hashlib.sha256(raw).hexdigest()
    name = blob_name(prefix, run_id, body["place_id"], body["attempts"], body["repair_gen"], sha)
    ref = {"place_id": body["place_id"], "attempts": body["attempts"],
           "repair_gen": body["repair_gen"], "blob": name, "sha256": sha,
           "outcome": outcome, "rank": rank_of(body)}
    return name, gzip.compress(raw, mtime=0), ref   # mtime=0: same body, same bytes


class Forbidden(Exception):
    """403 on the PUT: the SAS expired or lacks cw. Stop leasing at once."""


def blob_put(name: str, gz: bytes, sha256: str, sas: str | None = None) -> None:
    """One PUT, If-None-Match: *. Returns when acknowledged (201, or 409 BlobAlreadyExists
    for this content-addressed name); raises otherwise (Forbidden on 403)."""
    from cloud import blob_shots
    sas = sas or staging_sas()
    if not sas:
        raise RuntimeError("no result SAS (PHASE2_SHOT_BLOB_SAS / MEDIA_BLOB_WRITE_SAS)")
    try:
        blob_shots._put_blob(name, gz, "application/gzip", {"sha256": sha256, "schema": SCHEMA_VERSION},
                             sas_url=sas, skip_existing=True)   # True = 201, False = 409/412 exists
    except RuntimeError as e:
        if "forbidden" in str(e):
            raise Forbidden(str(e)) from e
        raise


def _default_spool_dir() -> Path:
    return Path(os.environ.get("PHASE2_SPOOL_DIR")
                or Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "p2spool")


class Stager:
    """Spool + one uploader thread + the refs waiting for /phase2/staged (mode 1 only)."""

    def __init__(self, run_id: str, mode_: str, *, worker: str = "", shard: str = "",
                 spool_dir=None, put=None, cap: int = SPOOL_CAP, start: bool = True):
        if mode_ not in PREFIX:
            raise ValueError(f"Stager needs PHASE2_STAGING shadow or 1, got {mode_!r}")
        self.run_id, self.mode = str(run_id), mode_
        self.prefix = PREFIX[mode_]
        self.worker, self.shard = str(worker), str(shard)
        self.dir = Path(spool_dir) if spool_dir else _default_spool_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._put = put or blob_put
        self.cap = cap
        self._cond = threading.Condition()
        self._spool: dict[str, dict] = {}       # blob name -> ref (object not yet acknowledged)
        self._next: dict[str, tuple[float, float]] = {}   # name -> (next try, backoff s)
        self._blocked = False
        self.refs: list[dict] = []               # acknowledged, not yet reported (mode 1)
        self.forbidden = False
        self.uploaded = self.put_errors = self.dropped = 0
        self.accepted = self.superseded = self.rejected = 0
        self._stop = False
        for p in sorted(self.dir.glob("*.ref.json")):   # leftovers (a replay, or a restart)
            ref = json.loads(p.read_text(encoding="ascii"))
            self._spool[ref["blob"]] = ref
        self._thread = threading.Thread(target=self._run, name="p2-stager", daemon=True)
        if start:
            self._thread.start()

    # -- files ---------------------------------------------------------------------
    def _paths(self, name: str) -> tuple[Path, Path]:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        return self.dir / f"{stem}", self.dir / f"{stem}.ref.json"

    def stage(self, result: dict, outcome: str, *, repair_gen: int = 0, attributes=None) -> dict:
        """Spool one outlet's object (durable on disk) and wake the uploader. Returns its ref."""
        name, gz, ref = build_object(self.run_id, result, outcome, repair_gen=repair_gen,
                                     attributes=attributes, worker=self.worker, shard=self.shard,
                                     prefix=self.prefix)
        obj, meta = self._paths(name)
        obj.write_bytes(gz)
        meta.write_text(json.dumps(ref), encoding="ascii")   # last: a ref file means a whole object
        with self._cond:
            self._spool[name] = ref
            self._cond.notify_all()
        return ref

    # -- uploader ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                now = time.monotonic()
                due = [n for n in self._spool if self._next.get(n, (0.0, 0.0))[0] <= now]
                if not due:
                    if self._stop:
                        return
                    wake = min((t for t, _ in self._next.values()), default=now + 1.0)
                    self._cond.wait(max(0.05, min(1.0, wake - now)))
                    continue
            for name in due:
                self.upload_one(name)

    def upload_one(self, name: str) -> bool:
        with self._cond:
            ref = self._spool.get(name)
        if ref is None:
            return True
        obj, meta = self._paths(name)
        try:
            self._put(name, obj.read_bytes(), ref["sha256"])
        except Exception as e:  # noqa: BLE001 - every failure keeps the object; 403 also stops leasing
            if self.mode == "shadow":   # compare-only: drop, never block (Codex M4)
                with self._cond:
                    self.put_errors += 1
                    self.dropped += 1
                    self._spool.pop(name, None)
                    self._next.pop(name, None)
                    self._cond.notify_all()
                meta.unlink(missing_ok=True)
                obj.unlink(missing_ok=True)
                logger.warning("phase2.shadow_put_dropped blob=%s err=%r", name, e)
                return False
            with self._cond:
                self.put_errors += 1
                if isinstance(e, Forbidden):
                    self.forbidden = True
                back = min(60.0, max(1.0, 2 * self._next.get(name, (0.0, 0.5))[1]))
                self._next[name] = (time.monotonic() + back, back)
            logger.warning("phase2.stage_put_failed blob=%s retry_in=%.0fs err=%r", name, back, e)
            return False
        with self._cond:
            self._spool.pop(name, None)
            self._next.pop(name, None)
            self.uploaded += 1
            if self.mode == "1":
                self.refs.append(ref)
            if not self._spool:
                self._blocked = False
            self._cond.notify_all()
        meta.unlink(missing_ok=True)   # ref file first: a crash here leaves a bare object, never a ref without one
        obj.unlink(missing_ok=True)
        return True

    # -- worker controls -----------------------------------------------------------
    def spooled(self) -> int:
        with self._cond:
            return len(self._spool)

    def full(self) -> bool:
        """Lease nothing while True: set at `cap` spooled objects, cleared when the spool is empty.
        Always False in shadow mode."""
        if self.mode == "shadow":
            return False
        with self._cond:
            if len(self._spool) >= self.cap:
                self._blocked = True
            elif not self._spool:
                self._blocked = False
            return self._blocked

    def lease_room(self, held: int) -> int:
        """How many places may be leased now, given `held` leased places not yet staged:
        0 while full(); else SPOOL_CAP + HELD_MAX - spooled - held (Codex M6: spooled + held
        never exceeds 70). Unbounded (a large number) in shadow mode."""
        if self.mode == "shadow":
            return 1 << 30
        if self.full():
            return 0
        with self._cond:
            return max(0, self.cap + HELD_MAX - len(self._spool) - int(held))

    def report_due(self, since_s: float) -> bool:
        with self._cond:
            return bool(self.refs) and (len(self.refs) >= REPORT_EVERY_N or since_s >= REPORT_EVERY_S)

    def report(self, state) -> dict:
        """POST every acknowledged ref to /phase2/staged (<= 500 per call). A failed call keeps
        its refs (and the ones after it) for the next report: refs are never dropped."""
        tot = {"accepted": 0, "superseded": 0, "rejected": 0}
        while True:
            with self._cond:
                chunk = self.refs[:REPORT_MAX]
            if not chunk:
                return tot
            try:
                out = state.submit_staged(self.run_id, chunk) or {}
            except Exception:  # noqa: BLE001
                logger.warning("phase2.staged_report_failed refs=%d kept", len(chunk), exc_info=True)
                return tot
            with self._cond:
                del self.refs[:len(chunk)]
                for k in tot:
                    n = int(out.get(k, 0) or 0)
                    tot[k] += n
                    setattr(self, k, getattr(self, k) + n)
            if out.get("rejected"):
                logger.error("phase2.staged_rejected n=%s (ref disagrees with its object name)",
                             out.get("rejected"))
            logger.info("phase2.staged_report refs=%d %s", len(chunk), out)

    def drain(self, timeout_s: float = DRAIN_S) -> int:
        """Shard end: retry every spooled object (backoff reset) until empty or timeout_s.
        Returns the objects still spooled (lost work: exit 3)."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            self._next.clear()
            self._cond.notify_all()
        while self.spooled() and time.monotonic() < deadline:
            if not self._thread.is_alive():
                with self._cond:
                    names = list(self._spool)
                for n in names:
                    self.upload_one(n)
                if self.spooled():
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            with self._cond:
                self._cond.wait(0.5)
        return self.spooled()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()


def replay_spool(spool_dir: str, run_id: str, state=None) -> dict:
    """--replay-spool: upload the objects of a shard artifact and report them to
    /phase2/staged (rule F accepts them after a re-lease). Shadow objects are uploaded only."""
    from cloud.http_state import HttpState
    out = {}
    for m in ("1", "shadow"):
        st = Stager(run_id, m, spool_dir=spool_dir, start=False)
        st._spool = {n: r for n, r in st._spool.items() if n.startswith(PREFIX[m] + "/")}
        if not st._spool:
            continue
        left = st.drain(DRAIN_S)
        rep = st.report(state or HttpState()) if m == "1" else {}
        out[m] = {"uploaded": st.uploaded, "left": left, "refs_unreported": len(st.refs), **rep}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay-spool", required=True, metavar="DIR")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out = replay_spool(a.replay_spool, a.run_id)
    print(json.dumps(out))
    return 3 if any(v.get("left") or v.get("refs_unreported") for v in out.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
