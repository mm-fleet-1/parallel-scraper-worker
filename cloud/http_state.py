"""HTTP state backend for the PUBLIC Phase 2 worker.

Implements the small surface the worker needs (get_run / claim_place_ids / submit /
reclaim_place_ids) by calling the lease/submit API over HTTPS. Holds only
`WORKER_API_BASE` + `WORKER_TOKEN` — no DB creds, no Drive creds. Retries transient
network/5xx errors with backoff (lease/submit are idempotent on the server side)."""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time

import requests

logger = logging.getLogger(__name__)


class PayloadTooLarge(Exception):
    """HTTP 413. Not a RequestException: _request never retries it (same body, same answer)."""


class HttpState:
    def __init__(self, base: str | None = None, token: str | None = None, timeout: float = 60.0):
        self.base = (base or os.environ["WORKER_API_BASE"]).rstrip("/")
        self.token = token or os.environ["WORKER_TOKEN"]
        self.timeout = timeout
        self.last_lease: dict = {}   # last /phase2/lease body: pending / retry_after_s on empty
        self.last_reclaim: dict = {}  # last /phase2/reclaim body
        self.api_503 = 0             # 503 + Retry-After answers (API shed the call before any DB work)
        self._n_lock = threading.Lock()
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {self.token}"
        # When the API uses a self-signed cert (e.g. Azure static IP, no domain), pin it:
        # WORKER_CA_BUNDLE points at that public cert so TLS is verified, not disabled.
        ca_bundle = os.environ.get("WORKER_CA_BUNDLE")
        if ca_bundle:
            self._s.verify = ca_bundle

    def fresh_connection(self) -> None:
        """Drop pooled keep-alive connections, so the next request may reach another API pod
        (a 'pending' empty lease means another pod may hold this run's leases)."""
        try:
            self._s.close()
        except Exception:  # noqa: BLE001
            pass

    # Retry transient 5xx / connection errors long enough to ride out a single API pod
    # restart or a brief overload (a single replica means there IS such a window). With
    # the defaults below the backoff sums to ~90s before giving up. lease/submit are
    # idempotent server-side (submit is lease-guarded; re-leasing at worst grabs a few
    # extra pids that get processed once), so retrying is safe.
    _ATTEMPTS = int(os.environ.get("WORKER_HTTP_ATTEMPTS", "12"))
    _BACKOFF_CAP = float(os.environ.get("WORKER_HTTP_BACKOFF_CAP_S", "10"))

    def _request(self, method: str, path: str, *, json=None, params=None):
        last = None
        for attempt in range(self._ATTEMPTS):
            final = attempt == self._ATTEMPTS - 1
            try:
                r = self._s.request(method, self.base + path, json=json, params=params, timeout=self.timeout)
                if r.status_code == 413:
                    raise PayloadTooLarge(f"413 {path}: {r.text[:200]}")
                ra = r.headers.get("Retry-After") if r.status_code == 503 else None
                if ra is not None:
                    # Load shed (cloud/api.py): no DB work was done. Wait what the API asks
                    # (+-30% jitter so shards do not return together). The last attempt
                    # raises below, and the caller keeps its rows for the next flush.
                    with self._n_lock:
                        self.api_503 += 1
                    if not final:
                        last = RuntimeError(f"503 Retry-After {ra}")
                        time.sleep(self._retry_after_s(ra))
                        continue
                if r.status_code >= 500 and not final:
                    last = RuntimeError(f"{r.status_code}: {r.text[:200]}")
                    time.sleep(min(self._BACKOFF_CAP, 0.5 * (2 ** attempt)) + random.uniform(0, 0.5))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                last = exc
                if not final:
                    time.sleep(min(self._BACKOFF_CAP, 0.5 * (2 ** attempt)) + random.uniform(0, 0.5))
                    continue
                raise
        if last is not None:
            raise last
        raise RuntimeError("unreachable")

    @staticmethod
    def _retry_after_s(value: str) -> float:
        try:
            base = max(1.0, float(value))
        except (TypeError, ValueError):
            base = 15.0
        return min(60.0, base * random.uniform(0.7, 1.3))

    # --- surface the worker calls -------------------------------------------------
    def get_run(self, run_id: str):
        try:
            self._request("GET", "/phase2/progress", params={"run_id": run_id})
            return {"run_id": run_id}
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                return None
            raise

    def claim_place_ids(self, run_id: str, worker, n: int) -> list[dict]:
        body = {"run_id": run_id, "worker": str(worker), "n": int(n)}
        mode = (os.environ.get("PHASE2_STAGING") or "0").strip().lower()
        if mode != "0":   # §13c: lets the API hold leases of staging runs during an ingest pause
            body["staging"] = mode
        out = self._request("POST", "/phase2/lease", json=body)
        self.last_lease = out
        return out.get("pids", [])

    # A 120 s flush is ~12 rows (~0.2-0.5 MB), but a failed flush requeues its rows, so
    # the next batch grows. Past cloud/api.py WORKER_MAX_RESULTS (200 rows) or the ingress
    # body limit (nginx default 1 MB) every retry was a 413 and the rows never landed.
    _SUBMIT_MAX_ROWS = int(os.environ.get("WORKER_SUBMIT_MAX_ROWS", "200"))
    _SUBMIT_MAX_BYTES = int(os.environ.get("WORKER_SUBMIT_MAX_BYTES", "900000"))

    def submit(self, run_id: str, done: list[dict], failed: list[dict]) -> dict:
        """POST results in chunks whose whole request body stays under the byte cap and
        the row cap; counts are summed. On 413 a chunk is halved and resent, down to one
        row. A single row that still gets 413 is logged with its place_id and returned in
        `rejected`: the caller does not requeue it (it would 413 forever) and the shard
        gate counts it as lost work (exit 3). Any other failure raises as before: the
        caller requeues all rows, and rows that already landed come back stale."""
        items = [("done", d) for d in done] + [("failed", f) for f in failed]
        if not items:
            return self._request("POST", "/phase2/results",
                                 json={"run_id": run_id, "done": [], "failed": []})
        total: dict = {"rejected": []}
        envelope = len(json.dumps({"run_id": run_id, "done": [], "failed": []}))
        chunk: list = []
        size = envelope
        for kind, item in items:
            # exact body length as requests' json.dumps writes it: ", " between list items
            b = len(json.dumps(item)) + (2 if any(k == kind for k, _ in chunk[-1:]) else 0)
            if chunk and (len(chunk) >= self._SUBMIT_MAX_ROWS or size + b > self._SUBMIT_MAX_BYTES):
                self._post_results(run_id, chunk, total)
                chunk, size = [], envelope
                b = len(json.dumps(item))
            chunk.append((kind, item))
            size += b
        self._post_results(run_id, chunk, total)
        return total

    def submit_staged(self, run_id: str, refs: list[dict]) -> dict:
        """POST refs to durable result objects (design §13c), <= 500 per call; counts summed.
        A failed chunk raises: the caller keeps every ref not yet answered (rule F makes a
        re-report of an answered ref a no-op)."""
        total: dict = {}
        for i in range(0, len(refs), 500):
            out = self._request("POST", "/phase2/staged", json={"run_id": run_id, "refs": refs[i:i + 500]})
            for k, v in (out or {}).items():
                if isinstance(v, int) and not isinstance(v, bool):
                    total[k] = total.get(k, 0) + v
        return total

    def _post_results(self, run_id: str, chunk: list, total: dict) -> None:
        body = {"run_id": run_id, "done": [i for k, i in chunk if k == "done"],
                "failed": [i for k, i in chunk if k == "failed"]}
        try:
            out = self._request("POST", "/phase2/results", json=body)
        except PayloadTooLarge:
            if len(chunk) == 1:
                pid = chunk[0][1].get("place_id")
                logger.error("phase2.submit_rejected_413 place_id=%s bytes=%d (dropped; counted as lost)",
                             pid, len(json.dumps(body)))
                total["rejected"].append(pid)
                return
            half = len(chunk) // 2
            self._post_results(run_id, chunk[:half], total)
            self._post_results(run_id, chunk[half:], total)
            return
        for k, v in (out or {}).items():
            if isinstance(v, int) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v

    def submit_attributes(self, run_id: str, rows: list[dict]) -> dict:
        """POST Overview-pass scalars. Best-effort at the call site: these are
        additive fields, never a reason to fail a scrape."""
        return self._request("POST", "/phase2/attributes",
                             json={"run_id": run_id, "rows": rows})

    def mark_delisted(self, run_id: str, place_id: str, attempts: int, observed_at: str,
                      evidence_text: str | None = None, page_url: str | None = None) -> dict:
        """Tombstone a delisted place (design §8.4); lease-gated server side. {accepted, stale}."""
        return self._request("POST", "/phase2/delisted",
                             json={"run_id": run_id, "place_id": place_id, "attempts": int(attempts),
                                   "observed_at": observed_at, "evidence_text": evidence_text,
                                   "page_url": page_url})

    def reclaim_place_ids(self, run_id: str, older_than_minutes: int = 30) -> int:
        out = self._request("POST", "/phase2/reclaim",
                            json={"run_id": run_id, "older_than_minutes": int(older_than_minutes)})
        self.last_reclaim = out   # may carry skipped=cooldown|busy (cloud/api.py _reclaim)
        return int(out.get("reclaimed", 0))

    def upload_screenshots(self, run_id: str, place_id: str, attempts: int, paths: dict) -> dict:
        """POST page-screenshot bytes (multipart) to the API, which uploads them to Drive.
        `paths` = {kind: local_png_path}. Carries the lease `attempts` so the server can
        reject stale uploads. Best-effort — the worker wraps this in try/except."""
        if not paths:
            return {"accepted": 0, "stale": 0, "kinds": []}
        handles = []
        try:
            files = []
            for kind, p in paths.items():
                fh = open(p, "rb")
                handles.append(fh)
                files.append((kind, (f"{place_id}_{kind}.png", fh, "image/png")))
            r = self._s.post(self.base + "/phase2/screenshots",
                             data={"run_id": run_id, "place_id": place_id, "attempts": int(attempts)},
                             files=files, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        finally:
            for fh in handles:
                try:
                    fh.close()
                except Exception:
                    pass
