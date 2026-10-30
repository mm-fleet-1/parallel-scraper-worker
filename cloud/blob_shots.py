"""Azure Blob media store for the PUBLIC Phase 2 worker (creds-free).

Blob is the PERMANENT store for Phase 2 media. The worker uploads during the
scrape; the private indexer (cloud/drain_screenshots_blob.py) records rows in
the DB. Layout mirrors the old per-place Drive folders:

    <container>/<run_id>/<place_id>/
        <place_id>_1.jpg            place photos (Google CDN -> Blob)
        <place_id>_2.jpg
        screenshot_overview.png     panel screenshots
        screenshot_reviews.png

Everything for one place is grouped under the ``<run_id>/<place_id>/`` prefix,
so "all media for a place" is one prefix-list call — no manifest needed.

Security model (matches the lease/submit API): public compute holds NO account
keys — only ``PHASE2_SHOT_BLOB_SAS``, a container SAS URL scoped to **create +
write only** (no read/list/delete), e.g.:

    https://<account>.blob.core.windows.net/<container>?sv=...&sp=cw&sig=...

Uploads set a real Content-Type + inline disposition so the stored URL renders
directly in a browser tab or an <img> tag (no download prompt).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

_PUT_ATTEMPTS = 3
_PUT_BACKOFF_S = 1.0
_PUT_TIMEOUT_S = 30
_PHOTO_GET_TIMEOUT_S = 20
_PHOTO_POOL_WORKERS = 6

_EXT_BY_CTYPE = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/webp": "webp", "image/gif": "gif",
}

# Reuse the engine's photo-URL dedupe + hi-res upscale when the engine is present
# (it always is in the worker); fall back to order-preserving dedupe / identity.
try:
    from parallel_scraper.phase2_metadata import (
        _dedupe_image_urls as _dedupe_urls,
        _upscale_image_url as _upscale_url,
    )
except Exception:  # pragma: no cover
    def _dedupe_urls(urls):
        return list(dict.fromkeys(urls))

    def _upscale_url(u):
        return u


def blob_sas_url() -> str | None:
    """The container SAS URL from env, or None when Blob media is off."""
    return os.environ.get("PHASE2_SHOT_BLOB_SAS") or None


def photos_enabled() -> bool:
    """Photo mirroring (CDN -> Blob) is opt-out; needs the SAS to be set anyway."""
    return os.environ.get("PHASE2_PHOTOS_TO_BLOB", "1") != "0" and bool(blob_sas_url())


def photos_max() -> int:
    try:
        return max(0, int(os.environ.get("PHASE2_PHOTOS_MAX", "15")))
    except ValueError:
        return 15


def _split_sas(sas_url: str) -> tuple[str, str]:
    """Split a container SAS URL into (container_base_url, sas_query)."""
    base, _, query = sas_url.partition("?")
    return base.rstrip("/"), query


def screenshot_name(run_id: str, place_id: str, kind: str) -> str:
    return f"{run_id}/{place_id}/screenshot_{kind}.png"


def photo_name(run_id: str, place_id: str, seq: int, ext: str) -> str:
    return f"{run_id}/{place_id}/{place_id}_{seq}.{ext}"


def photo_blob_name(place_id: str, sha1: str) -> str:
    """Content-addressed master: photos/<pid>/<sha1(source bytes)[:20]>.jpg (§4.1)."""
    return f"photos/{place_id}/{sha1[:20]}.jpg"


def _put_blob(name: str, body: bytes, content_type: str, meta: dict,
              sas_url: str | None = None, session=None,
              skip_existing: bool = False) -> bool:
    """PUT one blob (BlockBlob). Raises on final failure.

    `skip_existing` sends `If-None-Match: *`: Azure refuses with 409 (BlobAlreadyExists) when a blob
    of that name already exists, and that is returned as False (nothing
    written). Makes photo mirroring idempotent across re-runs — the SAS is cw
    only, so a HEAD exists-check is not possible; the CDN download still
    happens, only the upload/overwrite is saved. Returns True when written."""
    sas_url = sas_url or blob_sas_url()
    if not sas_url:
        raise RuntimeError("PHASE2_SHOT_BLOB_SAS not set")
    base, query = _split_sas(sas_url)
    url = f"{base}/{name}?{query}"
    headers = {
        "x-ms-blob-type": "BlockBlob",
        # Persisted onto the blob so the URL renders inline in browsers/<img>.
        "Content-Type": content_type,
        "x-ms-blob-content-type": content_type,
        "x-ms-blob-content-disposition": "inline",
    }
    # Metadata names must be simple identifiers; values ASCII.
    for k, v in meta.items():
        headers[f"x-ms-meta-{k}"] = str(v)
    if skip_existing:
        headers["If-None-Match"] = "*"
    http = session or requests
    last_exc: Exception | None = None
    for attempt in range(_PUT_ATTEMPTS):
        try:
            resp = http.put(url, data=body, headers=headers, timeout=_PUT_TIMEOUT_S)
            if resp.status_code == 201:
                return True
            # Only a real "already there" is a skip; any other 409 is a failure (§4.1).
            if skip_existing and (resp.status_code == 412 or (
                    resp.status_code == 409 and (getattr(resp, "headers", None) or {})
                    .get("x-ms-error-code") == "BlobAlreadyExists")):
                return False
            # 403 = SAS expired/wrong perms: retrying won't help.
            if resp.status_code == 403:
                raise RuntimeError(f"blob PUT forbidden (SAS expired or missing cw perms): {resp.status_code}")
            last_exc = RuntimeError(f"blob PUT failed status={resp.status_code}")
        except RuntimeError:
            raise
        except Exception as exc:  # network/timeout — retry
            last_exc = exc
        if attempt + 1 < _PUT_ATTEMPTS:
            time.sleep(_PUT_BACKOFF_S * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("blob PUT failed")


def put_shot(run_id: str, place_id: str, kind: str, attempts: int,
             local_path: str, sas_url: str | None = None, session=None) -> str:
    """PUT one screenshot PNG and return its blob name. Raises on final failure so the
    caller can fall back to the artifact-dir path."""
    with open(local_path, "rb") as f:
        body = f.read()
    name = screenshot_name(run_id, place_id, kind)
    _put_blob(name, body, "image/png",
              {"runid": run_id, "placeid": place_id, "kind": kind,
               "attempts": int(attempts)},
              sas_url=sas_url, session=session)
    return name


def upload_menu_photos(run_id: str, place_id: str, menu_photos,
                       sas_url: str | None = None, session=None) -> int:
    """Mirror menu-card photos into Blob as
    ``<run_id>/<place_id>/<place_id>_menu_N.jpg`` (0-based, matching the
    horeca capture order so DB rows pair 1:1). `menu_photos` is the horeca
    list of {url, date}. Full-res via =w2000. Best-effort like place photos."""
    items = [(i, p.get("url"), p.get("date") or "")
             for i, p in enumerate(menu_photos or []) if p.get("url")]
    if not items:
        return 0
    http = session or requests

    def _one(args) -> bool:
        seq, url, date = args
        try:
            full = url.split("=")[0] + "=w2000"
            r = http.get(full, timeout=_PHOTO_GET_TIMEOUT_S)
            if r.status_code != 200 or not r.content:
                return False
            _put_blob(f"{run_id}/{place_id}/{place_id}_menu_{seq}.jpg", r.content,
                      "image/jpeg",
                      {"runid": run_id, "placeid": place_id, "kind": "menu",
                       "seq": seq, "caption": date.encode("ascii", "ignore").decode()},
                      sas_url=sas_url, session=session, skip_existing=True)
            return True
        except Exception:
            logger.debug("menu_upload_failed pid=%s seq=%d", place_id, seq, exc_info=True)
            return False

    with ThreadPoolExecutor(max_workers=min(_PHOTO_POOL_WORKERS, len(items))) as ex:
        stored = sum(ex.map(_one, items))
    if stored < len(items):
        logger.info("menu_photos_partial pid=%s stored=%d of %d", place_id, stored, len(items))
    return stored


def upload_place_photos(run_id: str, place_id: str, image_urls,
                        max_photos: int | None = None,
                        sas_url: str | None = None, session=None) -> int:
    """Mirror a place's photos (Google CDN URLs) into Blob as
    ``<run_id>/<place_id>/<place_id>_N.jpg``. Best-effort: per-photo failures
    are logged and skipped; returns the number stored. Numbering follows the
    deduped URL order (1-based), matching the old local/Drive convention."""
    if max_photos is None:
        max_photos = photos_max()
    urls = [u for u in (image_urls or []) if u]
    try:
        # _dedupe_urls parses the engine's ';'-joined URL *string*; handing it the
        # list stringified the whole list into one junk URL and silently mirrored
        # zero photos (no exception, so the guard below never caught it).
        urls = [_upscale_url(u) for u in _dedupe_urls(";".join(urls))]
    except Exception:
        pass
    urls = urls[:max_photos]
    if not urls:
        return 0
    http = session or requests

    def _one(args) -> bool:
        seq, url = args
        try:
            r = http.get(url, timeout=_PHOTO_GET_TIMEOUT_S)
            if r.status_code != 200 or not r.content:
                return False
            ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()
            ext = _EXT_BY_CTYPE.get(ctype, "jpg")
            _put_blob(photo_name(run_id, place_id, seq, ext), r.content,
                      ctype if ctype in _EXT_BY_CTYPE else "image/jpeg",
                      {"runid": run_id, "placeid": place_id, "kind": "photo", "seq": seq},
                      sas_url=sas_url, session=session, skip_existing=True)
            return True
        except Exception:
            logger.debug("photo_upload_failed pid=%s seq=%d", place_id, seq, exc_info=True)
            return False

    with ThreadPoolExecutor(max_workers=min(_PHOTO_POOL_WORKERS, len(urls))) as ex:
        stored = sum(ex.map(_one, enumerate(urls, start=1)))
    if stored < len(urls):
        logger.info("photos_partial pid=%s stored=%d of %d", place_id, stored, len(urls))
    return stored


_MASTER_SIDE = 1600
_MASTER_QUALITY = 85


def resize_master(src: bytes) -> tuple[bytes, int, int]:
    """Longest side 1600 px, JPEG q85, EXIF orientation applied (the re-encode drops
    EXIF, so an unrotated phone photo showed 90 degrees off). Returns (jpeg_bytes, width, height)."""
    from PIL import Image, ImageOps  # lazy: the worker imports this module without Pillow
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(src))).convert("RGB")
    img.thumbnail((_MASTER_SIDE, _MASTER_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=_MASTER_QUALITY)
    return out.getvalue(), img.width, img.height


def _source_url(item) -> str:
    """CDN rewrite to =s1600; Street View thumbs are fetched as given."""
    if item.source == "street_view":
        return item.url
    return item.url.split("=")[0] + "=s1600"


# streetviewpixels answers 403 PERMISSION_DENIED to the python-requests User-Agent
# (any size, with or without Referer); a browser UA gets 200 (checked live, Oct 2026).
_SV_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
               "Referer": "https://www.google.com/"}


def _sv_urls(url: str) -> list[str]:
    """The Street View URL as given, then at w=1200 (h kept in proportion 4:3)."""
    out = [url]
    if "streetviewpixels" in url:
        alt = re.sub(r"([?&])h=\d+", r"\1h=900", re.sub(r"([?&])w=\d+", r"\1w=1200", url))
        if alt != url:
            out.append(alt)
    return out


def _sv_get(url: str) -> bytes | None:
    r = requests.get(url, headers=_SV_HEADERS, timeout=_PHOTO_GET_TIMEOUT_S)
    return r.content if r.status_code == 200 and r.content else None


def mirror_selected(run_id: str, place_id: str, selected, *, fetch=None, put=None,
                    resize=None, pool_workers: int = _PHOTO_POOL_WORKERS,
                    sv_fetch=None) -> list[dict]:
    """Mirror selected photos (gallery_select.Selected, plus the Street View item)
    to ``photos/<pid>/<sha1>.jpg``; one outcome per item, in input order (§4.1):
    {ordinal, blob_name, sha1, width, height, bytes, status}, ordinal = selection
    index, status written | exists | failed.

    Injected for tests: fetch(url) -> bytes | None; put(name, body, meta) ->
    'written' | 'exists' (raise on failure); resize(bytes) -> (bytes, w, h);
    sv_fetch(url) -> bytes | None, the Street View fallback (browser headers) tried on
    the URL as given and then at w=1200 when `fetch` returns nothing.
    """
    if fetch is None:
        def fetch(url):
            r = requests.get(url, timeout=_PHOTO_GET_TIMEOUT_S)
            return r.content if r.status_code == 200 and r.content else None
    if put is None:
        def put(name, body, meta):
            return "written" if _put_blob(name, body, "image/jpeg", meta,
                                          skip_existing=True) else "exists"
    resize = resize or resize_master
    sv_fetch = sv_fetch or _sv_get

    def one(args) -> dict:
        i, item = args
        out = {"ordinal": i, "blob_name": None, "sha1": None, "width": None,
               "height": None, "bytes": None, "status": "failed"}
        try:
            url = _source_url(item)
            src = fetch(url)
            if not src and item.source == "street_view":
                for u in _sv_urls(url):
                    try:
                        src = sv_fetch(u)
                    except Exception:
                        src = None
                    if src:
                        break
            if not src:
                return out
            sha1 = hashlib.sha1(src).hexdigest()[:20]
            body, w, h = resize(src)
            name = photo_blob_name(place_id, sha1)
            out.update(sha1=sha1, width=w, height=h, bytes=len(body))
            meta = {"runid": run_id, "placeid": place_id, "source": item.source,
                    "googleurl": (item.url or "").encode("ascii", "ignore").decode()[:2000],
                    "dhash": f"{item.dhash:016x}" if item.dhash is not None else "",
                    "photodate": item.photo_month or ""}
            status = put(name, body, meta)
            if status in ("written", "exists"):
                out.update(status=status, blob_name=name)
        except Exception:
            logger.debug("photo_mirror_failed pid=%s ordinal=%d", place_id, i, exc_info=True)
        return out

    items = list(enumerate(selected))
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(pool_workers, len(items)))) as ex:
        return list(ex.map(one, items))
