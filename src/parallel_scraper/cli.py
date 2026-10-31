"""argparse + interactive column picker."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from parallel_scraper.config import (
    ALWAYS_INCLUDED,
    COLUMN_CATALOG,
    LEAN_PROFILE,
    PROFILES,
    ParallelConfig,
)
from parallel_scraper.scraper import ParallelScraper

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parallel-scraper",
        description="Parallel Google Places API placeId discovery + deep metadata scraper.",
    )
    p.add_argument("--osm-id", type=int, default=13312356, help="OSM relation/way/node ID")
    p.add_argument("--osm-type", default="relation", choices=["relation", "way", "node"])
    p.add_argument("--city", default="Mumbai")
    p.add_argument("--grid-size-meters", type=int, default=1000)
    p.add_argument("--grid-type", default="square", choices=["square", "hex"])
    p.add_argument("--adaptive-grid", action="store_true",
                   help="Quadtree discovery: start at --adaptive-coarse-meters and "
                        "subdivide saturated cells down to --grid-size-meters")
    p.add_argument("--adaptive-coarse-meters", type=int, default=4000,
                   help="Starting cell edge for --adaptive-grid (default 4000)")
    p.add_argument("--queries", default=None,
                   help="Comma-separated list of search queries (e.g. shops,restaurants). "
                        "On resume, omit to inherit queries from run_config.json.")
    p.add_argument("--discovery-backend", default="places_api",
                   choices=["places_api", "maps_frontend"],
                   help="Phase 1 discovery backend")
    p.add_argument("--places-tier", default="ESSENTIALS",
                   choices=["ESSENTIALS", "PRO", "ENTERPRISE"],
                   help="Google Places API tier for Phase 1 Text Search")
    p.add_argument("--places-language", default="en",
                   help="Text Search languageCode (e.g. es for Argentina)")
    p.add_argument("--places-region", default="IN",
                   help="Text Search regionCode, CLDR two-letter (e.g. AR)")
    p.add_argument("--phase1-workers", type=int, default=3,
                   help="Concurrent Phase 1 workers")
    p.add_argument("--phase2-workers", type=int, default=5,
                   help="Concurrent Chromium browsers for deep metadata scrape (Phase 2)")
    # Back-compat: old --threads flag maps to --phase2-workers.
    p.add_argument("--threads", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--max-queue-size", type=int, default=1000)
    p.add_argument("--discovery-rps", type=float, default=2.0)
    p.add_argument("--consumer-rps", type=float, default=2.0, dest="consumer_rps_per_thread")
    p.add_argument("--consumer-delay-ms", type=int, default=500)
    p.add_argument("--max-retries-transient", type=int, default=4)
    p.add_argument("--heartbeat-interval-s", type=int, default=30)
    p.add_argument("--worker-browser-recycle-after", type=int, default=50,
                   help="Restart each Phase-2 Chromium after this many places "
                        "to bound memory growth (default 50, min 1).")
    p.add_argument("--no-image-download", action="store_true",
                   help="Metadata only: skip downloading image files to disk. "
                        "image_urls (Google CDN links) are still captured in the CSV.")
    p.add_argument("--capture-screenshots", action="store_true",
                   help="Save per-place panel screenshots (overview + reviews/histogram) "
                        "to outputs/<run>/screenshots/ for multimodal-LLM review.")
    p.add_argument("--master-dedup", default="inputs/known_placeids.csv",
                   dest="master_dedup_csv")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-id", default=None, help="Resume an existing run by ID")
    p.add_argument("--max-places", type=int, default=None,
                   help="Cap discovered placeIds (smoke-test)")
    p.add_argument("--dry-run", action="store_true",
                   help="Mock both phases — no live network")
    p.add_argument("--polygon-file", default=None,
                   help="Path to a JSON file with [[lat,lng],...] polygon coords, or "
                        "GeoJSON (Polygon/MultiPolygon/Feature/FeatureCollection). "
                        "When set, takes precedence over --osm-id.")
    p.add_argument("--columns", default=None,
                   help="Comma-separated column keys to save (skip the picker)")
    p.add_argument("--profile", default=None,
                   help=f"Preset column set: {', '.join(PROFILES.keys())}")
    p.add_argument("--list-columns", action="store_true",
                   help="Print the column catalog and exit")
    return p


def _print_catalog() -> None:
    print("Available Phase-2 columns (place_id always included):\n")
    for key, label in COLUMN_CATALOG.items():
        print(f"  {key:<22} {label}")
    print(f"\nProfiles: {', '.join(PROFILES.keys())}")


def _resolve_columns(args, run_dir: Path) -> tuple[str, ...]:
    """Resolve the operator-selected column tuple.

    Order of precedence:
      1. Existing run_config.json on resume.
      2. --columns CSV string.
      3. --profile preset.
      4. Interactive picker (questionary, with stdlib fallback).
    """
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        ex_cols = tuple(existing.get("selected_columns", []))
        if ex_cols:
            return tuple(c for c in ex_cols if c in COLUMN_CATALOG)

    if args.columns:
        return tuple(c.strip() for c in args.columns.split(",")
                     if c.strip() in COLUMN_CATALOG)

    if args.profile:
        if args.profile not in PROFILES:
            raise SystemExit(f"Unknown profile: {args.profile}. "
                             f"Available: {', '.join(PROFILES.keys())}")
        return PROFILES[args.profile]

    # Interactive picker
    return _interactive_picker()


def _resolve_queries(args, run_dir: Path | None) -> tuple[str, ...]:
    """Resolve the queries tuple.

    Precedence:
      1. Explicit --queries CLI arg (overrides everything; honor operator intent).
      2. queries from run_config.json on resume (prevents the 'forgot --queries
         on resume -> CLI default leaks a new query into Phase 1' footgun).
      3. Legacy default ('shops') for a fresh run that omits --queries.
    """
    # 1. Explicit CLI wins
    if args.queries is not None:
        parsed = tuple(q.strip() for q in args.queries.split(",") if q.strip())
        if parsed:
            return parsed

    # 2. Resume: inherit from run_config.json
    if run_dir is not None and (run_dir / "run_config.json").exists():
        try:
            existing = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            ex_queries = tuple(existing.get("queries") or ())
            if ex_queries:
                logger.info("queries.resume_from_config count=%d", len(ex_queries))
                return ex_queries
        except Exception:
            logger.exception("queries.run_config_read_failed")

    # 3. Fresh-run fallback
    return ("shops",)


def _interactive_picker() -> tuple[str, ...]:
    try:
        import questionary
        choices = [
            questionary.Choice(title=f"{label} ({key})",
                               value=key,
                               checked=(key in LEAN_PROFILE))
            for key, label in COLUMN_CATALOG.items()
        ]
        answer = questionary.checkbox(
            "Select Phase-2 metadata columns to save (place_id always included):",
            choices=choices,
        ).ask()
        if not answer:
            print("No columns selected — defaulting to lean profile.")
            return LEAN_PROFILE
        return tuple(answer)
    except ImportError:
        return _stdlib_picker()


def _stdlib_picker() -> tuple[str, ...]:
    print("\nSelect Phase-2 columns. Enter comma-separated numbers, or blank for lean profile:\n")
    keys = list(COLUMN_CATALOG.keys())
    for i, key in enumerate(keys):
        marker = "*" if key in LEAN_PROFILE else " "
        print(f"  {i+1:2d}. [{marker}] {key:<22} {COLUMN_CATALOG[key]}")
    print("\n(items marked * are in the lean default)")
    raw = input("> ").strip()
    if not raw:
        return LEAN_PROFILE
    chosen: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(keys):
                chosen.append(keys[idx])
        except ValueError:
            if tok in COLUMN_CATALOG:
                chosen.append(tok)
    return tuple(chosen) if chosen else LEAN_PROFILE


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_columns:
        _print_catalog()
        return 0

    polygon = None
    if args.polygon_file:
        from parallel_scraper.boundary import polygon_rings_from_geojson
        data = json.loads(Path(args.polygon_file).read_text(encoding="utf-8"))
        rings = polygon_rings_from_geojson(data)
        # Single ring keeps the legacy flat shape (round-trips through existing
        # run_config.json files); multiple rings nest one level deeper.
        if len(rings) == 1:
            polygon = tuple((lat, lng) for lat, lng in rings[0])
        else:
            polygon = tuple(tuple((lat, lng) for lat, lng in r) for r in rings)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If resuming, the run_dir already exists.
    if args.run_id:
        run_dir = output_dir / args.run_id
        if not run_dir.exists():
            print(f"Resume failed: run directory not found: {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = output_dir  # ParallelScraper will create the run_dir under here.

    # Resolve queries: on resume prefer run_config.json (prevents the leak where a
    # forgotten --queries on resume causes the CLI default 'shops' to add a brand
    # new query to Phase 1's work set).
    queries = _resolve_queries(args, run_dir if args.run_id else None)

    # Resume: run_config.json is authoritative for adaptive-grid settings —
    # same rationale as queries above. A forgotten --adaptive-grid on resume
    # would regenerate root cells at the floor size with different cell ids,
    # orphaning every done-cell state row.
    if args.run_id:
        try:
            cfg_prev = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            if "adaptive_grid" in cfg_prev:
                args.adaptive_grid = bool(cfg_prev["adaptive_grid"])
                args.adaptive_coarse_meters = int(
                    cfg_prev.get("adaptive_coarse_meters", args.adaptive_coarse_meters))
        except FileNotFoundError:
            pass

    # Resolve columns BEFORE constructing ParallelScraper (which writes run_config.json).
    selected_columns = _resolve_columns(args, run_dir if args.run_id else Path("/__never_exists__"))

    phase2 = args.threads if args.threads is not None else args.phase2_workers

    config = ParallelConfig(
        osm_relation_id=args.osm_id,
        osm_type=args.osm_type,
        city_name=args.city,
        grid_size_meters=args.grid_size_meters,
        grid_type=args.grid_type,
        adaptive_grid=args.adaptive_grid,
        adaptive_coarse_meters=args.adaptive_coarse_meters,
        queries=queries,
        discovery_backend=args.discovery_backend,
        places_tier=args.places_tier,
        places_language_code=args.places_language,
        places_region_code=args.places_region,
        phase1_workers=args.phase1_workers,
        phase2_workers=phase2,
        num_consumer_threads=phase2,    # back-compat field
        max_queue_size=args.max_queue_size,
        discovery_rps=args.discovery_rps,
        consumer_rps_per_thread=args.consumer_rps_per_thread,
        consumer_delay_ms=args.consumer_delay_ms,
        max_retries_transient=args.max_retries_transient,
        heartbeat_interval_s=args.heartbeat_interval_s,
        worker_browser_recycle_after=max(1, int(args.worker_browser_recycle_after)),
        selected_columns=selected_columns,
        master_dedup_csv=args.master_dedup_csv,
        output_dir=str(output_dir),
        run_id=args.run_id,
        max_places=args.max_places,
        dry_run=args.dry_run,
        download_images=not args.no_image_download,
        capture_screenshots=args.capture_screenshots,
        boundary_polygon=polygon,
    )

    print(f"\nStarting run with columns: {list(ALWAYS_INCLUDED) + list(selected_columns)}")
    if config.dry_run:
        print("(dry run — no live network)\n")

    try:
        scraper = ParallelScraper(config)
        result = scraper.run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception:
        logger.exception("scraper.fatal")
        return 1

    print(f"\nDone. Run dir: {result.get('run_dir')}")
    print(f"  discovered={result.get('discovered')} scraped={result.get('scraped')} "
          f"errors={result.get('errors')}")
    status = result.get("status")
    if status == "failed":
        return 1
    if status == "stopped":
        return 130
    if status == "incomplete":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
