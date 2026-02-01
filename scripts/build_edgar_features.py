#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to import path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.edgar_feature_store import build_edgar_feature_store_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EDGAR feature store (parquet-native)")
    parser.add_argument(
        "--edgar_path",
        type=Path,
        default=Path("data/raw/edgar/accessions"),
        help="Raw EDGAR accessions path (Delta or parquet)",
    )
    parser.add_argument(
        "--raw_edgar_delta",
        type=Path,
        default=None,
        help="Alias for --edgar_path",
    )
    parser.add_argument(
        "--snapshots_path",
        type=Path,
        default=Path("data/raw/offers"),
        help="Offers snapshots parquet/delta path",
    )
    parser.add_argument(
        "--snapshots_index_parquet",
        type=Path,
        default=None,
        help="Override: use this parquet as snapshots (cik, snapshot_ts or crawled_date_day)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output parquet/dir path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output dir (writes edgar_features/ inside); overridden by output_path",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Optional log directory (defaults to output_path parent / logs)",
    )
    parser.add_argument(
        "--snapshot_time_col",
        type=str,
        default="crawled_date",
        help="Snapshot timestamp column",
    )
    parser.add_argument("--batch_size", type=int, default=200_000)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument("--align_to_snapshots", action="store_true")
    parser.add_argument("--partition_by_year", action="store_true")
    parser.add_argument(
        "--cutoff_col",
        type=str,
        default=None,
        help="Optional cutoff column in snapshots (e.g., cutoff_ts) for leakage control",
    )
    args = parser.parse_args()
    if args.raw_edgar_delta is not None:
        args.edgar_path = args.raw_edgar_delta
    if args.output_dir is not None:
        args.output_path = args.output_dir / "edgar_features"
    elif args.output_path is None:
        args.output_path = Path("data/processed/edgar_features.parquet")

    if args.partition_by_year and args.output_path.suffix:
        args.output_path = args.output_path.with_suffix("")

    if args.log_dir is None:
        log_dir = args.output_path.parent / "logs"
    else:
        log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "build_edgar_features.log"

    logger = logging.getLogger("build_edgar_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    snapshots_path = args.snapshots_index_parquet if args.snapshots_index_parquet else args.snapshots_path
    if args.snapshots_index_parquet:
        args.snapshot_time_col = "crawled_date_day"

    logger.info("build_edgar_features start")
    logger.info("edgar_path=%s", args.edgar_path)
    logger.info("snapshots_path=%s", snapshots_path)
    logger.info("output_path=%s", args.output_path)
    logger.info("align_to_snapshots=%s", args.align_to_snapshots)
    logger.info("partition_by_year=%s", args.partition_by_year)
    logger.info("cutoff_col=%s", args.cutoff_col)

    out = build_edgar_feature_store_v2(
        edgar_path=args.edgar_path,
        snapshots_path=snapshots_path,
        output_path=args.output_path,
        snapshot_time_col=args.snapshot_time_col,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
        ema_alpha=args.ema_alpha,
        align_to_snapshots=args.align_to_snapshots,
        partition_by_year=args.partition_by_year,
        cutoff_col=args.cutoff_col,
        logger=logger,
    )
    logger.info("build_edgar_features done: %s", out)


if __name__ == "__main__":
    main()
