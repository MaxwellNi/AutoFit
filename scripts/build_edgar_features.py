#!/usr/bin/env python
"""Build EDGAR feature store from raw accessions. Writes MANIFEST.json to output_dir when --output_dir is used."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to import path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.edgar_feature_store import build_edgar_feature_store_v2


def _delta_version_and_files(path: Path) -> tuple[int | None, int]:
    try:
        from deltalake import DeltaTable
    except ImportError:
        return None, 0
    if not path.exists() or not (path / "_delta_log").exists():
        return None, 0
    try:
        dt = DeltaTable(str(path))
        ver = dt.version()
        files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
        return ver, len(files_list)
    except Exception:
        return None, 0


def _snapshots_stats(path: Path) -> dict:
    """Row count and unique keys from snapshots parquet or dataset."""
    out = {"rows": 0, "unique_cik": 0, "unique_keys": ""}
    if not path or not path.exists():
        return out
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        out["rows"] = len(df)
        if "cik" in df.columns:
            out["unique_cik"] = int(df["cik"].nunique())
        key_cols = [c for c in ["cik", "crawled_date_day", "snapshot_ts", "platform_name", "offer_id"] if c in df.columns]
        out["unique_keys"] = "|".join(key_cols[:3])
    except Exception:
        pass
    return out


def _output_stats(output_path: Path, partition_by_year: bool) -> dict:
    """Row count and column list from built output (dir or single parquet)."""
    out: dict = {"rows": 0, "columns": []}
    try:
        import pyarrow.dataset as ds
        part = "hive" if (partition_by_year or output_path.is_dir()) else None
        dset = ds.dataset(str(output_path), format="parquet", partitioning=part, exclude_invalid_files=True, ignore_prefixes=["_"])
        out["columns"] = dset.schema.names if hasattr(dset.schema, "names") else [f.name for f in dset.schema]
        total = sum(b.num_rows for b in dset.scanner(batch_size=200_000).to_batches())
        out["rows"] = total
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(output_path)
            out["rows"] = len(df)
            out["columns"] = list(df.columns)
        except Exception:
            pass
    return out


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

    id_cols: tuple = ("platform_name", "offer_id", "cik")
    if snapshots_path and str(snapshots_path).endswith(".parquet") and snapshots_path.exists():
        try:
            import pyarrow.parquet as pq
            snap_schema = pq.read_schema(snapshots_path).names
            if "platform_name" not in snap_schema or "offer_id" not in snap_schema:
                id_cols = ("cik",)
                logger.info("snapshots is cik-day only; using id_cols=(cik,)")
        except Exception:
            pass

    logger.info("build_edgar_features start")
    logger.info("edgar_path=%s", args.edgar_path)
    logger.info("snapshots_path=%s", snapshots_path)
    logger.info("output_path=%s", args.output_path)
    logger.info("align_to_snapshots=%s", args.align_to_snapshots)
    logger.info("partition_by_year=%s", args.partition_by_year)
    logger.info("cutoff_col=%s", args.cutoff_col)

    edgar_ver, edgar_files = _delta_version_and_files(args.edgar_path)
    snap_stats = _snapshots_stats(snapshots_path) if snapshots_path and str(snapshots_path).endswith(".parquet") else {"rows": 0, "unique_cik": 0, "unique_keys": ""}

    out = build_edgar_feature_store_v2(
        edgar_path=args.edgar_path,
        snapshots_path=snapshots_path,
        output_path=args.output_path,
        snapshot_time_col=args.snapshot_time_col,
        id_cols=id_cols,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
        ema_alpha=args.ema_alpha,
        align_to_snapshots=args.align_to_snapshots,
        partition_by_year=args.partition_by_year,
        cutoff_col=args.cutoff_col,
        logger=logger,
    )
    logger.info("build_edgar_features done: %s", out)

    manifest_dir = args.output_path.parent
    out_stats = _output_stats(args.output_path, args.partition_by_year)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
    manifest = {
        "raw_edgar_delta": str(args.edgar_path.resolve()),
        "raw_edgar_delta_version": edgar_ver,
        "raw_edgar_active_files": edgar_files,
        "snapshots_index_path": str(snapshots_path.resolve()) if snapshots_path else None,
        "snapshots_index_rows": snap_stats.get("rows", 0),
        "snapshots_unique_cik": snap_stats.get("unique_cik", 0),
        "snapshots_unique_keys": snap_stats.get("unique_keys", ""),
        "output_rows": out_stats.get("rows", 0),
        "output_columns": out_stats.get("columns", [])[:50],
        "partition_strategy": "snapshot_year" if args.partition_by_year else "none",
        "git_head": git_head,
        "cmd_args": {
            "align_to_snapshots": args.align_to_snapshots,
            "partition_by_year": args.partition_by_year,
            "snapshot_time_col": args.snapshot_time_col,
            "ema_alpha": args.ema_alpha,
        },
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    manifest_path = manifest_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote MANIFEST.json to %s", manifest_path)


if __name__ == "__main__":
    main()
