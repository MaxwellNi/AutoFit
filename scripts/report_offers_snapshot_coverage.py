from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.schema_profiler import profile_parquet_dataset


def _dataset_for_path(path: Path) -> ds.Dataset:
    return ds.dataset(
        str(path),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )


def _scan_basic_stats(
    dataset: ds.Dataset,
    *,
    columns: List[str],
    batch_size: int = 200_000,
    limit_rows: Optional[int] = None,
) -> Dict[str, object]:
    totals = {c: 0 for c in columns}
    nulls = {c: 0 for c in columns}
    seen = 0
    min_time = None
    max_time = None

    scanner = dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        if limit_rows is not None:
            remaining = limit_rows - seen
            if remaining <= 0:
                break
            df = df.head(remaining)

        seen += len(df)
        for col in columns:
            if col not in df.columns:
                continue
            totals[col] += len(df)
            nulls[col] += int(df[col].isna().sum())

        if "crawled_date" in df.columns:
            ts = pd.to_datetime(df["crawled_date"], errors="coerce", utc=True)
            if ts.notna().any():
                cur_min = ts.min()
                cur_max = ts.max()
                min_time = cur_min if min_time is None else min(min_time, cur_min)
                max_time = cur_max if max_time is None else max(max_time, cur_max)

        if limit_rows is not None and seen >= limit_rows:
            break

    null_rate = {
        col: (nulls[col] / max(1, totals[col])) if totals[col] else 0.0 for col in columns
    }

    return {
        "rows_scanned": int(seen),
        "null_rate": null_rate,
        "time_min": min_time.isoformat() if min_time is not None else None,
        "time_max": max_time.isoformat() if max_time is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offers snapshots coverage report")
    parser.add_argument("--offers_path", type=Path, default=Path("data/raw/offers"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=200_000)
    args = parser.parse_args()

    out_dir = args.output_dir or Path("runs/offers_snapshot_report") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    dataset = _dataset_for_path(args.offers_path)
    total_rows = dataset.count_rows()
    total_rows = int(total_rows)
    if args.limit_rows is not None:
        total_rows = min(total_rows, int(args.limit_rows))

    profile = profile_parquet_dataset(args.offers_path, name="offers_snapshots")
    schema_path = out_dir / "schema_profile.json"
    schema_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    columns = [
        c for c in ["platform_name", "offer_id", "cik", "crawled_date", "crawled_date_day"]
        if c in dataset.schema.names
    ]
    stats = _scan_basic_stats(
        dataset,
        columns=columns,
        batch_size=args.batch_size,
        limit_rows=args.limit_rows,
    )

    report = {
        "offers_path": str(args.offers_path),
        "total_rows": int(total_rows),
        "columns_scanned": columns,
        "stats": stats,
        "time_columns": profile.time_columns,
        "key_candidates": profile.key_candidates,
    }

    report_path = out_dir / "offers_snapshot_coverage.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    frame = pd.DataFrame(
        [
            {
                "offers_path": report["offers_path"],
                "total_rows": report["total_rows"],
                "rows_scanned": stats["rows_scanned"],
                "time_min": stats["time_min"],
                "time_max": stats["time_max"],
                "null_rate_platform_name": stats["null_rate"].get("platform_name"),
                "null_rate_offer_id": stats["null_rate"].get("offer_id"),
                "null_rate_cik": stats["null_rate"].get("cik"),
                "null_rate_crawled_date": stats["null_rate"].get("crawled_date"),
            }
        ]
    )
    frame.to_parquet(out_dir / "offers_snapshot_coverage.parquet", index=False)

    log_path = out_dir / "logs" / "report.log"
    log_path.write_text(
        "\n".join(
            [
                f"offers_path: {report['offers_path']}",
                f"total_rows: {report['total_rows']}",
                f"rows_scanned: {stats['rows_scanned']}",
                f"time_min: {stats['time_min']}",
                f"time_max: {stats['time_max']}",
                f"columns_scanned: {columns}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Schema profile: {schema_path}")
    print(f"Coverage report: {report_path}")
    print(f"Coverage parquet: {out_dir / 'offers_snapshot_coverage.parquet'}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
