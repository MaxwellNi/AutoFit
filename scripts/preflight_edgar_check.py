#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _pick_sample_file(edgar_dir: Path, seed: int = 42) -> Path:
    files = sorted(p for p in edgar_dir.rglob("*.parquet") if p.is_file())
    if not files:
        files = sorted(p for p in edgar_dir.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files found under {edgar_dir}")
    rng = np.random.RandomState(seed)
    return files[int(rng.randint(0, len(files)))]


def _numeric_stats(df: pd.DataFrame, max_cols: int = 8) -> List[Tuple[str, float, float]]:
    stats: List[Tuple[str, float, float]] = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols[:max_cols]:
        series = df[col]
        non_null_rate = float(series.notna().mean()) if len(series) else 0.0
        non_zero_rate = float((series.fillna(0) != 0).mean()) if len(series) else 0.0
        stats.append((col, non_null_rate, non_zero_rate))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight EDGAR feature store sanity check.")
    parser.add_argument("--edgar_dir", type=str, required=True)
    parser.add_argument("--sample_rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    edgar_dir = Path(args.edgar_dir)
    print(f"edgar_dir={edgar_dir}")
    if not edgar_dir.exists():
        raise SystemExit("FATAL: edgar_features_dir missing")
    file_count = len([p for p in edgar_dir.rglob("*") if p.is_file()])
    print(f"edgar_features_file_count={file_count}")
    if file_count == 0:
        raise SystemExit("FATAL: edgar_features exists but empty")

    sample_file = _pick_sample_file(edgar_dir, seed=args.seed)
    print(f"sample_file={sample_file}")
    table = pq.read_table(sample_file, memory_map=True)
    if args.sample_rows > 0 and table.num_rows > args.sample_rows:
        table = table.slice(0, args.sample_rows)
    df = table.to_pandas()
    print(f"sample_rows={len(df)}")
    print(f"sample_cols={len(df.columns)}")
    print(f"sample_col_names={list(df.columns)[:30]}")
    col_hash = hashlib.sha256(",".join(sorted(df.columns)).encode("utf-8")).hexdigest()
    print(f"col_hash={col_hash}")
    if len(df) == 0 or len(df.columns) == 0:
        raise SystemExit("FATAL: edgar_features sample has no rows or columns")

    stats = _numeric_stats(df)
    if not stats:
        raise SystemExit("FATAL: no numeric columns found in edgar_features sample")
    for col, non_null_rate, non_zero_rate in stats:
        print(f"numeric_col={col} non_null_rate={non_null_rate:.4f} non_zero_rate={non_zero_rate:.4f}")
    if all(non_null_rate == 0.0 for _, non_null_rate, _ in stats):
        raise SystemExit("FATAL: all numeric columns are entirely null in sample")
    if all(non_zero_rate == 0.0 for _, _, non_zero_rate in stats):
        raise SystemExit("FATAL: all numeric columns are zero in sample")
    if not any(
        non_null_rate > 0.01 and non_zero_rate > 0.01 for _, non_null_rate, non_zero_rate in stats
    ):
        raise SystemExit("FATAL: numeric columns fail non-null/non-zero >1% threshold")


if __name__ == "__main__":
    main()
