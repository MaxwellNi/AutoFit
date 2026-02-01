#!/usr/bin/env python
"""
Extract snapshots index (platform_name, offer_id, cik, snapshot_ts) from offers_core_daily.
For EDGAR alignment. Use --dedup_cik 0 for FULL snapshots (one row per offer snapshot with cik)
to match audit/joins; use --dedup_cik 1 to collapse by (cik, snapshot_ts) for smaller output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Make snapshots index from offers_core_daily.")
    parser.add_argument("--offers_core_parquet", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument(
        "--dedup_cik",
        type=int,
        default=0,
        help="0=full snapshots (one per offer; for Block3 audit); 1=dedup by (cik,snapshot_ts)",
    )
    args = parser.parse_args()

    core_path = args.offers_core_parquet
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    out = args.output_path or core_path.parent / "snapshots_cik_day" / "snapshots.parquet"

    cols = ["platform_name", "offer_id", "cik", "snapshot_ts"]
    df = pd.read_parquet(core_path, columns=cols)
    if "cik" not in df.columns:
        raise KeyError("offers_core must have cik")
    df["cik"] = df["cik"].astype(str)
    df = df[~df["cik"].isin(["", "nan", "None", "NaN"])]
    df = df.dropna(subset=["cik", "snapshot_ts"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    df["crawled_date_day"] = df["snapshot_ts"]
    df = df.dropna(subset=["snapshot_ts"])
    if args.dedup_cik == 1:
        df = df.drop_duplicates(subset=["cik", "snapshot_ts"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, dedup_cik={args.dedup_cik})", flush=True)


if __name__ == "__main__":
    main()
