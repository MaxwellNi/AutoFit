#!/usr/bin/env python
"""
Extract snapshots index (platform_name, offer_id, cik, snapshot_ts) from offers_core_daily.
For EDGAR alignment: drop null cik, drop_duplicates(cik, snapshot_ts) to avoid row explosion.
Output: snapshots_cik_day.parquet (minimal cols for edgar store input).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Make snapshots index from offers_core_daily.")
    parser.add_argument("--offers_core_parquet", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)
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
    df = df.drop_duplicates(subset=["cik", "snapshot_ts"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
