#!/usr/bin/env python
"""
Generate snapshots index from offers_core_daily for EDGAR alignment.
Outputs two versions:
  - snapshots_offer_day.parquet: full (entity_id, platform_name, offer_id, cik, snapshot_day)
  - snapshots_cik_day.parquet: dedup by (cik, snapshot_ts) for EDGAR aggregation only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Make snapshots index (offer_day + cik_day) from offers_core_daily.")
    parser.add_argument("--offers_core_daily", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    core_path = args.offers_core_daily
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    df = pd.read_parquet(core_path)
    cols = ["entity_id", "platform_name", "offer_id", "cik", "snapshot_ts"]
    read_cols = [c for c in cols if c in df.columns]
    df = df[read_cols].copy()
    if "cik" not in df.columns:
        raise KeyError("offers_core_daily must have cik")
    df["cik"] = df["cik"].astype(str)
    df = df[~df["cik"].isin(["", "nan", "None", "NaN"])]
    df = df.dropna(subset=["cik", "snapshot_ts"])
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    df["snapshot_day"] = df["snapshot_ts"].dt.date
    df["crawled_date_day"] = df["snapshot_ts"]
    df = df.dropna(subset=["snapshot_ts"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # snapshots_offer_day: full (no dedup)
    out_cols = [c for c in ["entity_id", "platform_name", "offer_id", "cik", "snapshot_day", "snapshot_ts", "crawled_date_day"] if c in df.columns]
    offer_day = df[out_cols].drop_duplicates().reset_index(drop=True)
    offer_day.to_parquet(args.output_dir / "snapshots_offer_day.parquet", index=False)
    print(f"Wrote {args.output_dir / 'snapshots_offer_day.parquet'} ({len(offer_day):,} rows)", flush=True)

    # snapshots_cik_day: dedup by (cik, snapshot_ts) for EDGAR
    cik_day = df[["cik", "snapshot_ts", "crawled_date_day"]].drop_duplicates().reset_index(drop=True)
    cik_day.to_parquet(args.output_dir / "snapshots_cik_day.parquet", index=False)
    print(f"Wrote {args.output_dir / 'snapshots_cik_day.parquet'} ({len(cik_day):,} rows)", flush=True)


if __name__ == "__main__":
    main()
