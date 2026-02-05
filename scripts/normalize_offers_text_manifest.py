#!/usr/bin/env python
"""
Normalize offers_text MANIFEST.json fields for clarity.
Ensures n_unique_entity_day is accurate (unique entity_id, crawled_date_day pairs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize offers_text MANIFEST.json")
    parser.add_argument("--offers_text_manifest", type=Path, required=True)
    parser.add_argument("--offers_text_parquet", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    if not args.offers_text_manifest.exists():
        print(f"MANIFEST not found: {args.offers_text_manifest}")
        return

    manifest = json.loads(args.offers_text_manifest.read_text())

    # Calculate actual unique entity-day pairs from parquet
    if args.offers_text_parquet.exists():
        # Check schema first
        import pyarrow.parquet as pq
        schema = pq.read_schema(args.offers_text_parquet)
        cols = schema.names
        
        # Use snapshot_ts if crawled_date_day not available
        day_col = "crawled_date_day" if "crawled_date_day" in cols else ("snapshot_ts" if "snapshot_ts" in cols else None)
        
        if day_col:
            df = pd.read_parquet(args.offers_text_parquet, columns=["entity_id", day_col])
            if day_col == "snapshot_ts":
                df["crawled_date_day"] = pd.to_datetime(df["snapshot_ts"], errors="coerce").dt.date.astype(str)
            unique_pairs = df[["entity_id", "crawled_date_day"]].drop_duplicates().shape[0]
        else:
            df = pd.read_parquet(args.offers_text_parquet, columns=["entity_id"])
            unique_pairs = len(df)
            
        unique_entities = df["entity_id"].nunique()

        # Add clarified fields
        manifest["n_unique_entity_day_pairs"] = unique_pairs
        manifest["n_unique_entity_id"] = unique_entities
        manifest["field_semantics"] = {
            "n_unique_entity_day_pairs": "Count of unique (entity_id, day) pairs in output",
            "n_unique_entity_id": "Count of unique entity_id values in output",
            "rows_emitted": "Total rows in output parquet",
        }

    if not args.overwrite:
        print("Use --overwrite 1 to write changes")
        print(json.dumps(manifest, indent=2))
        return

    args.offers_text_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {args.offers_text_manifest}")


if __name__ == "__main__":
    main()
