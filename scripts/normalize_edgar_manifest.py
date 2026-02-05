#!/usr/bin/env python
"""
Normalize edgar MANIFEST.json fields for clarity.
Converts snapshots_unique_keys from string to structured format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize edgar MANIFEST.json")
    parser.add_argument("--edgar_manifest", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    if not args.edgar_manifest.exists():
        print(f"MANIFEST not found: {args.edgar_manifest}")
        return

    manifest = json.loads(args.edgar_manifest.read_text())

    # Normalize snapshots_unique_keys if it's a string
    suk = manifest.get("snapshots_unique_keys")
    if isinstance(suk, str):
        # Parse "cik, crawled_date_day" -> structured format
        key_cols = [k.strip() for k in suk.split(",")]
        manifest["snapshots_key_cols"] = key_cols
        manifest["snapshots_unique_keys"] = {
            "description": "Unique key columns for alignment to snapshots",
            "columns": key_cols,
        }

    # Add field semantics
    manifest["field_semantics"] = manifest.get("field_semantics", {})
    manifest["field_semantics"].update({
        "snapshots_key_cols": "List of column names used as unique key for snapshot alignment",
        "output_rows": "Total rows in output edgar_features parquet",
        "raw_edgar_delta_version": "Delta table version of raw EDGAR source",
    })

    if not args.overwrite:
        print("Use --overwrite 1 to write changes")
        print(json.dumps(manifest, indent=2))
        return

    args.edgar_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {args.edgar_manifest}")


if __name__ == "__main__":
    main()
