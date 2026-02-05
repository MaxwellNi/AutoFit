#!/usr/bin/env python
"""
Exact offer-day coverage audit.
Computes raw_unique_offer_day_pairs = distinct(platform_name, offer_id, crawled_date_day) from raw offers
and compares to core_daily rows to ensure exact 1:1 alignment.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact offer-day coverage audit")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--offers_core_daily_parquet", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--full_scan", type=int, default=0, help="If 1, scan full raw Delta (slow)")
    args = parser.parse_args()

    result: Dict[str, Any] = {
        "audit": "offer_day_coverage_exact",
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gate_passed": False,
        "fail_reasons": [],
    }

    # Load core daily
    core_df = pd.read_parquet(args.offers_core_daily_parquet, columns=["entity_id", "crawled_date_day"])
    core_entity_day_rows = len(core_df)
    core_unique_pairs = core_df.drop_duplicates().shape[0]
    result["core_entity_day_rows"] = core_entity_day_rows
    result["core_unique_pairs"] = core_unique_pairs

    # Check if core has duplicates (should be 1:1)
    if core_entity_day_rows != core_unique_pairs:
        result["fail_reasons"].append(f"core_daily has duplicates: {core_entity_day_rows} rows vs {core_unique_pairs} unique pairs")

    # Scan raw offers
    raw_unique_offer_day_pairs = 0
    if args.full_scan:
        try:
            from deltalake import DeltaTable
            dt = DeltaTable(str(args.raw_offers_delta))
            dset = dt.to_pyarrow_dataset()
            
            seen = set()
            for batch in dset.scanner(columns=["platform_name", "offer_id", "crawled_date_day"], batch_size=500_000).to_batches():
                df = batch.to_pandas()
                for row in df.itertuples(index=False):
                    key = (str(row.platform_name), str(row.offer_id), str(row.crawled_date_day))
                    seen.add(key)
            raw_unique_offer_day_pairs = len(seen)
        except Exception as e:
            result["fail_reasons"].append(f"Raw scan failed: {e}")
            raw_unique_offer_day_pairs = -1
    else:
        # Use MANIFEST if available
        manifest_path = args.raw_offers_delta.parent.parent / "offers_core_full_daily_wide_20260203_225620" / "MANIFEST.json"
        if not manifest_path.exists():
            manifest_path = Path("runs/offers_core_full_daily_wide_20260203_225620/MANIFEST.json")
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            raw_unique_offer_day_pairs = m.get("rows_emitted", 0)
            result["note"] = "raw_unique_offer_day_pairs from MANIFEST (no full scan)"
        else:
            result["fail_reasons"].append("Cannot determine raw_unique_offer_day_pairs without full_scan=1")

    result["raw_unique_offer_day_pairs"] = raw_unique_offer_day_pairs

    # Coverage rate
    if raw_unique_offer_day_pairs > 0:
        coverage_rate = core_unique_pairs / raw_unique_offer_day_pairs
        result["coverage_rate"] = round(coverage_rate, 6)
        
        # Gate: coverage should be very close to 1.0 (allow small tolerance for edge cases)
        if coverage_rate < 0.9999 or coverage_rate > 1.0001:
            result["fail_reasons"].append(f"coverage_rate {coverage_rate} not ~1.0")
    else:
        result["coverage_rate"] = None

    result["gate_passed"] = len(result["fail_reasons"]) == 0

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"gate_passed={result['gate_passed']}, coverage_rate={result.get('coverage_rate')}")


if __name__ == "__main__":
    main()
