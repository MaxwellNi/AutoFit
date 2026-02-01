#!/usr/bin/env python
"""Update FULL_SCALE_POINTER.yaml to point to wide freeze artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Update FULL_SCALE_POINTER to wide artifacts.")
    parser.add_argument("--wide_stamp", type=str, required=True)
    args = parser.parse_args()

    stamp = args.wide_stamp
    wide_snapshot = f"runs/offers_core_full_snapshot_wide_{stamp}"
    wide_daily = f"runs/offers_core_full_daily_wide_{stamp}"
    wide_edgar = f"runs/edgar_feature_store_full_daily_wide_{stamp}"
    wide_multiscale = f"runs/multiscale_full_wide_{stamp}"
    analysis_base = "runs/orchestrator/20260129_073037/analysis"

    pointer = {
        "variant": "TRAIN_WIDE_FINAL",
        "stamp": stamp,
        "offers_text": {"dir": "runs/offers_text_v1_20260129_073037_full"},
        "offers_core_snapshot": {"dir": wide_snapshot},
        "offers_core_daily": {"dir": wide_daily},
        "snapshots_index": {
            "offer_day": f"{wide_daily}/snapshots_index/snapshots_offer_day.parquet",
            "cik_day": f"{wide_daily}/snapshots_index/snapshots_cik_day.parquet",
        },
        "edgar_store_full_daily": {"dir": wide_edgar},
        "multiscale_full": {"dir": wide_multiscale},
        "analysis": {
            "dir": f"{analysis_base}/wide_{stamp}",
            "freeze_candidates": f"{analysis_base}/wide_{stamp}/freeze_candidates.json",
            "column_manifest": f"{analysis_base}/wide_{stamp}/column_manifest_wide.json",
            "raw_cardinality_coverage": f"{analysis_base}/wide_{stamp}/raw_cardinality_coverage_wide_{stamp}.json",
        },
    }

    path = repo_root / "docs/audits/FULL_SCALE_POINTER.yaml"
    path.write_text(yaml.dump(pointer, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
