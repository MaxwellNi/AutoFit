#!/usr/bin/env python
"""
Generate snapshots_index from offers_core_daily for EDGAR alignment.
Wrapper: calls make_snapshots_index_from_offers_core with output to snapshots_index/snapshots.parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Make snapshots index from offers_core_daily.")
    parser.add_argument("--offers_core_parquet", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None, help="Dir for snapshots_index/ (default: parent of offers_core)")
    parser.add_argument("--dedup_cik", type=int, default=0, help="0=full snapshots; 1=dedup by (cik,snapshot_ts)")
    args = parser.parse_args()

    core_path = args.offers_core_parquet
    out_dir = args.output_dir or core_path.parent
    out_path = out_dir / "snapshots_index" / "snapshots.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from scripts.make_snapshots_index_from_offers_core import main as _real_main
    orig = sys.argv
    sys.argv = ["make_snapshots_index_from_offers_core", "--offers_core_parquet", str(core_path), "--output_path", str(out_path), "--dedup_cik", str(args.dedup_cik)]
    try:
        _real_main()
    finally:
        sys.argv = orig


if __name__ == "__main__":
    main()
