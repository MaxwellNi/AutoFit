#!/usr/bin/env python
"""
Build offers_core_full_daily from offers_core_snapshot.
Adds derived columns: delta_funding_raised, pct_change, count_snapshots_per_day.
For same granularity (entity-day), daily = snapshot + derived.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offers_core_daily from snapshot (add derived cols).")
    parser.add_argument("--snapshot_dir", type=Path, required=True, help="Dir with offers_core_snapshot.parquet")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    snap_path = args.snapshot_dir / "offers_core_snapshot.parquet"
    static_path = args.snapshot_dir / "offers_static.parquet"
    out_daily = args.output_dir / "offers_core_daily.parquet"
    out_static = args.output_dir / "offers_static.parquet"

    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found. Run build_offers_core_full_snapshot first.", file=sys.stderr)
        sys.exit(1)
    if out_daily.exists() and args.overwrite != 1:
        print(f"ERROR: {out_daily} exists and --overwrite 0.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(snap_path)
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")

    # Derived: delta_funding_raised, pct_change (vs prev day)
    if "funding_raised_usd" in df.columns:
        df["_prev_raised"] = df.groupby("entity_id")["funding_raised_usd"].shift(1)
        df["delta_funding_raised"] = pd.to_numeric(df["funding_raised_usd"], errors="coerce") - pd.to_numeric(df["_prev_raised"], errors="coerce")
        df["pct_change"] = df["delta_funding_raised"] / df["_prev_raised"].replace(0, float("nan"))
        df = df.drop(columns=["_prev_raised"])
    df["count_snapshots_per_day"] = 1

    df.to_parquet(out_daily, index=False)
    if static_path.exists():
        import shutil
        shutil.copy(static_path, out_static)
    else:
        static = df.drop_duplicates(subset=["entity_id"], keep="last")[["entity_id", "platform_name", "offer_id"] + [c for c in ["cik", "datetime_open_offering", "datetime_close_offering"] if c in df.columns]]
        static["static_snapshot_ts"] = static["snapshot_ts"]
        static.to_parquet(out_static, index=False)

    manifest_path = args.snapshot_dir / "MANIFEST.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_type"] = "daily"
    manifest["snapshot_dir"] = str(args.snapshot_dir)
    manifest["rows_emitted"] = len(df)
    manifest["built_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_daily} ({len(df):,} rows), {out_static}", flush=True)


if __name__ == "__main__":
    main()
