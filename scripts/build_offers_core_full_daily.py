#!/usr/bin/env python
"""
Build offers_core_full_daily from offers_core_snapshot (wide).
Adds derived columns: delta_funding_raised, pct_change, count_snapshots_per_day.
For same granularity (entity-day), daily = snapshot + derived, using last_non_null.
If offers_extras_snapshot exists, build offers_extras_daily similarly.
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
    parser.add_argument("--extras_snapshot", type=Path, default=None, help="Optional offers_extras_snapshot.parquet")
    args = parser.parse_args()

    snap_path = args.snapshot_dir / "offers_core_snapshot.parquet"
    static_path = args.snapshot_dir / "offers_static.parquet"
    out_daily = args.output_dir / "offers_core_daily.parquet"
    out_static = args.output_dir / "offers_static.parquet"
    extras_snap = args.extras_snapshot or (args.snapshot_dir / "offers_extras_snapshot.parquet")
    out_extras = args.output_dir / "offers_extras_daily.parquet"

    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found. Run build_offers_core_full_snapshot first.", file=sys.stderr)
        sys.exit(1)
    if out_daily.exists() and args.overwrite != 1:
        print(f"ERROR: {out_daily} exists and --overwrite 0.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(snap_path)
    if "entity_id" not in df.columns:
        if "platform_name" in df.columns and "offer_id" in df.columns:
            df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
        else:
            print("ERROR: snapshot missing entity_id and platform_name/offer_id", file=sys.stderr)
            sys.exit(1)
    if "snapshot_ts" not in df.columns:
        if "crawled_date_day" in df.columns:
            df["snapshot_ts"] = pd.to_datetime(df["crawled_date_day"], errors="coerce", utc=True)
        elif "crawled_date" in df.columns:
            df["snapshot_ts"] = pd.to_datetime(df["crawled_date"], errors="coerce", utc=True)
        else:
            print("ERROR: snapshot missing snapshot_ts/crawled_date_day/crawled_date", file=sys.stderr)
            sys.exit(1)
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")
    if "crawled_date_day" not in df.columns:
        df["crawled_date_day"] = pd.to_datetime(df["snapshot_ts"], errors="coerce").dt.date.astype(str)
    key_cols = ["entity_id", "crawled_date_day"]
    counts = df.groupby(key_cols, sort=False).size().reset_index(name="count_snapshots_per_day")

    # Ensure last_non_null within entity-day while preserving key columns
    non_key_cols = [c for c in df.columns if c not in key_cols]
    if non_key_cols:
        df[non_key_cols] = df.groupby(key_cols, sort=False)[non_key_cols].ffill()
    df = df.groupby(key_cols, sort=False).tail(1)
    # Defragment before adding derived columns
    df = df.copy()

    # Derived: delta_funding_raised, pct_change (vs prev day)
    if "funding_raised_usd" in df.columns:
        base = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
        prev = base.groupby(df["entity_id"]).shift(1)
        delta = base - prev
        pct = delta / prev.replace(0, float("nan"))
        df = df.assign(
            delta_funding_raised=delta,
            pct_change=pct,
        )
    df = df.merge(counts, on=key_cols, how="left")

    df.to_parquet(out_daily, index=False)
    output_columns = list(df.columns)
    if static_path.exists():
        import shutil
        shutil.copy(static_path, out_static)
    else:
        static = df.drop_duplicates(subset=["entity_id"], keep="last")[["entity_id", "platform_name", "offer_id"] + [c for c in ["cik", "datetime_open_offering", "datetime_close_offering"] if c in df.columns]]
        static["static_snapshot_ts"] = static["snapshot_ts"]
        static.to_parquet(out_static, index=False)

    extras_output_columns = []
    if extras_snap.exists():
        ex = pd.read_parquet(extras_snap)
        if "entity_id" not in ex.columns:
            if "platform_name" in ex.columns and "offer_id" in ex.columns:
                ex["entity_id"] = ex["platform_name"].astype(str) + "|" + ex["offer_id"].astype(str)
            else:
                print("ERROR: extras snapshot missing entity_id and platform_name/offer_id", file=sys.stderr)
                sys.exit(1)
        if "snapshot_ts" not in ex.columns:
            if "crawled_date_day" in ex.columns:
                ex["snapshot_ts"] = pd.to_datetime(ex["crawled_date_day"], errors="coerce", utc=True)
            elif "crawled_date" in ex.columns:
                ex["snapshot_ts"] = pd.to_datetime(ex["crawled_date"], errors="coerce", utc=True)
            else:
                print("ERROR: extras snapshot missing snapshot_ts/crawled_date_day/crawled_date", file=sys.stderr)
                sys.exit(1)
        ex = ex.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")
        if "crawled_date_day" not in ex.columns:
            ex["crawled_date_day"] = pd.to_datetime(ex["snapshot_ts"], errors="coerce").dt.date.astype(str)
        ex_key_cols = ["entity_id", "crawled_date_day"]
        ex_non_key = [c for c in ex.columns if c not in ex_key_cols]
        if ex_non_key:
            ex[ex_non_key] = ex.groupby(ex_key_cols, sort=False)[ex_non_key].ffill()
        ex = ex.groupby(ex_key_cols, sort=False).tail(1)
        ex.to_parquet(out_extras, index=False)
        extras_output_columns = list(ex.columns)

    manifest_path = args.snapshot_dir / "MANIFEST.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_type"] = "daily"
    manifest["snapshot_dir"] = str(args.snapshot_dir)
    manifest["rows_emitted"] = len(df)
    manifest["output_columns"] = output_columns
    if extras_output_columns:
        manifest["extras_output_columns"] = extras_output_columns
        manifest["extras_snapshot_path"] = str(extras_snap)
        manifest["extras_daily_path"] = str(out_extras)
    manifest["grain"] = "entity-day"
    manifest["partition_strategy"] = "single_file"
    manifest["cmd_args"] = sys.argv[1:]
    manifest["built_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_daily} ({len(df):,} rows), {out_static}", flush=True)
    if extras_output_columns:
        print(f"Wrote {out_extras} ({len(ex):,} rows)", flush=True)


if __name__ == "__main__":
    main()
