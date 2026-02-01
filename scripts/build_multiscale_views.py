#!/usr/bin/env python
"""
Build multi-scale derived views from offers_core_daily + edgar_features.
Output: offers_weekly, offers_monthly, edgar_weekly, edgar_monthly, stage_table, day_to_stage.
Stage/round: heuristic from open/close dates, status changes, funding_raised change-points.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys_path_inserted = False


def _ensure_path() -> None:
    global sys_path_inserted
    if not sys_path_inserted:
        import sys
        sys.path.insert(0, str(repo_root))
        sys_path_inserted = True


def _load_offers_daily(path: Path) -> pd.DataFrame:
    _ensure_path()
    df = pd.read_parquet(path)
    if "snapshot_ts" not in df.columns and "crawled_date_day" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["crawled_date_day"], errors="coerce", utc=True)
    elif "snapshot_ts" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    return df


def _load_edgar(path: Path) -> pd.DataFrame:
    _ensure_path()
    import pyarrow.dataset as ds
    dset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dset.to_table().to_pandas()


def _aggregate_weekly(df: pd.DataFrame, key_cols: List[str], time_col: str = "snapshot_ts") -> pd.DataFrame:
    df = df.copy()
    df["_week"] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.to_period("W")
    agg_cols = [c for c in ["funding_raised_usd", "funding_goal_usd", "investors_count"] if c in df.columns]
    if not agg_cols:
        return df.groupby(key_cols + ["_week"]).agg("last").reset_index()
    grp = df.groupby(key_cols + ["_week"])
    out = grp[agg_cols].last().reset_index()
    for c in key_cols + ["_week"]:
        if c not in out.columns and c in df.columns:
            out[c] = grp[c].first().values
    out["snapshot_ts"] = pd.to_datetime(out["_week"].astype(str) + "-1", errors="coerce", utc=True)
    out = out.drop(columns=["_week"], errors="ignore")
    return out


def _aggregate_monthly(df: pd.DataFrame, key_cols: List[str], time_col: str = "snapshot_ts") -> pd.DataFrame:
    df = df.copy()
    df["_month"] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.to_period("M")
    agg_cols = [c for c in ["funding_raised_usd", "funding_goal_usd", "investors_count"] if c in df.columns]
    if not agg_cols:
        return df.groupby(key_cols + ["_month"]).agg("last").reset_index()
    grp = df.groupby(key_cols + ["_month"])
    out = grp[agg_cols].last().reset_index()
    out["snapshot_ts"] = pd.to_datetime(out["_month"].astype(str) + "-01", errors="coerce", utc=True)
    out = out.drop(columns=["_month"], errors="ignore")
    return out


def _build_stage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic stage/round from open/close dates, status, funding_raised change-points."""
    key_cols = ["entity_id", "platform_name", "offer_id"]
    key_cols = [c for c in key_cols if c in df.columns]
    time_col = "snapshot_ts" if "snapshot_ts" in df.columns else "crawled_date_day"
    df = df.sort_values(key_cols + [time_col], kind="mergesort")
    df = df.copy()
    df["_day"] = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.date
    stages = []
    for (eid,), g in df.groupby(["entity_id"] if "entity_id" in df.columns else key_cols[:1]):
        g = g.sort_values(time_col).copy()
        if "funding_raised_usd" in g.columns:
            g["_delta"] = pd.to_numeric(g["funding_raised_usd"], errors="coerce").diff()
            g["_cp"] = (g["_delta"].abs() > 0) & g["_delta"].notna()
        else:
            g["_cp"] = False
        g["stage_idx"] = g["_cp"].cumsum().fillna(0)
        stages.append(g)
    out = pd.concat(stages, ignore_index=True) if stages else pd.DataFrame()
    if not out.empty:
        out = out[key_cols + [time_col, "_day", "stage_idx"]].drop_duplicates()
        out = out.rename(columns={"stage_idx": "stage", "_day": "snapshot_day"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-scale derived views.")
    parser.add_argument("--offers_core_daily", type=Path, required=True)
    parser.add_argument("--edgar_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stamp", type=str, default=None)
    args = parser.parse_args()

    _ensure_path()
    import sys
    sys.path.insert(0, str(repo_root))

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"multiscale_full_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_cols = ["entity_id", "platform_name", "offer_id"]
    time_col = "snapshot_ts"

    offers = _load_offers_daily(args.offers_core_daily)
    key_cols_offers = [c for c in key_cols if c in offers.columns]
    if offers.empty:
        print("WARN: offers_core_daily empty", flush=True)
    else:
        offers_weekly = _aggregate_weekly(offers, key_cols_offers, time_col)
        offers_monthly = _aggregate_monthly(offers, key_cols_offers, time_col)
        offers_weekly.to_parquet(out_dir / "offers_weekly.parquet", index=False)
        offers_monthly.to_parquet(out_dir / "offers_monthly.parquet", index=False)
        print(f"Wrote offers_weekly.parquet ({len(offers_weekly):,}), offers_monthly.parquet ({len(offers_monthly):,})", flush=True)

        stage_df = _build_stage_table(offers)
        if not stage_df.empty:
            stage_df.to_parquet(out_dir / "stage_table.parquet", index=False)
            day_to_stage = stage_df[["entity_id", "snapshot_day", "stage"]].drop_duplicates()
            day_to_stage.to_parquet(out_dir / "day_to_stage.parquet", index=False)
            print(f"Wrote stage_table.parquet, day_to_stage.parquet", flush=True)

    try:
        edgar = _load_edgar(args.edgar_dir)
        if not edgar.empty and "cik" in edgar.columns:
            snap_col = "crawled_date_day" if "crawled_date_day" in edgar.columns else "snapshot_year"
            if snap_col not in edgar.columns:
                snap_col = [c for c in edgar.columns if "date" in c.lower() or "time" in c.lower()][0] if edgar.columns.any() else None
            if snap_col:
                edgar["snapshot_ts"] = pd.to_datetime(edgar[snap_col], errors="coerce", utc=True)
                ec = ["cik", "snapshot_ts"] + [c for c in edgar.columns if c not in ("cik", snap_col)][:20]
                edgar_sub = edgar[[c for c in ec if c in edgar.columns]]
                edgar_weekly = _aggregate_weekly(edgar_sub, ["cik"], "snapshot_ts")
                edgar_monthly = _aggregate_monthly(edgar_sub, ["cik"], "snapshot_ts")
                edgar_weekly.to_parquet(out_dir / "edgar_weekly.parquet", index=False)
                edgar_monthly.to_parquet(out_dir / "edgar_monthly.parquet", index=False)
                print(f"Wrote edgar_weekly.parquet, edgar_monthly.parquet", flush=True)
    except Exception as e:
        print(f"WARN: edgar load failed: {e}", flush=True)

    manifest = {
        "stamp": stamp,
        "offers_core_daily": str(args.offers_core_daily),
        "edgar_dir": str(args.edgar_dir),
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote MANIFEST.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
