#!/usr/bin/env python3
"""Audit EDGAR alignment semantics relevant to V740 local-only work.

This script compares two EDGAR join policies on the same frozen Block 3 core
panel:

1. exact-day join on (cik, crawled_date_day)
2. benchmark-style as-of join on (cik, crawled_date_day) with backward lookup
   and a bounded tolerance window

The goal is not to train a model. It is to quantify whether V740's current
EDGAR-side difficulties are plausibly driven by join coverage / staleness
rather than by modeling alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.data_preprocessing.block3_dataset import Block3Dataset


TARGET_COLS = ["funding_raised_usd", "investors_count", "is_funded"]


def _to_day(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_convert(None).dt.normalize().astype("datetime64[ns]")


def _rate(num: int | float, den: int | float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _describe_numeric(series: pd.Series) -> Dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "max": None,
        }
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_scalar(v) for v in value]
    return value


def build_audit(pointer_path: Path, tolerance_days: int, top_entities: int, min_entity_rows: int) -> Dict[str, Any]:
    dataset = Block3Dataset.from_pointer(pointer_path)

    core_cols = ["entity_id", "cik", "crawled_date_day", *TARGET_COLS]
    core = dataset.get_offers_core_daily(columns=core_cols).copy()
    core["core_date"] = _to_day(core["crawled_date_day"])
    core = core[core["core_date"].notna()].reset_index(drop=True)
    core["core_row_id"] = np.arange(len(core), dtype=np.int64)

    core_with_cik = core[core["cik"].notna()].copy()
    core_with_cik = core_with_cik[["core_row_id", "entity_id", "cik", "core_date", *TARGET_COLS]]

    edgar = dataset.get_edgar_store(columns=["cik", "crawled_date_day"]).copy()
    edgar["edgar_date"] = _to_day(edgar["crawled_date_day"])
    edgar = edgar[edgar["cik"].notna() & edgar["edgar_date"].notna()][["cik", "edgar_date"]]
    edgar = edgar.drop_duplicates(["cik", "edgar_date"]).reset_index(drop=True)

    exact = core_with_cik[["core_row_id", "entity_id", "cik", "core_date", *TARGET_COLS]].merge(
        edgar.assign(exact_match=True),
        left_on=["cik", "core_date"],
        right_on=["cik", "edgar_date"],
        how="left",
    )
    exact = exact[["core_row_id", "exact_match"]]
    exact["exact_match"] = exact["exact_match"].fillna(False).astype(bool)

    asof_left = core_with_cik.sort_values(["cik", "core_date"]).reset_index(drop=True)
    right_groups = {
        cik: grp[["edgar_date"]].sort_values("edgar_date").reset_index(drop=True)
        for cik, grp in edgar.groupby("cik", sort=False)
    }
    asof_parts: List[pd.DataFrame] = []
    tolerance = pd.Timedelta(days=tolerance_days)
    for cik, left_grp in asof_left.groupby("cik", sort=False):
        left_sorted = left_grp.sort_values("core_date").reset_index(drop=True)
        right_grp = right_groups.get(cik)
        if right_grp is None or right_grp.empty:
            merged = left_sorted.copy()
            merged["edgar_date"] = pd.NaT
        else:
            merged = pd.merge_asof(
                left_sorted,
                right_grp,
                left_on="core_date",
                right_on="edgar_date",
                direction="backward",
                tolerance=tolerance,
            )
        asof_parts.append(merged)
    asof = pd.concat(asof_parts, ignore_index=True)
    asof["asof_match"] = asof["edgar_date"].notna()
    asof["lag_days"] = (asof["core_date"] - asof["edgar_date"]).dt.days
    audit_df = asof.merge(exact, on="core_row_id", how="left")
    audit_df["exact_match"] = audit_df["exact_match"].fillna(False).astype(bool)

    total_rows = len(core)
    total_rows_with_cik = len(core_with_cik)
    exact_matches = int(audit_df["exact_match"].sum())
    asof_matches = int(audit_df["asof_match"].sum())
    lag = audit_df.loc[audit_df["asof_match"], "lag_days"]

    entity_summary = (
        audit_df.groupby(["entity_id", "cik"], dropna=False)
        .agg(
            rows=("core_row_id", "size"),
            exact_rate=("exact_match", "mean"),
            asof_rate=("asof_match", "mean"),
            median_lag=("lag_days", "median"),
        )
        .reset_index()
    )
    entity_summary["coverage_gain"] = entity_summary["asof_rate"] - entity_summary["exact_rate"]
    entity_summary = entity_summary[entity_summary["rows"] >= min_entity_rows].copy()

    top_gain = entity_summary.sort_values(["coverage_gain", "rows"], ascending=[False, False]).head(top_entities)
    top_stale = entity_summary.sort_values(["median_lag", "rows"], ascending=[False, False]).head(top_entities)

    target_subset: Dict[str, Any] = {}
    for target in TARGET_COLS:
        mask = audit_df[target].notna()
        denom = int(mask.sum())
        target_subset[target] = {
            "eligible_rows": denom,
            "exact_match_rate": _rate(int((audit_df["exact_match"] & mask).sum()), denom),
            "asof_match_rate": _rate(int((audit_df["asof_match"] & mask).sum()), denom),
        }

    result = {
        "pointer_path": str(pointer_path),
        "tolerance_days": int(tolerance_days),
        "core": {
            "rows": int(total_rows),
            "rows_with_cik": int(total_rows_with_cik),
            "rows_without_cik": int(total_rows - total_rows_with_cik),
            "cik_row_rate": _rate(total_rows_with_cik, total_rows),
            "unique_entities": int(core["entity_id"].nunique()),
            "unique_ciks": int(core_with_cik["cik"].nunique()),
        },
        "edgar": {
            "rows": int(len(edgar)),
            "unique_ciks": int(edgar["cik"].nunique()),
            "core_cik_overlap_rate": _rate(
                len(set(core_with_cik["cik"].unique()) & set(edgar["cik"].unique())),
                core_with_cik["cik"].nunique(),
            ),
        },
        "join_comparison": {
            "exact_day": {
                "matched_rows": exact_matches,
                "match_rate_rows_with_cik": _rate(exact_matches, total_rows_with_cik),
            },
            "asof_backward": {
                "matched_rows": asof_matches,
                "match_rate_rows_with_cik": _rate(asof_matches, total_rows_with_cik),
                "coverage_gain_rows": int(asof_matches - exact_matches),
                "coverage_gain_rate": _rate(asof_matches - exact_matches, total_rows_with_cik),
                "lag_days": _describe_numeric(lag),
                "same_day_share_among_matches": _rate(int((lag == 0).sum()), int(lag.notna().sum())),
                "le_7_share_among_matches": _rate(int((lag <= 7).sum()), int(lag.notna().sum())),
                "le_30_share_among_matches": _rate(int((lag <= 30).sum()), int(lag.notna().sum())),
                "days_31_90_share_among_matches": _rate(int(((lag > 30) & (lag <= 90)).sum()), int(lag.notna().sum())),
            },
        },
        "target_subsets": target_subset,
        "entity_coverage": {
            "min_rows_per_entity": int(min_entity_rows),
            "eligible_entities": int(len(entity_summary)),
            "exact_rate": _describe_numeric(entity_summary["exact_rate"]),
            "asof_rate": _describe_numeric(entity_summary["asof_rate"]),
            "coverage_gain": _describe_numeric(entity_summary["coverage_gain"]),
        },
        "top_entity_gains": [
            {
                "entity_id": str(row.entity_id),
                "cik": str(row.cik),
                "rows": int(row.rows),
                "exact_rate": float(row.exact_rate),
                "asof_rate": float(row.asof_rate),
                "coverage_gain": float(row.coverage_gain),
                "median_lag": None if pd.isna(row.median_lag) else float(row.median_lag),
            }
            for row in top_gain.itertuples(index=False)
        ],
        "top_entity_staleness": [
            {
                "entity_id": str(row.entity_id),
                "cik": str(row.cik),
                "rows": int(row.rows),
                "exact_rate": float(row.exact_rate),
                "asof_rate": float(row.asof_rate),
                "coverage_gain": float(row.coverage_gain),
                "median_lag": None if pd.isna(row.median_lag) else float(row.median_lag),
            }
            for row in top_stale.itertuples(index=False)
        ],
    }
    return _normalize_scalar(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EDGAR exact-day vs as-of alignment for V740")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--tolerance-days", type=int, default=90)
    parser.add_argument("--top-entities", type=int, default=10)
    parser.add_argument("--min-entity-rows", type=int, default=5)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = build_audit(
        pointer_path=Path(args.pointer),
        tolerance_days=args.tolerance_days,
        top_entities=args.top_entities,
        min_entity_rows=args.min_entity_rows,
    )

    rendered = json.dumps(result, indent=2, sort_keys=False)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()