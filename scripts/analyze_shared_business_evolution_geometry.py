#!/usr/bin/env python3
"""Analyze shared business evolution geometry for Block 3 targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.data_preprocessing.block3_dataset import FreezePointer


TARGET_COLUMNS = [
    "entity_id",
    "crawled_date_day",
    "funding_raised_usd",
    "funding_goal_usd",
    "investors_count",
    "is_funded",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pointer-path",
        type=Path,
        default=Path("docs/audits/FULL_SCALE_POINTER.yaml"),
        help="Freeze pointer YAML.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the full JSON report.",
    )
    return ap.parse_args()


def _safe_rate(values: pd.Series) -> float | None:
    if len(values) <= 0:
        return None
    return float(values.mean())


def _safe_quantile(values: pd.Series, q: float) -> float | None:
    clean = values.dropna()
    if len(clean) <= 0:
        return None
    return float(clean.quantile(q))


def _safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 2:
        return None
    return float(np.corrcoef(left.loc[mask], right.loc[mask])[0, 1])


def _load_panel(pointer_path: Path) -> pd.DataFrame:
    pointer = FreezePointer.load(pointer_path)
    parquet_path = pointer.offers_core_daily_dir / "offers_core_daily.parquet"
    table = pq.read_table(parquet_path, columns=TARGET_COLUMNS)
    frame = table.to_pandas(strings_to_categorical=True)
    frame["crawled_date_day"] = pd.to_datetime(frame["crawled_date_day"], errors="coerce")
    frame = frame.dropna(subset=["entity_id", "crawled_date_day"]).copy()
    for column in ["funding_raised_usd", "funding_goal_usd", "investors_count", "is_funded"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["entity_id", "crawled_date_day"], kind="mergesort")
    return frame


def _final_state_relations(final_rows: pd.DataFrame) -> Dict[str, Any]:
    final = final_rows.copy()
    final["funding_pos"] = final["funding_raised_usd"].fillna(0.0) > 0.0
    final["investors_pos"] = final["investors_count"].fillna(0.0) > 0.0
    final["funded_pos"] = final["is_funded"].fillna(0.0) > 0.5
    both_positive = final["funding_pos"] & final["investors_pos"]
    return {
        "shares": {
            "funded_pos": _safe_rate(final["funded_pos"]),
            "funding_pos": _safe_rate(final["funding_pos"]),
            "investors_pos": _safe_rate(final["investors_pos"]),
        },
        "conditionals": {
            "p_funding_pos_given_funded": _safe_rate(final.loc[final["funded_pos"], "funding_pos"]),
            "p_investors_pos_given_funded": _safe_rate(final.loc[final["funded_pos"], "investors_pos"]),
            "p_funded_given_funding_pos": _safe_rate(final.loc[final["funding_pos"], "funded_pos"]),
            "p_funded_given_investors_pos": _safe_rate(final.loc[final["investors_pos"], "funded_pos"]),
            "p_investors_pos_given_funding_pos": _safe_rate(final.loc[final["funding_pos"], "investors_pos"]),
            "p_funding_pos_given_investors_pos": _safe_rate(final.loc[final["investors_pos"], "funding_pos"]),
        },
        "scale_correlation": {
            "funding_log_vs_investors_log_all": _safe_corr(
                np.log1p(final["funding_raised_usd"].fillna(0.0).clip(lower=0.0)),
                np.log1p(final["investors_count"].fillna(0.0).clip(lower=0.0)),
            ),
            "funding_log_vs_investors_log_positive_both": _safe_corr(
                np.log1p(final.loc[both_positive, "funding_raised_usd"].fillna(0.0).clip(lower=0.0)),
                np.log1p(final.loc[both_positive, "investors_count"].fillna(0.0).clip(lower=0.0)),
            ),
        },
    }


def _goal_alignment(final_rows: pd.DataFrame) -> Dict[str, Any]:
    valid = final_rows[final_rows["funding_goal_usd"].fillna(0.0) > 0.0].copy()
    if valid.empty:
        return {
            "rows": 0,
            "goal_met_share": None,
            "funded_share": None,
            "conditionals": {},
            "goal_ratio": {},
        }
    valid["funded_pos"] = valid["is_funded"].fillna(0.0) > 0.5
    valid["goal_ratio"] = (
        valid["funding_raised_usd"].fillna(0.0) / valid["funding_goal_usd"]
    ).replace([np.inf, -np.inf], np.nan)
    valid["goal_met_50"] = valid["goal_ratio"] >= 0.5
    valid["goal_met_100"] = valid["goal_ratio"] >= 1.0
    valid["goal_met_150"] = valid["goal_ratio"] >= 1.5
    return {
        "rows": int(len(valid)),
        "goal_met_share": {
            "ratio_ge_0_5": _safe_rate(valid["goal_met_50"]),
            "ratio_ge_1_0": _safe_rate(valid["goal_met_100"]),
            "ratio_ge_1_5": _safe_rate(valid["goal_met_150"]),
        },
        "funded_share": _safe_rate(valid["funded_pos"]),
        "conditionals": {
            "p_funded_given_ratio_ge_0_5": _safe_rate(valid.loc[valid["goal_met_50"], "funded_pos"]),
            "p_funded_given_ratio_ge_1_0": _safe_rate(valid.loc[valid["goal_met_100"], "funded_pos"]),
            "p_funded_given_ratio_ge_1_5": _safe_rate(valid.loc[valid["goal_met_150"], "funded_pos"]),
            "p_ratio_ge_0_5_given_funded": _safe_rate(valid.loc[valid["funded_pos"], "goal_met_50"]),
            "p_ratio_ge_1_0_given_funded": _safe_rate(valid.loc[valid["funded_pos"], "goal_met_100"]),
            "p_ratio_ge_1_5_given_funded": _safe_rate(valid.loc[valid["funded_pos"], "goal_met_150"]),
        },
        "goal_ratio": {
            "median_all": _safe_quantile(valid["goal_ratio"], 0.5),
            "median_funded": _safe_quantile(valid.loc[valid["funded_pos"], "goal_ratio"], 0.5),
            "median_unfunded": _safe_quantile(valid.loc[~valid["funded_pos"], "goal_ratio"], 0.5),
            "p90_funded": _safe_quantile(valid.loc[valid["funded_pos"], "goal_ratio"], 0.9),
            "p90_unfunded": _safe_quantile(valid.loc[~valid["funded_pos"], "goal_ratio"], 0.9),
        },
    }


def _process_geometry(panel: pd.DataFrame) -> Dict[str, Any]:
    work = panel[["entity_id", "funding_raised_usd", "investors_count", "is_funded"]].copy()
    work["funding_diff"] = work.groupby("entity_id", sort=False)["funding_raised_usd"].diff()
    work["investors_diff"] = work.groupby("entity_id", sort=False)["investors_count"].diff()
    work["funded_prev"] = work.groupby("entity_id", sort=False)["is_funded"].shift(1)
    work["funded_up"] = ((work["funded_prev"] <= 0.5) & (work["is_funded"] > 0.5)).fillna(False)
    work["funded_down"] = ((work["funded_prev"] > 0.5) & (work["is_funded"] <= 0.5)).fillna(False)

    per_entity = pd.DataFrame(
        {
            "rows": work.groupby("entity_id", sort=False).size(),
            "funding_nonmonotone": work.groupby("entity_id", sort=False)["funding_diff"].apply(
                lambda values: bool((values.dropna() < 0.0).any())
            ),
            "investors_nonmonotone": work.groupby("entity_id", sort=False)["investors_diff"].apply(
                lambda values: bool((values.dropna() < 0.0).any())
            ),
            "funded_up_events": work.groupby("entity_id", sort=False)["funded_up"].sum(),
            "funded_down_events": work.groupby("entity_id", sort=False)["funded_down"].sum(),
        }
    )
    per_entity["funding_monotone"] = ~per_entity["funding_nonmonotone"]
    per_entity["investors_monotone"] = ~per_entity["investors_nonmonotone"]
    per_entity["funded_absorbing"] = per_entity["funded_down_events"] == 0

    valid_delta = work.dropna(subset=["funding_diff", "investors_diff"]).copy()
    valid_delta["funding_jump"] = valid_delta["funding_diff"] > 0.0
    valid_delta["investor_jump"] = valid_delta["investors_diff"] > 0.0

    return {
        "monotonicity": {
            "funding_monotone_share": _safe_rate(per_entity["funding_monotone"]),
            "investors_monotone_share": _safe_rate(per_entity["investors_monotone"]),
            "funded_absorbing_share": _safe_rate(per_entity["funded_absorbing"]),
            "entities_with_funded_up": int((per_entity["funded_up_events"] > 0).sum()),
            "entities_with_funded_down": int((per_entity["funded_down_events"] > 0).sum()),
        },
        "increment_coupling": {
            "rows_with_valid_lag": int(len(valid_delta)),
            "share_funding_jump": _safe_rate(valid_delta["funding_jump"]),
            "share_investor_jump": _safe_rate(valid_delta["investor_jump"]),
            "share_both_jump": _safe_rate(valid_delta["funding_jump"] & valid_delta["investor_jump"]),
            "p_investor_jump_given_funding_jump": _safe_rate(
                valid_delta.loc[valid_delta["funding_jump"], "investor_jump"]
            ),
            "p_funding_jump_given_investor_jump": _safe_rate(
                valid_delta.loc[valid_delta["investor_jump"], "funding_jump"]
            ),
            "contingency": {
                "neither": int((~valid_delta["funding_jump"] & ~valid_delta["investor_jump"]).sum()),
                "funding_only": int((valid_delta["funding_jump"] & ~valid_delta["investor_jump"]).sum()),
                "investor_only": int((~valid_delta["funding_jump"] & valid_delta["investor_jump"]).sum()),
                "both": int((valid_delta["funding_jump"] & valid_delta["investor_jump"]).sum()),
            },
        },
    }


def build_report(pointer_path: Path) -> Dict[str, Any]:
    panel = _load_panel(pointer_path)
    final_rows = panel.groupby("entity_id", sort=False).tail(1).copy()
    return {
        "panel": {
            "rows": int(len(panel)),
            "entities": int(panel["entity_id"].nunique()),
            "mean_rows_per_entity": float(panel.groupby("entity_id", sort=False).size().mean()),
        },
        "final_state_relations": _final_state_relations(final_rows),
        "goal_alignment": _goal_alignment(final_rows),
        "process_geometry": _process_geometry(panel),
    }


def main() -> None:
    args = _parse_args()
    report = build_report(args.pointer_path)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()