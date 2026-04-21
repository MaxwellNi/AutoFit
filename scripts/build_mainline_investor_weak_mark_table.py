#!/usr/bin/env python3
"""Build a first weakly supervised marked-investor table from the freeze surface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.mainline_investor_mark_common import (
    BASE_INVESTOR_COLUMNS,
    MARK_PROXY_COLUMNS,
    available_panel_columns,
    load_investor_mark_panel,
    load_pointer,
    output_dir,
)
from src.narrative.block3.models.single_model_mainline.investor_mark_encoder import InvestorMarkEncoder


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pointer-path", type=Path, default=None)
    ap.add_argument("--entity-limit", type=int, default=512)
    ap.add_argument("--max-rows-per-entity", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--institutional-threshold", type=float, default=0.30)
    ap.add_argument("--retail-threshold", type=float, default=0.40)
    ap.add_argument("--lead-threshold", type=float, default=0.05)
    ap.add_argument("--international-threshold", type=float, default=0.01)
    ap.add_argument("--high-threshold-usd", type=float, default=10000.0)
    ap.add_argument("--concentrated-capital-threshold", type=float, default=0.55)
    ap.add_argument(
        "--output-table",
        type=Path,
        default=output_dir() / "investor_weak_mark_table.parquet",
    )
    ap.add_argument(
        "--output-summary-json",
        type=Path,
        default=output_dir() / "investor_weak_mark_summary.json",
    )
    ap.add_argument("--output-sample-csv", type=Path, default=None)
    return ap.parse_args()


def _build_primary_label(table: pd.DataFrame) -> pd.Series:
    labels = pd.Series("untyped", index=table.index, dtype="object")
    institutional_preferred = table["weak_institutional_like_flag"] & (
        table["mark_institutional_like_score"] >= table["mark_retail_like_score"]
    )
    retail_preferred = table["weak_retail_like_flag"] & ~institutional_preferred
    labels = labels.where(~retail_preferred, "retail_like")
    labels = labels.where(~institutional_preferred, "institutional_like")
    labels = labels.where(~table["weak_repeat_networked_flag"] | labels.ne("untyped"), "repeat_networked")
    labels = labels.where(~table["weak_high_threshold_offer_flag"] | labels.ne("untyped"), "high_threshold_offer")
    labels = labels.where(~table["weak_lead_like_flag"], "lead_like")
    return labels


def _example_rows(table: pd.DataFrame, flag_column: str, top_k: int = 10) -> list[dict[str, Any]]:
    flagged = table[table[flag_column]].copy()
    if flagged.empty:
        return []
    score_cols = [
        "mark_large_investor_event_score",
        "mark_institutional_like_score",
        "mark_retail_like_score",
        "mark_concentrated_capital_score",
    ]
    present_scores = [column for column in score_cols if column in flagged.columns]
    if present_scores:
        flagged = flagged.sort_values(present_scores, ascending=False)
    keep = [
        column
        for column in (
            "entity_id",
            "crawled_date_day",
            "investors__json",
            "investor_website",
            "investment_type",
            flag_column,
            "weak_primary_label",
            "mark_institutional_like_score",
            "mark_retail_like_score",
            "mark_large_investor_event_score",
            "mark_concentrated_capital_score",
            "mark_non_national_share",
        )
        if column in flagged.columns
    ]
    return flagged[keep].head(top_k).to_dict(orient="records")


def _value_counts_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def main() -> int:
    args = _parse_args()
    pointer = load_pointer(args.pointer_path)
    available = set(available_panel_columns(pointer))
    requested_columns = tuple(dict.fromkeys(BASE_INVESTOR_COLUMNS + MARK_PROXY_COLUMNS))
    panel = load_investor_mark_panel(
        args.pointer_path,
        requested_columns=requested_columns,
        entity_limit=args.entity_limit,
        max_rows_per_entity=args.max_rows_per_entity,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        selection_mode="mark_rich",
    )

    encoder = InvestorMarkEncoder()
    marks = encoder.build_mark_frame(panel)
    table = pd.concat([panel.reset_index(drop=True), marks.reset_index(drop=True)], axis=1)
    proxy_positive_mask = pd.Series(False, index=table.index)
    for column in MARK_PROXY_COLUMNS:
        if column in table.columns:
            proxy_positive_mask |= pd.to_numeric(table[column], errors="coerce").fillna(0.0) > 0.0
    raw_reference_mask = (
        pd.to_numeric(table.get("mark_list_present", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0) > 0.0
    ) | (
        pd.to_numeric(table.get("mark_website_present", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0) > 0.0
    ) | (
        pd.to_numeric(table.get("mark_hash_present", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0) > 0.0
    )
    surface_mode = pd.Series("surface_poor", index=table.index, dtype="object")
    surface_mode = surface_mode.where(~(proxy_positive_mask & ~raw_reference_mask), "proxy_only")
    surface_mode = surface_mode.where(~(raw_reference_mask & ~proxy_positive_mask), "raw_reference_only")
    surface_mode = surface_mode.where(~(raw_reference_mask & proxy_positive_mask), "hybrid")
    table["weak_surface_mode"] = surface_mode
    investment_type = table.get("investment_type", pd.Series("", index=table.index, dtype="object")).fillna("").astype(str)
    crowdfunding_type = investment_type.str.lower().str.contains("crowd")

    high_threshold_log = float(np.log1p(max(args.high_threshold_usd, 0.0)))
    institutional_proxy_threshold = max(float(args.institutional_threshold), 0.55)
    institutional_proxy_min_investment_usd = max(float(args.high_threshold_usd) / 10.0, 1000.0)
    institutional_proxy_min_investment_log = float(np.log1p(institutional_proxy_min_investment_usd))
    institutional_keyword_evidence = (
        (table["mark_institutional_keyword_score"] >= float(args.institutional_threshold))
        & (table["mark_institutional_like_score"] >= table["mark_retail_like_score"])
    )
    institutional_proxy_evidence = (
        proxy_positive_mask
        & (table["mark_large_investor_event_score"] >= institutional_proxy_threshold)
        & (table["mark_concentrated_capital_score"] >= float(args.concentrated_capital_threshold))
        & (table["mark_non_accredited_share"] <= 0.20)
        & (table["mark_minimum_investment_log"] >= institutional_proxy_min_investment_log)
        & ~crowdfunding_type
    )
    table["weak_institutional_like_flag"] = institutional_keyword_evidence | institutional_proxy_evidence
    table["weak_retail_like_flag"] = (
        (table["mark_retail_like_score"] >= float(args.retail_threshold))
        | crowdfunding_type
    )
    table["weak_lead_like_flag"] = table["mark_lead_keyword_score"] >= float(args.lead_threshold)
    table["weak_repeat_networked_flag"] = (
        (table["mark_repeat_list_flag"] >= 0.5)
        | (table["mark_syndicate_size_score"] >= 0.5)
    )
    table["weak_international_like_flag"] = table["mark_non_national_share"] >= float(args.international_threshold)
    table["weak_high_threshold_offer_flag"] = (
        (table["mark_minimum_investment_log"] >= high_threshold_log)
        | (table["mark_concentrated_capital_score"] >= float(args.concentrated_capital_threshold))
    )
    table["weak_primary_label"] = _build_primary_label(table)

    label_counts = table["weak_primary_label"].value_counts(dropna=False).to_dict()
    summary: Dict[str, Any] = {
        "panel": {
            "rows": int(len(table)),
            "entity_count": int(table["entity_id"].astype(str).nunique()) if not table.empty else 0,
            "min_date": None if table.empty else str(pd.to_datetime(table["crawled_date_day"], errors="coerce").min()),
            "max_date": None if table.empty else str(pd.to_datetime(table["crawled_date_day"], errors="coerce").max()),
            "entity_limit": int(args.entity_limit),
            "max_rows_per_entity": int(args.max_rows_per_entity),
            "max_rows": int(args.max_rows),
            "rows_with_any_edgar_proxy": int(proxy_positive_mask.sum()),
            "edgar_proxy_row_share": float(proxy_positive_mask.mean()) if len(table) else 0.0,
            "rows_with_any_raw_reference": int(raw_reference_mask.sum()),
            "raw_reference_row_share": float(raw_reference_mask.mean()) if len(table) else 0.0,
            "rows_with_proxy_only_surface": int((proxy_positive_mask & ~raw_reference_mask).sum()),
            "proxy_only_row_share": float((proxy_positive_mask & ~raw_reference_mask).mean()) if len(table) else 0.0,
            "surface_mode_counts": _value_counts_dict(surface_mode),
            "available_mark_columns": {
                column: bool(column in available)
                for column in requested_columns
                if column not in {"entity_id", "crawled_date_day"}
            },
        },
        "thresholds": {
            "institutional_threshold": float(args.institutional_threshold),
            "retail_threshold": float(args.retail_threshold),
            "lead_threshold": float(args.lead_threshold),
            "international_threshold": float(args.international_threshold),
            "high_threshold_usd": float(args.high_threshold_usd),
            "concentrated_capital_threshold": float(args.concentrated_capital_threshold),
            "institutional_proxy_threshold": float(institutional_proxy_threshold),
            "institutional_proxy_min_investment_usd": float(institutional_proxy_min_investment_usd),
            "crowdfunding_retail_prior_enabled": True,
        },
        "flag_counts": {
            "weak_institutional_like_flag": int(table["weak_institutional_like_flag"].sum()),
            "weak_retail_like_flag": int(table["weak_retail_like_flag"].sum()),
            "weak_lead_like_flag": int(table["weak_lead_like_flag"].sum()),
            "weak_repeat_networked_flag": int(table["weak_repeat_networked_flag"].sum()),
            "weak_international_like_flag": int(table["weak_international_like_flag"].sum()),
            "weak_high_threshold_offer_flag": int(table["weak_high_threshold_offer_flag"].sum()),
        },
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "label_counts_by_surface_mode": {
            mode: _value_counts_dict(table.loc[table["weak_surface_mode"] == mode, "weak_primary_label"])
            for mode in sorted(table["weak_surface_mode"].astype(str).unique())
        },
        "mark_score_means": {
            column: float(pd.to_numeric(table[column], errors="coerce").fillna(0.0).mean())
            for column in marks.columns
        },
        "proxy_column_coverage": {
            column: {
                "non_null_rows": int(pd.to_numeric(table[column], errors="coerce").notna().sum()),
                "positive_rows": int((pd.to_numeric(table[column], errors="coerce").fillna(0.0) > 0.0).sum()),
            }
            for column in MARK_PROXY_COLUMNS
            if column in table.columns
        },
        "examples": {
            "institutional_like": _example_rows(table, "weak_institutional_like_flag"),
            "retail_like": _example_rows(table, "weak_retail_like_flag"),
            "lead_like": _example_rows(table, "weak_lead_like_flag"),
            "repeat_networked": _example_rows(table, "weak_repeat_networked_flag"),
            "high_threshold_offer": _example_rows(table, "weak_high_threshold_offer_flag"),
        },
    }

    if args.output_table:
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.output_table, index=False)
    if args.output_summary_json:
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if args.output_sample_csv:
        args.output_sample_csv.parent.mkdir(parents=True, exist_ok=True)
        sample = table.head(min(len(table), 2000)).copy()
        sample.to_csv(args.output_sample_csv, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())