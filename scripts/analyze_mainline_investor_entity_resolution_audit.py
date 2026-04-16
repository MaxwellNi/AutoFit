#!/usr/bin/env python3
"""Build a first-pass entity-resolution audit for investor list and website references."""
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
    available_core_columns,
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
    ap.add_argument("--min-canonical-frequency", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=output_dir() / "investor_entity_resolution_audit.json",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=output_dir() / "investor_entity_resolution_candidates.csv",
    )
    ap.add_argument("--output-raw-csv", type=Path, default=None)
    return ap.parse_args()


def _safe_unique_nonempty(values: pd.Series) -> int:
    cleaned = values.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    return int(cleaned.nunique())


def _build_canonical_summary(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty:
        return pd.DataFrame(
            columns=[
                "canonical_reference",
                "mention_rows",
                "entity_count",
                "raw_variant_count",
                "domain_count",
                "source_kind_count",
                "institutional_keyword_score_mean",
                "first_seen",
                "last_seen",
                "sample_raw_reference",
                "sample_domain",
                "is_ambiguous",
            ]
        )
    summary = (
        resolved.groupby("canonical_reference", dropna=False)
        .agg(
            mention_rows=("raw_reference", "size"),
            entity_count=("entity_id", "nunique"),
            raw_variant_count=("raw_reference", "nunique"),
            domain_count=("domain", _safe_unique_nonempty),
            source_kind_count=("source_kind", "nunique"),
            institutional_keyword_score_mean=("institutional_keyword_score", "mean"),
            first_seen=("crawled_date_day", "min"),
            last_seen=("crawled_date_day", "max"),
            sample_raw_reference=("raw_reference", "first"),
            sample_domain=("domain", "first"),
        )
        .reset_index()
    )
    summary["is_ambiguous"] = (
        (summary["raw_variant_count"].fillna(0).astype(int) > 1)
        | (summary["domain_count"].fillna(0).astype(int) > 1)
    )
    return summary.sort_values(["mention_rows", "entity_count", "canonical_reference"], ascending=[False, False, True])


def _top_domains(resolved: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    if resolved.empty:
        return []
    domain_rows = resolved.copy()
    domain_rows["domain"] = domain_rows["domain"].fillna("").astype(str).str.strip()
    domain_rows = domain_rows[domain_rows["domain"].ne("")]
    if domain_rows.empty:
        return []
    grouped = (
        domain_rows.groupby("domain")
        .agg(
            mention_rows=("raw_reference", "size"),
            canonical_count=("canonical_reference", "nunique"),
            entity_count=("entity_id", "nunique"),
            institutional_keyword_score_mean=("institutional_keyword_score", "mean"),
        )
        .reset_index()
        .sort_values(["mention_rows", "canonical_count", "domain"], ascending=[False, False, True])
    )
    return grouped.head(top_k).to_dict(orient="records")


def main() -> int:
    args = _parse_args()
    pointer = load_pointer(args.pointer_path)
    available_core = set(available_core_columns(pointer))
    available_panel = set(available_panel_columns(pointer))
    panel = load_investor_mark_panel(
        args.pointer_path,
        requested_columns=BASE_INVESTOR_COLUMNS,
        entity_limit=args.entity_limit,
        max_rows_per_entity=args.max_rows_per_entity,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        selection_mode="reference_surface",
    )
    encoder = InvestorMarkEncoder()
    resolved = encoder.build_entity_resolution_frame(panel)
    canonical_summary = _build_canonical_summary(resolved)
    frequent = canonical_summary[canonical_summary["mention_rows"] >= max(int(args.min_canonical_frequency), 1)].copy()
    ambiguous = frequent[frequent["is_ambiguous"]].copy()

    report: Dict[str, Any] = {
        "panel": {
            "rows": int(len(panel)),
            "entity_count": int(panel["entity_id"].astype(str).nunique()) if not panel.empty else 0,
            "min_date": None if panel.empty else str(pd.to_datetime(panel["crawled_date_day"], errors="coerce").min()),
            "max_date": None if panel.empty else str(pd.to_datetime(panel["crawled_date_day"], errors="coerce").max()),
            "entity_limit": int(args.entity_limit),
            "max_rows_per_entity": int(args.max_rows_per_entity),
            "max_rows": int(args.max_rows),
            "available_reference_columns": {
                "investors__json": bool("investors__json" in available_core),
                "investor_website": bool("investor_website" in available_core),
                "investors__len": bool("investors__len" in available_core),
                "investors__hash": bool("investors__hash" in available_core),
            },
            "available_proxy_columns": {
                column: bool(column in available_panel)
                for column in MARK_PROXY_COLUMNS
            },
        },
        "resolution_summary": {
            "resolved_rows": int(len(resolved)),
            "unique_canonical_references": int(canonical_summary["canonical_reference"].nunique()) if not canonical_summary.empty else 0,
            "unique_domains": int(resolved["domain"].fillna("").astype(str).str.strip().replace("", np.nan).dropna().nunique()) if not resolved.empty else 0,
            "frequent_canonical_references": int(len(frequent)),
            "ambiguous_canonical_references": int(len(ambiguous)),
        },
        "top_canonical_references": frequent.head(args.top_k).to_dict(orient="records"),
        "top_ambiguous_canonical_references": ambiguous.head(args.top_k).to_dict(orient="records"),
        "top_domains": _top_domains(resolved, args.top_k),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        canonical_summary.to_csv(args.output_csv, index=False)
    if args.output_raw_csv:
        args.output_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        resolved.to_csv(args.output_raw_csv, index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())