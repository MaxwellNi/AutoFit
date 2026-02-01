#!/usr/bin/env python
"""
Build column_contract_wide.yaml (v4) from raw column inventory.
Wide-table principle: non_null >= 0.001, std=0 still keep, categorical distinct <= 200k.
Arrays/text: not raw in core; derived (text_len, num_tokens_est) go to core_daily.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parent.parent


def _load_inventory(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_profile_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _profile_to_inventory_df(profile: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for col, v in profile.get("columns", {}).items():
        r = {
            "column": col,
            "dtype": str(v.get("dtype", "")),
            "source": v.get("source", ""),
            "non_null_rate_sample": v.get("approx_non_null_rate") or v.get("non_null_rate_sample"),
            "std_sample": v.get("std_sample"),
            "distinct_sample": v.get("distinct_sample") or v.get("n_unique_est"),
        }
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build column_contract_wide from inventory.")
    parser.add_argument("--inventory_offers", type=Path, default=None, help="raw_offers_column_inventory.parquet")
    parser.add_argument("--inventory_edgar", type=Path, default=None)
    parser.add_argument("--profile_offers", type=Path, default=None, help="Fallback: raw_offers_profile.json")
    parser.add_argument("--profile_edgar", type=Path, default=None)
    parser.add_argument("--contract_v3", type=Path, default=repo_root / "configs/column_contract_v3.yaml")
    parser.add_argument("--output_yaml", type=Path, default=repo_root / "configs/column_contract_wide.yaml")
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--non_null_min", type=float, default=0.001)
    parser.add_argument("--categorical_distinct_max", type=int, default=200_000)
    args = parser.parse_args()

    analysis_dir = repo_root / "runs/orchestrator/20260129_073037/analysis"
    inv_offers = _load_inventory(args.inventory_offers or analysis_dir / "raw_offers_column_inventory.parquet")
    inv_edgar = _load_inventory(args.inventory_edgar or analysis_dir / "raw_edgar_column_inventory.parquet")

    if inv_offers is None or inv_offers.empty:
        prof = _load_profile_json(args.profile_offers or analysis_dir / "raw_offers_profile.json")
        inv_offers = _profile_to_inventory_df(prof)
    if inv_edgar is None or inv_edgar.empty:
        prof = _load_profile_json(args.profile_edgar or analysis_dir / "raw_edgar_profile.json")
        inv_edgar = _profile_to_inventory_df(prof)

    v3 = {}
    if args.contract_v3.exists():
        v3 = yaml.safe_load(args.contract_v3.read_text(encoding="utf-8")) or {}

    def candidate_cols(df: pd.DataFrame, nn_min: float, distinct_max: int) -> tuple[List[str], List[str], List[str]]:
        must = []
        high_value = []
        derived_only = []
        if df.empty or "column" not in df.columns:
            return must, high_value, derived_only
        nn_col = "non_null_rate_sample" if "non_null_rate_sample" in df.columns else "approx_non_null_rate"
        if nn_col not in df.columns:
            nn_col = [c for c in df.columns if "non_null" in c.lower() or "approx" in c.lower()][0] if any("non_null" in str(c).lower() for c in df.columns) else None
        for _, row in df.iterrows():
            col = str(row["column"])
            nnr = row.get(nn_col) if nn_col else None
            nnr = float(nnr) if nnr is not None and pd.notna(nnr) else 0.0
            if nnr < nn_min:
                continue
            std = row.get("std_sample")
            distinct = row.get("distinct_sample")
            distinct = int(distinct) if distinct is not None and pd.notna(distinct) else None
            dtype = str(row.get("dtype", ""))
            if distinct is not None and distinct > distinct_max:
                if "text" in col.lower() or "description" in col.lower() or "headline" in col.lower():
                    derived_only.append(col)
                continue
            must.append(col)
            if std is not None and float(std) > 0:
                high_value.append(col)
        return must, high_value, derived_only

    offers_must, offers_high, offers_derived = candidate_cols(inv_offers, args.non_null_min, args.categorical_distinct_max)
    edgar_must, edgar_high, _ = candidate_cols(inv_edgar, args.non_null_min, args.categorical_distinct_max)

    core_v3 = v3.get("offers_core_snapshot", v3.get("offers_core_daily", {}))
    v3_must = set(core_v3.get("must_keep", []))
    for c in v3_must:
        if c not in offers_must:
            if inv_offers.empty or c in inv_offers["column"].tolist():
                offers_must.append(c)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    wide = {
        "stamp": stamp,
        "non_null_min": args.non_null_min,
        "categorical_distinct_max": args.categorical_distinct_max,
        "offers_core_snapshot": {
            "must_keep": list(dict.fromkeys(offers_must)),
            "high_value": offers_high,
            "derived_only": offers_derived[:30],
            "can_drop": core_v3.get("can_drop", []),
        },
        "offers_core_daily": {
            "must_keep": list(dict.fromkeys(offers_must)),
            "high_value": offers_high,
            "derived_structured": ["text_len", "num_tokens_est", "num_urls", "num_hashtags"],
            "can_drop": core_v3.get("can_drop", []),
        },
        "offers_text": {"must_keep": v3.get("offers_text", {}).get("must_keep", ["headline", "title", "description_text", "company_description"])},
        "edgar_store": {
            "must_keep": list(dict.fromkeys(edgar_must)) if edgar_must else v3.get("edgar_store", {}).get("must_keep", []),
            "high_value": edgar_high,
        },
    }

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.dump(wide, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {args.output_yaml}", flush=True)

    out_md = args.output_md or repo_root / "docs" / "audits" / f"column_contract_wide_{stamp}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        f"# Column Contract Wide ({stamp})",
        "",
        "Wide-table principle: non_null >= 0.001, std=0 still keep, categorical distinct <= 200k.",
        "",
        "## offers_core_snapshot / offers_core_daily",
        f"- must_keep: {len(wide['offers_core_snapshot']['must_keep'])} columns",
        f"- high_value (std>0): {len(wide['offers_core_snapshot']['high_value'])}",
        f"- derived_only (text len/tokens): {wide['offers_core_daily']['derived_structured']}",
        "",
        "## edgar_store",
        f"- must_keep: {len(wide['edgar_store']['must_keep'])} columns",
        "",
        "## vs v3",
        "- v3 coverage_min=0.05; wide uses 0.001",
        "- Arrays/text: raw not in core; derived (text_len, num_tokens_est) in core_daily",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
