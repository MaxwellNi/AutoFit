#!/usr/bin/env python
"""
Build column_contract_v3.yaml from raw profiles and existing v2 contract.
Rules: coverage_min, variance_min, low_cardinality_max; arrays/text -> offers_text.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

repo_root = Path(__file__).resolve().parent.parent


def _load_profile(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v2() -> Dict[str, Any]:
    p = repo_root / "configs" / "column_contract_v2.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _cols_meeting_coverage(profile: Dict[str, Any], coverage_min: float) -> Set[str]:
    cols = set()
    for col, v in profile.get("columns", {}).items():
        nnr = v.get("approx_non_null_rate")
        if nnr is not None and nnr >= coverage_min:
            cols.add(col)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Build column_contract_v3 from profiles.")
    parser.add_argument("--raw_offers_profile", type=Path, required=True)
    parser.add_argument("--raw_edgar_profile", type=Path, required=True)
    parser.add_argument("--output_yaml", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument("--coverage_min", type=float, default=0.05)
    args = parser.parse_args()

    offers_prof = _load_profile(args.raw_offers_profile)
    edgar_prof = _load_profile(args.raw_edgar_profile)
    v2 = _load_v2()

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_yaml = args.output_yaml or repo_root / "configs" / "column_contract_v3.yaml"
    out_md = args.output_md or repo_root / "docs" / "audits" / f"column_contract_v3_{stamp}.md"

    offers_cols = _cols_meeting_coverage(offers_prof, args.coverage_min)
    edgar_cols = _cols_meeting_coverage(edgar_prof, args.coverage_min)

    core_must = list(v2.get("offers_core", {}).get("must_keep", []))
    static_must = list(v2.get("offers_static", {}).get("must_keep", []))
    text_must = list(v2.get("offers_text", {}).get("must_keep", []))
    edgar_must = list(v2.get("edgar_store", {}).get("must_keep", []))
    can_drop = list(v2.get("offers_core", {}).get("can_drop", []))
    should_add = list(v2.get("should_add_v3_candidates", []))

    for c in core_must:
        if c not in offers_cols and c in offers_prof.get("columns", {}):
            offers_cols.add(c)
    for c in text_must:
        if c not in offers_cols and c in offers_prof.get("columns", {}):
            offers_cols.add(c)

    v3 = {
        "stamp": stamp,
        "coverage_min": args.coverage_min,
        "offers_core_snapshot": {
            "must_keep": core_must,
            "should_add": [c for c in should_add if c in offers_cols],
            "can_drop": can_drop,
        },
        "offers_core_daily": {
            "must_keep": core_must,
            "should_add": [c for c in should_add if c in offers_cols],
            "can_drop": can_drop,
        },
        "offers_static": {
            "must_keep": static_must,
            "can_drop": can_drop,
        },
        "offers_text": {
            "must_keep": text_must,
        },
        "edgar_store": {
            "must_keep": edgar_must,
        },
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.dump(v3, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {out_yaml}", flush=True)

    md_lines = [
        f"# Column Contract v3 ({stamp})",
        "",
        "Built from raw profiles + column_contract_v2.",
        f"Coverage min: {args.coverage_min}",
        "",
        "## offers_core_snapshot / offers_core_daily",
        "- must_keep:", "  - " + "\n  - ".join(v3["offers_core_snapshot"]["must_keep"]),
        "- can_drop:", "  - " + "\n  - ".join(v3["offers_core_snapshot"]["can_drop"]),
        "",
        "## offers_text",
        "- must_keep:", "  - " + "\n  - ".join(v3["offers_text"]["must_keep"]),
        "",
        "## edgar_store",
        "- must_keep: (27 last/mean/ema features + id cols)",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
