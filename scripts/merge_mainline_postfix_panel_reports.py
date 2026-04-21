#!/usr/bin/env python3
"""Merge sharded mainline postfix panel JSON reports into a single report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_binary_postfix_panel import _build_summary as _build_binary_summary
from scripts.analyze_mainline_funding_postfix_panel import _build_summary as _build_funding_summary


TASK_ORDER = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
ABLATION_ORDER = {"core_only": 0, "core_edgar": 1, "core_text": 2, "full": 3}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("binary", "funding"), required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("inputs", nargs="+", type=Path)
    return ap.parse_args()


def _sort_key(report: Dict[str, Any]) -> tuple[Any, ...]:
    case = dict(report.get("case", {}))
    return (
        TASK_ORDER.get(str(case.get("task")), 99),
        int(case.get("horizon", 0)),
        ABLATION_ORDER.get(str(case.get("ablation")), 99),
        str(case.get("name", "")),
    )


def _merge_reports(payloads: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if not payloads:
        raise SystemExit("No panel payloads provided")

    merged_cases: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for report in payload.get("cases", []):
            case_name = str(report.get("case", {}).get("name", "")).strip()
            if case_name:
                merged_cases[case_name] = report

    case_reports = sorted(merged_cases.values(), key=_sort_key)
    base_panel = dict(payloads[0].get("panel", {}))
    base_panel["cases"] = int(len(case_reports))
    summary_builder = _build_binary_summary if mode == "binary" else _build_funding_summary
    return {
        "panel": base_panel,
        "summary": summary_builder(case_reports),
        "cases": case_reports,
    }


def main() -> int:
    args = _parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    merged = _merge_reports(payloads, args.mode)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "cases": len(merged.get("cases", []))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())