#!/usr/bin/env python3
"""Run a funding full-surface anchor-dominance audit on top of the mainline postfix panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_funding_postfix_panel import (
    NO_LEAK_CONTRACT,
    PANEL_SPECS,
    _build_summary as _build_panel_summary,
    _classify_against_baseline,
    _instantiate_incumbent,
    _make_temporal_config,
    _positive_int,
    _relative_gap_pct,
    _run_mainline_case,
    _run_model,
    _select_panel_cases,
)
from scripts.run_v740_alpha_minibenchmark import _build_case_frame
from scripts.run_v740_shared112_champion_loop import _load_shared112_surface


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-results-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv",
    )
    ap.add_argument("--panel", choices=tuple(PANEL_SPECS.keys()), default="all_shared112_funding")
    ap.add_argument("--profile", choices=("quick", "audit", "hard"), default="audit")
    ap.add_argument("--task", choices=("task1_outcome", "task2_forecast", "task3_risk_adjust"), default="")
    ap.add_argument("--ablation", choices=("core_only", "core_edgar", "core_text", "full"), default="")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--case-substr", default="")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    ap.add_argument("--skip-incumbent", action="store_true")
    ap.add_argument("--panel-report-json", type=Path, default=None)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "analysis" / "mainline_marked_investor" / "funding_anchor_dominance_audit.json",
    )
    ap.add_argument("--input-size", type=_positive_int, default=60)
    ap.add_argument("--hidden-dim", type=_positive_int, default=64)
    ap.add_argument("--max-epochs", type=_positive_int, default=3)
    ap.add_argument("--batch-size", type=_positive_int, default=128)
    ap.add_argument("--max-covariates", type=_positive_int, default=15)
    ap.add_argument("--max-windows", type=_positive_int, default=50000)
    ap.add_argument("--patience", type=_positive_int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default="mainline_alpha")
    ap.add_argument("--disable-teacher-distill", action="store_true")
    ap.add_argument("--disable-event-head", action="store_true")
    ap.add_argument("--disable-task-modulation", action="store_true")
    return ap.parse_args()


def _load_or_run_panel(args: argparse.Namespace) -> Dict[str, Any]:
    if args.panel_report_json is not None:
        return json.loads(args.panel_report_json.read_text(encoding="utf-8"))

    manifest = _load_shared112_surface(args.all_results_csv, args.profile)
    cases = _select_panel_cases(manifest["cells"], args)
    if not cases:
        raise SystemExit("No shared112 funding cases matched the selected panel filters.")

    temporal_config = _make_temporal_config()
    case_reports: List[Dict[str, Any]] = []
    for case in cases:
        train, val, test = _build_case_frame(case, temporal_config)
        case_report: Dict[str, Any] = {
            "case": {
                "name": case["name"],
                "task": case["task"],
                "ablation": case["ablation"],
                "target": case["target"],
                "horizon": int(case["horizon"]),
                "max_entities": int(case["max_entities"]),
                "max_rows": int(case["max_rows"]),
                "incumbent_model": case["incumbent_model"],
                "incumbent_benchmark_mae": float(case["incumbent_benchmark_mae"]),
                "runner_up_model": case.get("runner_up_model"),
                "runner_up_benchmark_mae": case.get("runner_up_benchmark_mae"),
            },
            "mainline": {},
            "incumbent": {},
            "comparisons": {
                "vs_anchor": {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None},
                "vs_incumbent": {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None},
            },
        }
        try:
            case_report["mainline"] = _run_mainline_case(case, train, val, test, args)
        except Exception as exc:
            case_report["mainline"] = {"error": str(exc)}

        if args.skip_incumbent:
            case_report["incumbent"] = {"skipped": True}
        else:
            try:
                incumbent_model = _instantiate_incumbent(case["incumbent_model"])
                case_report["incumbent"] = _run_model(incumbent_model, case, train, val, test)
            except Exception as exc:
                case_report["incumbent"] = {"error": str(exc)}

        mainline_mae = case_report["mainline"].get("metrics", {}).get("mae") if isinstance(case_report["mainline"], dict) else None
        anchor_mae = (
            case_report["mainline"].get("anchor", {}).get("metrics", {}).get("mae")
            if isinstance(case_report["mainline"], dict)
            else None
        )
        incumbent_mae = case_report["incumbent"].get("metrics", {}).get("mae") if isinstance(case_report["incumbent"], dict) else None
        if mainline_mae is not None and anchor_mae is not None:
            case_report["comparisons"]["vs_anchor"] = {
                "baseline_mae": float(anchor_mae),
                "gap_pct": float(_relative_gap_pct(mainline_mae, anchor_mae)),
                "outcome": _classify_against_baseline(mainline_mae, anchor_mae, args.tie_tolerance_pct),
            }
        if mainline_mae is not None and incumbent_mae is not None:
            case_report["comparisons"]["vs_incumbent"] = {
                "baseline_mae": float(incumbent_mae),
                "gap_pct": float(_relative_gap_pct(mainline_mae, incumbent_mae)),
                "outcome": _classify_against_baseline(mainline_mae, incumbent_mae, args.tie_tolerance_pct),
            }
        case_reports.append(case_report)

    return {
        "panel": {
            "name": args.panel,
            "description": PANEL_SPECS[args.panel]["description"],
            "profile": args.profile,
            "variant": args.variant,
            "cases": len(case_reports),
            "skip_incumbent": bool(args.skip_incumbent),
            "runtime_owner": "single_model_mainline",
            "required_runtime_contract": NO_LEAK_CONTRACT,
        },
        "summary": _build_panel_summary(case_reports),
        "cases": case_reports,
    }


def _anchor_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value >= 10.0:
        return "very_high_anchor"
    if value >= 3.0:
        return "high_anchor"
    if value >= 1.0:
        return "mixed"
    return "low_anchor"


def _case_row(report: Dict[str, Any]) -> Dict[str, Any]:
    mainline = dict(report.get("mainline", {}))
    funding = dict(mainline.get("funding_process_contract", {}))
    anchor = dict(mainline.get("anchor", {}))
    compare = dict(report.get("comparisons", {}).get("vs_anchor", {}))
    anchor_dominance = funding.get("lane_anchor_dominance")
    gap_pct = compare.get("gap_pct")
    return {
        "case": report.get("case", {}).get("name"),
        "task": report.get("case", {}).get("task"),
        "ablation": report.get("case", {}).get("ablation"),
        "horizon": int(report.get("case", {}).get("horizon", 0)),
        "candidate_mae": mainline.get("metrics", {}).get("mae"),
        "anchor_mae": anchor.get("metrics", {}).get("mae"),
        "gap_vs_anchor_pct": gap_pct,
        "outcome_vs_anchor": compare.get("outcome"),
        "anchor_dominance": anchor_dominance,
        "anchor_bucket": _anchor_bucket(anchor_dominance),
        "lane_residual_blend": funding.get("lane_residual_blend"),
        "lane_jump_event_rate": funding.get("lane_jump_event_rate"),
        "lane_positive_jump_rows": funding.get("lane_positive_jump_rows"),
        "lane_uses_jump_hurdle_head": funding.get("lane_uses_jump_hurdle_head"),
        "runtime_contract_ok": bool(mainline.get("runtime_contract_ok")),
        "anchor_history_available_share": anchor.get("history_available_share"),
        "anchor_lag1_nonzero_share": anchor.get("lag1_nonzero_share"),
    }


def _corr(rows: Iterable[Dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x_val = row.get(x_key)
        y_val = row.get(y_key)
        if x_val is None or y_val is None:
            continue
        if not np.isfinite(float(x_val)) or not np.isfinite(float(y_val)):
            continue
        xs.append(float(x_val))
        ys.append(float(y_val))
    if len(xs) < 2:
        return None
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = [row for row in rows if row.get("gap_vs_anchor_pct") is not None]
    better = [row for row in comparable if row.get("outcome_vs_anchor") == "better"]
    worse = [row for row in comparable if row.get("outcome_vs_anchor") == "worse"]
    ties = [row for row in comparable if row.get("outcome_vs_anchor") == "tie"]
    strong_anchor = [row for row in comparable if (row.get("anchor_dominance") or 0.0) >= 3.0]
    strong_anchor_failures = [row for row in strong_anchor if row.get("outcome_vs_anchor") == "worse"]
    gaps = [float(row["gap_vs_anchor_pct"]) for row in comparable]
    anchor_dominance = [float(row["anchor_dominance"]) for row in comparable if row.get("anchor_dominance") is not None]
    residual_blends = [float(row["lane_residual_blend"]) for row in comparable if row.get("lane_residual_blend") is not None]
    jump_rates = [float(row["lane_jump_event_rate"]) for row in comparable if row.get("lane_jump_event_rate") is not None]
    return {
        "cases": int(len(rows)),
        "comparable_cases": int(len(comparable)),
        "better_than_anchor_cases": int(len(better)),
        "worse_than_anchor_cases": int(len(worse)),
        "tie_cases": int(len(ties)),
        "strong_anchor_cases": int(len(strong_anchor)),
        "strong_anchor_failures": int(len(strong_anchor_failures)),
        "no_leak_runtime_pass": int(sum(bool(row.get("runtime_contract_ok")) for row in rows)),
        "jump_hurdle_active_cases": int(sum(bool(row.get("lane_uses_jump_hurdle_head")) for row in rows)),
        "mean_gap_vs_anchor_pct": float(np.mean(gaps)) if gaps else None,
        "median_gap_vs_anchor_pct": float(np.median(gaps)) if gaps else None,
        "mean_anchor_dominance": float(np.mean(anchor_dominance)) if anchor_dominance else None,
        "mean_residual_blend": float(np.mean(residual_blends)) if residual_blends else None,
        "mean_jump_event_rate": float(np.mean(jump_rates)) if jump_rates else None,
        "anchor_dominance_gap_correlation": _corr(comparable, "anchor_dominance", "gap_vs_anchor_pct"),
    }


def _group_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups = sorted({str(row.get(key)) for row in rows})
    return {group: _aggregate([row for row in rows if str(row.get(key)) == group]) for group in groups}


def _top_rows(rows: List[Dict[str, Any]], *, predicate, reverse: bool, limit: int = 10) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if predicate(row) and row.get("gap_vs_anchor_pct") is not None]
    filtered.sort(key=lambda row: (float(row["gap_vs_anchor_pct"]), str(row.get("case"))), reverse=reverse)
    return filtered[:limit]


def _build_anchor_audit(panel_report: Dict[str, Any]) -> Dict[str, Any]:
    rows = [_case_row(report) for report in panel_report.get("cases", [])]
    return {
        "overall": _aggregate(rows),
        "by_anchor_bucket": _group_summary(rows, "anchor_bucket"),
        "by_ablation": _group_summary(rows, "ablation"),
        "by_horizon": _group_summary(rows, "horizon"),
        "top_strong_anchor_failures": _top_rows(
            rows,
            predicate=lambda row: ((row.get("anchor_dominance") or 0.0) >= 3.0) and (row.get("outcome_vs_anchor") == "worse"),
            reverse=True,
        ),
        "top_low_anchor_wins": _top_rows(
            rows,
            predicate=lambda row: ((row.get("anchor_dominance") or 0.0) < 1.0) and (row.get("outcome_vs_anchor") == "better"),
            reverse=False,
        ),
        "case_rows": rows,
    }


def main() -> int:
    args = _parse_args()
    panel_report = _load_or_run_panel(args)
    audit = _build_anchor_audit(panel_report)
    report = {
        "panel": panel_report.get("panel", {}),
        "panel_summary": panel_report.get("summary", {}),
        "anchor_audit": audit,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())