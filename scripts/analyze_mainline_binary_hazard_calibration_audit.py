#!/usr/bin/env python3
"""Run a binary hazard/calibration audit on top of the mainline postfix panel."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_binary_postfix_panel import (
    METRIC_DIRECTIONS,
    NO_LEAK_CONTRACT,
    PANEL_SPECS,
    _build_summary as _build_panel_summary,
    _instantiate_incumbent,
    _make_temporal_config,
    _metric_report,
    _positive_int,
    _run_mainline_case,
    _run_model,
    _select_panel_cases,
)
from scripts.run_v740_alpha_minibenchmark import _build_case_frame
from scripts.run_v740_shared112_champion_loop import _load_shared112_surface
from scripts.build_shared112_scorecard import _is_severe_binary_calibration_case


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-results-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv",
    )
    ap.add_argument("--panel", choices=tuple(PANEL_SPECS.keys()), default="all_shared112_binary")
    ap.add_argument("--profile", choices=("quick", "audit", "hard"), default="audit")
    ap.add_argument("--task", choices=("task1_outcome", "task2_forecast", "task3_risk_adjust"), default="")
    ap.add_argument("--ablation", choices=("core_only", "core_edgar", "core_text", "full"), default="")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--case-substr", default="")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    ap.add_argument("--collapse-std-threshold", type=float, default=0.01)
    ap.add_argument("--skip-incumbent", action="store_true")
    ap.add_argument("--panel-report-json", type=Path, default=None)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "analysis" / "mainline_marked_investor" / "binary_hazard_calibration_audit.json",
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
        raise SystemExit("No shared112 binary cases matched the selected panel filters.")

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
            "calibration_issue": False,
            "comparisons": {"vs_incumbent": {}},
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
                case_report["incumbent"] = _run_model(
                    incumbent_model,
                    case,
                    train,
                    val,
                    test,
                    collapse_std_threshold=args.collapse_std_threshold,
                )
            except Exception as exc:
                case_report["incumbent"] = {"error": str(exc)}

        if isinstance(case_report["mainline"], dict) and isinstance(case_report["incumbent"], dict):
            mainline_metrics = case_report["mainline"].get("metrics", {})
            incumbent_metrics = case_report["incumbent"].get("metrics", {})
            case_report["comparisons"]["vs_incumbent"] = {
                metric_name: _metric_report(mainline_metrics, incumbent_metrics, metric_name, args.tie_tolerance_pct)
                for metric_name in METRIC_DIRECTIONS
            }
            case_report["calibration_issue"] = bool(
                _is_severe_binary_calibration_case(case_report["mainline"], case_report["incumbent"], case["target"])
            )
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
            "collapse_std_threshold": float(args.collapse_std_threshold),
        },
        "summary": _build_panel_summary(case_reports),
        "cases": case_reports,
    }


def _hazard_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value >= 0.60:
        return "high_hazard"
    if value >= 0.25:
        return "medium_hazard"
    if value > 0.0:
        return "light_hazard"
    return "no_hazard"


def _case_row(report: Dict[str, Any]) -> Dict[str, Any]:
    mainline = dict(report.get("mainline", {}))
    binary = dict(mainline.get("binary_process_contract", {}))
    comparisons = dict(report.get("comparisons", {}).get("vs_incumbent", {}))
    brier_cmp = dict(comparisons.get("brier", {}))
    ece_cmp = dict(comparisons.get("ece", {}))
    hazard_blend = binary.get("hazard_blend")
    return {
        "case": report.get("case", {}).get("name"),
        "task": report.get("case", {}).get("task"),
        "ablation": report.get("case", {}).get("ablation"),
        "horizon": int(report.get("case", {}).get("horizon", 0)),
        "runtime_contract_ok": bool(mainline.get("runtime_contract_ok")),
        "probability_collapse": bool(mainline.get("probability_collapse")),
        "calibration_issue": bool(report.get("calibration_issue")),
        "uses_hazard_adapter": bool(binary.get("uses_hazard_adapter")),
        "hazard_blend": hazard_blend,
        "hazard_bucket": _hazard_bucket(hazard_blend),
        "hazard_rows": binary.get("hazard_rows"),
        "calibration_method": binary.get("calibration_method"),
        "selected_brier": binary.get("selected_brier"),
        "selected_ece": binary.get("selected_ece"),
        "identity_brier": binary.get("identity_brier"),
        "identity_ece": binary.get("identity_ece"),
        "brier_gap_pct": brier_cmp.get("gap_pct"),
        "brier_outcome": brier_cmp.get("outcome"),
        "ece_gap_pct": ece_cmp.get("gap_pct"),
        "ece_outcome": ece_cmp.get("outcome"),
        "calibration_improved_vs_identity": bool(
            binary.get("selected_brier", np.inf) <= binary.get("identity_brier", np.inf)
            and binary.get("selected_ece", np.inf) <= binary.get("identity_ece", np.inf)
        ),
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
    brier_rows = [row for row in rows if row.get("brier_gap_pct") is not None]
    ece_rows = [row for row in rows if row.get("ece_gap_pct") is not None]
    brier_gaps = [float(row["brier_gap_pct"]) for row in brier_rows]
    ece_gaps = [float(row["ece_gap_pct"]) for row in ece_rows]
    hazard_blends = [float(row["hazard_blend"]) for row in rows if row.get("hazard_blend") is not None]
    return {
        "cases": int(len(rows)),
        "no_leak_runtime_pass": int(sum(bool(row.get("runtime_contract_ok")) for row in rows)),
        "hazard_adapter_active_cases": int(sum(bool(row.get("uses_hazard_adapter")) for row in rows)),
        "probability_collapse_cases": int(sum(bool(row.get("probability_collapse")) for row in rows)),
        "severe_calibration_cases": int(sum(bool(row.get("calibration_issue")) for row in rows)),
        "calibration_improved_vs_identity_cases": int(sum(bool(row.get("calibration_improved_vs_identity")) for row in rows)),
        "mean_hazard_blend": float(np.mean(hazard_blends)) if hazard_blends else None,
        "mean_brier_gap_pct": float(np.mean(brier_gaps)) if brier_gaps else None,
        "median_brier_gap_pct": float(np.median(brier_gaps)) if brier_gaps else None,
        "mean_ece_gap_pct": float(np.mean(ece_gaps)) if ece_gaps else None,
        "median_ece_gap_pct": float(np.median(ece_gaps)) if ece_gaps else None,
        "calibration_method_counts": dict(Counter(str(row.get("calibration_method", "unknown")) for row in rows)),
        "hazard_blend_brier_gap_correlation": _corr(brier_rows, "hazard_blend", "brier_gap_pct"),
        "hazard_blend_ece_gap_correlation": _corr(ece_rows, "hazard_blend", "ece_gap_pct"),
    }


def _group_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups = sorted({str(row.get(key)) for row in rows})
    return {group: _aggregate([row for row in rows if str(row.get(key)) == group]) for group in groups}


def _top_rows(rows: List[Dict[str, Any]], metric_key: str, *, reverse: bool, limit: int = 10) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if row.get(metric_key) is not None]
    filtered.sort(key=lambda row: (float(row[metric_key]), str(row.get("case"))), reverse=reverse)
    return filtered[:limit]


def _build_audit(panel_report: Dict[str, Any]) -> Dict[str, Any]:
    rows = [_case_row(report) for report in panel_report.get("cases", [])]
    severe_rows = [row for row in rows if row.get("calibration_issue")]
    return {
        "overall": _aggregate(rows),
        "by_hazard_bucket": _group_summary(rows, "hazard_bucket"),
        "by_calibration_method": _group_summary(rows, "calibration_method"),
        "by_ablation": _group_summary(rows, "ablation"),
        "by_horizon": _group_summary(rows, "horizon"),
        "top_severe_calibration_cases": _top_rows(severe_rows, "ece_gap_pct", reverse=True),
        "top_brier_failures": _top_rows(rows, "brier_gap_pct", reverse=True),
        "top_brier_wins": _top_rows(rows, "brier_gap_pct", reverse=False),
        "case_rows": rows,
    }


def main() -> int:
    args = _parse_args()
    panel_report = _load_or_run_panel(args)
    audit = _build_audit(panel_report)
    report = {
        "panel": panel_report.get("panel", {}),
        "panel_summary": panel_report.get("summary", {}),
        "hazard_calibration_audit": audit,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())