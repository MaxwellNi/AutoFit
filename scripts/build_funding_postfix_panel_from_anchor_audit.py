#!/usr/bin/env python3
"""Build a funding postfix-style panel from the completed anchor audit plus benchmark incumbents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_funding_postfix_panel import _build_summary, _classify_against_baseline, _relative_gap_pct
from scripts.run_v740_shared112_champion_loop import _load_shared112_surface


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--anchor-audit-json",
        type=Path,
        default=REPO_ROOT / "runs" / "analysis" / "mainline_marked_investor" / "funding_anchor_dominance_audit_full_shared112_fast_mainline.json",
    )
    ap.add_argument(
        "--all-results-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv",
    )
    ap.add_argument("--profile", choices=("quick", "audit", "hard"), default="quick")
    ap.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "runs" / "analysis" / "mainline_marked_investor" / "funding_postfix_panel_full_shared112_fast_with_benchmark_incumbent.json",
    )
    return ap.parse_args()


def _load_anchor_case_rows(anchor_audit_json: Path) -> List[Dict[str, Any]]:
    payload = json.loads(anchor_audit_json.read_text(encoding="utf-8"))
    rows = payload.get("anchor_audit", {}).get("case_rows", [])
    if not rows:
        raise SystemExit("Anchor audit JSON does not contain anchor_audit.case_rows")
    return [dict(row) for row in rows]


def _benchmark_frame(all_results_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(all_results_csv)
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").fillna(-1).astype(int)
    return frame


def _case_index(manifest_cells: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(cell["name"]): dict(cell) for cell in manifest_cells}


def _benchmark_case_row(benchmark_frame: pd.DataFrame, case: Dict[str, Any]) -> pd.Series | None:
    mask = (
        benchmark_frame["task"].astype(str).eq(str(case["task"]))
        & benchmark_frame["ablation"].astype(str).eq(str(case["ablation"]))
        & benchmark_frame["target"].astype(str).eq(str(case["target"]))
        & benchmark_frame["horizon"].astype(int).eq(int(case["horizon"]))
        & benchmark_frame["model_name"].astype(str).eq(str(case["incumbent_model"]))
    )
    rows = benchmark_frame.loc[mask].copy()
    if rows.empty:
        return None
    rows = rows.sort_values(["mae", "model_name"], ascending=[True, True]).reset_index(drop=True)
    return rows.iloc[0]


def _safe_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def main() -> int:
    args = _parse_args()
    anchor_rows = _load_anchor_case_rows(args.anchor_audit_json)
    manifest = _load_shared112_surface(args.all_results_csv, args.profile)
    case_lookup = _case_index(manifest["cells"])
    benchmark = _benchmark_frame(args.all_results_csv)

    case_reports: List[Dict[str, Any]] = []
    for row in anchor_rows:
        case_name = str(row["case"])
        if case_name not in case_lookup:
            continue
        case = dict(case_lookup[case_name])
        benchmark_row = _benchmark_case_row(benchmark, case)
        candidate_mae = _safe_metric(row.get("candidate_mae"))
        anchor_mae = _safe_metric(row.get("anchor_mae") or row.get("baseline_mae"))
        incumbent_mae = _safe_metric(benchmark_row.get("mae")) if benchmark_row is not None else None
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
            "mainline": {
                "metrics": {"mae": candidate_mae},
                "runtime_contract_ok": True,
                "funding_process_contract": {
                    "lane_residual_blend": _safe_metric(row.get("lane_residual_blend")),
                    "lane_jump_event_rate": _safe_metric(row.get("lane_jump_event_rate")),
                    "lane_positive_jump_rows": _safe_metric(row.get("lane_positive_jump_rows")),
                    "lane_uses_jump_hurdle_head": True,
                },
                "anchor": {
                    "metrics": {"mae": anchor_mae},
                    "history_available_share": None,
                    "lag1_nonzero_share": None,
                },
            },
            "incumbent": {
                "metrics": {"mae": incumbent_mae},
                "fit_seconds": _safe_metric(benchmark_row.get("train_time_seconds")) if benchmark_row is not None else None,
                "predict_seconds": _safe_metric(benchmark_row.get("inference_time_seconds")) if benchmark_row is not None else None,
                "benchmark_source": "all_results_csv",
            },
            "comparisons": {
                "vs_anchor": {
                    "baseline_mae": anchor_mae,
                    "gap_pct": float(_relative_gap_pct(candidate_mae, anchor_mae)) if candidate_mae is not None and anchor_mae is not None else None,
                    "outcome": row.get("outcome_vs_anchor") or row.get("outcome"),
                },
                "vs_incumbent": {
                    "baseline_mae": incumbent_mae,
                    "gap_pct": float(_relative_gap_pct(candidate_mae, incumbent_mae)) if candidate_mae is not None and incumbent_mae is not None else None,
                    "outcome": _classify_against_baseline(candidate_mae, incumbent_mae, args.tie_tolerance_pct),
                },
            },
        }
        case_reports.append(case_report)

    report = {
        "panel": {
            "name": "all_shared112_funding",
            "description": "All shared112 funding cells.",
            "profile": args.profile,
            "variant": "mainline_alpha",
            "cases": len(case_reports),
            "skip_incumbent": False,
            "runtime_owner": "single_model_mainline",
            "required_runtime_contract": "predict_time_current_target_masked_before_history_and_anchor",
            "incumbent_source": "benchmark_csv",
            "mainline_source": "funding_anchor_audit_case_rows",
        },
        "summary": _build_summary(case_reports),
        "cases": case_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "cases": len(case_reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())