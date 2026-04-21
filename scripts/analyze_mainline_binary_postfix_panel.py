#!/usr/bin/env python3
"""Run a post-fix no-leak binary panel for native single-model mainline on shared112 local slices."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_shared112_scorecard import _is_severe_binary_calibration_case
from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _compute_metrics, _prepare_features
from scripts.run_v740_shared112_champion_loop import _load_shared112_surface
from src.narrative.block3.models.nf_adaptive_champion import NFAdaptiveChampionV739
from src.narrative.block3.models.registry import check_model_available, get_model
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper


TASK_ORDER = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
ABLATION_ORDER = {"core_only": 0, "core_edgar": 1, "core_text": 2, "full": 3}
NO_LEAK_CONTRACT = "predict_time_current_target_masked_before_history_and_anchor"
METRIC_DIRECTIONS = {
    "mae": True,
    "brier": True,
    "logloss": True,
    "ece": True,
    "auc": False,
    "prauc": False,
}
PANEL_SPECS: Dict[str, Dict[str, Any]] = {
    "hardest_family": {
        "description": "Source-rich, longer-horizon shared112 binary cells: core_edgar/full x h14/h30 across all tasks.",
        "ablations": {"core_edgar", "full"},
        "horizons": {14, 30},
    },
    "source_rich_all": {
        "description": "All source-rich shared112 binary cells: core_edgar/full across all horizons.",
        "ablations": {"core_edgar", "full"},
        "horizons": set(),
    },
    "all_shared112_binary": {
        "description": "All shared112 binary cells.",
        "ablations": set(),
        "horizons": set(),
    },
}


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-results-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv",
    )
    ap.add_argument("--panel", choices=tuple(PANEL_SPECS.keys()), default="hardest_family")
    ap.add_argument("--profile", choices=("quick", "audit", "hard"), default="audit")
    ap.add_argument("--task", choices=tuple(TASK_ORDER.keys()), default="")
    ap.add_argument("--ablation", choices=tuple(ABLATION_ORDER.keys()), default="")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--case-substr", default="")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    ap.add_argument("--collapse-std-threshold", type=float, default=0.01)
    ap.add_argument(
        "--skip-incumbent",
        action="store_true",
        help="Only run native mainline. Useful for quick no-leak binary probes before incumbent comparison.",
    )
    ap.add_argument(
        "--existing-panel-json",
        type=Path,
        default=None,
        help="Reuse an existing binary postfix panel JSON and only fill missing incumbent/comparison fields.",
    )
    ap.add_argument(
        "--use-benchmark-incumbent",
        action="store_true",
        help="Use recorded incumbent benchmark metrics from all_results.csv instead of rerunning incumbent models.",
    )
    ap.add_argument("--output-json", type=Path, default=None)
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


def _sort_key(case: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        TASK_ORDER.get(str(case["task"]), 99),
        int(case["horizon"]),
        ABLATION_ORDER.get(str(case["ablation"]), 99),
        str(case["name"]),
    )


def _select_panel_cases(cells: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    spec = PANEL_SPECS[args.panel]
    selected: List[Dict[str, Any]] = []
    for cell in cells:
        if str(cell.get("target")) != "is_funded":
            continue
        if spec["ablations"] and str(cell.get("ablation")) not in spec["ablations"]:
            continue
        if spec["horizons"] and int(cell.get("horizon", 0)) not in spec["horizons"]:
            continue
        if args.task and str(cell.get("task")) != args.task:
            continue
        if args.ablation and str(cell.get("ablation")) != args.ablation:
            continue
        if args.horizon and int(cell.get("horizon", 0)) != int(args.horizon):
            continue
        if args.case_substr and args.case_substr not in str(cell.get("name", "")):
            continue
        selected.append(dict(cell))
    selected.sort(key=_sort_key)
    if args.max_cases and args.max_cases > 0:
        selected = selected[: args.max_cases]
    return selected


def _relative_gap_pct(candidate_value: float, baseline_value: float, *, lower_is_better: bool) -> float:
    if lower_is_better:
        return (float(candidate_value) / max(abs(float(baseline_value)), 1e-9) - 1.0) * 100.0
    return (float(baseline_value) / max(abs(float(candidate_value)), 1e-9) - 1.0) * 100.0


def _classify_against_baseline(
    candidate_value: float | None,
    baseline_value: float | None,
    tie_tol_pct: float,
    *,
    lower_is_better: bool,
) -> str:
    if candidate_value is None or baseline_value is None:
        return "incomplete"
    rel = abs(float(candidate_value) - float(baseline_value)) / max(abs(float(baseline_value)), 1e-9) * 100.0
    if rel <= float(tie_tol_pct) + 1e-12:
        return "tie"
    if lower_is_better:
        return "better" if float(candidate_value) < float(baseline_value) else "worse"
    return "better" if float(candidate_value) > float(baseline_value) else "worse"


def _metric_report(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    metric_name: str,
    tie_tol_pct: float,
) -> Dict[str, Any]:
    candidate_value = candidate_metrics.get(metric_name)
    baseline_value = baseline_metrics.get(metric_name)
    if candidate_value is None or baseline_value is None:
        return {"baseline_value": None, "candidate_value": None, "gap_pct": None, "outcome": "incomplete"}
    lower_is_better = METRIC_DIRECTIONS[metric_name]
    return {
        "baseline_value": float(baseline_value),
        "candidate_value": float(candidate_value),
        "gap_pct": float(
            _relative_gap_pct(candidate_value, baseline_value, lower_is_better=lower_is_better)
        ),
        "outcome": _classify_against_baseline(
            candidate_value,
            baseline_value,
            tie_tol_pct,
            lower_is_better=lower_is_better,
        ),
    }


def _instantiate_mainline(args: argparse.Namespace) -> SingleModelMainlineWrapper:
    return SingleModelMainlineWrapper(
        variant=args.variant,
        use_delegate=False,
        input_size=args.input_size,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_covariates=args.max_covariates,
        max_entities=3000,
        max_windows=args.max_windows,
        patience=args.patience,
        enable_teacher_distill=not args.disable_teacher_distill,
        enable_event_head=not args.disable_event_head,
        enable_task_modulation=not args.disable_task_modulation,
        seed=args.seed,
    )


def _instantiate_incumbent(resolved_name: str):
    if resolved_name == "AutoFitV739":
        return NFAdaptiveChampionV739(model_timeout=90)
    if not check_model_available(resolved_name):
        raise RuntimeError(f"Model {resolved_name} is not available in this environment")
    return get_model(resolved_name)


def _load_benchmark_frame(all_results_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(all_results_csv)
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").fillna(-1).astype(int)
    return frame


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


def _build_benchmark_incumbent_report(benchmark_row: pd.Series | None) -> Dict[str, Any]:
    if benchmark_row is None:
        return {"error": "benchmark incumbent row not found"}
    metrics: Dict[str, Any] = {
        "mae": _safe_metric(benchmark_row.get("mae")),
        "brier": _safe_metric(benchmark_row.get("binary_brier")),
        "logloss": _safe_metric(benchmark_row.get("binary_logloss")),
        "ece": _safe_metric(benchmark_row.get("binary_ece")),
        "auc": None,
        "prauc": _safe_metric(benchmark_row.get("binary_prauc")),
    }
    return {
        "metrics": metrics,
        "fit_seconds": _safe_metric(benchmark_row.get("train_time_seconds")),
        "predict_seconds": _safe_metric(benchmark_row.get("inference_time_seconds")),
        "prediction_std": None,
        "binary_prob_std": None,
        "constant_prediction": False,
        "probability_collapse": False,
        "train_rows": None,
        "val_rows": None,
        "test_rows": _safe_metric(benchmark_row.get("effective_eval_rows")),
        "train_matrix_rows": None,
        "test_matrix_rows": _safe_metric(benchmark_row.get("effective_eval_rows")),
        "feature_count": None,
        "pred_mean": None,
        "pred_min": None,
        "pred_max": None,
        "target_mean": None,
        "benchmark_source": "all_results_csv",
        "hazard_calibration_method": benchmark_row.get("hazard_calibration_method"),
    }


def _safe_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _run_model(
    model,
    case: Dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    collapse_std_threshold: float,
) -> Dict[str, Any]:
    X_train, y_train = _prepare_features(train, case["target"])
    X_test, y_test = _prepare_features(test, case["target"])
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows for {case['name']}: train={len(X_train)} test={len(X_test)}"
        )

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=case["target"],
        task=case["task"],
        ablation=case["ablation"],
        horizon=case["horizon"],
    )
    fit_seconds = time.time() - t0

    t1 = time.time()
    preds = np.asarray(
        model.predict(
            X_test,
            test_raw=test,
            target=case["target"],
            task=case["task"],
            ablation=case["ablation"],
            horizon=case["horizon"],
        ),
        dtype=np.float64,
    )
    predict_seconds = time.time() - t1
    clipped = np.clip(preds, 0.0, 1.0)
    pred_std = float(np.nanstd(clipped)) if len(clipped) else 0.0
    y_true = y_test.to_numpy(dtype=np.float64)
    return {
        "metrics": _compute_metrics(y_true, clipped),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_std": pred_std,
        "binary_prob_std": pred_std,
        "constant_prediction": bool(len(clipped) > 1 and pred_std < 1e-8),
        "probability_collapse": bool(len(clipped) > 1 and pred_std < float(collapse_std_threshold)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "pred_mean": float(np.mean(clipped)) if len(clipped) else 0.0,
        "pred_min": float(np.min(clipped)) if len(clipped) else 0.0,
        "pred_max": float(np.max(clipped)) if len(clipped) else 0.0,
        "target_mean": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def _run_mainline_case(
    case: Dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model = _instantiate_mainline(args)
    report = _run_model(
        model,
        case,
        train,
        val,
        test,
        collapse_std_threshold=args.collapse_std_threshold,
    )
    regime = model.get_regime_info()
    runtime = dict(regime.get("runtime", {}))
    runtime_contract_ok = bool(runtime.get("predict_time_current_target_masked") is True) and (
        runtime.get("runtime_no_leak_contract") == NO_LEAK_CONTRACT
    )
    report["runtime"] = runtime
    report["runtime_contract_ok"] = runtime_contract_ok
    report["source_stats"] = dict(regime.get("source_stats", {}))
    report["binary_process_contract"] = dict(regime.get("binary_process_contract", {}))
    return report


def _load_existing_case_reports(existing_panel_json: Path | None) -> Dict[str, Dict[str, Any]]:
    if existing_panel_json is None:
        return {}
    payload = json.loads(existing_panel_json.read_text(encoding="utf-8"))
    return {
        str(report.get("case", {}).get("name")): dict(report)
        for report in payload.get("cases", [])
        if isinstance(report, dict) and report.get("case", {}).get("name")
    }


def _rebuild_case_from_report(existing_report: Dict[str, Any]) -> Dict[str, Any]:
    case = dict(existing_report.get("case", {}))
    case["name"] = str(case["name"])
    case["task"] = str(case["task"])
    case["ablation"] = str(case["ablation"])
    case["target"] = str(case["target"])
    case["horizon"] = int(case["horizon"])
    case["max_entities"] = int(case["max_entities"])
    case["max_rows"] = int(case["max_rows"])
    case["incumbent_model"] = str(case["incumbent_model"])
    case["incumbent_benchmark_mae"] = float(case["incumbent_benchmark_mae"])
    return case


def _should_reuse_mainline(existing_report: Dict[str, Any]) -> bool:
    mainline = dict(existing_report.get("mainline", {}))
    return bool(mainline) and ("metrics" in mainline) and ("error" not in mainline)


def _merge_existing_panel_meta(existing_report: Dict[str, Any], args: argparse.Namespace, case_count: int) -> Dict[str, Any]:
    panel = dict(existing_report.get("panel", {})) if existing_report else {}
    panel.update(
        {
            "name": args.panel,
            "description": PANEL_SPECS[args.panel]["description"],
            "profile": args.profile,
            "variant": args.variant,
            "cases": int(case_count),
            "skip_incumbent": bool(args.skip_incumbent),
            "runtime_owner": "single_model_mainline",
            "required_runtime_contract": NO_LEAK_CONTRACT,
            "collapse_std_threshold": float(args.collapse_std_threshold),
        }
    )
    return panel


def _build_panel_report_payload(
    existing_report: Dict[str, Any],
    args: argparse.Namespace,
    case_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "panel": _merge_existing_panel_meta(existing_report, args, len(case_reports)),
        "summary": _build_summary(case_reports),
        "cases": case_reports,
    }


def _write_panel_report(output_json: Path | None, report: Dict[str, Any]) -> None:
    if output_json is None:
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _aggregate_group(case_reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    reports = list(case_reports)
    metric_outcomes = {
        metric_name: Counter(
            report.get("comparisons", {}).get("vs_incumbent", {}).get(metric_name, {}).get("outcome", "incomplete")
            for report in reports
        )
        for metric_name in METRIC_DIRECTIONS
    }
    metric_gaps = {
        metric_name: [
            float(report["comparisons"]["vs_incumbent"][metric_name]["gap_pct"])
            for report in reports
            if report.get("comparisons", {}).get("vs_incumbent", {}).get(metric_name, {}).get("gap_pct") is not None
        ]
        for metric_name in METRIC_DIRECTIONS
    }
    return {
        "cases": len(reports),
        "no_leak_runtime_pass": int(sum(bool(report.get("mainline", {}).get("runtime_contract_ok")) for report in reports)),
        "probability_collapse_cases": int(sum(bool(report.get("mainline", {}).get("probability_collapse")) for report in reports)),
        "severe_calibration_cases": int(sum(bool(report.get("calibration_issue")) for report in reports)),
        "hazard_adapter_active_cases": int(
            sum(
                bool(report.get("mainline", {}).get("binary_process_contract", {}).get("uses_hazard_adapter"))
                for report in reports
            )
        ),
        "calibration_method_counts": dict(
            Counter(
                str(
                    report.get("mainline", {})
                    .get("binary_process_contract", {})
                    .get("calibration_method", "unknown")
                )
                for report in reports
            )
        ),
        "metric_outcomes": {metric_name: dict(counter) for metric_name, counter in metric_outcomes.items()},
        "mean_gap_pct_by_metric": {
            metric_name: float(np.mean(values)) if values else None
            for metric_name, values in metric_gaps.items()
        },
        "median_gap_pct_by_metric": {
            metric_name: float(np.median(values)) if values else None
            for metric_name, values in metric_gaps.items()
        },
    }


def _top_cases(
    case_reports: Iterable[Dict[str, Any]],
    metric_name: str,
    *,
    reverse: bool,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in case_reports:
        metric_cmp = report.get("comparisons", {}).get("vs_incumbent", {}).get(metric_name, {})
        gap_pct = metric_cmp.get("gap_pct")
        if gap_pct is None:
            continue
        rows.append(
            {
                "case": report["case"]["name"],
                "task": report["case"]["task"],
                "ablation": report["case"]["ablation"],
                "horizon": int(report["case"]["horizon"]),
                "gap_pct": float(gap_pct),
                "outcome": metric_cmp.get("outcome"),
                "candidate_value": metric_cmp.get("candidate_value"),
                "baseline_value": metric_cmp.get("baseline_value"),
                "binary_prob_std": report.get("mainline", {}).get("binary_prob_std"),
                "probability_collapse": bool(report.get("mainline", {}).get("probability_collapse")),
                "calibration_issue": bool(report.get("calibration_issue")),
                "hazard_blend": report.get("mainline", {}).get("binary_process_contract", {}).get("hazard_blend"),
                "calibration_method": report.get("mainline", {}).get("binary_process_contract", {}).get("calibration_method"),
            }
        )
    rows.sort(key=lambda row: (float(row["gap_pct"]), row["case"]), reverse=reverse)
    return rows[:limit]


def _build_summary(case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task = {}
    for task in sorted({report["case"]["task"] for report in case_reports}, key=lambda value: TASK_ORDER.get(value, 99)):
        by_task[task] = _aggregate_group([report for report in case_reports if report["case"]["task"] == task])
    by_ablation = {}
    for ablation in sorted(
        {report["case"]["ablation"] for report in case_reports},
        key=lambda value: ABLATION_ORDER.get(value, 99),
    ):
        by_ablation[ablation] = _aggregate_group(
            [report for report in case_reports if report["case"]["ablation"] == ablation]
        )
    by_horizon = {}
    for horizon in sorted({int(report["case"]["horizon"]) for report in case_reports}):
        by_horizon[str(horizon)] = _aggregate_group(
            [report for report in case_reports if int(report["case"]["horizon"]) == int(horizon)]
        )
    return {
        "overall": _aggregate_group(case_reports),
        "by_task": by_task,
        "by_ablation": by_ablation,
        "by_horizon": by_horizon,
        "worst_vs_incumbent_brier": _top_cases(case_reports, "brier", reverse=True),
        "best_vs_incumbent_brier": _top_cases(case_reports, "brier", reverse=False),
        "worst_vs_incumbent_auc": _top_cases(case_reports, "auc", reverse=True),
        "best_vs_incumbent_auc": _top_cases(case_reports, "auc", reverse=False),
    }


def main() -> int:
    args = _parse_args()
    existing_case_reports = _load_existing_case_reports(args.existing_panel_json)
    existing_panel_report = (
        json.loads(args.existing_panel_json.read_text(encoding="utf-8")) if args.existing_panel_json else {}
    )
    manifest = _load_shared112_surface(args.all_results_csv, args.profile)
    cases = _select_panel_cases(manifest["cells"], args)
    benchmark_frame = _load_benchmark_frame(args.all_results_csv) if args.use_benchmark_incumbent else None
    if not cases:
        raise SystemExit("No shared112 binary cases matched the selected panel filters.")

    temporal_config = _make_temporal_config()
    case_reports: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[binary-postfix] preparing {case['name']} ({case['incumbent_model']} is incumbent)", flush=True)
        train, val, test = _build_case_frame(case, temporal_config)
        case_report = dict(existing_case_reports.get(case["name"], {}))
        if not case_report:
            case_report = {
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
        case_report.setdefault("comparisons", {})
        case_report["comparisons"].setdefault("vs_incumbent", {})
        if _should_reuse_mainline(case_report):
            print(f"[binary-postfix] reusing cached native mainline on {case['name']}", flush=True)
        else:
            try:
                print(f"[binary-postfix] running native mainline on {case['name']}", flush=True)
                case_report["mainline"] = _run_mainline_case(case, train, val, test, args)
            except Exception as exc:
                case_report["mainline"] = {"error": str(exc)}
                print(f"[binary-postfix] mainline failed on {case['name']}: {exc}", flush=True)

        if args.skip_incumbent:
            case_report["incumbent"] = {"skipped": True}
        elif args.use_benchmark_incumbent:
            case_report["incumbent"] = _build_benchmark_incumbent_report(_benchmark_case_row(benchmark_frame, case))
        else:
            try:
                print(f"[binary-postfix] running incumbent {case['incumbent_model']} on {case['name']}", flush=True)
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
                print(f"[binary-postfix] incumbent failed on {case['name']}: {exc}", flush=True)

        if isinstance(case_report["mainline"], dict) and isinstance(case_report["incumbent"], dict):
            mainline_metrics = case_report["mainline"].get("metrics", {})
            incumbent_metrics = case_report["incumbent"].get("metrics", {})
            case_report["comparisons"]["vs_incumbent"] = {
                metric_name: _metric_report(mainline_metrics, incumbent_metrics, metric_name, args.tie_tolerance_pct)
                for metric_name in METRIC_DIRECTIONS
            }
            case_report["calibration_issue"] = bool(
                _is_severe_binary_calibration_case(
                    case_report["mainline"],
                    case_report["incumbent"],
                    case["target"],
                )
            )
        case_reports.append(case_report)
        _write_panel_report(args.output_json, _build_panel_report_payload(existing_panel_report, args, case_reports))

    report = _build_panel_report_payload(existing_panel_report, args, case_reports)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    _write_panel_report(args.output_json, report)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())