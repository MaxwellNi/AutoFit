#!/usr/bin/env python3
"""Run a post-fix no-leak funding panel for native single-model mainline on shared112 local slices."""
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

from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _compute_metrics, _prepare_features
from scripts.run_v740_shared112_champion_loop import _load_shared112_surface
from src.narrative.block3.models.nf_adaptive_champion import NFAdaptiveChampionV739
from src.narrative.block3.models.registry import check_model_available, get_model
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper


TASK_ORDER = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
ABLATION_ORDER = {"core_only": 0, "core_edgar": 1, "core_text": 2, "full": 3}

NO_LEAK_CONTRACT = "predict_time_current_target_masked_before_history_and_anchor"
PANEL_SPECS: Dict[str, Dict[str, Any]] = {
    "hardest_family": {
        "description": "Source-rich, longer-horizon shared112 funding cells: core_edgar/full x h14/h30 across all tasks.",
        "ablations": {"core_edgar", "full"},
        "horizons": {14, 30},
    },
    "long_horizon_all": {
        "description": "All shared112 funding cells at h14/h30 across all ablations.",
        "ablations": set(),
        "horizons": {14, 30},
    },
    "all_shared112_funding": {
        "description": "All shared112 funding cells.",
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
    ap.add_argument(
        "--skip-incumbent",
        action="store_true",
        help="Only run native mainline plus its no-leak anchor baseline. Useful for fast funding probes before full incumbent comparison.",
    )
    ap.add_argument(
        "--existing-panel-json",
        type=Path,
        default=None,
        help="Reuse an existing funding postfix panel JSON and only fill missing incumbent/comparison fields.",
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
        if str(cell.get("target")) != "funding_raised_usd":
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


def _relative_gap_pct(candidate_mae: float, baseline_mae: float) -> float:
    return (float(candidate_mae) / max(abs(float(baseline_mae)), 1e-9) - 1.0) * 100.0


def _classify_against_baseline(candidate_mae: float | None, baseline_mae: float | None, tie_tol_pct: float) -> str:
    if candidate_mae is None or baseline_mae is None:
        return "incomplete"
    rel = abs(float(candidate_mae) - float(baseline_mae)) / max(abs(float(baseline_mae)), 1e-9) * 100.0
    if rel <= float(tie_tol_pct) + 1e-12:
        return "tie"
    return "better" if float(candidate_mae) < float(baseline_mae) else "worse"


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
    return {
        "metrics": {
            "mae": _safe_metric(benchmark_row.get("mae")),
            "rmse": _safe_metric(benchmark_row.get("rmse")),
            "mape": _safe_metric(benchmark_row.get("mape")),
            "smape": _safe_metric(benchmark_row.get("smape")),
        },
        "fit_seconds": _safe_metric(benchmark_row.get("train_time_seconds")),
        "predict_seconds": _safe_metric(benchmark_row.get("inference_time_seconds")),
        "prediction_std": None,
        "constant_prediction": False,
        "train_rows": None,
        "val_rows": None,
        "test_rows": _safe_metric(benchmark_row.get("effective_eval_rows")),
        "train_matrix_rows": None,
        "test_matrix_rows": _safe_metric(benchmark_row.get("effective_eval_rows")),
        "feature_count": None,
        "pred_mean": None,
        "target_mean": None,
        "benchmark_source": "all_results_csv",
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
    y_true = y_test.to_numpy(dtype=np.float64)
    pred_std = float(np.nanstd(preds)) if len(preds) else 0.0
    return {
        "metrics": _compute_metrics(y_true, preds),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_std": pred_std,
        "constant_prediction": bool(len(preds) > 1 and pred_std < 1e-8),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "pred_mean": float(np.mean(preds)) if len(preds) else 0.0,
        "target_mean": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def _build_anchor_report(
    model: SingleModelMainlineWrapper,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test: pd.DataFrame,
) -> Dict[str, Any]:
    runtime_frame = model._prepare_runtime_frame(X_test, raw_frame=test)
    runtime_frame = model._mask_runtime_target_for_prediction(runtime_frame)
    history_frame = model._build_target_history_features(runtime_frame, include_seed=True)
    anchor = model._resolve_anchor(history_frame)
    y_true = y_test.to_numpy(dtype=np.float64)
    lag1 = pd.to_numeric(history_frame.get("target_lag1"), errors="coerce")
    history_count = pd.to_numeric(history_frame.get("target_history_count"), errors="coerce")
    return {
        "metrics": _compute_metrics(y_true, anchor),
        "pred_mean": float(np.mean(anchor)) if len(anchor) else 0.0,
        "prediction_std": float(np.nanstd(anchor)) if len(anchor) else 0.0,
        "history_available_share": float(np.mean(history_count.fillna(0.0).to_numpy(dtype=np.float64) > 0.0))
        if len(history_count)
        else 0.0,
        "lag1_nonzero_share": float(np.mean(lag1.fillna(0.0).to_numpy(dtype=np.float64) > 0.0))
        if len(lag1)
        else 0.0,
    }


def _run_mainline_case(
    case: Dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    X_train, y_train = _prepare_features(train, case["target"])
    X_test, y_test = _prepare_features(test, case["target"])
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows for {case['name']}: train={len(X_train)} test={len(X_test)}"
        )

    model = _instantiate_mainline(args)
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
    y_true = y_test.to_numpy(dtype=np.float64)
    pred_std = float(np.nanstd(preds)) if len(preds) else 0.0
    regime = model.get_regime_info()
    runtime = dict(regime.get("runtime", {}))
    funding_contract = dict(regime.get("funding_process_contract", {}))
    source_stats = dict(regime.get("source_stats", {}))
    anchor_report = _build_anchor_report(model, X_test, y_test, test)
    runtime_contract_ok = bool(runtime.get("predict_time_current_target_masked") is True) and (
        runtime.get("runtime_no_leak_contract") == NO_LEAK_CONTRACT
    )
    return {
        "metrics": _compute_metrics(y_true, preds),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_std": pred_std,
        "constant_prediction": bool(len(preds) > 1 and pred_std < 1e-8),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "pred_mean": float(np.mean(preds)) if len(preds) else 0.0,
        "target_mean": float(np.mean(y_true)) if len(y_true) else 0.0,
        "runtime": runtime,
        "runtime_contract_ok": runtime_contract_ok,
        "funding_process_contract": funding_contract,
        "source_stats": source_stats,
        "anchor": anchor_report,
    }


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
    anchor_gaps = [
        float(report["comparisons"]["vs_anchor"]["gap_pct"])
        for report in reports
        if report["comparisons"]["vs_anchor"].get("gap_pct") is not None
    ]
    incumbent_gaps = [
        float(report["comparisons"]["vs_incumbent"]["gap_pct"])
        for report in reports
        if report["comparisons"]["vs_incumbent"].get("gap_pct") is not None
    ]
    anchor_outcomes = Counter(report["comparisons"]["vs_anchor"].get("outcome", "incomplete") for report in reports)
    incumbent_outcomes = Counter(
        report["comparisons"]["vs_incumbent"].get("outcome", "incomplete") for report in reports
    )
    return {
        "cases": len(reports),
        "no_leak_runtime_pass": int(sum(bool(report["mainline"].get("runtime_contract_ok")) for report in reports)),
        "anchor_fallback_cases": int(
            sum(
                abs(float(report["mainline"].get("funding_process_contract", {}).get("lane_residual_blend", 1.0)))
                <= 1e-12
                for report in reports
            )
        ),
        "jump_hurdle_active_cases": int(
            sum(
                bool(report["mainline"].get("funding_process_contract", {}).get("lane_uses_jump_hurdle_head"))
                for report in reports
            )
        ),
        "anchor_outcomes": dict(anchor_outcomes),
        "incumbent_outcomes": dict(incumbent_outcomes),
        "mean_gap_vs_anchor_pct": float(np.mean(anchor_gaps)) if anchor_gaps else None,
        "median_gap_vs_anchor_pct": float(np.median(anchor_gaps)) if anchor_gaps else None,
        "mean_gap_vs_incumbent_pct": float(np.mean(incumbent_gaps)) if incumbent_gaps else None,
        "median_gap_vs_incumbent_pct": float(np.median(incumbent_gaps)) if incumbent_gaps else None,
    }


def _top_cases(case_reports: Iterable[Dict[str, Any]], key: str, *, reverse: bool, limit: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in case_reports:
        cmp_row = report["comparisons"].get(key, {})
        gap_pct = cmp_row.get("gap_pct")
        if gap_pct is None:
            continue
        rows.append(
            {
                "case": report["case"]["name"],
                "task": report["case"]["task"],
                "ablation": report["case"]["ablation"],
                "horizon": int(report["case"]["horizon"]),
                "gap_pct": float(gap_pct),
                "outcome": cmp_row.get("outcome"),
                "candidate_mae": report["mainline"].get("metrics", {}).get("mae"),
                "baseline_mae": cmp_row.get("baseline_mae"),
                "lane_residual_blend": report["mainline"].get("funding_process_contract", {}).get("lane_residual_blend"),
                "lane_jump_event_rate": report["mainline"].get("funding_process_contract", {}).get("lane_jump_event_rate"),
                "lane_positive_jump_rows": report["mainline"].get("funding_process_contract", {}).get("lane_positive_jump_rows"),
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
        "worst_vs_anchor": _top_cases(case_reports, "vs_anchor", reverse=True),
        "best_vs_anchor": _top_cases(case_reports, "vs_anchor", reverse=False),
        "worst_vs_incumbent": _top_cases(case_reports, "vs_incumbent", reverse=True),
        "best_vs_incumbent": _top_cases(case_reports, "vs_incumbent", reverse=False),
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
        raise SystemExit("No shared112 funding cases matched the selected panel filters.")

    temporal_config = _make_temporal_config()
    case_reports: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[funding-postfix] preparing {case['name']} ({case['incumbent_model']} is incumbent)", flush=True)
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
                "comparisons": {
                    "vs_anchor": {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None},
                    "vs_incumbent": {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None},
                },
            }
        case_report.setdefault("comparisons", {})
        case_report["comparisons"].setdefault(
            "vs_anchor", {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None}
        )
        case_report["comparisons"].setdefault(
            "vs_incumbent", {"outcome": "incomplete", "gap_pct": None, "baseline_mae": None}
        )
        if _should_reuse_mainline(case_report):
            print(f"[funding-postfix] reusing cached native mainline on {case['name']}", flush=True)
        else:
            try:
                print(f"[funding-postfix] running native mainline on {case['name']}", flush=True)
                case_report["mainline"] = _run_mainline_case(case, train, val, test, args)
            except Exception as exc:
                case_report["mainline"] = {"error": str(exc)}
                print(f"[funding-postfix] mainline failed on {case['name']}: {exc}", flush=True)

        if args.skip_incumbent:
            case_report["incumbent"] = {"skipped": True}
        elif args.use_benchmark_incumbent:
            case_report["incumbent"] = _build_benchmark_incumbent_report(_benchmark_case_row(benchmark_frame, case))
        else:
            try:
                print(f"[funding-postfix] running incumbent {case['incumbent_model']} on {case['name']}", flush=True)
                incumbent_model = _instantiate_incumbent(case["incumbent_model"])
                case_report["incumbent"] = _run_model(incumbent_model, case, train, val, test)
            except Exception as exc:
                case_report["incumbent"] = {"error": str(exc)}
                print(f"[funding-postfix] incumbent failed on {case['name']}: {exc}", flush=True)

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
        _write_panel_report(args.output_json, _build_panel_report_payload(existing_panel_report, args, case_reports))

    report = _build_panel_report_payload(existing_panel_report, args, case_reports)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    _write_panel_report(args.output_json, report)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())