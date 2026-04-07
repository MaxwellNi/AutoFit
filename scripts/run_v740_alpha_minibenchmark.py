#!/usr/bin/env python3
"""
Run a freeze-backed, benchmark-like local mini-benchmark for V740-alpha.

This stays outside the live benchmark harness and does not write under `runs/`.
It uses the same temporal split semantics and ablation naming, then evaluates
V740-alpha against the current valid AutoFit baseline (V739) on identical local
slice data. The goal is not to create official leaderboard numbers, but to
obtain a fairer local signal than one-off smoke cases.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.models.nf_adaptive_champion import NFAdaptiveChampionV739
from src.narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper
from src.narrative.block3.unified_protocol import (
    TemporalSplitConfig,
    apply_temporal_split,
    load_block3_tasks_config,
)

from scripts.run_v740_alpha_smoke_slice import (
    add_v740_funding_regime_args,
    add_v740_target_regime_args,
    _compute_metrics,
    _downsample_binary_preserve_time,
    _downsample_preserve_time,
    _load_smoke_frame,
    _prepare_features,
    v740_funding_regime_kwargs_from_args,
    v740_target_regime_kwargs_from_args,
)

DEFAULT_CASES: List[Dict[str, Any]] = [
    {
        "name": "mb_t1_core_edgar_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "core_edgar",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 16,
        "max_rows": 1600,
    },
    {
        "name": "mb_t1_full_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "full",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 16,
        "max_rows": 1600,
    },
    {
        "name": "mb_t1_core_edgar_seed2_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "core_edgar_seed2",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 16,
        "max_rows": 1600,
    },
    {
        "name": "mb_t1_core_edgar_funding_h30",
        "task": "task1_outcome",
        "ablation": "core_edgar",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 24,
        "max_rows": 3000,
    },
    {
        "name": "mb_t1_full_funding_h30",
        "task": "task1_outcome",
        "ablation": "full",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 24,
        "max_rows": 3000,
    },
    {
        "name": "mb_t1_core_only_investors_h14",
        "task": "task1_outcome",
        "ablation": "core_only",
        "target": "investors_count",
        "horizon": 14,
        "max_entities": 24,
        "max_rows": 3000,
    },
    {
        "name": "mb_t1_full_investors_h14",
        "task": "task1_outcome",
        "ablation": "full",
        "target": "investors_count",
        "horizon": 14,
        "max_entities": 24,
        "max_rows": 3000,
    },
    {
        "name": "mb_t3_core_edgar_funding_h30",
        "task": "task3_risk_adjust",
        "ablation": "core_edgar",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 24,
        "max_rows": 3000,
    },
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "v740_alpha_minibenchmark_20260325",
    )
    ap.add_argument(
        "--summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V740_ALPHA_MINIBENCHMARK_20260325.md",
    )
    ap.add_argument(
        "--models",
        default="v740_alpha,v739",
        help="Comma-separated local models to run. Supported: v740_alpha,v741_lite,v742_unified,v743_factorized,v744_guarded_phase,v745_evidence_residual,v739",
    )
    ap.add_argument(
        "--case-substr",
        default="",
        help="Optional substring filter over case names.",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional limit on the number of cases to run after filtering.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a case/model pair when its JSON artifact already exists.",
    )
    add_v740_funding_regime_args(ap)
    add_v740_target_regime_args(ap)
    return ap.parse_args()


def _normalize_local_model_id(model_id: str, args: argparse.Namespace) -> str:
    model_id = model_id.strip()
    if model_id == "v740_alpha" and getattr(args, "enable_v745_evidence_residual", False):
        return "v745_evidence_residual"
    if model_id == "v740_alpha" and getattr(args, "enable_v744_guarded_phase", False):
        return "v744_guarded_phase"
    if model_id == "v740_alpha" and getattr(args, "enable_v743_factorized", False):
        return "v743_factorized"
    if model_id == "v740_alpha" and getattr(args, "enable_v742_unified", False):
        return "v742_unified"
    if model_id == "v740_alpha" and getattr(args, "enable_v741_lite", False):
        return "v741_lite"
    return model_id


def _make_temporal_config() -> TemporalSplitConfig:
    tasks_config = load_block3_tasks_config()
    split_cfg = tasks_config.get("split", {})
    return TemporalSplitConfig(
        train_end=split_cfg.get("train_end", "2025-06-30"),
        val_end=split_cfg.get("val_end", "2025-09-30"),
        test_end=split_cfg.get("test_end", "2025-12-31"),
        embargo_days=split_cfg.get("embargo_days", 7),
    )


def _build_case_frame(case: Dict[str, Any], temporal_config: TemporalSplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class _Args:
        task = case["task"]
        ablation = case["ablation"]
        target = case["target"]
        horizon = case["horizon"]
        max_entities = case["max_entities"]
        max_rows = case["max_rows"]

    df = _load_smoke_frame(_Args, temporal_config)
    train, val, test, _ = apply_temporal_split(df, temporal_config)
    if case["max_rows"] and len(train) > case["max_rows"]:
        if case["target"] == "is_funded":
            train = _downsample_binary_preserve_time(train, case["target"], case["max_rows"])
        else:
            train = _downsample_preserve_time(train, case["max_rows"])
    return train, val, test


def _instantiate_model(model_id: str, args: argparse.Namespace):
    if model_id in {"v740_alpha", "v741_lite", "v742_unified", "v743_factorized", "v744_guarded_phase", "v745_evidence_residual"}:
        target_kwargs = v740_target_regime_kwargs_from_args(args)
        if model_id == "v741_lite":
            target_kwargs["enable_v741_lite"] = True
        if model_id == "v742_unified":
            target_kwargs["enable_financing_consistency"] = True
            target_kwargs["enable_target_routing"] = False
            target_kwargs["enable_count_source_routing"] = False
            target_kwargs["enable_count_source_specialists"] = False
            target_kwargs["enable_count_hurdle_head"] = False
            target_kwargs["enable_window_repair"] = True
        if model_id == "v743_factorized":
            target_kwargs["enable_financing_consistency"] = True
            target_kwargs["enable_financing_factorization"] = True
            target_kwargs["enable_target_routing"] = False
            target_kwargs["enable_count_source_routing"] = False
            target_kwargs["enable_count_source_specialists"] = False
            target_kwargs["enable_count_hurdle_head"] = False
            target_kwargs["enable_window_repair"] = True
        if model_id == "v744_guarded_phase":
            target_kwargs["enable_financing_consistency"] = True
            target_kwargs["enable_financing_factorization"] = True
            target_kwargs["enable_financing_guarded_phase"] = True
            target_kwargs["enable_window_repair"] = True
        if model_id == "v745_evidence_residual":
            target_kwargs["enable_financing_consistency"] = True
            target_kwargs["enable_financing_factorization"] = True
            target_kwargs["enable_financing_evidence_residual"] = True
            target_kwargs["enable_window_repair"] = True
        return V740AlphaPrototypeWrapper(
            input_size=60,
            hidden_dim=64,
            max_epochs=3,
            batch_size=128,
            max_covariates=15,
            max_entities=3000,
            max_windows=50000,
            patience=3,
            enable_teacher_distill=True,
            enable_event_head=True,
            enable_task_modulation=True,
            **v740_funding_regime_kwargs_from_args(args),
            **target_kwargs,
            seed=42,
        )
    if model_id == "v739":
        return NFAdaptiveChampionV739(model_timeout=90)
    raise ValueError(f"Unsupported model_id={model_id}")


def _run_model(
    model_id: str,
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
            f"Insufficient rows for {case['name']} / {model_id}: "
            f"train={len(X_train)} test={len(X_test)}"
        )

    model = _instantiate_model(model_id, args)
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
    preds = model.predict(
        X_test,
        test_raw=test,
        target=case["target"],
        task=case["task"],
        ablation=case["ablation"],
        horizon=case["horizon"],
    )
    pred_seconds = time.time() - t1
    preds = np.asarray(preds, dtype=np.float64)
    metrics = _compute_metrics(y_test.to_numpy(dtype=np.float64), preds)
    pred_std = float(np.nanstd(preds)) if len(preds) else 0.0
    constant_prediction = bool(len(preds) > 1 and pred_std < 1e-8)

    selected_model = None
    routing = None
    if hasattr(model, "_routing_info"):
        routing = getattr(model, "_routing_info")
        if isinstance(routing, dict):
            selected_model = routing.get("selected_model")
    regime_info = model.get_regime_info() if hasattr(model, "get_regime_info") else None

    return {
        "model_id": model_id,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": pred_seconds,
        "prediction_std": pred_std,
        "constant_prediction": constant_prediction,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "selected_model": selected_model,
        "routing_info": routing,
        "binary_teacher_weight": float(getattr(model, "_binary_teacher_weight", 0.0)) if getattr(model, "_binary_target", False) else None,
        "binary_event_weight": float(getattr(model, "_binary_event_weight", 0.0)) if getattr(model, "_binary_target", False) else None,
        "task_mod_enabled": bool(getattr(model, "_effective_task_modulation", getattr(model, "enable_task_modulation", False))),
        "funding_log_domain": bool(getattr(model, "_funding_log_domain", False)),
        "funding_source_scaling": bool(getattr(model, "_funding_source_scaling", False)),
        "funding_anchor_enabled": bool(getattr(model, "_funding_anchor_enabled", False)),
        "funding_anchor_strength": float(getattr(model, "_effective_funding_anchor_strength", 0.0)),
        "target_routing_enabled": bool(getattr(model, "enable_target_routing", False)),
        "target_route_experts": int(getattr(model, "target_route_experts", 0)),
        "count_anchor_enabled": bool(getattr(model, "enable_count_anchor", False)),
        "count_anchor_strength": float(getattr(model, "count_anchor_strength", 0.0)),
        "count_jump_enabled": bool(getattr(model, "enable_count_jump", False)),
        "count_jump_strength": float(getattr(model, "count_jump_strength", 0.0)),
        "count_sparsity_gate_enabled": bool(getattr(model, "enable_count_sparsity_gate", False)),
        "count_sparsity_gate_strength": float(getattr(model, "count_sparsity_gate_strength", 0.0)),
        "count_source_routing_enabled": bool(getattr(model, "enable_count_source_routing", False)),
        "count_route_experts": int(getattr(model, "count_route_experts", 0)),
        "count_route_floor": float(getattr(model, "count_route_floor", 0.0)),
        "count_route_entropy_strength": float(getattr(model, "count_route_entropy_strength", 0.0)),
        "count_active_loss_strength": float(getattr(model, "count_active_loss_strength", 0.0)),
        "regime_info": regime_info,
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_summary(path: Path, results: List[Dict[str, Any]]) -> None:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_case.setdefault(row["case"], []).append(row)

    lines = [
        "# V740 Alpha Mini-Benchmark (2026-03-25)",
        "",
        "Generated by `scripts/run_v740_alpha_minibenchmark.py`.",
        "",
        "This is a freeze-backed, benchmark-like local comparison that stays outside the live benchmark harness.",
        "The numbers below are only comparable **within the same local slice**.",
        "",
        "## Summary",
        "",
        "| Case | Model | Task | Ablation | Target | H | MAE | RMSE | Fit(s) | Pred(s) | Const | PredStd | SelectedModel | TaskMod | TeacherW | EventW |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|",
    ]

    for case_name, case_rows in by_case.items():
        case_rows = sorted(case_rows, key=lambda r: r.get("metrics", {}).get("mae", float("inf")))
        for row in case_rows:
            metrics = row.get("metrics", {})
            lines.append(
                "| {case} | {model} | {task} | {ablation} | {target} | {h} | {mae} | {rmse} | {fit} | {pred} | {const} | {std} | {sel} | {taskmod} | {tw} | {ew} |".format(
                    case=case_name,
                    model=row["model_id"],
                    task=row["task"],
                    ablation=row["ablation"],
                    target=row["target"],
                    h=row["horizon"],
                    mae=_fmt(metrics.get("mae")),
                    rmse=_fmt(metrics.get("rmse")),
                    fit=_fmt(row.get("fit_seconds")),
                    pred=_fmt(row.get("predict_seconds")),
                    const=row.get("constant_prediction"),
                    std=_fmt(row.get("prediction_std")),
                    sel=row.get("selected_model") or "-",
                    taskmod=row.get("task_mod_enabled"),
                    tw=_fmt(row.get("binary_teacher_weight")),
                    ew=_fmt(row.get("binary_event_weight")),
                )
            )

    lines.extend(["", "## Per-Case Winners", ""])
    for case_name, case_rows in by_case.items():
        case_rows = [r for r in case_rows if r.get("metrics", {}).get("mae") is not None]
        if not case_rows:
            continue
        case_rows = sorted(case_rows, key=lambda r: r["metrics"]["mae"])
        best = case_rows[0]
        lines.append(
            f"- `{case_name}`: winner=`{best['model_id']}` mae={best['metrics']['mae']:.4f}"
        )

    lines.extend(["", "## Raw JSON Artifacts", ""])
    for case_name, case_rows in by_case.items():
        for row in case_rows:
            lines.append(f"- `{case_name} / {row['model_id']}`: `{row['json_path']}`")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    models = []
    seen_models = set()
    for raw_model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_id = _normalize_local_model_id(raw_model_id, args)
        if model_id in seen_models:
            continue
        seen_models.add(model_id)
        models.append(model_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal_config = _make_temporal_config()
    cases = DEFAULT_CASES
    if args.case_substr:
        cases = [c for c in cases if args.case_substr in c["name"]]
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("No mini-benchmark cases selected.")

    results: List[Dict[str, Any]] = []
    for case in cases:
        print(
            f"[v740-mini] preparing {case['name']} "
            f"({case['task']} {case['ablation']} {case['target']} h={case['horizon']})",
            flush=True,
        )
        train, val, test = _build_case_frame(case, temporal_config)
        for model_id in models:
            print(f"[v740-mini] running {case['name']} / {model_id}", flush=True)
            row = {
                "case": case["name"],
                "task": case["task"],
                "ablation": case["ablation"],
                "target": case["target"],
                "horizon": case["horizon"],
            }
            out_path = args.output_dir / f"{case['name']}__{model_id}.json"
            if args.skip_existing and out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    existing["json_path"] = str(out_path)
                    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
                    results.append(existing)
                    print(
                        f"[v740-mini] skip-existing {case['name']} / {model_id}",
                        flush=True,
                    )
                    continue
                except Exception as exc:
                    print(
                        f"[v740-mini] skip-existing read failed for {case['name']} / {model_id}: {exc}; recomputing",
                        flush=True,
                    )
            try:
                summary = _run_model(model_id, case, train, val, test, args)
                row.update(summary)
                row["json_path"] = str(out_path)
                out_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                print(
                    f"[v740-mini] finished {case['name']} / {model_id} "
                    f"mae={summary.get('metrics', {}).get('mae')}",
                    flush=True,
                )
            except Exception as exc:
                row.update({
                    "model_id": model_id,
                    "error": str(exc),
                    "json_path": "-",
                })
                print(f"[v740-mini] failed {case['name']} / {model_id}: {exc}", flush=True)
            results.append(row)

    _write_summary(args.summary_md, results)
    print(f"[v740-mini] wrote summary to {args.summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
