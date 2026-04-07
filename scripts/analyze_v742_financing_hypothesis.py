#!/usr/bin/env python3
"""Compare V740/V742/V743/V744/V745 financing-process diagnostics on one smoke slice."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v740_alpha_smoke_slice import (  # noqa: E402
    _compute_metrics,
    _downsample_binary_preserve_time,
    _load_smoke_frame,
    _nonnegative_float,
    _positive_int,
    _prepare_features,
    add_v740_funding_regime_args,
    v740_funding_regime_kwargs_from_args,
)
from src.narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper  # noqa: E402
from src.narrative.block3.unified_protocol import (  # noqa: E402
    TemporalSplitConfig,
    apply_temporal_split,
    load_block3_tasks_config,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["task1_outcome", "task2_forecast", "task3_risk_adjust"])
    ap.add_argument("--ablation", required=True, choices=[
        "core_only", "core_only_seed2", "core_text", "core_edgar", "core_edgar_seed2", "full",
    ])
    ap.add_argument("--target", required=True, choices=["funding_raised_usd", "investors_count", "is_funded"])
    ap.add_argument("--horizon", type=_positive_int, required=True)
    ap.add_argument("--max-entities", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--input-size", type=int, default=60)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-covariates", type=int, default=15)
    ap.add_argument("--max-windows", type=int, default=50000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-teacher-distill", action="store_true")
    ap.add_argument("--disable-event-head", action="store_true")
    ap.add_argument("--disable-task-modulation", action="store_true")
    add_v740_funding_regime_args(ap)
    ap.add_argument("--v742-blend", type=_nonnegative_float, default=0.20)
    ap.add_argument("--v742-consistency-strength", type=_nonnegative_float, default=0.10)
    ap.add_argument("--v742-auxiliary-strength", type=_nonnegative_float, default=0.12)
    ap.add_argument("--v743-blend", type=_nonnegative_float, default=0.20)
    ap.add_argument("--v743-consistency-strength", type=_nonnegative_float, default=0.12)
    ap.add_argument("--v743-auxiliary-strength", type=_nonnegative_float, default=0.14)
    ap.add_argument("--v743-scaffold-strength", type=_nonnegative_float, default=0.08)
    ap.add_argument("--v744-blend", type=_nonnegative_float, default=0.20)
    ap.add_argument("--v744-consistency-strength", type=_nonnegative_float, default=0.10)
    ap.add_argument("--v744-auxiliary-strength", type=_nonnegative_float, default=0.12)
    ap.add_argument("--v744-scaffold-strength", type=_nonnegative_float, default=0.08)
    ap.add_argument("--v745-blend", type=_nonnegative_float, default=0.20)
    ap.add_argument("--v745-consistency-strength", type=_nonnegative_float, default=0.12)
    ap.add_argument("--v745-auxiliary-strength", type=_nonnegative_float, default=0.14)
    ap.add_argument("--v745-scaffold-strength", type=_nonnegative_float, default=0.08)
    ap.add_argument("--output-json", type=Path, default=None)
    return ap.parse_args()


def _common_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "input_size": args.input_size,
        "hidden_dim": args.hidden_dim,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "max_covariates": args.max_covariates,
        "max_entities": args.max_entities,
        "max_windows": args.max_windows,
        "patience": args.patience,
        "enable_teacher_distill": not args.disable_teacher_distill,
        "enable_event_head": not args.disable_event_head,
        "enable_task_modulation": not args.disable_task_modulation,
        "seed": args.seed,
        **v740_funding_regime_kwargs_from_args(args),
    }


def _instantiate_model(label: str, args: argparse.Namespace) -> V740AlphaPrototypeWrapper:
    kwargs = _common_model_kwargs(args)
    if label == "v742_unified":
        kwargs.update(
            {
                "enable_financing_consistency": True,
                "financing_consistency_strength": float(args.v742_consistency_strength),
                "financing_auxiliary_strength": float(args.v742_auxiliary_strength),
                "financing_process_blend": float(args.v742_blend),
                "enable_target_routing": False,
                "enable_count_source_routing": False,
                "enable_count_source_specialists": False,
                "enable_count_hurdle_head": False,
                "enable_window_repair": True,
            }
        )
    if label == "v743_factorized":
        kwargs.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "financing_consistency_strength": float(args.v743_consistency_strength),
                "financing_auxiliary_strength": float(args.v743_auxiliary_strength),
                "financing_process_blend": float(args.v743_blend),
                "financing_scaffold_strength": float(args.v743_scaffold_strength),
                "enable_target_routing": False,
                "enable_count_source_routing": False,
                "enable_count_source_specialists": False,
                "enable_count_hurdle_head": False,
                "enable_window_repair": True,
            }
        )
    if label == "v744_guarded_phase":
        kwargs.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "enable_financing_guarded_phase": True,
                "financing_consistency_strength": float(args.v744_consistency_strength),
                "financing_auxiliary_strength": float(args.v744_auxiliary_strength),
                "financing_process_blend": float(args.v744_blend),
                "financing_scaffold_strength": float(args.v744_scaffold_strength),
                "enable_window_repair": True,
            }
        )
    if label == "v745_evidence_residual":
        kwargs.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "enable_financing_evidence_residual": True,
                "financing_consistency_strength": float(args.v745_consistency_strength),
                "financing_auxiliary_strength": float(args.v745_auxiliary_strength),
                "financing_process_blend": float(args.v745_blend),
                "financing_scaffold_strength": float(args.v745_scaffold_strength),
                "enable_window_repair": True,
            }
        )
    return V740AlphaPrototypeWrapper(**kwargs)


def _evaluate_one(
    label: str,
    args: argparse.Namespace,
    train: Any,
    val: Any,
    test: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, Any]:
    model = _instantiate_model(label, args)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=args.target,
        task=args.task,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    fit_seconds = round(time.time() - t0, 3)

    t_pred = time.time()
    preds = model.predict(
        X_test,
        test_raw=test,
        target=args.target,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    predict_seconds = round(time.time() - t_pred, 3)

    return {
        "label": label,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "metrics": _compute_metrics(y_test.values, preds),
        "regime_info": model.get_regime_info(),
        "financing_diagnostics": model.score_financing_diagnostics(test),
    }


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    tasks_config = load_block3_tasks_config()
    split_cfg = tasks_config.get("split", {})
    temporal_config = TemporalSplitConfig(
        train_end=split_cfg.get("train_end", "2025-06-30"),
        val_end=split_cfg.get("val_end", "2025-09-30"),
        test_end=split_cfg.get("test_end", "2025-12-31"),
        embargo_days=split_cfg.get("embargo_days", 7),
    )
    df = _load_smoke_frame(args, temporal_config)
    train, val, test, _ = apply_temporal_split(df, temporal_config)
    if args.max_rows and len(train) > args.max_rows:
        train = _downsample_binary_preserve_time(train, args.target, args.max_rows)

    X_train, y_train = _prepare_features(train, args.target)
    X_test, y_test = _prepare_features(test, args.target)
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows after preparation: train={len(X_train)}, test={len(X_test)}"
        )

    baseline = _evaluate_one("v740_baseline", args, train, val, test, X_train, y_train, X_test, y_test)
    unified = _evaluate_one("v742_unified", args, train, val, test, X_train, y_train, X_test, y_test)
    factorized = _evaluate_one("v743_factorized", args, train, val, test, X_train, y_train, X_test, y_test)
    guarded = _evaluate_one("v744_guarded_phase", args, train, val, test, X_train, y_train, X_test, y_test)
    evidence = _evaluate_one("v745_evidence_residual", args, train, val, test, X_train, y_train, X_test, y_test)

    baseline_mae = float(baseline["metrics"].get("mae", float("nan")))
    unified_mae = float(unified["metrics"].get("mae", float("nan")))
    factorized_mae = float(factorized["metrics"].get("mae", float("nan")))
    guarded_mae = float(guarded["metrics"].get("mae", float("nan")))
    evidence_mae = float(evidence["metrics"].get("mae", float("nan")))
    baseline_rmse = float(baseline["metrics"].get("rmse", float("nan")))
    unified_rmse = float(unified["metrics"].get("rmse", float("nan")))
    factorized_rmse = float(factorized["metrics"].get("rmse", float("nan")))
    guarded_rmse = float(guarded["metrics"].get("rmse", float("nan")))
    evidence_rmse = float(evidence["metrics"].get("rmse", float("nan")))

    summary: Dict[str, Any] = {
        "task": args.task,
        "ablation": args.ablation,
        "target": args.target,
        "horizon": args.horizon,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "selected_entities": int(
            pd.concat([train["entity_id"], val["entity_id"], test["entity_id"]], ignore_index=True).nunique()
        ),
        "v742_requested_regime": {
            "blend": float(args.v742_blend),
            "consistency_strength": float(args.v742_consistency_strength),
            "auxiliary_strength": float(args.v742_auxiliary_strength),
        },
        "v743_requested_regime": {
            "blend": float(args.v743_blend),
            "consistency_strength": float(args.v743_consistency_strength),
            "auxiliary_strength": float(args.v743_auxiliary_strength),
            "scaffold_strength": float(args.v743_scaffold_strength),
        },
        "v744_requested_regime": {
            "blend": float(args.v744_blend),
            "consistency_strength": float(args.v744_consistency_strength),
            "auxiliary_strength": float(args.v744_auxiliary_strength),
            "scaffold_strength": float(args.v744_scaffold_strength),
        },
        "v745_requested_regime": {
            "blend": float(args.v745_blend),
            "consistency_strength": float(args.v745_consistency_strength),
            "auxiliary_strength": float(args.v745_auxiliary_strength),
            "scaffold_strength": float(args.v745_scaffold_strength),
        },
        "models": {
            "v740_baseline": baseline,
            "v742_unified": unified,
            "v743_factorized": factorized,
            "v744_guarded_phase": guarded,
            "v745_evidence_residual": evidence,
        },
        "comparison": {
            "mae_delta_v742_minus_v740": unified_mae - baseline_mae,
            "mae_delta_v743_minus_v740": factorized_mae - baseline_mae,
            "mae_delta_v744_minus_v740": guarded_mae - baseline_mae,
            "mae_delta_v745_minus_v740": evidence_mae - baseline_mae,
            "mae_delta_v743_minus_v742": factorized_mae - unified_mae,
            "mae_delta_v744_minus_v743": guarded_mae - factorized_mae,
            "mae_delta_v745_minus_v743": evidence_mae - factorized_mae,
            "mae_delta_v745_minus_v744": evidence_mae - guarded_mae,
            "rmse_delta_v742_minus_v740": unified_rmse - baseline_rmse,
            "rmse_delta_v743_minus_v740": factorized_rmse - baseline_rmse,
            "rmse_delta_v744_minus_v740": guarded_rmse - baseline_rmse,
            "rmse_delta_v745_minus_v740": evidence_rmse - baseline_rmse,
            "rmse_delta_v743_minus_v742": factorized_rmse - unified_rmse,
            "rmse_delta_v744_minus_v743": guarded_rmse - factorized_rmse,
            "rmse_delta_v745_minus_v743": evidence_rmse - factorized_rmse,
            "rmse_delta_v745_minus_v744": evidence_rmse - guarded_rmse,
            "mae_ratio_v742_over_v740": (unified_mae / baseline_mae) if baseline_mae > 0 else None,
            "mae_ratio_v743_over_v740": (factorized_mae / baseline_mae) if baseline_mae > 0 else None,
            "mae_ratio_v744_over_v740": (guarded_mae / baseline_mae) if baseline_mae > 0 else None,
            "mae_ratio_v745_over_v740": (evidence_mae / baseline_mae) if baseline_mae > 0 else None,
            "rmse_ratio_v742_over_v740": (unified_rmse / baseline_rmse) if baseline_rmse > 0 else None,
            "rmse_ratio_v743_over_v740": (factorized_rmse / baseline_rmse) if baseline_rmse > 0 else None,
            "rmse_ratio_v744_over_v740": (guarded_rmse / baseline_rmse) if baseline_rmse > 0 else None,
            "rmse_ratio_v745_over_v740": (evidence_rmse / baseline_rmse) if baseline_rmse > 0 else None,
        },
        "wall_time_seconds": round(time.time() - t0, 3),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())