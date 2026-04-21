#!/usr/bin/env python3
"""Dawn Operation: Learnable Sparse MoE Trunk Gate Test.

Three-way comparison on each evaluation cell:
  1. Original (random projection backbone + barrier)  = incumbent
  2. NoTrunk (R²-IN → raw features → lane, skip backbone)  = no_trunk
  3. LearnableTrunk (SparseMoETrunk → lane)  = learnable

PASS criterion per cell: learnable MAE < incumbent MAE AND learnable MAE < no_trunk MAE

Usage:
    python scripts/run_dawn_learnable_trunk_gate.py \\
        --targets is_funded funding_raised_usd investors_count \\
        --ablations core_only core_edgar \\
        --horizons 1 7 \\
        --output-dir runs/dawn_gate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v740_alpha_minibenchmark import (
    _build_case_frame,
    _make_temporal_config,
)
from scripts.run_v740_alpha_smoke_slice import _prepare_features
from src.narrative.block3.models.single_model_mainline.wrapper import (
    SingleModelMainlineWrapper,
)

_LOG = logging.getLogger("dawn_gate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

_TASK_MAP = {
    "is_funded": "task1_outcome",
    "funding_raised_usd": "task2_outcome",
    "investors_count": "task3_outcome",
}
_MAX_ENTITIES = 5000
_MAX_ROWS = 8000


# ── Metrics ──────────────────────────────────────────────────────────
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error

    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "pred_std": float(np.nanstd(y_p)),
        "pred_mean": float(np.nanmean(y_p)),
    }
    # AUC / Brier for binary
    try:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        y_int = y_t.astype(int)
        if len(np.unique(y_int)) >= 2:
            y_clip = np.clip(y_p, 0.0, 1.0)
            metrics["auc"] = float(roc_auc_score(y_int, y_clip))
            metrics["brier"] = float(brier_score_loss(y_int, y_clip))
    except Exception:
        pass
    return metrics


# ── Single Cell ──────────────────────────────────────────────────────
def _run_cell(
    target: str,
    ablation: str,
    horizon: int,
    output_dir: Path,
) -> Dict[str, Any]:
    task = _TASK_MAP[target]
    cell_name = f"{task}__{ablation}__{target}__h{horizon}"
    _LOG.info(f"=== Cell: {cell_name} ===")

    case = {
        "task": task,
        "ablation": ablation,
        "target": target,
        "horizon": horizon,
        "max_entities": _MAX_ENTITIES,
        "max_rows": _MAX_ROWS,
        "name": cell_name,
    }
    temporal_config = _make_temporal_config()
    train, val, test = _build_case_frame(case, temporal_config)
    X_train, y_train = _prepare_features(train, target)
    X_test, y_test = _prepare_features(test, target)

    if len(X_train) < 20 or len(X_test) < 10:
        _LOG.warning(f"  Skipping {cell_name}: insufficient rows")
        return {"cell": cell_name, "status": "skipped"}

    result: Dict[str, Any] = {"cell": cell_name, "n_train": len(X_train), "n_test": len(X_test)}
    fit_kwargs_base = dict(
        train_raw=train,
        val_raw=val,
        target=target,
        task=task,
        ablation=ablation,
        horizon=horizon,
    )
    predict_kwargs_base = dict(
        test_raw=test,
        target=target,
        task=task,
        ablation=ablation,
        horizon=horizon,
    )

    # ── Arm 1: Original (random projection) ──
    _LOG.info("  [Arm 1] Original backbone+barrier ...")
    try:
        w_orig = SingleModelMainlineWrapper()
        t0 = time.time()
        w_orig.fit(X_train, y_train, **fit_kwargs_base)
        orig_fit_s = time.time() - t0
        preds_orig = w_orig.predict(X_test, **predict_kwargs_base)
        orig_metrics = _compute_metrics(y_test.to_numpy(), preds_orig)
        result["original"] = {"metrics": orig_metrics, "fit_seconds": orig_fit_s}
        _LOG.info(f"    Original: MAE={orig_metrics['mae']:.6f}")
    except Exception as e:
        _LOG.error(f"    Original FAILED: {e}")
        result["original"] = {"error": str(e)}

    # ── Arm 2: LearnableTrunk ──
    _LOG.info("  [Arm 2] Learnable Sparse MoE Trunk ...")
    try:
        w_learn = SingleModelMainlineWrapper(
            enable_learnable_trunk=True,
            learnable_compact_dim=64,
            learnable_n_experts=6,
            learnable_expert_dim=32,
            learnable_top_k=2,
            learnable_n_epochs=30,
        )
        t0 = time.time()
        w_learn.fit(X_train, y_train, **fit_kwargs_base)
        learn_fit_s = time.time() - t0
        preds_learn = w_learn.predict(X_test, **predict_kwargs_base)
        learn_metrics = _compute_metrics(y_test.to_numpy(), preds_learn)
        result["learnable"] = {
            "metrics": learn_metrics,
            "fit_seconds": learn_fit_s,
            "trunk_desc": w_learn._learnable_trunk.describe() if w_learn._learnable_trunk else {},
        }
        _LOG.info(f"    Learnable: MAE={learn_metrics['mae']:.6f}")
    except Exception as e:
        _LOG.error(f"    Learnable FAILED: {e}")
        result["learnable"] = {"error": str(e)}

    # ── Verdict ──
    if "metrics" in result.get("original", {}) and "metrics" in result.get("learnable", {}):
        orig_mae = result["original"]["metrics"]["mae"]
        learn_mae = result["learnable"]["metrics"]["mae"]
        delta_vs_orig = learn_mae - orig_mae
        pct_vs_orig = (delta_vs_orig / orig_mae * 100) if orig_mae > 0 else 0

        result["verdict"] = {
            "learn_vs_orig_delta": delta_vs_orig,
            "learn_vs_orig_pct": pct_vs_orig,
            "PASS": bool(learn_mae < orig_mae),
        }
        verdict_str = "PASS" if result["verdict"]["PASS"] else "FAIL"
        _LOG.info(
            f"  → {verdict_str}: "
            f"learnable MAE={learn_mae:.6f} vs original MAE={orig_mae:.6f} "
            f"(delta={delta_vs_orig:+.6f}, {pct_vs_orig:+.1f}%)"
        )
    else:
        result["verdict"] = {"PASS": False, "reason": "arm_error"}

    return result


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dawn Learnable Trunk Gate")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["is_funded", "funding_raised_usd", "investors_count"],
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["core_only", "core_edgar"],
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 7],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/dawn_gate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_pass, n_fail, n_skip = 0, 0, 0

    for target in args.targets:
        for ablation in args.ablations:
            for horizon in args.horizons:
                result = _run_cell(target, ablation, horizon, output_dir)
                all_results.append(result)
                if result.get("status") == "skipped":
                    n_skip += 1
                elif result.get("verdict", {}).get("PASS"):
                    n_pass += 1
                else:
                    n_fail += 1

    # ── Summary ──
    summary = {
        "total": len(all_results),
        "pass": n_pass,
        "fail": n_fail,
        "skip": n_skip,
        "pass_rate": f"{n_pass}/{n_pass + n_fail}" if (n_pass + n_fail) > 0 else "N/A",
        "cells": all_results,
    }

    out_file = output_dir / "dawn_gate_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _LOG.info(f"\n{'='*60}")
    _LOG.info(f"Dawn Gate Summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    _LOG.info(f"Results: {out_file}")

    # Return exit code
    if n_pass + n_fail == 0:
        _LOG.error("No valid cells — cannot conclude")
        return 1
    return 0 if n_pass > n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
