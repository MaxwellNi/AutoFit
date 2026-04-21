#!/usr/bin/env python3
"""Battle 1: The Revenge of the Trunk — ED-SSM + MoE Ablation Gate.

Three-arm comparison on each evaluation cell:
  1. Original    — random projection backbone + barrier (incumbent)
  2. ED-SSM+MoE  — SequentialMoETrunk (sequential temporal trunk)

PASS: ED-SSM+MoE MAE < Original MAE on majority of cells.

Usage:
    python scripts/run_edssm_battle1_revenge.py \\
        --targets is_funded funding_raised_usd investors_count \\
        --ablations core_only core_edgar \\
        --horizons 1 7 \\
        --output-dir runs/edssm_battle1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v740_alpha_minibenchmark import (
    _build_case_frame,
    _make_temporal_config,
)
from scripts.run_v740_alpha_smoke_slice import _prepare_features
from src.narrative.block3.models.single_model_mainline.wrapper import (
    SingleModelMainlineWrapper,
)

_LOG = logging.getLogger("battle1_revenge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

_TASK_MAP = {
    "is_funded": "task1_outcome",
    "funding_raised_usd": "task2_forecast",
    "investors_count": "task3_risk_adjust",
}
_MAX_ENTITIES = 5000
_MAX_ROWS = 8000


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error

    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "pred_std": float(np.nanstd(y_p)),
        "pred_mean": float(np.nanmean(y_p)),
    }
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


def _run_cell(
    target: str,
    ablation: str,
    horizon: int,
    output_dir: Path,
    seq_decoder_branch: str,
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

    # ── Arm 1: Original (random projection backbone + barrier) ──
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
        _LOG.error(f"    Original FAILED: {e}", exc_info=True)
        result["original"] = {"error": str(e)}

    # ── Arm 2: ED-SSM + MoE Sequential Trunk ──
    _LOG.info("  [Arm 2] ED-SSM + MoE Sequential Trunk ...")
    try:
        w_edssm = SingleModelMainlineWrapper(
            enable_sequential_trunk=True,
            learnable_compact_dim=64,
            learnable_n_experts=6,
            learnable_expert_dim=32,
            learnable_top_k=2,
            seq_window_size=30,
            seq_d_model=64,
            seq_d_state=16,
            seq_n_ssm_layers=2,
            seq_n_epochs=20,
            seq_batch_size=512,
            seq_decoder_branch=seq_decoder_branch,
            seq_freeze_unified_ssm=False,
        )
        t0 = time.time()
        w_edssm.fit(X_train, y_train, **fit_kwargs_base)
        edssm_fit_s = time.time() - t0
        preds_edssm = w_edssm.predict(X_test, **predict_kwargs_base)
        edssm_metrics = _compute_metrics(y_test.to_numpy(), preds_edssm)
        result["edssm"] = {
            "metrics": edssm_metrics,
            "fit_seconds": edssm_fit_s,
            "trunk_desc": w_edssm._sequential_trunk.describe() if w_edssm._sequential_trunk else {},
        }
        _LOG.info(f"    ED-SSM: MAE={edssm_metrics['mae']:.6f}")
    except Exception as e:
        _LOG.error(f"    ED-SSM FAILED: {e}", exc_info=True)
        result["edssm"] = {"error": str(e)}

    # ── Verdict ──
    if "metrics" in result.get("original", {}) and "metrics" in result.get("edssm", {}):
        orig_mae = result["original"]["metrics"]["mae"]
        edssm_mae = result["edssm"]["metrics"]["mae"]
        delta = edssm_mae - orig_mae
        pct = (delta / orig_mae * 100) if orig_mae > 0 else 0

        result["verdict"] = {
            "edssm_vs_orig_delta": delta,
            "edssm_vs_orig_pct": pct,
            "PASS": bool(edssm_mae < orig_mae),
        }
        verdict_str = "✓ PASS" if result["verdict"]["PASS"] else "✗ FAIL"
        _LOG.info(
            f"  → {verdict_str}: "
            f"ED-SSM MAE={edssm_mae:.6f} vs Original MAE={orig_mae:.6f} "
            f"(delta={delta:+.6f}, {pct:+.1f}%)"
        )
    else:
        result["verdict"] = {"PASS": False, "reason": "arm_error"}

    return result


def main():
    parser = argparse.ArgumentParser(description="Battle 1: The Revenge of the Trunk")
    parser.add_argument(
        "--targets", nargs="+",
        default=["is_funded", "funding_raised_usd", "investors_count"],
    )
    parser.add_argument(
        "--ablations", nargs="+",
        default=["core_only", "core_edgar"],
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int,
        default=[1, 7],
    )
    parser.add_argument(
        "--seq-decoder-branch",
        type=str,
        default="legacy",
        choices=["legacy", "alpha", "beta", "gamma", "icm_lognormal", "icm_iqn", "icm_cfm"],
        help="Sequential trunk auxiliary decoder branch",
    )
    parser.add_argument("--output-dir", type=str, default="runs/edssm_battle1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_pass, n_fail, n_skip = 0, 0, 0
    t_start = time.time()

    for target in args.targets:
        for ablation in args.ablations:
            for horizon in args.horizons:
                result = _run_cell(
                    target,
                    ablation,
                    horizon,
                    output_dir,
                    seq_decoder_branch=args.seq_decoder_branch,
                )
                all_results.append(result)
                if result.get("status") == "skipped":
                    n_skip += 1
                elif result.get("verdict", {}).get("PASS"):
                    n_pass += 1
                else:
                    n_fail += 1

    total_time = time.time() - t_start

    summary = {
        "battle": "1_trunk_ablation_revenge",
        "total": len(all_results),
        "pass": n_pass,
        "fail": n_fail,
        "skip": n_skip,
        "pass_rate": f"{n_pass}/{n_pass + n_fail}" if (n_pass + n_fail) > 0 else "N/A",
        "total_seconds": total_time,
        "cells": all_results,
    }

    out_file = output_dir / "battle1_revenge_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _LOG.info(f"\n{'='*60}")
    _LOG.info(f"Battle 1 — The Revenge of the Trunk")
    _LOG.info(f"  {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    _LOG.info(f"  Total time: {total_time:.1f}s")
    _LOG.info(f"  Results: {out_file}")
    _LOG.info(f"{'='*60}")


if __name__ == "__main__":
    main()
