#!/usr/bin/env python3
"""Battle 3: The Physical Consistency Audit — ED-SSM vs DeepNPTS.

Runs the full multi-task VC evaluation with ED-SSM + MoE enabled, then
executes the economic constraint audit to measure:
  1. Ghost Funding:    investors ≈ 0 but funding > 0
  2. Logical Inversion: P(funded) ≥ 90% but funding in bottom-10%

Compares violation rates between ED-SSM mainline and DeepNPTS baseline.

Usage:
    python scripts/run_edssm_battle3_audit.py \\
        --benchmark DeepNPTS \\
        --ablations core_only core_edgar \\
        --horizons 1 7 \\
        --output-dir runs/edssm_battle3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _prepare_features
from src.narrative.block3.models.registry import check_model_available, get_model
from src.narrative.block3.models.single_model_mainline.wrapper import SingleModelMainlineWrapper

_LOG = logging.getLogger("battle3_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

TARGETS = ["is_funded", "funding_raised_usd", "investors_count"]
_TASK_MAP = {
    "is_funded": "task1_outcome",
    "funding_raised_usd": "task2_forecast",
    "investors_count": "task3_risk_adjust",
}
_MAX_ENTITIES = 5000
_MAX_ROWS = 8000

# Violation thresholds (same as audit_economic_constraints.py)
_GHOST_INVESTOR_THRESHOLD = 0.5
_INVERSION_PROB_THRESHOLD = 0.90
_INVERSION_FUNDING_QUANTILE = 0.10


def detect_ghost_funding(pred_investors, pred_funding):
    return (pred_investors < _GHOST_INVESTOR_THRESHOLD) & (pred_funding > 0.0)


def detect_logical_inversion(pred_binary, pred_funding, funding_q10):
    return (pred_binary >= _INVERSION_PROB_THRESHOLD) & (pred_funding <= funding_q10)


def _fit_predict_all_targets(
    wrapper_factory,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    ablation: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Fit 3 wrappers (one per target), return {target: predictions}."""
    results = {}
    for target in TARGETS:
        X_train, y_train = _prepare_features(train, target)
        X_test, _ = _prepare_features(test, target)
        if len(X_train) < 10 or len(X_test) < 10:
            results[target] = np.full(len(test), np.nan)
            continue
        wrapper = wrapper_factory()
        wrapper.fit(
            X_train, y_train,
            train_raw=train, val_raw=val,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        preds = wrapper.predict(
            X_test, test_raw=test,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        results[target] = np.asarray(preds, dtype=np.float64)
    return results


def _fit_predict_benchmark(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    ablation: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Fit benchmark model for all targets."""
    results = {}
    for target in TARGETS:
        X_train, y_train = _prepare_features(train, target)
        X_test, _ = _prepare_features(test, target)
        if len(X_train) < 10 or len(X_test) < 10:
            results[target] = np.full(len(test), np.nan)
            continue
        model = get_model(model_name)
        model.fit(
            X_train, y_train,
            train_raw=train, val_raw=val,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        preds = model.predict(
            X_test, test_raw=test,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        results[target] = np.asarray(preds, dtype=np.float64)
    return results


def _compute_violations(
    preds: Dict[str, np.ndarray],
    funding_q10: float,
) -> Dict[str, Any]:
    """Compute ghost funding and logical inversion rates."""
    p_binary = preds["is_funded"]
    p_funding = preds["funding_raised_usd"]
    p_investors = preds["investors_count"]

    # Align to shortest prediction vector (different targets may have different NaN masks)
    min_len = min(len(p_binary), len(p_funding), len(p_investors))
    p_binary = p_binary[:min_len]
    p_funding = p_funding[:min_len]
    p_investors = p_investors[:min_len]

    n_valid = int(np.isfinite(p_binary).sum())
    ghost = detect_ghost_funding(p_investors, p_funding)
    inversion = detect_logical_inversion(p_binary, p_funding, funding_q10)

    return {
        "ghost_count": int(ghost.sum()),
        "ghost_rate": round(float(ghost.sum()) / max(n_valid, 1), 4),
        "inversion_count": int(inversion.sum()),
        "inversion_rate": round(float(inversion.sum()) / max(n_valid, 1), 4),
        "pred_binary_mean": round(float(np.nanmean(p_binary)), 4),
        "pred_funding_mean": round(float(np.nanmean(p_funding)), 2),
        "pred_investors_mean": round(float(np.nanmean(p_investors)), 4),
        "n_valid": n_valid,
    }


def run_audit_cell(
    ablation: str,
    horizon: int,
    benchmark_model: str,
    seq_decoder_branch: str,
) -> Dict[str, Any]:
    """Run full audit on one (ablation, horizon) cell."""
    task = "task1_outcome"  # Any task works; we run all 3 targets
    cell_name = f"{ablation}__h{horizon}"
    _LOG.info(f"\n{'='*60}")
    _LOG.info(f"  Auditing: {cell_name}")
    _LOG.info(f"{'='*60}")

    case = {
        "task": task, "ablation": ablation, "target": "is_funded",
        "horizon": horizon, "max_entities": _MAX_ENTITIES,
        "max_rows": _MAX_ROWS, "name": cell_name,
    }
    temporal_config = _make_temporal_config()
    train, val, test = _build_case_frame(case, temporal_config)

    # Funding Q10 from training data
    _, y_fund_train = _prepare_features(train, "funding_raised_usd")
    funding_q10 = float(np.nanquantile(y_fund_train.to_numpy(), _INVERSION_FUNDING_QUANTILE))

    cell_result = {"cell": cell_name, "test_rows": len(test), "funding_q10": funding_q10}

    # ── ED-SSM mainline ──
    _LOG.info("  [ED-SSM Mainline] fitting 3 targets ...")
    t0 = time.time()
    edssm_preds = _fit_predict_all_targets(
        lambda: SingleModelMainlineWrapper(
            enable_sequential_trunk=True,
            learnable_compact_dim=64,
            learnable_n_experts=6,
            learnable_expert_dim=32,
            learnable_top_k=2,
            seq_window_size=30,
            seq_d_model=64,
            seq_n_epochs=20,
            seq_decoder_branch=seq_decoder_branch,
            seq_freeze_unified_ssm=True,
        ),
        train, val, test, task, ablation, horizon,
    )
    edssm_time = time.time() - t0
    edssm_viol = _compute_violations(edssm_preds, funding_q10)
    cell_result["edssm_mainline"] = {**edssm_viol, "time_seconds": edssm_time}
    _LOG.info(f"    Ghost: {edssm_viol['ghost_count']} ({edssm_viol['ghost_rate']:.2%})")
    _LOG.info(f"    Inversion: {edssm_viol['inversion_count']} ({edssm_viol['inversion_rate']:.2%})")

    # ── Benchmark (DeepNPTS etc.) ──
    _LOG.info(f"  [{benchmark_model}] fitting 3 targets ...")
    t0 = time.time()
    try:
        bench_preds = _fit_predict_benchmark(
            benchmark_model, train, val, test, task, ablation, horizon,
        )
        bench_time = time.time() - t0
        bench_viol = _compute_violations(bench_preds, funding_q10)
        cell_result[benchmark_model] = {**bench_viol, "time_seconds": bench_time}
        _LOG.info(f"    Ghost: {bench_viol['ghost_count']} ({bench_viol['ghost_rate']:.2%})")
        _LOG.info(f"    Inversion: {bench_viol['inversion_count']} ({bench_viol['inversion_rate']:.2%})")
    except Exception as e:
        _LOG.error(f"    {benchmark_model} FAILED: {e}")
        cell_result[benchmark_model] = {"error": str(e)}

    # ── Verdict ──
    if benchmark_model in cell_result and "ghost_rate" in cell_result[benchmark_model]:
        e_total = edssm_viol["ghost_rate"] + edssm_viol["inversion_rate"]
        b_total = cell_result[benchmark_model]["ghost_rate"] + cell_result[benchmark_model]["inversion_rate"]
        cell_result["verdict"] = {
            "edssm_total_violation": round(e_total, 4),
            "bench_total_violation": round(b_total, 4),
            "PASS": e_total < b_total,
        }
        v = "✓ PASS" if cell_result["verdict"]["PASS"] else "✗ FAIL"
        _LOG.info(f"  → {v}: ED-SSM violations={e_total:.4f} vs {benchmark_model}={b_total:.4f}")
    else:
        cell_result["verdict"] = {"PASS": False, "reason": "benchmark_error"}

    return cell_result


def main():
    parser = argparse.ArgumentParser(description="Battle 3: Physical Consistency Audit")
    parser.add_argument("--benchmark", default="DeepNPTS")
    parser.add_argument("--ablations", nargs="+", default=["core_only", "core_edgar"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7])
    parser.add_argument(
        "--seq-decoder-branch",
        type=str,
        default="legacy",
        choices=["legacy", "alpha", "beta", "gamma", "icm_lognormal", "icm_iqn", "icm_cfm"],
        help="Sequential trunk auxiliary decoder branch",
    )
    parser.add_argument("--output-dir", default="runs/edssm_battle3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_pass, n_fail = 0, 0
    t_start = time.time()

    for ablation in args.ablations:
        for horizon in args.horizons:
            result = run_audit_cell(
                ablation,
                horizon,
                args.benchmark,
                seq_decoder_branch=args.seq_decoder_branch,
            )
            all_results.append(result)
            if result.get("verdict", {}).get("PASS"):
                n_pass += 1
            else:
                n_fail += 1

    total_time = time.time() - t_start

    # Aggregate violation rates
    edssm_ghost_total = sum(r.get("edssm_mainline", {}).get("ghost_count", 0) for r in all_results)
    edssm_inv_total = sum(r.get("edssm_mainline", {}).get("inversion_count", 0) for r in all_results)
    total_rows = sum(r.get("test_rows", 0) for r in all_results)

    summary = {
        "battle": "3_physical_consistency_audit",
        "benchmark": args.benchmark,
        "total": len(all_results),
        "pass": n_pass,
        "fail": n_fail,
        "aggregate_edssm_ghost_rate": round(edssm_ghost_total / max(total_rows, 1), 4),
        "aggregate_edssm_inversion_rate": round(edssm_inv_total / max(total_rows, 1), 4),
        "total_seconds": total_time,
        "cells": all_results,
    }

    out_file = output_dir / "battle3_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _LOG.info(f"\n{'='*60}")
    _LOG.info(f"Battle 3 — Physical Consistency Audit vs {args.benchmark}")
    _LOG.info(f"  {n_pass} PASS / {n_fail} FAIL")
    _LOG.info(f"  ED-SSM Ghost Rate: {edssm_ghost_total}/{total_rows}")
    _LOG.info(f"  ED-SSM Inversion Rate: {edssm_inv_total}/{total_rows}")
    _LOG.info(f"  Total time: {total_time:.1f}s")
    _LOG.info(f"  Results: {out_file}")
    _LOG.info(f"{'='*60}")


if __name__ == "__main__":
    main()
