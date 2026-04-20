#!/usr/bin/env python3
"""Battle 1: DeepNPTS Knowledge Distillation Gate Test.

Fits DeepNPTS as frozen teacher, extracts soft predictions on training data,
then trains our binary lane with KD loss at multiple alpha levels.
Compares test MAE/Brier/AUC against baseline (no KD) and incumbent (DeepNPTS).

Usage:
    python scripts/run_battle1_kd_gate.py \
        --ablations core_only core_edgar \
        --horizons 1 7 14 30 \
        --output-dir runs/benchmarks/.../battle1_kd_gate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v740_alpha_minibenchmark import (
    _build_case_frame,
    _make_temporal_config,
)
from scripts.run_v740_alpha_smoke_slice import _prepare_features
from src.narrative.block3.models.single_model_mainline.wrapper import (
    SingleModelMainlineWrapper,
)
from src.narrative.block3.models.registry import get_model

_LOG = logging.getLogger("battle1_kd")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

_TARGET = "is_funded"
_TASK = "task1_outcome"
_KD_ALPHAS = [0.0, 0.15, 0.30, 0.50, 0.70]
_MAX_ENTITIES = 5000
_MAX_ROWS = 8000


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_t, y_p)),
    }
    try:
        if len(np.unique(y_t.astype(int))) >= 2:
            metrics["auc"] = float(roc_auc_score(y_t.astype(int), y_p))
            metrics["brier"] = float(brier_score_loss(y_t.astype(int), y_p))
    except Exception:
        pass
    return metrics


# ------------------------------------------------------------------
# Teacher: fit DeepNPTS and get training soft labels
# ------------------------------------------------------------------
def _get_teacher_soft_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    horizon: int,
    ablation: str,
) -> np.ndarray:
    """Fit DeepNPTS and produce soft predictions on training features.

    Strategy: fit DeepNPTS on train, call predict(X_train) to get per-entity
    forecasts mapped back to training rows.  Clip to [0.01, 0.99].
    """
    X_train, y_train = _prepare_features(train_df, target)
    teacher_model = get_model("DeepNPTS")
    _LOG.info(f"  Fitting DeepNPTS teacher ({len(X_train)} rows) ...")
    t0 = time.time()
    teacher_model.fit(
        X_train, y_train,
        train_raw=train_df,
        val_raw=val_df,
        target=target,
        task=_TASK,
        ablation=ablation,
        horizon=horizon,
    )
    fit_s = time.time() - t0
    _LOG.info(f"  DeepNPTS teacher fitted in {fit_s:.1f}s")

    # Get teacher predictions on training data
    teacher_preds = teacher_model.predict(
        X_train,
        train_raw=train_df,
        test_raw=train_df,
        target=target,
        task=_TASK,
        ablation=ablation,
        horizon=horizon,
    )
    soft_labels = np.clip(np.asarray(teacher_preds, dtype=np.float64), 0.01, 0.99)
    _LOG.info(
        f"  Teacher soft labels: mean={soft_labels.mean():.4f}, "
        f"std={soft_labels.std():.4f}, min={soft_labels.min():.4f}, max={soft_labels.max():.4f}"
    )
    return soft_labels, teacher_model, fit_s


# ------------------------------------------------------------------
# Run one cell
# ------------------------------------------------------------------
def _run_cell(
    ablation: str,
    horizon: int,
    output_dir: Path,
) -> Dict[str, Any]:
    cell_name = f"{_TASK}__{ablation}__{_TARGET}__h{horizon}"
    _LOG.info(f"=== Cell: {cell_name} ===")

    case = {
        "task": _TASK,
        "ablation": ablation,
        "target": _TARGET,
        "horizon": horizon,
        "max_entities": _MAX_ENTITIES,
        "max_rows": _MAX_ROWS,
        "name": cell_name,
    }
    temporal_config = _make_temporal_config()
    train, val, test = _build_case_frame(case, temporal_config)
    X_train, y_train = _prepare_features(train, _TARGET)
    X_test, y_test = _prepare_features(test, _TARGET)

    if len(X_train) < 20 or len(X_test) < 10:
        _LOG.warning(f"  Skipping {cell_name}: insufficient rows")
        return {"cell": cell_name, "status": "skipped"}

    # 1. Fit teacher → get soft labels
    teacher_soft, teacher_model, teacher_fit_s = _get_teacher_soft_labels(
        train, val, _TARGET, horizon, ablation,
    )

    # 2. Get incumbent (DeepNPTS) test predictions
    incumbent_preds = teacher_model.predict(
        X_test, train_raw=train, test_raw=test,
        target=_TARGET, task=_TASK, ablation=ablation, horizon=horizon,
    )
    incumbent_metrics = _compute_metrics(y_test.to_numpy(), incumbent_preds)

    # 3. Run mainline at multiple KD alpha levels
    results_by_alpha: Dict[str, Dict] = {}
    for alpha in _KD_ALPHAS:
        label = f"alpha={alpha:.2f}"
        _LOG.info(f"  Fitting mainline with {label} ...")

        wrapper = SingleModelMainlineWrapper()
        t0 = time.time()
        fit_kwargs = dict(
            train_raw=train, val_raw=val,
            target=_TARGET, task=_TASK,
            ablation=ablation, horizon=horizon,
        )
        if alpha > 0:
            fit_kwargs["teacher_probs"] = teacher_soft
            fit_kwargs["kd_alpha"] = alpha
        wrapper.fit(X_train, y_train, **fit_kwargs)
        fit_s = time.time() - t0

        preds = wrapper.predict(
            X_test, test_raw=test,
            target=_TARGET, task=_TASK,
            ablation=ablation, horizon=horizon,
        )
        metrics = _compute_metrics(y_test.to_numpy(), preds)
        pred_std = float(np.nanstd(preds))

        results_by_alpha[label] = {
            "alpha": alpha,
            "metrics": metrics,
            "fit_seconds": fit_s,
            "prediction_std": pred_std,
            "constant": bool(pred_std < 1e-8),
        }
        _LOG.info(f"    {label}: MAE={metrics['mae']:.6f} AUC={metrics.get('auc','?')} Brier={metrics.get('brier','?')}")

    # 4. Compute deltas
    baseline_mae = results_by_alpha["alpha=0.00"]["metrics"]["mae"]
    incumbent_mae = incumbent_metrics["mae"]
    for label, res in results_by_alpha.items():
        res["mae_vs_baseline_pct"] = 100 * (res["metrics"]["mae"] - baseline_mae) / max(baseline_mae, 1e-12)
        res["mae_vs_incumbent_pct"] = 100 * (res["metrics"]["mae"] - incumbent_mae) / max(incumbent_mae, 1e-12)

    cell_result = {
        "cell": cell_name,
        "ablation": ablation,
        "horizon": horizon,
        "target": _TARGET,
        "task": _TASK,
        "status": "ok",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "incumbent_model": "DeepNPTS",
        "incumbent_metrics": incumbent_metrics,
        "incumbent_fit_seconds": teacher_fit_s,
        "results_by_alpha": results_by_alpha,
    }

    out_path = output_dir / f"{cell_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cell_result, f, indent=2, default=str)
    _LOG.info(f"  Saved → {out_path}")
    return cell_result


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def _print_summary(results: List[Dict], output_dir: Path) -> None:
    lines = [
        "# Battle 1: DeepNPTS Knowledge Distillation Gate\n",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "## Results\n",
        "| cell | incumbent MAE | baseline(α=0) | α=0.15 | α=0.30 | α=0.50 | α=0.70 | best α | best gap vs incumbent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    wins = 0
    total = 0
    for r in results:
        if r.get("status") != "ok":
            continue
        total += 1
        inc_mae = r["incumbent_metrics"]["mae"]
        row = [r["cell"], f"{inc_mae:.6f}"]
        best_alpha = None
        best_mae = float("inf")
        for label in [f"alpha={a:.2f}" for a in _KD_ALPHAS]:
            res = r["results_by_alpha"].get(label, {})
            mae = res.get("metrics", {}).get("mae", float("nan"))
            row.append(f"{mae:.6f}")
            if mae < best_mae:
                best_mae = mae
                best_alpha = label
        gap_vs_inc = 100 * (best_mae - inc_mae) / max(inc_mae, 1e-12)
        row.append(best_alpha or "?")
        row.append(f"{gap_vs_inc:+.2f}%")
        if best_mae <= inc_mae * 1.001:
            wins += 1
        lines.append("| " + " | ".join(row) + " |")

    lines.append(f"\n## Verdict\n")
    lines.append(f"- W/L vs incumbent: **{wins}/{total}** (need ≥50% to pass)\n")
    if wins >= total // 2:
        lines.append("- **PASS**: KD successfully narrows the gap to DeepNPTS\n")
    else:
        lines.append("- **FAIL**: KD does not sufficiently close the gap\n")

    summary_md = "\n".join(lines)
    summary_path = output_dir / "battle1_summary.md"
    summary_path.write_text(summary_md)
    print(summary_md)
    _LOG.info(f"Summary → {summary_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Battle 1: KD gate test")
    parser.add_argument("--ablations", nargs="+", default=["core_only", "core_edgar"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 14, 30])
    parser.add_argument("--output-dir", default="runs/benchmarks/single_model_mainline_localclear_20260420/battle1_kd_gate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ablation in args.ablations:
        for horizon in args.horizons:
            try:
                r = _run_cell(ablation, horizon, output_dir)
                results.append(r)
            except Exception as exc:
                _LOG.error(f"FAILED {ablation}/h{horizon}: {exc}", exc_info=True)
                results.append({
                    "cell": f"{_TASK}__{ablation}__{_TARGET}__h{horizon}",
                    "status": "error",
                    "error": str(exc),
                })

    _print_summary(results, output_dir)


if __name__ == "__main__":
    main()
