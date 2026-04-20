#!/usr/bin/env python3
"""Battle 2: Trunk Ablation Audit — "假冠军" 剥离审查.

Tests whether the Shared Trunk actually contributes learned representations,
or whether downstream heads (HGBR/HGBC) could achieve the same results
on raw R²-IN-standardized features (bypassing backbone + barrier).

For each cell × target:
  1. Full pipeline: backbone → barrier → lane → predict
  2. No-trunk baseline: R²-IN(raw features) → lane → predict
  3. Compare: if NoTrunk ≈ FullPipeline, trunk is not learning anything useful

Usage:
    python scripts/run_battle2_trunk_ablation.py \
        --ablations core_only core_edgar \
        --horizons 1 7 14 30 \
        --output-dir runs/benchmarks/.../battle2_trunk_ablation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

_LOG = logging.getLogger("battle2_trunk")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

_TARGETS = ["is_funded", "funding_raised_usd", "investors_count"]
_TASK = "task1_outcome"
_MAX_ENTITIES = 5000
_MAX_ROWS = 8000


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_binary: bool = False) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error

    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    metrics: Dict[str, float] = {"mae": float(mean_absolute_error(y_t, y_p))}
    if is_binary:
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            y_p_clip = np.clip(y_p, 0.0, 1.0)
            y_int = y_t.astype(int)
            if len(np.unique(y_int)) >= 2:
                metrics["auc"] = float(roc_auc_score(y_int, y_p_clip))
                metrics["brier"] = float(brier_score_loss(y_int, y_p_clip))
        except Exception:
            pass
    else:
        metrics["rmse"] = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    return metrics


# ------------------------------------------------------------------
# No-Trunk baseline: R²-IN standardize raw features, feed to lane
# ------------------------------------------------------------------
def _robust_standardize(X: np.ndarray) -> np.ndarray:
    """R²-IN standardization: (X - median) / (1.4826 * MAD)."""
    X = np.asarray(X, dtype=np.float64)
    loc = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - loc), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (X - loc) / scale


def _fit_predict_no_trunk(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    target: str,
    ablation: str,
    horizon: int,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    """Fit and predict using a mainline wrapper with trunk bypass.

    Strategy: create wrapper normally, but override the backbone transform
    to return raw R²-IN-standardized features instead of the full trunk output.
    This tests whether the trunk's random projections, summaries, and
    temporal/spectral features actually contribute to downstream lane quality.
    """
    # Build raw standardized features
    raw_train = X_train.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float64)
    raw_test = X_test.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float64)

    # Fit R²-IN on training data
    loc = np.nanmedian(raw_train, axis=0)
    mad = np.nanmedian(np.abs(raw_train - loc), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-6, 1.0, scale)

    std_train = (raw_train - loc) / scale
    std_test = (raw_test - loc) / scale

    # Use the appropriate downstream model directly
    is_binary = target == "is_funded"
    t0 = time.time()

    if is_binary:
        from src.narrative.block3.models.single_model_mainline.lanes.binary_lane import (
            BinaryLaneRuntime,
        )
        lane = BinaryLaneRuntime(random_state=42)
        y_arr = (np.asarray(y_train.to_numpy(), dtype=np.float64) > 0.5).astype(np.float64)
        lane.fit(
            lane_state=std_train.astype(np.float32),
            y=y_arr,
            aux_features=None,
            horizon=horizon,
        )
        preds = lane.predict(
            lane_state=std_test.astype(np.float32),
            aux_features=None,
        )
    elif target == "funding_raised_usd":
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        y_arr = np.asarray(y_train.to_numpy(), dtype=np.float64)
        anchor = np.zeros_like(y_arr)
        # Simple hurdle: event classifier + severity regressor
        has_funding = (y_arr > 0).astype(int)
        event_model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05, random_state=42,
        )
        if len(np.unique(has_funding)) >= 2:
            event_model.fit(std_train, has_funding)
            event_prob_test = event_model.predict_proba(std_test)[:, 1]
        else:
            event_prob_test = np.full(len(std_test), float(has_funding.mean()))

        pos_mask = y_arr > 0
        if pos_mask.sum() >= 5:
            severity_model = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05, random_state=42,
            )
            severity_model.fit(std_train[pos_mask], np.log1p(y_arr[pos_mask]))
            severity_test = np.expm1(severity_model.predict(std_test))
        else:
            severity_test = np.full(len(std_test), float(np.mean(y_arr[pos_mask])) if pos_mask.any() else 0.0)

        preds = event_prob_test * np.maximum(severity_test, 0.0)
    else:
        # investors_count: occurrence + positive regression
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        y_arr = np.asarray(y_train.to_numpy(), dtype=np.float64)
        has_investors = (y_arr > 0).astype(int)
        occurrence_model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05, random_state=42,
        )
        if len(np.unique(has_investors)) >= 2:
            occurrence_model.fit(std_train, has_investors)
            occ_prob_test = occurrence_model.predict_proba(std_test)[:, 1]
        else:
            occ_prob_test = np.full(len(std_test), float(has_investors.mean()))

        pos_mask = y_arr > 0
        if pos_mask.sum() >= 5:
            pos_model = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05, random_state=42,
            )
            pos_model.fit(std_train[pos_mask], np.log1p(y_arr[pos_mask]))
            pos_test = np.expm1(pos_model.predict(std_test))
        else:
            pos_test = np.full(len(std_test), float(np.mean(y_arr[pos_mask])) if pos_mask.any() else 0.0)

        preds = occ_prob_test * np.maximum(pos_test, 0.0)

    fit_s = time.time() - t0
    return np.asarray(preds, dtype=np.float64), fit_s


# ------------------------------------------------------------------
# Full pipeline baseline
# ------------------------------------------------------------------
def _fit_predict_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    target: str,
    ablation: str,
    horizon: int,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
    wrapper = SingleModelMainlineWrapper()
    t0 = time.time()
    wrapper.fit(
        X_train, y_train,
        train_raw=train_raw, val_raw=val_raw,
        target=target, task=_TASK,
        ablation=ablation, horizon=horizon,
    )
    fit_s = time.time() - t0
    preds = wrapper.predict(
        X_test, test_raw=test_raw,
        target=target, task=_TASK,
        ablation=ablation, horizon=horizon,
    )
    return np.asarray(preds, dtype=np.float64), fit_s


# ------------------------------------------------------------------
# Run one cell
# ------------------------------------------------------------------
def _run_cell(
    ablation: str,
    horizon: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    cell_results = []

    for target in _TARGETS:
        cell_name = f"{_TASK}__{ablation}__{target}__h{horizon}"
        _LOG.info(f"=== Cell: {cell_name} ===")

        case = {
            "task": _TASK, "ablation": ablation, "target": target,
            "horizon": horizon, "max_entities": _MAX_ENTITIES,
            "max_rows": _MAX_ROWS, "name": cell_name,
        }
        temporal_config = _make_temporal_config()
        train, val, test = _build_case_frame(case, temporal_config)
        X_train, y_train = _prepare_features(train, target)
        X_test, y_test = _prepare_features(test, target)

        if len(X_train) < 20 or len(X_test) < 10:
            _LOG.warning(f"  Skipping {cell_name}: insufficient rows")
            cell_results.append({"cell": cell_name, "status": "skipped"})
            continue

        is_binary = target == "is_funded"

        # 1. Full pipeline
        _LOG.info(f"  Full pipeline ...")
        full_preds, full_fit_s = _fit_predict_full_pipeline(
            X_train, y_train, X_test, target, ablation, horizon, train, val, test,
        )
        full_metrics = _compute_metrics(y_test.to_numpy(), full_preds, is_binary)

        # 2. No-trunk baseline
        _LOG.info(f"  No-trunk baseline ...")
        notrunk_preds, notrunk_fit_s = _fit_predict_no_trunk(
            X_train, y_train, X_test, target, ablation, horizon, train, val, test,
        )
        notrunk_metrics = _compute_metrics(y_test.to_numpy(), notrunk_preds, is_binary)

        # 3. Compute delta
        full_mae = full_metrics["mae"]
        notrunk_mae = notrunk_metrics["mae"]
        trunk_delta_pct = 100 * (full_mae - notrunk_mae) / max(notrunk_mae, 1e-12)

        verdict = "trunk_helps" if full_mae < notrunk_mae * 0.99 else (
            "trunk_neutral" if full_mae < notrunk_mae * 1.01 else "trunk_hurts"
        )

        result = {
            "cell": cell_name, "target": target, "ablation": ablation,
            "horizon": horizon, "task": _TASK, "status": "ok",
            "train_rows": len(X_train), "test_rows": len(X_test),
            "full_pipeline": {"metrics": full_metrics, "fit_seconds": full_fit_s},
            "no_trunk": {"metrics": notrunk_metrics, "fit_seconds": notrunk_fit_s},
            "trunk_delta_mae_pct": trunk_delta_pct,
            "verdict": verdict,
        }
        cell_results.append(result)

        out_path = output_dir / f"{cell_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        _LOG.info(
            f"  Full MAE={full_mae:.6f} | NoTrunk MAE={notrunk_mae:.6f} | "
            f"delta={trunk_delta_pct:+.2f}% | {verdict}"
        )

    return cell_results


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def _print_summary(all_results: List[Dict], output_dir: Path) -> None:
    lines = [
        "# Battle 2: Trunk Ablation Audit — 假冠军剥离审查\n",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "## Results\n",
        "| cell | target | Full MAE | NoTrunk MAE | delta% | verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    helps = neutral = hurts = 0
    by_target: Dict[str, List[float]] = {}
    for r in all_results:
        if r.get("status") != "ok":
            continue
        target = r["target"]
        full_mae = r["full_pipeline"]["metrics"]["mae"]
        notrunk_mae = r["no_trunk"]["metrics"]["mae"]
        delta = r["trunk_delta_mae_pct"]
        verdict = r["verdict"]
        lines.append(f"| {r['cell']} | {target} | {full_mae:.6f} | {notrunk_mae:.6f} | {delta:+.2f}% | {verdict} |")
        if verdict == "trunk_helps":
            helps += 1
        elif verdict == "trunk_neutral":
            neutral += 1
        else:
            hurts += 1
        by_target.setdefault(target, []).append(delta)

    lines.append(f"\n## Per-Target Summary\n")
    for t, deltas in by_target.items():
        med = float(np.median(deltas))
        lines.append(f"- **{t}**: median delta = {med:+.2f}% ({len(deltas)} cells)")

    total = helps + neutral + hurts
    lines.append(f"\n## Verdict\n")
    lines.append(f"- Trunk helps: {helps}/{total}")
    lines.append(f"- Trunk neutral: {neutral}/{total}")
    lines.append(f"- Trunk hurts: {hurts}/{total}")
    if helps > hurts:
        lines.append(f"\n**PASS**: Trunk provides genuine value (helps > hurts)\n")
    elif helps == 0 and hurts == 0:
        lines.append(f"\n**WARNING**: Trunk is neutral everywhere — may be doing nothing\n")
    else:
        lines.append(f"\n**FAIL**: Trunk does NOT provide clear value — consider restructuring\n")

    summary_md = "\n".join(lines)
    summary_path = output_dir / "battle2_summary.md"
    summary_path.write_text(summary_md)
    print(summary_md)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Battle 2: Trunk ablation audit")
    parser.add_argument("--ablations", nargs="+", default=["core_only", "core_edgar"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 14, 30])
    parser.add_argument("--output-dir", default="runs/benchmarks/single_model_mainline_localclear_20260420/battle2_trunk_ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ablation in args.ablations:
        for horizon in args.horizons:
            try:
                cell_results = _run_cell(ablation, horizon, output_dir)
                all_results.extend(cell_results)
            except Exception as exc:
                _LOG.error(f"FAILED {ablation}/h{horizon}: {exc}", exc_info=True)

    _print_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
