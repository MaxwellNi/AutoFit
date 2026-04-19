#!/usr/bin/env python3
"""P5 Adaptive Shrinkage Gate for investors lane.

Provides per-sample uncertainty-driven shrinkage toward anchor, inspired by:
- James-Stein shrinkage (Efron & Morris 1973)
- ASH adaptive shrinkage (Stephens 2016)
- Nash neural adaptive shrinkage (Denault 2025)

The key insight: when the learned prediction's residual has high local variance,
shrink toward the anchor (historical mean) to reduce MSE.  This is the lane-level
analog of "abstain when uncertain" but uses soft shrinkage instead of hard cutoff.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def fit_shrinkage_gate(
    design: np.ndarray,
    learned_pred: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    *,
    random_state: int = 0,
    n_neighbors: int = 30,
) -> tuple[HistGradientBoostingRegressor | None, dict[str, float]]:
    """Fit a model that predicts per-sample optimal shrinkage toward anchor.

    For each sample i, the oracle shrinkage is:
        alpha_i* = argmin_alpha  (alpha * anchor_i + (1-alpha) * learned_i - target_i)^2
                 = (learned_i - target_i)(learned_i - anchor_i)
                   / (learned_i - anchor_i)^2

    We clip alpha to [0, 1] and train a gradient boosted regressor to predict it
    from the design matrix.

    Returns (model, diagnostics) where model is None if fitting failed.
    """
    learned = np.asarray(learned_pred, dtype=np.float64).ravel()
    anch = np.asarray(anchor, dtype=np.float64).ravel()
    tgt = np.asarray(target, dtype=np.float64).ravel()
    n = len(learned)

    if n < 24:
        return None, {"shrinkage_status": "too_few_samples", "n_samples": n}

    diff = learned - anch
    denom = diff ** 2
    numer = (learned - tgt) * diff

    # Avoid division by zero: if learned == anchor, shrinkage is irrelevant
    safe_mask = denom > 1e-12
    if safe_mask.sum() < 16:
        return None, {"shrinkage_status": "learned_equals_anchor", "safe_count": int(safe_mask.sum())}

    oracle_alpha = np.zeros(n, dtype=np.float64)
    oracle_alpha[safe_mask] = np.clip(numer[safe_mask] / denom[safe_mask], 0.0, 1.0)

    # Smooth oracle alpha with local averaging for stability
    alpha_std = float(np.std(oracle_alpha[safe_mask]))
    if alpha_std < 1e-8:
        mean_alpha = float(np.mean(oracle_alpha[safe_mask]))
        return None, {
            "shrinkage_status": "constant_oracle",
            "mean_oracle_alpha": mean_alpha,
        }

    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        min_samples_leaf=max(8, n // 20),
        random_state=random_state,
    )
    model.fit(design[safe_mask], oracle_alpha[safe_mask])

    # Diagnostics: train-set shrinkage effect
    pred_alpha = np.clip(model.predict(design), 0.0, 1.0)
    shrunk_pred = (1.0 - pred_alpha) * learned + pred_alpha * anch
    base_mae = float(np.mean(np.abs(learned - tgt)))
    shrunk_mae = float(np.mean(np.abs(shrunk_pred - tgt)))
    anchor_mae = float(np.mean(np.abs(anch - tgt)))

    diagnostics = {
        "shrinkage_status": "converged",
        "n_safe_samples": int(safe_mask.sum()),
        "mean_oracle_alpha": float(np.mean(oracle_alpha[safe_mask])),
        "std_oracle_alpha": alpha_std,
        "mean_pred_alpha": float(np.mean(pred_alpha)),
        "base_mae": base_mae,
        "shrunk_mae": shrunk_mae,
        "anchor_mae": anchor_mae,
        "shrinkage_improvement_pct": (base_mae - shrunk_mae) / max(base_mae, 1e-12) * 100.0,
    }
    return model, diagnostics


def predict_shrinkage_alpha(
    model: HistGradientBoostingRegressor | None,
    design: np.ndarray,
    default_alpha: float = 0.0,
) -> np.ndarray:
    """Predict per-sample shrinkage coefficients.

    Returns array of alpha in [0, 1] where:
    - 0 = fully trust learned prediction
    - 1 = fully trust anchor
    """
    n = len(design)
    if model is None:
        return np.full(n, default_alpha, dtype=np.float64)
    raw = model.predict(design).astype(np.float64, copy=False)
    return np.clip(raw, 0.0, 1.0)


def apply_shrinkage(
    learned_pred: np.ndarray,
    anchor: np.ndarray,
    alpha: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply shrinkage: result = (1 - effective_alpha) * learned + effective_alpha * anchor.

    strength in [0, 1] scales the shrinkage effect:
    - strength=0: no shrinkage (return learned)
    - strength=1: full adaptive shrinkage
    """
    s = float(np.clip(strength, 0.0, 1.0))
    effective = np.asarray(alpha, dtype=np.float64) * s
    return (1.0 - effective) * np.asarray(learned_pred, dtype=np.float64) + effective * np.asarray(anchor, dtype=np.float64)


def shrinkage_diagnostics(alpha: np.ndarray) -> dict[str, float]:
    """Summary statistics for shrinkage alpha distribution."""
    a = np.asarray(alpha, dtype=np.float64)
    return {
        "shrinkage_mean_alpha": float(np.mean(a)),
        "shrinkage_median_alpha": float(np.median(a)),
        "shrinkage_std_alpha": float(np.std(a)),
        "shrinkage_frac_gt_0.5": float(np.mean(a > 0.5)),
        "shrinkage_frac_lt_0.1": float(np.mean(a < 0.1)),
    }
