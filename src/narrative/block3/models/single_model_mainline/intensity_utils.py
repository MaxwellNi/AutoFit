#!/usr/bin/env python3
"""Lightweight neural intensity utilities for investors lane marked TPP.

Implements a conditional intensity model λ(t|H_t) for investor event arrival:

  λ(t|H_t) = μ(x) + Σ_j α_j · exp(-δ_j · (t - t_j))

where:
  - μ(x) is a learnable baseline intensity conditioned on covariates x
  - α_j, δ_j are mark-dependent triggering parameters
  - t_j are historical event arrival times

Reference:
  - Du et al., "Recurrent Marked Temporal Point Processes" (KDD 2016)
  - Mei & Eisner, "The Neural Hawkes Process" (NeurIPS 2017)
  - Zuo et al., "Transformer Hawkes Process" (ICML 2020)

This module provides a simplified, non-neural implementation suitable for
integration into the HGBT-based investors lane without requiring GPU training.
The intensity is estimated via a parametric Hawkes kernel + HGBT baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def hawkes_intensity_features(
    aux_features: np.ndarray | None,
    anchor: np.ndarray,
    *,
    decay_scales: tuple[float, ...] = (7.0, 14.0, 30.0),
) -> np.ndarray:
    """Build intensity-inspired features from auxiliary history signals.

    Extracts multi-scale exponential decay features from the history
    components in aux_features (lag1, roll3, roll7, roll_std, history_count).

    Parameters
    ----------
    aux_features : (n, k) or None
        Auxiliary features where first 5 columns are
        [lag1, roll3, roll7, roll_std, history_count].
    anchor : (n,)
        Anchor prediction (baseline rate proxy).
    decay_scales : tuple of floats
        Time scales for exponential kernels.

    Returns
    -------
    intensity_features : (n, n_features) float32
        Columns: [baseline_log, excitation_7, excitation_14, excitation_30,
                  history_regularity, burst_indicator, arrival_rate_proxy]
    """
    n = anchor.shape[0]
    anchor_safe = np.maximum(np.asarray(anchor, dtype=np.float64), 1e-8)

    if aux_features is None or aux_features.shape[1] < 5:
        return np.zeros((n, 7), dtype=np.float32)

    aux = np.asarray(aux_features, dtype=np.float64)
    lag1 = aux[:, 0]
    roll3 = aux[:, 1]
    roll7 = aux[:, 2]
    roll_std = aux[:, 3]
    history_count = aux[:, 4]

    # Baseline log-intensity
    baseline_log = np.log1p(anchor_safe)

    # Multi-scale excitation: proxy for Σ α·exp(-δ·Δt) using rolling averages
    # Higher rolling average relative to anchor => recent excitation
    excitation_features = []
    for scale in decay_scales:
        if scale <= 7:
            ref = np.maximum(lag1, 1e-8)
        elif scale <= 14:
            ref = np.maximum(roll3, 1e-8)
        else:
            ref = np.maximum(roll7, 1e-8)
        excitation = np.log1p(ref) - np.log1p(anchor_safe)
        excitation_features.append(excitation)

    # History regularity: low std relative to mean => regular arrivals
    mean_proxy = np.maximum(roll7, 1e-8)
    regularity = 1.0 - np.clip(roll_std / mean_proxy, 0.0, 3.0) / 3.0

    # Burst indicator: lag1 >> roll7 => recent burst
    burst = np.clip((lag1 - roll7) / np.maximum(roll7, 1e-8), -2.0, 5.0)

    # Arrival rate proxy: history_count normalized
    rate_proxy = np.log1p(history_count)

    parts = [
        baseline_log.reshape(-1, 1),
        *[e.reshape(-1, 1) for e in excitation_features],
        regularity.reshape(-1, 1),
        burst.reshape(-1, 1),
        rate_proxy.reshape(-1, 1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def fit_intensity_baseline(
    intensity_features: np.ndarray,
    occurrence_target: np.ndarray,
    *,
    random_state: int = 0,
    min_samples: int = 20,
) -> tuple[HistGradientBoostingRegressor | None, bool]:
    """Fit a baseline intensity model μ(x) via quantile regression.

    Models the conditional probability of positive investor activity
    using intensity features derived from hawkes_intensity_features().

    Parameters
    ----------
    intensity_features : (n, k) float32
        Features from hawkes_intensity_features.
    occurrence_target : (n,) binary
        1 if investor activity > 0, else 0.
    random_state : int
        Random seed.
    min_samples : int
        Minimum samples to train.

    Returns
    -------
    (model, converged) : tuple
        model is a fitted HGBR or None if insufficient data.
        converged is True if model was fitted.
    """
    y = np.asarray(occurrence_target, dtype=np.float64)
    if y.size < min_samples or y.sum() < 3 or (y.size - y.sum()) < 3:
        return None, False

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=2,
        max_iter=80,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(intensity_features, y)
    return model, True


def predict_intensity_probability(
    intensity_model: HistGradientBoostingRegressor | None,
    intensity_features: np.ndarray,
    fallback_rate: float,
) -> np.ndarray:
    """Predict occurrence probability from the intensity baseline model.

    Clips output to [0.01, 0.99] to maintain valid probability range.

    Parameters
    ----------
    intensity_model : fitted HGBR or None
        If None, returns fallback_rate for all rows.
    intensity_features : (n, k) float32
        Features from hawkes_intensity_features.
    fallback_rate : float
        Default active rate if model is unavailable.

    Returns
    -------
    proba : (n,) float64
        Predicted probability of positive investor activity.
    """
    n = intensity_features.shape[0]
    if intensity_model is None:
        return np.full(n, np.clip(fallback_rate, 0.01, 0.99), dtype=np.float64)

    raw = intensity_model.predict(intensity_features)
    return np.clip(raw, 0.01, 0.99).astype(np.float64)


def intensity_diagnostics(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """Compute diagnostics for the intensity model.

    Returns
    -------
    dict with keys:
        - brier_score: mean((proba - y)^2)
        - calibration_gap: |mean(proba) - mean(y)|
        - discrimination: std(proba) — higher is better
        - n_positive: count of positive events
        - n_total: total observations
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if y.size == 0:
        return {
            "brier_score": 1.0,
            "calibration_gap": 1.0,
            "discrimination": 0.0,
            "n_positive": 0,
            "n_total": 0,
        }
    brier = float(np.mean((p - y) ** 2))
    cal_gap = float(abs(np.mean(p) - np.mean(y)))
    disc = float(np.std(p))
    return {
        "brier_score": brier,
        "calibration_gap": cal_gap,
        "discrimination": disc,
        "n_positive": int(y.sum()),
        "n_total": int(y.size),
    }
