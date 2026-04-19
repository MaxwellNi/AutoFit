"""GPD (Generalized Pareto Distribution) tail utilities for funding lane P2.

Implements Peaks-Over-Threshold (POT) tail correction:
- Fit GPD parameters (shape ξ, scale σ) from exceedances above a threshold u
- Apply tail correction to funding predictions: when residual > u,
  replace simple clip with GPD-informed bound
- Compute tail diagnostics (tail index, exceedance rate, expected shortfall)

References:
  Pickands (1975) — Statistical inference using extreme order statistics
  McNeil & Frey (2000) — Estimation of tail-related risk measures
  Embrechts et al. (1997) — Modelling Extremal Events
"""
from __future__ import annotations

import numpy as np


def fit_gpd_pot(
    exceedances: np.ndarray,
    *,
    min_exceedances: int = 15,
) -> dict[str, float]:
    """Fit GPD to exceedances (values already shifted: y - threshold).

    Uses Method of Moments (MOM) estimator which is simple and robust:
      ξ = 0.5 * (1 - mean²/var)
      σ = 0.5 * mean * (1 + mean²/var)
    Valid for ξ < 0.5 (covers all realistic financial tail shapes).

    Returns dict with keys: xi (shape), sigma (scale), n_exceedances, converged.
    """
    exc = np.asarray(exceedances, dtype=np.float64)
    exc = exc[np.isfinite(exc) & (exc > 0)]

    if exc.size < min_exceedances:
        return {"xi": 0.0, "sigma": 0.0, "n_exceedances": int(exc.size), "converged": False}

    mean_exc = float(np.mean(exc))
    var_exc = float(np.var(exc, ddof=1))

    if mean_exc < 1e-12 or var_exc < 1e-12:
        return {"xi": 0.0, "sigma": mean_exc, "n_exceedances": len(exc), "converged": bool(mean_exc > 0)}

    ratio = mean_exc ** 2 / var_exc
    xi = 0.5 * (1.0 - ratio)
    sigma = 0.5 * mean_exc * (1.0 + ratio)

    # Safety: clip xi to [-0.5, 0.5] — MOM is only valid for xi < 0.5
    xi = float(np.clip(xi, -0.5, 0.5))
    sigma = float(max(sigma, 1e-8))

    return {"xi": xi, "sigma": sigma, "n_exceedances": len(exc), "converged": True}


def gpd_quantile(xi: float, sigma: float, p: float) -> float:
    """GPD quantile function: Q(p) = σ/ξ * ((1-p)^(-ξ) - 1) for ξ ≠ 0."""
    if sigma <= 0 or p <= 0 or p >= 1:
        return 0.0
    if abs(xi) < 1e-8:
        # Exponential case
        return -sigma * np.log(1.0 - p)
    return (sigma / xi) * ((1.0 - p) ** (-xi) - 1.0)


def gpd_tail_bound(
    xi: float,
    sigma: float,
    threshold: float,
    confidence: float = 0.99,
) -> float:
    """Compute GPD-based upper bound: threshold + Q(confidence).

    This replaces the naive residual_cap with a statistically-grounded bound.
    """
    return threshold + gpd_quantile(xi, sigma, confidence)


def gpd_expected_shortfall(xi: float, sigma: float, threshold: float, p: float = 0.95) -> float:
    """Expected Shortfall (CVaR) at level p using GPD.

    ES_p = VaR_p/(1-xi) + (sigma - xi*threshold)/(1-xi)  for xi < 1.
    """
    if xi >= 1.0 or sigma <= 0:
        return float("inf")
    var_p = threshold + gpd_quantile(xi, sigma, p)
    es = var_p / (1.0 - xi) + (sigma - xi * threshold) / (1.0 - xi)
    return float(es)


def compute_tail_diagnostics(
    residuals: np.ndarray,
    threshold: float,
    gpd_params: dict[str, float],
) -> dict[str, float]:
    """Compute tail diagnostics for funding lane monitoring."""
    exc = np.asarray(residuals, dtype=np.float64)
    above = exc[exc > threshold]
    n_total = len(exc)
    n_above = len(above)

    xi = gpd_params.get("xi", 0.0)
    sigma = gpd_params.get("sigma", 0.0)

    tail_bound_99 = gpd_tail_bound(xi, sigma, threshold, confidence=0.99)
    tail_bound_95 = gpd_tail_bound(xi, sigma, threshold, confidence=0.95)
    es_95 = gpd_expected_shortfall(xi, sigma, threshold, p=0.95)

    return {
        "exceedance_rate": float(n_above / max(n_total, 1)),
        "threshold": float(threshold),
        "gpd_xi": float(xi),
        "gpd_sigma": float(sigma),
        "tail_bound_95": float(tail_bound_95),
        "tail_bound_99": float(tail_bound_99),
        "expected_shortfall_95": float(es_95),
        "n_exceedances": int(n_above),
        "gpd_converged": bool(gpd_params.get("converged", False)),
    }


def apply_gpd_tail_correction(
    predictions: np.ndarray,
    anchor: np.ndarray,
    threshold: float,
    gpd_params: dict[str, float],
    *,
    confidence: float = 0.99,
) -> np.ndarray:
    """Apply GPD tail correction to predictions.

    For predictions where residual (pred - anchor) exceeds the threshold,
    clip to GPD-based upper bound instead of naive cap.
    """
    pred = np.asarray(predictions, dtype=np.float64).copy()
    anc = np.asarray(anchor, dtype=np.float64)

    if not gpd_params.get("converged", False):
        return pred

    xi = gpd_params.get("xi", 0.0)
    sigma = gpd_params.get("sigma", 0.0)

    upper_bound = float(gpd_tail_bound(xi, sigma, threshold, confidence))
    residual = pred - anc
    mask = residual > threshold
    if mask.any():
        pred[mask] = anc[mask] + np.minimum(residual[mask], upper_bound)

    return pred
