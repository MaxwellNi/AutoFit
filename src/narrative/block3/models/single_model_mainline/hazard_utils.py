"""Discrete-time hazard transform utilities for binary lane survival modelling.

These functions bridge cumulative event probability and per-step hazard rate,
enabling calibration in hazard space where the time-to-event structure is
naturally respected.

Math
----
Given a cumulative event probability *p* at horizon *H* and an assumption of
constant per-step hazard *h*:

    p = 1 - (1 - h)^H          ⟹  h = 1 - (1 - p)^{1/H}

Calibrating in hazard space ensures that the resulting survival function
S(t) = (1-h)^t is monotonically decreasing — a structural guarantee absent
from raw probability calibration.
"""
from __future__ import annotations

import numpy as np


def prob_to_daily_hazard(prob: np.ndarray, horizon: int) -> np.ndarray:
    """Map cumulative event probability at horizon *H* to per-step hazard."""
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    h = max(1, int(horizon))
    return np.clip(1.0 - np.power(1.0 - p, 1.0 / float(h)), 0.0, 1.0)


def daily_hazard_to_cumulative(hazard: np.ndarray, horizon: int) -> np.ndarray:
    """Map per-step hazard to cumulative event probability at horizon *H*."""
    hz = np.asarray(hazard, dtype=np.float64).reshape(-1)
    hz = np.clip(hz, 0.0, 1.0)
    h = max(1, int(horizon))
    return np.clip(1.0 - np.power(1.0 - hz, float(h)), 0.0, 1.0)


def survival_nll(prob: np.ndarray, event: np.ndarray, horizon: int = 1) -> float:
    r"""Discrete-time survival negative log-likelihood.

    Given calibrated event probability *p* at horizon *H*, per-step hazard
    *h = 1 - (1-p)^{1/H}*, and binary event indicator *δ*:

        NLL = -mean[ δ·log(1 - S(H)) + (1-δ)·log(S(H)) ]

    where S(H) = (1-h)^H = 1 - p.  This is algebraically identical to
    standard binary cross-entropy, **but the key insight is evaluating it
    after hazard-space calibration** — ensuring the scoring criterion rewards
    calibrators that respect the time-to-event structure.

    For multi-horizon consistency analysis, the per-step hazard *h* should
    be approximately constant across horizons for the same entity, so
    comparing ``survival_nll`` across horizons reveals calibration drift.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    delta = np.asarray(event, dtype=np.float64).reshape(-1)
    # S(H) = 1 - p, f(H) = p
    nll = -(delta * np.log(p) + (1.0 - delta) * np.log(1.0 - p))
    return float(nll.mean())


def cross_horizon_hazard_consistency(
    prob_short: np.ndarray,
    prob_long: np.ndarray,
    horizon_short: int,
    horizon_long: int,
) -> dict:
    """Check consistency of calibrated probabilities across two horizons.

    If the per-step hazard rate h is constant, then:
        P(event ≤ H_long) >= P(event ≤ H_short) for H_long > H_short

    Returns a dict with:
      - violation_rate: fraction of samples where P_long < P_short
      - hazard_drift: mean |h_short - h_long| / mean(h_short, h_long)
      - monotonicity_satisfied: True if violation_rate <= 0.05
    """
    p_s = np.clip(np.asarray(prob_short, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    p_l = np.clip(np.asarray(prob_long, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)
    h_s = prob_to_daily_hazard(p_s, horizon_short)
    h_l = prob_to_daily_hazard(p_l, horizon_long)

    violations = p_l < p_s - 1e-6
    violation_rate = float(violations.mean())

    mean_h = 0.5 * (h_s.mean() + h_l.mean())
    hazard_drift = float(np.abs(h_s - h_l).mean() / max(mean_h, 1e-7))

    return {
        "violation_rate": violation_rate,
        "hazard_drift": hazard_drift,
        "monotonicity_satisfied": violation_rate <= 0.05,
        "mean_hazard_short": float(h_s.mean()),
        "mean_hazard_long": float(h_l.mean()),
    }
