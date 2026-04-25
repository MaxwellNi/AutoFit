"""NC-CoPo (Neural Compound-Poisson Conformal) calibrator — Round-12 stub.

Status: RESEARCH STUB (2026-04-24 Round 12).  The public API is committed
so downstream evaluation harnesses can import it.  The split-conformal
implementation itself is intentionally small and self-contained; the
finite-sample coverage guarantee (Theorem 1 of
`docs/NCCOPO_THEOREM_DRAFT_20260423.md`) is proved in the theory draft
and verified empirically by `scripts/verify_nccopo_gumbel.py` (pending).

The calibrator sits DOWNSTREAM of an arbitrary compound-Poisson point
predictor.  It does NOT retrain the predictor; it consumes calibration
pairs `(y_cal, y_pred_cal)` plus plug-in moment estimates
`(mu_hat_cal, sigma_hat_cal)` — obtained from the trained Poisson
intensity network lambda_theta and the severity GEV g_theta — and emits
a marginal (1 - alpha)-coverage prediction interval for every future
point.

Novelty target (Level 2 / Level 3 in the four-tier novelty ladder):

  Level 2 — finite-sample split-conformal coverage for random-sum
  responses under covariate-dependent intensity AND severity (fills the
  gap left by Gibbs 2023 / Romano 2019 / Lei 2018 who only cover
  marginal regression).

  Level 3 — Mondrian-conditional coverage on (lambda, g) quality slices
  + heavy-tail DRO envelope under the architectural constraint
  xi_theta <= 1/2 - delta.

References:
  * docs/NCCOPO_THEOREM_DRAFT_20260423.md (Theorem 1 + verification plan)
  * Gibbs, I. & Candes, E. (2023) "Conformal inference of counterfactuals"
  * Romano, Y. et al. (2019) "Conformalized quantile regression"
  * Lei, J. et al. (2018) "Distribution-free predictive inference"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class NCCoPoConfig:
    """Configuration for a single NC-CoPo calibration run.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate (default 0.10 → 90% prediction interval).
    use_studentized : bool
        If True, non-conformity score is |y - y_pred| / sigma_hat
        (Theorem 1-a, sharper under heavy tails).  If False, absolute
        residual only (Theorem 1-b baseline).
    mondrian_groups : Optional[np.ndarray]
        Per-sample integer group labels for Mondrian-conditional
        calibration (Level 3).  When provided, the conformal quantile is
        computed per-group rather than marginally.
    """

    alpha: float = 0.10
    use_studentized: bool = True
    mondrian_groups: Optional[np.ndarray] = None


@dataclass
class NCCoPoFitResult:
    """Fitted-state container for a NC-CoPo calibrator."""

    alpha: float
    conformal_q: float
    n_cal: int
    use_studentized: bool
    # Per-group conformal quantile for Mondrian mode (empty when marginal).
    group_conformal_q: dict = field(default_factory=dict)
    # Empirical coverage on the calibration set (sanity check).
    cal_coverage: float = 0.0


def _nonconformity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_hat: Optional[np.ndarray],
    use_studentized: bool,
) -> np.ndarray:
    resid = np.abs(y_true - y_pred)
    if use_studentized and sigma_hat is not None:
        denom = np.clip(np.asarray(sigma_hat, dtype=np.float64), 1e-8, None)
        return resid / denom
    return resid


class NCCoPoCalibrator:
    """Split-conformal calibrator for compound-Poisson random-sum outputs.

    Usage
    -----
    >>> cal = NCCoPoCalibrator(NCCoPoConfig(alpha=0.10))
    >>> cal.fit(y_cal, y_pred_cal, sigma_hat_cal=sigma_cal)
    >>> lo, hi = cal.predict_interval(y_pred_test, sigma_hat_test=sigma_test)
    >>> # Theorem 1: P(Y in [lo, hi]) >= 1 - alpha, finite-sample, marginal
    """

    def __init__(self, config: Optional[NCCoPoConfig] = None) -> None:
        self.config = config or NCCoPoConfig()
        self._result: Optional[NCCoPoFitResult] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        sigma_hat_cal: Optional[np.ndarray] = None,
    ) -> "NCCoPoCalibrator":
        """Compute the (1-alpha)-quantile of non-conformity scores.

        Theorem 1 guarantee: given n calibration samples drawn exchangeably
        with future test samples, returning the
        ceil((n+1)(1-alpha))/n-quantile of the non-conformity score
        sequence yields a marginal (1 - alpha)-coverage interval.
        """

        y_cal = np.asarray(y_cal, dtype=np.float64).ravel()
        y_pred_cal = np.asarray(y_pred_cal, dtype=np.float64).ravel()
        if y_cal.shape != y_pred_cal.shape:
            raise ValueError(
                f"y_cal shape {y_cal.shape} != y_pred_cal shape {y_pred_cal.shape}"
            )
        n_cal = int(y_cal.size)
        if n_cal < 2:
            raise ValueError("NC-CoPo requires >= 2 calibration samples.")

        scores = _nonconformity(
            y_cal, y_pred_cal, sigma_hat_cal, self.config.use_studentized
        )
        # Conformal quantile of level ceil((n+1)(1-alpha))/n.
        q_level = min(1.0, np.ceil((n_cal + 1) * (1.0 - self.config.alpha)) / n_cal)
        conformal_q = float(np.quantile(scores, q_level, method="higher"))

        # Mondrian-conditional quantiles (Level 3 extension).
        group_q: dict = {}
        groups = self.config.mondrian_groups
        if groups is not None:
            groups = np.asarray(groups).ravel()
            if groups.size != n_cal:
                raise ValueError("mondrian_groups length must equal n_cal.")
            for g in np.unique(groups):
                mask = groups == g
                if int(mask.sum()) >= 2:
                    ng = int(mask.sum())
                    qg = min(1.0, np.ceil((ng + 1) * (1.0 - self.config.alpha)) / ng)
                    group_q[int(g)] = float(
                        np.quantile(scores[mask], qg, method="higher")
                    )

        # Calibration-set empirical coverage sanity check.
        if self.config.use_studentized and sigma_hat_cal is not None:
            lo_cal = y_pred_cal - conformal_q * np.clip(sigma_hat_cal, 1e-8, None)
            hi_cal = y_pred_cal + conformal_q * np.clip(sigma_hat_cal, 1e-8, None)
        else:
            lo_cal = y_pred_cal - conformal_q
            hi_cal = y_pred_cal + conformal_q
        cal_cov = float(np.mean((y_cal >= lo_cal) & (y_cal <= hi_cal)))

        self._result = NCCoPoFitResult(
            alpha=float(self.config.alpha),
            conformal_q=conformal_q,
            n_cal=n_cal,
            use_studentized=bool(self.config.use_studentized),
            group_conformal_q=group_q,
            cal_coverage=cal_cov,
        )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict_interval(
        self,
        y_pred: np.ndarray,
        sigma_hat: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction interval of level 1 - alpha.

        If a Mondrian `groups` argument is supplied AND the calibrator was
        fitted with groups, per-sample quantiles are looked up; samples
        whose group was unseen during calibration fall back to the
        marginal quantile.
        """

        if self._result is None:
            raise RuntimeError("Calibrator must be fitted before predict_interval.")
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        q = np.full(y_pred.shape, self._result.conformal_q, dtype=np.float64)
        if groups is not None and self._result.group_conformal_q:
            groups = np.asarray(groups).ravel()
            for g_int, qv in self._result.group_conformal_q.items():
                q = np.where(groups == g_int, qv, q)
        if self._result.use_studentized and sigma_hat is not None:
            scale = np.clip(np.asarray(sigma_hat, dtype=np.float64), 1e-8, None)
            lo = y_pred - q * scale
            hi = y_pred + q * scale
        else:
            lo = y_pred - q
            hi = y_pred + q
        return lo, hi

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    @property
    def result(self) -> Optional[NCCoPoFitResult]:
        return self._result
