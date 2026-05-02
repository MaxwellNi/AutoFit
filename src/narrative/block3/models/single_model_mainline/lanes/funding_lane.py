#!/usr/bin/env python3
"""Funding lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..tail_utils import (
    apply_gpd_tail_correction,
    compute_tail_diagnostics,
    cqr_conformal_quantile,
    cqr_conformity_scores,
    cqr_prediction_interval,
    fit_gpd_pot,
)


@dataclass(frozen=True)
class FundingLaneSpec:
    lane_name: str = "funding"
    supports_anchor_residual: bool = True
    supports_jump_hurdle_process: bool = True
    supports_anchor_reliability_gate: bool = True
    supports_tail_aware_objective: bool = True
    supports_horizon_bucket_subpaths: bool = True
    supports_source_scaling_guard: bool = True
    guardrails: Tuple[str, ...] = ("source_rich_blowup", "log_domain_regression", "easy_cell_only_improvement")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_anchor_residual": self.supports_anchor_residual,
            "supports_jump_hurdle_process": self.supports_jump_hurdle_process,
            "supports_anchor_reliability_gate": self.supports_anchor_reliability_gate,
            "supports_tail_aware_objective": self.supports_tail_aware_objective,
            "supports_horizon_bucket_subpaths": self.supports_horizon_bucket_subpaths,
            "supports_source_scaling_guard": self.supports_source_scaling_guard,
            "guardrails": self.guardrails,
        }


class FundingLaneRuntime:
    def __init__(self, spec: FundingLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or FundingLaneSpec()
        self.random_state = int(random_state)
        self._event_model: HistGradientBoostingClassifier | None = None
        self._model: HistGradientBoostingRegressor | None = None
        self._fallback_value = 0.0
        self._residual_blend = 1.0
        self._residual_cap = np.inf
        self._anchor_calibration_mae = 0.0
        self._guarded_calibration_mae = 0.0
        self._anchor_dominance = 1.0
        self._calibration_rows = 0
        self._jump_event_rate = 0.0
        self._positive_jump_rows = 0
        self._positive_jump_median = 0.0
        self._jump_floor = 1e-6
        self._uses_jump_hurdle_head = False
        self._log_domain_enabled = False
        self._source_scaling_enabled = False
        self._tail_weight = 0.0
        self._tail_quantile = 0.90
        self._source_scale_strength = 0.0
        self._source_scale_reliability = 0.0
        self._source_scale_signed_mode = False
        self._gpd_params: dict[str, float] = {"xi": 0.0, "sigma": 0.0, "n_exceedances": 0, "converged": False}
        self._gpd_threshold = 0.0
        self._gpd_enabled = False
        # P2.1: CQR prediction interval state
        self._cqr_enabled = False
        self._cqr_conformal_q = 0.0
        self._cqr_alpha = 0.10
        self._cqr_q_lo_model: HistGradientBoostingRegressor | None = None
        self._cqr_q_hi_model: HistGradientBoostingRegressor | None = None
        self._cqr_converged = False
        self._fitted = False
        # 2026-04-21 18:57 CEST — anti-silent-collapse observability / trunk-fallback.
        # When the jump-hurdle head refuses to train (zero-inflated target, small N,
        # degenerate jump_target std), we MUST NOT silently return anchor_vec.
        # Instead we train a ridge readout on the trunk lane_state so trunk signal
        # quality can be independently measured.
        self._anchor_only_mode = False
        self._anchor_only_reason = ""
        self._jump_target_std = 0.0
        self._trunk_fallback_coef: np.ndarray | None = None
        self._trunk_fallback_intercept = 0.0
        self._trunk_fallback_fitted = False
        self._force_hurdle = False
        # Round-12 Route L2/K2 state (2026-04-24)
        self._source_scale_silently_dead = False
        self._ss_fallback_active = False
        # §0w (2026-04-24): audit-only note when SS_FALLBACK env is on but
        # _source_scaling_enabled is False (core_only silently-dead case).
        self._ss_fallback_env_requested_no_op = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        source_scale: np.ndarray | None = None,
        use_log_domain: bool = False,
        enable_source_scaling: bool = False,
        tail_weight: float = 0.0,
        tail_quantile: float = 0.90,
        enable_gpd_tail: bool = False,
        enable_cqr_interval: bool = False,
        cqr_alpha: float = 0.10,
        force_hurdle: bool = False,
    ) -> "FundingLaneRuntime":
        target = np.asarray(y, dtype=np.float64)
        finite = target[np.isfinite(target)]
        self._fallback_value = float(np.nanmedian(finite)) if finite.size else 0.0
        self._event_model = None
        self._model = None
        self._residual_blend = 0.0
        self._residual_cap = 0.0
        self._anchor_calibration_mae = 0.0
        self._guarded_calibration_mae = 0.0
        self._anchor_dominance = 1.0
        self._calibration_rows = 0
        self._jump_event_rate = 0.0
        self._positive_jump_rows = 0
        self._positive_jump_median = 0.0
        self._jump_floor = 1e-6
        self._uses_jump_hurdle_head = False
        self._log_domain_enabled = bool(use_log_domain)
        self._tail_weight = float(max(tail_weight, 0.0))
        self._tail_quantile = float(np.clip(tail_quantile, 0.50, 0.99))
        self._source_scale_strength = 0.0
        self._source_scale_reliability = 0.0
        self._source_scale_signed_mode = False
        self._gpd_params = {"xi": 0.0, "sigma": 0.0, "n_exceedances": 0, "converged": False}
        self._gpd_threshold = 0.0
        self._gpd_enabled = bool(enable_gpd_tail)
        self._cqr_enabled = bool(enable_cqr_interval)
        self._cqr_alpha = float(np.clip(cqr_alpha, 0.01, 0.50))
        self._cqr_conformal_q = 0.0
        self._cqr_q_lo_model = None
        self._cqr_q_hi_model = None
        self._cqr_converged = False
        if target.size == 0:
            self._fitted = True
            return self

        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=target.size)
        source_scale_vec = _resolve_source_scale(source_scale, length=target.size)
        self._source_scaling_enabled = bool(enable_source_scaling and np.any(source_scale_vec > 1e-8))
        design = _merge_features(lane_state, aux_features, anchor_vec)
        jump_target = _positive_jump_target(
            target=target,
            anchor_vec=anchor_vec,
            use_log_domain=self._log_domain_enabled,
        )
        self._anchor_dominance = _anchor_dominance(
            anchor_vec=anchor_vec,
            residual=jump_target,
            use_log_domain=self._log_domain_enabled,
        )
        self._jump_floor = _jump_event_floor(jump_target)
        positive_jump_mask = jump_target > self._jump_floor
        self._jump_event_rate = float(np.mean(positive_jump_mask)) if target.size else 0.0
        self._positive_jump_rows = int(positive_jump_mask.sum())
        self._positive_jump_median = (
            float(np.nanmedian(jump_target[positive_jump_mask])) if positive_jump_mask.any() else 0.0
        )
        anchor_mae = float(np.mean(np.abs(target - anchor_vec)))
        self._anchor_calibration_mae = anchor_mae
        self._guarded_calibration_mae = anchor_mae
        self._calibration_rows = int(target.size)

        self._jump_target_std = float(np.nanstd(jump_target)) if jump_target.size else 0.0
        self._force_hurdle = bool(force_hurdle)

        # Route G (2026-04-23): force_hurdle=True bypasses the positive-jump
        # row threshold and the jump_target_std degeneracy gate, but we keep
        # the hard `target.size < 12` gate because severity models with <12
        # rows are numerically non-identifiable regardless of toggle.
        jumps_below_threshold = self._positive_jump_rows < _minimum_positive_jump_rows(target.size)
        std_degenerate = self._jump_target_std < 1e-8
        short_circuit = target.size < 12 or (
            not self._force_hurdle and (jumps_below_threshold or std_degenerate)
        )

        if short_circuit:
            # 2026-04-21 18:57 CEST — trunk-fallback instead of silent anchor-only.
            # Without this, predict() used to collapse to a constant anchor_vec
            # regardless of lane_state (trunk output), which made it impossible
            # to measure trunk signal quality. Now we train a ridge on the
            # trunk-derived design matrix so the trunk gets a real test.
            self._anchor_only_mode = True
            if target.size < 12:
                self._anchor_only_reason = "n_too_small"
            elif self._positive_jump_rows < _minimum_positive_jump_rows(target.size):
                self._anchor_only_reason = "positive_jumps_below_threshold"
            else:
                self._anchor_only_reason = "jump_target_std_degenerate"
            self._trunk_fallback_fitted = _fit_trunk_fallback_ridge(
                self,
                design=design,
                target=target,
                anchor_vec=anchor_vec,
                use_log_domain=self._log_domain_enabled,
            )
            self._fitted = True
            return self

        calibration = _split_funding_calibration(
            design=design,
            target=target,
            anchor=anchor_vec,
            jump_target=jump_target,
            jump_floor=self._jump_floor,
            source_scale=source_scale_vec if self._source_scaling_enabled else None,
        )
        if calibration is not None:
            calibration_models = _fit_jump_process_models(
                design=calibration["train_design"],
                jump_target=calibration["train_jump"],
                jump_floor=float(calibration["jump_floor"]),
                random_state=self.random_state,
                use_log_domain=self._log_domain_enabled,
            )
            calibration_jump_pred = _predict_jump_process(
                design=calibration["calibration_design"],
                event_model=calibration_models["event_model"],
                severity_model=calibration_models["severity_model"],
                event_rate=float(calibration_models["event_rate"]),
                positive_jump_median=float(calibration_models["positive_jump_median"]),
                use_log_domain=self._log_domain_enabled,
            )
            (
                self._residual_blend,
                self._residual_cap,
                self._anchor_calibration_mae,
                self._guarded_calibration_mae,
            ) = _calibrate_anchor_residual_guard(
                anchor_vec=calibration["calibration_anchor"],
                target_vec=calibration["calibration_target"],
                residual_pred=calibration_jump_pred,
                residual_target=calibration["calibration_jump"],
                anchor_dominance=self._anchor_dominance,
                use_log_domain=self._log_domain_enabled,
                tail_weight=self._tail_weight,
                tail_quantile=self._tail_quantile,
            )
            if self._source_scaling_enabled and calibration.get("calibration_source_scale") is not None:
                import os as _os
                self._source_scale_signed_mode = _os.environ.get("MAINLINE_FUNDING_SS_SIGNED", "0") in ("1", "true", "True")
                (
                    self._source_scale_strength,
                    self._guarded_calibration_mae,
                ) = _calibrate_source_scaling_guard(
                    anchor_vec=calibration["calibration_anchor"],
                    target_vec=calibration["calibration_target"],
                    residual_pred=calibration_jump_pred,
                    residual_blend=self._residual_blend,
                    residual_cap=self._residual_cap,
                    source_scale=calibration["calibration_source_scale"],
                    use_log_domain=self._log_domain_enabled,
                    tail_weight=self._tail_weight,
                    tail_quantile=self._tail_quantile,
                    allow_signed_source_scale=self._source_scale_signed_mode,
                )
                self._source_scale_reliability = _guard_improvement_ratio(
                    baseline_mae=self._anchor_calibration_mae,
                    guarded_mae=self._guarded_calibration_mae,
                )
            self._calibration_rows = int(calibration["calibration_target"].size)

        # ------------------------------------------------------------------
        # Round-12 Route K2 fix (2026-04-24): calibration-rejection fallback.
        # When _calibrate_anchor_residual_guard picks best_blend=0 (the
        # severity path is rejected by grid-search), source_scaling and
        # tail_focus atoms silently die — their effect is multiplied by
        # residual_blend=0. K2 installs an explicit source-scaling fallback
        # that applies on the anchor (NOT on residual) so the atom remains
        # measurable even when severity is rejected.  Enabled via
        # MAINLINE_FUNDING_SS_FALLBACK=1 (default OFF to preserve prior
        # baselines). Also emits Route L2 warning when source-scaling is
        # enabled at flag level but silently dead at data level.
        # ------------------------------------------------------------------
        import os as _os
        import logging as _logging
        _log = _logging.getLogger("narrative.block3.mainline.funding_lane")
        if enable_source_scaling and not self._source_scaling_enabled:
            # Route L2 gate: flag=True but vector all-zero.
            _log.warning(
                "[funding_lane L2] enable_source_scaling=True but "
                "source_scale_vec all-zero (core_only ablation or missing "
                "edgar/text columns). Atom is silently dead."
            )
            self._source_scale_silently_dead = True
        else:
            self._source_scale_silently_dead = False

        self._ss_fallback_active = False
        if (
            _os.environ.get("MAINLINE_FUNDING_SS_FALLBACK", "0") in ("1", "true", "True")
            and self._source_scaling_enabled
            and self._residual_blend <= 1e-8
        ):
            self._residual_blend = 0.05
            self._source_scale_strength = max(self._source_scale_strength, 0.5)
            if not np.isfinite(self._residual_cap) or self._residual_cap <= 0.0:
                # Use median positive-jump as a scale anchor.
                self._residual_cap = max(
                    2.0 * float(self._positive_jump_median + 1e-8),
                    1.0,
                )
            self._ss_fallback_active = True
            _log.info(
                "[funding_lane K2] SS_FALLBACK activated: blend=0.05, "
                "strength=%.3f, cap=%.3f",
                self._source_scale_strength,
                self._residual_cap,
            )
        elif (
            _os.environ.get("MAINLINE_FUNDING_SS_FALLBACK", "0") in ("1", "true", "True")
            and not self._source_scaling_enabled
        ):
            # §0w (2026-04-24): env flag is on but data-level enablement
            # is False (core_only silently-dead case).  No numeric change,
            # but audit telemetry records the request so downstream
            # analyzers can correctly interpret sidecars.
            self._ss_fallback_env_requested_no_op = True
            _log.info(
                "[funding_lane K2] SS_FALLBACK env=1 requested but "
                "source_scaling_enabled=False (vector all-zero). "
                "No-op; audit note recorded."
            )

        full_models = _fit_jump_process_models(
            design=design,
            jump_target=jump_target,
            jump_floor=self._jump_floor,
            random_state=self.random_state,
            use_log_domain=self._log_domain_enabled,
        )
        self._event_model = full_models["event_model"]
        self._model = full_models["severity_model"]
        self._jump_event_rate = float(full_models["event_rate"])
        self._positive_jump_rows = int(full_models["positive_jump_rows"])
        self._positive_jump_median = float(full_models["positive_jump_median"])
        self._uses_jump_hurdle_head = bool(
            self._event_model is not None or self._model is not None or self._positive_jump_median > 0.0
        )

        # P2: Fit GPD to tail exceedances for distribution-aware tail correction
        if self._gpd_enabled and self._uses_jump_hurdle_head:
            self._gpd_threshold, self._gpd_params = _fit_gpd_for_funding(
                jump_target=jump_target,
                tail_quantile=self._tail_quantile,
            )

        # P2.1: CQR — train quantile regressors and compute conformal correction
        if self._cqr_enabled and self._uses_jump_hurdle_head and calibration is not None:
            self._cqr_q_lo_model, self._cqr_q_hi_model, self._cqr_conformal_q, self._cqr_converged = (
                _fit_cqr_for_funding(
                    train_design=calibration["train_design"],
                    train_jump=calibration["train_jump"],
                    cal_design=calibration["calibration_design"],
                    cal_jump=calibration["calibration_jump"],
                    alpha=self._cqr_alpha,
                    random_state=self.random_state,
                )
            )

        self._fitted = True
        return self

    def predict(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        source_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FundingLaneRuntime is not fitted")

        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=len(lane_state))
        source_scale_vec = _resolve_source_scale(source_scale, length=len(lane_state))
        if not self._uses_jump_hurdle_head:
            # 2026-04-21 18:57 CEST — use trunk ridge fallback instead of
            # silently returning a constant anchor. This makes trunk quality
            # measurable and breaks the horizon-invariant collapse.
            if self._trunk_fallback_fitted and self._trunk_fallback_coef is not None:
                design_pred = _merge_features(lane_state, aux_features, anchor_vec)
                residual_log = design_pred @ self._trunk_fallback_coef + self._trunk_fallback_intercept
                if self._log_domain_enabled:
                    pred = np.expm1(np.log1p(np.clip(anchor_vec, 0.0, None)) + residual_log)
                else:
                    pred = np.clip(anchor_vec, 0.0, None) + residual_log
                return np.clip(pred, 0.0, None).astype(np.float64, copy=False)
            return np.clip(anchor_vec, 0.0, None).astype(np.float64, copy=False)

        design = _merge_features(lane_state, aux_features, anchor_vec)
        jump_pred = _predict_jump_process(
            design=design,
            event_model=self._event_model,
            severity_model=self._model,
            event_rate=self._jump_event_rate,
            positive_jump_median=self._positive_jump_median,
            use_log_domain=self._log_domain_enabled,
        )
        return _guarded_funding_prediction(
            anchor_vec=anchor_vec,
            residual_pred=jump_pred,
            residual_blend=self._residual_blend,
            residual_cap=self._residual_cap,
            use_log_domain=self._log_domain_enabled,
            source_scale=source_scale_vec if self._source_scaling_enabled else None,
            source_scale_strength=self._source_scale_strength,
            gpd_params=self._gpd_params if self._gpd_enabled else None,
            gpd_threshold=self._gpd_threshold,
        )

    def predict_interval(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return CQR prediction interval (lower, upper) or None if CQR not fitted."""
        if not self._cqr_converged or self._cqr_q_lo_model is None or self._cqr_q_hi_model is None:
            return None
        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=len(lane_state))
        design = _merge_features(lane_state, aux_features, anchor_vec)
        q_lo_raw = self._cqr_q_lo_model.predict(design)
        q_hi_raw = self._cqr_q_hi_model.predict(design)
        lower, upper = cqr_prediction_interval(q_lo_raw, q_hi_raw, self._cqr_conformal_q)
        # Funding is non-negative
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, lower)
        return lower, upper

    def nccopo_inputs(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        source_scale: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return plug-in compound-Poisson moment inputs for NC-CoPo calibration.

        Round-12 (2026-04-24 02:10 CEST).  Emits per-row estimates of

            mu_hat(x)    = E[Y | x]  = lambda_hat(x) * E[M | x],
            sigma_hat(x) = sqrt( Var[Y | x] )
                         = sqrt( lambda_hat(x) * (Var[M|x] + E[M|x]^2) ),

        using (i) the anchor-guarded funding point prediction as
        `mu_hat` and (ii) the GPD tail params + jump_event_rate as a
        second-moment proxy when available.  When the hurdle head is
        disengaged (anchor-only mode), `sigma_hat` falls back to a
        local-sd estimate from residual_cap.

        Output is consumable directly by
        `narrative.block3.models.calibration.NCCoPoCalibrator`.
        """
        if not self._fitted:
            raise ValueError("FundingLaneRuntime is not fitted")

        mu_hat = self.predict(
            lane_state,
            aux_features=aux_features,
            anchor=anchor,
            source_scale=source_scale,
        )
        n = mu_hat.shape[0]

        # First-moment rescue: positive floor (funding is non-negative).
        mu_hat = np.clip(mu_hat, 0.0, None)

        # Second moment: E[M^2] = Var[M] + E[M]^2.  We approximate E[M]
        # by positive_jump_median (training-time empirical jump
        # magnitude); Var[M] ~ (residual_cap/2)^2 under a Laplace-like
        # envelope; lambda ~ jump_event_rate for a homogeneous rate.
        # These are conservative; Theorem 3 gives asymptotic accuracy
        # under the neural GEV training loss, while this closed form is
        # only a finite-n plug-in.
        E_M = max(self._positive_jump_median, 1e-6)
        cap = float(self._residual_cap) if np.isfinite(self._residual_cap) else max(2.0 * E_M, 1.0)
        Var_M = (cap / 2.0) ** 2
        lam = max(self._jump_event_rate, 1e-6)
        base_var = lam * (Var_M + E_M ** 2)
        # When GPD tail is fitted, scale variance by the GPD second moment
        # proxy (sigma_gpd^2 / (1 - 2*xi)) for xi < 0.5.
        if self._gpd_enabled:
            xi = float(self._gpd_params.get("xi", 0.0))
            sig = float(self._gpd_params.get("sigma", 0.0))
            if xi < 0.5 and sig > 0:
                tail_var = (sig ** 2) / max(1.0 - 2.0 * xi, 1e-3)
                base_var = max(base_var, tail_var)
        sigma_hat = np.full(n, np.sqrt(max(base_var, 1e-8)))

        # §0w (2026-04-24) per-row lambda_hat via trained event model so
        # sigma_hat is genuinely heteroscedastic: Var[Y|x] = lambda(x) *
        # (Var[M] + E[M]^2) + (small) severity-model variance proxy.  A
        # homoscedastic sigma_hat collapses studentized == absolute for
        # NC-CoPo and defeats Theorem 1-a's sharpness gain.
        lambda_per_row = np.full(n, lam)
        if self._event_model is not None:
            try:
                design_test = _merge_features(
                    lane_state,
                    aux_features,
                    _resolve_anchor(anchor, fallback=self._fallback_value, length=n),
                )
                proba = self._event_model.predict_proba(design_test)[:, 1]
                # Event probability is a plug-in for per-row lambda under
                # a homogeneous Bernoulli-thinning hurdle; clip to avoid
                # zero-variance blowups.
                lambda_per_row = np.clip(proba.astype(np.float64), 1e-4, 1.0)
            except Exception:  # noqa: BLE001 — best-effort heteroscedastic upgrade
                pass
        var_per_row = lambda_per_row * (Var_M + E_M ** 2)
        if self._gpd_enabled:
            xi = float(self._gpd_params.get("xi", 0.0))
            sig = float(self._gpd_params.get("sigma", 0.0))
            if xi < 0.5 and sig > 0:
                # §0w (2026-04-24): additive GPD second-moment correction
                # scaled by per-row lambda so heteroscedasticity survives.
                tail_add = (sig ** 2) / max(1.0 - 2.0 * xi, 1e-3)
                var_per_row = var_per_row + lambda_per_row * tail_add
        sigma_hat = np.sqrt(np.maximum(var_per_row, 1e-8))

        # Mondrian slice: predicted-intensity tertile of mu_hat itself.
        if n >= 3:
            q1, q2 = np.quantile(mu_hat, [1 / 3, 2 / 3])
            groups = np.where(mu_hat <= q1, 0, np.where(mu_hat <= q2, 1, 2))
        else:
            groups = np.zeros(n, dtype=int)

        return {
            "mu_hat": mu_hat.astype(np.float64, copy=False),
            "sigma_hat": sigma_hat.astype(np.float64, copy=False),
            "mondrian_groups": groups.astype(np.int64, copy=False),
            "lambda_hat": lambda_per_row.astype(np.float64, copy=False),
            "residual_cap_used": float(cap),
        }

    def describe_tail(self) -> dict[str, object]:
        """Return GPD + CQR tail diagnostics for funding lane monitoring."""
        return {
            "gpd_enabled": self._gpd_enabled,
            "gpd_threshold": self._gpd_threshold,
            "gpd_xi": self._gpd_params.get("xi", 0.0),
            "gpd_sigma": self._gpd_params.get("sigma", 0.0),
            "gpd_n_exceedances": self._gpd_params.get("n_exceedances", 0),
            "gpd_converged": self._gpd_params.get("converged", False),
            "tail_weight": self._tail_weight,
            "tail_quantile": self._tail_quantile,
            "cqr_enabled": self._cqr_enabled,
            "cqr_converged": self._cqr_converged,
            "cqr_conformal_q": self._cqr_conformal_q,
            "cqr_alpha": self._cqr_alpha,
            # 2026-04-21 anti-silent-collapse observability
            "anchor_only_mode": bool(self._anchor_only_mode),
            "anchor_only_reason": self._anchor_only_reason,
            "jump_target_std": float(self._jump_target_std),
            "positive_jump_rows": int(self._positive_jump_rows),
            "lane_hurdle_engaged": bool(self._uses_jump_hurdle_head),
            "trunk_fallback_fitted": bool(self._trunk_fallback_fitted),
            "force_hurdle": bool(self._force_hurdle),
            # 2026-04-24 Round-12 calibration-decision exposure: these
            # three scalars reveal whether the grid-search in
            # _calibrate_anchor_residual_guard accepted or rejected the
            # severity path. residual_blend=0 means severity output was
            # rejected (anchor-only prediction), which silently nullifies
            # source_scaling and tail_focus atoms even when
            # enable_source_scaling=True and tail_weight>0.
            "residual_blend": float(self._residual_blend),
            "residual_cap": float(self._residual_cap),
            "anchor_calibration_mae": float(self._anchor_calibration_mae),
            "guarded_calibration_mae": float(self._guarded_calibration_mae),
            "source_scaling_enabled": bool(self._source_scaling_enabled),
            "source_scale_strength": float(self._source_scale_strength),
            "source_scale_reliability": float(self._source_scale_reliability),
            "source_scale_signed_mode": bool(self._source_scale_signed_mode),
            "calibration_rows": int(self._calibration_rows),
            # Round-12 Route L2/K2 audit gates (2026-04-24)
            "source_scale_silently_dead": bool(self._source_scale_silently_dead),
            "ss_fallback_active": bool(self._ss_fallback_active),
            "ss_fallback_env_requested_no_op": bool(
                self._ss_fallback_env_requested_no_op
            ),
        }


def _build_residual_model(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        random_state=random_state,
    )


def _fit_trunk_fallback_ridge(
    lane: "FundingLaneRuntime",
    design: np.ndarray,
    target: np.ndarray,
    anchor_vec: np.ndarray,
    use_log_domain: bool,
    lambda_reg: float = 1e-2,
) -> bool:
    """Train a ridge readout on the trunk-derived design matrix.

    Purpose (2026-04-21): replace the old silent anchor-only fallback so that
    when the jump-hurdle head refuses to train, the lane still exposes a
    model that actually USES the trunk state. This makes trunk quality
    measurable and breaks the horizon-invariant constant-output collapse.

    The ridge learns a residual on top of the anchor, in log-domain when the
    lane is log-domain-enabled.
    """
    try:
        X = np.asarray(design, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        a = np.asarray(anchor_vec, dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(a) & np.all(np.isfinite(X), axis=1)
        if int(mask.sum()) < 8 or X.shape[1] == 0:
            return False
        X = X[mask]
        y = y[mask]
        a = a[mask]
        if use_log_domain:
            residual = np.log1p(np.clip(y, 0.0, None)) - np.log1p(np.clip(a, 0.0, None))
        else:
            residual = y - a
        if float(np.nanstd(residual)) < 1e-8:
            return False
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        gram = Xb.T @ Xb + lambda_reg * np.eye(Xb.shape[1], dtype=np.float64)
        rhs = Xb.T @ residual
        try:
            w = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        lane._trunk_fallback_coef = w[:-1].astype(np.float64, copy=False)
        lane._trunk_fallback_intercept = float(w[-1])
        return True
    except Exception:
        return False


def _build_event_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        random_state=random_state,
    )


def _split_funding_calibration(
    design: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    jump_target: np.ndarray,
    jump_floor: float,
    source_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray | float] | None:
    n_rows = int(design.shape[0])
    if n_rows < 12:
        return None
    calibration_rows = min(max(n_rows // 5, 4), 64)
    train_rows = n_rows - calibration_rows
    if train_rows < 8:
        return None
    positive_mask = np.asarray(jump_target, dtype=np.float64) > float(jump_floor)
    total_positive = int(positive_mask.sum())
    required_train_positive = min(2, max(total_positive - 1, 0))
    while calibration_rows > 4 and int(positive_mask[:train_rows].sum()) < required_train_positive:
        calibration_rows -= 1
        train_rows = n_rows - calibration_rows
    return {
        "train_design": design[:train_rows],
        "train_jump": jump_target[:train_rows],
        "calibration_design": design[train_rows:],
        "calibration_jump": jump_target[train_rows:],
        "calibration_target": target[train_rows:],
        "calibration_anchor": anchor[train_rows:].astype(np.float64, copy=False),
        "calibration_source_scale": None
        if source_scale is None
        else np.asarray(source_scale[train_rows:], dtype=np.float64),
        "jump_floor": float(jump_floor),
    }


def _positive_jump_target(target: np.ndarray, anchor_vec: np.ndarray, use_log_domain: bool = False) -> np.ndarray:
    target_arr = np.clip(np.asarray(target, dtype=np.float64), 0.0, None)
    anchor_arr = np.clip(np.asarray(anchor_vec, dtype=np.float64), 0.0, None)
    if use_log_domain:
        return np.clip(np.log1p(target_arr) - np.log1p(anchor_arr), 0.0, None)
    return np.clip(target_arr - anchor_arr, 0.0, None)


def _minimum_positive_jump_rows(n_rows: int) -> int:
    # Route F (2026-04-23 Round 10): allow env-override for audit probes that
    # need the hurdle+severity path to activate on sparse positive-jump panels.
    # Default behavior unchanged. Set MAINLINE_FUNDING_MIN_JUMP_ROWS=<int> to
    # force a smaller threshold for forensic ablation only.
    import os
    override = os.environ.get("MAINLINE_FUNDING_MIN_JUMP_ROWS", "").strip()
    if override:
        try:
            val = int(override)
            if val >= 1:
                return val
        except ValueError:
            pass
    return 4 if int(n_rows) < 128 else 6


def _jump_event_floor(jump_target: np.ndarray) -> float:
    _ = jump_target
    return 1e-6


def _fit_jump_process_models(
    design: np.ndarray,
    jump_target: np.ndarray,
    jump_floor: float,
    random_state: int,
    use_log_domain: bool = False,
) -> dict[str, object]:
    event_target = (np.asarray(jump_target, dtype=np.float64) > float(jump_floor)).astype(np.int32, copy=False)
    event_rate = float(event_target.mean()) if event_target.size else 0.0
    positive_jump = np.asarray(jump_target, dtype=np.float64)[event_target > 0]
    positive_jump_median = float(np.nanmedian(positive_jump)) if positive_jump.size else 0.0
    event_model: HistGradientBoostingClassifier | None = None
    severity_model: HistGradientBoostingRegressor | None = None
    if np.unique(event_target).size >= 2 and len(event_target) >= 12:
        event_model = _build_event_model(random_state=random_state)
        event_model.fit(design, event_target)
    if positive_jump.size >= 8:
        severity_target = positive_jump if use_log_domain else np.log1p(positive_jump)
        if np.nanstd(severity_target) >= 1e-8:
            severity_model = _build_residual_model(random_state=random_state)
            severity_model.fit(design[event_target > 0], severity_target)
    return {
        "event_model": event_model,
        "severity_model": severity_model,
        "event_rate": float(event_rate),
        "positive_jump_rows": int(positive_jump.size),
        "positive_jump_median": float(positive_jump_median),
    }


def _predict_jump_process(
    design: np.ndarray,
    event_model: HistGradientBoostingClassifier | None,
    severity_model: HistGradientBoostingRegressor | None,
    event_rate: float,
    positive_jump_median: float,
    use_log_domain: bool = False,
) -> np.ndarray:
    n_rows = int(design.shape[0])
    if event_model is None:
        event_prob = np.full(n_rows, float(np.clip(event_rate, 0.0, 1.0)), dtype=np.float64)
    else:
        event_prob = event_model.predict_proba(design)[:, 1].astype(np.float64, copy=False)
    if severity_model is None:
        jump_size = np.full(n_rows, float(max(positive_jump_median, 0.0)), dtype=np.float64)
    else:
        severity_pred = severity_model.predict(design).astype(np.float64, copy=False)
        jump_size = severity_pred if use_log_domain else np.expm1(severity_pred)
    return np.clip(event_prob, 0.0, 1.0) * np.clip(jump_size, 0.0, None)


def _guarded_funding_prediction(
    anchor_vec: np.ndarray,
    residual_pred: np.ndarray,
    residual_blend: float,
    residual_cap: float,
    use_log_domain: bool = False,
    source_scale: np.ndarray | None = None,
    source_scale_strength: float = 0.0,
    gpd_params: dict[str, float] | None = None,
    gpd_threshold: float = 0.0,
) -> np.ndarray:
    anchor_arr = np.asarray(anchor_vec, dtype=np.float64)
    guarded_residual = np.asarray(residual_pred, dtype=np.float64)
    if np.isfinite(residual_cap):
        guarded_residual = np.clip(guarded_residual, -residual_cap, residual_cap)
    if source_scale is not None and abs(float(source_scale_strength)) > 0.0:
        scale_arr = _resolve_source_scale(source_scale, length=len(anchor_arr))
        guarded_residual = guarded_residual * np.clip(
            1.0 - float(source_scale_strength) * scale_arr,
            0.0,
            2.0,
        )
    if use_log_domain:
        pred = np.expm1(np.log1p(np.clip(anchor_arr, 0.0, None)) + float(residual_blend) * guarded_residual)
    else:
        pred = anchor_arr + float(residual_blend) * guarded_residual
    pred = np.clip(pred, 0.0, None).astype(np.float64, copy=False)

    # P2: Apply GPD tail correction if available
    if gpd_params is not None and gpd_params.get("converged", False) and gpd_threshold > 0:
        pred = apply_gpd_tail_correction(
            predictions=pred,
            anchor=anchor_arr,
            threshold=gpd_threshold,
            gpd_params=gpd_params,
            confidence=0.99,
        )

    return pred


def _anchor_dominance(anchor_vec: np.ndarray, residual: np.ndarray, use_log_domain: bool = False) -> float:
    finite_anchor = np.asarray(anchor_vec, dtype=np.float64)
    if use_log_domain:
        finite_anchor = np.log1p(np.clip(finite_anchor, 0.0, None))
    finite_anchor = np.abs(finite_anchor)
    finite_residual = np.abs(np.asarray(residual, dtype=np.float64))
    anchor_scale = float(np.nanmedian(finite_anchor)) if finite_anchor.size else 0.0
    residual_scale = float(np.nanmedian(finite_residual)) if finite_residual.size else 0.0
    return float(anchor_scale / max(residual_scale, 1e-8)) if anchor_scale > 0.0 else 1.0


def _calibrate_anchor_residual_guard(
    anchor_vec: np.ndarray,
    target_vec: np.ndarray,
    residual_pred: np.ndarray,
    residual_target: np.ndarray,
    anchor_dominance: float,
    use_log_domain: bool = False,
    tail_weight: float = 0.0,
    tail_quantile: float = 0.90,
) -> tuple[float, float, float, float]:
    anchor_arr = np.asarray(anchor_vec, dtype=np.float64)
    target_arr = np.asarray(target_vec, dtype=np.float64)
    pred_arr = np.asarray(residual_pred, dtype=np.float64)
    target_residual = np.asarray(residual_target, dtype=np.float64)
    mask = np.isfinite(anchor_arr) & np.isfinite(target_arr) & np.isfinite(pred_arr) & np.isfinite(target_residual)
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0

    anchor_arr = anchor_arr[mask]
    target_arr = target_arr[mask]
    pred_arr = pred_arr[mask]
    target_residual = target_residual[mask]
    weights = _funding_error_weights(target_arr, tail_weight=tail_weight, tail_quantile=tail_quantile)
    anchor_mae = _weighted_mae(target_arr, anchor_arr, weights)
    positive_mask = target_residual > 1e-8
    positive_residual = np.abs(target_residual[positive_mask])
    residual_scale = float(np.quantile(positive_residual, 0.75)) if positive_residual.size else 0.0
    if residual_scale < 1e-8:
        return 0.0, 0.0, anchor_mae, anchor_mae

    if anchor_dominance >= 3.0:
        blend_grid = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
        cap_grid = (0.25, 0.50, 1.0, 1.5, 2.0, np.inf)
    else:
        blend_grid = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
        cap_grid = (0.50, 1.0, 1.5, 2.0, 4.0, np.inf)

    best_blend = 0.0
    best_cap = 0.0
    best_mae = anchor_mae
    best_jump_mae = (
        float(np.mean(np.abs(anchor_arr[positive_mask] - target_arr[positive_mask]))) if positive_mask.any() else anchor_mae
    )
    for blend in blend_grid:
        for cap_multiplier in cap_grid:
            cap = np.inf if not np.isfinite(cap_multiplier) else float(cap_multiplier * residual_scale)
            candidate = _guarded_funding_prediction(
                anchor_vec=anchor_arr,
                residual_pred=pred_arr,
                residual_blend=float(blend),
                residual_cap=cap,
                use_log_domain=use_log_domain,
            )
            candidate_mae = _weighted_mae(target_arr, candidate, weights)
            candidate_jump_mae = (
                _weighted_mae(target_arr[positive_mask], candidate[positive_mask], weights[positive_mask])
                if positive_mask.any()
                else candidate_mae
            )
            if candidate_mae + 1e-9 < best_mae:
                best_mae = candidate_mae
                best_blend = float(blend)
                best_cap = float(cap)
                best_jump_mae = candidate_jump_mae
            elif abs(candidate_mae - best_mae) <= 1e-9:
                if positive_mask.any() and candidate_jump_mae + 1e-9 < best_jump_mae:
                    best_blend = float(blend)
                    best_cap = float(cap)
                    best_jump_mae = candidate_jump_mae
                elif (
                    abs(candidate_jump_mae - best_jump_mae) <= 1e-9
                    and (
                        float(blend) < best_blend
                        or (abs(float(blend) - best_blend) <= 1e-9 and float(cap) < best_cap)
                    )
                ):
                    best_blend = float(blend)
                    best_cap = float(cap)
    return best_blend, best_cap, anchor_mae, best_mae


def _calibrate_source_scaling_guard(
    anchor_vec: np.ndarray,
    target_vec: np.ndarray,
    residual_pred: np.ndarray,
    residual_blend: float,
    residual_cap: float,
    source_scale: np.ndarray,
    *,
    use_log_domain: bool = False,
    tail_weight: float = 0.0,
    tail_quantile: float = 0.90,
    allow_signed_source_scale: bool = False,
) -> tuple[float, float]:
    anchor_arr = np.asarray(anchor_vec, dtype=np.float64)
    target_arr = np.asarray(target_vec, dtype=np.float64)
    pred_arr = np.asarray(residual_pred, dtype=np.float64)
    scale_arr = np.asarray(source_scale, dtype=np.float64)
    mask = np.isfinite(anchor_arr) & np.isfinite(target_arr) & np.isfinite(pred_arr) & np.isfinite(scale_arr)
    if not mask.any():
        return 0.0, 0.0

    anchor_arr = anchor_arr[mask]
    target_arr = target_arr[mask]
    pred_arr = pred_arr[mask]
    scale_arr = np.clip(scale_arr[mask], 0.0, 1.0)
    weights = _funding_error_weights(target_arr, tail_weight=tail_weight, tail_quantile=tail_quantile)
    best_strength = 0.0
    best_mae = _weighted_mae(
        target_arr,
        _guarded_funding_prediction(
            anchor_vec=anchor_arr,
            residual_pred=pred_arr,
            residual_blend=residual_blend,
            residual_cap=residual_cap,
            use_log_domain=use_log_domain,
        ),
        weights,
    )
    if not np.any(scale_arr > 1e-8):
        return best_strength, best_mae

    if allow_signed_source_scale:
        strength_grid = (-1.0, -0.75, -0.60, -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0)
    else:
        strength_grid = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0)

    for strength in strength_grid:
        candidate = _guarded_funding_prediction(
            anchor_vec=anchor_arr,
            residual_pred=pred_arr,
            residual_blend=residual_blend,
            residual_cap=residual_cap,
            use_log_domain=use_log_domain,
            source_scale=scale_arr,
            source_scale_strength=float(strength),
        )
        candidate_mae = _weighted_mae(target_arr, candidate, weights)
        if candidate_mae + 1e-9 < best_mae:
            best_strength = float(strength)
            best_mae = candidate_mae
        elif abs(candidate_mae - best_mae) <= 1e-9 and abs(float(strength)) < abs(best_strength):
            best_strength = float(strength)
    return best_strength, best_mae


def _funding_error_weights(target: np.ndarray, *, tail_weight: float, tail_quantile: float) -> np.ndarray:
    weights = np.ones(len(target), dtype=np.float64)
    if float(tail_weight) <= 0.0 or len(target) == 0:
        return weights
    threshold = float(np.quantile(np.asarray(target, dtype=np.float64), float(np.clip(tail_quantile, 0.50, 0.99))))
    weights[np.asarray(target, dtype=np.float64) >= threshold] += float(tail_weight)
    return weights


def _weighted_mae(target: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    target_arr = np.asarray(target, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    return float(np.average(np.abs(target_arr - pred_arr), weights=weight_arr))


def _resolve_source_scale(source_scale: np.ndarray | None, length: int) -> np.ndarray:
    if source_scale is None:
        return np.zeros(length, dtype=np.float64)
    scale_arr = np.asarray(source_scale, dtype=np.float64).reshape(-1)
    if scale_arr.size != length:
        raise ValueError("Funding source scale length does not match lane_state rows")
    return np.clip(np.nan_to_num(scale_arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _guard_improvement_ratio(baseline_mae: float, guarded_mae: float) -> float:
    baseline = float(max(abs(baseline_mae), 1e-9))
    return float(np.clip((float(baseline_mae) - float(guarded_mae)) / baseline, 0.0, 1.0))


def _merge_features(lane_state: np.ndarray, aux_features: np.ndarray | None, anchor: np.ndarray) -> np.ndarray:
    lane = np.asarray(lane_state, dtype=np.float32)
    parts = [lane]
    if aux_features is not None:
        aux = np.asarray(aux_features, dtype=np.float32)
        if aux.ndim == 1:
            aux = aux[:, None]
        parts.append(aux)
    anchor_vec = np.asarray(anchor, dtype=np.float32).reshape(-1, 1)
    parts.append(anchor_vec)
    parts.append(np.log1p(np.clip(anchor_vec, 0.0, None)).astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _resolve_anchor(anchor: np.ndarray | None, fallback: float, length: int) -> np.ndarray:
    if anchor is None:
        return np.full(length, fallback, dtype=np.float64)
    anchor_vec = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor_vec.size != length:
        raise ValueError("Funding anchor length does not match lane_state rows")
    missing = ~np.isfinite(anchor_vec)
    if missing.any():
        anchor_vec = anchor_vec.copy()
        anchor_vec[missing] = fallback
    return anchor_vec


def _fit_gpd_for_funding(
    jump_target: np.ndarray,
    tail_quantile: float,
    min_exceedances: int = 15,
) -> tuple[float, dict[str, float]]:
    """Fit GPD to funding jump exceedances above the tail_quantile threshold.

    Returns (threshold, gpd_params).
    """
    arr = np.asarray(jump_target, dtype=np.float64)
    positive = arr[arr > 1e-6]
    if positive.size < min_exceedances * 2:
        return 0.0, {"xi": 0.0, "sigma": 0.0, "n_exceedances": 0, "converged": False}

    threshold = float(np.quantile(positive, float(np.clip(tail_quantile, 0.50, 0.99))))
    if threshold < 1e-8:
        return 0.0, {"xi": 0.0, "sigma": 0.0, "n_exceedances": 0, "converged": False}

    exceedances = positive[positive > threshold] - threshold
    gpd_params = fit_gpd_pot(exceedances, min_exceedances=min_exceedances)
    return threshold, gpd_params


def _fit_cqr_for_funding(
    train_design: np.ndarray,
    train_jump: np.ndarray,
    cal_design: np.ndarray,
    cal_jump: np.ndarray,
    alpha: float = 0.10,
    random_state: int = 0,
    min_calibration_rows: int = 8,
) -> tuple[
    HistGradientBoostingRegressor | None,
    HistGradientBoostingRegressor | None,
    float,
    bool,
]:
    """Fit CQR quantile regressors for funding prediction intervals.

    1. Train two HGBR quantile regressors at α/2 and 1-α/2 on training jump data.
    2. On calibration set, compute conformity scores.
    3. Compute conformal quantile Q for finite-sample coverage.

    Returns (q_lo_model, q_hi_model, conformal_q, converged).
    """
    y_train = np.asarray(train_jump, dtype=np.float64)
    y_cal = np.asarray(cal_jump, dtype=np.float64)
    if y_cal.size < min_calibration_rows or y_train.size < min_calibration_rows:
        return None, None, 0.0, False

    lo_quantile = alpha / 2.0
    hi_quantile = 1.0 - alpha / 2.0

    base_params = dict(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        min_samples_leaf=4,
        random_state=random_state,
    )
    q_lo_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=lo_quantile, **base_params
    )
    q_hi_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=hi_quantile, **base_params
    )

    q_lo_model.fit(train_design, y_train)
    q_hi_model.fit(train_design, y_train)

    q_lo_cal = q_lo_model.predict(cal_design)
    q_hi_cal = q_hi_model.predict(cal_design)
    scores = cqr_conformity_scores(y_cal, q_lo_cal, q_hi_cal)
    conformal_q = cqr_conformal_quantile(scores, alpha=alpha)

    return q_lo_model, q_hi_model, conformal_q, True