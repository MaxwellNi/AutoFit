#!/usr/bin/env python3
"""Investors lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from ..intensity_utils import (
    fit_intensity_baseline,
    hawkes_intensity_features,
    intensity_diagnostics,
    predict_intensity_probability,
)
from ..shrinkage_utils import (
    apply_shrinkage,
    fit_shrinkage_gate,
    predict_shrinkage_alpha,
    shrinkage_diagnostics,
)


@dataclass(frozen=True)
class InvestorsLaneSpec:
    lane_name: str = "investors"
    supports_h1_occurrence_head: bool = True
    supports_short_horizon_exemplar_branch: bool = True
    supports_hurdle_decomposition: bool = True
    supports_transition_dynamics: bool = True
    supports_marked_event_features: bool = True
    supports_source_pollution_guard: bool = True
    guardrails: Tuple[str, ...] = ("shared_count_repair", "h1_only_sharpness", "source_rich_miscalibration")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_h1_occurrence_head": self.supports_h1_occurrence_head,
            "supports_short_horizon_exemplar_branch": self.supports_short_horizon_exemplar_branch,
            "supports_hurdle_decomposition": self.supports_hurdle_decomposition,
            "supports_transition_dynamics": self.supports_transition_dynamics,
            "supports_marked_event_features": self.supports_marked_event_features,
            "supports_source_pollution_guard": self.supports_source_pollution_guard,
            "guardrails": self.guardrails,
        }


class InvestorsLaneRuntime:
    def __init__(self, spec: InvestorsLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or InvestorsLaneSpec()
        self.random_state = int(random_state)
        self._occurrence_model: HistGradientBoostingClassifier | None = None
        self._positive_model: HistGradientBoostingRegressor | None = None
        self._positive_specialists: Dict[int, HistGradientBoostingRegressor] = {}
        self._exemplar_model: KNeighborsRegressor | None = None
        self._fallback_value = 0.0
        self._active_rate = 0.0
        self._anchor_blend = 0.0
        self._source_anchor_blend_by_profile: Dict[int, float] = {}
        self._source_anchor_blend_reliability_by_profile: Dict[int, float] = {}
        self._source_transition_strength_by_profile: Dict[int, float] = {}
        self._source_transition_reliability_by_profile: Dict[int, float] = {}
        self._horizon = 1
        self._use_learned_occurrence = True
        self._use_hurdle_head = False
        self._use_count_jump = False
        self._use_sparsity_gate = False
        self._count_jump_strength = 0.0
        self._count_sparsity_gate_strength = 0.0
        self._horizon_anchor_mix = 0.0
        self._horizon_anchor_mix_reliability = 0.0
        self._global_anchor_blend = 0.0
        self._global_anchor_blend_reliability = 0.0
        self._global_jump_strength = 0.0
        self._global_jump_reliability = 0.0
        self._use_intensity_baseline = False
        self._intensity_model: HistGradientBoostingRegressor | None = None
        self._intensity_converged = False
        self._intensity_blend = 0.5
        self._intensity_diagnostics: dict[str, float] = {}
        self._use_shrinkage_gate = False
        self._shrinkage_model: HistGradientBoostingRegressor | None = None
        self._shrinkage_converged = False
        self._shrinkage_strength = 0.8
        self._shrinkage_diagnostics: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        source_features: np.ndarray | None = None,
        mark_features: np.ndarray | None = None,
        enable_source_features: bool = False,
        enable_source_specialists: bool = False,
        enable_source_guard: bool = False,
        enable_source_read_policy: bool = False,
        enable_source_transition_correction: bool = False,
        enable_mark_features: bool = False,
        enable_hurdle_head: bool = False,
        enable_count_jump: bool = False,
        count_jump_strength: float = 0.0,
        enable_count_sparsity_gate: bool = False,
        count_sparsity_gate_strength: float = 0.0,
        horizon: int = 1,
        anchor_blend: float = 0.0,
        task_name: str = "",
        enable_intensity_baseline: bool = False,
        intensity_blend: float = 0.5,
        enable_shrinkage_gate: bool = False,
        shrinkage_strength: float = 0.8,
    ) -> "InvestorsLaneRuntime":
        target = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
        self._fallback_value = float(np.nanmedian(target)) if target.size else 0.0
        self._horizon = int(horizon)
        self._anchor_blend = float(np.clip(anchor_blend, 0.0, 1.0))
        self._use_learned_occurrence = str(task_name) == "task2_forecast"
        self._use_hurdle_head = bool(enable_hurdle_head and self._horizon > 1)
        self._use_count_jump = bool(enable_count_jump and self._horizon > 1)
        self._use_sparsity_gate = bool(enable_count_sparsity_gate and self._horizon > 1)
        self._count_jump_strength = float(np.clip(count_jump_strength, 0.0, 1.0))
        self._count_sparsity_gate_strength = float(np.clip(count_sparsity_gate_strength, 0.0, 1.0))
        self._horizon_anchor_mix = 0.0
        self._horizon_anchor_mix_reliability = 0.0
        self._global_anchor_blend = self._anchor_blend
        self._global_anchor_blend_reliability = 0.0
        self._global_jump_strength = 0.0
        self._global_jump_reliability = 0.0
        self._use_intensity_baseline = bool(enable_intensity_baseline)
        self._intensity_model = None
        self._intensity_converged = False
        self._intensity_blend = float(np.clip(intensity_blend, 0.0, 1.0))
        self._intensity_diagnostics = {}
        self._use_shrinkage_gate = bool(enable_shrinkage_gate and self._horizon > 1)
        self._shrinkage_model = None
        self._shrinkage_converged = False
        self._shrinkage_strength = float(np.clip(shrinkage_strength, 0.0, 1.0))
        self._shrinkage_diagnostics = {}
        if target.size == 0:
            self._fitted = True
            return self

        has_source_features = source_features is not None
        use_source_features = bool(enable_source_features and has_source_features)
        has_mark_features = mark_features is not None
        use_mark_features = bool(enable_mark_features and has_mark_features)
        use_source_specialists = bool(enable_source_specialists and has_source_features)
        use_source_guard = bool(enable_source_guard and has_source_features)
        use_source_read_policy = bool(enable_source_read_policy and has_source_features)
        use_source_transition_correction = bool(enable_source_transition_correction and has_source_features)
        normalized_source = _normalize_source_features(source_features) if has_source_features else None
        normalized_mark = _normalize_mark_features(mark_features) if has_mark_features else None
        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=target.size)
        smoothed_anchor = _horizon_model_anchor(anchor_vec, aux_features, self._horizon, self._use_hurdle_head)
        (
            self._horizon_anchor_mix,
            self._horizon_anchor_mix_reliability,
        ) = _calibrate_horizon_anchor_mix(
            raw_anchor=anchor_vec,
            smoothed_anchor=smoothed_anchor,
            target=target,
            enabled=self._use_hurdle_head,
        )
        model_anchor = _blend_horizon_anchor(anchor_vec, smoothed_anchor, self._horizon_anchor_mix)
        design = _merge_features(
            lane_state,
            aux_features,
            model_anchor if self._use_hurdle_head else anchor_vec,
            normalized_source if use_source_features else None,
            normalized_mark if use_mark_features else None,
        )
        active = (target > 0.0).astype(np.int32)
        self._active_rate = float(active.mean())
        self._positive_specialists = {}
        self._source_anchor_blend_by_profile = {}
        self._source_anchor_blend_reliability_by_profile = {}
        self._source_transition_strength_by_profile = {}
        self._source_transition_reliability_by_profile = {}

        if self._use_learned_occurrence and np.unique(active).size >= 2 and len(active) >= 12:
            self._occurrence_model = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=150,
                learning_rate=0.05,
                random_state=self.random_state,
            )
            self._occurrence_model.fit(design, active)

        # --- Intensity baseline (P3 Marked TPP) ---
        if self._use_intensity_baseline:
            intensity_feats = hawkes_intensity_features(aux_features, model_anchor)
            self._intensity_model, self._intensity_converged = fit_intensity_baseline(
                intensity_feats,
                active,
                random_state=self.random_state,
            )
            if self._intensity_converged:
                intensity_proba = predict_intensity_probability(
                    self._intensity_model, intensity_feats, self._active_rate
                )
                self._intensity_diagnostics = intensity_diagnostics(active, intensity_proba)

        positive_mask = target > 0.0
        if positive_mask.any():
            positive_design = design[positive_mask]
            positive_target = target[positive_mask]
            positive_anchor = model_anchor[positive_mask]
            positive_residual = np.log1p(positive_target) - np.log1p(np.clip(positive_anchor, 0.0, None))
            if positive_target.size >= 8 and np.nanstd(positive_residual) >= 1e-8:
                self._positive_model = HistGradientBoostingRegressor(
                    max_depth=3,
                    max_iter=200,
                    learning_rate=0.05,
                    random_state=self.random_state,
                )
                self._positive_model.fit(positive_design, positive_residual)
                if normalized_source is not None and use_source_specialists:
                    positive_source = normalized_source[positive_mask]
                    positive_profiles = _source_profile_ids(positive_source)
                    for profile_id in (1, 2, 3):
                        profile_mask = positive_profiles == profile_id
                        if int(profile_mask.sum()) < 24:
                            continue
                        profile_residual = positive_residual[profile_mask]
                        if np.nanstd(profile_residual) < 1e-8:
                            continue
                        specialist = HistGradientBoostingRegressor(
                            max_depth=3,
                            max_iter=150,
                            learning_rate=0.05,
                            random_state=self.random_state + 17 * profile_id,
                        )
                        specialist.fit(positive_design[profile_mask], profile_residual)
                        self._positive_specialists[profile_id] = specialist
            if self._horizon == 1 and positive_target.size >= 5:
                neighbors = min(15, positive_target.size)
                self._exemplar_model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
                self._exemplar_model.fit(positive_design, positive_residual)

            base_positive_pred = _compose_positive_predictions(
                positive_design=design,
                base_log=np.log1p(np.clip(model_anchor, 0.0, None)),
                positive_model=self._positive_model,
                exemplar_model=self._exemplar_model if self._horizon == 1 else None,
            )
            active_prob = _compute_active_probability(
                design=design,
                anchor=model_anchor,
                aux_features=aux_features,
                occurrence_model=self._occurrence_model,
                use_learned_occurrence=self._use_learned_occurrence,
                active_rate=self._active_rate,
                use_sparsity_gate=self._use_sparsity_gate,
                count_sparsity_gate_strength=self._count_sparsity_gate_strength,
                horizon=self._horizon,
            )
            raw_pred = np.clip(active_prob * np.clip(base_positive_pred, 0.0, None), 0.0, None)
            if self._use_count_jump:
                jump_target = _jump_target(model_anchor, aux_features, self._horizon)
                (
                    self._global_jump_strength,
                    self._global_jump_reliability,
                ) = _calibrate_global_jump_strength(
                    base_pred=raw_pred,
                    jump_target=jump_target,
                    target=target,
                    requested_strength=self._count_jump_strength,
                )
                jump_strength = _global_jump_effective_strength(
                    strength=self._global_jump_strength,
                    reliability=self._global_jump_reliability,
                    stability=_history_anchor_stability(aux_features),
                )
                raw_pred = (1.0 - jump_strength) * raw_pred + jump_strength * jump_target
            (
                self._global_anchor_blend,
                self._global_anchor_blend_reliability,
            ) = _calibrate_global_anchor_blend(
                base_pred=raw_pred,
                anchor=model_anchor,
                target=target,
                default_blend=self._anchor_blend,
            )

            # --- P5 Adaptive Shrinkage Gate ---
            if self._use_shrinkage_gate:
                self._shrinkage_model, self._shrinkage_diagnostics = fit_shrinkage_gate(
                    design=design,
                    learned_pred=raw_pred,
                    anchor=model_anchor,
                    target=target,
                    random_state=self.random_state + 997,
                )
                self._shrinkage_converged = (
                    self._shrinkage_model is not None
                    and self._shrinkage_diagnostics.get("shrinkage_status") == "converged"
                )

            if normalized_source is not None and use_source_read_policy:
                positive_profiles = _source_profile_ids(normalized_source[positive_mask])
                positive_confidence = _source_confidence(normalized_source[positive_mask])
                positive_stability = _history_anchor_stability(
                    aux_features[positive_mask]
                    if aux_features is not None
                    else np.zeros((positive_target.size, 0), dtype=np.float32)
                )
                calibration_mask = _read_policy_calibration_mask(positive_profiles)
                calibration_design = positive_design
                calibration_anchor = positive_anchor
                calibration_target = positive_target
                calibration_profiles = positive_profiles
                calibration_confidence = positive_confidence
                calibration_stability = positive_stability
                if calibration_mask.any() and (~calibration_mask).sum() >= 24:
                    calibration_design = positive_design[calibration_mask]
                    calibration_anchor = positive_anchor[calibration_mask]
                    calibration_target = positive_target[calibration_mask]
                    calibration_profiles = positive_profiles[calibration_mask]
                    calibration_confidence = positive_confidence[calibration_mask]
                    calibration_stability = positive_stability[calibration_mask]
                    base_positive_pred = _fit_reference_positive_predictions(
                        train_design=positive_design[~calibration_mask],
                        train_target=positive_target[~calibration_mask],
                        train_anchor=positive_anchor[~calibration_mask],
                        test_design=calibration_design,
                        test_anchor=calibration_anchor,
                        horizon=self._horizon,
                        random_state=self.random_state + 503,
                    )
                else:
                    base_positive_pred = _compose_positive_predictions(
                        positive_design=positive_design,
                        base_log=np.log1p(np.clip(positive_anchor, 0.0, None)),
                        positive_model=self._positive_model,
                        exemplar_model=self._exemplar_model if self._horizon == 1 else None,
                    )
                default_blend = float(np.clip(self._global_anchor_blend, 0.0, 1.0))
                candidate_blends = sorted({0.0, 0.2, 0.4, 0.6, default_blend, 0.85, 1.0})
                for profile_id in np.unique(calibration_profiles):
                    profile_mask = calibration_profiles == profile_id
                    if int(profile_mask.sum()) < 24:
                        continue
                    default_mae = float(
                        np.mean(
                            np.abs(
                                ((1.0 - default_blend) * base_positive_pred[profile_mask] + default_blend * calibration_anchor[profile_mask])
                                - calibration_target[profile_mask]
                            )
                        )
                    )
                    best_blend = default_blend
                    best_mae = default_mae
                    for blend in candidate_blends:
                        candidate_pred = (
                            (1.0 - blend) * base_positive_pred[profile_mask] + blend * calibration_anchor[profile_mask]
                        )
                        candidate_mae = float(np.mean(np.abs(candidate_pred - calibration_target[profile_mask])))
                        if candidate_mae + 1e-9 < best_mae:
                            best_mae = candidate_mae
                            best_blend = float(blend)
                    prior_blend = _source_read_policy_prior_blend(
                        profile_id=int(profile_id),
                        default_blend=default_blend,
                        confidence=float(np.mean(calibration_confidence[profile_mask])),
                        stability=float(np.mean(calibration_stability[profile_mask])),
                    )
                    target_blend = float(
                        _regularize_read_policy_blend(
                            profile_id=int(profile_id),
                            calibrated_blend=float(best_blend),
                            prior_blend=float(prior_blend),
                        )
                    )
                    reliability = _read_policy_reliability(default_mae=default_mae, best_mae=best_mae)
                    self._source_anchor_blend_by_profile[int(profile_id)] = float(target_blend)
                    self._source_anchor_blend_reliability_by_profile[int(profile_id)] = float(reliability)

            if normalized_source is not None and use_source_transition_correction:
                transition_profiles = _source_profile_ids(normalized_source)
                transition_confidence = _source_confidence(normalized_source)
                transition_stability = _history_anchor_stability(
                    aux_features if aux_features is not None else np.zeros((target.size, 0), dtype=np.float32)
                )
                calibration_mask = _read_policy_calibration_mask(transition_profiles)

                prev_fitted = self._fitted
                self._fitted = True
                try:
                    base_pred = self.predict(
                        lane_state,
                        aux_features=aux_features,
                        anchor=anchor,
                        source_features=source_features,
                        mark_features=mark_features,
                        enable_source_features=enable_source_features,
                        enable_source_specialists=enable_source_specialists,
                        enable_source_guard=enable_source_guard,
                        enable_source_read_policy=enable_source_read_policy,
                        enable_source_transition_correction=False,
                        enable_mark_features=enable_mark_features,
                    )
                finally:
                    self._fitted = prev_fitted

                calibration_pred = base_pred
                calibration_target = target
                calibration_anchor = model_anchor
                calibration_profiles = transition_profiles
                calibration_confidence = transition_confidence
                calibration_stability = transition_stability
                calibration_aux = aux_features
                if calibration_mask.any() and (~calibration_mask).sum() >= 24:
                    calibration_pred = base_pred[calibration_mask]
                    calibration_target = target[calibration_mask]
                    calibration_anchor = model_anchor[calibration_mask]
                    calibration_profiles = transition_profiles[calibration_mask]
                    calibration_confidence = transition_confidence[calibration_mask]
                    calibration_stability = transition_stability[calibration_mask]
                    if aux_features is not None:
                        calibration_aux = aux_features[calibration_mask]

                transition_signal = _transition_signal(calibration_aux, calibration_anchor)
                transition_target = np.clip(calibration_anchor + transition_signal, 0.0, None)
                candidate_strengths = sorted({0.0, 0.08, 0.15, 0.25, 0.40, 0.55})
                for profile_id in np.unique(calibration_profiles):
                    profile_mask = calibration_profiles == profile_id
                    if int(profile_mask.sum()) < 24:
                        continue
                    default_mae = float(np.mean(np.abs(calibration_pred[profile_mask] - calibration_target[profile_mask])))
                    best_strength = 0.0
                    best_mae = default_mae
                    for strength in candidate_strengths:
                        candidate_pred = (
                            (1.0 - float(strength)) * calibration_pred[profile_mask]
                            + float(strength) * transition_target[profile_mask]
                        )
                        candidate_mae = float(np.mean(np.abs(candidate_pred - calibration_target[profile_mask])))
                        if candidate_mae + 1e-9 < best_mae:
                            best_mae = candidate_mae
                            best_strength = float(strength)
                    prior_strength = _source_transition_prior_strength(
                        profile_id=int(profile_id),
                        confidence=float(np.mean(calibration_confidence[profile_mask])),
                        stability=float(np.mean(calibration_stability[profile_mask])),
                    )
                    target_strength = _regularize_transition_strength(
                        profile_id=int(profile_id),
                        calibrated_strength=float(best_strength),
                        prior_strength=float(prior_strength),
                    )
                    reliability = _transition_reliability(default_mae=default_mae, best_mae=best_mae)
                    self._source_transition_strength_by_profile[int(profile_id)] = float(target_strength)
                    self._source_transition_reliability_by_profile[int(profile_id)] = float(reliability)

        self._fitted = True
        return self

    def predict(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        source_features: np.ndarray | None = None,
        mark_features: np.ndarray | None = None,
        enable_source_features: bool = False,
        enable_source_specialists: bool = False,
        enable_source_guard: bool = False,
        enable_source_read_policy: bool = False,
        enable_source_transition_correction: bool = False,
        enable_mark_features: bool = False,
    ) -> np.ndarray:
        if not self._fitted:
            raise ValueError("InvestorsLaneRuntime is not fitted")

        has_source_features = source_features is not None
        use_source_features = bool(enable_source_features and has_source_features)
        has_mark_features = mark_features is not None
        use_mark_features = bool(enable_mark_features and has_mark_features)
        use_source_specialists = bool(enable_source_specialists and has_source_features)
        use_source_guard = bool(enable_source_guard and has_source_features)
        use_source_read_policy = bool(enable_source_read_policy and has_source_features)
        use_source_transition_correction = bool(enable_source_transition_correction and has_source_features)
        normalized_source = _normalize_source_features(source_features) if has_source_features else None
        normalized_mark = _normalize_mark_features(mark_features) if has_mark_features else None
        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=len(lane_state))
        smoothed_anchor = _horizon_model_anchor(anchor_vec, aux_features, self._horizon, self._use_hurdle_head)
        model_anchor = _blend_horizon_anchor(anchor_vec, smoothed_anchor, self._horizon_anchor_mix)
        design = _merge_features(
            lane_state,
            aux_features,
            model_anchor if self._use_hurdle_head else anchor_vec,
            normalized_source if use_source_features else None,
            normalized_mark if use_mark_features else None,
        )
        active_prob = _compute_active_probability(
            design=design,
            anchor=model_anchor,
            aux_features=aux_features,
            occurrence_model=self._occurrence_model,
            use_learned_occurrence=self._use_learned_occurrence,
            active_rate=self._active_rate,
            use_sparsity_gate=self._use_sparsity_gate,
            count_sparsity_gate_strength=self._count_sparsity_gate_strength,
            horizon=self._horizon,
        )

        # --- Intensity blend (P3 Marked TPP) ---
        if self._use_intensity_baseline and self._intensity_converged:
            intensity_feats = hawkes_intensity_features(aux_features, model_anchor)
            intensity_prob = predict_intensity_probability(
                self._intensity_model, intensity_feats, self._active_rate
            )
            blend = self._intensity_blend
            active_prob = (1.0 - blend) * active_prob + blend * intensity_prob

        base_log = np.log1p(np.clip(model_anchor, 0.0, None))
        if self._positive_model is None:
            positive_pred = np.clip(model_anchor, 0.0, None).astype(np.float64, copy=False)
        else:
            positive_residual = self._positive_model.predict(design).astype(np.float64, copy=False)
            if self._exemplar_model is not None and self._horizon == 1:
                exemplar_residual = self._exemplar_model.predict(design).astype(np.float64, copy=False)
                positive_residual = 0.6 * positive_residual + 0.4 * exemplar_residual
            if normalized_source is not None and self._positive_specialists:
                profile_ids = _source_profile_ids(normalized_source)
                source_confidence = _source_confidence(normalized_source)
                for profile_id, specialist in self._positive_specialists.items():
                    profile_mask = profile_ids == profile_id
                    if not profile_mask.any():
                        continue
                    specialist_residual = specialist.predict(design[profile_mask]).astype(np.float64, copy=False)
                    blend = _specialist_blend(profile_id, source_confidence[profile_mask])
                    positive_residual[profile_mask] = (
                        (1.0 - blend) * positive_residual[profile_mask] + blend * specialist_residual
                    )
            positive_pred = np.expm1(base_log + positive_residual)
            if normalized_source is not None and use_source_guard:
                guard = _source_guard_weight(normalized_source)
                positive_pred = guard * positive_pred + (1.0 - guard) * np.clip(model_anchor, 0.0, None)

        pred = np.clip(active_prob * np.clip(positive_pred, 0.0, None), 0.0, None)
        if self._use_count_jump and self._global_jump_strength > 0.0:
            jump_target = _jump_target(model_anchor, aux_features, self._horizon)
            jump_strength = _global_jump_effective_strength(
                strength=self._global_jump_strength,
                reliability=self._global_jump_reliability,
                stability=_history_anchor_stability(aux_features),
            )
            pred = (1.0 - jump_strength) * pred + jump_strength * jump_target

        # --- P5 Adaptive Shrinkage Gate ---
        if self._use_shrinkage_gate and self._shrinkage_converged:
            shrinkage_alpha = predict_shrinkage_alpha(
                self._shrinkage_model, design
            )
            pred = apply_shrinkage(pred, model_anchor, shrinkage_alpha, self._shrinkage_strength)
            pred = np.clip(pred, 0.0, None)

        if anchor is not None and (
            self._global_anchor_blend > 0.0
            or (normalized_source is not None and use_source_read_policy and self._source_anchor_blend_by_profile)
        ):
            anchor_blend = np.full(len(pred), self._global_anchor_blend, dtype=np.float64)
            if normalized_source is not None and use_source_read_policy and self._source_anchor_blend_by_profile:
                profile_ids = _source_profile_ids(normalized_source)
                confidence = _source_confidence(normalized_source)
                for profile_id, profile_blend in self._source_anchor_blend_by_profile.items():
                    profile_mask = profile_ids == int(profile_id)
                    if not profile_mask.any():
                        continue
                    reliability = self._source_anchor_blend_reliability_by_profile.get(int(profile_id), 0.0)
                    effective_strength = _read_policy_strength(
                        default_blend=self._global_anchor_blend,
                        target_blend=float(profile_blend),
                        reliability=float(reliability),
                    )
                    anchor_blend[profile_mask] = np.clip(
                        self._global_anchor_blend
                        + effective_strength * confidence[profile_mask] * (float(profile_blend) - self._global_anchor_blend),
                        0.0,
                        1.0,
                    )
            pred = (1.0 - anchor_blend) * pred + anchor_blend * model_anchor
        if normalized_source is not None and use_source_transition_correction and self._source_transition_strength_by_profile:
            profile_ids = _source_profile_ids(normalized_source)
            confidence = _source_confidence(normalized_source)
            stability = _history_anchor_stability(
                aux_features if aux_features is not None else np.zeros((len(pred), 0), dtype=np.float32)
            )
            transition_signal = _transition_signal(aux_features, model_anchor)
            transition_target = np.clip(model_anchor + transition_signal, 0.0, None)
            transition_strength = np.zeros(len(pred), dtype=np.float64)
            for profile_id, profile_strength in self._source_transition_strength_by_profile.items():
                profile_mask = profile_ids == int(profile_id)
                if not profile_mask.any():
                    continue
                reliability = self._source_transition_reliability_by_profile.get(int(profile_id), 0.0)
                transition_strength[profile_mask] = _transition_effective_strength(
                    strength=float(profile_strength),
                    reliability=float(reliability),
                    confidence=confidence[profile_mask],
                    stability=stability[profile_mask],
                )
            pred = (1.0 - transition_strength) * pred + transition_strength * transition_target
        return pred.astype(np.float64, copy=False)

    def describe_intensity(self) -> dict:
        """Return intensity model diagnostics."""
        base = {
            "intensity_enabled": self._use_intensity_baseline,
            "intensity_converged": self._intensity_converged,
            "intensity_blend": self._intensity_blend,
        }
        base.update(self._intensity_diagnostics)
        return base

    def describe_shrinkage(self) -> dict:
        """Return shrinkage gate diagnostics."""
        base = {
            "shrinkage_enabled": self._use_shrinkage_gate,
            "shrinkage_converged": self._shrinkage_converged,
            "shrinkage_strength": self._shrinkage_strength,
        }
        base.update(self._shrinkage_diagnostics)
        return base


def _merge_features(
    lane_state: np.ndarray,
    aux_features: np.ndarray | None,
    anchor: np.ndarray | None,
    source_features: np.ndarray | None = None,
    mark_features: np.ndarray | None = None,
) -> np.ndarray:
    lane = np.asarray(lane_state, dtype=np.float32)
    parts = [lane]
    if aux_features is not None:
        aux = np.asarray(aux_features, dtype=np.float32)
        if aux.ndim == 1:
            aux = aux[:, None]
        parts.append(aux)
    if anchor is not None:
        anchor_vec = np.asarray(anchor, dtype=np.float32).reshape(-1, 1)
        parts.append(anchor_vec)
        parts.append(np.log1p(np.clip(anchor_vec, 0.0, None)).astype(np.float32, copy=False))
    if source_features is not None:
        source = _normalize_source_features(source_features)
        parts.append(source)
    if mark_features is not None:
        marks = _normalize_mark_features(mark_features)
        parts.append(marks)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _normalize_source_features(source_features: np.ndarray) -> np.ndarray:
    source = np.asarray(source_features, dtype=np.float32)
    if source.ndim == 1:
        source = source[:, None]
    return source


def _normalize_mark_features(mark_features: np.ndarray) -> np.ndarray:
    marks = np.asarray(mark_features, dtype=np.float32)
    if marks.ndim == 1:
        marks = marks[:, None]
    return marks


def _resolve_anchor(anchor: np.ndarray | None, fallback: float, length: int) -> np.ndarray:
    if anchor is None:
        return np.full(length, max(fallback, 0.0), dtype=np.float64)
    anchor_vec = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor_vec.size != length:
        raise ValueError("Investors anchor length does not match lane_state rows")
    missing = ~np.isfinite(anchor_vec)
    if missing.any():
        anchor_vec = anchor_vec.copy()
        anchor_vec[missing] = max(fallback, 0.0)
    return np.clip(anchor_vec, 0.0, None)


def _anchor_activity_gate(anchor: np.ndarray, fallback: float) -> np.ndarray:
    anchor_vec = np.clip(np.asarray(anchor, dtype=np.float64).reshape(-1), 0.0, None)
    gate = 1.0 - np.exp(-anchor_vec)
    return np.clip(np.maximum(gate, float(np.clip(fallback, 0.0, 1.0))), 0.0, 1.0)


def _history_components(
    aux_features: np.ndarray | None,
    fallback_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_vec = np.clip(np.asarray(fallback_anchor, dtype=np.float64).reshape(-1), 0.0, None)
    if aux_features is None:
        zeros = np.zeros_like(anchor_vec)
        ones = np.ones_like(anchor_vec)
        return anchor_vec, anchor_vec, anchor_vec, zeros, ones
    aux = np.asarray(aux_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux[:, None]
    if aux.shape[0] != anchor_vec.shape[0]:
        raise ValueError("Investors aux_features length does not match anchor length")
    lag1 = np.clip(aux[:, 0], 0.0, None) if aux.shape[1] >= 1 else anchor_vec
    roll3 = np.clip(aux[:, 1], 0.0, None) if aux.shape[1] >= 2 else lag1
    roll7 = np.clip(aux[:, 2], 0.0, None) if aux.shape[1] >= 3 else roll3
    roll_std = np.clip(np.abs(aux[:, 3]), 0.0, None) if aux.shape[1] >= 4 else np.zeros_like(anchor_vec)
    history_count = np.clip(aux[:, 4], 0.0, None) if aux.shape[1] >= 5 else np.ones_like(anchor_vec)
    return lag1, roll3, roll7, roll_std, history_count


def _horizon_model_anchor(
    anchor: np.ndarray,
    aux_features: np.ndarray | None,
    horizon: int,
    enabled: bool,
) -> np.ndarray:
    anchor_vec = np.clip(np.asarray(anchor, dtype=np.float64).reshape(-1), 0.0, None)
    if not enabled or int(horizon) <= 1:
        return anchor_vec
    lag1, roll3, roll7, _, history_count = _history_components(aux_features, anchor_vec)
    if int(horizon) <= 7:
        weights = (0.25, 0.45, 0.30)
    elif int(horizon) <= 14:
        weights = (0.10, 0.35, 0.55)
    else:
        weights = (0.05, 0.20, 0.75)
    smoothed = weights[0] * lag1 + weights[1] * roll3 + weights[2] * roll7
    stability = np.clip(1.0 - np.exp(-history_count / 3.0), 0.0, 1.0)
    blended = stability * smoothed + (1.0 - stability) * anchor_vec
    return np.clip(blended, 0.0, None).astype(np.float64, copy=False)


def _blend_horizon_anchor(
    raw_anchor: np.ndarray,
    smoothed_anchor: np.ndarray,
    mix: float,
) -> np.ndarray:
    raw = np.clip(np.asarray(raw_anchor, dtype=np.float64).reshape(-1), 0.0, None)
    smoothed = np.clip(np.asarray(smoothed_anchor, dtype=np.float64).reshape(-1), 0.0, None)
    blend = float(np.clip(mix, 0.0, 1.0))
    return np.clip((1.0 - blend) * raw + blend * smoothed, 0.0, None).astype(np.float64, copy=False)


def _calibrate_horizon_anchor_mix(
    raw_anchor: np.ndarray,
    smoothed_anchor: np.ndarray,
    target: np.ndarray,
    enabled: bool,
) -> tuple[float, float]:
    if not enabled:
        return 0.0, 0.0
    raw = np.clip(np.asarray(raw_anchor, dtype=np.float64).reshape(-1), 0.0, None)
    smoothed = np.clip(np.asarray(smoothed_anchor, dtype=np.float64).reshape(-1), 0.0, None)
    target_vec = np.clip(np.asarray(target, dtype=np.float64).reshape(-1), 0.0, None)
    if raw.shape != smoothed.shape or raw.shape != target_vec.shape:
        raise ValueError("Horizon anchor calibration arrays must have matching lengths")
    if raw.size == 0 or np.allclose(raw, smoothed):
        return 0.0, 0.0
    candidate_mixes = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0)
    default_mae = float(np.mean(np.abs(raw - target_vec)))
    best_mix = 0.0
    best_mae = default_mae
    for mix in candidate_mixes:
        candidate_anchor = _blend_horizon_anchor(raw, smoothed, mix)
        candidate_mae = float(np.mean(np.abs(candidate_anchor - target_vec)))
        if candidate_mae + 1e-9 < best_mae:
            best_mae = candidate_mae
            best_mix = float(mix)
    return float(best_mix), _read_policy_reliability(default_mae=default_mae, best_mae=best_mae)


def _hurdle_activity_gate(
    anchor: np.ndarray,
    aux_features: np.ndarray | None,
    fallback: float,
    horizon: int,
) -> np.ndarray:
    anchor_vec = np.clip(np.asarray(anchor, dtype=np.float64).reshape(-1), 0.0, None)
    lag1, roll3, roll7, roll_std, history_count = _history_components(aux_features, anchor_vec)
    history_gate = np.clip(1.0 - np.exp(-history_count / 3.0), 0.0, 1.0)
    smoothed = np.clip(0.15 * lag1 + 0.50 * roll3 + 0.35 * roll7, 0.0, None)
    change = np.abs(lag1 - roll3) + 0.5 * np.abs(roll3 - roll7) + 0.5 * roll_std
    horizon_scale = float(np.clip(np.log1p(max(int(horizon), 1)) / np.log1p(30.0), 0.0, 1.0))
    logit = (
        -1.35
        + 0.95 * np.log1p(smoothed)
        + 0.30 * np.log1p(change)
        + 1.10 * history_gate
        - 0.15 * horizon_scale
    )
    gate = 1.0 / (1.0 + np.exp(-logit))
    floor = np.clip(float(fallback), 0.0, 1.0)
    floor_gate = floor * (0.60 + 0.40 * history_gate)
    return np.clip(np.maximum(gate, floor_gate), 0.0, 1.0).astype(np.float64, copy=False)


def _compute_active_probability(
    design: np.ndarray,
    anchor: np.ndarray,
    aux_features: np.ndarray | None,
    occurrence_model: HistGradientBoostingClassifier | None,
    use_learned_occurrence: bool,
    active_rate: float,
    use_sparsity_gate: bool,
    count_sparsity_gate_strength: float,
    horizon: int,
) -> np.ndarray:
    if occurrence_model is None:
        if use_learned_occurrence:
            base_prob = np.full(len(design), float(np.clip(active_rate, 0.0, 1.0)), dtype=np.float64)
        else:
            base_prob = _anchor_activity_gate(anchor, fallback=active_rate)
    else:
        base_prob = occurrence_model.predict_proba(design)[:, 1].astype(np.float64, copy=False)
    if not use_sparsity_gate:
        return np.clip(base_prob, 0.0, 1.0).astype(np.float64, copy=False)
    prior = _hurdle_activity_gate(anchor=anchor, aux_features=aux_features, fallback=active_rate, horizon=horizon)
    strength = float(np.clip(count_sparsity_gate_strength, 0.0, 1.0))
    return np.clip((1.0 - strength) * base_prob + strength * prior, 0.0, 1.0).astype(np.float64, copy=False)


def _jump_target(anchor: np.ndarray, aux_features: np.ndarray | None, horizon: int) -> np.ndarray:
    anchor_vec = np.clip(np.asarray(anchor, dtype=np.float64).reshape(-1), 0.0, None)
    if aux_features is None:
        return anchor_vec
    horizon_scale = float(np.clip(np.log1p(max(int(horizon), 1)) / np.log1p(30.0), 0.0, 1.0))
    signal = _transition_signal(aux_features, anchor_vec)
    return np.clip(anchor_vec + (0.35 + 0.65 * horizon_scale) * signal, 0.0, None).astype(np.float64, copy=False)


def _calibrate_global_anchor_blend(
    base_pred: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    default_blend: float,
) -> tuple[float, float]:
    default = float(np.clip(default_blend, 0.0, 1.0))
    candidate_blends = sorted({0.0, default, max(default, 0.55), max(default, 0.75), 0.90})
    default_mae = float(np.mean(np.abs(((1.0 - default) * base_pred + default * anchor) - target)))
    best_blend = default
    best_mae = default_mae
    for blend in candidate_blends:
        candidate_pred = (1.0 - blend) * base_pred + blend * anchor
        candidate_mae = float(np.mean(np.abs(candidate_pred - target)))
        if candidate_mae + 1e-9 < best_mae:
            best_mae = candidate_mae
            best_blend = float(blend)
    return float(best_blend), _read_policy_reliability(default_mae=default_mae, best_mae=best_mae)


def _calibrate_global_jump_strength(
    base_pred: np.ndarray,
    jump_target: np.ndarray,
    target: np.ndarray,
    requested_strength: float,
) -> tuple[float, float]:
    requested = float(np.clip(requested_strength, 0.0, 1.0))
    if requested <= 0.0:
        return 0.0, 0.0
    candidate_strengths = sorted({0.0, 0.08, 0.15, 0.5 * requested, requested, min(0.60, requested + 0.15)})
    default_mae = float(np.mean(np.abs(base_pred - target)))
    best_strength = 0.0
    best_mae = default_mae
    for strength in candidate_strengths:
        candidate_pred = (1.0 - strength) * base_pred + strength * jump_target
        candidate_mae = float(np.mean(np.abs(candidate_pred - target)))
        if candidate_mae + 1e-9 < best_mae:
            best_mae = candidate_mae
            best_strength = float(strength)
    return float(best_strength), _transition_reliability(default_mae=default_mae, best_mae=best_mae)


def _global_jump_effective_strength(
    strength: float,
    reliability: float,
    stability: np.ndarray,
) -> np.ndarray:
    base = float(np.clip(strength, 0.0, 1.0))
    rel = float(np.clip(reliability, 0.0, 1.0))
    stab = np.asarray(stability, dtype=np.float64)
    if stab.size == 0:
        return np.asarray(base * rel, dtype=np.float64)
    volatility_gate = 0.35 + 0.65 * (1.0 - np.clip(stab, 0.0, 1.0))
    return np.clip(base * rel * volatility_gate, 0.0, 1.0).astype(np.float64, copy=False)


def _source_profile_ids(source_features: np.ndarray) -> np.ndarray:
    if source_features.size == 0 or source_features.shape[1] < 2:
        return np.zeros(source_features.shape[0], dtype=np.int8)
    edgar_active = source_features[:, 0] >= 0.5
    text_active = source_features[:, 1] >= 0.5
    profile = np.zeros(source_features.shape[0], dtype=np.int8)
    profile[np.logical_and(edgar_active, ~text_active)] = 1
    profile[np.logical_and(~edgar_active, text_active)] = 2
    profile[np.logical_and(edgar_active, text_active)] = 3
    return profile


def _source_confidence(source_features: np.ndarray) -> np.ndarray:
    if source_features.size == 0:
        return np.zeros(source_features.shape[0], dtype=np.float64)
    edgar_active = source_features[:, 0] if source_features.shape[1] >= 1 else np.zeros(source_features.shape[0], dtype=np.float32)
    text_active = source_features[:, 1] if source_features.shape[1] >= 2 else np.zeros(source_features.shape[0], dtype=np.float32)
    edgar_recency = source_features[:, 2] if source_features.shape[1] >= 3 else np.full(source_features.shape[0], 9999.0, dtype=np.float32)
    text_recency = source_features[:, 3] if source_features.shape[1] >= 4 else np.full(source_features.shape[0], 9999.0, dtype=np.float32)
    edgar_conf = np.clip(edgar_active, 0.0, 1.0) * np.exp(-np.clip(edgar_recency, 0.0, 365.0) / 30.0)
    text_conf = np.clip(text_active, 0.0, 1.0) * np.exp(-np.clip(text_recency, 0.0, 365.0) / 21.0)
    return np.clip(np.maximum(edgar_conf, text_conf), 0.0, 1.0).astype(np.float64, copy=False)


def _compose_positive_predictions(
    positive_design: np.ndarray,
    base_log: np.ndarray,
    positive_model: HistGradientBoostingRegressor | None,
    exemplar_model: KNeighborsRegressor | None,
) -> np.ndarray:
    if positive_model is None:
        return np.expm1(base_log)
    positive_residual = positive_model.predict(positive_design).astype(np.float64, copy=False)
    if exemplar_model is not None:
        exemplar_residual = exemplar_model.predict(positive_design).astype(np.float64, copy=False)
        positive_residual = 0.6 * positive_residual + 0.4 * exemplar_residual
    return np.expm1(base_log + positive_residual)


def _history_anchor_stability(aux_features: np.ndarray | None) -> np.ndarray:
    if aux_features is None:
        return np.zeros(0, dtype=np.float64)
    aux = np.asarray(aux_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux[:, None]
    if aux.shape[1] < 4:
        return np.full(aux.shape[0], 0.5, dtype=np.float64)
    lag1 = np.abs(aux[:, 0]) if aux.shape[1] >= 1 else np.zeros(aux.shape[0], dtype=np.float64)
    roll_std = np.abs(aux[:, 3])
    history_count = np.clip(aux[:, 4], 0.0, None) if aux.shape[1] >= 5 else np.ones(aux.shape[0], dtype=np.float64)
    stability = np.exp(-roll_std / (lag1 + 1.0)) * (1.0 - np.exp(-history_count / 3.0))
    return np.clip(stability, 0.0, 1.0).astype(np.float64, copy=False)


def _source_read_policy_prior_blend(
    profile_id: int,
    default_blend: float,
    confidence: float,
    stability: float,
) -> float:
    conf = float(np.clip(confidence, 0.0, 1.0))
    stab = float(np.clip(stability, 0.0, 1.0))
    base = float(np.clip(default_blend, 0.0, 1.0))
    if int(profile_id) == 1:
        return float(np.clip(base + 0.18 * conf + 0.12 * stab, base, 0.95))
    if int(profile_id) == 2:
        return float(np.clip(base - 0.25 * conf - 0.15 * (1.0 - stab), 0.15, base))
    if int(profile_id) == 3:
        return float(np.clip(base + 0.08 * conf + 0.06 * stab, base, 0.90))
    return float(np.clip(base - 0.18 * (1.0 - stab), 0.20, base))


def _regularize_read_policy_blend(profile_id: int, calibrated_blend: float, prior_blend: float) -> float:
    calibrated = float(np.clip(calibrated_blend, 0.0, 1.0))
    prior = float(np.clip(prior_blend, 0.0, 1.0))
    if int(profile_id) in (1, 3):
        return float(np.clip(max(calibrated, prior), 0.0, 1.0))
    if int(profile_id) == 2:
        return float(np.clip(0.35 * calibrated + 0.65 * prior, 0.0, 1.0))
    return float(np.clip(0.25 * calibrated + 0.75 * prior, 0.0, 1.0))


def _read_policy_reliability(default_mae: float, best_mae: float) -> float:
    default = float(max(default_mae, 1e-8))
    best = float(max(best_mae, 0.0))
    improvement = max(0.0, default - best)
    return float(np.clip(improvement / default, 0.0, 1.0))


def _read_policy_strength(default_blend: float, target_blend: float, reliability: float) -> float:
    default = float(np.clip(default_blend, 0.0, 1.0))
    target = float(np.clip(target_blend, 0.0, 1.0))
    rel = float(np.clip(reliability, 0.0, 1.0))
    if target >= default:
        return rel
    return float(np.clip((rel - 0.25) / 0.75, 0.0, 1.0))


def _transition_signal(aux_features: np.ndarray | None, anchor: np.ndarray) -> np.ndarray:
    anchor_vec = np.clip(np.asarray(anchor, dtype=np.float64).reshape(-1), 0.0, None)
    if aux_features is None:
        return np.zeros_like(anchor_vec)
    aux = np.asarray(aux_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux[:, None]
    if aux.shape[0] != anchor_vec.shape[0]:
        raise ValueError("Transition aux_features length does not match anchor length")
    lag1 = aux[:, 0] if aux.shape[1] >= 1 else anchor_vec
    roll3 = aux[:, 1] if aux.shape[1] >= 2 else lag1
    roll7 = aux[:, 2] if aux.shape[1] >= 3 else roll3
    roll_std = np.abs(aux[:, 3]) if aux.shape[1] >= 4 else np.zeros(aux.shape[0], dtype=np.float64)
    history_count = np.clip(aux[:, 4], 0.0, None) if aux.shape[1] >= 5 else np.ones(aux.shape[0], dtype=np.float64)
    delta_short = lag1 - roll3
    delta_medium = roll3 - roll7
    jump_direction = np.sign(delta_short + 0.5 * delta_medium)
    history_gate = 0.45 + 0.55 * (1.0 - np.exp(-history_count / 3.0))
    raw_signal = 0.60 * delta_short + 0.30 * delta_medium + 0.10 * jump_direction * roll_std
    return (history_gate * raw_signal).astype(np.float64, copy=False)


def _source_transition_prior_strength(profile_id: int, confidence: float, stability: float) -> float:
    conf = float(np.clip(confidence, 0.0, 1.0))
    stab = float(np.clip(stability, 0.0, 1.0))
    volatility = 1.0 - stab
    if int(profile_id) == 2:
        return float(np.clip(0.18 + 0.20 * conf + 0.22 * volatility, 0.10, 0.60))
    if int(profile_id) == 3:
        return float(np.clip(0.10 + 0.14 * conf + 0.16 * volatility, 0.05, 0.45))
    if int(profile_id) == 1:
        return float(np.clip(0.04 + 0.08 * conf + 0.08 * volatility, 0.0, 0.25))
    return float(np.clip(0.03 + 0.08 * volatility, 0.0, 0.20))


def _regularize_transition_strength(profile_id: int, calibrated_strength: float, prior_strength: float) -> float:
    calibrated = float(np.clip(calibrated_strength, 0.0, 1.0))
    prior = float(np.clip(prior_strength, 0.0, 1.0))
    if int(profile_id) == 2:
        return float(np.clip(max(calibrated, 0.75 * prior), 0.0, 0.70))
    if int(profile_id) == 3:
        return float(np.clip(max(calibrated, 0.55 * prior), 0.0, 0.55))
    if int(profile_id) == 1:
        return float(np.clip(0.70 * calibrated + 0.30 * prior, 0.0, 0.35))
    return float(np.clip(0.60 * calibrated + 0.40 * prior, 0.0, 0.30))


def _transition_reliability(default_mae: float, best_mae: float) -> float:
    return _read_policy_reliability(default_mae=default_mae, best_mae=best_mae)


def _transition_effective_strength(
    strength: float,
    reliability: float,
    confidence: np.ndarray,
    stability: np.ndarray,
) -> np.ndarray:
    base = float(np.clip(strength, 0.0, 1.0))
    rel = float(np.clip(reliability, 0.0, 1.0))
    conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    stab = np.clip(np.asarray(stability, dtype=np.float64), 0.0, 1.0)
    volatility_gate = 0.35 + 0.65 * (1.0 - stab)
    return np.clip(base * rel * conf * volatility_gate, 0.0, 1.0).astype(np.float64, copy=False)


def _fit_reference_positive_predictions(
    train_design: np.ndarray,
    train_target: np.ndarray,
    train_anchor: np.ndarray,
    test_design: np.ndarray,
    test_anchor: np.ndarray,
    horizon: int,
    random_state: int,
) -> np.ndarray:
    if len(train_target) < 8:
        return np.clip(test_anchor, 0.0, None).astype(np.float64, copy=False)
    positive_residual = np.log1p(np.clip(train_target, 0.0, None)) - np.log1p(np.clip(train_anchor, 0.0, None))
    if np.nanstd(positive_residual) < 1e-8:
        return np.clip(test_anchor, 0.0, None).astype(np.float64, copy=False)
    positive_model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        random_state=random_state,
    )
    positive_model.fit(train_design, positive_residual)
    exemplar_model: KNeighborsRegressor | None = None
    if int(horizon) == 1 and len(train_target) >= 5:
        neighbors = min(15, len(train_target))
        exemplar_model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
        exemplar_model.fit(train_design, positive_residual)
    return _compose_positive_predictions(
        positive_design=test_design,
        base_log=np.log1p(np.clip(test_anchor, 0.0, None)),
        positive_model=positive_model,
        exemplar_model=exemplar_model,
    )


def _read_policy_calibration_mask(profile_ids: np.ndarray) -> np.ndarray:
    profile_ids = np.asarray(profile_ids, dtype=np.int64)
    mask = np.zeros(len(profile_ids), dtype=bool)
    for profile_id in np.unique(profile_ids):
        idx = np.flatnonzero(profile_ids == profile_id)
        if idx.size < 12:
            continue
        stride = 4 if idx.size >= 24 else 3
        chosen = idx[::stride]
        if chosen.size == 0:
            continue
        mask[chosen] = True
    return mask


def _specialist_blend(profile_id: int, confidence: np.ndarray) -> np.ndarray:
    conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    if int(profile_id) == 3:
        return np.clip(0.55 + 0.25 * conf, 0.55, 0.80)
    return np.clip(0.35 + 0.25 * conf, 0.35, 0.60)


def _source_guard_weight(source_features: np.ndarray) -> np.ndarray:
    profile_ids = _source_profile_ids(source_features)
    confidence = _source_confidence(source_features)
    weight = np.clip(0.85 + 0.10 * confidence, 0.85, 0.95)
    mixed_mask = profile_ids == 3
    weight[mixed_mask] = np.clip(0.72 + 0.18 * confidence[mixed_mask], 0.72, 0.90)
    return weight.astype(np.float64, copy=False)