#!/usr/bin/env python3
"""Binary lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from narrative.nbi_nci.calibration import reliability_metrics


@dataclass(frozen=True)
class BinaryLaneSpec:
    lane_name: str = "binary"
    supports_calibration: bool = True
    supports_hazard_adapter: bool = True
    supports_event_consistency: bool = True
    guardrails: Tuple[str, ...] = ("probability_collapse", "threshold_only_alignment", "cross_lane_collateral_damage")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_calibration": self.supports_calibration,
            "supports_hazard_adapter": self.supports_hazard_adapter,
            "supports_event_consistency": self.supports_event_consistency,
            "guardrails": self.guardrails,
        }


class BinaryLaneRuntime:
    def __init__(self, spec: BinaryLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or BinaryLaneSpec()
        self.random_state = int(random_state)
        self._model: LogisticRegression | None = None
        self._hazard_model: LogisticRegression | None = None
        self._calibrator_name = "identity"
        self._calibrator: Any | None = None
        self._constant_probability = 0.5
        self._train_positive_rate = 0.0
        self._event_rate = 0.0
        self._transition_rate = 0.0
        self._persistence_rate = 0.0
        self._hazard_rows = 0
        self._hazard_blend = 0.0
        self._temperature = 1.0
        self._uses_hazard_adapter = False
        self._identity_metrics: Dict[str, float] = {"brier": 0.25, "logloss": 0.6931471805599453, "ece": 0.0}
        self._selected_metrics: Dict[str, float] = dict(self._identity_metrics)
        self._calibration_shrinkage_target = "none"
        self._calibration_shrinkage_strength = 0.0
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        enable_calibration_shrinkage: bool = False,
        calibration_shrinkage_target: str = "auto",
    ) -> "BinaryLaneRuntime":
        target = (np.asarray(y, dtype=np.float32) > 0.5).astype(np.int32, copy=False)
        self._model = None
        self._hazard_model = None
        self._calibrator_name = "identity"
        self._calibrator = None
        self._temperature = 1.0
        self._hazard_blend = 0.0
        self._uses_hazard_adapter = False
        self._calibration_shrinkage_target = "none"
        self._calibration_shrinkage_strength = 0.0
        self._train_positive_rate = float(target.mean()) if target.size else 0.0
        self._event_rate = self._train_positive_rate
        lag1_active = _lag1_active(aux_features, len(target))
        at_risk_mask = lag1_active < 0.5
        persisted_mask = ~at_risk_mask
        self._hazard_rows = int(at_risk_mask.sum())
        self._transition_rate = float(target[at_risk_mask].mean()) if at_risk_mask.any() else self._train_positive_rate
        self._persistence_rate = float(target[persisted_mask].mean()) if persisted_mask.any() else self._train_positive_rate
        if target.size == 0:
            self._constant_probability = 0.5
            self._identity_metrics = {"brier": 0.25, "logloss": 0.6931471805599453, "ece": 0.0}
            self._selected_metrics = dict(self._identity_metrics)
            self._fitted = True
            return self

        self._constant_probability = float(target.mean())
        if np.unique(target).size < 2:
            constant_prob = np.full(target.size, self._constant_probability, dtype=np.float64)
            self._identity_metrics = _binary_metrics(constant_prob, target)
            self._selected_metrics = dict(self._identity_metrics)
            self._fitted = True
            return self

        design = _merge_features(lane_state, aux_features, lag1_active)
        calibration = _split_binary_calibration(design, target, lag1_active)
        if calibration is not None:
            train_design = np.asarray(calibration["train_design"], dtype=np.float32)
            train_target = np.asarray(calibration["train_target"], dtype=np.int32)
            train_lag1 = np.asarray(calibration["train_lag1"], dtype=np.float64)
            calibration_design = np.asarray(calibration["calibration_design"], dtype=np.float32)
            calibration_target = np.asarray(calibration["calibration_target"], dtype=np.int32)
            calibration_lag1 = np.asarray(calibration["calibration_lag1"], dtype=np.float64)

            base_model = _build_binary_model(self.random_state)
            base_model.fit(train_design, train_target)
            hazard_model = _fit_hazard_model(
                design=train_design,
                target=train_target,
                lag1_active=train_lag1,
                random_state=self.random_state,
            )
            base_prob = _predict_binary_probability(
                design=calibration_design,
                model=base_model,
                fallback_rate=float(train_target.mean()),
            )
            hazard_prior = _predict_hazard_prior(
                design=calibration_design,
                lag1_active=calibration_lag1,
                hazard_model=hazard_model,
                transition_rate=float(train_target[train_lag1 < 0.5].mean()) if np.any(train_lag1 < 0.5) else float(train_target.mean()),
                persistence_rate=float(train_target[train_lag1 >= 0.5].mean()) if np.any(train_lag1 >= 0.5) else float(train_target.mean()),
            )
            self._hazard_blend = _calibrate_hazard_blend(base_prob, hazard_prior, calibration_target)
            blended_prob = _blend_binary_probability(base_prob, hazard_prior, self._hazard_blend)
            (
                self._calibrator_name,
                self._identity_metrics,
                self._selected_metrics,
            ) = _select_binary_calibrator(blended_prob, calibration_target, random_state=self.random_state)
            if enable_calibration_shrinkage:
                calibration_calibrator = _fit_named_binary_calibrator(
                    blended_prob,
                    calibration_target,
                    self._calibrator_name,
                    random_state=self.random_state,
                )
                calibrated_prob = _apply_binary_calibrator(
                    self._calibrator_name,
                    calibration_calibrator,
                    blended_prob,
                )
                (
                    self._calibration_shrinkage_target,
                    self._calibration_shrinkage_strength,
                    self._selected_metrics,
                ) = _select_probability_shrinkage(
                    calibrated_prob,
                    calibration_target,
                    hazard_prior=hazard_prior,
                    base_rate=float(train_target.mean()),
                    requested_target=calibration_shrinkage_target,
                )

        self._model = _build_binary_model(self.random_state)
        self._model.fit(design, target)
        self._hazard_model = _fit_hazard_model(
            design=design,
            target=target,
            lag1_active=lag1_active,
            random_state=self.random_state,
        )
        self._uses_hazard_adapter = bool(self._hazard_model is not None and self._hazard_blend > 0.0)

        raw_prob = _predict_binary_probability(
            design=design,
            model=self._model,
            fallback_rate=self._train_positive_rate,
        )
        hazard_prior = _predict_hazard_prior(
            design=design,
            lag1_active=lag1_active,
            hazard_model=self._hazard_model,
            transition_rate=self._transition_rate,
            persistence_rate=self._persistence_rate,
        )
        blended_prob = _blend_binary_probability(raw_prob, hazard_prior, self._hazard_blend)
        self._calibrator = _fit_named_binary_calibrator(
            probs=blended_prob,
            labels=target,
            name=self._calibrator_name,
            random_state=self.random_state,
        )
        if self._selected_metrics == self._identity_metrics:
            self._selected_metrics = _binary_metrics(
                _apply_binary_calibrator(self._calibrator_name, self._calibrator, blended_prob),
                target,
            )
        self._fitted = True
        return self

    def predict(self, lane_state: np.ndarray, aux_features: np.ndarray | None = None) -> np.ndarray:
        if not self._fitted:
            raise ValueError("BinaryLaneRuntime is not fitted")
        if self._model is None:
            return np.full(len(lane_state), self._constant_probability, dtype=np.float64)
        lag1_active = _lag1_active(aux_features, len(lane_state))
        design = _merge_features(lane_state, aux_features, lag1_active)
        base_prob = _predict_binary_probability(
            design=design,
            model=self._model,
            fallback_rate=self._train_positive_rate,
        )
        hazard_prior = _predict_hazard_prior(
            design=design,
            lag1_active=lag1_active,
            hazard_model=self._hazard_model,
            transition_rate=self._transition_rate,
            persistence_rate=self._persistence_rate,
        )
        blended_prob = _blend_binary_probability(base_prob, hazard_prior, self._hazard_blend)
        calibrated = _apply_binary_calibrator(self._calibrator_name, self._calibrator, blended_prob)
        calibrated = _apply_probability_shrinkage(
            calibrated,
            _resolve_probability_shrinkage_target(
                self._calibration_shrinkage_target,
                hazard_prior,
                self._train_positive_rate,
            ),
            self._calibration_shrinkage_strength,
        )
        return np.clip(calibrated, 0.0, 1.0).astype(np.float64, copy=False)


def _build_binary_model(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        random_state=random_state,
    )


def _split_binary_calibration(
    design: np.ndarray,
    target: np.ndarray,
    lag1_active: np.ndarray,
) -> dict[str, np.ndarray] | None:
    n_rows = int(design.shape[0])
    if n_rows < 24:
        return None
    calibration_rows = min(max(n_rows // 4, 6), 64)
    train_rows = n_rows - calibration_rows
    if train_rows < 12:
        return None
    calibration_target = np.asarray(target[train_rows:], dtype=np.int32)
    if np.unique(calibration_target).size < 2:
        return None
    return {
        "train_design": design[:train_rows],
        "train_target": target[:train_rows],
        "train_lag1": lag1_active[:train_rows],
        "calibration_design": design[train_rows:],
        "calibration_target": calibration_target,
        "calibration_lag1": lag1_active[train_rows:],
    }


def _fit_hazard_model(
    design: np.ndarray,
    target: np.ndarray,
    lag1_active: np.ndarray,
    random_state: int,
) -> LogisticRegression | None:
    at_risk_mask = np.asarray(lag1_active, dtype=np.float64) < 0.5
    if int(at_risk_mask.sum()) < 12:
        return None
    at_risk_target = np.asarray(target[at_risk_mask], dtype=np.int32)
    if np.unique(at_risk_target).size < 2:
        return None
    model = _build_binary_model(random_state + 19)
    model.fit(np.asarray(design[at_risk_mask], dtype=np.float32), at_risk_target)
    return model


def _predict_binary_probability(
    design: np.ndarray,
    model: LogisticRegression | None,
    fallback_rate: float,
) -> np.ndarray:
    if model is None:
        return np.full(int(design.shape[0]), float(np.clip(fallback_rate, 0.0, 1.0)), dtype=np.float64)
    return model.predict_proba(design)[:, 1].astype(np.float64, copy=False)


def _predict_hazard_prior(
    design: np.ndarray,
    lag1_active: np.ndarray,
    hazard_model: LogisticRegression | None,
    transition_rate: float,
    persistence_rate: float,
) -> np.ndarray:
    lag1 = np.asarray(lag1_active, dtype=np.float64).reshape(-1)
    at_risk_mask = lag1 < 0.5
    prior = np.full(lag1.shape[0], float(np.clip(persistence_rate, 0.0, 1.0)), dtype=np.float64)
    if at_risk_mask.any():
        if hazard_model is None:
            prior[at_risk_mask] = float(np.clip(transition_rate, 0.0, 1.0))
        else:
            prior[at_risk_mask] = hazard_model.predict_proba(design[at_risk_mask])[:, 1].astype(np.float64, copy=False)
    return np.clip(prior, 0.0, 1.0)


def _calibrate_hazard_blend(base_prob: np.ndarray, hazard_prior: np.ndarray, labels: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.int32)
    best_blend = 0.0
    best_score = float("inf")
    for blend in (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0):
        candidate = _blend_binary_probability(base_prob, hazard_prior, float(blend))
        metrics = _binary_metrics(candidate, target)
        score = 0.65 * float(metrics["brier"]) + 0.35 * float(metrics["logloss"])
        if score + 1e-12 < best_score:
            best_score = score
            best_blend = float(blend)
        elif abs(score - best_score) <= 1e-12 and float(blend) < best_blend:
            best_blend = float(blend)
    return float(best_blend)


def _blend_binary_probability(base_prob: np.ndarray, hazard_prior: np.ndarray, blend: float) -> np.ndarray:
    base_arr = np.asarray(base_prob, dtype=np.float64)
    prior_arr = np.asarray(hazard_prior, dtype=np.float64)
    mix = float(np.clip(blend, 0.0, 1.0))
    return np.clip((1.0 - mix) * base_arr + mix * prior_arr, 0.0, 1.0)


def _select_binary_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> tuple[str, Dict[str, float], Dict[str, float],]:
    target = np.asarray(labels, dtype=np.int32)
    identity_metrics = _binary_metrics(probs, target)
    candidates: Dict[str, Dict[str, float]] = {"identity": dict(identity_metrics)}
    if np.unique(np.asarray(probs, dtype=np.float64)).size >= 2:
        platt = _fit_named_binary_calibrator(probs, target, "platt", random_state=random_state)
        if platt is not None:
            candidates["platt"] = _binary_metrics(_apply_binary_calibrator("platt", platt, probs), target)
        isotonic = _fit_named_binary_calibrator(probs, target, "isotonic", random_state=random_state)
        if isotonic is not None:
            candidates["isotonic"] = _binary_metrics(_apply_binary_calibrator("isotonic", isotonic, probs), target)
    selected = _select_calibration_name(candidates)
    return selected, identity_metrics, dict(candidates[selected])


def _select_calibration_name(candidates: Dict[str, Dict[str, float]], ece_threshold: float = 0.05) -> str:
    scored = []
    for name, metrics in candidates.items():
        score = 0.65 * float(metrics.get("brier", float("inf"))) + 0.35 * float(metrics.get("logloss", float("inf")))
        scored.append((str(name), float(score), float(metrics.get("ece", float("inf")))))
    scored.sort(key=lambda item: item[1])
    identity_ece = float(candidates.get("identity", {}).get("ece", 1.0))
    for name, _, ece in scored:
        if name == "identity":
            return name
        if ece <= identity_ece + float(ece_threshold):
            return name
    return "identity"


def _select_probability_shrinkage(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    hazard_prior: np.ndarray,
    base_rate: float,
    requested_target: str = "auto",
) -> tuple[str, float, Dict[str, float]]:
    baseline_metrics = _binary_metrics(probs, labels)
    best_target = "none"
    best_strength = 0.0
    best_metrics = dict(baseline_metrics)
    best_score = _calibration_guard_score(best_metrics)

    requested = str(requested_target).strip().lower() or "auto"
    target_candidates: Dict[str, np.ndarray] = {}
    if requested in {"auto", "hazard_prior", "hazard"}:
        target_candidates["hazard_prior"] = np.asarray(hazard_prior, dtype=np.float64)
    if requested in {"auto", "base_rate", "base"}:
        target_candidates["base_rate"] = np.full(len(probs), float(np.clip(base_rate, 0.0, 1.0)), dtype=np.float64)

    for target_name, target_probs in target_candidates.items():
        for strength in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
            candidate = _apply_probability_shrinkage(probs, target_probs, float(strength))
            metrics = _binary_metrics(candidate, labels)
            score = _calibration_guard_score(metrics)
            if score + 1e-12 < best_score:
                best_target = str(target_name)
                best_strength = float(strength)
                best_metrics = dict(metrics)
                best_score = float(score)
            elif abs(score - best_score) <= 1e-12 and float(strength) < best_strength:
                best_target = str(target_name)
                best_strength = float(strength)
                best_metrics = dict(metrics)
    return best_target, best_strength, best_metrics


def _calibration_guard_score(metrics: Dict[str, float]) -> float:
    return (
        0.45 * float(metrics.get("brier", float("inf")))
        + 0.20 * float(metrics.get("logloss", float("inf")))
        + 0.35 * float(metrics.get("ece", float("inf")))
    )


def _resolve_probability_shrinkage_target(
    target_name: str,
    hazard_prior: np.ndarray,
    base_rate: float,
) -> np.ndarray:
    target = str(target_name).strip().lower()
    if target == "hazard_prior":
        return np.asarray(hazard_prior, dtype=np.float64)
    if target == "base_rate":
        return np.full(len(hazard_prior), float(np.clip(base_rate, 0.0, 1.0)), dtype=np.float64)
    return np.asarray(hazard_prior, dtype=np.float64)


def _apply_probability_shrinkage(probs: np.ndarray, target_probs: np.ndarray, strength: float) -> np.ndarray:
    mix = float(np.clip(strength, 0.0, 1.0))
    if mix <= 0.0:
        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    base = np.asarray(probs, dtype=np.float64)
    target = np.asarray(target_probs, dtype=np.float64)
    if base.shape != target.shape:
        raise ValueError("Probability shrinkage target must match probability shape")
    return np.clip((1.0 - mix) * base + mix * target, 0.0, 1.0)


def _fit_named_binary_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
    name: str,
    random_state: int,
) -> Any | None:
    target = np.asarray(labels, dtype=np.int32)
    candidate_probs = np.asarray(probs, dtype=np.float64)
    if np.unique(target).size < 2 or np.unique(candidate_probs).size < 2:
        return None
    if name == "platt":
        model = LogisticRegression(max_iter=400, random_state=random_state + 41)
        model.fit(candidate_probs.reshape(-1, 1), target)
        return model
    if name == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception:
            return None
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(candidate_probs, target)
        return model
    return None


def _apply_binary_calibrator(name: str, calibrator: Any | None, probs: np.ndarray) -> np.ndarray:
    candidate_probs = np.asarray(probs, dtype=np.float64)
    if calibrator is None or name == "identity":
        return np.clip(candidate_probs, 0.0, 1.0)
    if name == "platt" and hasattr(calibrator, "predict_proba"):
        return np.clip(calibrator.predict_proba(candidate_probs.reshape(-1, 1))[:, 1], 0.0, 1.0).astype(
            np.float64,
            copy=False,
        )
    if name == "isotonic":
        return np.clip(np.asarray(calibrator.predict(candidate_probs), dtype=np.float64), 0.0, 1.0)
    return np.clip(candidate_probs, 0.0, 1.0)


def _binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    candidate_probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.int32)
    reliability = reliability_metrics(candidate_probs, target, n_bins=10)
    return {
        "brier": float(reliability["brier"]),
        "logloss": _binary_logloss(candidate_probs, target),
        "ece": float(reliability["ece"]),
    }


def _binary_logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    candidate_probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.float64)
    return float(-(target * np.log(candidate_probs) + (1.0 - target) * np.log(1.0 - candidate_probs)).mean())


def _lag1_active(aux_features: np.ndarray | None, n_rows: int) -> np.ndarray:
    if aux_features is None:
        return np.zeros(n_rows, dtype=np.float64)
    aux = np.asarray(aux_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux[:, None]
    if aux.shape[0] != n_rows or aux.shape[1] == 0:
        return np.zeros(n_rows, dtype=np.float64)
    lag1 = np.asarray(aux[:, 0], dtype=np.float64)
    return (np.nan_to_num(lag1, nan=0.0) > 0.5).astype(np.float64, copy=False)


def _merge_features(lane_state: np.ndarray, aux_features: np.ndarray | None, lag1_active: np.ndarray) -> np.ndarray:
    lane = np.asarray(lane_state, dtype=np.float32)
    parts = [lane]
    if aux_features is None:
        aux = np.zeros((lane.shape[0], 0), dtype=np.float32)
    else:
        aux = np.asarray(aux_features, dtype=np.float32)
        if aux.ndim == 1:
            aux = aux[:, None]
    if aux.shape[1] > 0:
        parts.append(aux)
    lag1 = np.asarray(lag1_active, dtype=np.float32).reshape(-1, 1)
    parts.append(lag1)
    parts.append(1.0 - lag1)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)