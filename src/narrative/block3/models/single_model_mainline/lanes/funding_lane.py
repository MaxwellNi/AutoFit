#!/usr/bin/env python3
"""Funding lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


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
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
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
        if target.size == 0:
            self._fitted = True
            return self

        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=target.size)
        design = _merge_features(lane_state, aux_features, anchor_vec)
        jump_target = _positive_jump_target(target=target, anchor_vec=anchor_vec)
        self._anchor_dominance = _anchor_dominance(anchor_vec=anchor_vec, residual=jump_target)
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

        if target.size < 12 or self._positive_jump_rows < 8 or np.nanstd(jump_target) < 1e-8:
            self._fitted = True
            return self

        calibration = _split_funding_calibration(
            design=design,
            target=target,
            anchor=anchor_vec,
            jump_target=jump_target,
            jump_floor=self._jump_floor,
        )
        if calibration is not None:
            calibration_models = _fit_jump_process_models(
                design=calibration["train_design"],
                jump_target=calibration["train_jump"],
                jump_floor=float(calibration["jump_floor"]),
                random_state=self.random_state,
            )
            calibration_jump_pred = _predict_jump_process(
                design=calibration["calibration_design"],
                event_model=calibration_models["event_model"],
                severity_model=calibration_models["severity_model"],
                event_rate=float(calibration_models["event_rate"]),
                positive_jump_median=float(calibration_models["positive_jump_median"]),
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
            )
            self._calibration_rows = int(calibration["calibration_target"].size)

        full_models = _fit_jump_process_models(
            design=design,
            jump_target=jump_target,
            jump_floor=self._jump_floor,
            random_state=self.random_state,
        )
        self._event_model = full_models["event_model"]
        self._model = full_models["severity_model"]
        self._jump_event_rate = float(full_models["event_rate"])
        self._positive_jump_rows = int(full_models["positive_jump_rows"])
        self._positive_jump_median = float(full_models["positive_jump_median"])
        self._uses_jump_hurdle_head = bool(
            self._event_model is not None or self._model is not None or self._positive_jump_median > 0.0
        )
        self._fitted = True
        return self

    def predict(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FundingLaneRuntime is not fitted")

        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=len(lane_state))
        if not self._uses_jump_hurdle_head:
            return np.clip(anchor_vec, 0.0, None).astype(np.float64, copy=False)

        design = _merge_features(lane_state, aux_features, anchor_vec)
        jump_pred = _predict_jump_process(
            design=design,
            event_model=self._event_model,
            severity_model=self._model,
            event_rate=self._jump_event_rate,
            positive_jump_median=self._positive_jump_median,
        )
        return _guarded_funding_prediction(
            anchor_vec=anchor_vec,
            residual_pred=jump_pred,
            residual_blend=self._residual_blend,
            residual_cap=self._residual_cap,
        )


def _build_residual_model(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        random_state=random_state,
    )


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
) -> dict[str, np.ndarray | float] | None:
    n_rows = int(design.shape[0])
    if n_rows < 12:
        return None
    calibration_rows = min(max(n_rows // 5, 4), 64)
    train_rows = n_rows - calibration_rows
    if train_rows < 8:
        return None
    return {
        "train_design": design[:train_rows],
        "train_jump": jump_target[:train_rows],
        "calibration_design": design[train_rows:],
        "calibration_jump": jump_target[train_rows:],
        "calibration_target": target[train_rows:],
        "calibration_anchor": anchor[train_rows:].astype(np.float64, copy=False),
        "jump_floor": float(jump_floor),
    }


def _positive_jump_target(target: np.ndarray, anchor_vec: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(target, dtype=np.float64) - np.asarray(anchor_vec, dtype=np.float64), 0.0, None)


def _jump_event_floor(jump_target: np.ndarray) -> float:
    _ = jump_target
    return 1e-6


def _fit_jump_process_models(
    design: np.ndarray,
    jump_target: np.ndarray,
    jump_floor: float,
    random_state: int,
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
        log_jump = np.log1p(positive_jump)
        if np.nanstd(log_jump) >= 1e-8:
            severity_model = _build_residual_model(random_state=random_state)
            severity_model.fit(design[event_target > 0], log_jump)
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
) -> np.ndarray:
    n_rows = int(design.shape[0])
    if event_model is None:
        event_prob = np.full(n_rows, float(np.clip(event_rate, 0.0, 1.0)), dtype=np.float64)
    else:
        event_prob = event_model.predict_proba(design)[:, 1].astype(np.float64, copy=False)
    if severity_model is None:
        jump_size = np.full(n_rows, float(max(positive_jump_median, 0.0)), dtype=np.float64)
    else:
        jump_size = np.expm1(severity_model.predict(design)).astype(np.float64, copy=False)
    return np.clip(event_prob, 0.0, 1.0) * np.clip(jump_size, 0.0, None)


def _guarded_funding_prediction(
    anchor_vec: np.ndarray,
    residual_pred: np.ndarray,
    residual_blend: float,
    residual_cap: float,
) -> np.ndarray:
    anchor_arr = np.asarray(anchor_vec, dtype=np.float64)
    guarded_residual = np.asarray(residual_pred, dtype=np.float64)
    if np.isfinite(residual_cap):
        guarded_residual = np.clip(guarded_residual, -residual_cap, residual_cap)
    pred = anchor_arr + float(residual_blend) * guarded_residual
    return np.clip(pred, 0.0, None).astype(np.float64, copy=False)


def _anchor_dominance(anchor_vec: np.ndarray, residual: np.ndarray) -> float:
    finite_anchor = np.abs(np.asarray(anchor_vec, dtype=np.float64))
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
    anchor_mae = float(np.mean(np.abs(target_arr - anchor_arr)))
    residual_scale = float(np.quantile(np.abs(target_residual), 0.75)) if target_residual.size else 0.0
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
    for blend in blend_grid:
        for cap_multiplier in cap_grid:
            cap = np.inf if not np.isfinite(cap_multiplier) else float(cap_multiplier * residual_scale)
            candidate = _guarded_funding_prediction(
                anchor_vec=anchor_arr,
                residual_pred=pred_arr,
                residual_blend=float(blend),
                residual_cap=cap,
            )
            candidate_mae = float(np.mean(np.abs(candidate - target_arr)))
            if candidate_mae + 1e-9 < best_mae:
                best_mae = candidate_mae
                best_blend = float(blend)
                best_cap = float(cap)
            elif abs(candidate_mae - best_mae) <= 1e-9:
                if float(blend) < best_blend or (
                    abs(float(blend) - best_blend) <= 1e-9 and float(cap) < best_cap
                ):
                    best_blend = float(blend)
                    best_cap = float(cap)
    return best_blend, best_cap, anchor_mae, best_mae


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