#!/usr/bin/env python3
"""Investors lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


@dataclass(frozen=True)
class InvestorsLaneSpec:
    lane_name: str = "investors"
    supports_h1_occurrence_head: bool = True
    supports_short_horizon_exemplar_branch: bool = True
    supports_hurdle_decomposition: bool = True
    supports_transition_dynamics: bool = True
    supports_source_pollution_guard: bool = True
    guardrails: Tuple[str, ...] = ("shared_count_repair", "h1_only_sharpness", "source_rich_miscalibration")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_h1_occurrence_head": self.supports_h1_occurrence_head,
            "supports_short_horizon_exemplar_branch": self.supports_short_horizon_exemplar_branch,
            "supports_hurdle_decomposition": self.supports_hurdle_decomposition,
            "supports_transition_dynamics": self.supports_transition_dynamics,
            "supports_source_pollution_guard": self.supports_source_pollution_guard,
            "guardrails": self.guardrails,
        }


class InvestorsLaneRuntime:
    def __init__(self, spec: InvestorsLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or InvestorsLaneSpec()
        self.random_state = int(random_state)
        self._occurrence_model: HistGradientBoostingClassifier | None = None
        self._positive_model: HistGradientBoostingRegressor | None = None
        self._exemplar_model: KNeighborsRegressor | None = None
        self._fallback_value = 0.0
        self._active_rate = 0.0
        self._anchor_blend = 0.0
        self._horizon = 1
        self._use_learned_occurrence = True
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        horizon: int = 1,
        anchor_blend: float = 0.0,
        task_name: str = "",
    ) -> "InvestorsLaneRuntime":
        target = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
        self._fallback_value = float(np.nanmedian(target)) if target.size else 0.0
        self._horizon = int(horizon)
        self._anchor_blend = float(np.clip(anchor_blend, 0.0, 1.0))
        self._use_learned_occurrence = str(task_name) == "task2_forecast"
        if target.size == 0:
            self._fitted = True
            return self

        design = _merge_features(lane_state, aux_features, anchor)
        active = (target > 0.0).astype(np.int32)
        self._active_rate = float(active.mean())

        if self._use_learned_occurrence and np.unique(active).size >= 2 and len(active) >= 12:
            self._occurrence_model = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=150,
                learning_rate=0.05,
                random_state=self.random_state,
            )
            self._occurrence_model.fit(design, active)

        positive_mask = target > 0.0
        if positive_mask.any():
            positive_design = design[positive_mask]
            positive_target = target[positive_mask]
            anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=target.size)
            positive_anchor = anchor_vec[positive_mask]
            positive_residual = np.log1p(positive_target) - np.log1p(np.clip(positive_anchor, 0.0, None))
            if positive_target.size >= 8 and np.nanstd(positive_residual) >= 1e-8:
                self._positive_model = HistGradientBoostingRegressor(
                    max_depth=3,
                    max_iter=200,
                    learning_rate=0.05,
                    random_state=self.random_state,
                )
                self._positive_model.fit(positive_design, positive_residual)
            if self._horizon == 1 and positive_target.size >= 5:
                neighbors = min(15, positive_target.size)
                self._exemplar_model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
                self._exemplar_model.fit(positive_design, positive_residual)

        self._fitted = True
        return self

    def predict(
        self,
        lane_state: np.ndarray,
        aux_features: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise ValueError("InvestorsLaneRuntime is not fitted")

        design = _merge_features(lane_state, aux_features, anchor)
        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=len(design))
        if self._occurrence_model is None:
            if self._use_learned_occurrence:
                active_prob = np.full(len(design), self._active_rate, dtype=np.float64)
            else:
                active_prob = _anchor_activity_gate(anchor_vec, fallback=self._active_rate)
        else:
            active_prob = self._occurrence_model.predict_proba(design)[:, 1].astype(np.float64, copy=False)

        base_log = np.log1p(np.clip(anchor_vec, 0.0, None))
        if self._positive_model is None:
            positive_pred = np.clip(anchor_vec, 0.0, None).astype(np.float64, copy=False)
        else:
            positive_residual = self._positive_model.predict(design).astype(np.float64, copy=False)
            if self._exemplar_model is not None and self._horizon == 1:
                exemplar_residual = self._exemplar_model.predict(design).astype(np.float64, copy=False)
                positive_residual = 0.6 * positive_residual + 0.4 * exemplar_residual
            positive_pred = np.expm1(base_log + positive_residual)

        pred = np.clip(active_prob * np.clip(positive_pred, 0.0, None), 0.0, None)
        if anchor is not None and self._anchor_blend > 0.0:
            pred = (1.0 - self._anchor_blend) * pred + self._anchor_blend * anchor_vec
        return pred.astype(np.float64, copy=False)


def _merge_features(lane_state: np.ndarray, aux_features: np.ndarray | None, anchor: np.ndarray | None) -> np.ndarray:
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
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


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