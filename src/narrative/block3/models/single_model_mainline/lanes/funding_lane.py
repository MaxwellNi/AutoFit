#!/usr/bin/env python3
"""Funding lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class FundingLaneSpec:
    lane_name: str = "funding"
    supports_anchor_residual: bool = True
    supports_tail_aware_objective: bool = True
    supports_horizon_bucket_subpaths: bool = True
    supports_source_scaling_guard: bool = True
    guardrails: Tuple[str, ...] = ("source_rich_blowup", "log_domain_regression", "easy_cell_only_improvement")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_anchor_residual": self.supports_anchor_residual,
            "supports_tail_aware_objective": self.supports_tail_aware_objective,
            "supports_horizon_bucket_subpaths": self.supports_horizon_bucket_subpaths,
            "supports_source_scaling_guard": self.supports_source_scaling_guard,
            "guardrails": self.guardrails,
        }


class FundingLaneRuntime:
    def __init__(self, spec: FundingLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or FundingLaneSpec()
        self.random_state = int(random_state)
        self._model: HistGradientBoostingRegressor | None = None
        self._fallback_value = 0.0
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
        if target.size == 0:
            self._fitted = True
            return self

        anchor_vec = _resolve_anchor(anchor, fallback=self._fallback_value, length=target.size)
        design = _merge_features(lane_state, aux_features, anchor_vec)
        residual = target - anchor_vec

        if target.size < 8 or np.nanstd(residual) < 1e-8:
            self._fitted = True
            return self

        self._model = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        self._model.fit(design, residual)
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
        if self._model is None:
            return np.clip(anchor_vec, 0.0, None).astype(np.float64, copy=False)

        design = _merge_features(lane_state, aux_features, anchor_vec)
        residual = self._model.predict(design)
        return np.clip(anchor_vec + residual, 0.0, None).astype(np.float64, copy=False)


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