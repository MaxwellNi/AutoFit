#!/usr/bin/env python3
"""Binary lane contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


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
        self._constant_probability = 0.5
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
    ) -> "BinaryLaneRuntime":
        target = (np.asarray(y, dtype=np.float32) > 0.5).astype(np.int32, copy=False)
        if target.size == 0:
            self._constant_probability = 0.5
            self._fitted = True
            return self

        self._constant_probability = float(target.mean())
        if np.unique(target).size < 2:
            self._fitted = True
            return self

        design = _merge_features(lane_state, aux_features)
        self._model = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._model.fit(design, target)
        self._fitted = True
        return self

    def predict(self, lane_state: np.ndarray, aux_features: np.ndarray | None = None) -> np.ndarray:
        if not self._fitted:
            raise ValueError("BinaryLaneRuntime is not fitted")
        if self._model is None:
            return np.full(len(lane_state), self._constant_probability, dtype=np.float64)
        design = _merge_features(lane_state, aux_features)
        return self._model.predict_proba(design)[:, 1].astype(np.float64, copy=False)


def _merge_features(lane_state: np.ndarray, aux_features: np.ndarray | None) -> np.ndarray:
    lane = np.asarray(lane_state, dtype=np.float32)
    if aux_features is None:
        return lane
    aux = np.asarray(aux_features, dtype=np.float32)
    if aux.ndim == 1:
        aux = aux[:, None]
    return np.concatenate([lane, aux], axis=1).astype(np.float32, copy=False)