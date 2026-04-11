#!/usr/bin/env python3
"""Hard target barrier contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class HardTargetBarrierSpec:
    lane_names: Tuple[str, ...] = ("binary", "funding", "investors")
    shared_to_lane_dim: int = 64
    requires_isolated_parameters: bool = True
    prohibits_shared_output_process: bool = True
    requires_lane_private_transition_block: bool = True
    transition_block_stage: str = "contract_stub"

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_names": self.lane_names,
            "shared_to_lane_dim": self.shared_to_lane_dim,
            "requires_isolated_parameters": self.requires_isolated_parameters,
            "prohibits_shared_output_process": self.prohibits_shared_output_process,
            "requires_lane_private_transition_block": self.requires_lane_private_transition_block,
            "transition_block_stage": self.transition_block_stage,
        }


class TargetIsolatedBarrier:
    """Lane-specific projection barrier for the mainline runtime owner."""

    def __init__(self, spec: HardTargetBarrierSpec | None = None):
        self.spec = spec or HardTargetBarrierSpec()
        self._shared_dim = 0
        self._condition_dim = 0
        self._source_dim = 0
        self._shared_weights: Dict[str, np.ndarray] = {}
        self._condition_weights: Dict[str, np.ndarray] = {}
        self._source_weights: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, shared_dim: int, condition_dim: int = 0, source_dim: int = 0) -> "TargetIsolatedBarrier":
        self._shared_dim = int(shared_dim)
        self._condition_dim = int(condition_dim)
        self._source_dim = int(source_dim)
        self._shared_weights = {}
        self._condition_weights = {}
        self._source_weights = {}

        for lane_idx, lane_name in enumerate(self.spec.lane_names):
            self._shared_weights[lane_name] = self._projection_matrix(
                lane_name=lane_name,
                salt=lane_idx + 1,
                input_dim=self._shared_dim,
                output_dim=self.spec.shared_to_lane_dim,
            )
            if self._condition_dim:
                self._condition_weights[lane_name] = self._projection_matrix(
                    lane_name=lane_name,
                    salt=101 + lane_idx,
                    input_dim=self._condition_dim,
                    output_dim=max(4, min(12, self._condition_dim)),
                )
            if self._source_dim:
                self._source_weights[lane_name] = self._projection_matrix(
                    lane_name=lane_name,
                    salt=211 + lane_idx,
                    input_dim=self._source_dim,
                    output_dim=max(4, min(16, self._source_dim)),
                )
        self._fitted = True
        return self

    def split(
        self,
        shared_state: np.ndarray,
        condition_state: np.ndarray | None = None,
        source_state: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        shared = np.asarray(shared_state, dtype=np.float32)
        condition = None if condition_state is None else np.asarray(condition_state, dtype=np.float32)
        source = None if source_state is None else np.asarray(source_state, dtype=np.float32)

        if not self._fitted:
            self.fit(
                shared_dim=shared.shape[1],
                condition_dim=0 if condition is None else condition.shape[1],
                source_dim=0 if source is None else source.shape[1],
            )

        lane_states: Dict[str, np.ndarray] = {}
        for lane_idx, lane_name in enumerate(self.spec.lane_names):
            parts = [shared @ self._shared_weights[lane_name]]
            if condition is not None and lane_name in self._condition_weights:
                parts.append(condition @ self._condition_weights[lane_name])
            if source is not None and lane_name in self._source_weights:
                parts.append(source @ self._source_weights[lane_name])
            lane_id = np.full((shared.shape[0], 1), float(lane_idx), dtype=np.float32)
            lane_states[lane_name] = np.concatenate([*parts, lane_id], axis=1).astype(np.float32, copy=False)
        return lane_states

    def _projection_matrix(self, lane_name: str, salt: int, input_dim: int, output_dim: int) -> np.ndarray:
        seed = self._lane_seed(lane_name, salt)
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((input_dim, output_dim)).astype(np.float32)
            / max(1.0, np.sqrt(float(input_dim)))
        )

    def _lane_seed(self, lane_name: str, salt: int) -> int:
        base = sum((idx + 1) * byte for idx, byte in enumerate(lane_name.encode("utf-8")))
        return int((base + 7919 * salt) % (2**32 - 1))