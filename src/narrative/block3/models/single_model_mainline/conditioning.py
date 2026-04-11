#!/usr/bin/env python3
"""Explicit task/target/horizon/ablation conditioning for the mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ConditionKey:
    task: str
    target: str
    horizon: int
    ablation: str


@dataclass(frozen=True)
class ConditioningSchema:
    task_vocab: Tuple[str, ...] = ("task1_outcome", "task2_forecast", "task3_risk_adjust")
    target_vocab: Tuple[str, ...] = ("funding_raised_usd", "investors_count", "is_funded")
    ablation_vocab: Tuple[str, ...] = (
        "core_only",
        "core_only_seed2",
        "core_text",
        "core_edgar",
        "core_edgar_seed2",
        "full",
    )
    horizon_vocab: Tuple[int, ...] = (1, 7, 14, 30)

    def as_dict(self) -> Dict[str, object]:
        return {
            "task_vocab": self.task_vocab,
            "target_vocab": self.target_vocab,
            "ablation_vocab": self.ablation_vocab,
            "horizon_vocab": self.horizon_vocab,
        }


class MainlineConditionEncoder:
    """Encodes the fixed mainline condition contract into stable ids."""

    def __init__(self, schema: ConditioningSchema | None = None):
        self.schema = schema or ConditioningSchema()
        self._task_map = {name: idx for idx, name in enumerate(self.schema.task_vocab)}
        self._target_map = {name: idx for idx, name in enumerate(self.schema.target_vocab)}
        self._ablation_map = {name: idx for idx, name in enumerate(self.schema.ablation_vocab)}
        self._horizon_map = {value: idx for idx, value in enumerate(self.schema.horizon_vocab)}

    def ids(self, key: ConditionKey) -> Dict[str, int]:
        return {
            "task_id": self._task_map[key.task],
            "target_id": self._target_map[key.target],
            "horizon_id": self._horizon_map[key.horizon],
            "ablation_id": self._ablation_map[key.ablation],
        }

    def encode(self, key: ConditionKey) -> np.ndarray:
        encoded = self.ids(key)
        return np.asarray(
            [
                encoded["task_id"],
                encoded["target_id"],
                encoded["horizon_id"],
                encoded["ablation_id"],
            ],
            dtype=np.int64,
        )

    def broadcast(self, key: ConditionKey, n_rows: int) -> np.ndarray:
        if n_rows <= 0:
            return np.zeros((0, self.feature_width()), dtype=np.float32)

        ids = self.ids(key)
        width = self.feature_width()
        matrix = np.zeros((n_rows, width), dtype=np.float32)
        task_offset = 0
        target_offset = task_offset + len(self.schema.task_vocab)
        ablation_offset = target_offset + len(self.schema.target_vocab)
        horizon_offset = ablation_offset + len(self.schema.ablation_vocab)
        scalar_offset = horizon_offset + len(self.schema.horizon_vocab)

        matrix[:, task_offset + ids["task_id"]] = 1.0
        matrix[:, target_offset + ids["target_id"]] = 1.0
        matrix[:, ablation_offset + ids["ablation_id"]] = 1.0
        matrix[:, horizon_offset + ids["horizon_id"]] = 1.0
        matrix[:, scalar_offset] = float(key.horizon) / float(max(self.schema.horizon_vocab))
        return matrix

    def encode_many(self, keys: Iterable[ConditionKey]) -> np.ndarray:
        keys_list = list(keys)
        if not keys_list:
            return np.zeros((0, self.feature_width()), dtype=np.float32)
        return np.concatenate([self.broadcast(key, 1) for key in keys_list], axis=0)

    def feature_width(self) -> int:
        return (
            len(self.schema.task_vocab)
            + len(self.schema.target_vocab)
            + len(self.schema.ablation_vocab)
            + len(self.schema.horizon_vocab)
            + 1
        )