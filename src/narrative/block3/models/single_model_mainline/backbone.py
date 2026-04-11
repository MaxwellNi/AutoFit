#!/usr/bin/env python3
"""Shared trunk contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SharedTemporalBackboneSpec:
    """Minimal contract for the shared temporal trunk.

    The backbone is deliberately target-agnostic after conditioning and before
    the hard target barrier.
    """

    decomposition_stages: int = 3
    multi_resolution_levels: Tuple[int, ...] = (1, 7, 30)
    patch_sizes: Tuple[int, ...] = (4, 8, 16)
    compact_state_dim: int = 64
    target_agnostic: bool = True
    source_native_memory_external: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "decomposition_stages": self.decomposition_stages,
            "multi_resolution_levels": self.multi_resolution_levels,
            "patch_sizes": self.patch_sizes,
            "compact_state_dim": self.compact_state_dim,
            "target_agnostic": self.target_agnostic,
            "source_native_memory_external": self.source_native_memory_external,
        }


class SharedTemporalBackbone:
    """Compact shared-state backbone for the mainline runtime owner.

    The current runtime works on freeze-safe dense feature rows. It keeps source
    columns outside the trunk and turns the remaining state stream into a stable,
    low-dimensional shared representation with deterministic projections and
    multi-resolution summaries.
    """

    def __init__(self, spec: SharedTemporalBackboneSpec | None = None, random_state: int = 0):
        self.spec = spec or SharedTemporalBackboneSpec()
        self.random_state = int(random_state)
        self.feature_cols: list[str] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._projection: np.ndarray | None = None
        self._summary_splits: Tuple[int, ...] = ()
        self._fitted = False

    def fit(self, frame: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> "SharedTemporalBackbone":
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(frame.columns)
        if not self.feature_cols:
            raise ValueError("SharedTemporalBackbone requires at least one feature column")

        numeric = self._numeric_matrix(frame)
        self._feature_mean = numeric.mean(axis=0).astype(np.float32, copy=False)
        scale = numeric.std(axis=0).astype(np.float32, copy=False)
        scale[scale < 1e-6] = 1.0
        self._feature_scale = scale

        input_dim = numeric.shape[1]
        compact_dim = max(8, min(self.spec.compact_state_dim, input_dim))
        rng = np.random.default_rng(self.random_state + input_dim + compact_dim)
        self._projection = (
            rng.standard_normal((input_dim, compact_dim)).astype(np.float32)
            / max(1.0, np.sqrt(float(input_dim)))
        )
        self._summary_splits = tuple(
            max(1, min(input_dim, int(level))) for level in self.spec.multi_resolution_levels
        )
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._feature_mean is None or self._feature_scale is None or self._projection is None:
            raise ValueError("SharedTemporalBackbone is not fitted")

        numeric = self._numeric_matrix(frame)
        standardized = (numeric - self._feature_mean) / self._feature_scale
        compact = standardized @ self._projection
        summaries = self._build_summaries(standardized)
        return np.concatenate([compact, summaries], axis=1).astype(np.float32, copy=False)

    def fit_transform(self, frame: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> np.ndarray:
        return self.fit(frame, feature_cols=feature_cols).transform(frame)

    def _numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = frame.reindex(columns=self.feature_cols, fill_value=0.0)
        numeric = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return numeric.to_numpy(dtype=np.float32, copy=False)

    def _build_summaries(self, standardized: np.ndarray) -> np.ndarray:
        n_rows = standardized.shape[0]
        scalar_parts = [
            standardized.mean(axis=1, keepdims=True),
            standardized.std(axis=1, keepdims=True),
            standardized.min(axis=1, keepdims=True),
            standardized.max(axis=1, keepdims=True),
        ]
        chunk_parts = []
        for n_chunks in self._summary_splits:
            indices = np.array_split(np.arange(standardized.shape[1]), n_chunks)
            chunk_mean = np.column_stack(
                [
                    standardized[:, idx].mean(axis=1) if len(idx) else np.zeros(n_rows, dtype=np.float32)
                    for idx in indices
                ]
            )
            chunk_parts.append(chunk_mean.astype(np.float32, copy=False))
        return np.concatenate([*scalar_parts, *chunk_parts], axis=1).astype(np.float32, copy=False)