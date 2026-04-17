#!/usr/bin/env python3
"""Shared trunk contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

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
    event_state_schema_version: str = "event_state_v2"
    event_state_atoms: Tuple[str, ...] = (
        "financing_phase",
        "event_boundary",
        "funded_flip",
        "funding_jump_severity",
        "investor_jump_process",
        "goal_crossing",
        "financing_persistence",
        "source_topology",
        "source_arrival_decay",
        "shared_state_energy",
    )
    enable_multiscale_temporal_state: bool = False
    temporal_state_windows: Tuple[int, ...] = (3, 7, 30)

    def as_dict(self) -> Dict[str, object]:
        return {
            "decomposition_stages": self.decomposition_stages,
            "multi_resolution_levels": self.multi_resolution_levels,
            "patch_sizes": self.patch_sizes,
            "compact_state_dim": self.compact_state_dim,
            "target_agnostic": self.target_agnostic,
            "source_native_memory_external": self.source_native_memory_external,
            "event_state_schema_version": self.event_state_schema_version,
            "event_state_atoms": self.event_state_atoms,
            "enable_multiscale_temporal_state": self.enable_multiscale_temporal_state,
            "temporal_state_windows": self.temporal_state_windows,
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
        self._state_layout: Dict[str, Any] = self._empty_state_layout()

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
        self._state_layout = self._empty_state_layout(compact_dim=compact_dim, summary_dim=self._summary_dim())
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
        seed_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._fitted or self._feature_mean is None or self._feature_scale is None or self._projection is None:
            raise ValueError("SharedTemporalBackbone is not fitted")

        numeric = self._numeric_matrix(frame)
        standardized = (numeric - self._feature_mean) / self._feature_scale
        compact = standardized @ self._projection
        summaries = self._build_summaries(standardized)
        temporal_state, spectral_state, state_names = self._build_multiscale_state(
            standardized,
            context_frame=context_frame,
            seed_frame=seed_frame,
        )
        state = np.concatenate([compact, summaries, temporal_state, spectral_state], axis=1).astype(np.float32, copy=False)
        self._state_layout = {
            "uses_multiscale_temporal_state": bool(self.spec.enable_multiscale_temporal_state),
            "compact_state_dim": int(compact.shape[1]),
            "summary_state_dim": int(summaries.shape[1]),
            "temporal_state_dim": int(temporal_state.shape[1]),
            "spectral_state_dim": int(spectral_state.shape[1]),
            "shared_state_dim": int(state.shape[1]),
            "temporal_feature_names": tuple(state_names["temporal"]),
            "spectral_feature_names": tuple(state_names["spectral"]),
            "seed_rows_required": int(self.required_seed_rows()),
        }
        return state

    def fit_transform(
        self,
        frame: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        return self.fit(frame, feature_cols=feature_cols).transform(frame, context_frame=context_frame)

    def describe_state_layout(self) -> Dict[str, Any]:
        return dict(self._state_layout)

    def required_seed_rows(self) -> int:
        if not self.spec.enable_multiscale_temporal_state:
            return 0
        return max(self._normalized_multiscale_windows())

    def build_context_seed(self, context_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame | None:
        if not self._fitted or not self.spec.enable_multiscale_temporal_state:
            return None
        required = {"entity_id", "crawled_date_day"}
        if not required.issubset(context_frame.columns):
            return None
        if len(context_frame) != len(feature_frame):
            raise ValueError("Backbone context seed requires aligned context and feature frames")
        seed = context_frame[["entity_id", "crawled_date_day"]].copy()
        seed["entity_id"] = seed["entity_id"].astype(str)
        seed["crawled_date_day"] = pd.to_datetime(seed["crawled_date_day"], errors="coerce")
        aligned = feature_frame.reindex(columns=self.feature_cols, fill_value=0.0)
        for col in self.feature_cols:
            seed[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        seed.sort_values(["entity_id", "crawled_date_day"], inplace=True, kind="mergesort")
        return seed.groupby("entity_id", sort=False).tail(self.required_seed_rows()).reset_index(drop=True)

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

    def _summary_dim(self) -> int:
        return 4 + int(sum(self._summary_splits))

    def _build_multiscale_state(
        self,
        standardized: np.ndarray,
        *,
        context_frame: pd.DataFrame | None,
        seed_frame: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Tuple[str, ...]]]:
        temporal_names = (
            "entity_progress",
            "state_level",
            "state_energy",
            "state_velocity",
            "state_acceleration",
            "state_persistence",
            "state_shock_pressure",
        )
        spectral_names = (
            "low_band_level",
            "mid_band_level",
            "high_band_level",
            "low_band_energy",
            "mid_band_energy",
            "high_band_energy",
        )
        if not self.spec.enable_multiscale_temporal_state or standardized.shape[0] == 0:
            empty = np.zeros((standardized.shape[0], 0), dtype=np.float32)
            return empty, empty, {"temporal": (), "spectral": ()}

        current = self._build_context_state_frame(
            standardized,
            context_frame=context_frame,
            row_order=np.arange(standardized.shape[0], dtype=np.int64),
            is_runtime_row=True,
        )
        frames = [current]
        if seed_frame is not None and len(seed_frame) > 0:
            seed_numeric = self._numeric_matrix(seed_frame)
            seed_standardized = (seed_numeric - self._feature_mean) / self._feature_scale
            seed = self._build_context_state_frame(
                seed_standardized,
                context_frame=seed_frame,
                row_order=-np.arange(seed_standardized.shape[0], 0, -1, dtype=np.int64),
                is_runtime_row=False,
            )
            frames.insert(0, seed)

        work = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        work.sort_values(["entity_id", "crawled_date_day", "_row_order"], inplace=True, kind="mergesort")
        grouped = work.groupby("entity_id", sort=False)
        entity_size = grouped["state_level"].transform("size").astype(np.float32)
        entity_step = grouped.cumcount().astype(np.float32)
        work["entity_progress"] = np.where(entity_size > 1.0, entity_step / np.maximum(entity_size - 1.0, 1.0), 0.0)
        work["state_velocity"] = grouped["state_level"].diff().fillna(0.0)
        grouped = work.groupby("entity_id", sort=False)
        work["state_acceleration"] = grouped["state_velocity"].diff().fillna(0.0)

        short_window, mid_window, long_window = self._normalized_multiscale_windows()
        level_short = grouped["state_level"].transform(lambda series: series.rolling(short_window, min_periods=1).mean())
        level_mid = grouped["state_level"].transform(lambda series: series.rolling(mid_window, min_periods=1).mean())
        level_long = grouped["state_level"].transform(lambda series: series.rolling(long_window, min_periods=1).mean())
        energy_short = grouped["state_energy"].transform(lambda series: series.rolling(short_window, min_periods=1).mean())
        energy_mid = grouped["state_energy"].transform(lambda series: series.rolling(mid_window, min_periods=1).mean())
        energy_long = grouped["state_energy"].transform(lambda series: series.rolling(long_window, min_periods=1).mean())
        grouped = work.groupby("entity_id", sort=False)
        work["state_persistence"] = grouped["state_velocity"].transform(
            lambda series: (1.0 / (1.0 + np.abs(series))).rolling(mid_window, min_periods=1).mean()
        )
        work["state_shock_pressure"] = (
            np.abs(work["state_level"] - level_short)
            + np.abs(work["state_energy"] - energy_short)
            + np.abs(work["state_acceleration"])
        )
        work["low_band_level"] = level_long
        work["mid_band_level"] = level_mid - level_long
        work["high_band_level"] = level_short - level_mid
        work["low_band_energy"] = energy_long
        work["mid_band_energy"] = energy_mid - energy_long
        work["high_band_energy"] = energy_short - energy_mid

        runtime = work[work["_is_runtime_row"]].copy()
        runtime.sort_values("_row_order", inplace=True, kind="mergesort")
        temporal_state = runtime.loc[:, list(temporal_names)].to_numpy(dtype=np.float32, copy=False)
        spectral_state = runtime.loc[:, list(spectral_names)].to_numpy(dtype=np.float32, copy=False)
        return temporal_state, spectral_state, {
            "temporal": temporal_names,
            "spectral": spectral_names,
        }

    def _build_context_state_frame(
        self,
        standardized: np.ndarray,
        *,
        context_frame: pd.DataFrame | None,
        row_order: np.ndarray,
        is_runtime_row: bool,
    ) -> pd.DataFrame:
        n_rows = standardized.shape[0]
        if context_frame is not None and len(context_frame) != n_rows:
            raise ValueError("Backbone context frame must align with the number of rows being transformed")
        if context_frame is not None and "entity_id" in context_frame.columns:
            entity_id = context_frame["entity_id"].astype(str).to_numpy(copy=False)
        else:
            entity_id = np.repeat("global", n_rows)
        if context_frame is not None and "crawled_date_day" in context_frame.columns:
            crawled_date_day = pd.to_datetime(context_frame["crawled_date_day"], errors="coerce")
        else:
            crawled_date_day = pd.Series(pd.NaT, index=np.arange(n_rows))
        return pd.DataFrame(
            {
                "entity_id": entity_id,
                "crawled_date_day": pd.to_datetime(crawled_date_day, errors="coerce"),
                "_row_order": np.asarray(row_order, dtype=np.int64),
                "_is_runtime_row": np.repeat(bool(is_runtime_row), n_rows),
                "state_level": standardized.mean(axis=1).astype(np.float32, copy=False),
                "state_energy": np.sqrt(np.mean(np.square(standardized), axis=1)).astype(np.float32, copy=False),
            }
        )

    def _normalized_multiscale_windows(self) -> Tuple[int, int, int]:
        windows = sorted({max(1, int(window)) for window in self.spec.temporal_state_windows})
        if not windows:
            return (1, 1, 1)
        if len(windows) == 1:
            return (windows[0], windows[0], windows[0])
        if len(windows) == 2:
            return (windows[0], windows[1], windows[1])
        return (windows[0], windows[1], windows[-1])

    def _empty_state_layout(self, compact_dim: int = 0, summary_dim: int = 0) -> Dict[str, Any]:
        return {
            "uses_multiscale_temporal_state": bool(self.spec.enable_multiscale_temporal_state),
            "compact_state_dim": int(compact_dim),
            "summary_state_dim": int(summary_dim),
            "temporal_state_dim": 0,
            "spectral_state_dim": 0,
            "shared_state_dim": int(compact_dim + summary_dim),
            "temporal_feature_names": (),
            "spectral_feature_names": (),
            "seed_rows_required": int(self.required_seed_rows()),
        }