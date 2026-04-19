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
    enable_temporal_state_features: bool = True
    enable_spectral_state_features: bool = True
    enable_process_state_feedback: bool = False
    process_state_feedback_strength: float = 0.0
    process_state_feedback_source_decay: float = 0.65
    process_state_feedback_min_horizon: int = 7
    process_state_feedback_state_weights: Tuple[float, ...] = (0.25, 0.25, 0.0, 0.0, 0.5)
    process_state_feedback_max_norm_share: float = 0.02
    process_state_feedback_predict_scale_cap: float = 4.0
    # Hawkes-process-inspired asymmetric event-driven intensity state.
    # Unlike rolling-window statistics that treat positive/negative shocks
    # symmetrically, this accumulates only POSITIVE jumps with exponential
    # decay at multiple timescales.  Captures the self-exciting structure of
    # investor-arrival and financing-momentum processes.
    # Reference: Hawkes (1971); Aït-Sahalia et al. (2014); Neural Hawkes
    # (NeurIPS 2017); EasyTPP (NeurIPS 2023).
    enable_hawkes_financing_state: bool = False
    hawkes_financing_decay_halflives: Tuple[float, ...] = (7.0, 30.0, 90.0)
    hawkes_positive_shock_threshold: float = 0.5

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
            "enable_temporal_state_features": self.enable_temporal_state_features,
            "enable_spectral_state_features": self.enable_spectral_state_features,
            "enable_process_state_feedback": self.enable_process_state_feedback,
            "process_state_feedback_strength": self.process_state_feedback_strength,
            "process_state_feedback_source_decay": self.process_state_feedback_source_decay,
            "process_state_feedback_min_horizon": self.process_state_feedback_min_horizon,
            "process_state_feedback_state_weights": self.process_state_feedback_state_weights,
            "process_state_feedback_max_norm_share": self.process_state_feedback_max_norm_share,
            "process_state_feedback_predict_scale_cap": self.process_state_feedback_predict_scale_cap,
            "enable_hawkes_financing_state": self.enable_hawkes_financing_state,
            "hawkes_financing_decay_halflives": self.hawkes_financing_decay_halflives,
            "hawkes_positive_shock_threshold": self.hawkes_positive_shock_threshold,
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
        hawkes_state, hawkes_names = self._build_event_driven_intensity_state(
            standardized, context_frame=context_frame
        )
        state = np.concatenate([compact, summaries, temporal_state, spectral_state, hawkes_state], axis=1).astype(np.float32, copy=False)
        self._state_layout = {
            "uses_multiscale_temporal_state": bool(self.spec.enable_multiscale_temporal_state),
            "uses_temporal_state_features": bool(
                self.spec.enable_multiscale_temporal_state and self.spec.enable_temporal_state_features
            ),
            "uses_spectral_state_features": bool(
                self.spec.enable_multiscale_temporal_state and self.spec.enable_spectral_state_features
            ),
            "uses_hawkes_financing_state": bool(self.spec.enable_hawkes_financing_state),
            "compact_state_dim": int(compact.shape[1]),
            "summary_state_dim": int(summaries.shape[1]),
            "temporal_state_dim": int(temporal_state.shape[1]),
            "spectral_state_dim": int(spectral_state.shape[1]),
            "hawkes_state_dim": int(hawkes_state.shape[1]),
            "shared_state_dim": int(state.shape[1]),
            "temporal_feature_names": tuple(state_names["temporal"]),
            "spectral_feature_names": tuple(state_names["spectral"]),
            "hawkes_feature_names": tuple(hawkes_names),
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
        if not self.spec.enable_temporal_state_features and not self.spec.enable_spectral_state_features:
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
        if not self.spec.enable_temporal_state_features and not self.spec.enable_spectral_state_features:
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
        if self.spec.enable_temporal_state_features:
            temporal_state = runtime.loc[:, list(temporal_names)].to_numpy(dtype=np.float32, copy=False)
        else:
            temporal_state = np.zeros((len(runtime), 0), dtype=np.float32)
            temporal_names = ()
        if self.spec.enable_spectral_state_features:
            spectral_state = runtime.loc[:, list(spectral_names)].to_numpy(dtype=np.float32, copy=False)
        else:
            spectral_state = np.zeros((len(runtime), 0), dtype=np.float32)
            spectral_names = ()
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

    def _build_event_driven_intensity_state(
        self,
        standardized: np.ndarray,
        context_frame: pd.DataFrame | None,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Hawkes-process-inspired asymmetric event-driven intensity state.

        For each entity time series, compute exponentially decaying sums of
        POSITIVE state-level jumps at multiple timescales.  The recursive
        formula I_t = decay * I_{t-1} + max(0, velocity_t - threshold)
        approximates a discrete-time Hawkes intensity (Hawkes 1971).

        Features:
          - hawkes_intensity_h<N>: decayed positive-shock sum at halflife N steps
          - hawkes_time_since_event: steps since last positive shock, normalised
            by entity tenure (0 = just happened, 1 = never happened / very old)
        """
        halflives = self.spec.hawkes_financing_decay_halflives
        threshold = float(self.spec.hawkes_positive_shock_threshold)
        n_rows = standardized.shape[0]
        n_scales = len(halflives)
        feature_names: Tuple[str, ...] = tuple(
            f"hawkes_intensity_h{int(h)}" for h in halflives
        ) + ("hawkes_time_since_event",)

        if not self.spec.enable_hawkes_financing_state or n_rows == 0:
            return np.zeros((n_rows, 0), dtype=np.float32), ()

        # State level = mean of all standardised features (scalar per row)
        state_level = standardized.mean(axis=1).astype(np.float32)
        # Positive velocity only (investor/funding arrivals are unidirectional)
        velocity = np.empty(n_rows, dtype=np.float32)
        velocity[0] = 0.0
        velocity[1:] = state_level[1:] - state_level[:-1]
        positive_shock = np.maximum(0.0, velocity - threshold)

        # Decay factor per halflife: alpha = exp(-ln2 / h)
        decay_factors = np.array(
            [float(np.exp(-np.log(2.0) / max(1.0, float(h)))) for h in halflives],
            dtype=np.float32,
        )

        # Per-entity grouping
        if context_frame is not None and "entity_id" in context_frame.columns:
            entity_ids = context_frame["entity_id"].astype(str).to_numpy(copy=False)
        else:
            entity_ids = np.repeat("__global__", n_rows)

        intensities = np.zeros((n_rows, n_scales), dtype=np.float32)
        time_since = np.ones(n_rows, dtype=np.float32)  # default = 1 (no event)

        # Build ordered list of unique entities (preserve encounter order)
        seen: dict[str, list[int]] = {}
        for i, eid in enumerate(entity_ids):
            if eid not in seen:
                seen[eid] = []
            seen[eid].append(i)

        for row_indices in seen.values():
            e_n = len(row_indices)
            e_shock = positive_shock[row_indices]

            # Recursive Hawkes intensity
            e_intensities = np.zeros((e_n, n_scales), dtype=np.float32)
            last_I = np.zeros(n_scales, dtype=np.float32)
            for t in range(e_n):
                last_I = decay_factors * last_I + e_shock[t]
                e_intensities[t] = last_I
            intensities[row_indices] = e_intensities

            # Time-since-last-positive-event (normalised by entity tenure)
            e_time_since = np.ones(e_n, dtype=np.float32)
            last_event_step = -1
            for t in range(e_n):
                if e_shock[t] > 0.0:
                    last_event_step = t
                if last_event_step < 0:
                    e_time_since[t] = 1.0  # no event yet
                else:
                    e_time_since[t] = float(t - last_event_step) / max(1.0, float(e_n - 1))
            time_since[row_indices] = e_time_since

        result = np.concatenate([intensities, time_since[:, None]], axis=1).astype(np.float32)
        return result, feature_names

    def _empty_state_layout(self, compact_dim: int = 0, summary_dim: int = 0) -> Dict[str, Any]:
        return {
            "uses_multiscale_temporal_state": bool(self.spec.enable_multiscale_temporal_state),
            "uses_temporal_state_features": bool(
                self.spec.enable_multiscale_temporal_state and self.spec.enable_temporal_state_features
            ),
            "uses_spectral_state_features": bool(
                self.spec.enable_multiscale_temporal_state and self.spec.enable_spectral_state_features
            ),
            "uses_hawkes_financing_state": bool(self.spec.enable_hawkes_financing_state),
            "compact_state_dim": int(compact_dim),
            "summary_state_dim": int(summary_dim),
            "temporal_state_dim": 0,
            "spectral_state_dim": 0,
            "hawkes_state_dim": 0,
            "shared_state_dim": int(compact_dim + summary_dim),
            "temporal_feature_names": (),
            "spectral_feature_names": (),
            "hawkes_feature_names": (),
            "seed_rows_required": int(self.required_seed_rows()),
        }