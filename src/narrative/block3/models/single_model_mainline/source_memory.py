#!/usr/bin/env python3
"""Sparse source-memory boundary for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..v740_multisource_features import (
    DualClockConfig,
    build_dual_clock_memory_for_entity,
    build_source_native_edgar_memory,
    build_source_native_text_memory,
    infer_edgar_columns,
    infer_text_columns,
)


@dataclass(frozen=True)
class SourceMemoryContract:
    edgar_as_sparse_event_memory: bool = True
    text_as_sparse_event_memory: bool = True
    uses_recency_tokens: bool = True
    uses_novelty_proxy: bool = True
    uses_target_conditioned_readout: bool = True
    exposes_read_confidence: bool = True
    exposes_coverage_adequacy: bool = True
    low_coverage_threshold: float = 0.05
    adequate_coverage_threshold: float = 0.20
    read_low_confidence_threshold: float = 0.10
    read_high_confidence_threshold: float = 0.55
    no_read_fallback_mode: str = "zero_features_with_metadata_flag"

    def as_dict(self) -> Dict[str, object]:
        return {
            "edgar_as_sparse_event_memory": self.edgar_as_sparse_event_memory,
            "text_as_sparse_event_memory": self.text_as_sparse_event_memory,
            "uses_recency_tokens": self.uses_recency_tokens,
            "uses_novelty_proxy": self.uses_novelty_proxy,
            "uses_target_conditioned_readout": self.uses_target_conditioned_readout,
            "exposes_read_confidence": self.exposes_read_confidence,
            "exposes_coverage_adequacy": self.exposes_coverage_adequacy,
            "low_coverage_threshold": self.low_coverage_threshold,
            "adequate_coverage_threshold": self.adequate_coverage_threshold,
            "read_low_confidence_threshold": self.read_low_confidence_threshold,
            "read_high_confidence_threshold": self.read_high_confidence_threshold,
            "no_read_fallback_mode": self.no_read_fallback_mode,
        }


@dataclass(frozen=True)
class SourceColumnLayout:
    edgar_cols: List[str]
    text_cols: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {"edgar_cols": list(self.edgar_cols), "text_cols": list(self.text_cols)}


@dataclass(frozen=True)
class SourceMemoryBatch:
    edgar_recent: np.ndarray
    edgar_bucket: np.ndarray
    text_recent: np.ndarray
    text_bucket: np.ndarray
    density: Dict[str, float]


class SourceMemoryAssembler:
    """Stable sparse-memory interface backed by the current V740 feature builders."""

    def __init__(self, cfg: DualClockConfig | None = None, contract: SourceMemoryContract | None = None):
        self.cfg = cfg or DualClockConfig()
        self.contract = contract or SourceMemoryContract()

    def infer_layout(self, frame: pd.DataFrame) -> SourceColumnLayout:
        edgar_cols = infer_edgar_columns(frame) if self.contract.edgar_as_sparse_event_memory else []
        text_cols = infer_text_columns(frame) if self.contract.text_as_sparse_event_memory else []
        return SourceColumnLayout(edgar_cols=edgar_cols, text_cols=text_cols)

    def build_runtime_features(
        self,
        frame: pd.DataFrame,
        layout: SourceColumnLayout | None = None,
    ) -> pd.DataFrame:
        layout = layout or self.infer_layout(frame)
        edgar = self._summarize_source_block(frame, layout.edgar_cols, prefix="edgar")
        text = self._summarize_source_block(frame, layout.text_cols, prefix="text", with_novelty=True)
        features = pd.concat([edgar, text], axis=1)
        return features.fillna(0.0).astype(np.float32)

    def build_target_read_policy_features(
        self,
        source_frame: pd.DataFrame,
        *,
        ablation: str,
        horizon: int | float = 1,
    ) -> pd.DataFrame:
        out = self.apply_ablation_permissions(source_frame, ablation=ablation)
        text_conf = self._runtime_source_confidence(out, "text")
        edgar_conf = self._runtime_source_confidence(out, "edgar")
        text_allowed, edgar_allowed = self.ablation_permissions(ablation)

        out["text_confidence"] = text_conf.astype(np.float32)
        out["edgar_confidence"] = edgar_conf.astype(np.float32)
        out["text_ablation_allowed"] = np.full(len(out), float(text_allowed), dtype=np.float32)
        out["edgar_ablation_allowed"] = np.full(len(out), float(edgar_allowed), dtype=np.float32)
        out["text_effective_confidence"] = (text_conf * float(text_allowed)).astype(np.float32)
        out["edgar_effective_confidence"] = (edgar_conf * float(edgar_allowed)).astype(np.float32)
        out["source_any_allowed"] = np.full(len(out), float(text_allowed or edgar_allowed), dtype=np.float32)
        text_active = pd.to_numeric(out.get("text_active", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        edgar_active = pd.to_numeric(out.get("edgar_active", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        text_nonzero = pd.to_numeric(out.get("text_nonzero_share", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        edgar_nonzero = pd.to_numeric(out.get("edgar_nonzero_share", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        out["source_any_available"] = (
            (text_active > 0.0) | (edgar_active > 0.0)
        ).astype(np.float32)
        out["source_any_nonzero"] = (
            (text_nonzero > 0.0) | (edgar_nonzero > 0.0)
        ).astype(np.float32)

        horizon_value = float(horizon)
        text_eff = out["text_effective_confidence"].to_numpy(dtype=np.float32, copy=False)
        edgar_eff = out["edgar_effective_confidence"].to_numpy(dtype=np.float32, copy=False)
        short_weight = 0.75 if horizon_value <= 1.0 else 0.55
        out["binary_read_confidence"] = np.maximum(text_eff, edgar_eff).astype(np.float32)
        out["funding_read_confidence"] = (0.45 * text_eff + 0.55 * edgar_eff).astype(np.float32)
        out["investors_read_confidence"] = (short_weight * text_eff + (1.0 - short_weight) * edgar_eff).astype(np.float32)
        for lane in ("binary", "funding", "investors"):
            conf = out[f"{lane}_read_confidence"].to_numpy(dtype=np.float32, copy=False)
            out[f"{lane}_no_read_fallback"] = (conf < self.contract.read_low_confidence_threshold).astype(np.float32)
            out[f"{lane}_low_confidence_read"] = (
                (conf >= self.contract.read_low_confidence_threshold)
                & (conf < self.contract.read_high_confidence_threshold)
            ).astype(np.float32)
            out[f"{lane}_high_confidence_read"] = (
                conf >= self.contract.read_high_confidence_threshold
            ).astype(np.float32)
        return out.fillna(0.0).astype(np.float32)

    def apply_ablation_permissions(self, source_frame: pd.DataFrame, *, ablation: str) -> pd.DataFrame:
        out = source_frame.copy()
        text_allowed, edgar_allowed = self.ablation_permissions(ablation)
        if not text_allowed:
            for column in [col for col in out.columns if col.startswith("text_")]:
                out[column] = 0.0
        if not edgar_allowed:
            for column in [col for col in out.columns if col.startswith("edgar_")]:
                out[column] = 0.0
        return out

    def ablation_permissions(self, ablation: str) -> tuple[bool, bool]:
        name = str(ablation)
        text_allowed = name in {"core_text", "full"}
        edgar_allowed = name in {"core_edgar", "full"}
        return text_allowed, edgar_allowed

    def build_entity_memory(
        self,
        entity_df: pd.DataFrame,
        prediction_time: pd.Timestamp,
        layout: SourceColumnLayout,
    ) -> SourceMemoryBatch:
        edgar_recent, edgar_bucket = self._build_edgar_memory(entity_df, prediction_time, layout.edgar_cols)
        text_recent, text_bucket = self._build_text_memory(entity_df, prediction_time, layout.text_cols)
        density = {
            "edgar_row_density": self._row_density(entity_df, layout.edgar_cols),
            "text_row_density": self._row_density(entity_df, layout.text_cols),
        }
        return SourceMemoryBatch(
            edgar_recent=edgar_recent,
            edgar_bucket=edgar_bucket,
            text_recent=text_recent,
            text_bucket=text_bucket,
            density=density,
        )

    def summarize_contract_from_stats(
        self,
        edgar_cols: List[str],
        text_cols: List[str],
        edgar_row_density: float,
        text_row_density: float,
    ) -> Dict[str, object]:
        return {
            "fallback_policy": {
                "mode": self.contract.no_read_fallback_mode,
                "low_coverage_threshold": self.contract.low_coverage_threshold,
                "adequate_coverage_threshold": self.contract.adequate_coverage_threshold,
            },
            "sources": {
                "edgar": self._contract_source_view(
                    source_cols=edgar_cols,
                    row_density=edgar_row_density,
                    enabled=self.contract.edgar_as_sparse_event_memory,
                ),
                "text": self._contract_source_view(
                    source_cols=text_cols,
                    row_density=text_row_density,
                    enabled=self.contract.text_as_sparse_event_memory,
                ),
            },
        }

    def _build_edgar_memory(
        self,
        entity_df: pd.DataFrame,
        prediction_time: pd.Timestamp,
        source_cols: List[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not source_cols:
            return self._empty_recent(3), self._empty_bucket(4)
        if "edgar_filed_date" in entity_df.columns or "cutoff_ts" in entity_df.columns:
            memory = build_source_native_edgar_memory(entity_df, prediction_time, source_cols, self.cfg)
        else:
            memory = build_dual_clock_memory_for_entity(entity_df, prediction_time, source_cols, self.cfg)
        return memory["recent_tokens"], memory["bucket_tokens"]

    def _build_text_memory(
        self,
        entity_df: pd.DataFrame,
        prediction_time: pd.Timestamp,
        source_cols: List[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not source_cols:
            return self._empty_recent(5), self._empty_bucket(6)
        event_col = "snapshot_ts" if "snapshot_ts" in entity_df.columns else self.cfg.time_col
        memory = build_source_native_text_memory(entity_df, prediction_time, source_cols, self.cfg, event_col=event_col)
        return memory["recent_tokens"], memory["bucket_tokens"]

    def _row_density(self, frame: pd.DataFrame, source_cols: List[str]) -> float:
        if frame.empty or not source_cols:
            return 0.0
        present = frame[source_cols].notna().any(axis=1)
        return float(present.mean()) if len(present) else 0.0

    def _contract_source_view(
        self,
        source_cols: List[str],
        row_density: float,
        enabled: bool,
    ) -> Dict[str, object]:
        density = float(np.clip(row_density, 0.0, 1.0))
        col_count = len(source_cols)
        if (not enabled) or col_count == 0:
            read_confidence = "absent"
            coverage_adequacy = "fallback_only"
            no_read_fallback_active = True
        elif density < self.contract.low_coverage_threshold:
            read_confidence = "low"
            coverage_adequacy = "fallback_only"
            no_read_fallback_active = True
        elif density < self.contract.adequate_coverage_threshold:
            read_confidence = "medium"
            coverage_adequacy = "partial"
            no_read_fallback_active = False
        else:
            read_confidence = "high"
            coverage_adequacy = "adequate"
            no_read_fallback_active = False
        return {
            "enabled": enabled,
            "column_count": col_count,
            "row_density": density,
            "read_confidence": read_confidence if self.contract.exposes_read_confidence else "disabled",
            "coverage_adequacy": coverage_adequacy if self.contract.exposes_coverage_adequacy else "disabled",
            "no_read_fallback_active": no_read_fallback_active,
        }

    def _runtime_source_confidence(self, source_frame: pd.DataFrame, prefix: str) -> pd.Series:
        index = source_frame.index

        def _col(name: str, default: float = 0.0) -> pd.Series:
            return pd.to_numeric(
                source_frame.get(name, pd.Series(default, index=index, dtype=np.float32)),
                errors="coerce",
            ).fillna(default).astype(np.float32)

        active = _col(f"{prefix}_active").clip(lower=0.0, upper=1.0)
        nonzero = _col(f"{prefix}_nonzero_share").clip(lower=0.0, upper=1.0)
        abs_max = _col(f"{prefix}_abs_max").clip(lower=0.0)
        recency = _col(f"{prefix}_recency_days", 9999.0).clip(lower=0.0)
        freshness = pd.Series(np.exp(-recency.to_numpy(dtype=np.float64, copy=False) / 365.0), index=index)
        magnitude = pd.Series(np.tanh(abs_max.to_numpy(dtype=np.float64, copy=False)), index=index)
        confidence = active * (0.45 * nonzero + 0.35 * freshness.astype(np.float32) + 0.20 * magnitude.astype(np.float32))
        return confidence.clip(lower=0.0, upper=1.0).astype(np.float32)

    def _empty_recent(self, leading_width: int) -> np.ndarray:
        width = leading_width
        return np.zeros((self.cfg.max_events, width), dtype=np.float32)

    def _empty_bucket(self, leading_width: int) -> np.ndarray:
        width = len(self.cfg.recency_buckets) + 1
        return np.zeros((width, leading_width), dtype=np.float32)

    def _summarize_source_block(
        self,
        frame: pd.DataFrame,
        source_cols: List[str],
        prefix: str,
        with_novelty: bool = False,
    ) -> pd.DataFrame:
        index = frame.index
        base = pd.DataFrame(index=index)
        if not source_cols:
            return self._empty_runtime_block(index, prefix, with_novelty=with_novelty)

        numeric = frame.reindex(columns=source_cols).apply(pd.to_numeric, errors="coerce")
        filled = numeric.fillna(0.0)
        active = numeric.notna().any(axis=1).astype(np.float32)
        nonzero_share = (filled != 0.0).mean(axis=1).astype(np.float32)
        abs_mean = np.abs(filled).mean(axis=1).astype(np.float32)
        abs_max = np.abs(filled).max(axis=1).astype(np.float32)
        recency_days = self._compute_recency_days(frame, numeric)

        base[f"{prefix}_active"] = active
        base[f"{prefix}_nonzero_share"] = nonzero_share
        base[f"{prefix}_abs_mean"] = abs_mean
        base[f"{prefix}_abs_max"] = abs_max
        base[f"{prefix}_recency_days"] = recency_days.astype(np.float32)
        bucket_frame = self._recency_bucket_frame(recency_days, prefix)
        base = pd.concat([base, bucket_frame], axis=1)

        if with_novelty:
            base[f"{prefix}_novelty"] = self._compute_novelty(frame, filled)
        return base.astype(np.float32)

    def _empty_runtime_block(self, index: pd.Index, prefix: str, with_novelty: bool = False) -> pd.DataFrame:
        cols = {
            f"{prefix}_active": np.zeros(len(index), dtype=np.float32),
            f"{prefix}_nonzero_share": np.zeros(len(index), dtype=np.float32),
            f"{prefix}_abs_mean": np.zeros(len(index), dtype=np.float32),
            f"{prefix}_abs_max": np.zeros(len(index), dtype=np.float32),
            f"{prefix}_recency_days": np.zeros(len(index), dtype=np.float32),
        }
        for bucket_col in self._bucket_column_names(prefix):
            cols[bucket_col] = np.zeros(len(index), dtype=np.float32)
        if with_novelty:
            cols[f"{prefix}_novelty"] = np.zeros(len(index), dtype=np.float32)
        return pd.DataFrame(cols, index=index)

    def _compute_recency_days(self, frame: pd.DataFrame, source_block: pd.DataFrame) -> pd.Series:
        if "entity_id" not in frame.columns or "crawled_date_day" not in frame.columns:
            return pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)

        work = pd.DataFrame(
            {
                "entity_id": frame["entity_id"].astype(str),
                "crawled_date_day": pd.to_datetime(frame["crawled_date_day"], errors="coerce"),
                "_active": source_block.notna().any(axis=1).to_numpy(dtype=bool),
                "_orig_index": np.arange(len(frame), dtype=np.int64),
            },
            index=frame.index,
        )
        work.sort_values(["entity_id", "crawled_date_day", "_orig_index"], inplace=True, kind="mergesort")
        last_active = work["crawled_date_day"].where(work["_active"]).groupby(work["entity_id"]).ffill()
        recency = (work["crawled_date_day"] - last_active).dt.days.astype("float64")
        recency = recency.where(np.isfinite(recency), 9999.0).clip(lower=0.0, upper=9999.0)
        work["_recency"] = recency
        restored = work.sort_values("_orig_index", kind="mergesort")["_recency"]
        restored.index = frame.index
        return restored.astype(np.float32)

    def _compute_novelty(self, frame: pd.DataFrame, filled_block: pd.DataFrame) -> pd.Series:
        if "entity_id" not in frame.columns or "crawled_date_day" not in frame.columns:
            return pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)

        work = filled_block.copy()
        work["entity_id"] = frame["entity_id"].astype(str).values
        work["crawled_date_day"] = pd.to_datetime(frame["crawled_date_day"], errors="coerce").values
        work["_orig_index"] = np.arange(len(frame), dtype=np.int64)
        value_cols = list(filled_block.columns)
        work.sort_values(["entity_id", "crawled_date_day", "_orig_index"], inplace=True, kind="mergesort")
        prev = work.groupby("entity_id", sort=False)[value_cols].shift(1).fillna(0.0)
        delta = work[value_cols].to_numpy(dtype=np.float32, copy=False) - prev.to_numpy(dtype=np.float32, copy=False)
        novelty = np.sqrt(np.mean(np.square(delta), axis=1)).astype(np.float32, copy=False)
        work["_novelty"] = novelty
        restored = work.sort_values("_orig_index", kind="mergesort")["_novelty"]
        restored.index = frame.index
        return restored.astype(np.float32)

    def _bucket_column_names(self, prefix: str) -> List[str]:
        bucket_cols = []
        lower = -1.0
        for upper in self.cfg.recency_buckets:
            bucket_cols.append(f"{prefix}_recency_{int(lower + 1)}_{int(upper)}")
            lower = float(upper)
        bucket_cols.append(f"{prefix}_recency_gt_{int(self.cfg.recency_buckets[-1])}")
        return bucket_cols

    def _recency_bucket_frame(self, recency: pd.Series, prefix: str) -> pd.DataFrame:
        values = recency.to_numpy(dtype=np.float32, copy=False)
        lower = -1.0
        bucket_data: Dict[str, np.ndarray] = {}
        for upper in self.cfg.recency_buckets:
            bucket_data[f"{prefix}_recency_{int(lower + 1)}_{int(upper)}"] = (
                (values > lower) & (values <= float(upper))
            ).astype(np.float32)
            lower = float(upper)
        bucket_data[f"{prefix}_recency_gt_{int(self.cfg.recency_buckets[-1])}"] = (
            values > float(self.cfg.recency_buckets[-1])
        ).astype(np.float32)
        return pd.DataFrame(bucket_data, index=recency.index)