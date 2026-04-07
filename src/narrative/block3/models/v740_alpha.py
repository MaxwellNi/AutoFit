#!/usr/bin/env python3
"""
V740-alpha prototype.

This file intentionally stays out of the active registry for now.
It is a code skeleton for the single-model synthesis direction:
  - decomposition-first shared trunk
  - multi-resolution mixing
  - lightweight patch/context pathway
  - condition tokens for task/target/horizon/ablation
  - target-specific heads

The goal is to provide a concrete engineering foundation without polluting the
current clean benchmark line before coverage stabilizes.
"""
from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .samformer_model import (
    _detect_binary,
    _detect_nonnegative,
    _sanitize_predictions,
    _select_feature_cols,
)
from .v740_multisource_features import (
    DualClockConfig,
    build_dual_clock_memory_for_entity,
    build_source_native_edgar_memory,
    build_source_native_text_memory,
    infer_edgar_columns,
    infer_text_columns,
)

logger = logging.getLogger(__name__)


_FINANCING_TARGETS = ("funding_raised_usd", "investors_count", "is_funded")


def _build_edgar_memory(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: List[str],
    cfg: DualClockConfig,
) -> Dict[str, np.ndarray]:
    if not source_cols:
        return {
            "recent_tokens": np.zeros((cfg.max_events, 3), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, 4), dtype=np.float32),
        }
    if "edgar_filed_date" in entity_df.columns or "cutoff_ts" in entity_df.columns:
        return build_source_native_edgar_memory(entity_df, prediction_time, source_cols, cfg)
    return build_dual_clock_memory_for_entity(entity_df, prediction_time, source_cols, cfg)


def _build_text_memory(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: List[str],
    cfg: DualClockConfig,
) -> Dict[str, np.ndarray]:
    event_col = "snapshot_ts" if "snapshot_ts" in entity_df.columns else cfg.time_col
    return build_source_native_text_memory(entity_df, prediction_time, source_cols, cfg, event_col=event_col)


def _select_state_feature_cols(
    train_raw: pd.DataFrame,
    target: str,
    max_covariates: int,
    source_exclude: List[str],
) -> List[str]:
    if not source_exclude:
        return _select_feature_cols(train_raw, target, max_covariates)
    state_df = train_raw.drop(columns=[c for c in source_exclude if c in train_raw.columns], errors="ignore")
    return _select_feature_cols(state_df, target, max_covariates)


@dataclass
class _V740Windows:
    train_x: List[np.ndarray]
    train_y: List[np.ndarray]
    train_funding_aux: List[np.ndarray]
    train_investors_aux: List[np.ndarray]
    train_binary_aux: List[np.ndarray]
    val_x: List[np.ndarray]
    val_y: List[np.ndarray]
    val_funding_aux: List[np.ndarray]
    val_investors_aux: List[np.ndarray]
    val_binary_aux: List[np.ndarray]
    train_edgar_recent: List[np.ndarray]
    train_edgar_bucket: List[np.ndarray]
    train_text_recent: List[np.ndarray]
    train_text_bucket: List[np.ndarray]
    val_edgar_recent: List[np.ndarray]
    val_edgar_bucket: List[np.ndarray]
    val_text_recent: List[np.ndarray]
    val_text_bucket: List[np.ndarray]
    contexts: Dict[str, np.ndarray]
    context_memory: Dict[str, Dict[str, np.ndarray]]
    window_stats: Dict[str, object]


def _left_pad_last_dim(arr: np.ndarray, target_width: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-1] >= target_width:
        return arr[..., -target_width:].astype(np.float32, copy=False)
    pad = target_width - arr.shape[-1]
    if arr.shape[-1] <= 0:
        out_shape = list(arr.shape)
        out_shape[-1] = target_width
        return np.zeros(tuple(out_shape), dtype=np.float32)
    pad_block = np.repeat(arr[..., :1], pad, axis=-1)
    return np.concatenate([pad_block, arr], axis=-1).astype(np.float32, copy=False)


def _window_teacher_features(
    windows: List[np.ndarray],
    edgar_recent: List[np.ndarray],
    edgar_bucket: List[np.ndarray],
    text_recent: List[np.ndarray],
    text_bucket: List[np.ndarray],
) -> np.ndarray:
    feats: List[np.ndarray] = []
    for i, x_win in enumerate(windows):
        target_hist = x_win[0].astype(np.float32, copy=False)
        diffs = np.diff(target_hist) if len(target_hist) > 1 else np.zeros((1,), dtype=np.float32)
        row = [
            float(target_hist[-1]),
            float(np.mean(target_hist)),
            float(np.std(target_hist)),
            float(np.min(target_hist)),
            float(np.max(target_hist)),
            float(target_hist[-1] - target_hist[0]),
            float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
            float(np.mean(target_hist > 0.0)),
        ]
        if x_win.shape[0] > 1:
            cov_block = x_win[1:].astype(np.float32, copy=False)
            row.extend([
                float(np.mean(cov_block)),
                float(np.std(cov_block)),
                float(np.max(cov_block)),
            ])
        else:
            row.extend([0.0, 0.0, 0.0])

        er = edgar_recent[i] if i < len(edgar_recent) else None
        eb = edgar_bucket[i] if i < len(edgar_bucket) else None
        tr = text_recent[i] if i < len(text_recent) else None
        tb = text_bucket[i] if i < len(text_bucket) else None
        for recent, bucket in ((er, eb), (tr, tb)):
            if recent is not None and recent.size:
                active_mask = recent[:, 2] > 0.0
                row.extend([
                    float(np.sum(active_mask)),
                    float(np.mean(recent[:, 0][active_mask])) if np.any(active_mask) else 0.0,
                    float(np.mean(np.abs(recent[:, 3][active_mask]))) if np.any(active_mask) and recent.shape[1] > 3 else 0.0,
                    float(np.mean(np.abs(recent[:, 4][active_mask]))) if np.any(active_mask) and recent.shape[1] > 4 else 0.0,
                ])
            else:
                row.extend([0.0, 0.0, 0.0, 0.0])
            if bucket is not None and bucket.size:
                row.extend([
                    float(np.sum(bucket[:, 0])),
                    float(np.max(bucket[:, 0])),
                    float(np.mean(bucket[:, 0])),
                    float(np.mean(np.abs(bucket[:, 4]))) if bucket.shape[1] > 4 else 0.0,
                    float(np.mean(np.abs(bucket[:, 5]))) if bucket.shape[1] > 5 else 0.0,
                ])
            else:
                row.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        feats.append(np.asarray(row, dtype=np.float32))
    if not feats:
        return np.zeros((0, 1), dtype=np.float32)
    arr = np.vstack(feats).astype(np.float32, copy=False)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _build_v740_windows(
    train_raw: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    input_size: int,
    horizon: int,
    step: int,
    max_entities: int,
    max_windows: int,
    val_frac: float,
    seed: int,
    edgar_cols: List[str],
    text_cols: List[str],
    dual_clock_cfg: DualClockConfig,
    aux_target_raw: Optional[pd.DataFrame] = None,
    enable_window_repair: bool = False,
    min_history_points: int = 8,
    target_windows_per_entity: int = 6,
) -> _V740Windows:
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    train_funding_aux: List[np.ndarray] = []
    train_investors_aux: List[np.ndarray] = []
    train_binary_aux: List[np.ndarray] = []
    val_x: List[np.ndarray] = []
    val_y: List[np.ndarray] = []
    val_funding_aux: List[np.ndarray] = []
    val_investors_aux: List[np.ndarray] = []
    val_binary_aux: List[np.ndarray] = []
    train_edgar_recent: List[np.ndarray] = []
    train_edgar_bucket: List[np.ndarray] = []
    train_text_recent: List[np.ndarray] = []
    train_text_bucket: List[np.ndarray] = []
    val_edgar_recent: List[np.ndarray] = []
    val_edgar_bucket: List[np.ndarray] = []
    val_text_recent: List[np.ndarray] = []
    val_text_bucket: List[np.ndarray] = []
    contexts: Dict[str, np.ndarray] = {}
    context_memory: Dict[str, Dict[str, np.ndarray]] = {}
    entities_considered = 0
    entities_with_windows = 0
    short_context_entities = 0
    step_reduced_entities = 0
    repaired_entities = 0
    effective_contexts: List[int] = []
    effective_steps: List[int] = []
    min_history_points = max(2, int(min_history_points))
    target_windows_per_entity = max(1, int(target_windows_per_entity))

    if train_raw is None or "entity_id" not in train_raw.columns:
        return _V740Windows(
            train_x, train_y, train_funding_aux, train_investors_aux, train_binary_aux,
            val_x, val_y, val_funding_aux, val_investors_aux, val_binary_aux,
            train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket,
            val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket,
            contexts, context_memory,
            {
                "repair_enabled": bool(enable_window_repair),
                "requested_input_size": int(input_size),
                "requested_step": int(step),
                "min_history_points": int(min_history_points),
                "target_windows_per_entity": int(target_windows_per_entity),
                "entities_considered": 0,
                "entities_with_windows": 0,
                "short_context_entities": 0,
                "step_reduced_entities": 0,
                "repaired_entities": 0,
                "train_windows": 0,
                "val_windows": 0,
            },
        )

    rng = np.random.RandomState(seed)
    groups = train_raw.groupby("entity_id", sort=False)
    aux_source = aux_target_raw if aux_target_raw is not None else train_raw
    aux_groups = None
    if aux_source is not None and "entity_id" in aux_source.columns:
        aux_groups = aux_source.groupby("entity_id", sort=False)
    for i, (eid, grp) in enumerate(groups):
        if i >= max_entities:
            break
        grp = grp.sort_values("crawled_date_day").reset_index(drop=True)
        if target not in grp.columns:
            continue
        aux_grp = grp
        if aux_groups is not None:
            try:
                aux_grp = aux_groups.get_group(eid).sort_values("crawled_date_day").reset_index(drop=True)
            except KeyError:
                aux_grp = grp
        if len(aux_grp) != len(grp):
            aux_grp = grp
        entities_considered += 1

        y_arr = pd.Series(grp[target], dtype="float64").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

        def _aux_array(frame: pd.DataFrame, col: str) -> np.ndarray:
            if col not in frame.columns:
                return np.zeros((len(frame),), dtype=np.float32)
            return pd.Series(frame[col], dtype="float64").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

        funding_arr = _aux_array(aux_grp, "funding_raised_usd")
        investors_arr = _aux_array(aux_grp, "investors_count")
        binary_arr = _aux_array(aux_grp, "is_funded")
        times = pd.to_datetime(grp["crawled_date_day"], errors="coerce")
        if len(y_arr) <= horizon or times.isna().all():
            continue
        available_history = len(y_arr) - horizon
        if available_history <= 0:
            continue
        if enable_window_repair:
            effective_input_size = min(int(input_size), int(available_history))
            if effective_input_size < min_history_points:
                continue
        else:
            if len(y_arr) < input_size + horizon:
                continue
            effective_input_size = int(input_size)

        channels = [y_arr]
        for col in feature_cols:
            if col in grp.columns:
                vals = pd.Series(grp[col], dtype="float64").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
                channels.append(vals)
        series = np.stack(channels, axis=0)
        context_width = min(int(input_size), series.shape[1])
        contexts[str(eid)] = _left_pad_last_dim(series[:, -context_width:], int(input_size))

        last_time = times.iloc[-1]
        edgar_context_mem = _build_edgar_memory(grp, last_time, edgar_cols, dual_clock_cfg)
        text_context_mem = _build_text_memory(grp, last_time, text_cols, dual_clock_cfg)
        context_memory[str(eid)] = {
            "edgar_recent": edgar_context_mem["recent_tokens"],
            "edgar_bucket": edgar_context_mem["bucket_tokens"],
            "text_recent": text_context_mem["recent_tokens"],
            "text_bucket": text_context_mem["bucket_tokens"],
        }

        entity_x: List[np.ndarray] = []
        entity_y: List[np.ndarray] = []
        entity_funding_aux: List[np.ndarray] = []
        entity_investors_aux: List[np.ndarray] = []
        entity_binary_aux: List[np.ndarray] = []
        entity_edgar_recent: List[np.ndarray] = []
        entity_edgar_bucket: List[np.ndarray] = []
        entity_text_recent: List[np.ndarray] = []
        entity_text_bucket: List[np.ndarray] = []
        limit = len(y_arr) - effective_input_size - horizon + 1
        if limit <= 0:
            continue
        entity_step = int(step)
        if enable_window_repair:
            desired_windows = min(target_windows_per_entity, limit)
            if desired_windows > 0:
                entity_step = max(1, min(int(step), int(math.ceil(limit / desired_windows))))
        for t in range(0, limit, entity_step):
            x_slice = series[:, t : t + effective_input_size]
            x_win = _left_pad_last_dim(x_slice, int(input_size))
            y_win = y_arr[t + effective_input_size : t + effective_input_size + horizon]
            funding_win = funding_arr[t + effective_input_size : t + effective_input_size + horizon]
            investors_win = investors_arr[t + effective_input_size : t + effective_input_size + horizon]
            binary_win = binary_arr[t + effective_input_size : t + effective_input_size + horizon]
            end_idx = t + effective_input_size - 1
            pred_time = times.iloc[end_idx]
            if pd.isna(pred_time):
                continue
            if (
                np.any(~np.isfinite(x_win))
                or np.any(~np.isfinite(y_win))
                or np.any(~np.isfinite(funding_win))
                or np.any(~np.isfinite(investors_win))
                or np.any(~np.isfinite(binary_win))
            ):
                continue

            prefix = grp.iloc[: end_idx + 1]
            if edgar_cols:
                edgar_mem = _build_edgar_memory(prefix, pred_time, edgar_cols, dual_clock_cfg)
                edgar_recent = edgar_mem["recent_tokens"]
                edgar_bucket = edgar_mem["bucket_tokens"]
            else:
                edgar_recent = np.zeros((dual_clock_cfg.max_events, 3), dtype=np.float32)
                edgar_bucket = np.zeros((len(dual_clock_cfg.recency_buckets) + 1, 4), dtype=np.float32)
            if text_cols:
                text_mem = _build_text_memory(prefix, pred_time, text_cols, dual_clock_cfg)
                text_recent = text_mem["recent_tokens"]
                text_bucket = text_mem["bucket_tokens"]
            else:
                text_recent = np.zeros((dual_clock_cfg.max_events, 5), dtype=np.float32)
                text_bucket = np.zeros((len(dual_clock_cfg.recency_buckets) + 1, 6), dtype=np.float32)

            entity_x.append(x_win.astype(np.float32, copy=False))
            entity_y.append(y_win.astype(np.float32, copy=False))
            entity_funding_aux.append(funding_win.astype(np.float32, copy=False))
            entity_investors_aux.append(investors_win.astype(np.float32, copy=False))
            entity_binary_aux.append(binary_win.astype(np.float32, copy=False))
            entity_edgar_recent.append(edgar_recent.astype(np.float32, copy=False))
            entity_edgar_bucket.append(edgar_bucket.astype(np.float32, copy=False))
            entity_text_recent.append(text_recent.astype(np.float32, copy=False))
            entity_text_bucket.append(text_bucket.astype(np.float32, copy=False))

        if not entity_x:
            continue
        entities_with_windows += 1
        effective_contexts.append(int(effective_input_size))
        effective_steps.append(int(entity_step))
        if effective_input_size < int(input_size):
            short_context_entities += 1
            repaired_entities += 1
        elif entity_step < int(step):
            repaired_entities += 1
        if entity_step < int(step):
            step_reduced_entities += 1

        n_val = 0
        if len(entity_x) >= 4:
            n_val = max(1, int(round(len(entity_x) * val_frac)))
            n_val = min(n_val, len(entity_x) - 1)
        split = len(entity_x) - n_val
        train_x.extend(entity_x[:split])
        train_y.extend(entity_y[:split])
        train_funding_aux.extend(entity_funding_aux[:split])
        train_investors_aux.extend(entity_investors_aux[:split])
        train_binary_aux.extend(entity_binary_aux[:split])
        train_edgar_recent.extend(entity_edgar_recent[:split])
        train_edgar_bucket.extend(entity_edgar_bucket[:split])
        train_text_recent.extend(entity_text_recent[:split])
        train_text_bucket.extend(entity_text_bucket[:split])
        val_x.extend(entity_x[split:])
        val_y.extend(entity_y[split:])
        val_funding_aux.extend(entity_funding_aux[split:])
        val_investors_aux.extend(entity_investors_aux[split:])
        val_binary_aux.extend(entity_binary_aux[split:])
        val_edgar_recent.extend(entity_edgar_recent[split:])
        val_edgar_bucket.extend(entity_edgar_bucket[split:])
        val_text_recent.extend(entity_text_recent[split:])
        val_text_bucket.extend(entity_text_bucket[split:])

    def _cap_indices(n: int, cap: int) -> np.ndarray:
        if n <= cap:
            return np.arange(n)
        idx = rng.choice(n, size=cap, replace=False)
        idx.sort()
        return idx

    def _apply_cap(idx: np.ndarray, *lists: List[np.ndarray]) -> List[List[np.ndarray]]:
        out = []
        for xs in lists:
            out.append([xs[j] for j in idx])
        return out

    train_idx = _cap_indices(len(train_x), max_windows)
    train_x, train_y, train_funding_aux, train_investors_aux, train_binary_aux, train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket = _apply_cap(
        train_idx,
        train_x,
        train_y,
        train_funding_aux,
        train_investors_aux,
        train_binary_aux,
        train_edgar_recent,
        train_edgar_bucket,
        train_text_recent,
        train_text_bucket,
    )
    if val_x:
        val_cap = max(2048, max_windows // 5)
        val_idx = _cap_indices(len(val_x), val_cap)
        val_x, val_y, val_funding_aux, val_investors_aux, val_binary_aux, val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket = _apply_cap(
            val_idx,
            val_x,
            val_y,
            val_funding_aux,
            val_investors_aux,
            val_binary_aux,
            val_edgar_recent,
            val_edgar_bucket,
            val_text_recent,
            val_text_bucket,
        )

    window_stats: Dict[str, object] = {
        "repair_enabled": bool(enable_window_repair),
        "requested_input_size": int(input_size),
        "requested_step": int(step),
        "min_history_points": int(min_history_points),
        "target_windows_per_entity": int(target_windows_per_entity),
        "entities_considered": int(entities_considered),
        "entities_with_windows": int(entities_with_windows),
        "short_context_entities": int(short_context_entities),
        "step_reduced_entities": int(step_reduced_entities),
        "repaired_entities": int(repaired_entities),
        "mean_effective_input_size": float(np.mean(effective_contexts)) if effective_contexts else 0.0,
        "min_effective_input_size": int(min(effective_contexts)) if effective_contexts else 0,
        "mean_effective_step": float(np.mean(effective_steps)) if effective_steps else float(step),
        "min_effective_step": int(min(effective_steps)) if effective_steps else int(step),
        "train_windows": int(len(train_x)),
        "val_windows": int(len(val_x)),
    }

    return _V740Windows(
        train_x=train_x,
        train_y=train_y,
        train_funding_aux=train_funding_aux,
        train_investors_aux=train_investors_aux,
        train_binary_aux=train_binary_aux,
        val_x=val_x,
        val_y=val_y,
        val_funding_aux=val_funding_aux,
        val_investors_aux=val_investors_aux,
        val_binary_aux=val_binary_aux,
        train_edgar_recent=train_edgar_recent,
        train_edgar_bucket=train_edgar_bucket,
        train_text_recent=train_text_recent,
        train_text_bucket=train_text_bucket,
        val_edgar_recent=val_edgar_recent,
        val_edgar_bucket=val_edgar_bucket,
        val_text_recent=val_text_recent,
        val_text_bucket=val_text_bucket,
        contexts=contexts,
        context_memory=context_memory,
        window_stats=window_stats,
    )


def _torch_imports():
    import torch
    from torch import nn

    class ConditionEncoder(nn.Module):
        def __init__(self, emb_dim: int = 16, seq_len: int = 60, max_horizon: int = 365):
            super().__init__()
            self.task_emb = nn.Embedding(4, emb_dim)
            self.target_emb = nn.Embedding(4, emb_dim)
            self.ablation_emb = nn.Embedding(8, emb_dim)
            self.seq_len = max(1, int(seq_len))
            self.max_horizon = max(365, int(max_horizon))
            self.register_buffer(
                "horizon_bounds",
                torch.tensor([1.0, 7.0, 14.0, 30.0, 45.0, 60.0, 90.0, 180.0, 365.0], dtype=torch.float32),
                persistent=False,
            )
            self.horizon_bucket_emb = nn.Embedding(len(self.horizon_bounds) + 1, emb_dim)
            self.horizon_proj = nn.Sequential(
                nn.Linear(5, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.proj = nn.Sequential(
                nn.Linear(emb_dim * 4, emb_dim * 2),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim),
            )

        def forward(
            self,
            task_idx: torch.Tensor,
            target_idx: torch.Tensor,
            horizon_value: torch.Tensor,
            ablation_idx: torch.Tensor,
        ) -> torch.Tensor:
            horizon_value = horizon_value.float().clamp_min(1.0)
            bucket_idx = torch.bucketize(horizon_value, self.horizon_bounds.to(horizon_value.device))
            horizon_scaled = (horizon_value / float(self.max_horizon)).unsqueeze(-1)
            horizon_log = (torch.log1p(horizon_value) / math.log1p(float(self.max_horizon))).unsqueeze(-1)
            horizon_sqrt = (torch.sqrt(horizon_value) / math.sqrt(float(self.max_horizon))).unsqueeze(-1)
            ratio_to_context = (horizon_value / float(self.seq_len)).clamp(0.0, 4.0).div(4.0).unsqueeze(-1)
            context_to_h = (float(self.seq_len) / horizon_value).clamp(0.0, 4.0).div(4.0).unsqueeze(-1)
            horizon_feat = self.horizon_proj(
                torch.cat(
                    [horizon_scaled, horizon_log, horizon_sqrt, ratio_to_context, context_to_h],
                    dim=-1,
                )
            ) + self.horizon_bucket_emb(bucket_idx)
            x = torch.cat([
                self.task_emb(task_idx),
                self.target_emb(target_idx),
                horizon_feat,
                self.ablation_emb(ablation_idx),
            ], dim=-1)
            return self.proj(x)

    class MovingAverageDecomposition(nn.Module):
        def __init__(self, kernel_size: int = 5):
            super().__init__()
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        def forward(self, x: torch.Tensor):
            trend = self.avg(x)
            seasonal = x - trend
            return trend, seasonal

    class MultiResolutionBlock(nn.Module):
        def __init__(self, channels: int, hidden_dim: int):
            super().__init__()
            self.scales = nn.ModuleList([
                nn.Conv1d(channels, hidden_dim, kernel_size=1),
                nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1),
                nn.Conv1d(channels, hidden_dim, kernel_size=5, padding=2),
            ])
            self.out = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = [layer(x) for layer in self.scales]
            return self.out(torch.cat(feats, dim=1))

    class PatchContextMixer(nn.Module):
        def __init__(self, hidden_dim: int, patch_size: int = 8):
            super().__init__()
            self.patch_size = patch_size
            self.depthwise = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=patch_size,
                stride=max(1, patch_size // 2),
                groups=hidden_dim,
                padding=patch_size // 2,
            )
            self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mixed = self.pointwise(self.depthwise(x))
            gate = self.gate(mixed)
            up = nn.functional.interpolate(mixed, size=x.shape[-1], mode="linear", align_corners=False)
            return x + up * gate

    class CASALocalContextBlock(nn.Module):
        """Lightweight CASA-inspired local context attention.

        This is intentionally not a paper-faithful CASA reimplementation.
        The goal is to add a cheap local-context gating path that conditions
        local sequence mixing on the current task/horizon regime.
        """

        def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int = 7):
            super().__init__()
            self.query = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.value = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.local = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=max(1, hidden_dim // 8),
            )
            self.cond_gate = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            q = self.query(x)
            local = self.local(x)
            gate = self.cond_gate(cond).unsqueeze(-1)
            score = torch.sigmoid((q * local) / math.sqrt(max(1, x.shape[1])))
            mixed = self.value(x) * score * gate
            return x + self.out(mixed)

    class CompactTemporalMemory(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, L] -> [B, L, C]
            seq = x.transpose(1, 2)
            _, h = self.gru(seq)
            return self.proj(h[-1])

    class ValueBucketEncoder(nn.Module):
        def __init__(self, num_bins: int = 32, hidden_dim: int = 64):
            super().__init__()
            self.num_bins = num_bins
            self.emb = nn.Embedding(num_bins + 1, hidden_dim)
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, target_hist: torch.Tensor) -> torch.Tensor:
            # target_hist: [B, L]
            mean = target_hist.mean(dim=-1, keepdim=True)
            std = target_hist.std(dim=-1, keepdim=True).clamp_min(1e-5)
            z = ((target_hist - mean) / std).clamp(-4.0, 4.0)
            boundaries = torch.linspace(-4.0, 4.0, self.num_bins, device=target_hist.device)
            bucket_ids = torch.bucketize(z, boundaries)
            emb = self.emb(bucket_ids)
            pooled = emb.mean(dim=1)
            return self.proj(pooled)

    class StaticSummaryProjector(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, static_vec: torch.Tensor) -> torch.Tensor:
            return self.net(static_vec)

    class StaticDynamicTimeFusion(nn.Module):
        """TimeEmb-inspired static/dynamic/source fusion gate."""

        def __init__(self, hidden_dim: int, cond_dim: int):
            super().__init__()
            self.dynamic_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.source_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.cond_gate = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.residual = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(
            self,
            pooled: torch.Tensor,
            memory_feat: torch.Tensor,
            value_feat: torch.Tensor,
            static_feat: torch.Tensor,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            cond: torch.Tensor,
        ) -> torch.Tensor:
            dynamic_feat = self.dynamic_proj(torch.cat([pooled, memory_feat, value_feat], dim=-1))
            source_feat = self.source_proj(torch.cat([static_feat, edgar_feat, text_feat], dim=-1))
            gate = self.cond_gate(cond)
            fused = dynamic_feat + gate * source_feat
            return fused + 0.1 * self.residual(torch.cat([dynamic_feat, source_feat], dim=-1))

    class EventMemoryEncoder(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.recent_proj = nn.LazyLinear(hidden_dim)
            self.bucket_proj = nn.LazyLinear(hidden_dim)
            self.out = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(
            self,
            recent_tokens: Optional[torch.Tensor],
            bucket_tokens: Optional[torch.Tensor],
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor:
            if recent_tokens is None and bucket_tokens is None:
                return torch.zeros((batch_size, self.out[-1].out_features), device=device)

            if recent_tokens is None:
                recent_feat = torch.zeros((batch_size, self.out[-1].out_features), device=device)
            else:
                recent_flat = recent_tokens.reshape(recent_tokens.shape[0], recent_tokens.shape[1], -1)
                recent_feat = self.recent_proj(recent_flat).mean(dim=1)

            if bucket_tokens is None:
                bucket_feat = torch.zeros((batch_size, self.out[-1].out_features), device=device)
            else:
                bucket_flat = bucket_tokens.reshape(bucket_tokens.shape[0], bucket_tokens.shape[1], -1)
                bucket_feat = self.bucket_proj(bucket_flat).mean(dim=1)

            return self.out(torch.cat([recent_feat, bucket_feat], dim=-1))

    class BinaryHistoryKernel(nn.Module):
        """DeepNPTS-style convex-combination shortcut for binary targets."""

        def __init__(self, seq_len: int, hidden_dim: int, cond_dim: int, horizon: int):
            super().__init__()
            self.seq_len = seq_len
            self.horizon = horizon
            self.summary_dim = 7
            self.history_proj = nn.Sequential(
                nn.Linear(seq_len * 2 + self.summary_dim + cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.weight_head = nn.Linear(hidden_dim, seq_len * horizon)
            self.prior_head = nn.Sequential(
                nn.Linear(hidden_dim + self.summary_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.mix_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3 + cond_dim + self.summary_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )

        def forward(
            self,
            target_hist: torch.Tensor,
            pooled: torch.Tensor,
            memory_feat: torch.Tensor,
            value_feat: torch.Tensor,
            cond: torch.Tensor,
            event_logits: torch.Tensor,
            base_logits: torch.Tensor,
        ) -> torch.Tensor:
            B, L = target_hist.shape
            if L != self.seq_len:
                raise ValueError(f"BinaryHistoryKernel expected seq_len={self.seq_len}, got {L}")

            hist = target_hist.float().clamp(0.0, 1.0)
            delta = torch.cat([hist[:, :1] * 0.0, hist[:, 1:] - hist[:, :-1]], dim=1)

            def _recent_mean(width: int) -> torch.Tensor:
                width = min(width, L)
                return hist[:, -width:].mean(dim=1, keepdim=True)

            summary = torch.cat(
                [
                    hist[:, -1:].detach(),
                    hist.mean(dim=1, keepdim=True),
                    _recent_mean(3),
                    _recent_mean(7),
                    _recent_mean(14),
                    delta.abs().mean(dim=1, keepdim=True),
                    hist.std(dim=1, keepdim=True, unbiased=False),
                ],
                dim=-1,
            )
            history_ctx = self.history_proj(torch.cat([hist, delta, summary, cond], dim=-1))
            weight_logits = self.weight_head(history_ctx).reshape(B, L, self.horizon)
            recency_bias = torch.linspace(-0.15, 0.15, L, device=hist.device, dtype=hist.dtype).view(1, L, 1)
            weights = torch.softmax(weight_logits + recency_bias, dim=1)
            kernel_probs = (weights * hist.unsqueeze(-1)).sum(dim=1)

            event_prob = torch.sigmoid(event_logits).unsqueeze(-1)
            prior_probs = torch.sigmoid(self.prior_head(torch.cat([history_ctx, summary, event_prob], dim=-1)))
            base_probs = torch.sigmoid(base_logits)
            gate = self.mix_gate(
                torch.cat([pooled, memory_feat, value_feat, cond, summary, event_prob], dim=-1)
            )
            shortcut_probs = 0.7 * kernel_probs + 0.3 * prior_probs
            final_probs = (1.0 - gate) * base_probs + gate * shortcut_probs
            return final_probs.clamp(1e-4, 1.0 - 1e-4)

    class InvariantVariantFusion(nn.Module):
        def __init__(self, hidden_dim: int, cond_dim: int):
            super().__init__()
            self.invariant_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.variant_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.cond_gate = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            inv = self.invariant_proj(x)
            var = self.variant_proj(x)
            gate = self.cond_gate(cond).unsqueeze(-1)
            return inv + gate * var

    class TaskSpecificModulator(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.task_emb = nn.Embedding(4, hidden_dim)
            self.scale = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.bias = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.residual = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, pooled: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
            task_vec = self.task_emb(task_idx)
            scale = 0.2 * self.scale(task_vec)
            bias = 0.1 * self.bias(task_vec)
            mod = pooled * (1.0 + scale) + bias
            return mod + 0.1 * self.residual(torch.cat([pooled, task_vec], dim=-1))

    class TargetRoutedDecoder(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            num_targets: int = 4,
            num_experts: int = 3,
        ):
            super().__init__()
            self.summary_dim = 6
            self.num_experts = max(1, int(num_experts))
            route_in_dim = hidden_dim * 5 + cond_dim + self.summary_dim
            self.route_ctx = nn.Sequential(
                nn.Linear(route_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.route_logits = nn.Sequential(
                nn.Linear(route_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_experts),
            )
            self.target_route_bias = nn.Embedding(num_targets, self.num_experts)
            self.target_residual = nn.Embedding(num_targets, hidden_dim)
            self.shared_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim + self.summary_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_experts)
            ])
            self.mix_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + cond_dim + self.summary_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def _history_summary(self, target_hist: torch.Tensor) -> torch.Tensor:
            recent_width = min(7, target_hist.shape[1])
            long_width = min(21, target_hist.shape[1])
            recent = target_hist[:, -recent_width:]
            long = target_hist[:, -long_width:]
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            return torch.cat(
                [
                    target_hist[:, -1:],
                    recent.mean(dim=1, keepdim=True),
                    long.mean(dim=1, keepdim=True),
                    recent.std(dim=1, keepdim=True, unbiased=False),
                    step.mean(dim=1, keepdim=True),
                    (recent.abs() > 1e-6).float().mean(dim=1, keepdim=True),
                ],
                dim=-1,
            )

        def forward(
            self,
            pooled: torch.Tensor,
            memory_feat: torch.Tensor,
            value_feat: torch.Tensor,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            target_idx: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hist_summary = self._history_summary(target_hist)
            route_in = torch.cat(
                [pooled, memory_feat, value_feat, edgar_feat, text_feat, cond, hist_summary],
                dim=-1,
            )
            route_ctx = self.route_ctx(route_in)
            route_logits = self.route_logits(route_in) + self.target_route_bias(target_idx)
            route_weights = torch.softmax(route_logits, dim=-1)
            expert_in = torch.cat([route_ctx, hist_summary], dim=-1)
            expert_outputs = torch.stack([expert(expert_in) for expert in self.experts], dim=1)
            routed = torch.sum(expert_outputs * route_weights.unsqueeze(-1), dim=1)
            shared = self.shared_proj(pooled)
            gate = self.mix_gate(torch.cat([shared, routed, cond, hist_summary], dim=-1))
            decoded = shared + gate * routed + 0.05 * self.target_residual(target_idx)
            return self.out(torch.cat([pooled, decoded, route_ctx], dim=-1)), route_weights, gate

    class CountStructureHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            horizon: int,
            anchor_strength: float = 0.70,
            jump_strength: float = 0.30,
        ):
            super().__init__()
            self.horizon = horizon
            self.anchor_strength = float(max(0.0, anchor_strength))
            self.jump_strength = float(max(0.0, jump_strength))
            anchor_ctx_dim = hidden_dim + cond_dim + 7
            self.level_refine = nn.Sequential(
                nn.Linear(anchor_ctx_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.growth_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.growth_gate = nn.Sequential(
                nn.Linear(anchor_ctx_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )
            self.jump_basis = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.jump_gate = nn.Sequential(
                nn.Linear(anchor_ctx_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )
            self.jump_scale = nn.Sequential(
                nn.Linear(anchor_ctx_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )

        def forward(self, hidden: torch.Tensor, cond: torch.Tensor, target_hist: torch.Tensor) -> torch.Tensor:
            recent_width = min(5, target_hist.shape[1])
            recent = target_hist[:, -recent_width:]
            anchor_last = target_hist[:, -1:]
            anchor_mean = recent.mean(dim=1, keepdim=True)
            anchor_std = recent.std(dim=1, keepdim=True, unbiased=False)
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            jump_last = step[:, -1:] if step.shape[1] > 0 else anchor_last * 0.0
            jump_mean = step.mean(dim=1, keepdim=True) if step.shape[1] > 0 else anchor_last * 0.0
            jump_energy = step.abs().mean(dim=1, keepdim=True) if step.shape[1] > 0 else anchor_last * 0.0
            jump_peak = step.abs().amax(dim=1, keepdim=True) if step.shape[1] > 0 else anchor_last * 0.0
            anchor_ctx = torch.cat(
                [
                    hidden,
                    cond,
                    anchor_last,
                    anchor_mean,
                    anchor_std,
                    jump_last,
                    jump_mean,
                    jump_energy,
                    jump_peak,
                ],
                dim=-1,
            )
            anchor_level = 0.75 * anchor_last + 0.25 * anchor_mean
            anchor_path = anchor_level.expand(-1, self.horizon) + 0.10 * self.level_refine(anchor_ctx)
            growth = torch.cumsum(torch.nn.functional.softplus(self.growth_head(hidden)), dim=-1)
            growth_gate = self.growth_gate(anchor_ctx)
            jump_entry = 0.60 * jump_last + 0.30 * jump_mean + 0.10 * (anchor_last - anchor_mean)
            jump_profile = self.jump_basis(hidden)
            jump_gate = self.jump_gate(anchor_ctx)
            jump_scale = self.jump_scale(anchor_ctx) * torch.log1p(jump_peak + jump_energy + anchor_std)
            jump_path = jump_gate * (jump_profile + jump_entry.expand(-1, self.horizon))
            residual = 0.15 * self.residual_head(hidden)
            return (
                anchor_path
                + self.anchor_strength * growth_gate * growth
                + self.jump_strength * jump_scale * jump_path
                + residual
            )

    class CountSourceRoutedDecoder(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            num_ablations: int,
            num_experts: int = 3,
            route_floor: float = 0.10,
        ):
            super().__init__()
            self.num_experts = max(1, int(num_experts))
            self.route_floor = float(min(0.45, max(0.0, route_floor)))
            self.history_dim = 6
            self.source_dim = 6
            self.ablation_dim = max(8, hidden_dim // 4)
            self.ablation_embed = nn.Embedding(num_ablations, self.ablation_dim)
            self.ablation_bias = nn.Embedding(num_ablations, self.num_experts)

            route_in_dim = hidden_dim * 6 + cond_dim + self.history_dim + self.source_dim + self.ablation_dim
            expert_in_dim = hidden_dim * 2 + self.history_dim + self.source_dim + self.ablation_dim
            self.shared_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.route_ctx = nn.Sequential(
                nn.Linear(route_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.route_logits = nn.Sequential(
                nn.Linear(route_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_experts),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.num_experts)
            ])
            self.mix_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2 + cond_dim + self.history_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def _history_summary(self, target_hist: torch.Tensor) -> torch.Tensor:
            recent_width = min(7, target_hist.shape[1])
            recent = torch.relu(target_hist[:, -recent_width:])
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            jump_peak = step.abs().amax(dim=1, keepdim=True) if step.shape[1] > 0 else recent[:, -1:] * 0.0
            jump_mean = step.abs().mean(dim=1, keepdim=True) if step.shape[1] > 0 else recent[:, -1:] * 0.0
            return torch.cat(
                [
                    torch.log1p(recent[:, -1:]),
                    torch.log1p(recent.mean(dim=1, keepdim=True)),
                    torch.log1p(recent.amax(dim=1, keepdim=True)),
                    (recent > 0.5).float().mean(dim=1, keepdim=True),
                    torch.log1p(jump_mean),
                    torch.log1p(jump_peak),
                ],
                dim=-1,
            )

        def _source_summary(
            self,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            edgar_active_flag: Optional[torch.Tensor] = None,
            text_active_flag: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            edgar_energy = torch.sqrt(edgar_feat.pow(2).mean(dim=1, keepdim=True).clamp_min(0.0))
            text_energy = torch.sqrt(text_feat.pow(2).mean(dim=1, keepdim=True).clamp_min(0.0))
            if edgar_active_flag is None:
                edgar_active = (edgar_energy > 1e-6).to(edgar_feat.dtype)
            else:
                edgar_active = edgar_active_flag.to(device=edgar_feat.device, dtype=edgar_feat.dtype)
            if text_active_flag is None:
                text_active = (text_energy > 1e-6).to(text_feat.dtype)
            else:
                text_active = text_active_flag.to(device=text_feat.device, dtype=text_feat.dtype)
            edgar_energy = edgar_energy * edgar_active
            text_energy = text_energy * text_active
            both_active = edgar_active * text_active
            source_gap = torch.log1p(edgar_energy) - torch.log1p(text_energy)
            return torch.cat(
                [
                    torch.log1p(edgar_energy),
                    torch.log1p(text_energy),
                    edgar_active,
                    text_active,
                    both_active,
                    source_gap,
                ],
                dim=-1,
            )

        def forward(
            self,
            count_feat: torch.Tensor,
            decoded: torch.Tensor,
            memory_feat: torch.Tensor,
            value_feat: torch.Tensor,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            ablation_idx: torch.Tensor,
            edgar_active_flag: Optional[torch.Tensor] = None,
            text_active_flag: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hist_summary = self._history_summary(target_hist)
            source_summary = self._source_summary(
                edgar_feat,
                text_feat,
                edgar_active_flag=edgar_active_flag,
                text_active_flag=text_active_flag,
            )
            ablation_feat = self.ablation_embed(ablation_idx)
            route_in = torch.cat(
                [
                    count_feat,
                    decoded,
                    memory_feat,
                    value_feat,
                    edgar_feat,
                    text_feat,
                    cond,
                    hist_summary,
                    source_summary,
                    ablation_feat,
                ],
                dim=-1,
            )
            shared = self.shared_proj(torch.cat([count_feat, decoded], dim=-1))
            route_ctx = self.route_ctx(route_in)
            route_logits = self.route_logits(route_in) + self.ablation_bias(ablation_idx)
            route_weights = torch.softmax(route_logits, dim=-1)
            if self.route_floor > 0.0 and self.num_experts > 1:
                route_weights = (1.0 - self.route_floor) * route_weights + (self.route_floor / self.num_experts)
            expert_in = torch.cat([shared, route_ctx, hist_summary, source_summary, ablation_feat], dim=-1)
            expert_outputs = torch.stack([expert(expert_in) for expert in self.experts], dim=1)
            routed = torch.sum(expert_outputs * route_weights.unsqueeze(-1), dim=1)
            gate = self.mix_gate(torch.cat([shared, routed, cond, hist_summary, source_summary], dim=-1))
            mixed = shared + gate * routed
            return self.out(torch.cat([count_feat, mixed, route_ctx], dim=-1)), route_weights, gate

    class HurdleCountHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            horizon: int,
            occurrence_prior_strength: float = 0.75,
            enable_source_specialists: bool = False,
        ):
            super().__init__()
            self.horizon = horizon
            self.occurrence_prior_strength = float(max(0.0, occurrence_prior_strength))
            self.enable_source_specialists = bool(enable_source_specialists)
            self.history_dim = 6
            self.source_dim = 5
            self.num_profiles = 3
            self.num_source_classes = 4
            self.source_class_names = ("no_source", "text_light", "edgar_rich", "full_rich")

            profile_in_dim = cond_dim + self.history_dim + self.source_dim
            expert_in_dim = hidden_dim + cond_dim + self.history_dim + self.source_dim
            temporal_in_dim = expert_in_dim + 1

            self.profile_router = nn.Sequential(
                nn.Linear(profile_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_profiles),
            )
            self.occurrence_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, horizon),
                )
                for _ in range(self.num_profiles)
            ])
            self.intensity_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, horizon),
                )
                for _ in range(self.num_profiles)
            ])
            self.source_occurrence_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, horizon),
                )
                for _ in range(self.num_source_classes)
            ])
            self.source_intensity_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, horizon),
                )
                for _ in range(self.num_source_classes)
            ])
            self.short_occurrence = nn.Sequential(
                nn.Linear(temporal_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.long_occurrence = nn.Sequential(
                nn.Linear(temporal_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.short_intensity = nn.Sequential(
                nn.Linear(temporal_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.long_intensity = nn.Sequential(
                nn.Linear(temporal_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.horizon_gate = nn.Sequential(
                nn.Linear(cond_dim + self.history_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.zero_bias = nn.Sequential(
                nn.Linear(profile_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.source_occurrence_gate = nn.Sequential(
                nn.Linear(profile_in_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.source_intensity_gate = nn.Sequential(
                nn.Linear(profile_in_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.positive_residual = nn.Sequential(
                nn.Linear(expert_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.positive_gate = nn.Sequential(
                nn.Linear(expert_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )

        def _history_summary(self, target_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            recent_width = min(7, target_hist.shape[1])
            recent = torch.relu(target_hist[:, -recent_width:])
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            last = torch.log1p(recent[:, -1:])
            mean = torch.log1p(recent.mean(dim=1, keepdim=True))
            peak = torch.log1p(recent.amax(dim=1, keepdim=True))
            nonzero = (recent > 0.5).float().mean(dim=1, keepdim=True)
            jump_mean = torch.log1p(step.abs().mean(dim=1, keepdim=True)) if step.shape[1] > 0 else last * 0.0
            jump_peak = torch.log1p(step.abs().amax(dim=1, keepdim=True)) if step.shape[1] > 0 else last * 0.0
            activity_prior = (
                -1.20
                + 2.15 * nonzero
                + 0.30 * last
                + 0.20 * mean
                + 0.12 * peak
                + 0.15 * jump_mean
            )
            hist_summary = torch.cat([last, mean, peak, nonzero, jump_mean, jump_peak], dim=-1)
            return hist_summary, activity_prior

        def _source_summary(
            self,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            edgar_active_flag: Optional[torch.Tensor] = None,
            text_active_flag: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            edgar_energy = torch.sqrt(edgar_feat.pow(2).mean(dim=1, keepdim=True).clamp_min(0.0))
            text_energy = torch.sqrt(text_feat.pow(2).mean(dim=1, keepdim=True).clamp_min(0.0))
            if edgar_active_flag is None:
                edgar_active = (edgar_energy > 1e-6).to(edgar_feat.dtype)
            else:
                edgar_active = edgar_active_flag.to(device=edgar_feat.device, dtype=edgar_feat.dtype)
            if text_active_flag is None:
                text_active = (text_energy > 1e-6).to(text_feat.dtype)
            else:
                text_active = text_active_flag.to(device=text_feat.device, dtype=text_feat.dtype)
            edgar_energy = edgar_energy * edgar_active
            text_energy = text_energy * text_active
            source_gap = torch.log1p(edgar_energy) - torch.log1p(text_energy)
            return torch.cat(
                [
                    torch.log1p(edgar_energy),
                    torch.log1p(text_energy),
                    edgar_active,
                    text_active,
                    source_gap,
                ],
                dim=-1,
            )

        def _source_class_weights(self, source_summary: torch.Tensor) -> torch.Tensor:
            edgar_active = source_summary[:, 2:3]
            text_active = source_summary[:, 3:4]
            no_source = (1.0 - edgar_active) * (1.0 - text_active)
            text_light = text_active * (1.0 - edgar_active)
            edgar_rich = edgar_active * (1.0 - text_active)
            full_rich = edgar_active * text_active
            weights = torch.cat([no_source, text_light, edgar_rich, full_rich], dim=-1)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

        def forward(
            self,
            hidden: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            edgar_feat: torch.Tensor,
            text_feat: torch.Tensor,
            horizon_value: torch.Tensor,
            edgar_active_flag: Optional[torch.Tensor] = None,
            text_active_flag: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            hist_summary, activity_prior = self._history_summary(target_hist)
            source_summary = self._source_summary(
                edgar_feat,
                text_feat,
                edgar_active_flag=edgar_active_flag,
                text_active_flag=text_active_flag,
            )
            profile_in = torch.cat([cond, hist_summary, source_summary], dim=-1)
            expert_in = torch.cat([hidden, cond, hist_summary, source_summary], dim=-1)
            horizon_feat = (
                torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(365.0)
            )
            temporal_in = torch.cat([expert_in, horizon_feat], dim=-1)

            profile_logits = self.profile_router(profile_in)
            profile_weights = torch.softmax(profile_logits, dim=-1)
            occurrence_expert = torch.stack([head(expert_in) for head in self.occurrence_experts], dim=1)
            intensity_expert = torch.stack([head(expert_in) for head in self.intensity_experts], dim=1)
            occurrence_profile = torch.sum(occurrence_expert * profile_weights.unsqueeze(-1), dim=1)
            intensity_profile = torch.sum(intensity_expert * profile_weights.unsqueeze(-1), dim=1)

            source_class_weights = profile_in.new_zeros((profile_in.shape[0], self.num_source_classes))
            source_occurrence = occurrence_profile * 0.0
            source_intensity = intensity_profile * 0.0
            if self.enable_source_specialists:
                source_class_weights = self._source_class_weights(source_summary)
                source_occurrence_expert = torch.stack(
                    [head(expert_in) for head in self.source_occurrence_experts],
                    dim=1,
                )
                source_intensity_expert = torch.stack(
                    [head(expert_in) for head in self.source_intensity_experts],
                    dim=1,
                )
                source_occ_gate = self.source_occurrence_gate(torch.cat([profile_in, horizon_feat], dim=-1))
                source_int_gate = self.source_intensity_gate(torch.cat([profile_in, horizon_feat], dim=-1))
                source_occurrence = source_occ_gate * torch.sum(
                    source_occurrence_expert * source_class_weights.unsqueeze(-1),
                    dim=1,
                )
                source_intensity = source_int_gate * torch.sum(
                    source_intensity_expert * source_class_weights.unsqueeze(-1),
                    dim=1,
                )

            horizon_mix = self.horizon_gate(torch.cat([cond, hist_summary, horizon_feat], dim=-1))
            occurrence_base = (
                (1.0 - horizon_mix) * self.short_occurrence(temporal_in)
                + horizon_mix * self.long_occurrence(temporal_in)
            )
            intensity_base = (
                (1.0 - horizon_mix) * self.short_intensity(temporal_in)
                + horizon_mix * self.long_intensity(temporal_in)
            )

            active_anchor = (
                0.55 * hist_summary[:, 0:1]
                + 0.30 * hist_summary[:, 1:2]
                + 0.15 * hist_summary[:, 2:3]
            )
            positive_raw = (
                active_anchor.expand(-1, self.horizon)
                + intensity_base
                + intensity_profile
                + source_intensity
                + self.positive_gate(expert_in) * self.positive_residual(expert_in)
            )
            occurrence_logits = (
                occurrence_base
                + occurrence_profile
                + source_occurrence
                + self.zero_bias(profile_in)
                + self.occurrence_prior_strength * activity_prior.expand(-1, self.horizon)
            )
            return (
                positive_raw,
                occurrence_logits,
                profile_weights,
                horizon_mix.expand(-1, self.horizon),
                source_class_weights,
            )

    class LiteInvestorsHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            horizon: int,
            delta_bucket_values: Optional[List[float]] = None,
        ):
            super().__init__()
            self.horizon = int(horizon)
            if delta_bucket_values is None:
                delta_bucket_values = [-16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
            bucket_tensor = torch.tensor(delta_bucket_values, dtype=torch.float32)
            self.register_buffer("delta_bucket_values", bucket_tensor)
            self.history_dim = 7
            self.source_dim = 4
            core_in_dim = hidden_dim + cond_dim + self.history_dim + self.source_dim + 1
            self.anchor_refine = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.occurrence_head = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.state_head = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon * len(delta_bucket_values)),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Tanh(),
            )
            self.residual_gate = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )

        def _history_summary(self, target_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            recent_width = min(7, target_hist.shape[1])
            recent = torch.relu(target_hist[:, -recent_width:])
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            last = recent[:, -1:]
            mean = recent.mean(dim=1, keepdim=True)
            peak = recent.amax(dim=1, keepdim=True)
            nonzero = (recent > 0.5).float().mean(dim=1, keepdim=True)
            jump_last = step[:, -1:] if step.shape[1] > 0 else last * 0.0
            jump_mean = step.mean(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            jump_peak = step.abs().amax(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            hist_summary = torch.cat(
                [
                    torch.log1p(last),
                    torch.log1p(mean),
                    torch.log1p(peak),
                    nonzero,
                    jump_last,
                    jump_mean,
                    torch.log1p(jump_peak.abs()),
                ],
                dim=-1,
            )
            active_prior = (
                -1.15
                + 2.25 * nonzero
                + 0.35 * torch.log1p(last)
                + 0.20 * torch.log1p(mean)
                + 0.12 * torch.log1p(peak)
                + 0.15 * torch.log1p(jump_peak.abs())
            )
            return hist_summary, active_prior

        def forward(
            self,
            hidden: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            horizon_value: torch.Tensor,
            edgar_active_flag: torch.Tensor,
            text_active_flag: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            hist_summary, active_prior = self._history_summary(target_hist)
            horizon_feat = (
                torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(365.0)
            )
            source_summary = torch.cat(
                [
                    edgar_active_flag,
                    text_active_flag,
                    edgar_active_flag * text_active_flag,
                    text_active_flag - edgar_active_flag,
                ],
                dim=-1,
            )
            core = torch.cat([hidden, cond, hist_summary, source_summary, horizon_feat], dim=-1)
            last_level = torch.relu(target_hist[:, -1:])
            recent_mean = torch.relu(target_hist[:, -min(5, target_hist.shape[1]):]).mean(dim=1, keepdim=True)
            base_anchor = 0.80 * last_level + 0.20 * recent_mean
            anchor_path = base_anchor.expand(-1, self.horizon) + 0.10 * self.anchor_refine(core)
            occurrence_logits = self.occurrence_head(core) + active_prior.expand(-1, self.horizon)
            state_logits = self.state_head(core).view(-1, self.horizon, self.delta_bucket_values.numel())
            state_probs = torch.softmax(state_logits, dim=-1)
            expected_delta = torch.sum(
                state_probs * self.delta_bucket_values.view(1, 1, -1),
                dim=-1,
            )
            residual = 0.35 * self.residual_gate(core) * self.residual_head(core)
            count_positive = torch.relu(anchor_path + expected_delta + residual)
            count_occurrence_prob = torch.sigmoid(occurrence_logits)
            count_out = count_occurrence_prob * count_positive
            return count_out, occurrence_logits, count_positive, state_logits, anchor_path

    class UnifiedFinancingProcessHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            horizon: int,
        ):
            super().__init__()
            self.horizon = int(horizon)
            self.history_dim = 7
            self.source_dim = 4
            core_in_dim = hidden_dim + cond_dim + self.history_dim + self.source_dim + 1
            self.shared = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.event_head = nn.Linear(hidden_dim, horizon)
            self.count_head = nn.Sequential(
                nn.Linear(hidden_dim + self.history_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.amount_head = nn.Sequential(
                nn.Linear(hidden_dim + self.history_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )

        def _history_summary(self, target_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            recent_width = min(7, target_hist.shape[1])
            recent = torch.relu(target_hist[:, -recent_width:])
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            last = recent[:, -1:]
            mean = recent.mean(dim=1, keepdim=True)
            peak = recent.amax(dim=1, keepdim=True)
            nonzero = (recent > 0.5).float().mean(dim=1, keepdim=True)
            jump_last = step[:, -1:] if step.shape[1] > 0 else last * 0.0
            jump_mean = step.mean(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            jump_peak = step.abs().amax(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            hist_summary = torch.cat(
                [
                    torch.log1p(last),
                    torch.log1p(mean),
                    torch.log1p(peak),
                    nonzero,
                    jump_last,
                    jump_mean,
                    torch.log1p(jump_peak.abs()),
                ],
                dim=-1,
            )
            active_prior = (
                -1.35
                + 2.10 * nonzero
                + 0.30 * torch.log1p(last)
                + 0.15 * torch.log1p(mean)
                + 0.10 * torch.log1p(jump_peak.abs())
            )
            amount_prior = 0.35 * torch.log1p(last) + 0.35 * torch.log1p(mean) + 0.30 * torch.log1p(peak)
            return hist_summary, active_prior, amount_prior

        def forward(
            self,
            hidden: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            horizon_value: torch.Tensor,
            edgar_active_flag: torch.Tensor,
            text_active_flag: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]:
            hist_summary, active_prior, amount_prior = self._history_summary(target_hist)
            horizon_feat = (
                torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(365.0)
            )
            source_summary = torch.cat(
                [
                    edgar_active_flag,
                    text_active_flag,
                    edgar_active_flag * text_active_flag,
                    text_active_flag - edgar_active_flag,
                ],
                dim=-1,
            )
            core = torch.cat([hidden, cond, hist_summary, source_summary, horizon_feat], dim=-1)
            shared = self.shared(core)
            shared_with_context = torch.cat([shared, hist_summary, source_summary], dim=-1)
            event_logits = self.event_head(shared) + active_prior.expand(-1, self.horizon)
            event_prob = torch.sigmoid(event_logits)
            count_positive = torch.nn.functional.softplus(self.count_head(shared_with_context))
            amount_positive_log = torch.nn.functional.softplus(
                self.amount_head(shared_with_context) + amount_prior.expand(-1, self.horizon)
            )
            count_out = event_prob * count_positive
            amount_out_log = event_prob * amount_positive_log
            return (
                event_logits,
                event_prob,
                count_positive,
                count_out,
                amount_positive_log,
                amount_out_log,
                None,
                None,
                None,
                None,
            )


    class FactorizedFinancingProcessHead(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            cond_dim: int,
            horizon: int,
        ):
            super().__init__()
            self.horizon = int(horizon)
            self.history_dim = 7
            self.source_dim = 4
            core_in_dim = hidden_dim + cond_dim + self.history_dim + self.source_dim + 1
            self.shared = nn.Sequential(
                nn.Linear(core_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.event_head = nn.Linear(hidden_dim, horizon)
            self.breadth_anchor_head = nn.Sequential(
                nn.Linear(hidden_dim + self.history_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.breadth_delta_head = nn.Sequential(
                nn.Linear(hidden_dim + cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.intensity_head = nn.Sequential(
                nn.Linear(hidden_dim + self.history_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.amount_coupling_head = nn.Sequential(
                nn.Linear(hidden_dim + self.source_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )

        def _history_summary(
            self,
            target_hist: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            recent_width = min(7, target_hist.shape[1])
            recent = torch.relu(target_hist[:, -recent_width:])
            step = recent[:, 1:] - recent[:, :-1] if recent.shape[1] > 1 else recent[:, :1] * 0.0
            last = recent[:, -1:]
            mean = recent.mean(dim=1, keepdim=True)
            peak = recent.amax(dim=1, keepdim=True)
            nonzero = (recent > 0.5).float().mean(dim=1, keepdim=True)
            jump_last = step[:, -1:] if step.shape[1] > 0 else last * 0.0
            jump_mean = step.mean(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            jump_peak = step.abs().amax(dim=1, keepdim=True) if step.shape[1] > 0 else last * 0.0
            hist_summary = torch.cat(
                [
                    torch.log1p(last),
                    torch.log1p(mean),
                    torch.log1p(peak),
                    nonzero,
                    jump_last,
                    jump_mean,
                    torch.log1p(jump_peak.abs()),
                ],
                dim=-1,
            )
            active_prior = (
                -1.20
                + 2.00 * nonzero
                + 0.30 * torch.log1p(last)
                + 0.20 * torch.log1p(mean)
                + 0.10 * torch.log1p(jump_peak.abs())
            )
            breadth_prior = 0.65 * torch.log1p(last) + 0.25 * torch.log1p(mean) + 0.10 * torch.log1p(peak)
            amount_prior = 0.35 * torch.log1p(last) + 0.35 * torch.log1p(mean) + 0.30 * torch.log1p(peak)
            return hist_summary, active_prior, breadth_prior, amount_prior

        def forward(
            self,
            hidden: torch.Tensor,
            cond: torch.Tensor,
            target_hist: torch.Tensor,
            horizon_value: torch.Tensor,
            edgar_active_flag: torch.Tensor,
            text_active_flag: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            hist_summary, active_prior, breadth_prior, amount_prior = self._history_summary(target_hist)
            horizon_feat = (
                torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(365.0)
            )
            source_summary = torch.cat(
                [
                    edgar_active_flag,
                    text_active_flag,
                    edgar_active_flag * text_active_flag,
                    text_active_flag - edgar_active_flag,
                ],
                dim=-1,
            )
            core = torch.cat([hidden, cond, hist_summary, source_summary, horizon_feat], dim=-1)
            shared = self.shared(core)
            shared_with_context = torch.cat([shared, hist_summary, source_summary], dim=-1)

            event_logits = self.event_head(shared) + active_prior.expand(-1, self.horizon)
            event_prob = torch.sigmoid(event_logits)

            breadth_anchor_log = torch.nn.functional.softplus(
                self.breadth_anchor_head(shared_with_context) + breadth_prior.expand(-1, self.horizon)
            )
            breadth_delta = 0.35 * torch.tanh(self.breadth_delta_head(torch.cat([shared, cond], dim=-1)))
            breadth_log = torch.clamp(
                breadth_anchor_log + breadth_delta * (0.30 + 0.70 * event_prob),
                min=0.0,
                max=12.0,
            )
            breadth_positive = torch.expm1(breadth_log)

            intensity_log = torch.nn.functional.softplus(
                self.intensity_head(shared_with_context) + amount_prior.expand(-1, self.horizon)
            )
            amount_coupling = 0.20 + 0.60 * self.amount_coupling_head(torch.cat([shared, source_summary], dim=-1))
            amount_positive_log = intensity_log + amount_coupling * breadth_log

            count_out = event_prob * breadth_positive
            amount_out_log = event_prob * amount_positive_log
            return (
                event_logits,
                event_prob,
                breadth_positive,
                count_out,
                amount_positive_log,
                amount_out_log,
                breadth_log,
                breadth_anchor_log,
                intensity_log,
                amount_coupling,
            )

    class V740AlphaNet(nn.Module):
        def __init__(
            self,
            in_channels: int,
            seq_len: int,
            horizon: int,
            hidden_dim: int = 64,
            cond_dim: int = 16,
            enable_continuous_anchor: bool = True,
            continuous_anchor_strength: float = 0.85,
            enable_target_routing: bool = True,
            target_route_experts: int = 3,
            enable_count_anchor: bool = True,
            count_anchor_strength: float = 0.70,
            enable_count_jump: bool = True,
            count_jump_strength: float = 0.30,
            enable_count_sparsity_gate: bool = True,
            count_sparsity_gate_strength: float = 0.75,
            enable_count_source_routing: bool = True,
            count_route_experts: int = 3,
            count_route_floor: float = 0.10,
            enable_count_hurdle_head: bool = False,
            enable_count_source_specialists: bool = False,
            enable_v741_lite: bool = False,
            enable_financing_consistency: bool = False,
            enable_financing_factorization: bool = False,
            enable_financing_guarded_phase: bool = False,
            enable_financing_evidence_residual: bool = False,
            financing_process_blend: float = 0.0,
            financing_investor_blend_scale: float = 1.0,
            financing_binary_blend_scale: float = 0.35,
            financing_funding_blend_scale: float = 0.20,
            funding_log_domain_enabled: bool = False,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.horizon = horizon
            self.task_mod_enabled = True
            self.continuous_anchor_enabled = bool(enable_continuous_anchor)
            self.continuous_anchor_strength = float(max(0.0, continuous_anchor_strength))
            self.target_routing_enabled = bool(enable_target_routing)
            self.count_anchor_enabled = bool(enable_count_anchor)
            self.count_jump_enabled = bool(enable_count_jump)
            self.count_sparsity_gate_enabled = bool(enable_count_sparsity_gate)
            self.count_sparsity_gate_strength = float(min(1.0, max(0.0, count_sparsity_gate_strength)))
            self.count_source_routing_enabled = bool(enable_count_source_routing)
            self.count_hurdle_head_enabled = bool(enable_count_hurdle_head)
            self.count_source_specialists_enabled = bool(enable_count_source_specialists)
            self.v741_lite_enabled = bool(enable_v741_lite)
            self.financing_process_enabled = bool(enable_financing_consistency)
            self.financing_factorization_enabled = bool(enable_financing_factorization)
            self.financing_guarded_phase_enabled = bool(enable_financing_guarded_phase)
            self.financing_evidence_residual_enabled = bool(enable_financing_evidence_residual)
            self.financing_process_blend = float(min(1.0, max(0.0, financing_process_blend)))
            self.financing_investor_blend_scale = float(max(0.0, financing_investor_blend_scale))
            self.financing_binary_blend_scale = float(max(0.0, financing_binary_blend_scale))
            self.financing_funding_blend_scale = float(max(0.0, financing_funding_blend_scale))
            self.funding_log_domain_enabled = bool(funding_log_domain_enabled)
            self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.decomp = MovingAverageDecomposition(kernel_size=5)
            self.multi_res = MultiResolutionBlock(hidden_dim, hidden_dim)
            self.patch_mixer = PatchContextMixer(hidden_dim, patch_size=8)
            self.casa_local = CASALocalContextBlock(hidden_dim, cond_dim, kernel_size=7)
            self.memory_branch = CompactTemporalMemory(hidden_dim)
            self.value_branch = ValueBucketEncoder(num_bins=32, hidden_dim=hidden_dim)
            self.cond_encoder = ConditionEncoder(emb_dim=cond_dim, seq_len=seq_len)
            self.edgar_memory = EventMemoryEncoder(hidden_dim)
            self.text_memory = EventMemoryEncoder(hidden_dim)
            self.binary_kernel = BinaryHistoryKernel(
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                horizon=horizon,
            )
            self.fusion = InvariantVariantFusion(hidden_dim, cond_dim)
            self.task_mod = TaskSpecificModulator(hidden_dim)
            self.static_proj = StaticSummaryProjector(max(1, in_channels - 1), hidden_dim)
            self.time_fusion = StaticDynamicTimeFusion(hidden_dim, cond_dim)
            self.backbone_norm = nn.LayerNorm(hidden_dim)
            self.target_decoder = TargetRoutedDecoder(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_targets=4,
                num_experts=target_route_experts,
            )
            self.continuous_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.count_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.binary_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.combine = nn.Sequential(
                nn.Linear(hidden_dim * 7 + cond_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.shared_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.continuous_anchor_gate = nn.Sequential(
                nn.Linear(hidden_dim + cond_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
                nn.Sigmoid(),
            )
            self.binary_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.count_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.count_occurrence_head = nn.Sequential(
                nn.Linear(hidden_dim + cond_dim + 5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )
            self.lite_investors_head = LiteInvestorsHead(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                horizon=horizon,
            )
            if self.financing_factorization_enabled:
                self.financing_process_head = FactorizedFinancingProcessHead(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    horizon=horizon,
                )
            else:
                self.financing_process_head = UnifiedFinancingProcessHead(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    horizon=horizon,
                )
            self.count_hurdle_head = HurdleCountHead(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                horizon=horizon,
                occurrence_prior_strength=count_sparsity_gate_strength,
                enable_source_specialists=enable_count_source_specialists,
            )
            self.count_structure_head = CountStructureHead(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                horizon=horizon,
                anchor_strength=count_anchor_strength,
                jump_strength=count_jump_strength if enable_count_jump else 0.0,
            )
            self.count_source_decoder = CountSourceRoutedDecoder(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_ablations=6,
                num_experts=count_route_experts,
                route_floor=count_route_floor,
            )
            self.event_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.task_shared_bias = nn.Embedding(4, horizon)
            self.task_count_bias = nn.Embedding(4, horizon)
            self.task_binary_bias = nn.Embedding(4, horizon)
            self.task_event_bias = nn.Embedding(4, 1)
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
            )

        def _source_activity_flag(
            self,
            recent_tokens: Optional[torch.Tensor],
            bucket_tokens: Optional[torch.Tensor],
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            active = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            if recent_tokens is not None and recent_tokens.numel() > 0 and recent_tokens.shape[-1] > 2:
                active = torch.maximum(
                    active,
                    (recent_tokens[:, :, 2] > 0.0).any(dim=1, keepdim=True).to(dtype=dtype),
                )
            if bucket_tokens is not None and bucket_tokens.numel() > 0 and bucket_tokens.shape[-1] > 0:
                active = torch.maximum(
                    active,
                    (bucket_tokens[:, :, 0] > 0.0).any(dim=1, keepdim=True).to(dtype=dtype),
                )
            return active

        def forward(
            self,
            x: torch.Tensor,
            task_idx: torch.Tensor,
            target_idx: torch.Tensor,
            horizon_value: torch.Tensor,
            ablation_idx: torch.Tensor,
            edgar_recent: Optional[torch.Tensor] = None,
            edgar_bucket: Optional[torch.Tensor] = None,
            text_recent: Optional[torch.Tensor] = None,
            text_bucket: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            # x: [B, C, L]
            target_hist = x[:, 0, :]
            if x.shape[1] > 1:
                static_summary = x[:, 1:, :].mean(dim=-1)
            else:
                static_summary = x.new_zeros((x.shape[0], 1))
            x = self.input_proj(x)
            trend, seasonal = self.decomp(x)
            cond = self.cond_encoder(task_idx, target_idx, horizon_value, ablation_idx)
            fused = self.multi_res(trend + seasonal)
            fused = self.patch_mixer(fused)
            fused = self.casa_local(fused, cond)
            memory_feat = self.memory_branch(fused)
            value_feat = self.value_branch(target_hist)
            fused = self.fusion(fused, cond)
            pooled = fused.mean(dim=-1)
            static_feat = self.static_proj(static_summary)
            edgar_feat = self.edgar_memory(edgar_recent, edgar_bucket, x.shape[0], x.device)
            text_feat = self.text_memory(text_recent, text_bucket, x.shape[0], x.device)
            edgar_active_flag = self._source_activity_flag(
                edgar_recent,
                edgar_bucket,
                x.shape[0],
                x.device,
                pooled.dtype,
            )
            text_active_flag = self._source_activity_flag(
                text_recent,
                text_bucket,
                x.shape[0],
                x.device,
                pooled.dtype,
            )
            time_fused = self.time_fusion(
                pooled,
                memory_feat,
                value_feat,
                static_feat,
                edgar_feat,
                text_feat,
                cond,
            )
            pooled = self.backbone_norm(pooled)
            pooled = self.combine(torch.cat([
                pooled, memory_feat, value_feat, static_feat, edgar_feat, text_feat, time_fused, cond,
            ], dim=-1))
            if self.task_mod_enabled:
                pooled = self.task_mod(pooled, task_idx)
                shared_bias = self.task_shared_bias(task_idx)
                count_bias = self.task_count_bias(task_idx)
                binary_bias = self.task_binary_bias(task_idx)
                event_bias = self.task_event_bias(task_idx).squeeze(-1)
            else:
                shared_bias = torch.zeros((x.shape[0], self.horizon), device=x.device, dtype=pooled.dtype)
                count_bias = torch.zeros((x.shape[0], self.horizon), device=x.device, dtype=pooled.dtype)
                binary_bias = torch.zeros((x.shape[0], self.horizon), device=x.device, dtype=pooled.dtype)
                event_bias = torch.zeros((x.shape[0],), device=x.device, dtype=pooled.dtype)
            pooled = self.backbone_norm(pooled)
            if self.target_routing_enabled:
                decoded, route_weights, route_gate = self.target_decoder(
                    pooled,
                    memory_feat,
                    value_feat,
                    edgar_feat,
                    text_feat,
                    cond,
                    target_hist,
                    target_idx,
                )
            else:
                decoded = pooled
                route_weights = None
                route_gate = None
            continuous_feat = self.continuous_decoder(decoded)
            count_feat = self.count_decoder(decoded)
            binary_feat = self.binary_decoder(decoded)
            count_route_weights = None
            count_route_gate = None
            if self.count_source_routing_enabled:
                count_feat, count_route_weights, count_route_gate = self.count_source_decoder(
                    count_feat,
                    decoded,
                    memory_feat,
                    value_feat,
                    edgar_feat,
                    text_feat,
                    cond,
                    target_hist,
                    ablation_idx,
                    edgar_active_flag=edgar_active_flag,
                    text_active_flag=text_active_flag,
                )
            recent_width = min(4, target_hist.shape[1])
            anchor_term = torch.zeros((x.shape[0], self.horizon), device=x.device, dtype=pooled.dtype)
            if self.continuous_anchor_enabled and self.continuous_anchor_strength > 0.0:
                anchor_level = 0.7 * target_hist[:, -1:] + 0.3 * target_hist[:, -recent_width:].mean(dim=1, keepdim=True)
                continuous_anchor = self.continuous_anchor_gate(torch.cat([continuous_feat, cond, anchor_level], dim=-1))
                anchor_term = self.continuous_anchor_strength * continuous_anchor * anchor_level.expand(-1, self.horizon)
            event_logits = self.event_head(binary_feat).squeeze(-1) + event_bias
            binary_logits = self.binary_head(binary_feat) + binary_bias
            binary_probs = self.binary_kernel(
                target_hist,
                binary_feat,
                memory_feat,
                value_feat,
                cond,
                event_logits=event_logits,
                base_logits=binary_logits,
            )
            count_base_raw = self.count_head(count_feat) + count_bias
            if self.count_anchor_enabled:
                count_base_raw = count_base_raw + self.count_structure_head(count_feat, cond, target_hist)
            count_base = torch.nn.functional.softplus(count_base_raw)
            count_sparsity_gate = None
            count_occurrence_logits = None
            count_positive = None
            count_occurrence_prob = None
            count_profile_weights = None
            count_horizon_mix = None
            count_hurdle_blend = None
            count_source_class_weights = None
            count_state_logits = None
            count_anchor_path = None
            if self.count_sparsity_gate_enabled:
                count_recent_width = min(7, target_hist.shape[1])
                count_recent = torch.relu(target_hist[:, -count_recent_width:])
                count_last = count_recent[:, -1:]
                count_mean = count_recent.mean(dim=1, keepdim=True)
                count_peak = count_recent.amax(dim=1, keepdim=True)
                count_nonzero = (count_recent > 0.5).float().mean(dim=1, keepdim=True)
                count_step = count_recent[:, 1:] - count_recent[:, :-1] if count_recent.shape[1] > 1 else count_recent[:, :1] * 0.0
                count_jump = count_step.abs().mean(dim=1, keepdim=True) if count_step.shape[1] > 0 else count_last * 0.0
                activity_logit = (
                    -1.35
                    + 2.50 * count_nonzero
                    + 0.30 * torch.log1p(count_last)
                    + 0.20 * torch.log1p(count_mean)
                    + 0.20 * torch.log1p(count_peak)
                    + 0.15 * torch.log1p(count_jump)
                )
                count_occurrence_input = torch.cat(
                    [
                        count_feat,
                        cond,
                        torch.cat(
                            [
                                torch.log1p(count_last),
                                torch.log1p(count_mean),
                                torch.log1p(count_peak),
                                count_nonzero,
                                torch.log1p(count_jump),
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                )
                legacy_count_occurrence_logits = self.count_occurrence_head(count_occurrence_input)
                legacy_count_occurrence_logits = legacy_count_occurrence_logits + self.count_sparsity_gate_strength * activity_logit.expand(-1, self.horizon)
                legacy_count_gate = torch.sigmoid(legacy_count_occurrence_logits)
                legacy_count_scale = 1.0 - self.count_sparsity_gate_strength * (1.0 - legacy_count_gate)
                count_base = count_base * legacy_count_scale
            else:
                legacy_count_gate = None
            if self.count_hurdle_head_enabled:
                (
                    count_positive_raw,
                    count_occurrence_logits,
                    count_profile_weights,
                    count_horizon_mix,
                    count_source_class_weights,
                ) = self.count_hurdle_head(
                    count_feat,
                    cond,
                    target_hist,
                    edgar_feat,
                    text_feat,
                    horizon_value,
                    edgar_active_flag=edgar_active_flag,
                    text_active_flag=text_active_flag,
                )
                if self.count_anchor_enabled:
                    count_positive_raw = count_positive_raw + 0.25 * self.count_structure_head(count_feat, cond, target_hist)
                count_positive = torch.nn.functional.softplus(count_positive_raw + count_bias)
                count_occurrence_prob = torch.sigmoid(count_occurrence_logits)
                hurdle_count = count_occurrence_prob * count_positive
                horizon_scale = torch.clamp(
                    torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(30.0),
                    0.0,
                    1.0,
                )
                task1_mask = (task_idx == 0).to(pooled.dtype).unsqueeze(-1)
                task2_mask = (task_idx == 1).to(pooled.dtype).unsqueeze(-1)
                task3_mask = (task_idx == 2).to(pooled.dtype).unsqueeze(-1)
                edgar_active = (edgar_feat.pow(2).mean(dim=1, keepdim=True) > 1e-6).to(pooled.dtype)
                if self.count_source_specialists_enabled and count_source_class_weights is not None:
                    no_source = count_source_class_weights[:, 0:1]
                    text_light = count_source_class_weights[:, 1:2]
                    edgar_rich = count_source_class_weights[:, 2:3]
                    full_rich = count_source_class_weights[:, 3:4]
                    count_hurdle_blend = (
                        0.08
                        + 0.12 * horizon_scale
                        + 0.04 * task1_mask
                        + 0.10 * task3_mask
                        + 0.22 * no_source * horizon_scale
                        + 0.10 * text_light * horizon_scale
                        + 0.06 * full_rich * torch.maximum(horizon_scale, torch.full_like(horizon_scale, 0.35))
                        - 0.18 * task2_mask * edgar_rich
                        - 0.12 * (1.0 - horizon_scale) * edgar_rich
                        - 0.08 * (1.0 - horizon_scale) * full_rich
                    )
                else:
                    count_hurdle_blend = (
                        0.18
                        + 0.28 * horizon_scale
                        + 0.05 * task1_mask
                        + 0.15 * task3_mask
                        - 0.12 * task2_mask * edgar_active
                        - 0.08 * (1.0 - horizon_scale) * edgar_active
                        + 0.10 * (1.0 - edgar_active) * horizon_scale
                    )
                count_hurdle_blend = torch.clamp(count_hurdle_blend, 0.05, 0.70)
                count_out = (1.0 - count_hurdle_blend) * count_base + count_hurdle_blend * hurdle_count
                if legacy_count_gate is not None:
                    count_sparsity_gate = (
                        (1.0 - count_hurdle_blend) * legacy_count_gate
                        + count_hurdle_blend * count_occurrence_prob
                    )
                else:
                    count_sparsity_gate = count_occurrence_prob
            else:
                count_out = count_base
                count_occurrence_logits = legacy_count_occurrence_logits if self.count_sparsity_gate_enabled else None
                count_sparsity_gate = legacy_count_gate
            investor_mask = (target_idx == 1).to(pooled.dtype).unsqueeze(-1)
            funding_mask = (target_idx == 0).to(pooled.dtype).unsqueeze(-1)
            binary_mask = (target_idx == 2).to(pooled.dtype).unsqueeze(-1)
            if self.v741_lite_enabled:
                (
                    lite_count_out,
                    lite_count_occurrence_logits,
                    lite_count_positive,
                    lite_count_state_logits,
                    lite_anchor_path,
                ) = self.lite_investors_head(
                    count_feat,
                    cond,
                    target_hist,
                    horizon_value,
                    edgar_active_flag=edgar_active_flag,
                    text_active_flag=text_active_flag,
                )
                count_out = (1.0 - investor_mask) * count_out + investor_mask * lite_count_out
                count_occurrence_logits = lite_count_occurrence_logits
                count_occurrence_prob = torch.sigmoid(lite_count_occurrence_logits)
                count_positive = lite_count_positive
                count_sparsity_gate = count_occurrence_prob
                count_state_logits = lite_count_state_logits
                count_anchor_path = lite_anchor_path
                count_profile_weights = None
                count_horizon_mix = None
                count_hurdle_blend = None
                count_source_class_weights = None
            continuous = (
                self.shared_head(continuous_feat)
                + shared_bias
                + anchor_term
            )
            legacy_continuous = continuous
            legacy_count = count_out
            legacy_binary_prob = binary_probs
            financing_event_logits = None
            financing_event_prob = None
            financing_count_positive = None
            financing_count = None
            financing_amount_log = None
            financing_amount_positive_log = None
            financing_breadth_log = None
            financing_breadth_anchor_log = None
            financing_intensity_log = None
            financing_amount_coupling = None
            financing_investor_gate = None
            financing_binary_gate = None
            financing_funding_gate = None
            if self.financing_process_enabled:
                guarded_investors_target = bool((target_idx == 1).all().item())
                financing_hidden = decoded
                financing_cond = cond
                if (
                    (self.financing_guarded_phase_enabled or self.financing_evidence_residual_enabled)
                    and not guarded_investors_target
                ):
                    financing_hidden = decoded.detach()
                    financing_cond = cond.detach()
                (
                    financing_event_logits,
                    financing_event_prob,
                    financing_count_positive,
                    financing_count,
                    financing_amount_positive_log,
                    financing_amount_log,
                    financing_breadth_log,
                    financing_breadth_anchor_log,
                    financing_intensity_log,
                    financing_amount_coupling,
                ) = self.financing_process_head(
                    financing_hidden,
                    financing_cond,
                    target_hist,
                    horizon_value,
                    edgar_active_flag=edgar_active_flag,
                    text_active_flag=text_active_flag,
                )
                blend = self.financing_process_blend
                count_blend = blend
                funding_blend = blend
                binary_blend = blend
                if self.financing_factorization_enabled:
                    count_blend *= self.financing_investor_blend_scale
                    funding_blend *= self.financing_funding_blend_scale
                    binary_blend *= self.financing_binary_blend_scale
                if self.financing_evidence_residual_enabled and self.financing_factorization_enabled:
                    no_source = (1.0 - edgar_active_flag) * (1.0 - text_active_flag)
                    edgar_only = edgar_active_flag * (1.0 - text_active_flag)
                    text_only = text_active_flag * (1.0 - edgar_active_flag)
                    full_source = edgar_active_flag * text_active_flag

                    zero_gate = torch.zeros_like(financing_event_prob)
                    financing_binary_gate = zero_gate
                    financing_funding_gate = zero_gate
                    binary_probs = legacy_binary_prob
                    continuous = legacy_continuous

                    if guarded_investors_target:
                        breadth_agreement = torch.ones_like(financing_event_prob)
                        if financing_breadth_log is not None and financing_breadth_anchor_log is not None:
                            breadth_gap = torch.abs(financing_breadth_log - financing_breadth_anchor_log)
                            breadth_agreement = torch.exp(-breadth_gap / 0.75)
                        event_support = torch.clamp(0.25 + 0.75 * financing_event_prob.detach(), 0.0, 1.0)
                        investor_source_gate = torch.clamp(
                            0.95 * edgar_only
                            + 0.35 * full_source
                            + 0.20 * text_only
                            + 0.10 * no_source,
                            0.0,
                            1.0,
                        )
                        financing_investor_gate = torch.clamp(
                            count_blend * investor_source_gate * event_support * breadth_agreement,
                            0.0,
                            1.0,
                        )
                        count_out = legacy_count + investor_mask * financing_investor_gate * (financing_count - legacy_count)
                    else:
                        financing_investor_gate = zero_gate
                        count_out = legacy_count
                elif self.financing_guarded_phase_enabled and self.financing_factorization_enabled:
                    horizon_scale = torch.clamp(
                        torch.log1p(horizon_value.float()).unsqueeze(-1) / math.log1p(30.0),
                        0.0,
                        1.0,
                    )
                    no_source = (1.0 - edgar_active_flag) * (1.0 - text_active_flag)
                    edgar_only = edgar_active_flag * (1.0 - text_active_flag)
                    text_only = text_active_flag * (1.0 - edgar_active_flag)
                    full_source = edgar_active_flag * text_active_flag

                    legacy_count_ref = legacy_count.detach().clamp_min(0.0)
                    count_gap = torch.abs(
                        torch.log1p(financing_count.clamp_min(0.0)) - torch.log1p(legacy_count_ref)
                    )
                    count_agreement = torch.exp(-count_gap / 0.60)
                    investor_backoff = torch.clamp(
                        0.05 * edgar_only
                        + 0.25 * full_source
                        + 0.35 * text_only
                        + 0.55 * no_source
                        + 0.20 * (1.0 - count_agreement),
                        0.0,
                        0.90,
                    )
                    financing_investor_gate = torch.clamp(
                        count_blend * (1.0 - investor_backoff),
                        0.0,
                        1.0,
                    )
                    count_out = legacy_count + investor_mask * financing_investor_gate * (financing_count - legacy_count)

                    legacy_binary_ref = legacy_binary_prob.detach().clamp(1e-4, 1.0 - 1e-4)
                    legacy_binary_logits = torch.logit(legacy_binary_prob.clamp(1e-4, 1.0 - 1e-4))
                    event_gap = torch.abs(financing_event_prob - legacy_binary_ref)
                    event_agreement = torch.exp(-event_gap / 0.18)
                    binary_source_gate = torch.clamp(
                        0.05
                        + 0.55 * no_source
                        + 0.25 * text_only
                        + 0.10 * edgar_only
                        + 0.05 * full_source,
                        0.0,
                        1.0,
                    )
                    binary_horizon_gate = 0.20 + 0.80 * horizon_scale
                    financing_binary_gate = torch.clamp(
                        binary_blend * event_agreement * binary_source_gate * binary_horizon_gate,
                        0.0,
                        1.0,
                    )
                    binary_delta = torch.tanh(financing_event_logits - legacy_binary_logits.detach())
                    binary_logits = legacy_binary_logits + binary_mask * financing_binary_gate * binary_delta
                    binary_probs = torch.sigmoid(binary_logits)

                    legacy_continuous_ref = legacy_continuous.detach()
                    amount_gap = torch.abs(financing_amount_log - legacy_continuous_ref)
                    amount_agreement = torch.exp(-amount_gap / 1.50)
                    funding_source_gate = torch.clamp(
                        0.02
                        + 0.90 * edgar_only
                        + 0.08 * text_only
                        + 0.05 * no_source
                        + 0.04 * full_source,
                        0.0,
                        1.0,
                    )
                    funding_event_gate = 0.15 + 0.85 * financing_event_prob.detach()
                    financing_funding_gate = torch.clamp(
                        funding_blend * amount_agreement * funding_source_gate * funding_event_gate,
                        0.0,
                        1.0,
                    )
                    if self.funding_log_domain_enabled:
                        funding_delta = torch.tanh(financing_amount_log - legacy_continuous_ref)
                        continuous = legacy_continuous + funding_mask * financing_funding_gate * funding_delta
                else:
                    count_out = count_out + investor_mask * count_blend * (financing_count - count_out)
                    if self.funding_log_domain_enabled:
                        continuous = continuous + funding_mask * funding_blend * (financing_amount_log - continuous)
                    binary_probs = binary_probs + binary_mask * binary_blend * (financing_event_prob - binary_probs)
            outputs = {
                "continuous": continuous,
                "count": count_out,
                "binary": torch.logit(binary_probs),
                "event": event_logits,
                "uncertainty": torch.nn.functional.softplus(self.uncertainty_head(decoded)),
                "target_hist_last": target_hist[:, -1:],
            }
            if count_sparsity_gate is not None:
                outputs["count_sparsity_gate"] = count_sparsity_gate.expand(-1, self.horizon)
            if count_occurrence_logits is not None:
                outputs["count_occurrence_logits"] = count_occurrence_logits
            if count_occurrence_prob is not None:
                outputs["count_occurrence_prob"] = count_occurrence_prob
            if count_positive is not None:
                outputs["count_positive"] = count_positive
            if count_state_logits is not None:
                outputs["count_state_logits"] = count_state_logits
                outputs["count_state_bucket_values"] = self.lite_investors_head.delta_bucket_values
            if count_anchor_path is not None:
                outputs["count_anchor_path"] = count_anchor_path
            if count_profile_weights is not None:
                outputs["count_profile_weights"] = count_profile_weights
            if count_horizon_mix is not None:
                outputs["count_horizon_mix"] = count_horizon_mix
            if count_hurdle_blend is not None:
                outputs["count_hurdle_blend"] = count_hurdle_blend.expand(-1, self.horizon)
            if count_source_class_weights is not None:
                outputs["count_source_class_weights"] = count_source_class_weights
            if route_weights is not None:
                outputs["target_route_weights"] = route_weights
            if route_gate is not None:
                outputs["target_route_gate"] = route_gate
            if count_route_weights is not None:
                outputs["count_route_weights"] = count_route_weights
            if count_route_gate is not None:
                outputs["count_route_gate"] = count_route_gate
            if financing_event_logits is not None:
                outputs["financing_event_logits"] = financing_event_logits
            if financing_event_prob is not None:
                outputs["financing_event_prob"] = financing_event_prob
            if financing_count_positive is not None:
                outputs["financing_count_positive"] = financing_count_positive
            if financing_count is not None:
                outputs["financing_count"] = financing_count
            if financing_amount_positive_log is not None:
                outputs["financing_amount_positive_log"] = financing_amount_positive_log
            if financing_amount_log is not None:
                outputs["financing_amount_log"] = financing_amount_log
            if financing_breadth_log is not None:
                outputs["financing_breadth_log"] = financing_breadth_log
            if financing_breadth_anchor_log is not None:
                outputs["financing_breadth_anchor_log"] = financing_breadth_anchor_log
            if financing_intensity_log is not None:
                outputs["financing_intensity_log"] = financing_intensity_log
            if financing_amount_coupling is not None:
                outputs["financing_amount_coupling"] = financing_amount_coupling
            if financing_investor_gate is not None:
                outputs["financing_investor_gate"] = financing_investor_gate
            if financing_binary_gate is not None:
                outputs["financing_binary_gate"] = financing_binary_gate
            if financing_funding_gate is not None:
                outputs["financing_funding_gate"] = financing_funding_gate
            if self.financing_process_enabled:
                outputs["legacy_continuous"] = legacy_continuous
                outputs["legacy_count"] = legacy_count
                outputs["legacy_binary_prob"] = legacy_binary_prob
                if financing_investor_gate is not None or financing_binary_gate is not None or financing_funding_gate is not None:
                    applied_blend = (
                        investor_mask * (financing_investor_gate if financing_investor_gate is not None else 0.0)
                        + funding_mask * (financing_funding_gate if financing_funding_gate is not None else 0.0)
                        + binary_mask * (financing_binary_gate if financing_binary_gate is not None else 0.0)
                    )
                else:
                    applied_blend = (
                        investor_mask * count_blend
                        + funding_mask * funding_blend
                        + binary_mask * binary_blend
                    )
                outputs["financing_process_blend"] = applied_blend.expand(-1, self.horizon)
            return outputs

    return torch, nn, V740AlphaNet


class V740AlphaPrototypeWrapper(ModelBase):
    """Prototype wrapper for V740-alpha.

    This wrapper is intentionally pre-benchmark. It is useful for local
    prototyping and sanity checks, but is not yet part of the active registry.
    """

    _TASK_MAP = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
    _TARGET_MAP = {"funding_raised_usd": 0, "investors_count": 1, "is_funded": 2}
    _ABLATION_MAP = {
        "core_only": 0,
        "core_only_seed2": 1,
        "core_text": 2,
        "core_edgar": 3,
        "core_edgar_seed2": 4,
        "full": 5,
    }

    def __init__(
        self,
        input_size: int = 60,
        hidden_dim: int = 64,
        max_epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_covariates: int = 15,
        max_entities: int = 3000,
        max_windows: int = 50000,
        patience: int = 4,
        max_event_tokens: int = 4,
        enable_teacher_distill: bool = True,
        enable_event_head: bool = True,
        enable_task_modulation: bool = True,
        enable_selective_learning: bool = True,
        enable_distdf_loss: bool = True,
        enable_funding_log_domain: bool = True,
        enable_funding_source_scaling: bool = True,
        enable_funding_anchor: bool = True,
        funding_anchor_strength: float = 0.85,
        enable_target_routing: bool = True,
        target_route_experts: int = 3,
        enable_count_anchor: bool = True,
        count_anchor_strength: float = 0.70,
        enable_count_jump: bool = True,
        count_jump_strength: float = 0.30,
        enable_count_sparsity_gate: bool = True,
        count_sparsity_gate_strength: float = 0.75,
        enable_count_source_routing: bool = True,
        count_route_experts: int = 3,
        count_route_floor: float = 0.10,
        count_route_entropy_strength: float = 0.03,
        count_active_loss_strength: float = 0.08,
        enable_count_hurdle_head: bool = False,
        enable_count_source_specialists: bool = False,
        enable_v741_lite: bool = False,
        enable_window_repair: bool = False,
        min_window_history: int = 8,
        target_windows_per_entity: int = 6,
        enable_financing_consistency: bool = False,
        enable_financing_factorization: bool = False,
        enable_financing_guarded_phase: bool = False,
        enable_financing_evidence_residual: bool = False,
        financing_consistency_strength: float = 0.10,
        financing_auxiliary_strength: float = 0.12,
        financing_process_blend: float = 0.20,
        financing_investor_blend_scale: float = 1.0,
        financing_binary_blend_scale: float = 0.35,
        financing_funding_blend_scale: float = 0.20,
        financing_scaffold_strength: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        config = ModelConfig(
            name="V740AlphaPrototype",
            model_type="forecasting",
            params=kwargs,
            optional_dependency="torch",
        )
        super().__init__(config)
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_covariates = max_covariates
        self.max_entities = max_entities
        self.max_windows = max_windows
        self.patience = patience
        self.max_event_tokens = max_event_tokens
        self.enable_teacher_distill = enable_teacher_distill
        self.enable_event_head = enable_event_head
        self.enable_task_modulation = enable_task_modulation
        self.enable_selective_learning = enable_selective_learning
        self.enable_distdf_loss = enable_distdf_loss
        self.enable_funding_log_domain = enable_funding_log_domain
        self.enable_funding_source_scaling = enable_funding_source_scaling
        self.enable_funding_anchor = enable_funding_anchor
        self.funding_anchor_strength = max(0.0, float(funding_anchor_strength))
        self.enable_target_routing = enable_target_routing
        self.target_route_experts = max(1, int(target_route_experts))
        self.enable_count_anchor = enable_count_anchor
        self.count_anchor_strength = max(0.0, float(count_anchor_strength))
        self.enable_count_jump = enable_count_jump
        self.count_jump_strength = max(0.0, float(count_jump_strength))
        self.enable_count_sparsity_gate = enable_count_sparsity_gate
        self.count_sparsity_gate_strength = min(1.0, max(0.0, float(count_sparsity_gate_strength)))
        self.enable_count_source_routing = enable_count_source_routing
        self.count_route_experts = max(1, int(count_route_experts))
        self.count_route_floor = min(0.45, max(0.0, float(count_route_floor)))
        self.count_route_entropy_strength = max(0.0, float(count_route_entropy_strength))
        self.count_active_loss_strength = max(0.0, float(count_active_loss_strength))
        self.enable_count_hurdle_head = enable_count_hurdle_head
        self.enable_count_source_specialists = enable_count_source_specialists
        self.enable_v741_lite = enable_v741_lite
        self.enable_window_repair = enable_window_repair
        self.min_window_history = max(2, int(min_window_history))
        self.target_windows_per_entity = max(1, int(target_windows_per_entity))
        self.enable_financing_consistency = bool(enable_financing_consistency)
        self.enable_financing_factorization = bool(enable_financing_factorization)
        self.enable_financing_guarded_phase = bool(enable_financing_guarded_phase)
        self.enable_financing_evidence_residual = bool(enable_financing_evidence_residual)
        self.financing_consistency_strength = max(0.0, float(financing_consistency_strength))
        self.financing_auxiliary_strength = max(0.0, float(financing_auxiliary_strength))
        self.financing_process_blend = float(min(1.0, max(0.0, financing_process_blend)))
        self.financing_investor_blend_scale = max(0.0, float(financing_investor_blend_scale))
        self.financing_binary_blend_scale = max(0.0, float(financing_binary_blend_scale))
        self.financing_funding_blend_scale = max(0.0, float(financing_funding_blend_scale))
        self.financing_scaffold_strength = max(0.0, float(financing_scaffold_strength))
        self._effective_count_sparsity_gate_strength = self.count_sparsity_gate_strength
        self._effective_target_routing = bool(self.enable_target_routing)
        self._effective_count_source_routing = False
        self._effective_count_source_specialists = False
        self._effective_count_route_floor = 0.0
        self._effective_count_route_entropy_strength = 0.0
        self._effective_count_active_loss_strength = 0.0
        self._effective_financing_consistency = False
        self._effective_financing_factorization = False
        self._effective_financing_guarded_phase = False
        self._effective_financing_evidence_residual = False
        self._effective_financing_process_blend = 0.0
        self._effective_financing_scaffold_strength = 0.0
        self._effective_financing_target_scale = 1.0
        self.seed = seed
        self._network = None
        self._device = None
        self._contexts: Dict[str, np.ndarray] = {}
        self._context_memory: Dict[str, Dict[str, np.ndarray]] = {}
        self._feature_cols: List[str] = []
        self._edgar_cols: List[str] = []
        self._text_cols: List[str] = []
        self._source_state_exclude_cols: List[str] = []
        self._fallback_value = 0.0
        self._binary_target = False
        self._nonnegative_target = False
        self._binary_train_rate = 0.5
        self._binary_pos_weight = 1.0
        self._binary_rate_floor = 0.05
        self._binary_temperature = 1.0
        self._binary_logit_bias = 0.0
        self._binary_teacher_weight = 0.10
        self._binary_event_weight = 0.15
        self._teacher_logistic_mix = 0.4
        self._teacher_tree_mix = 0.6
        self._effective_task_modulation = enable_task_modulation
        self._binary_event_rate = 0.5
        self._binary_transition_rate = 0.5
        self._edgar_source_density = 0.0
        self._edgar_event_density = 0.0
        self._text_source_density = 0.0
        self._text_event_density = 0.0
        self._text_change_density = 0.0
        self._target_name = "funding_raised_usd"
        self._task_name = "task1_outcome"
        self._ablation_name = "core_only"
        self._horizon = 1
        self._funding_target = False
        self._funding_log_domain = False
        self._funding_source_scaling = False
        self._funding_anchor_enabled = False
        self._effective_funding_anchor_strength = 0.0
        self._funding_edgar_scale = 1.0
        self._funding_text_scale = 1.0
        self._target_route_stats: Dict[str, object] = {}
        self._count_route_stats: Dict[str, object] = {}
        self._count_specialist_stats: Dict[str, object] = {}
        self._window_stats: Dict[str, object] = {}
        self._count_source_specialist_names = ["no_source", "text_light", "edgar_rich", "full_rich"]
        self._v741_lite_delta_buckets = [-16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
        self._dual_clock_cfg = DualClockConfig(max_events=max_event_tokens)

    def _refresh_target_route_regime(self) -> None:
        if (
            self.enable_financing_evidence_residual
            and self.enable_financing_factorization
            and self._target_name == "investors_count"
        ):
            self._effective_target_routing = False
            return
        self._effective_target_routing = bool(self.enable_target_routing and not self.enable_v741_lite)

    def _count_delta_bucket_targets(self, torch, target: torch.Tensor, hist_last: torch.Tensor) -> torch.Tensor:
        bucket_values = torch.tensor(
            self._v741_lite_delta_buckets,
            dtype=target.dtype,
            device=target.device,
        )
        delta = target - hist_last.expand(-1, target.shape[1])
        distances = torch.abs(delta.unsqueeze(-1) - bucket_values.view(1, 1, -1))
        return distances.argmin(dim=-1)

    def _weighted_reduce(self, torch, values, sample_weight=None):
        if values.ndim > 1:
            dims = tuple(range(1, values.ndim))
            values = values.mean(dim=dims)
        if sample_weight is None:
            return values.mean()
        weight = sample_weight.to(device=values.device, dtype=values.dtype).view(-1)
        weight = weight / weight.mean().clamp_min(1e-6)
        return (values * weight).mean()

    def _financing_auxiliary_loss(self, torch, outputs, financing_targets=None, sample_weight=None):
        if not self._effective_financing_consistency or not financing_targets or not isinstance(outputs, dict):
            ref = outputs["count"] if isinstance(outputs, dict) and "count" in outputs else None
            if ref is None:
                return None
            return torch.zeros((), dtype=ref.dtype, device=ref.device)
        event_logits = outputs.get("financing_event_logits")
        event_prob = outputs.get("financing_event_prob")
        count_pred = outputs.get("financing_count")
        count_positive = outputs.get("financing_count_positive")
        amount_log = outputs.get("financing_amount_log")
        breadth_log = outputs.get("financing_breadth_log")
        intensity_log = outputs.get("financing_intensity_log")
        legacy_count = outputs.get("legacy_count")
        if event_logits is None or event_prob is None or count_pred is None or amount_log is None:
            ref = outputs["count"] if "count" in outputs else outputs.get("continuous")
            return torch.zeros((), dtype=ref.dtype, device=ref.device)

        binary_target = (financing_targets["is_funded"] > 0.5).to(event_logits.dtype)
        investors_target = financing_targets["investors_count"].clamp_min(0.0).to(event_logits.dtype)
        funding_target_log = torch.log1p(financing_targets["funding_raised_usd"].clamp_min(0.0)).to(event_logits.dtype)

        event_bce = self._weighted_reduce(
            torch,
            torch.nn.functional.binary_cross_entropy_with_logits(
                event_logits,
                binary_target,
                reduction="none",
            ),
            sample_weight=sample_weight,
        )
        count_aux = self._weighted_reduce(
            torch,
            torch.nn.functional.smooth_l1_loss(
                torch.log1p(count_pred.clamp_min(0.0)),
                torch.log1p(investors_target),
                reduction="none",
            ),
            sample_weight=sample_weight,
        )
        breadth_aux = torch.zeros((), dtype=event_logits.dtype, device=event_logits.device)
        if breadth_log is not None:
            breadth_aux = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(
                    breadth_log,
                    torch.log1p(investors_target),
                    reduction="none",
                ),
                sample_weight=sample_weight,
            )
        amount_aux = self._weighted_reduce(
            torch,
            torch.nn.functional.smooth_l1_loss(
                amount_log,
                funding_target_log,
                reduction="none",
            ),
            sample_weight=sample_weight,
        )
        intensity_aux = torch.zeros((), dtype=event_logits.dtype, device=event_logits.device)
        if intensity_log is not None:
            intensity_target_log = torch.relu(
                funding_target_log - 0.35 * torch.log1p(investors_target)
            )
            intensity_aux = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(
                    intensity_log,
                    intensity_target_log,
                    reduction="none",
                ),
                sample_weight=sample_weight,
            )

        count_presence = 1.0 - torch.exp(-count_pred.clamp_min(0.0))
        amount_presence = torch.sigmoid(amount_log - math.log1p(1.0))
        count_presence_gap = self._weighted_reduce(
            torch,
            (event_prob - count_presence) ** 2,
            sample_weight=sample_weight,
        )
        amount_presence_gap = self._weighted_reduce(
            torch,
            (event_prob - amount_presence) ** 2,
            sample_weight=sample_weight,
        )
        count_amount_gap = self._weighted_reduce(
            torch,
            (count_presence - amount_presence) ** 2,
            sample_weight=sample_weight,
        )
        coherence = self._weighted_reduce(
            torch,
            (event_prob - count_presence) ** 2
            + (event_prob - amount_presence) ** 2
            + 0.5 * (count_presence - amount_presence) ** 2,
            sample_weight=sample_weight,
        )
        inactive_count_penalty = self._weighted_reduce(
            torch,
            torch.log1p(count_pred.clamp_min(0.0)) * (1.0 - binary_target),
            sample_weight=sample_weight,
        )
        inactive_amount_penalty = self._weighted_reduce(
            torch,
            amount_log * (1.0 - binary_target),
            sample_weight=sample_weight,
        )
        inactive_penalty = self._weighted_reduce(
            torch,
            (
                torch.log1p(count_pred.clamp_min(0.0))
                + amount_log
            ) * (1.0 - binary_target),
            sample_weight=sample_weight,
        )
        trajectory = torch.zeros((), dtype=event_logits.dtype, device=event_logits.device)
        count_latent = count_positive if count_positive is not None else count_pred
        if count_latent.shape[1] > 1:
            count_step = torch.diff(torch.log1p(count_latent.clamp_min(0.0)), dim=1)
            amount_step = torch.diff(amount_log, dim=1)
            step_weight = 0.25 + binary_target[:, 1:]
            trajectory = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(
                    count_step,
                    amount_step,
                    reduction="none",
                ) * step_weight,
                sample_weight=sample_weight,
            )

        scaffold = torch.zeros((), dtype=event_logits.dtype, device=event_logits.device)
        if (
            self._effective_financing_scaffold_strength > 0.0
            and legacy_count is not None
            and self._target_name == "investors_count"
        ):
            scaffold = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(
                    torch.log1p(count_pred.clamp_min(0.0)),
                    torch.log1p(legacy_count.detach().clamp_min(0.0)),
                    reduction="none",
                ),
                sample_weight=sample_weight,
            )
        target_scale = float(self._effective_financing_target_scale)
        if self._effective_financing_guarded_phase:
            if self._target_name == "investors_count":
                auxiliary = target_scale * self.financing_auxiliary_strength * (
                    0.20 * event_bce
                    + 0.35 * count_aux
                    + 0.25 * breadth_aux
                    + 0.15 * amount_aux
                    + 0.05 * intensity_aux
                )
                consistency = target_scale * self.financing_consistency_strength * (
                    0.30 * count_presence_gap
                    + 0.10 * amount_presence_gap
                    + 0.15 * count_amount_gap
                    + 0.15 * inactive_count_penalty
                    + 0.30 * trajectory
                )
            elif self._target_name == "is_funded":
                auxiliary = target_scale * self.financing_auxiliary_strength * (
                    0.75 * event_bce
                    + 0.10 * count_aux
                    + 0.05 * amount_aux
                    + 0.05 * breadth_aux
                    + 0.05 * intensity_aux
                )
                consistency = target_scale * self.financing_consistency_strength * (
                    0.55 * count_presence_gap
                    + 0.25 * amount_presence_gap
                    + 0.10 * count_amount_gap
                    + 0.10 * inactive_penalty
                )
            else:
                auxiliary = target_scale * self.financing_auxiliary_strength * (
                    0.10 * event_bce
                    + 0.10 * count_aux
                    + 0.45 * amount_aux
                    + 0.10 * breadth_aux
                    + 0.25 * intensity_aux
                )
                consistency = target_scale * self.financing_consistency_strength * (
                    0.20 * count_presence_gap
                    + 0.35 * amount_presence_gap
                    + 0.15 * count_amount_gap
                    + 0.10 * inactive_amount_penalty
                    + 0.20 * trajectory
                )
        else:
            auxiliary = target_scale * self.financing_auxiliary_strength * (
                0.35 * event_bce
                + 0.20 * count_aux
                + 0.20 * amount_aux
                + 0.15 * breadth_aux
                + 0.10 * intensity_aux
            )
            consistency = target_scale * self.financing_consistency_strength * (
                0.45 * coherence + 0.35 * inactive_penalty + 0.20 * trajectory
            )
        return auxiliary + consistency + self._effective_financing_scaffold_strength * scaffold

    def _augment_with_financing_loss(self, torch, base_loss, outputs, financing_targets=None, sample_weight=None):
        extra = self._financing_auxiliary_loss(
            torch,
            outputs,
            financing_targets=financing_targets,
            sample_weight=sample_weight,
        )
        if extra is None:
            return base_loss
        return base_loss + extra

    def _build_window_sample_weights(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_edgar_recent: Optional[np.ndarray] = None,
        train_edgar_bucket: Optional[np.ndarray] = None,
        train_text_recent: Optional[np.ndarray] = None,
        train_text_bucket: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(train_x)
        weights = np.ones((n,), dtype=np.float32)
        if n == 0 or not self.enable_selective_learning:
            return weights

        target_mag = np.mean(np.abs(train_y), axis=1)
        hist_vol = np.std(train_x[:, 0, :], axis=1)

        def _normalize(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            if not np.isfinite(arr).any():
                return np.zeros_like(arr)
            lo = float(np.nanpercentile(arr, 5))
            hi = float(np.nanpercentile(arr, 95))
            if hi <= lo + 1e-8:
                return np.zeros_like(arr)
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        mag_norm = _normalize(target_mag)
        vol_norm = _normalize(hist_vol)

        weights = 1.0 + 0.10 * mag_norm - 0.20 * vol_norm
        horizon_factor = float(
            np.clip(np.log1p(max(1, self._horizon)) / np.log1p(365.0), 0.0, 1.0)
        )

        def _recent_activity(recent: Optional[np.ndarray]) -> np.ndarray:
            if recent is None or len(recent) == 0:
                return np.zeros((n,), dtype=np.float32)
            present = (recent[:, :, 2] > 0.0).astype(np.float32)
            active = present.any(axis=1).astype(np.float32)
            active_count = present.sum(axis=1).astype(np.float32)
            if float(active_count.max()) > 0.0:
                active_count = active_count / float(active_count.max())
            return 0.6 * active + 0.4 * active_count

        def _bucket_activity(bucket: Optional[np.ndarray]) -> np.ndarray:
            if bucket is None or len(bucket) == 0:
                return np.zeros((n,), dtype=np.float32)
            counts = bucket[:, :, 0].sum(axis=1).astype(np.float32)
            if float(counts.max()) > 0.0:
                counts = counts / float(counts.max())
            return counts

        def _recent_feature_mean(recent: Optional[np.ndarray], feature_idx: int) -> np.ndarray:
            if recent is None or len(recent) == 0 or recent.shape[2] <= feature_idx:
                return np.zeros((n,), dtype=np.float32)
            present = (recent[:, :, 2] > 0.0).astype(np.float32)
            denom = np.clip(present.sum(axis=1), 1.0, None)
            values = np.abs(recent[:, :, feature_idx]).astype(np.float32) * present
            return values.sum(axis=1) / denom

        edgar_activity = 0.7 * _recent_activity(train_edgar_recent) + 0.3 * _bucket_activity(train_edgar_bucket)
        text_activity = 0.7 * _recent_activity(train_text_recent) + 0.3 * _bucket_activity(train_text_bucket)
        text_novelty = np.clip(_recent_feature_mean(train_text_recent, 3), 0.0, 1.0)
        text_activity = text_activity * (0.5 + 0.5 * text_novelty)

        if self._binary_target:
            event_mask = (train_y.max(axis=1) > 0.5).astype(np.float32)
            state_now = (train_x[:, 0, -1] > 0.5).astype(np.float32)
            transition_mask = ((event_mask > 0.5) & (state_now < 0.5)).astype(np.float32)
            weights += 0.12 * event_mask + 0.10 * transition_mask
            weights += 0.05 * edgar_activity
            weights += 0.03 * text_activity
        else:
            target_delta = (
                np.mean(np.abs(np.diff(train_y, axis=1)), axis=1)
                if train_y.shape[1] > 1
                else np.zeros((n,), dtype=np.float32)
            )
            delta_norm = _normalize(target_delta)
            weights += 0.08 * delta_norm
            if self._funding_log_domain:
                last_level = _normalize(train_x[:, 0, -1])
                recent_active = np.mean(train_x[:, 0, :] > math.log1p(1.0), axis=1).astype(np.float32)
                future_active = np.mean(train_y > math.log1p(1.0), axis=1).astype(np.float32)
                weights += 0.10 * last_level + 0.06 * recent_active + 0.08 * future_active
            weights += (0.04 + 0.08 * horizon_factor) * edgar_activity
            weights += (0.02 + 0.05 * horizon_factor) * text_activity
        if self._has_edgar_path():
            weights += 0.04 * float(self._edgar_source_density)
        if self._has_text_path():
            weights += 0.02 * float(self._text_source_density)
        return np.clip(weights.astype(np.float32), 0.6, 1.3)

    def _token_tensor(self, torch, batch_size: int):
        task_idx = torch.full((batch_size,), self._TASK_MAP.get(self._task_name, 0), dtype=torch.long, device=self._device)
        target_idx = torch.full((batch_size,), self._TARGET_MAP.get(self._target_name, 0), dtype=torch.long, device=self._device)
        horizon_value = torch.full((batch_size,), float(max(1, self._horizon)), dtype=torch.float32, device=self._device)
        ablation_idx = torch.full((batch_size,), self._ABLATION_MAP.get(self._ablation_name, 0), dtype=torch.long, device=self._device)
        return task_idx, target_idx, horizon_value, ablation_idx

    def _has_text_path(self) -> bool:
        return self._ablation_name in {"core_text", "full"}

    def _has_edgar_path(self) -> bool:
        return self._ablation_name in {"core_edgar", "core_edgar_seed2", "full"}

    def _memory_batch(self, torch, entries: List[np.ndarray], default_shape: tuple[int, int]) -> torch.Tensor:
        if entries:
            arr = np.stack(entries).astype(np.float32, copy=False)
        else:
            arr = np.zeros((0, *default_shape), dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def _summarize_recent_memory(
        self,
        recent: Optional[np.ndarray],
        novelty_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        if recent is None or len(recent) == 0:
            return {"coverage": 0.0, "event_density": 0.0, "novelty": 0.0}
        arr = np.asarray(recent, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return {"coverage": 0.0, "event_density": 0.0, "novelty": 0.0}
        present = arr[:, :, 2] > 0.0
        coverage = float(present.any(axis=1).mean())
        event_density = float(present.sum(axis=1).mean() / max(arr.shape[1], 1))
        novelty = 0.0
        if novelty_idx is not None and arr.shape[2] > novelty_idx and present.any():
            novelty = float(np.clip(np.mean(np.abs(arr[:, :, novelty_idx][present])), 0.0, 1.0))
        return {
            "coverage": coverage,
            "event_density": event_density,
            "novelty": novelty,
        }

    def _configure_continuous_regime(self) -> None:
        self._funding_target = self._target_name == "funding_raised_usd" and not self._binary_target
        self._funding_log_domain = self._funding_target and self.enable_funding_log_domain
        self._funding_source_scaling = self._funding_target and self.enable_funding_source_scaling
        self._funding_anchor_enabled = (
            self._funding_target
            and self.enable_funding_anchor
            and self.funding_anchor_strength > 0.0
        )
        self._effective_funding_anchor_strength = (
            self.funding_anchor_strength if self._funding_anchor_enabled else 0.0
        )

    def _refresh_count_gate_regime(self) -> None:
        if self.enable_v741_lite and self._target_name == "investors_count":
            self._effective_count_sparsity_gate_strength = 0.0
            return
        if not self.enable_count_sparsity_gate and not self.enable_count_hurdle_head:
            self._effective_count_sparsity_gate_strength = 0.0
            return
        if self._target_name != "investors_count":
            self._effective_count_sparsity_gate_strength = (
                self.count_sparsity_gate_strength if self.enable_count_sparsity_gate else 0.0
            )
            return
        source_signal = float(
            np.clip(
                max(
                    self._edgar_source_density,
                    self._edgar_event_density,
                    0.5 * (self._text_source_density + self._text_event_density),
                ),
                0.0,
                1.0,
            )
        )
        if self.enable_count_hurdle_head:
            task_scale = 0.25
            if self._task_name == "task2_forecast":
                task_scale = 0.70
            elif self._task_name == "task3_risk_adjust":
                task_scale = 0.45
            source_scale = 0.35 + 0.65 * source_signal
            if self._has_edgar_path():
                source_scale = max(source_scale, 0.55)
            elif self._has_text_path():
                source_scale = max(source_scale, 0.40)
            self._effective_count_sparsity_gate_strength = float(
                np.clip(self.count_sparsity_gate_strength * task_scale * source_scale, 0.10, 0.90)
            )
            return
        if not self.enable_count_sparsity_gate:
            self._effective_count_sparsity_gate_strength = 0.0
            return
        if self._task_name != "task2_forecast":
            self._effective_count_sparsity_gate_strength = 0.0
            return
        source_scale = source_signal if self._has_edgar_path() else 0.0
        self._effective_count_sparsity_gate_strength = float(
            np.clip(self.count_sparsity_gate_strength * source_scale, 0.0, 1.0)
        )

    def _refresh_count_route_regime(self) -> None:
        if self.enable_v741_lite and self._target_name == "investors_count":
            self._effective_count_source_routing = False
            self._effective_count_route_floor = 0.0
            self._effective_count_route_entropy_strength = 0.0
            self._effective_count_active_loss_strength = 0.0
            return
        if (
            self.enable_financing_evidence_residual
            and self.enable_financing_factorization
            and self._target_name == "investors_count"
        ):
            self._effective_count_source_routing = False
            self._effective_count_route_floor = 0.0
            self._effective_count_route_entropy_strength = 0.0
            self._effective_count_active_loss_strength = 0.0
            return
        if self._target_name != "investors_count":
            self._effective_count_source_routing = False
            self._effective_count_route_floor = 0.0
            self._effective_count_route_entropy_strength = 0.0
            self._effective_count_active_loss_strength = 0.0
            return

        task_scale = 1.10
        if self._task_name == "task2_forecast":
            task_scale = 1.00
        elif self._task_name == "task3_risk_adjust":
            task_scale = 0.85
        self._effective_count_active_loss_strength = float(
            np.clip(self.count_active_loss_strength * task_scale, 0.0, 0.30)
        )

        if self.enable_count_hurdle_head:
            if not self.enable_count_source_routing or self.count_route_experts <= 1:
                self._effective_count_source_routing = False
                self._effective_count_route_floor = 0.0
                self._effective_count_route_entropy_strength = 0.0
                return
            source_signal = float(
                np.clip(
                    max(
                        self._edgar_source_density,
                        self._edgar_event_density,
                        0.5 * (self._text_source_density + self._text_event_density),
                    ),
                    0.0,
                    1.0,
                )
            )
            route_scale = 0.65 + 0.35 * (1.0 - source_signal)
            if self._has_edgar_path():
                route_scale = max(route_scale, 0.55)
            elif self._has_text_path():
                route_scale = max(route_scale, 0.45)
            self._effective_count_source_routing = True
            self._effective_count_route_floor = float(
                np.clip(self.count_route_floor * route_scale, 0.0, 0.35)
            )
            self._effective_count_route_entropy_strength = float(
                np.clip(self.count_route_entropy_strength * (0.75 + 0.25 * (1.0 - source_signal)), 0.0, 0.25)
            )
            self._effective_count_active_loss_strength = float(
                np.clip(self.count_active_loss_strength * (task_scale + 0.10), 0.0, 0.35)
            )
            return

        if not self.enable_count_source_routing or self.count_route_experts <= 1:
            self._effective_count_source_routing = False
            self._effective_count_route_floor = 0.0
            self._effective_count_route_entropy_strength = 0.0
            return

        source_signal = float(
            np.clip(
                max(
                    self._edgar_source_density,
                    self._edgar_event_density,
                ),
                0.0,
                1.0,
            )
        )
        has_source_path = self._has_edgar_path()
        if has_source_path and source_signal >= 0.75:
            self._effective_count_source_routing = False
            self._effective_count_route_floor = 0.0
            self._effective_count_route_entropy_strength = 0.0
            return

        route_scale = 1.0 if not has_source_path else float(np.clip(1.0 - 0.75 * source_signal, 0.25, 1.0))
        self._effective_count_source_routing = True
        self._effective_count_route_floor = float(
            np.clip(self.count_route_floor * route_scale, 0.0, 0.45)
        )
        self._effective_count_route_entropy_strength = float(
            np.clip(self.count_route_entropy_strength * route_scale, 0.0, 0.25)
        )

    def _refresh_count_specialist_regime(self) -> None:
        if (
            self.enable_financing_evidence_residual
            and self.enable_financing_factorization
            and self._target_name == "investors_count"
        ):
            self._effective_count_source_specialists = False
            return
        self._effective_count_source_specialists = bool(
            self.enable_count_source_specialists
            and self.enable_count_hurdle_head
            and self._target_name == "investors_count"
            and not self.enable_v741_lite
        )

    def _refresh_financing_consistency_regime(self, train_raw: Optional[pd.DataFrame] = None) -> None:
        has_all_targets = True
        if train_raw is not None:
            has_all_targets = all(col in train_raw.columns for col in _FINANCING_TARGETS)
        self._effective_financing_consistency = bool(
            self.enable_financing_consistency
            and has_all_targets
            and self._target_name in _FINANCING_TARGETS
        )
        self._effective_financing_factorization = bool(
            self._effective_financing_consistency and self.enable_financing_factorization
        )
        self._effective_financing_guarded_phase = bool(
            self._effective_financing_factorization and self.enable_financing_guarded_phase
        )
        self._effective_financing_evidence_residual = bool(
            self._effective_financing_factorization and self.enable_financing_evidence_residual
        )
        self._effective_financing_process_blend = (
            float(self.financing_process_blend) if self._effective_financing_consistency else 0.0
        )
        self._effective_financing_scaffold_strength = (
            float(self.financing_scaffold_strength) if self._effective_financing_factorization else 0.0
        )
        if self._effective_financing_evidence_residual:
            if self._target_name == "investors_count":
                self._effective_financing_target_scale = 1.0
            else:
                self._effective_financing_target_scale = 0.0
        elif self._effective_financing_guarded_phase:
            if self._target_name == "investors_count":
                self._effective_financing_target_scale = 0.90
            elif self._target_name == "funding_raised_usd":
                self._effective_financing_target_scale = 0.30
            else:
                self._effective_financing_target_scale = 0.15
        elif self._effective_financing_factorization:
            if self._target_name == "investors_count":
                self._effective_financing_target_scale = 1.0
            elif self._target_name == "funding_raised_usd":
                self._effective_financing_target_scale = 0.45
            else:
                self._effective_financing_target_scale = 0.35
        else:
            self._effective_financing_target_scale = 1.0

    def get_regime_info(self) -> Dict[str, object]:
        count_source_signal = float(
            np.clip(max(self._edgar_source_density, self._edgar_event_density), 0.0, 1.0)
        )
        return {
            "task": self._task_name,
            "target": self._target_name,
            "ablation": self._ablation_name,
            "horizon": int(self._horizon),
            "binary_target": bool(self._binary_target),
            "funding_target": bool(self._funding_target),
            "teacher_distill_enabled": bool(self.enable_teacher_distill),
            "event_head_enabled": bool(self.enable_event_head),
            "task_mod_enabled": bool(self._effective_task_modulation),
            "requested": {
                "funding_log_domain": bool(self.enable_funding_log_domain),
                "funding_source_scaling": bool(self.enable_funding_source_scaling),
                "funding_anchor": bool(self.enable_funding_anchor),
                "funding_anchor_strength": float(self.funding_anchor_strength),
            },
            "effective": {
                "funding_log_domain": bool(self._funding_log_domain),
                "funding_source_scaling": bool(self._funding_source_scaling),
                "funding_anchor": bool(self._funding_anchor_enabled),
                "funding_anchor_strength": float(self._effective_funding_anchor_strength),
            },
            "source_scales": {
                "edgar": float(self._funding_edgar_scale),
                "text": float(self._funding_text_scale),
            },
            "source_stats": {
                "row_density": {
                    "edgar": float(self._edgar_source_density),
                    "text": float(self._text_source_density),
                },
                "event_density": {
                    "edgar": float(self._edgar_event_density),
                    "text": float(self._text_event_density),
                },
                "text_change_density": float(self._text_change_density),
            },
            "state_stream": {
                "feature_cols": len(self._feature_cols),
                "excluded_source_cols": len(self._source_state_exclude_cols),
            },
            "target_routing": {
                "enabled": bool(self.enable_target_routing),
                "effective_enabled": bool(self._effective_target_routing),
                "experts": int(self.target_route_experts),
                "count_head_type": (
                    "evidence_financing_residual" if self._effective_financing_evidence_residual
                    else ("guarded_financing_phase_residual" if self._effective_financing_guarded_phase
                    else (
                    "shared_financing_process_blend" if self._effective_financing_consistency
                    else (
                    "typed_state_delta" if self.enable_v741_lite and self._target_name == "investors_count"
                    else ("hurdle_zero_inflated" if self.enable_count_hurdle_head else "softplus_regression")
                    )
                    ))
                ),
                "count_anchor_enabled": bool(self.enable_count_anchor),
                "count_anchor_strength": float(self.count_anchor_strength),
                "count_jump_enabled": bool(self.enable_count_jump),
                "count_jump_strength": float(self.count_jump_strength),
                "count_sparsity_gate_enabled": bool(self.enable_count_sparsity_gate),
                "count_sparsity_gate_strength": float(self.count_sparsity_gate_strength),
                "effective_count_sparsity_gate_strength": float(self._effective_count_sparsity_gate_strength),
                "count_positive_transform": "softplus",
                "count_sparsity_gate_type": (
                    "learned_occurrence_with_history_prior_task2_only"
                    if self.enable_count_sparsity_gate
                    else "disabled"
                ),
                "count_sparsity_gate_source_signal": count_source_signal,
                "stats": self._target_route_stats,
            },
            "count_source_routing": {
                "enabled": bool(self.enable_count_source_routing),
                "effective_enabled": bool(self._effective_count_source_routing),
                "experts": int(self.count_route_experts),
                "requested_route_floor": float(self.count_route_floor),
                "effective_route_floor": float(self._effective_count_route_floor),
                "requested_entropy_strength": float(self.count_route_entropy_strength),
                "effective_entropy_strength": float(self._effective_count_route_entropy_strength),
                "requested_active_loss_strength": float(self.count_active_loss_strength),
                "effective_active_loss_strength": float(self._effective_count_active_loss_strength),
                "source_signal": count_source_signal,
                "stats": self._count_route_stats,
            },
            "count_source_specialists": {
                "enabled": bool(self.enable_count_source_specialists),
                "effective_enabled": bool(self._effective_count_source_specialists),
                "class_names": list(self._count_source_specialist_names),
                "stats": self._count_specialist_stats,
            },
            "financing_process": {
                "enabled": bool(self.enable_financing_consistency),
                "effective_enabled": bool(self._effective_financing_consistency),
                "factorized_state_enabled": bool(self._effective_financing_factorization),
                "guarded_phase_enabled": bool(self._effective_financing_guarded_phase),
                "evidence_residual_enabled": bool(self._effective_financing_evidence_residual),
                "consistency_strength": float(self.financing_consistency_strength),
                "auxiliary_strength": float(self.financing_auxiliary_strength),
                "requested_blend": float(self.financing_process_blend),
                "effective_blend": float(self._effective_financing_process_blend),
                "investor_blend_scale": float(self.financing_investor_blend_scale),
                "binary_blend_scale": float(self.financing_binary_blend_scale),
                "funding_blend_scale": float(self.financing_funding_blend_scale),
                "target_loss_scale": float(self._effective_financing_target_scale),
                "scaffold_strength": float(self._effective_financing_scaffold_strength),
                "shared_targets": list(_FINANCING_TARGETS),
                "single_path_bias": not bool(self.enable_target_routing),
            },
            "single_model_lite": {
                "enabled": bool(self.enable_v741_lite),
                "delta_buckets": list(self._v741_lite_delta_buckets),
                "target_head_types": {
                    "is_funded": "calibrated_logit",
                    "funding_raised_usd": "anchor_residual",
                    "investors_count": "typed_state_delta" if self.enable_v741_lite else "legacy_count_path",
                },
            },
            "windowing": {
                "repair_enabled": bool(self.enable_window_repair),
                "min_window_history": int(self.min_window_history),
                "target_windows_per_entity": int(self.target_windows_per_entity),
                "stats": self._window_stats,
            },
        }

    def _summarize_route_stats(self, route_weights, route_gate=None) -> Dict[str, object]:
        if route_weights is None:
            return {}
        route_np = route_weights.detach().cpu().numpy()
        if route_np.size == 0:
            return {}
        route_mean = route_np.mean(axis=0)
        top_idx = route_np.argmax(axis=1)
        top_counts = np.bincount(top_idx, minlength=route_np.shape[1])
        dominant = int(route_mean.argmax())
        stats: Dict[str, object] = {
            "mean_weights": [float(x) for x in route_mean.tolist()],
            "dominant_expert": dominant,
            "dominant_share": float(top_counts[dominant] / max(len(top_idx), 1)),
            "entropy": float(-np.mean(np.sum(route_np * np.log(np.clip(route_np, 1e-8, 1.0)), axis=1))),
        }
        if route_gate is not None:
            stats["mean_gate"] = float(route_gate.detach().mean().cpu())
        return stats

    def _update_target_route_stats(self, outputs) -> None:
        route_weights = outputs.get("target_route_weights") if isinstance(outputs, dict) else None
        if route_weights is None:
            self._target_route_stats = {}
            return
        self._target_route_stats = self._summarize_route_stats(
            route_weights,
            outputs.get("target_route_gate"),
        )

    def _update_count_route_stats(self, outputs) -> None:
        route_weights = outputs.get("count_route_weights") if isinstance(outputs, dict) else None
        if route_weights is None:
            self._count_route_stats = {}
            return
        self._count_route_stats = self._summarize_route_stats(
            route_weights,
            outputs.get("count_route_gate"),
        )

    def _update_count_specialist_stats(self, outputs) -> None:
        specialist_weights = outputs.get("count_source_class_weights") if isinstance(outputs, dict) else None
        if specialist_weights is None:
            self._count_specialist_stats = {}
            return
        stats = self._summarize_route_stats(specialist_weights)
        if not stats:
            self._count_specialist_stats = {}
            return
        mean_weights = stats.get("mean_weights", [])
        dominant_idx = int(stats.get("dominant_expert", 0))
        dominant_class = (
            self._count_source_specialist_names[dominant_idx]
            if 0 <= dominant_idx < len(self._count_source_specialist_names)
            else str(dominant_idx)
        )
        self._count_specialist_stats = {
            "mean_class_weights": {
                name: float(mean_weights[idx]) if idx < len(mean_weights) else 0.0
                for idx, name in enumerate(self._count_source_specialist_names)
            },
            "dominant_class": dominant_class,
            "dominant_share": float(stats.get("dominant_share", 0.0)),
            "entropy": float(stats.get("entropy", 0.0)),
        }

    def _refresh_source_scales(self) -> None:
        edgar_scale = 1.0
        text_scale = 1.0
        edgar_signal = max(float(self._edgar_source_density), float(self._edgar_event_density))
        text_signal = 0.5 * float(self._text_source_density) + 0.5 * float(self._text_event_density)
        text_change = float(np.clip(self._text_change_density, 0.0, 1.0))
        if self._funding_source_scaling:
            if self._has_edgar_path():
                edgar_scale = 1.03 + 0.12 * edgar_signal
                if self._horizon >= 14:
                    edgar_scale += 0.05
                if self._horizon >= 30:
                    edgar_scale += 0.05
            if self._has_text_path():
                text_scale = 0.35 if self._ablation_name == "core_text" else 0.22
                if self._horizon >= 14:
                    text_scale *= 0.60
                if self._horizon >= 30:
                    text_scale *= 0.60
                text_scale *= 1.0 - 0.25 * text_signal
                text_scale *= 0.35 + 0.65 * text_change
                text_scale *= 0.80 + 0.20 * text_signal
        self._funding_edgar_scale = float(np.clip(edgar_scale, 0.75, 1.25))
        self._funding_text_scale = float(np.clip(text_scale, 0.05, 1.0))

    def _transform_continuous_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self._funding_log_domain:
            return np.log1p(np.clip(arr, 0.0, None)).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _inverse_continuous_array(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if self._funding_log_domain:
            return np.expm1(np.clip(arr, 0.0, 25.0)).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def _apply_source_tensor_scales(self, edgar_recent, edgar_bucket, text_recent, text_bucket):
        if self._funding_source_scaling:
            if edgar_recent is not None:
                edgar_recent = edgar_recent * self._funding_edgar_scale
            if edgar_bucket is not None:
                edgar_bucket = edgar_bucket * self._funding_edgar_scale
            if text_recent is not None:
                text_recent = text_recent * self._funding_text_scale
            if text_bucket is not None:
                text_bucket = text_bucket * self._funding_text_scale
        return edgar_recent, edgar_bucket, text_recent, text_bucket

    def _configure_binary_regime(self) -> None:
        teacher_weight = 0.08
        event_weight = 0.12
        logistic_mix = 0.45
        tree_mix = 0.55
        edgar_signal = max(float(self._edgar_source_density), float(self._edgar_event_density))
        text_signal = 0.5 * float(self._text_source_density) + 0.5 * float(self._text_event_density)
        text_signal *= 0.5 + 0.5 * float(np.clip(self._text_change_density, 0.0, 1.0))

        pos_balance = 1.0 - abs(self._binary_train_rate - 0.5) / 0.5
        event_sparsity = 1.0 - self._binary_event_rate
        transition_sparsity = 1.0 - self._binary_transition_rate

        teacher_weight += 0.03 * event_sparsity
        teacher_weight += 0.02 * transition_sparsity
        teacher_weight += 0.04 * edgar_signal
        teacher_weight -= 0.03 * text_signal
        teacher_weight += 0.01 * pos_balance

        event_weight += 0.04 * transition_sparsity
        event_weight += 0.02 * edgar_signal
        event_weight -= 0.03 * text_signal
        event_weight += 0.01 * pos_balance

        logistic_mix += 0.25 * text_signal
        logistic_mix += 0.10 * pos_balance
        logistic_mix -= 0.20 * edgar_signal
        logistic_mix = float(np.clip(logistic_mix, 0.20, 0.80))
        tree_mix = 1.0 - logistic_mix

        if self._task_name == "task2_forecast":
            teacher_weight *= 0.95
        elif self._task_name == "task3_risk_adjust":
            teacher_weight *= 0.90
            event_weight *= 0.90

        if not self.enable_teacher_distill:
            teacher_weight = 0.0
        if not self.enable_event_head:
            event_weight = 0.0

        self._binary_teacher_weight = float(np.clip(teacher_weight, 0.0, 0.30))
        self._binary_event_weight = float(np.clip(event_weight, 0.0, 0.30))
        self._teacher_logistic_mix = float(logistic_mix)
        self._teacher_tree_mix = float(tree_mix)

    def _loss(self, torch, outputs, target, teacher_probs=None, sample_weight=None, financing_targets=None):
        if self._binary_target:
            logits = outputs["binary"]
            probs = torch.sigmoid(logits)
            pos_weight = torch.tensor(
                self._binary_pos_weight,
                dtype=target.dtype,
                device=target.device,
            )
            bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight,
                reduction="none",
            )
            pt = probs * target + (1.0 - probs) * (1.0 - target)
            focal = ((1.0 - pt).clamp_min(1e-4) ** 1.5) * bce_raw
            bce = self._weighted_reduce(torch, focal, sample_weight=sample_weight)
            brier = self._weighted_reduce(torch, (probs - target) ** 2, sample_weight=sample_weight)
            rate_ref = torch.tensor(
                self._binary_train_rate,
                dtype=target.dtype,
                device=target.device,
            )
            rate_penalty = torch.nn.functional.smooth_l1_loss(
                probs.mean(dim=0),
                torch.full_like(probs.mean(dim=0), rate_ref),
            )
            std_penalty = torch.relu(self._binary_rate_floor - probs.std())
            flat_target = target.reshape(-1)
            flat_probs = probs.reshape(-1)
            if bool((flat_target > 0.5).any()) and bool((flat_target < 0.5).any()):
                pos_mean = flat_probs[flat_target > 0.5].mean()
                neg_mean = flat_probs[flat_target < 0.5].mean()
                separation = torch.relu(0.15 - (pos_mean - neg_mean))
            else:
                separation = torch.zeros((), dtype=target.dtype, device=target.device)
            event_bce = torch.zeros((), dtype=target.dtype, device=target.device)
            if self.enable_event_head:
                event_target = (target.max(dim=1).values > 0.5).to(target.dtype)
                event_logits = outputs["event"]
                event_bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    event_logits,
                    event_target,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                event_bce = self._weighted_reduce(torch, event_bce_raw, sample_weight=sample_weight)
            teacher_align = torch.zeros((), dtype=target.dtype, device=target.device)
            if teacher_probs is not None and self.enable_teacher_distill:
                teacher_probs = teacher_probs.to(device=target.device, dtype=target.dtype).view(-1)
                teacher_probs = teacher_probs.clamp(1e-4, 1.0 - 1e-4)
                avg_probs = probs.mean(dim=1)
                teacher_align = self._weighted_reduce(
                    torch,
                    (avg_probs - teacher_probs) ** 2,
                    sample_weight=sample_weight,
                )
                if self.enable_event_head:
                    event_probs = torch.sigmoid(outputs["event"])
                    teacher_align = teacher_align + 0.5 * self._weighted_reduce(
                        torch,
                        (event_probs - teacher_probs) ** 2,
                        sample_weight=sample_weight,
                    )
            primary_loss = (
                bce
                + 0.15 * brier
                + 0.10 * rate_penalty
                + 0.05 * std_penalty
                + 0.10 * separation
                + self._binary_event_weight * event_bce
                + self._binary_teacher_weight * teacher_align
            )
            return self._augment_with_financing_loss(
                torch,
                primary_loss,
                outputs,
                financing_targets=financing_targets,
                sample_weight=sample_weight,
            )
        if self._target_name == "investors_count":
            pred = outputs["count"].clamp_min(0.0)
            gate_strength_ratio = float(
                np.clip(
                    self._effective_count_sparsity_gate_strength / max(self.count_sparsity_gate_strength, 1e-6),
                    0.0,
                    1.0,
                )
            )
            occurrence_target = (target > 0.5).to(target.dtype)
            occurrence_logits = outputs.get("count_occurrence_logits") if isinstance(outputs, dict) else None
            occurrence_bce = torch.zeros((), dtype=target.dtype, device=target.device)
            if occurrence_logits is not None:
                occurrence_rate = occurrence_target.mean().clamp(1e-4, 1.0 - 1e-4)
                occurrence_pos_weight = ((1.0 - occurrence_rate) / occurrence_rate).clamp(1.0, 25.0)
                occurrence_bce = self._weighted_reduce(
                    torch,
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        occurrence_logits,
                        occurrence_target,
                        pos_weight=occurrence_pos_weight,
                        reduction="none",
                    ),
                    sample_weight=sample_weight,
                )
            if self.enable_count_hurdle_head:
                count_positive = outputs.get("count_positive") if isinstance(outputs, dict) else None
                if count_positive is None:
                    count_positive = pred
                base = self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred, target, reduction="none"),
                    sample_weight=sample_weight,
                )
                sparse_over = self._weighted_reduce(
                    torch,
                    torch.relu(pred - target) / (1.0 + torch.log1p(torch.relu(target))),
                    sample_weight=sample_weight,
                )
                active_mass = occurrence_target.mean().clamp_min(1e-4)
                positive_log_mae = self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(
                        torch.log1p(count_positive),
                        torch.log1p(target.clamp_min(0.0)),
                        reduction="none",
                    ) * occurrence_target,
                    sample_weight=sample_weight,
                ) / active_mass
                inactive_penalty = self._weighted_reduce(
                    torch,
                    pred * (1.0 - occurrence_target),
                    sample_weight=sample_weight,
                )
                quiet_mask = (target.max(dim=1, keepdim=True).values < 0.5).to(target.dtype)
                quiet_penalty = self._weighted_reduce(
                    torch,
                    pred * quiet_mask,
                    sample_weight=sample_weight,
                )
                entry_jump = torch.zeros((), dtype=target.dtype, device=target.device)
                hist_last = outputs.get("target_hist_last") if isinstance(outputs, dict) else None
                if hist_last is not None:
                    pred_entry = pred[:, :1] - hist_last
                    tgt_entry = target[:, :1] - hist_last
                    entry_weight = 1.0 + torch.log1p(torch.abs(tgt_entry))
                    entry_jump = self._weighted_reduce(
                        torch,
                        torch.nn.functional.smooth_l1_loss(pred_entry, tgt_entry, reduction="none") * entry_weight,
                        sample_weight=sample_weight,
                    )
                route_entropy_penalty = torch.zeros((), dtype=target.dtype, device=target.device)
                route_weights = outputs.get("count_route_weights") if isinstance(outputs, dict) else None
                if (
                    route_weights is not None
                    and route_weights.shape[1] > 1
                    and self._effective_count_route_entropy_strength > 0.0
                ):
                    entropy = -torch.sum(route_weights * torch.log(route_weights.clamp_min(1e-6)), dim=1)
                    entropy_floor = 0.60 * math.log(route_weights.shape[1])
                    route_entropy_penalty = self._weighted_reduce(
                        torch,
                        torch.relu(torch.full_like(entropy, entropy_floor) - entropy),
                        sample_weight=sample_weight,
                    )
                distdf = torch.zeros((), dtype=target.dtype, device=target.device)
                if self.enable_distdf_loss:
                    pred_mean = pred.mean(dim=1)
                    tgt_mean = target.mean(dim=1)
                    pred_std = pred.std(dim=1, unbiased=False)
                    tgt_std = target.std(dim=1, unbiased=False)
                    pred_step = pred[:, 1:] - pred[:, :-1] if pred.shape[1] > 1 else pred[:, :1] * 0.0
                    tgt_step = target[:, 1:] - target[:, :-1] if target.shape[1] > 1 else target[:, :1] * 0.0
                    distdf = self._weighted_reduce(
                        torch,
                        torch.nn.functional.smooth_l1_loss(pred_mean, tgt_mean, reduction="none")
                        + 0.5 * torch.nn.functional.smooth_l1_loss(
                            torch.log1p(pred_std),
                            torch.log1p(tgt_std),
                            reduction="none",
                        ),
                        sample_weight=sample_weight,
                    ) + 0.25 * self._weighted_reduce(
                        torch,
                        torch.nn.functional.smooth_l1_loss(pred_step, tgt_step, reduction="none"),
                        sample_weight=sample_weight,
                    )
                primary_loss = (
                    base
                    + (0.10 + 0.08 * gate_strength_ratio) * occurrence_bce
                    + 0.20 * positive_log_mae
                    + 0.08 * sparse_over
                    + 0.10 * inactive_penalty
                    + 0.10 * quiet_penalty
                    + 0.08 * entry_jump
                    + 0.05 * distdf
                    + self._effective_count_route_entropy_strength * route_entropy_penalty
                )
                return self._augment_with_financing_loss(
                    torch,
                    primary_loss,
                    outputs,
                    financing_targets=financing_targets,
                    sample_weight=sample_weight,
                )
            base = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred, target, reduction="none"),
                sample_weight=sample_weight,
            )
            sparse_over = self._weighted_reduce(
                torch,
                torch.relu(pred - target) / (1.0 + torch.log1p(torch.relu(target))),
                sample_weight=sample_weight,
            )
            quiet_mask = (target.max(dim=1, keepdim=True).values < 0.5).to(target.dtype)
            quiet_penalty = self._weighted_reduce(
                torch,
                pred * quiet_mask,
                sample_weight=sample_weight,
            )
            gate_penalty = torch.zeros((), dtype=target.dtype, device=target.device)
            gate = outputs.get("count_sparsity_gate") if isinstance(outputs, dict) else None
            if gate is not None:
                target_activity = (target > 0.5).to(target.dtype).mean(dim=1, keepdim=True)
                gate_penalty = self._weighted_reduce(
                    torch,
                    (gate.mean(dim=1, keepdim=True) - target_activity) ** 2,
                    sample_weight=sample_weight,
                )
            active_log_mae = torch.zeros((), dtype=target.dtype, device=target.device)
            if self._effective_count_active_loss_strength > 0.0 and bool((occurrence_target > 0.5).any()):
                active_log_loss = torch.nn.functional.smooth_l1_loss(
                    torch.log1p(pred),
                    torch.log1p(target.clamp_min(0.0)),
                    reduction="none",
                ) * occurrence_target
                active_mass = occurrence_target.mean().clamp_min(1e-4)
                active_log_mae = self._weighted_reduce(
                    torch,
                    active_log_loss,
                    sample_weight=sample_weight,
                ) / active_mass
            entry_jump = torch.zeros((), dtype=target.dtype, device=target.device)
            hist_last = outputs.get("target_hist_last") if isinstance(outputs, dict) else None
            if hist_last is not None:
                pred_entry = pred[:, :1] - hist_last
                tgt_entry = target[:, :1] - hist_last
                entry_weight = 1.0 + torch.log1p(torch.abs(tgt_entry))
                entry_jump = self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_entry, tgt_entry, reduction="none") * entry_weight,
                    sample_weight=sample_weight,
                )
            distdf = torch.zeros((), dtype=target.dtype, device=target.device)
            if self.enable_distdf_loss:
                pred_mean = pred.mean(dim=1)
                tgt_mean = target.mean(dim=1)
                pred_std = pred.std(dim=1, unbiased=False)
                tgt_std = target.std(dim=1, unbiased=False)
                pred_last = pred[:, -1]
                tgt_last = target[:, -1]
                pred_step = pred[:, 1:] - pred[:, :-1] if pred.shape[1] > 1 else pred[:, :1] * 0.0
                tgt_step = target[:, 1:] - target[:, :-1] if target.shape[1] > 1 else target[:, :1] * 0.0
                distdf = self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_mean, tgt_mean, reduction="none")
                    + 0.5 * torch.nn.functional.smooth_l1_loss(
                        torch.log1p(pred_std),
                        torch.log1p(tgt_std),
                        reduction="none",
                    ),
                    sample_weight=sample_weight,
                ) + 0.5 * self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_last, tgt_last, reduction="none"),
                    sample_weight=sample_weight,
                ) + 0.25 * self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_step, tgt_step, reduction="none"),
                    sample_weight=sample_weight,
                )
            route_entropy_penalty = torch.zeros((), dtype=target.dtype, device=target.device)
            route_weights = outputs.get("count_route_weights") if isinstance(outputs, dict) else None
            if (
                route_weights is not None
                and route_weights.shape[1] > 1
                and self._effective_count_route_entropy_strength > 0.0
            ):
                entropy = -torch.sum(route_weights * torch.log(route_weights.clamp_min(1e-6)), dim=1)
                entropy_floor = 0.60 * math.log(route_weights.shape[1])
                route_entropy_penalty = self._weighted_reduce(
                    torch,
                    torch.relu(torch.full_like(entropy, entropy_floor) - entropy),
                    sample_weight=sample_weight,
                )
            primary_loss = (
                base
                + (0.06 * gate_strength_ratio) * occurrence_bce
                + self._effective_count_active_loss_strength * active_log_mae
                + 0.08 * entry_jump
                + 0.08 * sparse_over
                + 0.08 * quiet_penalty
                + 0.05 * distdf
                + (0.04 * gate_strength_ratio) * gate_penalty
                + self._effective_count_route_entropy_strength * route_entropy_penalty
            )
            return self._augment_with_financing_loss(
                torch,
                primary_loss,
                outputs,
                financing_targets=financing_targets,
                sample_weight=sample_weight,
            )
        pred = outputs["continuous"]
        if self._funding_log_domain:
            base = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred, target, reduction="none"),
                sample_weight=sample_weight,
            )
            pred_pos = torch.clamp(pred, min=0.0)
            target_pos = torch.clamp(target, min=0.0)
            floor_penalty = self._weighted_reduce(torch, torch.relu(-pred), sample_weight=sample_weight)
            last_align = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred_pos[:, -1], target_pos[:, -1], reduction="none"),
                sample_weight=sample_weight,
            )
            distdf = torch.zeros((), dtype=target.dtype, device=target.device)
            if self.enable_distdf_loss:
                pred_mean = pred_pos.mean(dim=1)
                tgt_mean = target_pos.mean(dim=1)
                pred_std = pred_pos.std(dim=1, unbiased=False).clamp_min(1e-6)
                tgt_std = target_pos.std(dim=1, unbiased=False).clamp_min(1e-6)
                pred_step = pred_pos[:, 1:] - pred_pos[:, :-1] if pred_pos.shape[1] > 1 else pred_pos[:, :1] * 0.0
                tgt_step = target_pos[:, 1:] - target_pos[:, :-1] if target_pos.shape[1] > 1 else target_pos[:, :1] * 0.0
                distdf = self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_mean, tgt_mean, reduction="none")
                    + 0.5 * torch.nn.functional.smooth_l1_loss(
                        torch.log1p(pred_std),
                        torch.log1p(tgt_std),
                        reduction="none",
                    ),
                    sample_weight=sample_weight,
                ) + 0.25 * self._weighted_reduce(
                    torch,
                    torch.nn.functional.smooth_l1_loss(pred_step, tgt_step, reduction="none"),
                    sample_weight=sample_weight,
                )
            primary_loss = base + 0.08 * last_align + 0.05 * floor_penalty + 0.05 * distdf
            return self._augment_with_financing_loss(
                torch,
                primary_loss,
                outputs,
                financing_targets=financing_targets,
                sample_weight=sample_weight,
            )
        base = self._weighted_reduce(
            torch,
            torch.nn.functional.smooth_l1_loss(pred, target, reduction="none"),
            sample_weight=sample_weight,
        )
        pred_pos = torch.relu(pred)
        target_pos = torch.relu(target)
        align = self._weighted_reduce(
            torch,
            torch.nn.functional.smooth_l1_loss(
                torch.log1p(pred_pos),
                torch.log1p(target_pos),
                reduction="none",
            ),
            sample_weight=sample_weight,
        )
        distdf = torch.zeros((), dtype=target.dtype, device=target.device)
        if self.enable_distdf_loss:
            pred_mean = pred.mean(dim=1)
            tgt_mean = target.mean(dim=1)
            pred_std = pred.std(dim=1, unbiased=False).clamp_min(1e-6)
            tgt_std = target.std(dim=1, unbiased=False).clamp_min(1e-6)
            pred_last = pred[:, -1]
            tgt_last = target[:, -1]
            pred_step = pred[:, 1:] - pred[:, :-1] if pred.shape[1] > 1 else pred[:, :1] * 0.0
            tgt_step = target[:, 1:] - target[:, :-1] if target.shape[1] > 1 else target[:, :1] * 0.0
            distdf = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred_mean, tgt_mean, reduction="none")
                + 0.5 * torch.nn.functional.smooth_l1_loss(
                    torch.log1p(pred_std),
                    torch.log1p(tgt_std),
                    reduction="none",
                ),
                sample_weight=sample_weight,
            ) + 0.5 * self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred_last, tgt_last, reduction="none"),
                sample_weight=sample_weight,
            ) + 0.25 * self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred_step, tgt_step, reduction="none"),
                sample_weight=sample_weight,
            )
        primary_loss = base + 0.1 * align + 0.05 * distdf
        return self._augment_with_financing_loss(
            torch,
            primary_loss,
            outputs,
            financing_targets=financing_targets,
            sample_weight=sample_weight,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "V740AlphaPrototypeWrapper":
        torch, nn, V740AlphaNet = _torch_imports()

        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_raw = kwargs.get("train_raw")
        self._target_name = kwargs.get("target", y.name or "funding_raised_usd")
        self._task_name = kwargs.get("task", "task1_outcome")
        self._ablation_name = kwargs.get("ablation", "core_only")
        self._horizon = int(kwargs.get("horizon", 7))

        y_arr = np.asarray(y.values, dtype=np.float32)
        finite = y_arr[np.isfinite(y_arr)]
        self._fallback_value = float(np.nanmedian(finite)) if len(finite) else 0.0
        self._binary_target = _detect_binary(y_arr)
        self._nonnegative_target = _detect_nonnegative(y_arr)
        self._configure_continuous_regime()
        self._refresh_target_route_regime()
        self._binary_temperature = 1.0
        self._effective_task_modulation = self.enable_task_modulation and not self._binary_target

        if train_raw is None or "entity_id" not in train_raw.columns:
            logger.warning("  [V740-alpha] Missing train_raw/entity_id; fallback-only mode")
            self._fitted = True
            return self

        self._refresh_financing_consistency_regime(train_raw)
        self._text_cols = infer_text_columns(train_raw) if self._has_text_path() else []
        self._edgar_cols = infer_edgar_columns(train_raw) if self._has_edgar_path() else []
        self._source_state_exclude_cols = sorted(set(self._text_cols) | set(self._edgar_cols))
        self._feature_cols = _select_state_feature_cols(
            train_raw,
            self._target_name,
            self.max_covariates,
            self._source_state_exclude_cols,
        )
        if self._effective_financing_consistency:
            for finance_col in _FINANCING_TARGETS:
                if (
                    finance_col != self._target_name
                    and finance_col in train_raw.columns
                    and finance_col not in self._feature_cols
                ):
                    self._feature_cols.append(finance_col)
        window_train_raw = train_raw
        if self._funding_log_domain and self._target_name in train_raw.columns:
            window_train_raw = train_raw.copy()
            window_train_raw[self._target_name] = self._transform_continuous_array(
                pd.Series(train_raw[self._target_name], dtype="float64").to_numpy()
            )
        entity_windows = _build_v740_windows(
            train_raw=window_train_raw,
            target=self._target_name,
            feature_cols=self._feature_cols,
            input_size=self.input_size,
            horizon=self._horizon,
            step=max(1, self._horizon),
            max_entities=self.max_entities,
            max_windows=self.max_windows,
            val_frac=0.15,
            seed=seed,
            edgar_cols=self._edgar_cols,
            text_cols=self._text_cols,
            dual_clock_cfg=self._dual_clock_cfg,
            aux_target_raw=train_raw,
            enable_window_repair=self.enable_window_repair,
            min_history_points=self.min_window_history,
            target_windows_per_entity=self.target_windows_per_entity,
        )
        self._contexts = entity_windows.contexts
        self._context_memory = entity_windows.context_memory
        self._window_stats = entity_windows.window_stats
        if not entity_windows.train_x:
            logger.warning("  [V740-alpha] No windows; fallback-only mode")
            self._fitted = True
            return self

        if self._edgar_cols:
            self._edgar_source_density = float(train_raw[self._edgar_cols].notna().any(axis=1).mean())
        else:
            self._edgar_source_density = 0.0
        if self._text_cols:
            self._text_source_density = float(train_raw[self._text_cols].notna().any(axis=1).mean())
        else:
            self._text_source_density = 0.0
        if self._binary_target:
            train_window_y = np.stack(entity_windows.train_y).astype(np.float32, copy=False)
            binary_train = np.isfinite(train_window_y)
            if binary_train.any():
                train_pos_rate = float(np.clip(np.nanmean(train_window_y[binary_train]), 1e-4, 1.0 - 1e-4))
            else:
                train_pos_rate = 0.5
            self._binary_train_rate = train_pos_rate
            self._binary_pos_weight = float(np.clip((1.0 - train_pos_rate) / train_pos_rate, 1.0, 25.0))
            event_targets = (train_window_y.max(axis=1) > 0.5).astype(np.float32)
            current_state = np.stack(entity_windows.train_x).astype(np.float32, copy=False)[:, 0, -1]
            transition_targets = ((event_targets > 0.5) & (current_state <= 0.5)).astype(np.float32)
            self._binary_event_rate = float(event_targets.mean()) if len(event_targets) else 0.5
            self._binary_transition_rate = float(transition_targets.mean()) if len(transition_targets) else 0.5

        train_x_np = np.stack(entity_windows.train_x).astype(np.float32, copy=False)
        train_y_np = np.stack(entity_windows.train_y).astype(np.float32, copy=False)
        train_funding_aux_np = np.stack(entity_windows.train_funding_aux).astype(np.float32, copy=False)
        train_investors_aux_np = np.stack(entity_windows.train_investors_aux).astype(np.float32, copy=False)
        train_binary_aux_np = np.stack(entity_windows.train_binary_aux).astype(np.float32, copy=False)
        train_edgar_recent_np = (
            np.stack(entity_windows.train_edgar_recent).astype(np.float32, copy=False)
            if entity_windows.train_edgar_recent
            else None
        )
        train_edgar_bucket_np = (
            np.stack(entity_windows.train_edgar_bucket).astype(np.float32, copy=False)
            if entity_windows.train_edgar_bucket
            else None
        )
        train_text_recent_np = (
            np.stack(entity_windows.train_text_recent).astype(np.float32, copy=False)
            if entity_windows.train_text_recent
            else None
        )
        train_text_bucket_np = (
            np.stack(entity_windows.train_text_bucket).astype(np.float32, copy=False)
            if entity_windows.train_text_bucket
            else None
        )
        edgar_stats = self._summarize_recent_memory(train_edgar_recent_np, novelty_idx=None)
        text_stats = self._summarize_recent_memory(train_text_recent_np, novelty_idx=3)
        self._edgar_event_density = float(edgar_stats["event_density"])
        self._text_event_density = float(text_stats["event_density"])
        self._text_change_density = float(text_stats["novelty"])
        self._refresh_source_scales()
        self._refresh_count_gate_regime()
        self._refresh_count_route_regime()
        self._refresh_count_specialist_regime()
        if self._binary_target:
            self._configure_binary_regime()
        train_x = torch.tensor(train_x_np, dtype=torch.float32)
        train_y = torch.tensor(train_y_np, dtype=torch.float32)
        train_funding_aux = torch.tensor(train_funding_aux_np, dtype=torch.float32)
        train_investors_aux = torch.tensor(train_investors_aux_np, dtype=torch.float32)
        train_binary_aux = torch.tensor(train_binary_aux_np, dtype=torch.float32)
        train_sample_weight = torch.tensor(
            self._build_window_sample_weights(
                train_x_np,
                train_y_np,
                train_edgar_recent=train_edgar_recent_np,
                train_edgar_bucket=train_edgar_bucket_np,
                train_text_recent=train_text_recent_np,
                train_text_bucket=train_text_bucket_np,
            ),
            dtype=torch.float32,
        )
        train_edgar_recent = self._memory_batch(
            torch,
            entity_windows.train_edgar_recent,
            (self._dual_clock_cfg.max_events, 3 + len(self._edgar_cols)),
        )
        train_edgar_bucket = self._memory_batch(
            torch,
            entity_windows.train_edgar_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._edgar_cols)),
        )
        train_text_recent = self._memory_batch(
            torch,
            entity_windows.train_text_recent,
            (self._dual_clock_cfg.max_events, 5 + len(self._text_cols)),
        )
        train_text_bucket = self._memory_batch(
            torch,
            entity_windows.train_text_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 6 + len(self._text_cols)),
        )
        train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket = self._apply_source_tensor_scales(
            train_edgar_recent,
            train_edgar_bucket,
            train_text_recent,
            train_text_bucket,
        )
        val_x = torch.tensor(np.stack(entity_windows.val_x), dtype=torch.float32) if entity_windows.val_x else None
        val_y = torch.tensor(np.stack(entity_windows.val_y), dtype=torch.float32) if entity_windows.val_y else None
        val_funding_aux = torch.tensor(np.stack(entity_windows.val_funding_aux), dtype=torch.float32) if entity_windows.val_funding_aux else None
        val_investors_aux = torch.tensor(np.stack(entity_windows.val_investors_aux), dtype=torch.float32) if entity_windows.val_investors_aux else None
        val_binary_aux = torch.tensor(np.stack(entity_windows.val_binary_aux), dtype=torch.float32) if entity_windows.val_binary_aux else None
        val_edgar_recent = self._memory_batch(
            torch,
            entity_windows.val_edgar_recent,
            (self._dual_clock_cfg.max_events, 3 + len(self._edgar_cols)),
        ) if entity_windows.val_edgar_recent else None
        val_edgar_bucket = self._memory_batch(
            torch,
            entity_windows.val_edgar_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._edgar_cols)),
        ) if entity_windows.val_edgar_bucket else None
        val_text_recent = self._memory_batch(
            torch,
            entity_windows.val_text_recent,
            (self._dual_clock_cfg.max_events, 5 + len(self._text_cols)),
        ) if entity_windows.val_text_recent else None
        val_text_bucket = self._memory_batch(
            torch,
            entity_windows.val_text_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 6 + len(self._text_cols)),
        ) if entity_windows.val_text_bucket else None
        val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket = self._apply_source_tensor_scales(
            val_edgar_recent,
            val_edgar_bucket,
            val_text_recent,
            val_text_bucket,
        )

        train_teacher_probs = None
        if self._binary_target and self.enable_teacher_distill and len(entity_windows.train_x) >= 16:
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                teacher_x = _window_teacher_features(
                    entity_windows.train_x,
                    entity_windows.train_edgar_recent,
                    entity_windows.train_edgar_bucket,
                    entity_windows.train_text_recent,
                    entity_windows.train_text_bucket,
                )
                teacher_targets = np.stack(entity_windows.train_y).astype(np.float32, copy=False)
                teacher_y = (teacher_targets.max(axis=1) > 0.5).astype(np.int64)
                if len(np.unique(teacher_y)) > 1:
                    class_counts = np.bincount(teacher_y, minlength=2).astype(np.float32)
                    sample_weight = np.where(teacher_y > 0, 1.0 / max(class_counts[1], 1.0), 1.0 / max(class_counts[0], 1.0))
                    teacher = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=300,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    )
                    teacher.fit(teacher_x, teacher_y)
                    teacher_log_probs = teacher.predict_proba(teacher_x)[:, 1].astype(np.float32)
                    tree_teacher = HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=3,
                        max_iter=120,
                        min_samples_leaf=8,
                        random_state=seed,
                    )
                    tree_teacher.fit(teacher_x, teacher_y, sample_weight=sample_weight)
                    teacher_tree_probs = tree_teacher.predict_proba(teacher_x)[:, 1].astype(np.float32)
                    train_teacher_probs = (
                        self._teacher_logistic_mix * teacher_log_probs
                        + self._teacher_tree_mix * teacher_tree_probs
                    ).astype(np.float32)
            except Exception as exc:
                logger.warning(f"  [V740-alpha] binary teacher unavailable: {exc}")

        train_teacher = torch.tensor(
            train_teacher_probs if train_teacher_probs is not None else np.full((len(train_x),), 0.5, dtype=np.float32),
            dtype=torch.float32,
        )

        self._network = V740AlphaNet(
            in_channels=train_x.shape[1],
            seq_len=train_x.shape[2],
            horizon=self._horizon,
            hidden_dim=self.hidden_dim,
            enable_continuous_anchor=self._funding_anchor_enabled,
            continuous_anchor_strength=self._effective_funding_anchor_strength,
            enable_target_routing=self._effective_target_routing,
            target_route_experts=self.target_route_experts,
            enable_count_anchor=self.enable_count_anchor,
            count_anchor_strength=self.count_anchor_strength,
            enable_count_jump=self.enable_count_jump,
            count_jump_strength=self.count_jump_strength,
            enable_count_sparsity_gate=self.enable_count_sparsity_gate,
            count_sparsity_gate_strength=self._effective_count_sparsity_gate_strength,
            enable_count_source_routing=self._effective_count_source_routing,
            count_route_experts=self.count_route_experts,
            count_route_floor=self._effective_count_route_floor,
            enable_count_hurdle_head=self.enable_count_hurdle_head and self._target_name == "investors_count",
            enable_count_source_specialists=self._effective_count_source_specialists,
            enable_v741_lite=self.enable_v741_lite,
            enable_financing_consistency=self._effective_financing_consistency,
            enable_financing_factorization=self._effective_financing_factorization,
            enable_financing_guarded_phase=self._effective_financing_guarded_phase,
            enable_financing_evidence_residual=self._effective_financing_evidence_residual,
            financing_process_blend=self._effective_financing_process_blend,
            financing_investor_blend_scale=self.financing_investor_blend_scale,
            financing_binary_blend_scale=self.financing_binary_blend_scale,
            financing_funding_blend_scale=self.financing_funding_blend_scale,
            funding_log_domain_enabled=self._funding_log_domain,
        ).to(self._device)
        self._network.task_mod_enabled = self._effective_task_modulation
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_dataset = torch.utils.data.TensorDataset(
            train_x,
            train_y,
            train_edgar_recent,
            train_edgar_bucket,
            train_text_recent,
            train_text_bucket,
            train_teacher,
            train_sample_weight,
            train_funding_aux,
            train_investors_aux,
            train_binary_aux,
        )
        sampler = None
        shuffle = True
        if self._binary_target and len(train_dataset) > 1:
            window_targets = train_y.max(dim=1).values.detach().cpu().numpy()
            classes = (window_targets > 0.5).astype(np.int64)
            class_counts = np.bincount(classes, minlength=2)
            if np.all(class_counts > 0):
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[classes]
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
        )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(self.max_epochs):
            self._network.train()
            losses = []
            for xb, yb, er, eb, tr, tb, tp, sw, fb, ib, bb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                er = er.to(self._device)
                eb = eb.to(self._device)
                tr = tr.to(self._device)
                tb = tb.to(self._device)
                tp = tp.to(self._device)
                sw = sw.to(self._device)
                fb = fb.to(self._device)
                ib = ib.to(self._device)
                bb = bb.to(self._device)
                task_idx, target_idx, horizon_value, ablation_idx = self._token_tensor(torch, len(xb))
                outputs = self._network(
                    xb,
                    task_idx,
                    target_idx,
                    horizon_value,
                    ablation_idx,
                    edgar_recent=er,
                    edgar_bucket=eb,
                    text_recent=tr,
                    text_bucket=tb,
                )
                loss = self._loss(
                    torch,
                    outputs,
                    yb,
                    teacher_probs=tp if self._binary_target else None,
                    sample_weight=sw,
                    financing_targets={
                        "funding_raised_usd": fb,
                        "investors_count": ib,
                        "is_funded": bb,
                    },
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if val_x is None or val_y is None:
                continue

            self._network.eval()
            with torch.no_grad():
                task_idx, target_idx, horizon_value, ablation_idx = self._token_tensor(torch, len(val_x))
                val_out = self._network(
                    val_x.to(self._device),
                    task_idx,
                    target_idx,
                    horizon_value,
                    ablation_idx,
                    edgar_recent=val_edgar_recent.to(self._device) if val_edgar_recent is not None else None,
                    edgar_bucket=val_edgar_bucket.to(self._device) if val_edgar_bucket is not None else None,
                    text_recent=val_text_recent.to(self._device) if val_text_recent is not None else None,
                    text_bucket=val_text_bucket.to(self._device) if val_text_bucket is not None else None,
                )
                val_loss = float(
                    self._loss(
                        torch,
                        val_out,
                        val_y.to(self._device),
                        teacher_probs=None,
                        financing_targets={
                            "funding_raised_usd": val_funding_aux.to(self._device) if val_funding_aux is not None else val_y.to(self._device) * 0.0,
                            "investors_count": val_investors_aux.to(self._device) if val_investors_aux is not None else val_y.to(self._device) * 0.0,
                            "is_funded": val_binary_aux.to(self._device) if val_binary_aux is not None else val_y.to(self._device) * 0.0,
                        },
                    ).cpu()
                )

            logger.info(
                f"  [V740-alpha] epoch={epoch + 1}/{self.max_epochs} "
                f"train={np.mean(losses):.6f} val={val_loss:.6f}"
            )
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = copy.deepcopy(self._network.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        if self._binary_target and val_x is not None and val_y is not None and len(val_x) > 0:
            self._network.eval()
            with torch.no_grad():
                task_idx, target_idx, horizon_value, ablation_idx = self._token_tensor(torch, len(val_x))
                val_out = self._network(
                    val_x.to(self._device),
                    task_idx,
                    target_idx,
                    horizon_value,
                    ablation_idx,
                    edgar_recent=val_edgar_recent.to(self._device) if val_edgar_recent is not None else None,
                    edgar_bucket=val_edgar_bucket.to(self._device) if val_edgar_bucket is not None else None,
                    text_recent=val_text_recent.to(self._device) if val_text_recent is not None else None,
                    text_bucket=val_text_bucket.to(self._device) if val_text_bucket is not None else None,
                )
                val_logits = val_out["binary"]
                val_target = val_y.to(self._device)
                best_temp = 1.0
                best_bias = 0.0
                best_score = float("inf")
                if self._horizon <= 7:
                    temp_candidates = (0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.2, 3.0)
                    bias_candidates = (-0.35, -0.2, -0.1, 0.0, 0.1, 0.2, 0.35)
                else:
                    temp_candidates = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0)
                    bias_candidates = (-0.2, -0.1, 0.0, 0.1, 0.2)
                target_rate = val_target.mean()
                for temp in temp_candidates:
                    for bias in bias_candidates:
                        scaled = val_logits / temp + bias
                        probs = torch.sigmoid(scaled)
                        mae = torch.mean(torch.abs(probs - val_target))
                        brier = torch.mean((probs - val_target) ** 2)
                        rate_gap = torch.abs(probs.mean() - target_rate)
                        score = float((mae + 0.20 * brier + 0.05 * rate_gap).cpu())
                        if score < best_score:
                            best_score = score
                            best_temp = float(temp)
                            best_bias = float(bias)
                self._binary_temperature = best_temp
                self._binary_logit_bias = best_bias

        self._fitted = True
        return self

    def _prepare_prediction_entities(
        self,
        X: pd.DataFrame,
        test_raw: Optional[pd.DataFrame],
        target: Optional[str],
    ) -> tuple[Optional[np.ndarray], List[str]]:
        h = len(X)
        if test_raw is None or "entity_id" not in test_raw.columns:
            return None, []
        if target and target in test_raw.columns:
            valid_mask = test_raw[target].notna()
            test_entities = test_raw.loc[valid_mask, "entity_id"].values
        else:
            test_entities = test_raw["entity_id"].values
        if len(test_entities) != h:
            return None, []

        unique_entities: List[str] = []
        seen = set()
        for eid in test_entities:
            sid = str(eid)
            if sid in self._contexts and sid not in seen:
                seen.add(sid)
                unique_entities.append(sid)
        return test_entities, unique_entities

    def _context_tensor_for_entities(
        self,
        torch,
        unique_entities: List[str],
        key: str,
        default_shape: tuple[int, int],
    ):
        return torch.tensor(
            np.stack(
                [
                    self._context_memory.get(eid, {}).get(
                        key,
                        np.zeros(default_shape, dtype=np.float32),
                    )
                    for eid in unique_entities
                ]
            ),
            dtype=torch.float32,
            device=self._device,
        )

    def _forward_entity_contexts(self, torch, unique_entities: List[str]):
        if not unique_entities:
            return {}
        x_batch = torch.tensor(
            np.stack([self._contexts[eid] for eid in unique_entities]),
            dtype=torch.float32,
            device=self._device,
        )
        edgar_recent = self._context_tensor_for_entities(
            torch,
            unique_entities,
            "edgar_recent",
            (self._dual_clock_cfg.max_events, 3 + len(self._edgar_cols)),
        )
        edgar_bucket = self._context_tensor_for_entities(
            torch,
            unique_entities,
            "edgar_bucket",
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._edgar_cols)),
        )
        text_recent = self._context_tensor_for_entities(
            torch,
            unique_entities,
            "text_recent",
            (self._dual_clock_cfg.max_events, 5 + len(self._text_cols)),
        )
        text_bucket = self._context_tensor_for_entities(
            torch,
            unique_entities,
            "text_bucket",
            (len(self._dual_clock_cfg.recency_buckets) + 1, 6 + len(self._text_cols)),
        )
        edgar_recent, edgar_bucket, text_recent, text_bucket = self._apply_source_tensor_scales(
            edgar_recent,
            edgar_bucket,
            text_recent,
            text_bucket,
        )
        with torch.no_grad():
            task_idx, target_idx, horizon_value, ablation_idx = self._token_tensor(torch, len(unique_entities))
            outputs = self._network(
                x_batch,
                task_idx,
                target_idx,
                horizon_value,
                ablation_idx,
                edgar_recent=edgar_recent,
                edgar_bucket=edgar_bucket,
                text_recent=text_recent,
                text_bucket=text_bucket,
            )
        self._update_target_route_stats(outputs)
        self._update_count_route_stats(outputs)
        self._update_count_specialist_stats(outputs)
        return outputs

    def _build_eval_windows(self, raw_df: pd.DataFrame) -> _V740Windows:
        eval_raw = raw_df
        if self._funding_log_domain and self._target_name in raw_df.columns:
            eval_raw = raw_df.copy()
            eval_raw[self._target_name] = self._transform_continuous_array(
                pd.Series(raw_df[self._target_name], dtype="float64").to_numpy()
            )
        max_eval_entities = self.max_entities
        if raw_df is not None and "entity_id" in raw_df.columns:
            max_eval_entities = max(max_eval_entities, int(raw_df["entity_id"].nunique()))
        return _build_v740_windows(
            train_raw=eval_raw,
            target=self._target_name,
            feature_cols=self._feature_cols,
            input_size=self.input_size,
            horizon=self._horizon,
            step=max(1, self._horizon),
            max_entities=max_eval_entities,
            max_windows=self.max_windows,
            val_frac=0.0,
            seed=self.seed,
            edgar_cols=self._edgar_cols,
            text_cols=self._text_cols,
            dual_clock_cfg=self._dual_clock_cfg,
            aux_target_raw=raw_df,
            enable_window_repair=self.enable_window_repair,
            min_history_points=self.min_window_history,
            target_windows_per_entity=self.target_windows_per_entity,
        )

    def collect_window_outputs(self, raw_df: pd.DataFrame) -> Dict[str, object]:
        if not self._fitted:
            raise ValueError("V740AlphaPrototypeWrapper is not fitted")
        if self._network is None:
            return {
                "available": False,
                "window_count": 0,
                "horizon": int(self._horizon),
            }

        eval_windows = self._build_eval_windows(raw_df)
        if not eval_windows.train_x:
            return {
                "available": False,
                "window_count": 0,
                "horizon": int(self._horizon),
                "window_stats": copy.deepcopy(eval_windows.window_stats),
            }

        import torch

        n_windows = len(eval_windows.train_x)
        x_batch = torch.tensor(
            np.stack(eval_windows.train_x).astype(np.float32, copy=False),
            dtype=torch.float32,
            device=self._device,
        )

        def _window_tensor(entries: List[np.ndarray], default_shape: tuple[int, int]):
            if entries:
                arr = np.stack(entries).astype(np.float32, copy=False)
            else:
                arr = np.zeros((n_windows, *default_shape), dtype=np.float32)
            return torch.tensor(arr, dtype=torch.float32, device=self._device)

        edgar_recent = _window_tensor(
            eval_windows.train_edgar_recent,
            (self._dual_clock_cfg.max_events, 3 + len(self._edgar_cols)),
        )
        edgar_bucket = _window_tensor(
            eval_windows.train_edgar_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._edgar_cols)),
        )
        text_recent = _window_tensor(
            eval_windows.train_text_recent,
            (self._dual_clock_cfg.max_events, 5 + len(self._text_cols)),
        )
        text_bucket = _window_tensor(
            eval_windows.train_text_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 6 + len(self._text_cols)),
        )
        edgar_recent, edgar_bucket, text_recent, text_bucket = self._apply_source_tensor_scales(
            edgar_recent,
            edgar_bucket,
            text_recent,
            text_bucket,
        )

        with torch.no_grad():
            task_idx, target_idx, horizon_value, ablation_idx = self._token_tensor(torch, n_windows)
            outputs = self._network(
                x_batch,
                task_idx,
                target_idx,
                horizon_value,
                ablation_idx,
                edgar_recent=edgar_recent,
                edgar_bucket=edgar_bucket,
                text_recent=text_recent,
                text_bucket=text_bucket,
            )
        self._update_target_route_stats(outputs)
        self._update_count_route_stats(outputs)
        self._update_count_specialist_stats(outputs)

        artifacts: Dict[str, object] = {
            "available": True,
            "window_count": int(n_windows),
            "horizon": int(self._horizon),
            "window_stats": copy.deepcopy(eval_windows.window_stats),
            "target_true": np.stack(eval_windows.train_y).astype(np.float64, copy=False),
            "funding_true": np.stack(eval_windows.train_funding_aux).astype(np.float64, copy=False),
            "investors_true": np.stack(eval_windows.train_investors_aux).astype(np.float64, copy=False),
            "binary_true": np.stack(eval_windows.train_binary_aux).astype(np.float64, copy=False),
            "primary_binary_prob": (
                outputs["legacy_binary_prob"].cpu().numpy().astype(np.float64, copy=False)
                if outputs.get("legacy_binary_prob") is not None
                else torch.sigmoid(outputs["binary"]).cpu().numpy().astype(np.float64, copy=False)
            ),
            "blended_binary_prob": torch.sigmoid(outputs["binary"]).cpu().numpy().astype(np.float64, copy=False),
            "primary_count": (
                outputs["legacy_count"].clamp_min(0.0).cpu().numpy().astype(np.float64, copy=False)
                if outputs.get("legacy_count") is not None
                else outputs["count"].clamp_min(0.0).cpu().numpy().astype(np.float64, copy=False)
            ),
            "blended_count": outputs["count"].clamp_min(0.0).cpu().numpy().astype(np.float64, copy=False),
            "primary_continuous_raw": (
                outputs["legacy_continuous"].cpu().numpy().astype(np.float64, copy=False)
                if outputs.get("legacy_continuous") is not None
                else outputs["continuous"].cpu().numpy().astype(np.float64, copy=False)
            ),
            "blended_continuous_raw": outputs["continuous"].cpu().numpy().astype(np.float64, copy=False),
            "edgar_active_rate": float(
                np.mean([
                    float(np.any(entry[:, 2] > 0.0))
                    for entry in eval_windows.train_edgar_recent
                ]) if eval_windows.train_edgar_recent else 0.0
            ),
            "text_active_rate": float(
                np.mean([
                    float(np.any(entry[:, 2] > 0.0))
                    for entry in eval_windows.train_text_recent
                ]) if eval_windows.train_text_recent else 0.0
            ),
        }
        if self._target_name == "funding_raised_usd":
            artifacts["primary_continuous"] = self._inverse_continuous_array(
                artifacts["primary_continuous_raw"]
            ).astype(np.float64, copy=False)
            artifacts["blended_continuous"] = self._inverse_continuous_array(
                artifacts["blended_continuous_raw"]
            ).astype(np.float64, copy=False)
        else:
            artifacts["primary_continuous"] = artifacts["primary_continuous_raw"]
            artifacts["blended_continuous"] = artifacts["blended_continuous_raw"]

        for key in (
            "financing_event_prob",
            "financing_count",
            "financing_count_positive",
            "financing_amount_log",
            "financing_amount_positive_log",
            "financing_breadth_log",
            "financing_breadth_anchor_log",
            "financing_intensity_log",
            "financing_amount_coupling",
            "financing_process_blend",
            "financing_investor_gate",
            "financing_binary_gate",
            "financing_funding_gate",
        ):
            value = outputs.get(key)
            if value is not None:
                artifacts[key] = value.cpu().numpy().astype(np.float64, copy=False)
        if "financing_amount_log" in artifacts:
            artifacts["financing_amount"] = np.expm1(
                np.clip(artifacts["financing_amount_log"], 0.0, 25.0)
            ).astype(np.float64, copy=False)
        return artifacts

    def score_financing_diagnostics(self, raw_df: pd.DataFrame) -> Dict[str, object]:
        artifacts = self.collect_window_outputs(raw_df)
        summary: Dict[str, object] = {
            "available": bool(artifacts.get("available", False)),
            "window_count": int(artifacts.get("window_count", 0)),
            "horizon": int(artifacts.get("horizon", self._horizon)),
            "edgar_active_rate": float(artifacts.get("edgar_active_rate", 0.0)),
            "text_active_rate": float(artifacts.get("text_active_rate", 0.0)),
        }
        if not summary["available"] or "financing_event_prob" not in artifacts:
            summary["has_financing_head"] = False
            return summary

        event_prob = np.clip(np.asarray(artifacts["financing_event_prob"], dtype=np.float64), 1e-6, 1.0 - 1e-6)
        count_pred = np.clip(np.asarray(artifacts["financing_count"], dtype=np.float64), 0.0, None)
        count_positive = np.clip(
            np.asarray(artifacts.get("financing_count_positive", artifacts["financing_count"]), dtype=np.float64),
            0.0,
            None,
        )
        amount_log = np.clip(np.asarray(artifacts["financing_amount_log"], dtype=np.float64), 0.0, 25.0)
        binary_target = (np.asarray(artifacts["binary_true"], dtype=np.float64) > 0.5).astype(np.float64)
        investors_target = np.clip(np.asarray(artifacts["investors_true"], dtype=np.float64), 0.0, None)
        funding_target_log = np.log1p(np.clip(np.asarray(artifacts["funding_true"], dtype=np.float64), 0.0, None))

        count_presence = 1.0 - np.exp(-count_pred)
        amount_presence = 1.0 / (1.0 + np.exp(-(amount_log - math.log1p(1.0))))
        inactive_mask = binary_target <= 0.5

        summary.update(
            {
                "has_financing_head": True,
                "point_count": int(binary_target.size),
                "true_event_rate": float(binary_target.mean()),
                "pred_event_rate": float(event_prob.mean()),
                "event_bce": float(
                    np.mean(
                        -binary_target * np.log(event_prob)
                        - (1.0 - binary_target) * np.log(1.0 - event_prob)
                    )
                ),
                "count_log_mae": float(np.mean(np.abs(np.log1p(count_pred) - np.log1p(investors_target)))),
                "amount_log_mae": float(np.mean(np.abs(amount_log - funding_target_log))),
                "count_presence_gap": float(np.mean(np.abs(event_prob - count_presence))),
                "amount_presence_gap": float(np.mean(np.abs(event_prob - amount_presence))),
                "count_amount_presence_gap": float(np.mean(np.abs(count_presence - amount_presence))),
                "coherence_mse": float(
                    np.mean(
                        (event_prob - count_presence) ** 2
                        + (event_prob - amount_presence) ** 2
                        + 0.5 * (count_presence - amount_presence) ** 2
                    )
                ),
                "inactive_count_log_mass": float(
                    np.mean(np.log1p(count_pred)[inactive_mask])
                ) if np.any(inactive_mask) else 0.0,
                "inactive_amount_log_mass": float(
                    np.mean(amount_log[inactive_mask])
                ) if np.any(inactive_mask) else 0.0,
                "inactive_joint_mass": float(
                    np.mean((np.log1p(count_pred) + amount_log)[inactive_mask])
                ) if np.any(inactive_mask) else 0.0,
                "blend_strength_mean": float(
                    np.mean(np.asarray(artifacts.get("financing_process_blend", 0.0), dtype=np.float64))
                ) if "financing_process_blend" in artifacts else 0.0,
            }
        )
        if "financing_investor_gate" in artifacts:
            summary["investor_gate_mean"] = float(
                np.mean(np.asarray(artifacts["financing_investor_gate"], dtype=np.float64))
            )
        if "financing_binary_gate" in artifacts:
            summary["binary_gate_mean"] = float(
                np.mean(np.asarray(artifacts["financing_binary_gate"], dtype=np.float64))
            )
        if "financing_funding_gate" in artifacts:
            summary["funding_gate_mean"] = float(
                np.mean(np.asarray(artifacts["financing_funding_gate"], dtype=np.float64))
            )
        if count_positive.shape[1] > 1:
            count_step = np.diff(np.log1p(count_positive), axis=1)
            amount_step = np.diff(amount_log, axis=1)
            step_weight = 0.25 + binary_target[:, 1:]
            summary["trajectory_gap"] = float(np.mean(np.abs(count_step - amount_step) * step_weight))
        else:
            summary["trajectory_gap"] = 0.0
        if "financing_breadth_log" in artifacts:
            breadth_log = np.asarray(artifacts["financing_breadth_log"], dtype=np.float64)
            summary["breadth_log_mae"] = float(np.mean(np.abs(breadth_log - np.log1p(investors_target))))
            summary["breadth_mean"] = float(np.mean(np.expm1(np.clip(breadth_log, 0.0, 12.0))))
        if "financing_intensity_log" in artifacts:
            intensity_log = np.asarray(artifacts["financing_intensity_log"], dtype=np.float64)
            intensity_target = np.clip(funding_target_log - 0.35 * np.log1p(investors_target), 0.0, None)
            summary["intensity_log_mae"] = float(np.mean(np.abs(intensity_log - intensity_target)))
        if "financing_amount_coupling" in artifacts:
            summary["amount_coupling_mean"] = float(
                np.mean(np.asarray(artifacts["financing_amount_coupling"], dtype=np.float64))
            )
        if self._target_name == "investors_count":
            primary_count = np.clip(np.asarray(artifacts["primary_count"], dtype=np.float64), 0.0, None)
            summary["primary_vs_financing_count_gap"] = float(np.mean(np.abs(primary_count - count_pred)))
            blended_count = np.clip(
                np.asarray(artifacts.get("blended_count", artifacts["primary_count"]), dtype=np.float64),
                0.0,
                None,
            )
            summary["blended_vs_financing_count_gap"] = float(np.mean(np.abs(blended_count - count_pred)))
        elif self._target_name == "is_funded":
            primary_event = np.clip(np.asarray(artifacts["primary_binary_prob"], dtype=np.float64), 1e-6, 1.0 - 1e-6)
            summary["primary_vs_financing_event_gap"] = float(np.mean(np.abs(primary_event - event_prob)))
        elif self._target_name == "funding_raised_usd":
            primary_amount = np.asarray(artifacts["primary_continuous_raw"], dtype=np.float64)
            summary["primary_vs_financing_amount_log_gap"] = float(np.mean(np.abs(primary_amount - amount_log)))
        return summary

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("V740AlphaPrototypeWrapper is not fitted")
        h = len(X)
        if self._network is None or not self._contexts:
            return np.full(h, self._fallback_value, dtype=np.float64)

        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        req_horizon = int(kwargs.get("horizon", self._horizon))
        test_entities, unique_entities = self._prepare_prediction_entities(X, test_raw, target)
        if test_entities is None:
            return np.full(h, self._fallback_value, dtype=np.float64)

        import torch

        preds_map: Dict[str, float] = {}
        if unique_entities:
            outputs = self._forward_entity_contexts(torch, unique_entities)
            if self._binary_target:
                raw = torch.sigmoid(outputs["binary"] / self._binary_temperature + self._binary_logit_bias).cpu().numpy()
            elif self._target_name == "investors_count":
                raw = outputs["count"].cpu().numpy()
            else:
                raw = self._inverse_continuous_array(outputs["continuous"].cpu().numpy())
            idx = min(req_horizon - 1, raw.shape[1] - 1)
            preds_map = {eid: float(raw[i, idx]) for i, eid in enumerate(unique_entities)}

        out = np.empty(h, dtype=np.float64)
        for i, eid in enumerate(test_entities):
            out[i] = preds_map.get(str(eid), self._fallback_value)
        if self._nonnegative_target:
            out = np.clip(out, 0.0, None)
        return _sanitize_predictions(out, self._fallback_value, "V740AlphaPrototype")


__all__ = ["V740AlphaPrototypeWrapper"]
