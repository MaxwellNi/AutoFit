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
    infer_edgar_columns,
    infer_text_columns,
)

logger = logging.getLogger(__name__)


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


@dataclass
class _V740Windows:
    train_x: List[np.ndarray]
    train_y: List[np.ndarray]
    val_x: List[np.ndarray]
    val_y: List[np.ndarray]
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
                row.extend([
                    float(np.sum(recent[:, 2] > 0.0)),
                    float(np.mean(recent[:, 0][recent[:, 2] > 0.0])) if np.any(recent[:, 2] > 0.0) else 0.0,
                ])
            else:
                row.extend([0.0, 0.0])
            if bucket is not None and bucket.size:
                row.extend([
                    float(np.sum(bucket[:, 0])),
                    float(np.max(bucket[:, 0])),
                    float(np.mean(bucket[:, 0])),
                ])
            else:
                row.extend([0.0, 0.0, 0.0])
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
) -> _V740Windows:
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    val_x: List[np.ndarray] = []
    val_y: List[np.ndarray] = []
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

    if train_raw is None or "entity_id" not in train_raw.columns:
        return _V740Windows(
            train_x, train_y, val_x, val_y,
            train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket,
            val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket,
            contexts, context_memory,
        )

    rng = np.random.RandomState(seed)
    groups = train_raw.groupby("entity_id", sort=False)
    for i, (eid, grp) in enumerate(groups):
        if i >= max_entities:
            break
        grp = grp.sort_values("crawled_date_day").reset_index(drop=True)
        if target not in grp.columns:
            continue

        y_arr = pd.Series(grp[target], dtype="float64").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        times = pd.to_datetime(grp["crawled_date_day"], errors="coerce")
        if len(y_arr) < input_size + horizon or times.isna().all():
            continue

        channels = [y_arr]
        for col in feature_cols:
            if col in grp.columns:
                vals = pd.Series(grp[col], dtype="float64").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
                channels.append(vals)
        series = np.stack(channels, axis=0)
        contexts[str(eid)] = series[:, -input_size:].astype(np.float32, copy=False)

        last_time = times.iloc[-1]
        edgar_context_mem = _build_edgar_memory(grp, last_time, edgar_cols, dual_clock_cfg)
        text_context_mem = build_dual_clock_memory_for_entity(grp, last_time, text_cols, dual_clock_cfg)
        context_memory[str(eid)] = {
            "edgar_recent": edgar_context_mem["recent_tokens"],
            "edgar_bucket": edgar_context_mem["bucket_tokens"],
            "text_recent": text_context_mem["recent_tokens"],
            "text_bucket": text_context_mem["bucket_tokens"],
        }

        entity_x: List[np.ndarray] = []
        entity_y: List[np.ndarray] = []
        entity_edgar_recent: List[np.ndarray] = []
        entity_edgar_bucket: List[np.ndarray] = []
        entity_text_recent: List[np.ndarray] = []
        entity_text_bucket: List[np.ndarray] = []
        limit = len(y_arr) - input_size - horizon + 1
        for t in range(0, limit, step):
            x_win = series[:, t : t + input_size]
            y_win = y_arr[t + input_size : t + input_size + horizon]
            end_idx = t + input_size - 1
            pred_time = times.iloc[end_idx]
            if pd.isna(pred_time):
                continue
            if np.any(~np.isfinite(x_win)) or np.any(~np.isfinite(y_win)):
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
                text_mem = build_dual_clock_memory_for_entity(prefix, pred_time, text_cols, dual_clock_cfg)
                text_recent = text_mem["recent_tokens"]
                text_bucket = text_mem["bucket_tokens"]
            else:
                text_recent = np.zeros((dual_clock_cfg.max_events, 3), dtype=np.float32)
                text_bucket = np.zeros((len(dual_clock_cfg.recency_buckets) + 1, 4), dtype=np.float32)

            entity_x.append(x_win.astype(np.float32, copy=False))
            entity_y.append(y_win.astype(np.float32, copy=False))
            entity_edgar_recent.append(edgar_recent.astype(np.float32, copy=False))
            entity_edgar_bucket.append(edgar_bucket.astype(np.float32, copy=False))
            entity_text_recent.append(text_recent.astype(np.float32, copy=False))
            entity_text_bucket.append(text_bucket.astype(np.float32, copy=False))

        if not entity_x:
            continue

        n_val = 0
        if len(entity_x) >= 4:
            n_val = max(1, int(round(len(entity_x) * val_frac)))
            n_val = min(n_val, len(entity_x) - 1)
        split = len(entity_x) - n_val
        train_x.extend(entity_x[:split])
        train_y.extend(entity_y[:split])
        train_edgar_recent.extend(entity_edgar_recent[:split])
        train_edgar_bucket.extend(entity_edgar_bucket[:split])
        train_text_recent.extend(entity_text_recent[:split])
        train_text_bucket.extend(entity_text_bucket[:split])
        val_x.extend(entity_x[split:])
        val_y.extend(entity_y[split:])
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
    train_x, train_y, train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket = _apply_cap(
        train_idx, train_x, train_y, train_edgar_recent, train_edgar_bucket, train_text_recent, train_text_bucket,
    )
    if val_x:
        val_cap = max(2048, max_windows // 5)
        val_idx = _cap_indices(len(val_x), val_cap)
        val_x, val_y, val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket = _apply_cap(
            val_idx, val_x, val_y, val_edgar_recent, val_edgar_bucket, val_text_recent, val_text_bucket,
        )

    return _V740Windows(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
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

    class V740AlphaNet(nn.Module):
        def __init__(
            self,
            in_channels: int,
            seq_len: int,
            horizon: int,
            hidden_dim: int = 64,
            cond_dim: int = 16,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.horizon = horizon
            self.task_mod_enabled = True
            self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.decomp = MovingAverageDecomposition(kernel_size=5)
            self.multi_res = MultiResolutionBlock(hidden_dim, hidden_dim)
            self.patch_mixer = PatchContextMixer(hidden_dim, patch_size=8)
            self.memory_branch = CompactTemporalMemory(hidden_dim)
            self.value_branch = ValueBucketEncoder(num_bins=32, hidden_dim=hidden_dim)
            self.cond_encoder = ConditionEncoder(emb_dim=cond_dim, seq_len=seq_len)
            self.edgar_memory = EventMemoryEncoder(hidden_dim)
            self.text_memory = EventMemoryEncoder(hidden_dim)
            self.fusion = InvariantVariantFusion(hidden_dim, cond_dim)
            self.task_mod = TaskSpecificModulator(hidden_dim)
            self.static_proj = StaticSummaryProjector(max(1, in_channels - 1), hidden_dim)
            self.backbone_norm = nn.LayerNorm(hidden_dim)
            self.combine = nn.Sequential(
                nn.Linear(hidden_dim * 6 + cond_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.shared_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, horizon),
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
            fused = self.multi_res(trend + seasonal)
            fused = self.patch_mixer(fused)
            memory_feat = self.memory_branch(fused)
            value_feat = self.value_branch(target_hist)
            cond = self.cond_encoder(task_idx, target_idx, horizon_value, ablation_idx)
            fused = self.fusion(fused, cond)
            pooled = fused.mean(dim=-1)
            static_feat = self.static_proj(static_summary)
            edgar_feat = self.edgar_memory(edgar_recent, edgar_bucket, x.shape[0], x.device)
            text_feat = self.text_memory(text_recent, text_bucket, x.shape[0], x.device)
            pooled = self.backbone_norm(pooled)
            pooled = self.combine(torch.cat([
                pooled, memory_feat, value_feat, static_feat, edgar_feat, text_feat, cond,
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
            return {
                "continuous": self.shared_head(pooled) + shared_bias,
                "count": self.count_head(pooled) + count_bias,
                "binary": self.binary_head(pooled) + binary_bias,
                "event": self.event_head(pooled).squeeze(-1) + event_bias,
                "uncertainty": torch.nn.functional.softplus(self.uncertainty_head(pooled)),
            }

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
        self.seed = seed
        self._network = None
        self._device = None
        self._contexts: Dict[str, np.ndarray] = {}
        self._context_memory: Dict[str, Dict[str, np.ndarray]] = {}
        self._feature_cols: List[str] = []
        self._edgar_cols: List[str] = []
        self._text_cols: List[str] = []
        self._fallback_value = 0.0
        self._binary_target = False
        self._nonnegative_target = False
        self._binary_train_rate = 0.5
        self._binary_pos_weight = 1.0
        self._binary_rate_floor = 0.05
        self._binary_temperature = 1.0
        self._binary_teacher_weight = 0.10
        self._binary_event_weight = 0.15
        self._teacher_logistic_mix = 0.4
        self._teacher_tree_mix = 0.6
        self._effective_task_modulation = enable_task_modulation
        self._binary_event_rate = 0.5
        self._binary_transition_rate = 0.5
        self._edgar_source_density = 0.0
        self._text_source_density = 0.0
        self._target_name = "funding_raised_usd"
        self._task_name = "task1_outcome"
        self._ablation_name = "core_only"
        self._horizon = 1
        self._dual_clock_cfg = DualClockConfig(max_events=max_event_tokens)

    def _weighted_reduce(self, torch, values, sample_weight=None):
        if values.ndim > 1:
            dims = tuple(range(1, values.ndim))
            values = values.mean(dim=dims)
        if sample_weight is None:
            return values.mean()
        weight = sample_weight.to(device=values.device, dtype=values.dtype).view(-1)
        weight = weight / weight.mean().clamp_min(1e-6)
        return (values * weight).mean()

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

        edgar_activity = 0.7 * _recent_activity(train_edgar_recent) + 0.3 * _bucket_activity(train_edgar_bucket)
        text_activity = 0.7 * _recent_activity(train_text_recent) + 0.3 * _bucket_activity(train_text_bucket)

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

    def _configure_binary_regime(self) -> None:
        teacher_weight = 0.08
        event_weight = 0.12
        logistic_mix = 0.45
        tree_mix = 0.55

        pos_balance = 1.0 - abs(self._binary_train_rate - 0.5) / 0.5
        event_sparsity = 1.0 - self._binary_event_rate
        transition_sparsity = 1.0 - self._binary_transition_rate

        teacher_weight += 0.03 * event_sparsity
        teacher_weight += 0.02 * transition_sparsity
        teacher_weight += 0.04 * self._edgar_source_density
        teacher_weight -= 0.03 * self._text_source_density
        teacher_weight += 0.01 * pos_balance

        event_weight += 0.04 * transition_sparsity
        event_weight += 0.02 * self._edgar_source_density
        event_weight -= 0.03 * self._text_source_density
        event_weight += 0.01 * pos_balance

        logistic_mix += 0.25 * self._text_source_density
        logistic_mix += 0.10 * pos_balance
        logistic_mix -= 0.20 * self._edgar_source_density
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

    def _loss(self, torch, outputs, target, teacher_probs=None, sample_weight=None):
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
            return (
                bce
                + 0.15 * brier
                + 0.10 * rate_penalty
                + 0.05 * std_penalty
                + 0.10 * separation
                + self._binary_event_weight * event_bce
                + self._binary_teacher_weight * teacher_align
            )
        if self._target_name == "investors_count":
            pred = torch.relu(outputs["count"])
            base = self._weighted_reduce(
                torch,
                torch.nn.functional.smooth_l1_loss(pred, target, reduction="none"),
                sample_weight=sample_weight,
            )
            neg_penalty = self._weighted_reduce(torch, torch.relu(-outputs["count"]), sample_weight=sample_weight)
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
            return base + 0.05 * neg_penalty + 0.05 * distdf
        pred = outputs["continuous"]
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
        return base + 0.1 * align + 0.05 * distdf

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
        self._binary_temperature = 1.0
        self._effective_task_modulation = self.enable_task_modulation and not self._binary_target

        if train_raw is None or "entity_id" not in train_raw.columns:
            logger.warning("  [V740-alpha] Missing train_raw/entity_id; fallback-only mode")
            self._fitted = True
            return self

        self._feature_cols = _select_feature_cols(train_raw, self._target_name, self.max_covariates)
        self._text_cols = infer_text_columns(train_raw) if self._has_text_path() else []
        self._edgar_cols = infer_edgar_columns(train_raw) if self._has_edgar_path() else []
        entity_windows = _build_v740_windows(
            train_raw=train_raw,
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
        )
        self._contexts = entity_windows.contexts
        self._context_memory = entity_windows.context_memory
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
            self._configure_binary_regime()

        train_x_np = np.stack(entity_windows.train_x).astype(np.float32, copy=False)
        train_y_np = np.stack(entity_windows.train_y).astype(np.float32, copy=False)
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
        train_x = torch.tensor(train_x_np, dtype=torch.float32)
        train_y = torch.tensor(train_y_np, dtype=torch.float32)
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
            (self._dual_clock_cfg.max_events, 3 + len(self._text_cols)),
        )
        train_text_bucket = self._memory_batch(
            torch,
            entity_windows.train_text_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._text_cols)),
        )
        val_x = torch.tensor(np.stack(entity_windows.val_x), dtype=torch.float32) if entity_windows.val_x else None
        val_y = torch.tensor(np.stack(entity_windows.val_y), dtype=torch.float32) if entity_windows.val_y else None
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
            (self._dual_clock_cfg.max_events, 3 + len(self._text_cols)),
        ) if entity_windows.val_text_recent else None
        val_text_bucket = self._memory_batch(
            torch,
            entity_windows.val_text_bucket,
            (len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._text_cols)),
        ) if entity_windows.val_text_bucket else None

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
        )
        sampler = None
        shuffle = True
        if self._binary_target and len(train_dataset) > 1:
            window_targets = train_y[:, -1].detach().cpu().numpy()
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
            for xb, yb, er, eb, tr, tb, tp, sw in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                er = er.to(self._device)
                eb = eb.to(self._device)
                tr = tr.to(self._device)
                tb = tb.to(self._device)
                tp = tp.to(self._device)
                sw = sw.to(self._device)
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
                val_loss = float(self._loss(torch, val_out, val_y.to(self._device), teacher_probs=None).cpu())

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
                best_score = float("inf")
                for temp in (0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0):
                    scaled = val_logits / temp
                    probs = torch.sigmoid(scaled)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(scaled, val_target)
                    brier = torch.mean((probs - val_target) ** 2)
                    score = float((bce + 0.2 * brier).cpu())
                    if score < best_score:
                        best_score = score
                        best_temp = float(temp)
                self._binary_temperature = best_temp

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("V740AlphaPrototypeWrapper is not fitted")
        h = len(X)
        if self._network is None or not self._contexts:
            return np.full(h, self._fallback_value, dtype=np.float64)

        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        req_horizon = int(kwargs.get("horizon", self._horizon))
        if test_raw is None or "entity_id" not in test_raw.columns:
            return np.full(h, self._fallback_value, dtype=np.float64)

        if target and target in test_raw.columns:
            valid_mask = test_raw[target].notna()
            test_entities = test_raw.loc[valid_mask, "entity_id"].values
        else:
            test_entities = test_raw["entity_id"].values
        if len(test_entities) != h:
            return np.full(h, self._fallback_value, dtype=np.float64)

        import torch

        unique_entities = []
        seen = set()
        for eid in test_entities:
            sid = str(eid)
            if sid in self._contexts and sid not in seen:
                seen.add(sid)
                unique_entities.append(sid)

        preds_map: Dict[str, float] = {}
        if unique_entities:
            x_batch = torch.tensor(
                np.stack([self._contexts[eid] for eid in unique_entities]),
                dtype=torch.float32,
                device=self._device,
            )
            edgar_recent = torch.tensor(
                np.stack([
                    self._context_memory.get(eid, {}).get(
                        "edgar_recent",
                        np.zeros((self._dual_clock_cfg.max_events, 3 + len(self._edgar_cols)), dtype=np.float32),
                    )
                    for eid in unique_entities
                ]),
                dtype=torch.float32,
                device=self._device,
            )
            edgar_bucket = torch.tensor(
                np.stack([
                    self._context_memory.get(eid, {}).get(
                        "edgar_bucket",
                        np.zeros((len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._edgar_cols)), dtype=np.float32),
                    )
                    for eid in unique_entities
                ]),
                dtype=torch.float32,
                device=self._device,
            )
            text_recent = torch.tensor(
                np.stack([
                    self._context_memory.get(eid, {}).get(
                        "text_recent",
                        np.zeros((self._dual_clock_cfg.max_events, 3 + len(self._text_cols)), dtype=np.float32),
                    )
                    for eid in unique_entities
                ]),
                dtype=torch.float32,
                device=self._device,
            )
            text_bucket = torch.tensor(
                np.stack([
                    self._context_memory.get(eid, {}).get(
                        "text_bucket",
                        np.zeros((len(self._dual_clock_cfg.recency_buckets) + 1, 4 + len(self._text_cols)), dtype=np.float32),
                    )
                    for eid in unique_entities
                ]),
                dtype=torch.float32,
                device=self._device,
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
            if self._binary_target:
                raw = torch.sigmoid(outputs["binary"] / self._binary_temperature).cpu().numpy()
            elif self._target_name == "investors_count":
                raw = torch.relu(outputs["count"]).cpu().numpy()
            else:
                raw = outputs["continuous"].cpu().numpy()
            idx = min(req_horizon - 1, raw.shape[1] - 1)
            preds_map = {eid: float(raw[i, idx]) for i, eid in enumerate(unique_entities)}

        out = np.empty(h, dtype=np.float64)
        for i, eid in enumerate(test_entities):
            out[i] = preds_map.get(str(eid), self._fallback_value)
        if self._nonnegative_target:
            out = np.clip(out, 0.0, None)
        return _sanitize_predictions(out, self._fallback_value, "V740AlphaPrototype")


__all__ = ["V740AlphaPrototypeWrapper"]
