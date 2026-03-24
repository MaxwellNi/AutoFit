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
    infer_edgar_columns,
    infer_text_columns,
)

logger = logging.getLogger(__name__)


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
        context_memory[str(eid)] = {
            "edgar_recent": build_dual_clock_memory_for_entity(grp, last_time, edgar_cols, dual_clock_cfg)["recent_tokens"]
            if edgar_cols else np.zeros((dual_clock_cfg.max_events, 3), dtype=np.float32),
            "edgar_bucket": build_dual_clock_memory_for_entity(grp, last_time, edgar_cols, dual_clock_cfg)["bucket_tokens"]
            if edgar_cols else np.zeros((len(dual_clock_cfg.recency_buckets) + 1, 4), dtype=np.float32),
            "text_recent": build_dual_clock_memory_for_entity(grp, last_time, text_cols, dual_clock_cfg)["recent_tokens"]
            if text_cols else np.zeros((dual_clock_cfg.max_events, 3), dtype=np.float32),
            "text_bucket": build_dual_clock_memory_for_entity(grp, last_time, text_cols, dual_clock_cfg)["bucket_tokens"]
            if text_cols else np.zeros((len(dual_clock_cfg.recency_buckets) + 1, 4), dtype=np.float32),
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
                edgar_mem = build_dual_clock_memory_for_entity(prefix, pred_time, edgar_cols, dual_clock_cfg)
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
        def __init__(self, emb_dim: int = 16):
            super().__init__()
            self.task_emb = nn.Embedding(4, emb_dim)
            self.target_emb = nn.Embedding(4, emb_dim)
            self.horizon_emb = nn.Embedding(5, emb_dim)
            self.ablation_emb = nn.Embedding(8, emb_dim)
            self.proj = nn.Sequential(
                nn.Linear(emb_dim * 4, emb_dim * 2),
                nn.GELU(),
                nn.Linear(emb_dim * 2, emb_dim),
            )

        def forward(
            self,
            task_idx: torch.Tensor,
            target_idx: torch.Tensor,
            horizon_idx: torch.Tensor,
            ablation_idx: torch.Tensor,
        ) -> torch.Tensor:
            x = torch.cat([
                self.task_emb(task_idx),
                self.target_emb(target_idx),
                self.horizon_emb(horizon_idx),
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
            self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
            self.decomp = MovingAverageDecomposition(kernel_size=5)
            self.multi_res = MultiResolutionBlock(hidden_dim, hidden_dim)
            self.patch_mixer = PatchContextMixer(hidden_dim, patch_size=8)
            self.memory_branch = CompactTemporalMemory(hidden_dim)
            self.value_branch = ValueBucketEncoder(num_bins=32, hidden_dim=hidden_dim)
            self.cond_encoder = ConditionEncoder(emb_dim=cond_dim)
            self.edgar_memory = EventMemoryEncoder(hidden_dim)
            self.text_memory = EventMemoryEncoder(hidden_dim)
            self.fusion = InvariantVariantFusion(hidden_dim, cond_dim)
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
            horizon_idx: torch.Tensor,
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
            cond = self.cond_encoder(task_idx, target_idx, horizon_idx, ablation_idx)
            fused = self.fusion(fused, cond)
            pooled = fused.mean(dim=-1)
            static_feat = self.static_proj(static_summary)
            edgar_feat = self.edgar_memory(edgar_recent, edgar_bucket, x.shape[0], x.device)
            text_feat = self.text_memory(text_recent, text_bucket, x.shape[0], x.device)
            pooled = self.backbone_norm(pooled)
            pooled = self.combine(torch.cat([
                pooled, memory_feat, value_feat, static_feat, edgar_feat, text_feat, cond,
            ], dim=-1))
            return {
                "continuous": self.shared_head(pooled),
                "count": self.count_head(pooled),
                "binary": self.binary_head(pooled),
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
    _HORIZON_MAP = {1: 0, 7: 1, 14: 2, 30: 3}
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
        self._target_name = "funding_raised_usd"
        self._task_name = "task1_outcome"
        self._ablation_name = "core_only"
        self._horizon = 1
        self._dual_clock_cfg = DualClockConfig(max_events=max_event_tokens)

    def _token_tensor(self, torch, batch_size: int):
        task_idx = torch.full((batch_size,), self._TASK_MAP.get(self._task_name, 0), dtype=torch.long, device=self._device)
        target_idx = torch.full((batch_size,), self._TARGET_MAP.get(self._target_name, 0), dtype=torch.long, device=self._device)
        horizon_idx = torch.full((batch_size,), self._HORIZON_MAP.get(self._horizon, 0), dtype=torch.long, device=self._device)
        ablation_idx = torch.full((batch_size,), self._ABLATION_MAP.get(self._ablation_name, 0), dtype=torch.long, device=self._device)
        return task_idx, target_idx, horizon_idx, ablation_idx

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

    def _loss(self, torch, outputs, target):
        if self._binary_target:
            logits = outputs["binary"]
            probs = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            brier = torch.mean((probs - target) ** 2)
            return bce + 0.1 * brier
        if self._target_name == "investors_count":
            pred = torch.relu(outputs["count"])
            base = torch.nn.functional.smooth_l1_loss(pred, target)
            neg_penalty = torch.relu(-outputs["count"]).mean()
            return base + 0.05 * neg_penalty
        pred = outputs["continuous"]
        base = torch.nn.functional.smooth_l1_loss(pred, target)
        pred_pos = torch.relu(pred)
        target_pos = torch.relu(target)
        align = torch.nn.functional.smooth_l1_loss(
            torch.log1p(pred_pos),
            torch.log1p(target_pos),
        )
        return base + 0.1 * align

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

        train_x = torch.tensor(np.stack(entity_windows.train_x), dtype=torch.float32)
        train_y = torch.tensor(np.stack(entity_windows.train_y), dtype=torch.float32)
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

        self._network = V740AlphaNet(
            in_channels=train_x.shape[1],
            seq_len=train_x.shape[2],
            horizon=self._horizon,
            hidden_dim=self.hidden_dim,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_x,
                train_y,
                train_edgar_recent,
                train_edgar_bucket,
                train_text_recent,
                train_text_bucket,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(self.max_epochs):
            self._network.train()
            losses = []
            for xb, yb, er, eb, tr, tb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                er = er.to(self._device)
                eb = eb.to(self._device)
                tr = tr.to(self._device)
                tb = tb.to(self._device)
                task_idx, target_idx, horizon_idx, ablation_idx = self._token_tensor(torch, len(xb))
                outputs = self._network(
                    xb,
                    task_idx,
                    target_idx,
                    horizon_idx,
                    ablation_idx,
                    edgar_recent=er,
                    edgar_bucket=eb,
                    text_recent=tr,
                    text_bucket=tb,
                )
                loss = self._loss(torch, outputs, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            if val_x is None or val_y is None:
                continue

            self._network.eval()
            with torch.no_grad():
                task_idx, target_idx, horizon_idx, ablation_idx = self._token_tensor(torch, len(val_x))
                val_out = self._network(
                    val_x.to(self._device),
                    task_idx,
                    target_idx,
                    horizon_idx,
                    ablation_idx,
                    edgar_recent=val_edgar_recent.to(self._device) if val_edgar_recent is not None else None,
                    edgar_bucket=val_edgar_bucket.to(self._device) if val_edgar_bucket is not None else None,
                    text_recent=val_text_recent.to(self._device) if val_text_recent is not None else None,
                    text_bucket=val_text_bucket.to(self._device) if val_text_bucket is not None else None,
                )
                val_loss = float(self._loss(torch, val_out, val_y.to(self._device)).cpu())

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
                task_idx, target_idx, horizon_idx, ablation_idx = self._token_tensor(torch, len(unique_entities))
                outputs = self._network(
                    x_batch,
                    task_idx,
                    target_idx,
                    horizon_idx,
                    ablation_idx,
                    edgar_recent=edgar_recent,
                    edgar_bucket=edgar_bucket,
                    text_recent=text_recent,
                    text_bucket=text_bucket,
                )
            if self._binary_target:
                raw = torch.sigmoid(outputs["binary"]).cpu().numpy()
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
