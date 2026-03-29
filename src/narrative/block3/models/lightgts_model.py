#!/usr/bin/env python3
"""
LightGTS local integration scaffold for Block 3.

This file intentionally stays out of the active registry until it passes:
  1. vendor import smoke,
  2. tiny local audited smoke,
  3. narrow benchmark-clear.

The goal here is not to overclaim benchmark readiness; it is to turn the
verified official repo audit into a real Block 3 wrapper surface that can be
smoked and iterated on without mutating the shared benchmark lanes.
"""
from __future__ import annotations

import importlib
import logging
import math
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .optional_runtime import ensure_lightgts_repo_on_path
from .samformer_model import (
    _build_entity_windows,
    _detect_binary,
    _detect_nonnegative,
    _sanitize_predictions,
    _select_feature_cols,
)

logger = logging.getLogger(__name__)


def _import_lightgts():
    repo = ensure_lightgts_repo_on_path()
    if not repo.exists():
        raise ImportError(
            "LightGTS official repo not found. Set BLOCK3_LIGHTGTS_REPO or place "
            "the audited repo under ~/.cache/block3_optional_repos/LightGTS"
        )

    vendor_keys = [k for k in sys.modules if k == "src" or k.startswith("src.")]
    original_modules = {k: sys.modules[k] for k in vendor_keys}
    for key in vendor_keys:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(repo))

    last_error: Exception | None = None
    try:
        for module_name in ("src.models.LightGTS_resample", "src.models.LightGTS"):
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, "LightGTS", None)
                if cls is not None:
                    return cls
            except Exception as exc:  # pragma: no cover - import surface varies by repo rev
                last_error = exc
                continue
        raise ImportError(f"Unable to import LightGTS from vendor repo: {last_error}")
    finally:
        for key in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            sys.modules.pop(key, None)
        sys.modules.update(original_modules)
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass


def _patchify_context(x: np.ndarray, patch_len: int, stride: int) -> np.ndarray:
    """Convert [channels, time] into [num_patch, channels, patch_len]."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D [channels,time] context, got shape={arr.shape}")

    c, t = arr.shape
    if t < patch_len:
        pad = patch_len - t
        arr = np.pad(arr, ((0, 0), (pad, 0)), mode="constant")
        t = arr.shape[1]

    starts = list(range(0, max(t - patch_len, 0) + 1, stride))
    if not starts:
        starts = [0]
    final_start = max(t - patch_len, 0)
    if starts[-1] != final_start:
        starts.append(final_start)

    patches = [arr[:, s : s + patch_len] for s in starts]
    return np.stack(patches, axis=0).astype(np.float32, copy=False)


def _stack_patches(windows: Sequence[np.ndarray], patch_len: int, stride: int) -> np.ndarray:
    return np.stack([_patchify_context(x, patch_len=patch_len, stride=stride) for x in windows])


class LightGTSWrapper(ModelBase):
    """Local-only LightGTS wrapper pending audited smoke and narrow clear."""

    def __init__(
        self,
        input_size: int = 96,
        horizon: int = 30,
        patch_len: int = 24,
        stride: int = 24,
        max_covariates: int = 8,
        max_entities: int = 512,
        max_windows: int = 8192,
        step: int = 1,
        val_frac: float = 0.15,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        d_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        head_dropout: float = 0.0,
        lr: float = 3e-4,
        batch_size: int = 64,
        max_epochs: int = 5,
        patience: int = 2,
        seed: int = 42,
        device: str | None = None,
        **kwargs,
    ):
        config = ModelConfig(
            name="LightGTS",
            model_type="forecasting",
            params={
                "input_size": input_size,
                "horizon": horizon,
                "patch_len": patch_len,
                "stride": stride,
                "max_covariates": max_covariates,
                "d_model": d_model,
                "d_ff": d_ff,
                "e_layers": e_layers,
                "d_layers": d_layers,
                "n_heads": n_heads,
                "dropout": dropout,
                "attn_dropout": attn_dropout,
                "head_dropout": head_dropout,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
            },
            optional_dependency="LightGTS vendor repo + torch",
        )
        super().__init__(config)
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.max_covariates = int(max_covariates)
        self.max_entities = int(max_entities)
        self.max_windows = int(max_windows)
        self.step = int(step)
        self.val_frac = float(val_frac)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.e_layers = int(e_layers)
        self.d_layers = int(d_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.head_dropout = float(head_dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device_override = device
        self.extra_kwargs = kwargs

        self._network = None
        self._device = None
        self._contexts: Dict[str, np.ndarray] = {}
        self._fallback_value = 0.0
        self._binary_target = False
        self._nonnegative_target = False
        self._feature_cols: List[str] = []
        self._patch_count = 0

    def _make_network(self, c_in: int, horizon: int, num_patch: int):
        LightGTS = _import_lightgts()
        return LightGTS(
            c_in=c_in,
            target_dim=horizon,
            patch_len=self.patch_len,
            stride=self.stride,
            num_patch=num_patch,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            head_dropout=self.head_dropout,
            res_attention=False,
            learn_pe=False,
            head_type="prediction",
            **self.extra_kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LightGTSWrapper":
        import torch
        import torch.nn.functional as F

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        self._fallback_value = float(pd.to_numeric(y, errors="coerce").dropna().median()) if len(y) else 0.0
        self._binary_target = _detect_binary(pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy())
        self._nonnegative_target = (not self._binary_target) and _detect_nonnegative(
            pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()
        )

        if train_raw is None or target is None or "entity_id" not in train_raw.columns:
            logger.warning("  [LightGTS] Missing train_raw/entity_id, using fallback-only mode")
            self._fitted = True
            return self

        self._feature_cols = _select_feature_cols(train_raw, target, self.max_covariates)
        windows = _build_entity_windows(
            train_raw=train_raw,
            target=target,
            feature_cols=self._feature_cols,
            input_size=self.input_size,
            horizon=self.horizon,
            step=self.step,
            max_entities=self.max_entities,
            max_windows=self.max_windows,
            val_frac=self.val_frac,
            seed=self.seed,
        )
        self._contexts = windows.contexts
        if not windows.train_x:
            logger.warning("  [LightGTS] No training windows, using fallback-only mode")
            self._fitted = True
            return self

        train_x = _stack_patches(windows.train_x, patch_len=self.patch_len, stride=self.stride)
        train_y = np.stack(windows.train_y).astype(np.float32)
        val_x = _stack_patches(windows.val_x, patch_len=self.patch_len, stride=self.stride) if windows.val_x else None
        val_y = np.stack(windows.val_y).astype(np.float32) if windows.val_y else None

        self._patch_count = int(train_x.shape[1])
        c_in = int(train_x.shape[2])

        self._device = torch.device(
            self.device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._network = self._make_network(c_in=c_in, horizon=self.horizon, num_patch=self._patch_count).to(self._device)
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)

        best_state = None
        best_val = math.inf
        bad_epochs = 0

        logger.info(
            "  [LightGTS] windows train=%d val=%d channels=%d num_patch=%d",
            len(train_x),
            0 if val_x is None else len(val_x),
            c_in,
            self._patch_count,
        )

        for epoch in range(self.max_epochs):
            self._network.train()
            perm = np.random.permutation(len(train_x))
            losses: List[float] = []
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.tensor(train_x[idx], dtype=torch.float32, device=self._device)
                yb = torch.tensor(train_y[idx], dtype=torch.float32, device=self._device)

                optimizer.zero_grad(set_to_none=True)
                pred = self._network(xb)[:, : self.horizon, 0]
                if self._binary_target:
                    loss = F.binary_cross_entropy_with_logits(pred, yb)
                else:
                    loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_loss = float(np.mean(losses)) if losses else math.inf
            if val_x is not None and val_y is not None and len(val_x):
                self._network.eval()
                with torch.no_grad():
                    xb = torch.tensor(val_x, dtype=torch.float32, device=self._device)
                    yb = torch.tensor(val_y, dtype=torch.float32, device=self._device)
                    pred = self._network(xb)[:, : self.horizon, 0]
                    if self._binary_target:
                        vloss = F.binary_cross_entropy_with_logits(pred, yb)
                    else:
                        vloss = F.mse_loss(pred, yb)
                    val_loss = float(vloss.detach().cpu())

            logger.info(
                "  [LightGTS] epoch=%d/%d train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                self.max_epochs,
                float(np.mean(losses)) if losses else math.inf,
                val_loss,
            )

            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._network.state_dict().items()}
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
            raise ValueError("LightGTSWrapper is not fitted")

        h = len(X)
        if self._network is None or not self._contexts:
            return np.full(h, self._fallback_value, dtype=np.float64)

        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        req_horizon = int(kwargs.get("horizon", self.horizon))
        if test_raw is None or "entity_id" not in test_raw.columns:
            return np.full(h, self._fallback_value, dtype=np.float64)

        if target and target in test_raw.columns:
            valid_mask = test_raw[target].notna()
            test_entities = test_raw.loc[valid_mask, "entity_id"].values
        else:
            test_entities = test_raw["entity_id"].values
        if len(test_entities) != h:
            return np.full(h, self._fallback_value, dtype=np.float64)

        unique_entities = []
        seen = set()
        for eid in test_entities:
            sid = str(eid)
            if sid in self._contexts and sid not in seen:
                seen.add(sid)
                unique_entities.append(sid)

        entity_forecasts: Dict[str, float] = {}
        if unique_entities:
            import torch

            x_batch = np.stack(
                [_patchify_context(self._contexts[eid], patch_len=self.patch_len, stride=self.stride) for eid in unique_entities]
            ).astype(np.float32)
            with torch.no_grad():
                preds = self._network(
                    torch.tensor(x_batch, dtype=torch.float32, device=self._device)
                ).detach().cpu().numpy()
            idx = min(req_horizon - 1, preds.shape[1] - 1)
            point_preds = preds[:, idx, 0]
            if self._binary_target:
                point_preds = 1.0 / (1.0 + np.exp(-point_preds))
            if self._nonnegative_target:
                point_preds = np.clip(point_preds, 0.0, None)
            entity_forecasts = {eid: float(pred) for eid, pred in zip(unique_entities, point_preds)}

        out = np.empty(h, dtype=np.float64)
        for i, eid in enumerate(test_entities):
            out[i] = entity_forecasts.get(str(eid), self._fallback_value)
        return _sanitize_predictions(out, self._fallback_value, "LightGTS")


def create_lightgts(**kwargs) -> ModelBase:
    model = LightGTSWrapper(**kwargs)
    model.config = ModelConfig(
        name="LightGTS",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="LightGTS vendor repo + torch",
    )
    return model


__all__ = ["LightGTSWrapper", "create_lightgts"]
