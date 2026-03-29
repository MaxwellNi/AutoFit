#!/usr/bin/env python3
"""
ElasTST local integration scaffold for Block 3.

This wrapper intentionally stays lightweight and local-first:
  1. vendor repo audit
  2. import smoke
  3. tiny real-data smoke
  4. narrow benchmark-clear

It uses the official ProbTS ElasTST backbone, but avoids importing the full
`probts` package surface because that package-level import currently drags in
data-manager code paths that are not required for Block 3 local clears.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .optional_runtime import ensure_insider_libstdcpp, get_probts_repo_dir
from .samformer_model import (
    _build_entity_windows,
    _detect_binary,
    _detect_nonnegative,
    _sanitize_predictions,
    _select_feature_cols,
)

logger = logging.getLogger(__name__)


def _ensure_namespace(name: str, path: Path | None = None):
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = mod
    return mod


def _load_vendor_module(module_name: str, file_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_elastst_backbone():
    repo = get_probts_repo_dir()
    if not repo.exists():
        raise ImportError(
            "ProbTS official repo not found. Set BLOCK3_PROBTS_REPO or place "
            "the audited repo under ~/.cache/block3_optional_repos/ProbTS"
        )
    ensure_insider_libstdcpp()
    base = repo / "probts"
    if not base.exists():
        raise ImportError(f"ProbTS repo missing expected probts package at {base}")

    _ensure_namespace("probts", base)
    _ensure_namespace("probts.utils", base / "utils")
    _ensure_namespace("probts.model", base / "model")
    _ensure_namespace("probts.model.nn", base / "model" / "nn")
    _ensure_namespace("probts.model.nn.arch", base / "model" / "nn" / "arch")
    _ensure_namespace(
        "probts.model.nn.arch.ElasTSTModule",
        base / "model" / "nn" / "arch" / "ElasTSTModule",
    )

    _load_vendor_module("probts.utils.position_emb", base / "utils" / "position_emb.py")
    _load_vendor_module(
        "probts.model.nn.arch.ElasTSTModule.TRoPE",
        base / "model" / "nn" / "arch" / "ElasTSTModule" / "TRoPE.py",
    )
    _load_vendor_module(
        "probts.model.nn.arch.ElasTSTModule.Modules",
        base / "model" / "nn" / "arch" / "ElasTSTModule" / "Modules.py",
    )
    _load_vendor_module(
        "probts.model.nn.arch.ElasTSTModule.SubLayers",
        base / "model" / "nn" / "arch" / "ElasTSTModule" / "SubLayers.py",
    )
    _load_vendor_module(
        "probts.model.nn.arch.ElasTSTModule.Layers",
        base / "model" / "nn" / "arch" / "ElasTSTModule" / "Layers.py",
    )
    backbone_mod = _load_vendor_module(
        "probts.model.nn.arch.ElasTSTModule.ElasTST_backbone",
        base / "model" / "nn" / "arch" / "ElasTSTModule" / "ElasTST_backbone.py",
    )
    return getattr(backbone_mod, "ElasTST_backbone")


def _parse_patch_sizes(spec: str | int | List[int]) -> List[int]:
    if isinstance(spec, int):
        return [int(spec)]
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    return [int(x) for x in str(spec).split("_") if str(x).strip()]


class ElasTSTWrapper(ModelBase):
    """Local-only ElasTST wrapper pending audited smoke and narrow clear."""

    def __init__(
        self,
        input_size: int = 96,
        horizon: int = 30,
        max_covariates: int = 8,
        max_entities: int = 512,
        max_windows: int = 8192,
        step: int = 1,
        val_frac: float = 0.15,
        l_patch_size: str = "8_16_32",
        stride: int | None = None,
        t_layers: int = 2,
        v_layers: int = 0,
        n_heads: int = 8,
        d_k: int = 16,
        d_v: int = 16,
        d_inner: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.0,
        rotate: bool = True,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 5,
        patience: int = 2,
        seed: int = 42,
        device: str | None = None,
        **kwargs,
    ):
        config = ModelConfig(
            name="ElasTST",
            model_type="forecasting",
            params=kwargs,
            optional_dependency="ProbTS vendor repo + torch + einops",
        )
        super().__init__(config)
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.max_covariates = int(max_covariates)
        self.max_entities = int(max_entities)
        self.max_windows = int(max_windows)
        self.step = int(step)
        self.val_frac = float(val_frac)
        self.l_patch_size = _parse_patch_sizes(l_patch_size)
        self.stride = None if stride is None else int(stride)
        self.t_layers = int(t_layers)
        self.v_layers = int(v_layers)
        self.n_heads = int(n_heads)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.d_inner = int(d_inner)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.rotate = bool(rotate)
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
        self._feature_cols: List[str] = []
        self._fallback_value = 0.0
        self._binary_target = False
        self._nonnegative_target = False

    def _make_network(self):
        ElasTSTBackbone = _import_elastst_backbone()
        return ElasTSTBackbone(
            l_patch_size=self.l_patch_size,
            stride=self.stride,
            k_patch_size=1,
            # Official ElasTST treats the input as a single-channel
            # time-by-variable image and patches over the variable axis.
            in_channels=1,
            t_layers=self.t_layers,
            v_layers=self.v_layers,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_inner=self.d_inner,
            dropout=self.dropout,
            rotate=self.rotate,
            max_seq_len=max(self.input_size + self.horizon, 1024),
            structured_mask=True,
            **self.extra_kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ElasTSTWrapper":
        import torch
        import torch.nn.functional as F

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        self._fallback_value = float(pd.to_numeric(y, errors="coerce").dropna().median()) if len(y) else 0.0
        y_np = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()
        self._binary_target = _detect_binary(y_np)
        self._nonnegative_target = (not self._binary_target) and _detect_nonnegative(y_np)

        if train_raw is None or target is None or "entity_id" not in train_raw.columns:
            logger.warning("  [ElasTST] Missing train_raw/entity_id, using fallback-only mode")
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
            logger.warning("  [ElasTST] No training windows, using fallback-only mode")
            self._fitted = True
            return self

        train_x = np.transpose(np.stack(windows.train_x).astype(np.float32), (0, 2, 1))  # [B, L, C]
        train_y = np.stack(windows.train_y).astype(np.float32)  # [B, H]
        val_x = np.transpose(np.stack(windows.val_x).astype(np.float32), (0, 2, 1)) if windows.val_x else None
        val_y = np.stack(windows.val_y).astype(np.float32) if windows.val_y else None

        self._device = torch.device(self.device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._network = self._make_network().to(self._device)
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)

        x_t = torch.tensor(train_x, dtype=torch.float32)
        y_t = torch.tensor(train_y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(x_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for _epoch in range(self.max_epochs):
            self._network.train()
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                future_placeholder = torch.zeros((xb.shape[0], self.horizon, xb.shape[-1]), dtype=xb.dtype, device=xb.device)
                past_obs = torch.ones_like(xb)
                future_obs = torch.zeros_like(future_placeholder)
                out, _ = self._network(xb, future_placeholder, past_obs, future_obs, dataset_name=None)
                pred = out[:, : self.horizon, 0]
                if self._binary_target:
                    loss = F.binary_cross_entropy_with_logits(pred, yb)
                elif target == "investors_count":
                    loss = F.smooth_l1_loss(torch.relu(pred), yb)
                else:
                    loss = F.smooth_l1_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()

            if val_x is None or val_y is None:
                continue
            self._network.eval()
            with torch.no_grad():
                vx = torch.tensor(val_x, dtype=torch.float32, device=self._device)
                vy = torch.tensor(val_y, dtype=torch.float32, device=self._device)
                future_placeholder = torch.zeros((vx.shape[0], self.horizon, vx.shape[-1]), dtype=vx.dtype, device=vx.device)
                past_obs = torch.ones_like(vx)
                future_obs = torch.zeros_like(future_placeholder)
                out, _ = self._network(vx, future_placeholder, past_obs, future_obs, dataset_name=None)
                pred = out[:, : self.horizon, 0]
                if self._binary_target:
                    val_loss = float(F.binary_cross_entropy_with_logits(pred, vy).cpu())
                elif target == "investors_count":
                    val_loss = float(F.smooth_l1_loss(torch.relu(pred), vy).cpu())
                else:
                    val_loss = float(F.smooth_l1_loss(pred, vy).cpu())
            if val_loss < best_val - 1e-8:
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
            raise ValueError("ElasTSTWrapper is not fitted")

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

        unique_entities: List[str] = []
        seen = set()
        for eid in test_entities:
            sid = str(eid)
            if sid in self._contexts and sid not in seen:
                seen.add(sid)
                unique_entities.append(sid)

        preds_map: Dict[str, float] = {}
        if unique_entities:
            import torch

            xb = np.transpose(np.stack([self._contexts[eid] for eid in unique_entities]).astype(np.float32), (0, 2, 1))
            xb_t = torch.tensor(xb, dtype=torch.float32, device=self._device)
            future_placeholder = torch.zeros((xb_t.shape[0], self.horizon, xb_t.shape[-1]), dtype=xb_t.dtype, device=xb_t.device)
            past_obs = torch.ones_like(xb_t)
            future_obs = torch.zeros_like(future_placeholder)
            with torch.no_grad():
                out, _ = self._network(xb_t, future_placeholder, past_obs, future_obs, dataset_name=None)
            raw = out[:, : self.horizon, 0].detach().cpu().numpy()
            idx = min(req_horizon - 1, raw.shape[1] - 1)
            point = raw[:, idx]
            if self._binary_target:
                point = 1.0 / (1.0 + np.exp(-point))
            if self._nonnegative_target:
                point = np.clip(point, 0.0, None)
            preds_map = {eid: float(pred) for eid, pred in zip(unique_entities, point)}

        out = np.empty(h, dtype=np.float64)
        for i, eid in enumerate(test_entities):
            out[i] = preds_map.get(str(eid), self._fallback_value)
        return _sanitize_predictions(out, self._fallback_value, "ElasTST")


def create_elastst(**kwargs) -> ModelBase:
    model = ElasTSTWrapper(**kwargs)
    model.config = ModelConfig(
        name="ElasTST",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="ProbTS vendor repo + torch + einops",
    )
    return model


__all__ = ["ElasTSTWrapper", "create_elastst"]
