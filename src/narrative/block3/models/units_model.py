#!/usr/bin/env python3
"""
UniTS forecasting-only local integration scaffold for Block 3.

The official UniTS repo is a broader unified multi-task framework. For Block 3
we start with the narrowest useful slice:
  1. vendor repo audit
  2. forecasting-only import
  3. tiny real-data smoke
  4. narrow benchmark-clear if the first smoke is healthy

This wrapper intentionally does not claim full UniTS task coverage yet.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .optional_runtime import ensure_units_repo_on_path
from .samformer_model import (
    _build_entity_windows,
    _detect_binary,
    _detect_nonnegative,
    _sanitize_predictions,
    _select_feature_cols,
)

logger = logging.getLogger(__name__)


def _import_units_model():
    repo = ensure_units_repo_on_path()
    if not repo.exists():
        raise ImportError(
            "UniTS official repo not found. Set BLOCK3_UNITS_REPO or place "
            "the audited repo under ~/.cache/block3_optional_repos/UniTS"
        )
    file_path = repo / "models" / "UniTS.py"
    if not file_path.exists():
        raise ImportError(f"UniTS repo missing expected model file: {file_path}")

    module_name = "block3_vendor_units_model"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for UniTS from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "Model")


def _make_args(
    *,
    prompt_num: int,
    d_model: int,
    patch_len: int,
    stride: int,
    e_layers: int,
    n_heads: int,
    dropout: float,
):
    return SimpleNamespace(
        prompt_num=int(prompt_num),
        d_model=int(d_model),
        patch_len=int(patch_len),
        stride=int(stride),
        e_layers=int(e_layers),
        n_heads=int(n_heads),
        dropout=float(dropout),
    )


class UniTSWrapper(ModelBase):
    """Forecasting-only UniTS wrapper for Block 3 local clears."""

    def __init__(
        self,
        input_size: int = 60,
        horizon: int = 30,
        max_covariates: int = 8,
        max_entities: int = 512,
        max_windows: int = 8192,
        step: int = 1,
        val_frac: float = 0.15,
        prompt_num: int = 4,
        d_model: int = 64,
        patch_len: int = 10,
        stride: int = 10,
        e_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 5,
        patience: int = 2,
        seed: int = 42,
        device: str | None = None,
        **kwargs,
    ):
        config = ModelConfig(
            name="UniTS",
            model_type="forecasting",
            params=kwargs,
            optional_dependency="UniTS vendor repo + torch + timm",
        )
        super().__init__(config)
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.max_covariates = int(max_covariates)
        self.max_entities = int(max_entities)
        self.max_windows = int(max_windows)
        self.step = int(step)
        self.val_frac = float(val_frac)
        self.prompt_num = int(prompt_num)
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.e_layers = int(e_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
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
        self._feature_dim = 1

    def _make_network(self, n_vars: int):
        import torch

        UniTSModel = _import_units_model()
        args = _make_args(
            prompt_num=self.prompt_num,
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        configs_list = [
            (
                "block3_forecast",
                {
                    "task_name": "long_term_forecast",
                    "dataset": "block3",
                    "seq_len": self.input_size,
                    "label_len": 0,
                    "pred_len": self.horizon,
                    "enc_in": n_vars,
                    "c_out": n_vars,
                },
            )
        ]
        torch.manual_seed(self.seed)
        return UniTSModel(args=args, configs_list=configs_list, pretrain=False)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "UniTSWrapper":
        import torch
        import torch.nn.functional as F

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        self._fallback_value = float(pd.to_numeric(y, errors="coerce").dropna().median()) if len(y) else 0.0
        y_np = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()
        self._binary_target = _detect_binary(y_np)
        self._nonnegative_target = (not self._binary_target) and _detect_nonnegative(y_np)

        if train_raw is None or target is None or "entity_id" not in train_raw.columns:
            logger.warning("  [UniTS] Missing train_raw/entity_id, using fallback-only mode")
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
            logger.warning("  [UniTS] No training windows, using fallback-only mode")
            self._fitted = True
            return self

        train_x = np.transpose(np.stack(windows.train_x).astype(np.float32), (0, 2, 1))
        train_y = np.stack(windows.train_y).astype(np.float32)
        val_x = np.transpose(np.stack(windows.val_x).astype(np.float32), (0, 2, 1)) if windows.val_x else None
        val_y = np.stack(windows.val_y).astype(np.float32) if windows.val_y else None

        self._feature_dim = int(train_x.shape[-1])
        self._device = torch.device(self.device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._network = self._make_network(n_vars=self._feature_dim).to(self._device)
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)
        best_val = np.inf
        best_state = None
        bad_epochs = 0

        x_t = torch.tensor(train_x, dtype=torch.float32)
        y_t = torch.tensor(train_y, dtype=torch.float32)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        def _loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            if self._binary_target:
                return F.binary_cross_entropy_with_logits(pred, tgt)
            if target == "investors_count":
                return F.smooth_l1_loss(torch.relu(pred), tgt)
            return F.smooth_l1_loss(pred, tgt)

        for epoch in range(self.max_epochs):
            self._network.train()
            for xb, yb in dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                x_mark = torch.zeros((xb.shape[0], xb.shape[1], 1), dtype=xb.dtype, device=xb.device)
                pred = self._network(
                    xb,
                    x_mark,
                    task_id=0,
                    task_name="long_term_forecast",
                )[:, : self.horizon, 0]
                loss = _loss_fn(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
                optimizer.step()

            if val_x is None or val_y is None or not len(val_x):
                continue

            self._network.eval()
            with torch.no_grad():
                vx = torch.tensor(val_x, dtype=torch.float32, device=self._device)
                vy = torch.tensor(val_y, dtype=torch.float32, device=self._device)
                vmark = torch.zeros((vx.shape[0], vx.shape[1], 1), dtype=vx.dtype, device=vx.device)
                vpred = self._network(
                    vx,
                    vmark,
                    task_id=0,
                    task_name="long_term_forecast",
                )[:, : self.horizon, 0]
                val_loss = float(_loss_fn(vpred, vy).detach().cpu())

            logger.info(
                "  [UniTS] epoch=%d/%d val_loss=%.6f",
                epoch + 1,
                self.max_epochs,
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
            raise ValueError("UniTSWrapper is not fitted")

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

        entity_forecasts: Dict[str, float] = {}
        if unique_entities:
            import torch

            xb = np.transpose(
                np.stack([self._contexts[eid] for eid in unique_entities]).astype(np.float32),
                (0, 2, 1),
            )
            xb_t = torch.tensor(xb, dtype=torch.float32, device=self._device)
            x_mark = torch.zeros((xb_t.shape[0], xb_t.shape[1], 1), dtype=xb_t.dtype, device=xb_t.device)
            with torch.no_grad():
                preds = self._network(
                    xb_t,
                    x_mark,
                    task_id=0,
                    task_name="long_term_forecast",
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
        return _sanitize_predictions(out, self._fallback_value, "UniTS")


def create_units(**kwargs) -> ModelBase:
    model = UniTSWrapper(**kwargs)
    model.config = ModelConfig(
        name="UniTS",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="UniTS vendor repo + torch + timm",
    )
    return model


__all__ = ["UniTSWrapper", "create_units"]
