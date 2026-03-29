#!/usr/bin/env python3
"""
OLinear local integration scaffold for Block 3.

This wrapper intentionally targets the base `OLinear` path first, not the
heavier `OLinear-C` variant. The goal is to make the dataset-specific
orthogonal transformation artifacts explicit and reproducible on Block 3,
then validate the model through local smokes before any benchmark expansion.
"""
from __future__ import annotations

import importlib
import logging
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .optional_runtime import ensure_olinear_repo_on_path
from .samformer_model import (
    _build_entity_windows,
    _detect_binary,
    _detect_nonnegative,
    _sanitize_predictions,
    _select_feature_cols,
)

logger = logging.getLogger(__name__)


def _import_olinear_model_cls():
    repo = ensure_olinear_repo_on_path()
    if not repo.exists():
        raise ImportError(
            "OLinear official repo not found. Set BLOCK3_OLINEAR_REPO or place "
            "the audited repo under ~/.cache/block3_optional_repos/OLinear"
        )

    vendor_prefixes = (
        "model",
        "layers",
        "utils",
        "data_provider",
        "experiments",
    )
    vendor_keys = [
        key for key in list(sys.modules)
        if any(key == p or key.startswith(f"{p}.") for p in vendor_prefixes)
    ]
    original_modules = {key: sys.modules[key] for key in vendor_keys}
    for key in vendor_keys:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(repo))

    last_error: Exception | None = None
    try:
        try:
            module = importlib.import_module("model.OLinear")
            cls = getattr(module, "Model", None)
            if cls is not None:
                return cls
        except Exception as exc:  # pragma: no cover - vendor surface can change
            last_error = exc
        raise ImportError(f"Unable to import OLinear from vendor repo: {last_error}")
    finally:
        for key in [
            k for k in list(sys.modules)
            if any(k == p or k.startswith(f"{p}.") for p in vendor_prefixes)
        ]:
            sys.modules.pop(key, None)
        sys.modules.update(original_modules)
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass


def _safe_corrcoef(mat: np.ndarray, expected_dim: int) -> np.ndarray:
    """Compute a numerically safe correlation matrix."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_dim or arr.shape[0] < 2:
        return np.eye(expected_dim, dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(arr, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    if corr.shape != (expected_dim, expected_dim):
        return np.eye(expected_dim, dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32, copy=False)


def _orthogonal_basis_from_corr(corr: np.ndarray) -> np.ndarray:
    """Turn a symmetric correlation matrix into an orthogonal basis."""
    corr = np.asarray(corr, dtype=np.float64)
    corr = 0.5 * (corr + corr.T)
    corr += np.eye(corr.shape[0], dtype=np.float64) * 1e-6
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    q = vecs[:, order]
    q = np.real_if_close(q).astype(np.float32, copy=False)
    if q.shape[0] != q.shape[1]:
        return np.eye(corr.shape[0], dtype=np.float32)
    return q


def _build_olinear_artifacts(
    train_x: Sequence[np.ndarray],
    train_y: Sequence[np.ndarray],
    out_dir: Path,
    seq_len: int,
    pred_len: int,
) -> tuple[Path, Path]:
    """Create `Q_mat` and `Q_out_mat` artifacts for Block 3 windows."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_x:
        x_stack = np.stack(train_x).astype(np.float32)  # [B, C, T]
        x_for_corr = np.transpose(x_stack, (0, 1, 2)).reshape(-1, seq_len)
    else:
        x_for_corr = np.empty((0, seq_len), dtype=np.float32)
    q_mat = _orthogonal_basis_from_corr(_safe_corrcoef(x_for_corr, expected_dim=seq_len))

    if train_y:
        y_stack = np.stack(train_y).astype(np.float32)  # [B, H]
        y_for_corr = y_stack.reshape(-1, pred_len)
    else:
        y_for_corr = np.empty((0, pred_len), dtype=np.float32)
    q_out_mat = _orthogonal_basis_from_corr(_safe_corrcoef(y_for_corr, expected_dim=pred_len))

    q_mat_path = out_dir / f"block3_q_mat_seq{seq_len}.npy"
    q_out_path = out_dir / f"block3_q_out_mat_pred{pred_len}.npy"
    np.save(q_mat_path, q_mat)
    np.save(q_out_path, q_out_mat)
    return q_mat_path, q_out_path


class OLinearWrapper(ModelBase):
    """Local-only OLinear wrapper pending audited smoke and narrow clear."""

    def __init__(
        self,
        input_size: int = 60,
        horizon: int = 30,
        max_covariates: int = 8,
        max_entities: int = 512,
        max_windows: int = 8192,
        step: int = 1,
        val_frac: float = 0.15,
        d_model: int = 128,
        d_ff: int = 128,
        e_layers: int = 2,
        embed_size: int = 8,
        temp_patch_len: int = 16,
        temp_stride: int = 8,
        dropout: float = 0.0,
        lr: float = 5e-4,
        batch_size: int = 64,
        max_epochs: int = 5,
        patience: int = 2,
        seed: int = 42,
        device: str | None = None,
        artifact_dir: str | None = None,
        **kwargs,
    ):
        config = ModelConfig(
            name="OLinear",
            model_type="forecasting",
            params={
                "input_size": input_size,
                "horizon": horizon,
                "max_covariates": max_covariates,
                "d_model": d_model,
                "d_ff": d_ff,
                "e_layers": e_layers,
                "embed_size": embed_size,
                "temp_patch_len": temp_patch_len,
                "temp_stride": temp_stride,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
            },
            optional_dependency="OLinear vendor repo + torch + generated Q matrices",
        )
        super().__init__(config)
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.max_covariates = int(max_covariates)
        self.max_entities = int(max_entities)
        self.max_windows = int(max_windows)
        self.step = int(step)
        self.val_frac = float(val_frac)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.e_layers = int(e_layers)
        self.embed_size = int(embed_size)
        self.temp_patch_len = int(temp_patch_len)
        self.temp_stride = int(temp_stride)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device_override = device
        self.artifact_dir = artifact_dir
        self.extra_kwargs = kwargs

        self._network = None
        self._device = None
        self._contexts: Dict[str, np.ndarray] = {}
        self._fallback_value = 0.0
        self._binary_target = False
        self._nonnegative_target = False
        self._feature_cols: List[str] = []
        self._artifact_root: Path | None = None

    def _make_vendor_config(self, c_in: int, q_mat_path: Path, q_out_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            pred_len=self.horizon,
            enc_in=c_in,
            seq_len=self.input_size,
            d_model=self.d_model,
            d_ff=self.d_ff,
            Q_chan_indep=0,
            q_mat_file=str(q_mat_path),
            q_out_mat_file=str(q_out_path),
            Q_MAT_file=None,
            Q_OUT_MAT_file=None,
            root_path=str(self._artifact_root or q_mat_path.parent),
            temp_patch_len=self.temp_patch_len,
            temp_stride=self.temp_stride,
            embed_size=self.embed_size,
            dropout=self.dropout,
            activation="gelu",
            e_layers=self.e_layers,
            CKA_flag=0,
            n_heads=2,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "OLinearWrapper":
        import torch
        import torch.nn.functional as F

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        self._fallback_value = float(pd.to_numeric(y, errors="coerce").dropna().median()) if len(y) else 0.0
        y_np = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()
        self._binary_target = _detect_binary(y_np)
        self._nonnegative_target = (not self._binary_target) and _detect_nonnegative(y_np)

        if train_raw is None or target is None or "entity_id" not in train_raw.columns:
            logger.warning("  [OLinear] Missing train_raw/entity_id, using fallback-only mode")
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
            logger.warning("  [OLinear] No training windows, using fallback-only mode")
            self._fitted = True
            return self

        artifact_root = (
            Path(self.artifact_dir).expanduser().resolve()
            if self.artifact_dir
            else Path(tempfile.mkdtemp(prefix="block3_olinear_"))
        )
        self._artifact_root = artifact_root
        q_mat_path, q_out_path = _build_olinear_artifacts(
            windows.train_x,
            windows.train_y,
            out_dir=artifact_root,
            seq_len=self.input_size,
            pred_len=self.horizon,
        )

        train_x = np.stack(windows.train_x).astype(np.float32)  # [B, C, T]
        train_x = np.transpose(train_x, (0, 2, 1))  # [B, T, C]
        train_y = np.stack(windows.train_y).astype(np.float32)
        val_x = None
        val_y = None
        if windows.val_x:
            val_x = np.transpose(np.stack(windows.val_x).astype(np.float32), (0, 2, 1))
            val_y = np.stack(windows.val_y).astype(np.float32)

        self._device = torch.device(self.device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        OLinearModel = _import_olinear_model_cls()
        vendor_cfg = self._make_vendor_config(c_in=int(train_x.shape[-1]), q_mat_path=q_mat_path, q_out_path=q_out_path)
        self._network = OLinearModel(vendor_cfg).to(self._device)
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)

        best_state = None
        best_val = math.inf
        bad_epochs = 0

        logger.info(
            "  [OLinear] windows train=%d val=%d channels=%d artifact_root=%s",
            len(train_x),
            0 if val_x is None else len(val_x),
            int(train_x.shape[-1]),
            artifact_root,
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
                "  [OLinear] epoch=%d/%d train_loss=%.6f val_loss=%.6f",
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
            raise ValueError("OLinearWrapper is not fitted")

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

            x_batch = np.stack([self._contexts[eid].T for eid in unique_entities]).astype(np.float32)
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
        return _sanitize_predictions(out, self._fallback_value, "OLinear")


def create_olinear(**kwargs) -> ModelBase:
    model = OLinearWrapper(**kwargs)
    model.config = ModelConfig(
        name="OLinear",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="OLinear vendor repo + torch + generated Q matrices",
    )
    return model


__all__ = [
    "OLinearWrapper",
    "create_olinear",
]
