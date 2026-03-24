#!/usr/bin/env python3
"""
SAMformer wrapper for Block 3 benchmark.

Adapted from the official MIT-licensed implementation:
  SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting
  with Sharpness-Aware Minimization and Channel-Wise Attention
  ICML 2024 Oral
  https://github.com/romilbert/samformer

This integration is intentionally lightweight and benchmark-oriented:
  - one entity-aware panel forecaster
  - temporal validation split inside training windows
  - channel-wise attention over target history + top numeric covariates
  - single-model inference mapped back to test entities
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

logger = logging.getLogger(__name__)


_EXOG_EXCLUDE = {
    "entity_id", "crawled_date_day", "cik", "date",
    "offer_id", "snapshot_ts", "crawled_date", "processed_datetime",
    "funding_raised_usd", "funding_raised", "investors_count",
    "non_national_investors", "is_funded",
    "funding_goal_usd", "funding_goal",
    "funding_goal_maximum", "funding_goal_maximum_usd",
}


def _sanitize_predictions(
    preds: np.ndarray,
    fill_value: float,
    model_name: str,
) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not np.isfinite(fill_value):
        fill_value = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    n_bad = int((~finite).sum())
    arr = arr.copy()
    arr[~finite] = fill_value
    logger.warning(
        f"  [{model_name}] Sanitized {n_bad} non-finite predictions "
        f"(fill={fill_value:.6g})"
    )
    return arr


def _fill_numeric(values: Sequence[float]) -> np.ndarray:
    series = pd.Series(values, dtype="float64").ffill().bfill().fillna(0.0)
    return series.to_numpy(dtype=np.float32)


def _detect_binary(y: np.ndarray) -> bool:
    if len(y) == 0:
        return False
    uniq = np.unique(y[np.isfinite(y)])
    return len(uniq) <= 3 and set(np.round(uniq).tolist()).issubset({0.0, 1.0})


def _detect_nonnegative(y: np.ndarray) -> bool:
    if len(y) == 0:
        return False
    finite = y[np.isfinite(y)]
    return bool(len(finite) and (finite >= 0).all())


def _select_feature_cols(
    train_raw: pd.DataFrame,
    target: str,
    max_covariates: int,
) -> List[str]:
    num_cols = [
        c for c in train_raw.select_dtypes(include=[np.number]).columns
        if c not in (_EXOG_EXCLUDE | {target})
    ]
    if not num_cols:
        return []

    stats = []
    for col in num_cols:
        vals = pd.to_numeric(train_raw[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) < 32:
            continue
        var = float(finite.var())
        if var <= 1e-12:
            continue
        stats.append((col, var))

    stats.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in stats[:max_covariates]]


@dataclass
class _EntityWindows:
    train_x: List[np.ndarray]
    train_y: List[np.ndarray]
    val_x: List[np.ndarray]
    val_y: List[np.ndarray]
    contexts: Dict[str, np.ndarray]


def _build_entity_windows(
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
) -> _EntityWindows:
    train_x: List[np.ndarray] = []
    train_y: List[np.ndarray] = []
    val_x: List[np.ndarray] = []
    val_y: List[np.ndarray] = []
    contexts: Dict[str, np.ndarray] = {}

    if train_raw is None or "entity_id" not in train_raw.columns:
        return _EntityWindows(train_x, train_y, val_x, val_y, contexts)

    rng = np.random.RandomState(seed)
    groups = train_raw.groupby("entity_id", sort=False)
    for i, (eid, grp) in enumerate(groups):
        if i >= max_entities:
            break
        grp = grp.sort_values("crawled_date_day")
        if target not in grp.columns:
            continue

        y_arr = _fill_numeric(grp[target].values)
        if len(y_arr) < input_size + horizon:
            continue

        channels = [y_arr]
        for col in feature_cols:
            if col in grp.columns:
                channels.append(_fill_numeric(grp[col].values))
        series = np.stack(channels, axis=0)  # [C, T]
        contexts[str(eid)] = series[:, -input_size:].astype(np.float32, copy=False)

        entity_x: List[np.ndarray] = []
        entity_y: List[np.ndarray] = []
        limit = len(y_arr) - input_size - horizon + 1
        for t in range(0, limit, step):
            x_win = series[:, t : t + input_size]
            y_win = y_arr[t + input_size : t + input_size + horizon]
            if np.any(~np.isfinite(x_win)) or np.any(~np.isfinite(y_win)):
                continue
            entity_x.append(x_win.astype(np.float32, copy=False))
            entity_y.append(y_win.astype(np.float32, copy=False))

        if not entity_x:
            continue

        n_val = 0
        if len(entity_x) >= 4:
            n_val = max(1, int(round(len(entity_x) * val_frac)))
            n_val = min(n_val, len(entity_x) - 1)

        split = len(entity_x) - n_val
        train_x.extend(entity_x[:split])
        train_y.extend(entity_y[:split])
        val_x.extend(entity_x[split:])
        val_y.extend(entity_y[split:])

    def _cap(xs: List[np.ndarray], ys: List[np.ndarray], cap: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(xs) <= cap:
            return xs, ys
        idx = rng.choice(len(xs), size=cap, replace=False)
        idx.sort()
        return [xs[j] for j in idx], [ys[j] for j in idx]

    train_x, train_y = _cap(train_x, train_y, max_windows)
    if val_x:
        val_cap = max(2048, max_windows // 5)
        val_x, val_y = _cap(val_x, val_y, val_cap)

    return _EntityWindows(train_x, train_y, val_x, val_y, contexts)


class _RevIN:
    """Minimal RevIN adapted from the official SAMformer repository."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        import torch
        from torch import nn

        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.mean = None
        self.stdev = None

    def parameters(self):
        return [self.affine_weight, self.affine_bias]


def _build_torch_modules():
    import torch
    from torch import nn

    class RevIN(nn.Module):
        def __init__(self, num_features: int, eps: float = 1e-5):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
            self.mean = None
            self.stdev = None

        def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
            if mode == "norm":
                dim2reduce = tuple(range(1, x.ndim - 1))
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
                self.stdev = torch.sqrt(
                    torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
                    + self.eps
                ).detach()
                x = (x - self.mean) / self.stdev
                return x * self.affine_weight + self.affine_bias
            if mode == "denorm":
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
                return x * self.stdev + self.mean
            raise ValueError(f"Unknown RevIN mode: {mode}")

    class SAMOptimizer(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            base_optimizer,
            rho: float = 0.05,
            adaptive: bool = False,
            **kwargs,
        ):
            assert rho >= 0.0
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups

        @torch.no_grad()
        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norms = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    scale = torch.abs(p) if group["adaptive"] else 1.0
                    norms.append((scale * p.grad).norm(p=2).to(shared_device))
            if not norms:
                return torch.tensor(0.0, device=shared_device)
            return torch.norm(torch.stack(norms), p=2)

        @torch.no_grad()
        def first_step(self, zero_grad: bool = False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    e_w = (
                        (torch.pow(p, 2) if group["adaptive"] else 1.0)
                        * p.grad
                        * scale.to(p)
                    )
                    p.add_(e_w)
                    self.state[p]["e_w"] = e_w
            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad: bool = False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.sub_(self.state[p]["e_w"])
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()

    class SAMformerBackbone(nn.Module):
        def __init__(
            self,
            num_channels: int,
            seq_len: int,
            hid_dim: int,
            pred_horizon: int,
            use_revin: bool = True,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.revin = RevIN(num_features=num_channels)
            self.use_revin = use_revin
            self.compute_keys = nn.Linear(seq_len, hid_dim)
            self.compute_queries = nn.Linear(seq_len, hid_dim)
            self.compute_values = nn.Linear(seq_len, seq_len)
            self.linear_forecaster = nn.Linear(seq_len, pred_horizon)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, L]
            if self.use_revin:
                x_norm = self.revin(x.transpose(1, 2), mode="norm").transpose(1, 2)
            else:
                x_norm = x

            queries = self.compute_queries(x_norm)
            keys = self.compute_keys(x_norm)
            values = self.compute_values(x_norm)
            attn = nn.functional.scaled_dot_product_attention(queries, keys, values)
            out = x_norm + self.dropout(attn)
            out = self.linear_forecaster(out)  # [B, C, H]

            if self.use_revin:
                out = self.revin(out.transpose(1, 2), mode="denorm").transpose(1, 2)

            return out[:, 0, :]  # target channel forecast only

    return torch, nn, RevIN, SAMOptimizer, SAMformerBackbone


class SAMformerWrapper(ModelBase):
    """Benchmark-oriented SAMformer wrapper."""

    def __init__(
        self,
        input_size: int = 60,
        hidden_size: int = 16,
        max_epochs: int = 25,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        rho: float = 0.5,
        use_revin: bool = True,
        dropout: float = 0.0,
        max_covariates: int = 15,
        max_entities: int = 3000,
        max_windows: int = 60000,
        val_frac: float = 0.15,
        patience: int = 5,
        seed: int = 42,
        **kwargs,
    ):
        config = ModelConfig(
            name="SAMformer",
            model_type="forecasting",
            params={
                "input_size": input_size,
                "hidden_size": hidden_size,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "rho": rho,
                "use_revin": use_revin,
                "dropout": dropout,
                "max_covariates": max_covariates,
                "max_entities": max_entities,
                "max_windows": max_windows,
            },
            optional_dependency="torch",
        )
        super().__init__(config)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.dropout = dropout
        self.max_covariates = max_covariates
        self.max_entities = max_entities
        self.max_windows = max_windows
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed
        self._device = None
        self._network = None
        self._contexts: Dict[str, np.ndarray] = {}
        self._feature_cols: List[str] = []
        self._fallback_value = 0.0
        self._nonnegative_target = False
        self._binary_target = False
        self._horizon = 1

    def _make_optimizer(self, torch_mod, params):
        return self._sam_cls(
            params,
            base_optimizer=torch_mod.optim.Adam,
            rho=self.rho,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SAMformerWrapper":
        torch, nn, _RevIN, SAMOptimizer, SAMformerBackbone = _build_torch_modules()
        self._sam_cls = SAMOptimizer

        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = int(kwargs.get("horizon", 7))
        self._horizon = horizon
        y_arr = np.asarray(y.values, dtype=np.float32)
        finite = y_arr[np.isfinite(y_arr)]
        self._fallback_value = float(np.nanmedian(finite)) if len(finite) else 0.0
        self._binary_target = _detect_binary(y_arr)
        self._nonnegative_target = _detect_nonnegative(y_arr)

        if train_raw is None or "entity_id" not in train_raw.columns:
            logger.warning("  [SAMformer] No train_raw/entity_id; using fallback-only mode")
            self._fitted = True
            return self

        self._feature_cols = _select_feature_cols(train_raw, target, self.max_covariates)
        entity_windows = _build_entity_windows(
            train_raw=train_raw,
            target=target,
            feature_cols=self._feature_cols,
            input_size=self.input_size,
            horizon=horizon,
            step=max(1, horizon),
            max_entities=self.max_entities,
            max_windows=self.max_windows,
            val_frac=self.val_frac,
            seed=seed,
        )
        self._contexts = entity_windows.contexts

        if not entity_windows.train_x:
            logger.warning("  [SAMformer] No training windows; using fallback-only mode")
            self._fitted = True
            return self

        train_x = torch.tensor(np.stack(entity_windows.train_x), dtype=torch.float32)
        train_y = torch.tensor(np.stack(entity_windows.train_y), dtype=torch.float32)
        val_x = (
            torch.tensor(np.stack(entity_windows.val_x), dtype=torch.float32)
            if entity_windows.val_x else None
        )
        val_y = (
            torch.tensor(np.stack(entity_windows.val_y), dtype=torch.float32)
            if entity_windows.val_y else None
        )

        self._network = SAMformerBackbone(
            num_channels=train_x.shape[1],
            seq_len=train_x.shape[2],
            hid_dim=self.hidden_size,
            pred_horizon=horizon,
            use_revin=self.use_revin,
            dropout=self.dropout,
        ).to(self._device)

        criterion = nn.MSELoss() if not self._binary_target else nn.BCEWithLogitsLoss()
        optimizer = self._make_optimizer(torch, self._network.parameters())

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        logger.info(
            f"  [SAMformer] windows train={len(train_x):,} "
            f"val={0 if val_x is None else len(val_x):,} "
            f"channels={train_x.shape[1]} horizon={horizon} "
            f"device={self._device}"
        )

        for epoch in range(self.max_epochs):
            self._network.train()
            losses: List[float] = []
            for xb, yb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                out = self._network(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                out2 = self._network(xb)
                loss2 = criterion(out2, yb)
                loss2.backward()
                optimizer.second_step(zero_grad=True)
                losses.append(float(loss2.detach().cpu()))

            if val_x is None or val_y is None:
                continue

            self._network.eval()
            with torch.no_grad():
                pred_val = self._network(val_x.to(self._device))
                val_loss = float(criterion(pred_val, val_y.to(self._device)).cpu())

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = copy.deepcopy(self._network.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            logger.info(
                f"  [SAMformer] epoch={epoch + 1}/{self.max_epochs} "
                f"train={np.mean(losses):.6f} val={val_loss:.6f}"
            )

            if bad_epochs >= self.patience:
                break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("SAMformerWrapper is not fitted")

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

            x_batch = np.stack([self._contexts[eid] for eid in unique_entities]).astype(np.float32)
            with torch.no_grad():
                preds = self._network(
                    torch.tensor(x_batch, dtype=torch.float32, device=self._device)
                ).detach().cpu().numpy()
            idx = min(req_horizon - 1, preds.shape[1] - 1)
            point_preds = preds[:, idx]
            if self._binary_target:
                point_preds = 1.0 / (1.0 + np.exp(-point_preds))
            if self._nonnegative_target:
                point_preds = np.clip(point_preds, 0.0, None)
            entity_forecasts = {
                eid: float(pred) for eid, pred in zip(unique_entities, point_preds)
            }

        out = np.empty(h, dtype=np.float64)
        for i, eid in enumerate(test_entities):
            out[i] = entity_forecasts.get(str(eid), self._fallback_value)

        return _sanitize_predictions(out, self._fallback_value, "SAMformer")


def create_samformer(**kwargs) -> ModelBase:
    cfg = ModelConfig(
        name="SAMformer",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="torch",
    )
    model = SAMformerWrapper(**kwargs)
    model.config = cfg
    return model


__all__ = ["SAMformerWrapper", "create_samformer"]
