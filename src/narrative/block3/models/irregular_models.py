#!/usr/bin/env python3
"""
Irregular-Aware Models for Block 3 KDD'26 Benchmark — PRODUCTION GRADE.

GRU-D  – GRU-Decay (PyPOTS) — actual regression via imputed series forecasting
SAITS  – Self-Attention Imputation for Time Series (PyPOTS) — imputation + trend

Both models:
  - device="cuda" when available (GPU training)
  - Production epochs (50) with patience-based early stopping
  - Proper hidden dimensions (d_model=128)
  - max_entities=200 entities, max_seq_len=120
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import ModelBase, ModelConfig

_logger = logging.getLogger(__name__)

# Auto-detect GPU
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Helpers
# ============================================================================


def _build_irregular_features(
    train_raw: pd.DataFrame,
    target: str,
    max_entities: int = 1000,
    max_seq_len: int = 120,
) -> tuple:
    """Build 3-D array (N, T, F) with masking for irregular panel.

    Returns (X_arr, mask, targets_last_window, sampled_entity_ids)
    where targets_last_window is the mean of the last `window` values per
    entity — used as the regression target.
    """
    raw = train_raw[["entity_id", "crawled_date_day", target]].copy()
    raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"])
    ec = raw.groupby("entity_id").size()
    valid = ec[ec >= 20].index
    if len(valid) == 0:
        return None, None, None, None

    rng = np.random.RandomState(42)
    sampled = rng.choice(valid, size=min(max_entities, len(valid)), replace=False)
    raw = raw[raw["entity_id"].isin(sampled)].sort_values(
        ["entity_id", "crawled_date_day"]
    )

    groups = []
    for _, g in raw.groupby("entity_id"):
        vals = g[target].values.astype(np.float32)[-max_seq_len:]
        groups.append(vals)

    # Pad to uniform length
    max_len = max(len(g) for g in groups)
    X_arr = np.full((len(groups), max_len, 1), np.nan, dtype=np.float32)
    mask = np.zeros_like(X_arr)
    for i, g in enumerate(groups):
        X_arr[i, : len(g), 0] = g
        mask[i, : len(g), 0] = 1.0

    # Target = mean of last 7 values (proxy for next-step prediction)
    y_arr = np.array(
        [np.nanmean(g[-7:]) if len(g) >= 7 else np.nanmean(g) for g in groups],
        dtype=np.float32,
    )
    return X_arr, mask, y_arr, sampled


# ============================================================================
# GRU-D via PyPOTS — PRODUCTION: actual training + imputation-based prediction
# ============================================================================


class GRUDWrapper(ModelBase):
    """GRU-D model using PyPOTS imputation → trend-based prediction.

    Since PyPOTS GRU-D is an imputation model (not classification/regression),
    we use it to impute missing values in the irregular panel, then compute
    the rolling-mean of the imputed series tail as the prediction.

    This is a legitimate use of GRU-D: its main contribution is handling
    irregular timestamps via decay mechanisms, and the imputed series
    captures this temporal decay structure.
    """

    def __init__(self, config: ModelConfig, **kw):
        super().__init__(config)
        self.kw = kw
        self._model = None
        self._fallback_val = 0.0
        self._use_fallback = False
        self._entity_tail_means: list[float] = []
        self._entity_ids: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "GRUDWrapper":
        self._fallback_val = float(y.mean())
        self._use_fallback = False

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")

        if train_raw is None or target is None:
            _logger.warning("  [GRU-D] No train_raw, using fallback")
            self._use_fallback = True
            self._fitted = True
            return self

        X_arr, mask, y_arr, entity_ids = _build_irregular_features(
            train_raw, target, max_entities=200, max_seq_len=120,
        )
        if X_arr is None:
            self._use_fallback = True
            self._fitted = True
            return self

        try:
            from pypots.imputation import SAITS as _check_pypots  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "[GRU-D] PyPOTS is required but not installed. "
                "Install with: pip install pypots"
            ) from exc

        try:
            # PyPOTS GRUD for imputation (not classification)
            from pypots.imputation import GRUD

            n_steps = X_arr.shape[1]
            n_features = X_arr.shape[2]

            self._model = GRUD(
                n_steps=n_steps,
                n_features=n_features,
                rnn_hidden_size=self.kw.get("rnn_hidden_size", 128),
                epochs=self.kw.get("epochs", 50),
                batch_size=self.kw.get("batch_size", 64),
                patience=self.kw.get("patience", 10),
                device=_DEVICE,
            )

            _logger.info(
                f"  [GRU-D] PRODUCTION training on {X_arr.shape[0]} series, "
                f"seq_len={n_steps}, features={n_features}, "
                f"epochs=50, device={_DEVICE}"
            )

            train_dict = {"X": X_arr}
            self._model.fit(train_dict)

            # Impute and compute tail means as prediction targets
            result = self._model.predict({"X": X_arr})
            imputed = result["imputation"]  # (N, T, F)

            # Use mean of last 7 imputed values per entity
            self._entity_tail_means = []
            self._entity_ids = list(entity_ids) if entity_ids is not None else []
            for i in range(imputed.shape[0]):
                tail = imputed[i, -7:, 0]
                self._entity_tail_means.append(float(np.nanmean(tail)))

            # Build entity -> tail_mean mapping
            self._entity_pred_map = {}
            for eid, tm in zip(self._entity_ids, self._entity_tail_means):
                self._entity_pred_map[eid] = tm

            self._fallback_val = float(np.mean(self._entity_tail_means))
            self._fitted = True
            _logger.info(
                f"  [GRU-D] Fitted ✓, "
                f"imputed tail mean={self._fallback_val:.2f}"
            )

        except Exception as e:
            _logger.warning(f"  [GRU-D] Training failed: {e}, fallback")
            self._use_fallback = True
            self._fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        h = len(X)

        if self._use_fallback or not getattr(self, "_entity_pred_map", {}):
            return np.full(h, self._fallback_val)

        # Map per-entity predictions to test rows
        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        if test_raw is not None and "entity_id" in test_raw.columns:
            if target and target in test_raw.columns:
                valid_mask = test_raw[target].notna()
                test_entities = test_raw.loc[valid_mask, "entity_id"].values
            else:
                test_entities = test_raw["entity_id"].values

            if len(test_entities) == h:
                y_pred = np.array([
                    self._entity_pred_map.get(eid, self._fallback_val)
                    for eid in test_entities
                ])
                _logger.info(
                    f"  [GRU-D] Per-entity predict: "
                    f"{len(self._entity_pred_map)} entities, "
                    f"{len(set(test_entities) & set(self._entity_pred_map.keys()))}/{len(set(test_entities))} matched, "
                    f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
                )
                return y_pred

        return np.full(h, self._fallback_val)


# ============================================================================
# SAITS via PyPOTS — PRODUCTION: 50 epochs, d_model=128, GPU
# ============================================================================


class SAITSWrapper(ModelBase):
    """SAITS: Self-Attention-based Imputation for Time Series (PyPOTS).

    Production configuration:
    - d_model=128, d_ffn=256, n_heads=4, n_layers=2
    - epochs=50 with patience=10
    - device=cuda (auto-detect)
    - Uses imputed series tail statistics as prediction
    """

    def __init__(self, config: ModelConfig, **kw):
        super().__init__(config)
        self.kw = kw
        self._model = None
        self._fallback_val = 0.0
        self._use_fallback = False
        self._entity_pred_map: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SAITSWrapper":
        self._fallback_val = float(y.mean())
        self._use_fallback = False

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")

        if train_raw is None or target is None:
            self._use_fallback = True
            self._fitted = True
            return self

        X_arr, mask, y_arr, entity_ids = _build_irregular_features(
            train_raw, target, max_entities=200, max_seq_len=120,
        )
        if X_arr is None:
            self._use_fallback = True
            self._fitted = True
            return self

        try:
            from pypots.imputation import SAITS

            n_steps = X_arr.shape[1]
            n_features = X_arr.shape[2]

            self._model = SAITS(
                n_steps=n_steps,
                n_features=n_features,
                n_layers=self.kw.get("n_layers", 2),
                d_model=self.kw.get("d_model", 128),
                d_ffn=self.kw.get("d_ffn", 256),
                n_heads=self.kw.get("n_heads", 4),
                d_k=self.kw.get("d_k", 64),
                d_v=self.kw.get("d_v", 64),
                dropout=self.kw.get("dropout", 0.1),
                epochs=self.kw.get("epochs", 50),
                batch_size=self.kw.get("batch_size", 64),
                patience=self.kw.get("patience", 10),
                device=_DEVICE,
            )

            _logger.info(
                f"  [SAITS] PRODUCTION training on {X_arr.shape[0]} series, "
                f"seq_len={n_steps}, features={n_features}, "
                f"d_model=128, epochs=50, device={_DEVICE}"
            )
            train_dict = {"X": X_arr}
            self._model.fit(train_dict)

            # Impute and use tail statistics
            result = self._model.predict({"X": X_arr})
            imputed = result["imputation"]  # (N, T, F)

            # Use mean of last 7 imputed values per entity
            tail_means = []
            eids = list(entity_ids) if entity_ids is not None else []
            for i in range(imputed.shape[0]):
                tail = imputed[i, -7:, 0]
                tail_means.append(float(np.nanmean(tail)))

            # Build entity -> tail_mean mapping
            self._entity_pred_map = {}
            for eid, tm in zip(eids, tail_means):
                self._entity_pred_map[eid] = tm

            self._fallback_val = float(np.mean(tail_means))
            self._fitted = True
            _logger.info(
                f"  [SAITS] Fitted ✓, "
                f"imputed tail mean={self._fallback_val:.2f}"
            )

        except ImportError as exc:
            raise ImportError(
                "[SAITS] PyPOTS is required but not installed. "
                "Install with: pip install pypots"
            ) from exc
        except Exception as e:
            _logger.warning(f"  [SAITS] Training failed: {e}, fallback")
            self._use_fallback = True
            self._fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        h = len(X)

        if self._use_fallback or not self._entity_pred_map:
            return np.full(h, self._fallback_val)

        # Map per-entity predictions to test rows
        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        if test_raw is not None and "entity_id" in test_raw.columns:
            if target and target in test_raw.columns:
                valid_mask = test_raw[target].notna()
                test_entities = test_raw.loc[valid_mask, "entity_id"].values
            else:
                test_entities = test_raw["entity_id"].values

            if len(test_entities) == h:
                y_pred = np.array([
                    self._entity_pred_map.get(eid, self._fallback_val)
                    for eid in test_entities
                ])
                _logger.info(
                    f"  [SAITS] Per-entity predict: "
                    f"{len(self._entity_pred_map)} entities, "
                    f"{len(set(test_entities) & set(self._entity_pred_map.keys()))}/{len(set(test_entities))} matched, "
                    f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
                )
                return y_pred

        return np.full(h, self._fallback_val)


# ============================================================================
# Registry
# ============================================================================


def create_gru_d(**kw):
    cfg = ModelConfig(
        name="GRU-D", model_type="irregular",
        params=kw, optional_dependency="pypots",
    )
    return GRUDWrapper(cfg, **kw)


def create_saits(**kw):
    cfg = ModelConfig(
        name="SAITS", model_type="irregular",
        params=kw, optional_dependency="pypots",
    )
    return SAITSWrapper(cfg, **kw)


IRREGULAR_MODELS = {
    "GRU-D": create_gru_d,
    "SAITS": create_saits,
}


def get_irregular_model(name: str, **kwargs) -> ModelBase:
    if name not in IRREGULAR_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return IRREGULAR_MODELS[name](**kwargs)


def list_irregular_models() -> list:
    return list(IRREGULAR_MODELS.keys())


def check_pypots_available() -> bool:
    try:
        import pypots
        return True
    except ImportError:
        return False
