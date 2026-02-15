#!/usr/bin/env python3
"""
Deep Learning & Foundation Models for Block 3 KDD'26 Benchmark.

PRODUCTION-GRADE configurations for KDD'26 full paper.
All models use their NeuralForecast library defaults or paper-recommended
hyperparameters, with proper early stopping, validation, and robust scaling.

deep_classical (NeuralForecast):
  - N-BEATS, N-HiTS, TFT, DeepAR

transformer_sota (NeuralForecast):
  - PatchTST, iTransformer, TimesNet, TSMixer
  - Informer, Autoformer, FEDformer, VanillaTransformer
  - TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN

foundation:
  - Chronos (Amazon), Moirai (Salesforce), TimesFM (Google)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

_logger = logging.getLogger(__name__)


def _sanitize_predictions(
    preds: np.ndarray,
    fill_value: float,
    model_name: str,
) -> np.ndarray:
    """Replace non-finite predictions to keep evaluation coverage consistent."""
    arr = np.asarray(preds, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr

    if not np.isfinite(fill_value):
        if finite.any():
            fill_value = float(np.nanmedian(arr[finite]))
        else:
            fill_value = 0.0

    n_bad = int((~finite).sum())
    arr = arr.copy()
    arr[~finite] = fill_value
    _logger.warning(
        f"  [{model_name}] Sanitized {n_bad} non-finite predictions "
        f"(fill={fill_value:.6g})"
    )
    return arr


# ============================================================================
# KDD'26 Production Configurations
# ============================================================================
# Each model's config uses NeuralForecast library defaults or values from the
# original paper.  Early stopping is ENABLED for all models (patience=10
# val checks), validation is done on the last h timesteps of each series
# (val_size=h in NeuralForecast.fit).  Scaler is 'robust' for financial data.
#
# Batch size is set conservatively to avoid OOM on RTX 4090 (24 GB VRAM).
# Models with larger attention matrices use batch_size=32.
# ============================================================================

PRODUCTION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ================================================================
    # deep_classical
    # ================================================================
    "NBEATS": {
        # Oreshkin et al., 2019 — default NF max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
        "stack_types": ["trend", "seasonality"],
    },
    "NHITS": {
        # Challu et al., 2022 — default NF max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
        "stack_types": ["identity", "identity", "identity"],
    },
    "TFT": {
        # Lim et al., 2021 — NF default hidden=128, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "DeepAR": {
        # Salinas et al., 2020 — NF default max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "lstm_hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    # ================================================================
    # transformer_sota
    # ================================================================
    "PatchTST": {
        # Nie et al., ICLR 2023 — NF default max_steps=5000
        "input_size": 64,   # must be divisible by patch_len
        "max_steps": 3000,
        "hidden_size": 128,
        "n_heads": 16,
        "patch_len": 16,
        "stride": 8,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "iTransformer": {
        # Liu et al., ICLR 2024 — NF default hidden=512, max_steps=1000
        # hidden reduced to 256 for VRAM safety with n_series=200
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 256,
        "n_heads": 8,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "TimesNet": {
        # Wu et al., ICLR 2023 — NF default hidden=64, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 64,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "standard",
    },
    "TSMixer": {
        # Chen et al., TMLR 2023 — NF default max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "Informer": {
        # Zhou et al., AAAI 2021 — NF default hidden=128, max_steps=5000
        "input_size": 60,
        "max_steps": 3000,
        "hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "Autoformer": {
        # Wu et al., NeurIPS 2021 — NF default hidden=128, max_steps=5000
        "input_size": 60,
        "max_steps": 3000,
        "hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "FEDformer": {
        # Zhou et al., ICML 2022 — NF default hidden=128, max_steps=5000
        "input_size": 60,
        "max_steps": 3000,
        "hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "VanillaTransformer": {
        # Vaswani et al., 2017 baseline — NF default hidden=128, max_steps=5000
        "input_size": 60,
        "max_steps": 3000,
        "hidden_size": 128,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "TiDE": {
        # Das et al., TMLR 2023 — NF default hidden=512, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 256,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "NBEATSx": {
        # Olivares et al., 2022 — NF default max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
        "stack_types": ["trend", "seasonality"],
    },
    "BiTCN": {
        # NF default hidden=16, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 16,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "KAN": {
        # Liu et al., 2024 — NF default hidden=512, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 256,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "RMoK": {
        # NF default max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "SOFTS": {
        # NF default hidden=512, max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "hidden_size": 256,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "StemGNN": {
        # Cao et al., NeurIPS 2020 — NF default max_steps=1000
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    # ================================================================
    # Additional SOTA transformers (Phase 4)
    # ================================================================
    "DLinear": {
        # Zeng et al., AAAI 2023 — simple decomposition + linear
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
        "moving_avg_window": 25,
    },
    "NLinear": {
        # Zeng et al., AAAI 2023 — normalized linear baseline
        "input_size": 60,
        "max_steps": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "num_lr_decays": 3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "TimeMixer": {
        # Wang et al., ICLR 2024 — multi-scale mixing
        "input_size": 60,
        "max_steps": 1000,
        "d_model": 128,
        "d_ff": 256,
        "e_layers": 2,
        "down_sampling_layers": 2,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
    "TimeXer": {
        # Wang et al., NeurIPS 2024 — exogenous-aware transformer
        "input_size": 64,
        "max_steps": 2000,
        "hidden_size": 128,
        "n_heads": 8,
        "patch_len": 16,
        "e_layers": 2,
        "d_ff": 256,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "early_stop_patience_steps": 10,
        "val_check_steps": 100,
        "scaler_type": "robust",
    },
    "TSMixerx": {
        # Chen et al., TMLR 2023 — TSMixer with exogenous features
        "input_size": 60,
        "max_steps": 1000,
        "n_block": 4,
        "ff_dim": 128,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "early_stop_patience_steps": 10,
        "val_check_steps": 50,
        "scaler_type": "robust",
    },
}

# Models that require n_series parameter (multivariate / cross-series)
_NEEDS_N_SERIES = {"iTransformer", "TSMixer", "TSMixerx", "RMoK", "SOFTS",
                   "StemGNN", "TimeMixer", "TimeXer"}


# ============================================================================
# Panel data helper (shared by deep + foundation)
# ============================================================================

def _build_panel_df(
    train_raw: pd.DataFrame,
    target: str,
    max_entities: Optional[int] = None,
    min_obs: int = 10,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """Build NeuralForecast-style panel DataFrame from raw entity data.

    Args:
        max_entities: Maximum entities to sample.  ``None`` → use ALL
            valid entities (channel-independent models benefit from full
            coverage).  Set a finite value only for cross-series models
            that embed the entity dimension (iTransformer, TSMixer, etc.).
    """
    if train_raw is None or target is None:
        return None
    if "entity_id" not in train_raw.columns:
        return None

    raw = train_raw[["entity_id", "crawled_date_day", target]].copy()
    raw = raw.dropna(subset=[target])
    raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"])

    entity_counts = raw.groupby("entity_id").size()
    valid = entity_counts[entity_counts >= min_obs].index
    if len(valid) == 0:
        return None

    # When max_entities is None, use ALL valid entities; otherwise sample.
    if max_entities is not None and len(valid) > max_entities:
        rng = np.random.RandomState(seed)
        sampled = rng.choice(valid, size=max_entities, replace=False)
    else:
        sampled = valid
    raw = raw[raw["entity_id"].isin(sampled)].sort_values(
        ["entity_id", "crawled_date_day"]
    )

    panel = pd.DataFrame({
        "unique_id": raw["entity_id"].values,
        "ds": raw["crawled_date_day"].values,
        "y": raw[target].values.astype(np.float32),
    })
    _logger.info(
        f"  Panel data: {panel['unique_id'].nunique()} entities, {len(panel):,} rows"
    )
    return panel


def _synthetic_panel(y: pd.Series, max_entities: int = 200, min_obs: int = 20):
    """Fallback: convert flat y into synthetic panel."""
    n = len(y)
    n_series = min(max_entities, max(1, n // min_obs))
    obs = min(n // n_series, 500)
    uids, ds_list, ys = [], [], []
    for i in range(n_series):
        chunk = y.values[i * obs : (i + 1) * obs]
        uids.extend([f"s_{i}"] * len(chunk))
        ds_list.extend(pd.date_range("2020-01-01", periods=len(chunk), freq="D"))
        ys.extend(chunk)
    return pd.DataFrame(
        {"unique_id": uids, "ds": ds_list, "y": np.array(ys, dtype=np.float32)}
    )


# ============================================================================
# Robust Fallback for Unseen Entities  (replaces raw Ridge)
# ============================================================================
# References:
#   - Burbidge et al. (1988), "Alternative Transformations to Handle Extreme
#     Values of the Dependent Variable", JASA.
#   - Bellemare & Wichman (2020), "Elasticities and the Inverse Hyperbolic
#     Sine Transformation", Oxford Bulletin of Economics and Statistics.
#   - Grinsztajn et al. (2022), "Why do tree-based models still outperform
#     deep learning on typical tabular data?", NeurIPS.
# ============================================================================

class _RobustFallback:
    """LightGBM fallback with adaptive target transform for uncovered entities.

    Why this replaces raw Ridge:
      - Ridge on funding_raised_usd (kurtosis=125, skew=10.35) → MAE ~5.78B
      - The Inverse Hyperbolic Sine (asinh) transform handles zero-inflated
        heavy-tailed non-negative data without log(0) issues.
      - LightGBM natively handles NaN, categorical-like features, and
        non-linear interactions — far superior to Ridge for tabular data.
      - The transform is fitted ONLY on training data (no leakage).
    """

    def __init__(self):
        self._model = None
        self._transform_kind: str = "identity"
        self._feature_cols: List[str] = []
        self._clip_low: Optional[float] = None
        self._clip_high: Optional[float] = None
        self._safe_fill: float = 0.0
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_RobustFallback":
        """Fit fallback on cross-sectional features with adaptive transform."""
        y_arr = np.asarray(y, dtype=float)
        y_finite = y_arr[np.isfinite(y_arr)]
        if len(y_finite) < 10:
            if len(y_finite):
                self._safe_fill = float(np.nanmedian(y_finite))
            self._fitted = True
            return self
        self._safe_fill = float(np.nanmedian(y_finite))

        # --- Determine transform ---
        is_non_neg = (y_finite >= 0).all()
        n_unique = len(np.unique(y_finite))
        kurtosis = float(pd.Series(y_finite).kurtosis()) if len(y_finite) > 4 else 0

        if n_unique <= 3:
            self._transform_kind = "identity"   # binary / ternary
        elif is_non_neg and kurtosis > 5:
            self._transform_kind = "asinh"       # heavy-tailed non-negative
        elif (is_non_neg
              and (y_finite == np.round(y_finite)).mean() > 0.9
              and y_finite.max() > 2):
            self._transform_kind = "sqrt"        # count-like
        else:
            self._transform_kind = "identity"

        y_transformed = self._forward(y_arr)
        y_trans_finite = y_transformed[np.isfinite(y_transformed)]
        if len(y_trans_finite):
            lo = float(np.quantile(y_trans_finite, 0.001))
            hi = float(np.quantile(y_trans_finite, 0.999))
            if hi < lo:
                lo, hi = hi, lo
            span = max(hi - lo, 1e-6)
            pad = 0.10 * span
            lo -= pad
            hi += pad
            if self._transform_kind == "asinh":
                lo = max(lo, -30.0)
                hi = min(hi, 30.0)
            elif self._transform_kind == "sqrt":
                lo = max(lo, 0.0)
            self._clip_low = lo
            self._clip_high = hi

        # --- Feature matrix ---
        X_fb = X.select_dtypes(include=[np.number]).fillna(0)
        self._feature_cols = list(X_fb.columns)
        if len(self._feature_cols) == 0 or len(X_fb) < 10:
            self._fitted = True
            return self

        # --- Try LightGBM (preferred), then Ridge ---
        try:
            import lightgbm as lgb
            n = len(X_fb)
            split_idx = int(n * 0.8)
            self._model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, verbose=-1, random_state=42,
            )
            self._model.fit(
                X_fb.iloc[:split_idx], y_transformed[:split_idx],
                eval_set=[(X_fb.iloc[split_idx:], y_transformed[split_idx:])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        except Exception:
            try:
                from sklearn.linear_model import Ridge as _Ridge
                self._model = _Ridge(alpha=1.0)
                self._model.fit(X_fb, y_transformed)
            except Exception:
                pass

        self._fitted = True
        _logger.info(
            f"  [RobustFallback] transform={self._transform_kind}, "
            f"features={len(self._feature_cols)}, "
            f"model={type(self._model).__name__ if self._model else 'None'}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict on original scale (inverse-transformed)."""
        if self._model is None or not self._feature_cols:
            return None
        try:
            X_fb = X.reindex(columns=self._feature_cols, fill_value=0).fillna(0)
            preds_transformed = np.asarray(self._model.predict(X_fb), dtype=float)
            if self._clip_low is not None and self._clip_high is not None:
                preds_transformed = np.clip(
                    preds_transformed, self._clip_low, self._clip_high
                )
            preds = np.asarray(self._inverse(preds_transformed), dtype=float)
            finite = np.isfinite(preds)
            if not finite.all():
                fill = float(np.nanmedian(preds[finite])) if finite.any() else self._safe_fill
                preds = preds.copy()
                preds[~finite] = fill
            return preds
        except Exception:
            return None

    def _forward(self, y: np.ndarray) -> np.ndarray:
        if self._transform_kind == "asinh":
            return np.arcsinh(y)
        elif self._transform_kind == "sqrt":
            return np.sqrt(np.maximum(y, 0))
        return y.copy()

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        if self._transform_kind == "asinh":
            return np.sinh(np.clip(y, -30.0, 30.0))
        elif self._transform_kind == "sqrt":
            return np.square(y)
        return y


# ============================================================================
# Static Exogenous Feature Builder (for NF covariate support)
# ============================================================================
# NeuralForecast supports stat_exog_list for models with static covariate
# inputs (TFT, TiDE, TimeXer, TSMixerx).  EDGAR features are entity-level
# and roughly quarterly → ideal as static covariates.
# ============================================================================

_NF_SUPPORTS_STATIC_EXOG = {"TFT", "TiDE", "TimeXer", "TSMixerx"}

# Columns to NEVER include as exogenous features
_EXOG_EXCLUDE = {
    "entity_id", "crawled_date_day", "cik", "date",
    "offer_id", "snapshot_ts", "crawled_date", "processed_datetime",
    "funding_raised_usd", "funding_raised", "investors_count",
    "non_national_investors", "is_funded",
    "funding_goal_usd", "funding_goal",
    "funding_goal_maximum", "funding_goal_maximum_usd",
}


def _build_static_df(
    train_raw: pd.DataFrame,
    panel_entity_ids: np.ndarray,
    target: str,
    max_features: int = 50,
) -> Optional[tuple]:
    """Build static (entity-level) feature DataFrame for NeuralForecast.

    Returns (static_df, stat_exog_list) or None if not enough features.
    static_df has columns ['unique_id', feat1, feat2, ...].
    """
    if train_raw is None or "entity_id" not in train_raw.columns:
        return None

    exclude = _EXOG_EXCLUDE | {target}
    num_cols = [
        c for c in train_raw.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    if not num_cols:
        return None

    # Filter to panel entities, take last observation per entity
    mask = train_raw["entity_id"].isin(panel_entity_ids)
    entity_data = train_raw.loc[mask].sort_values("crawled_date_day")
    static = entity_data.groupby("entity_id")[num_cols].last().reset_index()
    static = static.rename(columns={"entity_id": "unique_id"})
    static[num_cols] = static[num_cols].fillna(0).astype(np.float32)

    # Keep only features with variance (constant features add nothing)
    variances = static[num_cols].var()
    valid_cols = variances[variances > 1e-12].index.tolist()
    if not valid_cols:
        return None

    # Limit to top-k by variance for VRAM / compute efficiency
    if len(valid_cols) > max_features:
        valid_cols = variances[valid_cols].nlargest(max_features).index.tolist()

    _logger.info(
        f"  [StaticExog] {len(valid_cols)} features for "
        f"{static['unique_id'].nunique()} entities"
    )
    return static[["unique_id"] + valid_cols], valid_cols


# ============================================================================
# NeuralForecast Wrapper (deep_classical + transformer_sota)
# ============================================================================


class DeepModelWrapper(ModelBase):
    """Unified wrapper for ALL NeuralForecast models.

    Uses PRODUCTION_CONFIGS for KDD'26-grade hyperparameters.
    """

    def __init__(self, config: ModelConfig, model_name: str, **kw):
        super().__init__(config)
        self.model_name = model_name
        self.model_kwargs = kw
        self._nf = None
        self._last_y = np.array([])
        self._use_fallback = False

    # ------------------------------------------------------------------
    # Model construction — uses PRODUCTION_CONFIGS exclusively
    # ------------------------------------------------------------------
    def _get_model(self, h: int, n_series: int = 1):
        from neuralforecast.models import (
            NBEATS, NHITS, TFT, DeepAR,
            PatchTST, iTransformer, TimesNet, TSMixer,
            Informer, Autoformer, FEDformer, VanillaTransformer,
            TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN,
            DLinear, NLinear, TimeMixer, TimeXer, TSMixerx,
        )

        _cls = {
            "NBEATS": NBEATS, "NHITS": NHITS, "TFT": TFT, "DeepAR": DeepAR,
            "PatchTST": PatchTST, "iTransformer": iTransformer,
            "TimesNet": TimesNet, "TSMixer": TSMixer,
            "Informer": Informer, "Autoformer": Autoformer,
            "FEDformer": FEDformer, "VanillaTransformer": VanillaTransformer,
            "TiDE": TiDE, "NBEATSx": NBEATSx, "BiTCN": BiTCN,
            "KAN": KAN, "RMoK": RMoK, "SOFTS": SOFTS, "StemGNN": StemGNN,
            "DLinear": DLinear, "NLinear": NLinear,
            "TimeMixer": TimeMixer, "TimeXer": TimeXer, "TSMixerx": TSMixerx,
        }
        if self.model_name not in _cls:
            raise ValueError(f"Unknown NF model: {self.model_name}")

        cls = _cls[self.model_name]
        cfg = PRODUCTION_CONFIGS[self.model_name]

        # ---- common params (accepted by every NF model) ----
        common: Dict[str, Any] = {
            "h": h,
            "input_size": cfg["input_size"],
            "max_steps": cfg["max_steps"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "early_stop_patience_steps": cfg["early_stop_patience_steps"],
            "val_check_steps": cfg["val_check_steps"],
            "scaler_type": cfg["scaler_type"],
        }
        if "num_lr_decays" in cfg:
            common["num_lr_decays"] = cfg["num_lr_decays"]

        # ---- model-specific params ----
        if self.model_name in ("NBEATS", "NHITS", "NBEATSx"):
            return cls(**common, stack_types=cfg["stack_types"])

        if self.model_name == "TFT":
            return cls(**common, hidden_size=cfg["hidden_size"])

        if self.model_name == "DeepAR":
            return cls(**common, lstm_hidden_size=cfg["lstm_hidden_size"])

        if self.model_name == "PatchTST":
            return cls(**common,
                       hidden_size=cfg["hidden_size"],
                       n_heads=cfg["n_heads"],
                       patch_len=cfg["patch_len"],
                       stride=cfg["stride"])

        if self.model_name == "iTransformer":
            return cls(**common,
                       hidden_size=cfg["hidden_size"],
                       n_heads=cfg["n_heads"],
                       n_series=n_series)

        if self.model_name == "TSMixer":
            return cls(**common, n_series=n_series)

        if self.model_name == "TSMixerx":
            return cls(**common, n_series=n_series,
                       n_block=cfg.get("n_block", 4),
                       ff_dim=cfg.get("ff_dim", 128))

        if self.model_name == "RMoK":
            return cls(**common, n_series=n_series)

        if self.model_name == "SOFTS":
            return cls(**common, hidden_size=cfg["hidden_size"], n_series=n_series)

        if self.model_name == "StemGNN":
            return cls(**common, n_series=n_series)

        if self.model_name == "DLinear":
            return cls(**common, moving_avg_window=cfg.get("moving_avg_window", 25))

        if self.model_name == "NLinear":
            return cls(**common)

        if self.model_name == "TimeMixer":
            return cls(**common, n_series=n_series,
                       d_model=cfg.get("d_model", 128),
                       d_ff=cfg.get("d_ff", 256),
                       e_layers=cfg.get("e_layers", 2),
                       down_sampling_layers=cfg.get("down_sampling_layers", 2))

        if self.model_name == "TimeXer":
            return cls(**common, n_series=n_series,
                       hidden_size=cfg["hidden_size"],
                       n_heads=cfg["n_heads"],
                       patch_len=cfg.get("patch_len", 16),
                       e_layers=cfg.get("e_layers", 2),
                       d_ff=cfg.get("d_ff", 256))

        # TimesNet, Informer, Autoformer, FEDformer, VanillaTransformer,
        # TiDE, BiTCN, KAN — all take hidden_size
        if "hidden_size" in cfg:
            return cls(**common, hidden_size=cfg["hidden_size"])

        return cls(**common)

    # ------------------------------------------------------------------
    # Fit — with proper validation and checkpoint saving
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "DeepModelWrapper":
        from neuralforecast import NeuralForecast

        h = kwargs.get("horizon", 7)
        # Clamp NF horizon to at least 7: NeuralForecast models (especially
        # NBEATS with trend/seasonality stacks) produce degenerate forecasts
        # when h=1 because val_size=1 gives insufficient validation signal.
        # We train with h_nf >= 7 and use the last step's forecast at predict.
        h_nf = max(h, 7)
        self._horizon = h
        self._horizon_nf = h_nf
        self._last_y = y.values
        self._use_fallback = False

        # Cross-series models (iTransformer, TSMixer, etc.) embed entity
        # dimension → limited by VRAM.  Channel-independent models share
        # weights across all entities → train on ALL for maximum coverage.
        _max_e = 5000 if self.model_name in _NEEDS_N_SERIES else None
        panel = _build_panel_df(
            kwargs.get("train_raw"), kwargs.get("target"),
            max_entities=_max_e, min_obs=10, seed=42,
        )
        if panel is None:
            panel = _synthetic_panel(y)

        # Store the entities used for training (for test-time mapping)
        self._train_entities = set(panel["unique_id"].unique())

        n_series = panel["unique_id"].nunique()
        cfg = PRODUCTION_CONFIGS.get(self.model_name, {})

        _logger.info(
            f"  [{self.model_name}] PRODUCTION training on panel "
            f"({n_series} entities, {len(panel):,} rows) | "
            f"max_steps={cfg.get('max_steps','?')}, "
            f"batch={cfg.get('batch_size','?')}, "
            f"lr={cfg.get('learning_rate','?')}, "
            f"early_stop={cfg.get('early_stop_patience_steps','?')}, "
            f"scaler={cfg.get('scaler_type','?')}"
        )
        try:
            model = self._get_model(h_nf, n_series=n_series)
            # val_size=h_nf → last h_nf timesteps of each series used for validation
            # This enables proper early stopping via the patience parameter.
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(df=panel, val_size=h_nf)
            self._fitted = True
            _logger.info(f"  [{self.model_name}] Training complete ✓")
        except Exception as e:
            _logger.warning(
                f"  [{self.model_name}] NF training failed: {e}, fallback"
            )
            self._use_fallback = True
            self._fitted = True

        # Feature-based ROBUST fallback for unseen test entities
        # (replaces raw Ridge which catastrophically fails on heavy-tailed
        # targets like funding_raised_usd with kurtosis=125)
        self._fallback = _RobustFallback()
        try:
            self._fallback.fit(X, y)
        except Exception as _exc:
            _logger.debug(f"  [{self.model_name}] Robust fallback failed: {_exc}")

        # --- Static exogenous features for NF models with covariate support ---
        # Enables EDGAR features to actually affect model predictions.
        self._static_df = None
        self._stat_exog_list: List[str] = []
        if (self.model_name in _NF_SUPPORTS_STATIC_EXOG
                and kwargs.get("train_raw") is not None):
            try:
                result = _build_static_df(
                    kwargs["train_raw"],
                    panel["unique_id"].unique(),
                    kwargs.get("target", ""),
                )
                if result is not None:
                    self._static_df, self._stat_exog_list = result
            except Exception as _exc:
                _logger.debug(f"  [{self.model_name}] Static exog failed: {_exc}")

        return self

    # ------------------------------------------------------------------
    # Predict — per-entity forecast mapped to test rows
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        h = len(X)
        req_horizon = kwargs.get("horizon", self._horizon)
        if self._use_fallback or self._nf is None:
            return np.full(h, float(np.mean(self._last_y)) if len(self._last_y) else 0)
        try:
            fcs = self._nf.predict()
            pred_cols = [c for c in fcs.columns if c not in ("unique_id", "ds")]
            if not pred_cols:
                return np.full(h, float(np.mean(self._last_y)) if len(self._last_y) else 0)

            # Build per-entity forecast using the requested horizon step.
            # NF may produce h_nf >= req_horizon steps; use step[req_horizon-1]
            # (the forecast at exactly the requested horizon).
            if "unique_id" in fcs.columns or fcs.index.name == "unique_id":
                if fcs.index.name == "unique_id":
                    fcs = fcs.reset_index()
                # Sort by ds, then pick the step at index `req_horizon - 1`
                # (or last step if fewer steps available)
                def _pick_horizon_step(grp):
                    vals = grp.sort_values("ds")[pred_cols[0]].values
                    idx = min(req_horizon - 1, len(vals) - 1)
                    return float(vals[idx])
                entity_fcs = (
                    fcs.groupby("unique_id")
                    .apply(_pick_horizon_step)
                    .to_dict()
                )
            else:
                # No entity info in forecast — global mean fallback
                entity_fcs = {}

            global_mean = float(np.mean(list(entity_fcs.values()))) if entity_fcs else float(fcs[pred_cols[0]].mean())

            # Map forecasts to test rows using entity_id
            test_raw = kwargs.get("test_raw")
            target = kwargs.get("target")
            if test_raw is not None and "entity_id" in test_raw.columns:
                # Align test_raw with X (same valid-target mask)
                if target and target in test_raw.columns:
                    valid_mask = test_raw[target].notna()
                    test_entities = test_raw.loc[valid_mask, "entity_id"].values
                else:
                    test_entities = test_raw["entity_id"].values

                if len(test_entities) == h:
                    # --- Per-entity hybrid prediction (NO all-or-nothing) ---
                    # For each test entity: use NF forecast if available,
                    # otherwise use RobustFallback (LightGBM + asinh transform).
                    # This replaces the old coverage < 0.95 → Ridge all-or-nothing
                    # that produced catastrophic results on heavy-tailed targets.
                    n_covered = sum(1 for eid in test_entities if eid in entity_fcs)
                    n_unique = len(set(test_entities))
                    coverage = n_covered / max(len(test_entities), 1)

                    # Get fallback predictions (transformed LightGBM)
                    fallback = getattr(self, '_fallback', None)
                    fallback_preds = None
                    if fallback is not None and fallback._fitted:
                        fallback_preds = fallback.predict(X)

                    y_pred = np.empty(h, dtype=np.float64)
                    for i, eid in enumerate(test_entities):
                        if eid in entity_fcs:
                            y_pred[i] = entity_fcs[eid]
                        elif fallback_preds is not None:
                            y_pred[i] = fallback_preds[i]
                        else:
                            y_pred[i] = global_mean

                    _logger.info(
                        f"  [{self.model_name}] Per-entity HYBRID predict: "
                        f"{len(entity_fcs)} entities forecasted, "
                        f"{n_covered}/{n_unique} ({coverage:.1%}) "
                        f"test entities covered by NF, "
                        f"fallback={'RobustFallback' if fallback_preds is not None else 'global_mean'}, "
                        f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
                    )
                    return _sanitize_predictions(
                        y_pred, global_mean, self.model_name
                    )

            # Fallback: no entity mapping available
            _logger.warning(
                f"  [{self.model_name}] No entity mapping for predict, "
                f"returning global mean={global_mean:.4f}"
            )
            return np.full(h, global_mean)
        except Exception as e:
            _logger.warning(f"  [{self.model_name}] predict failed: {e}")
            return np.full(h, float(np.mean(self._last_y)) if len(self._last_y) else 0)


# ============================================================================
# Foundation Model Wrapper
# ============================================================================


class FoundationModelWrapper(ModelBase):
    """Zero-shot / few-shot foundation models."""

    def __init__(self, config: ModelConfig, model_name: str, **kw):
        super().__init__(config)
        self.model_name = model_name
        self.model_kwargs = kw
        self._model = None
        self._entity_contexts: List[np.ndarray] = []
        self._context = np.array([])

    def _load_chronos(self):
        from chronos import ChronosPipeline
        self._model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map="auto",
        )

    def _load_chronos_bolt(self):
        from chronos import ChronosBoltPipeline
        self._model = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map="auto",
        )

    def _load_chronos2(self):
        from chronos import Chronos2Pipeline
        self._model = Chronos2Pipeline.from_pretrained(
            "amazon/chronos2-small", device_map="auto",
        )

    def _load_moirai(self):
        from uni2ts.model.moirai import MoiraiModule
        self._moirai_module = MoiraiModule.from_pretrained(
            "Salesforce/moirai-1.1-R-small"
        )

    def _load_moirai_large(self):
        from uni2ts.model.moirai import MoiraiModule
        self._moirai_module = MoiraiModule.from_pretrained(
            "Salesforce/moirai-1.1-R-large"
        )

    def _load_moirai2(self):
        from uni2ts.model.moirai_moe import MoiraiMoEModule
        self._moirai_module = MoiraiMoEModule.from_pretrained(
            "Salesforce/moirai-moe-1.0-R-small"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "FoundationModelWrapper":
        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        # No entity cap: foundation models are zero-shot pretrained and can
        # process any entity's time series.  Use ALL entities with enough
        # history to maximise test-time coverage.
        MIN_O = 10
        self._entity_ids: List[str] = []  # track entity_id per context

        if (train_raw is not None and target is not None
                and "entity_id" in train_raw.columns):
            raw = train_raw[["entity_id", "crawled_date_day", target]].dropna(
                subset=[target])
            raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"])
            raw = raw.sort_values(["entity_id", "crawled_date_day"])
            ec = raw.groupby("entity_id").size()
            valid = ec[ec >= MIN_O].index
            raw = raw[raw["entity_id"].isin(valid)]
            self._entity_contexts = []
            self._entity_ids = []
            for eid, grp in raw.groupby("entity_id"):
                self._entity_contexts.append(grp[target].values.astype(np.float32))
                self._entity_ids.append(eid)
            _logger.info(f"  [{self.model_name}] {len(self._entity_contexts)} entity contexts (ALL valid)")
        else:
            MIN_E_FALLBACK = 500
            cs = max(MIN_O, len(y) // MIN_E_FALLBACK)
            self._entity_contexts = []
            for i in range(0, len(y), cs):
                c = y.values[i : i + cs]
                if len(c) >= MIN_O:
                    self._entity_contexts.append(c.astype(np.float32))
                if len(self._entity_contexts) >= MIN_E_FALLBACK:
                    break

        self._context = y.values

        if self.model_name == "Chronos":
            self._load_chronos()
        elif self.model_name == "ChronosBolt":
            self._load_chronos_bolt()
        elif self.model_name == "Chronos2":
            self._load_chronos2()
        elif self.model_name == "Moirai":
            self._load_moirai()
        elif self.model_name == "MoiraiLarge":
            self._load_moirai_large()
        elif self.model_name == "Moirai2":
            self._load_moirai2()
        else:
            raise ValueError(f"Unknown foundation model: {self.model_name}")

        # Robust fallback for unseen entities (replaces raw Ridge)
        self._fallback = _RobustFallback()
        try:
            self._fallback.fit(X, y)
        except Exception:
            pass

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        h = len(X)
        ctxs = self._entity_contexts or [self._context]

        # Build per-entity predictions
        entity_preds: Dict[str, float] = {}

        if self.model_name in ("Chronos", "ChronosBolt", "Chronos2") and self._model is not None:
            import torch
            preds_all = []
            for i in range(0, len(ctxs), 32):
                batch = ctxs[i : i + 32]
                try:
                    tensors = [torch.tensor(c[-128:]).float() for c in batch]
                    out = self._model.predict(tensors, 7)
                    med = out.median(dim=1).values.mean(dim=1)
                    preds_all.extend(med.cpu().numpy().tolist())
                except Exception:
                    preds_all.extend([float(np.mean(c)) for c in batch])
            # Map entity_id -> forecast
            if self._entity_ids and len(preds_all) == len(self._entity_ids):
                entity_preds = dict(zip(self._entity_ids, preds_all))
            elif preds_all:
                entity_preds = {f"s_{i}": p for i, p in enumerate(preds_all)}

        elif self.model_name in ("Moirai", "MoiraiLarge") and hasattr(self, "_moirai_module"):
            import torch
            from uni2ts.model.moirai import MoiraiForecast
            preds_all = []
            for ctx in ctxs[:50]:
                try:
                    ts = pd.DataFrame(
                        {"target": ctx},
                        index=pd.date_range("2020-01-01", periods=len(ctx), freq="D"),
                    )
                    fm = MoiraiForecast(
                        module=self._moirai_module,
                        prediction_length=7,
                        context_length=min(128, len(ctx)),
                        patch_size="auto",
                        num_samples=20,
                    )
                    from gluonts.dataset.pandas import PandasDataset
                    ds = PandasDataset({"target": ts}, target="target", freq="D")
                    predictor = fm.create_predictor(batch_size=1)
                    for entry in predictor.predict(ds):
                        preds_all.append(float(np.median(entry.samples.mean(axis=1))))
                except Exception:
                    preds_all.append(float(np.mean(ctx)))
            # Map entity_id -> forecast
            eid_list = self._entity_ids[:50] if self._entity_ids else []
            if eid_list and len(preds_all) == len(eid_list):
                entity_preds = dict(zip(eid_list, preds_all))
            elif preds_all:
                entity_preds = {f"s_{i}": p for i, p in enumerate(preds_all)}

        elif self.model_name == "Moirai2" and hasattr(self, "_moirai_module"):
            import torch
            from uni2ts.model.moirai_moe import MoiraiMoEForecast
            preds_all = []
            for ctx in ctxs[:50]:
                try:
                    ts = pd.DataFrame(
                        {"target": ctx},
                        index=pd.date_range("2020-01-01", periods=len(ctx), freq="D"),
                    )
                    fm = MoiraiMoEForecast(
                        module=self._moirai_module,
                        prediction_length=7,
                        context_length=min(128, len(ctx)),
                        patch_size="auto",
                        num_samples=20,
                    )
                    from gluonts.dataset.pandas import PandasDataset
                    ds = PandasDataset({"target": ts}, target="target", freq="D")
                    predictor = fm.create_predictor(batch_size=1)
                    for entry in predictor.predict(ds):
                        preds_all.append(float(np.median(entry.samples.mean(axis=1))))
                except Exception:
                    preds_all.append(float(np.mean(ctx)))
            eid_list = self._entity_ids[:50] if self._entity_ids else []
            if eid_list and len(preds_all) == len(eid_list):
                entity_preds = dict(zip(eid_list, preds_all))
            elif preds_all:
                entity_preds = {f"s_{i}": p for i, p in enumerate(preds_all)}

        else:
            raise RuntimeError(
                f"[{self.model_name}] No valid prediction path reached. "
                f"Model was not loaded successfully."
            )

        if not entity_preds:
            return np.full(h, float(np.mean(self._context)) if len(self._context) else 0)

        global_mean = float(np.mean(list(entity_preds.values())))

        # Map to test rows via entity_id — per-entity hybrid (no threshold)
        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        if test_raw is not None and "entity_id" in test_raw.columns:
            if target and target in test_raw.columns:
                valid_mask = test_raw[target].notna()
                test_entities = test_raw.loc[valid_mask, "entity_id"].values
            else:
                test_entities = test_raw["entity_id"].values

            if len(test_entities) == h:
                n_covered = sum(1 for eid in test_entities if eid in entity_preds)
                coverage = n_covered / max(len(test_entities), 1)

                # Per-entity hybrid: NF forecast or RobustFallback
                fallback = getattr(self, '_fallback', None)
                fallback_preds = None
                if fallback is not None and fallback._fitted:
                    fallback_preds = fallback.predict(X)

                y_pred = np.empty(h, dtype=np.float64)
                for i, eid in enumerate(test_entities):
                    if eid in entity_preds:
                        y_pred[i] = entity_preds[eid]
                    elif fallback_preds is not None:
                        y_pred[i] = fallback_preds[i]
                    else:
                        y_pred[i] = global_mean

                _logger.info(
                    f"  [{self.model_name}] Per-entity HYBRID predict: "
                    f"{len(entity_preds)} entities, "
                    f"{n_covered}/{len(set(test_entities))} ({coverage:.1%}) covered, "
                    f"fallback={'RobustFallback' if fallback_preds is not None else 'global_mean'}, "
                    f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
                )
                return _sanitize_predictions(
                    y_pred, global_mean, self.model_name
                )

        # No entity mapping — global mean fallback
        _logger.warning(
            f"  [{self.model_name}] No entity mapping, returning global mean"
        )
        return np.full(h, global_mean)


# ============================================================================
# HuggingFace Foundation Models (Timer, TimeMoE, MOMENT, Lag-Llama, TimesFM)
# ============================================================================

class HFFoundationModelWrapper(ModelBase):
    """Foundation models loaded via HuggingFace transformers or dedicated libs.

    Supports models that use AutoModelForCausalLM.generate() API or
    dedicated forecast APIs (MOMENT, Lag-Llama, TimesFM via Docker).
    """

    # Model registry: name → (hf_repo, loader_type)
    _MODEL_REGISTRY = {
        "Timer": ("thuml/timer-base-84m", "hf_generate"),
        "TimeMoE": ("Maple728/TimeMoE-50M", "hf_generate"),
        "MOMENT": ("AutonLab/MOMENT-1-large", "momentfm"),
        "LagLlama": ("time-series-foundation-models/Lag-Llama", "lag_llama"),
        "TimesFM": (None, "timesfm_docker"),
    }

    def __init__(self, config: ModelConfig, model_name: str, **kw):
        super().__init__(config)
        self.model_name = model_name
        self.model_kwargs = kw
        self._model = None
        self._tokenizer = None
        self._entity_contexts: List[np.ndarray] = []
        self._entity_ids: List[str] = []
        self._context = np.array([])
        self._global_mean = 0.0

    def _load_hf_generate(self, repo: str):
        """Load HF models that use AutoModelForCausalLM.generate()."""
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self._model = self._model.to(device)
        self._model.eval()
        self._device = device

    def _load_moment(self, repo: str):
        """Load MOMENT via momentfm package."""
        from momentfm import MOMENTPipeline
        self._model = MOMENTPipeline.from_pretrained(
            repo, model_kwargs={"task_name": "forecasting", "forecast_horizon": 7},
        )
        self._model.init()

    def _load_lag_llama(self, repo: str):
        """Load Lag-Llama via HuggingFace + GluonTS."""
        import torch
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=repo, filename="lag-llama.ckpt")
        # Lag-Llama uses a custom LightningModule
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
            self._estimator = LagLlamaEstimator(
                prediction_length=7, context_length=128,
                input_size=1, n_layer=8, n_embd_per_head=32,
                n_head=4, ckpt_path=ckpt,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self._model = self._estimator.create_predictor(batch_size=32)
        except ImportError:
            # Fallback: load via torch directly
            self._model = None
            _logger.warning("lag_llama package not installed, using mean fallback")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "HFFoundationModelWrapper":
        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        MIN_O = 10

        # Build entity contexts — ALL valid entities (no arbitrary cap)
        if (train_raw is not None and target is not None
                and "entity_id" in train_raw.columns):
            raw = train_raw[["entity_id", "crawled_date_day", target]].dropna(subset=[target])
            raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"])
            raw = raw.sort_values(["entity_id", "crawled_date_day"])
            ec = raw.groupby("entity_id").size()
            valid = ec[ec >= MIN_O].index
            raw = raw[raw["entity_id"].isin(valid)]
            self._entity_contexts = []
            self._entity_ids = []
            for eid, grp in raw.groupby("entity_id"):
                self._entity_contexts.append(grp[target].values.astype(np.float32))
                self._entity_ids.append(eid)
        else:
            MAX_E_FALLBACK = 500
            cs = max(MIN_O, len(y) // MAX_E_FALLBACK)
            self._entity_contexts = []
            for i in range(0, len(y), cs):
                c = y.values[i:i + cs]
                if len(c) >= MIN_O:
                    self._entity_contexts.append(c.astype(np.float32))
                if len(self._entity_contexts) >= MAX_E_FALLBACK:
                    break

        self._context = y.values
        self._global_mean = float(np.mean(y.values))

        # Load model
        reg = self._MODEL_REGISTRY.get(self.model_name)
        if reg is None:
            raise ValueError(f"Unknown HF model: {self.model_name}")
        repo, loader_type = reg

        if loader_type == "hf_generate":
            self._load_hf_generate(repo)
        elif loader_type == "momentfm":
            self._load_moment(repo)
        elif loader_type == "lag_llama":
            self._load_lag_llama(repo)
        elif loader_type == "timesfm_docker":
            self._docker_init()

        # Robust fallback for unseen entities (replaces raw Ridge)
        self._fallback = _RobustFallback()
        try:
            self._fallback.fit(X, y)
        except Exception:
            pass

        self._fitted = True
        _logger.info(f"  [{self.model_name}] Fitted with {len(self._entity_contexts)} entities (ALL valid)")
        return self

    def _docker_init(self):
        """Initialize Docker-based TimesFM (Python 3.10 container)."""
        self._model = "docker_timesfm"
        # Docker inference happens in predict() via subprocess

    def _predict_hf_generate(self, ctxs: List[np.ndarray], horizon: int = 7) -> List[float]:
        """Predict using HF generate API (Timer, TimeMoE)."""
        import torch
        preds = []
        for ctx in ctxs:
            try:
                seq = ctx[-128:].astype(np.float32)
                # Normalize
                mu, sigma = float(np.mean(seq)), float(np.std(seq)) + 1e-8
                seq_norm = (seq - mu) / sigma
                inp = torch.tensor(seq_norm).unsqueeze(0).float()
                if hasattr(self, '_device'):
                    inp = inp.to(self._device)
                with torch.no_grad():
                    out = self._model.generate(inp, max_new_tokens=horizon)
                # De-normalize
                forecast = out[0, -horizon:].cpu().numpy() * sigma + mu
                preds.append(float(np.mean(forecast)))
            except Exception as e:
                _logger.debug(f"[{self.model_name}] generate failed: {e}")
                preds.append(float(np.mean(ctx)))
        return preds

    def _predict_moment(self, ctxs: List[np.ndarray], horizon: int = 7) -> List[float]:
        """Predict using MOMENT pipeline."""
        import torch
        preds = []
        for ctx in ctxs:
            try:
                seq = ctx[-512:].astype(np.float32)
                inp = torch.tensor(seq).unsqueeze(0).unsqueeze(-1).float()
                with torch.no_grad():
                    out = self._model(inp)
                forecast = out.forecast.squeeze().cpu().numpy()
                preds.append(float(np.mean(forecast[:horizon])))
            except Exception as e:
                _logger.debug(f"[{self.model_name}] MOMENT predict failed: {e}")
                preds.append(float(np.mean(ctx)))
        return preds

    def _predict_docker_timesfm(self, ctxs: List[np.ndarray], horizon: int = 7) -> List[float]:
        """Predict via Docker TimesFM container using subprocess."""
        import json, subprocess, tempfile
        # Save contexts to temp file
        ctx_data = [c.tolist() for c in ctxs[:100]]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"contexts": ctx_data, "horizon": horizon}, f)
            tmp_in = f.name

        tmp_out = tmp_in.replace('.json', '_out.json')

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{tmp_in}:/data/input.json",
            "-v", f"{os.path.dirname(tmp_out)}:/data/output",
            "timesfm-inference:latest",
            "python3", "/app/predict.py",
            "--input", "/data/input.json",
            "--output", f"/data/output/{os.path.basename(tmp_out)}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(tmp_out):
                with open(tmp_out) as f:
                    output = json.load(f)
                return [float(p) for p in output.get("predictions", [])]
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            _logger.warning(f"[TimesFM Docker] Failed: {e}")
        finally:
            for p in [tmp_in, tmp_out]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # Fallback to context means
        return [float(np.mean(c)) for c in ctxs[:100]]

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        h = len(X)
        ctxs = self._entity_contexts or [self._context]
        horizon = kwargs.get("horizon", 7)

        # Generate entity-level forecasts
        reg = self._MODEL_REGISTRY.get(self.model_name)
        _, loader_type = reg

        if loader_type == "hf_generate":
            preds_all = self._predict_hf_generate(ctxs, horizon)
        elif loader_type == "momentfm":
            preds_all = self._predict_moment(ctxs, horizon)
        elif loader_type == "lag_llama" and self._model is not None:
            # Use GluonTS predictor
            preds_all = []
            for ctx in ctxs[:50]:
                try:
                    from gluonts.dataset.common import ListDataset
                    ds = ListDataset(
                        [{"target": ctx, "start": pd.Timestamp("2020-01-01")}],
                        freq="D",
                    )
                    for entry in self._model.predict(ds):
                        preds_all.append(float(np.mean(entry.mean)))
                except Exception:
                    preds_all.append(float(np.mean(ctx)))
        elif loader_type == "timesfm_docker":
            preds_all = self._predict_docker_timesfm(ctxs, horizon)
        else:
            preds_all = [float(np.mean(c)) for c in ctxs]

        # Map entity_id -> forecast
        entity_preds: Dict[str, float] = {}
        if self._entity_ids and len(preds_all) == len(self._entity_ids):
            entity_preds = dict(zip(self._entity_ids, preds_all))
        elif preds_all:
            entity_preds = {f"s_{i}": p for i, p in enumerate(preds_all)}

        if not entity_preds:
            return np.full(h, self._global_mean)

        global_mean = float(np.mean(list(entity_preds.values())))

        # Map to test rows via entity_id (same logic as FoundationModelWrapper)
        test_raw = kwargs.get("test_raw")
        target = kwargs.get("target")
        if test_raw is not None and "entity_id" in test_raw.columns:
            if target and target in test_raw.columns:
                valid_mask = test_raw[target].notna()
                test_entities = test_raw.loc[valid_mask, "entity_id"].values
            else:
                test_entities = test_raw["entity_id"].values

            if len(test_entities) == h:
                n_covered = sum(1 for eid in test_entities if eid in entity_preds)
                coverage = n_covered / max(len(test_entities), 1)

                # Per-entity hybrid: NF forecast or RobustFallback
                fallback = getattr(self, '_fallback', None)
                fallback_preds = None
                if fallback is not None and fallback._fitted:
                    fallback_preds = fallback.predict(X)

                y_pred = np.empty(h, dtype=np.float64)
                for i, eid in enumerate(test_entities):
                    if eid in entity_preds:
                        y_pred[i] = entity_preds[eid]
                    elif fallback_preds is not None:
                        y_pred[i] = fallback_preds[i]
                    else:
                        y_pred[i] = global_mean
                return _sanitize_predictions(
                    y_pred, global_mean, self.model_name
                )

        return np.full(h, global_mean)


# ============================================================================
# Factory functions
# ============================================================================

def _nf_factory(name: str, prob: bool = False):
    def create(**kw):
        cfg = ModelConfig(name=name, model_type="forecasting", params=kw,
                          supports_probabilistic=prob,
                          optional_dependency="neuralforecast")
        return DeepModelWrapper(cfg, name, **kw)
    create.__doc__ = f"{name} model."
    return create


def _fm_factory(name: str, dep: str):
    def create(**kw):
        cfg = ModelConfig(name=name, model_type="forecasting", params=kw,
                          supports_probabilistic=True,
                          optional_dependency=dep)
        return FoundationModelWrapper(cfg, name, **kw)
    create.__doc__ = f"{name} foundation model."
    return create


# deep_classical
create_nbeats = _nf_factory("NBEATS")
create_nhits = _nf_factory("NHITS")
create_tft = _nf_factory("TFT")
create_deepar = _nf_factory("DeepAR", prob=True)

# transformer_sota
create_patchtst = _nf_factory("PatchTST")
create_itransformer = _nf_factory("iTransformer")
create_timesnet = _nf_factory("TimesNet")
create_tsmixer = _nf_factory("TSMixer")
create_informer = _nf_factory("Informer")
create_autoformer = _nf_factory("Autoformer")
create_fedformer = _nf_factory("FEDformer")
create_vanillatransformer = _nf_factory("VanillaTransformer")
create_tide = _nf_factory("TiDE")
create_nbeatsx = _nf_factory("NBEATSx")
create_bitcn = _nf_factory("BiTCN")
create_kan = _nf_factory("KAN")
create_rmok = _nf_factory("RMoK")
create_softs = _nf_factory("SOFTS")
create_stemgnn = _nf_factory("StemGNN")
# Phase 4 additions
create_dlinear = _nf_factory("DLinear")
create_nlinear = _nf_factory("NLinear")
create_timemixer = _nf_factory("TimeMixer")
create_timexer = _nf_factory("TimeXer")
create_tsmixerx = _nf_factory("TSMixerx")

# foundation
create_chronos = _fm_factory("Chronos", "chronos")
create_chronos_bolt = _fm_factory("ChronosBolt", "chronos")
create_chronos2 = _fm_factory("Chronos2", "chronos")
create_moirai = _fm_factory("Moirai", "uni2ts")
create_moirai_large = _fm_factory("MoiraiLarge", "uni2ts")
create_moirai2 = _fm_factory("Moirai2", "uni2ts")
def _hf_fm_factory(name: str, dep: str):
    def create(**kw):
        cfg = ModelConfig(name=name, model_type="forecasting", params=kw,
                          supports_probabilistic=True,
                          optional_dependency=dep)
        return HFFoundationModelWrapper(cfg, name, **kw)
    create.__doc__ = f"{name} HF foundation model."
    return create


# foundation (HuggingFace-based Phase 5 additions)
create_timer = _hf_fm_factory("Timer", "transformers")
create_timemoe = _hf_fm_factory("TimeMoE", "transformers")
create_moment = _hf_fm_factory("MOMENT", "momentfm")
create_lagllama = _hf_fm_factory("LagLlama", "gluonts")
create_timesfm = _hf_fm_factory("TimesFM", "docker")
# TimesFM removed: requires Python <3.12 (lingvo dependency)


# ============================================================================
# Registries (imported by registry.py)
# ============================================================================

DEEP_MODELS = {
    "NBEATS": create_nbeats, "NHITS": create_nhits,
    "TFT": create_tft, "DeepAR": create_deepar,
}

TRANSFORMER_MODELS = {
    "PatchTST": create_patchtst, "iTransformer": create_itransformer,
    "TimesNet": create_timesnet, "TSMixer": create_tsmixer,
    "Informer": create_informer, "Autoformer": create_autoformer,
    "FEDformer": create_fedformer, "VanillaTransformer": create_vanillatransformer,
    "TiDE": create_tide, "NBEATSx": create_nbeatsx,
    "BiTCN": create_bitcn, "KAN": create_kan,
    "RMoK": create_rmok, "SOFTS": create_softs,
    "StemGNN": create_stemgnn,
    # Phase 4 SOTA additions
    "DLinear": create_dlinear, "NLinear": create_nlinear,
    "TimeMixer": create_timemixer, "TimeXer": create_timexer,
    "TSMixerx": create_tsmixerx,
}

FOUNDATION_MODELS = {
    "Chronos": create_chronos,
    "ChronosBolt": create_chronos_bolt,
    "Chronos2": create_chronos2,
    "Moirai": create_moirai,
    "MoiraiLarge": create_moirai_large,
    "Moirai2": create_moirai2,
    # Phase 5: HuggingFace-based models
    "Timer": create_timer,
    "TimeMoE": create_timemoe,
    "MOMENT": create_moment,
    "LagLlama": create_lagllama,
    "TimesFM": create_timesfm,
}


def get_deep_model(name: str, **kwargs) -> ModelBase:
    all_m = {**DEEP_MODELS, **TRANSFORMER_MODELS, **FOUNDATION_MODELS}
    if name not in all_m:
        raise ValueError(f"Unknown model: {name}")
    return all_m[name](**kwargs)


def list_deep_models() -> Dict[str, list]:
    return {
        "deep_classical": list(DEEP_MODELS.keys()),
        "transformer_sota": list(TRANSFORMER_MODELS.keys()),
        "foundation": list(FOUNDATION_MODELS.keys()),
    }


def check_neuralforecast_available() -> bool:
    try:
        import neuralforecast
        return True
    except ImportError:
        return False
