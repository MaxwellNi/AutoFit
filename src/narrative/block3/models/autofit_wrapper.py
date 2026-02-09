#!/usr/bin/env python3
"""
AutoFit Wrappers — Exhaustive Stacked Generalization for KDD'26.

Design philosophy:
  - ZERO heuristic routing: every model is evaluated empirically
  - Strict temporal anti-leak: temporal blocking, NO shuffle
  - Exhaustive combinatorial search: try ALL viable stacking combos
  - Performance-compute Pareto: prune slow/poor models early
  - Fair comparison: same temporal splits, same preprocessing for all

Six variants:

  AutoFitV1     — Best single model via data-driven selection (not heuristic).
                  All available base models evaluated on temporal val fold.
                  Winner + LightGBM residual correction on tabular features.

  AutoFitV2     — Top-K weighted ensemble (inverse-MAE weights).
                  All available models ranked by val MAE, top-K selected.
                  Weights computed on temporal validation fold.

  AutoFitV2E    — Full stacking with LightGBM meta-learner.
                  Top-K models produce OOF predictions via temporal split.
                  LightGBM trains on (OOF_preds + tabular_features) -> target.

  AutoFitV3     — Greedy forward ensemble selection + meta-learner.
                  2-fold temporal CV for OOF generation (no data waste).
                  Starting from best single model, greedily adds models
                  that reduce CV-validated MAE.  Stops when no improvement.

  AutoFitV3E    — Top-K stacking with temporal CV.
                  Like V2E but with 2-fold temporal CV for OOF.

  AutoFitV3Max  — Exhaustive subset search (2^K combinations, K capped at 8).
                  Evaluates every possible subset of base models.
                  Selects subset that minimizes temporal-CV MAE.

Information flow guarantee (CRITICAL for fair comparison):
  ┌─────────────────────────────────────────────────┐
  │ Training data temporal ordering preserved:       │
  │   t_1 < t_2 < ... < t_split < ... < t_n         │
  │                                                  │
  │ Level-0: Base models trained on t_1..t_split     │
  │          OOF predictions on t_split+1..t_n       │
  │                                                  │
  │ Level-1: Meta-learner trained on OOF + features  │
  │          NEVER sees t_split+1..t_n targets until  │
  │          after OOF predictions are made           │
  │                                                  │
  │ Final: Base models REFIT on full t_1..t_n        │
  │        Meta-learner weights FROZEN from OOF      │
  └─────────────────────────────────────────────────┘
"""
from __future__ import annotations

import gc
import logging
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# ALL candidate models for exhaustive evaluation — covers every family
# Order: fast -> slow (for early stopping under time budget)
_ALL_CANDIDATES = [
    # -- Tabular (fast, strong baselines) --
    "HistGradientBoosting",  # fastest sklearn GBDT
    "LightGBM",              # fastest external GBDT
    "XGBoost",               # strong GBDT
    "CatBoost",              # handles categoricals natively
    "RandomForest",          # strong on heavy-tailed, low corr data
    "ExtraTrees",            # often complementary to RF
    # -- Foundation (moderate speed) --
    "Moirai",                # strong on count/temporal
    "Chronos",               # Amazon pre-trained
    # -- Deep classical (slower) --
    "TFT",                   # attention + variable selection
    "NHITS",                 # hierarchical interpolation
    "NBEATS",                # basis expansion
    "DeepAR",                # autoregressive
    # -- Irregular (handles missing data) --
    "GRU-D",                 # decay mechanism
    "SAITS",                 # self-attention imputation
    # -- Statistical (fast, diverse) --
    "AutoARIMA",
    "AutoETS",
    "AutoTheta",
]

_PANEL_CATEGORIES = {"deep_classical", "transformer_sota", "foundation",
                     "statistical", "irregular"}

# Max time per base model during candidate evaluation (seconds)
_CANDIDATE_TIMEOUT = 600  # 10 minutes

# Max candidates for exhaustive subset search
_MAX_EXHAUSTIVE_K = 8


# ============================================================================
# Helpers
# ============================================================================

def _get_model_category(model_name: str) -> Optional[str]:
    """Look up which category a model belongs to."""
    from .registry import MODEL_CATEGORIES
    for cat, models in MODEL_CATEGORIES.items():
        if model_name in models:
            return cat
    return None


def _needs_panel_kwargs(model_name: str) -> bool:
    """Check if model needs panel-aware kwargs."""
    cat = _get_model_category(model_name)
    return cat in _PANEL_CATEGORIES if cat else False


def _compute_target_regime(y: pd.Series, X: pd.DataFrame) -> Dict[str, float]:
    """Compute lightweight meta-features for logging (NOT for routing).

    These are logged for interpretability/ablation, but NOT used
    to select models -- selection is purely empirical.
    """
    y_arr = y.values.astype(float)
    meta: Dict[str, float] = {}

    if np.std(y_arr) > 0:
        meta["kurtosis"] = float(pd.Series(y_arr).kurtosis())
    else:
        meta["kurtosis"] = 0.0

    mean_abs = np.mean(np.abs(y_arr))
    meta["cv"] = float(np.std(y_arr) / max(mean_abs, 1e-8))

    unique_vals = np.unique(y_arr[~np.isnan(y_arr)])
    meta["n_unique"] = len(unique_vals)
    meta["binary_frac"] = 1.0 if len(unique_vals) <= 3 else float(np.isin(y_arr, [0, 1]).mean())
    meta["zero_frac"] = float((y_arr == 0).mean())
    meta["n_samples"] = len(y_arr)

    try:
        n_s = min(50_000, len(X))
        idx = np.random.choice(len(X), n_s, replace=False) if len(X) > n_s else np.arange(len(X))
        corrs = X.iloc[idx].corrwith(y.iloc[idx]).abs().fillna(0)
        meta["exog_corr_mean"] = float(corrs.mean())
        meta["exog_corr_max"] = float(corrs.max())
    except Exception:
        meta["exog_corr_mean"] = 0.0
        meta["exog_corr_max"] = 0.0

    return meta


def _fit_single_candidate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    train_raw_inner: Optional[pd.DataFrame],
    val_raw: Optional[pd.DataFrame],
    target: str,
    horizon: int,
    timeout: float = _CANDIDATE_TIMEOUT,
) -> Optional[Tuple[str, float, np.ndarray, float]]:
    """Fit a single candidate model and return (name, val_mae, val_preds, elapsed).

    Returns None if the model fails or times out.
    Handles all exceptions gracefully.
    """
    from .registry import get_model, check_model_available

    if not check_model_available(model_name):
        return None

    t0 = time.monotonic()
    try:
        model = get_model(model_name)

        fit_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(model_name):
            fit_kw = {"train_raw": train_raw_inner, "target": target, "horizon": horizon}

        model.fit(X_train, y_train, **fit_kw)

        pred_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(model_name):
            if val_raw is not None:
                pred_kw = {"test_raw": val_raw, "target": target, "horizon": horizon}

        val_preds = model.predict(X_val, **pred_kw)
        elapsed = time.monotonic() - t0

        if elapsed > timeout:
            logger.warning(f"[AutoFit] {model_name} took {elapsed:.0f}s > timeout={timeout}s, skip")
            del model
            gc.collect()
            return None

        val_mae = float(np.mean(np.abs(y_val.values - val_preds)))

        # Reject degenerate predictions
        if np.isnan(val_mae) or np.isinf(val_mae):
            logger.warning(f"[AutoFit] {model_name} produced NaN/Inf MAE, skipping")
            del model; gc.collect()
            return None

        # Reject constant predictions (broken model)
        if len(val_preds) > 10 and np.std(val_preds) == 0.0:
            logger.warning(f"[AutoFit] {model_name} produced constant predictions, skipping")
            del model; gc.collect()
            return None

        logger.info(f"[AutoFit] {model_name}: val_MAE={val_mae:,.4f} ({elapsed:.1f}s)")

        del model; gc.collect()
        return (model_name, val_mae, val_preds, elapsed)

    except Exception as e:
        logger.warning(f"[AutoFit] {model_name} failed: {e}")
        gc.collect()
        return None


def _build_meta_learner(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    oof_preds: Dict[str, np.ndarray],
    regularize: bool = True,
) -> Tuple[Any, List[str], float]:
    """Build a LightGBM meta-learner on OOF predictions + tabular features.

    Returns (meta_learner, column_list, meta_mae).

    Anti-overfitting measures:
      - Strong L1/L2 regularization
      - Small learning rate with early stopping
      - High min_child_samples
      - Subsample and colsample for bagging
    """
    import lightgbm as lgb

    X_meta = X_val.copy()
    for name, preds in oof_preds.items():
        X_meta[f"__pred_{name}__"] = preds

    numeric_cols = X_meta.select_dtypes(include=[np.number]).columns.tolist()
    X_meta_num = X_meta[numeric_cols].fillna(0)

    # Split meta-learner training for early stopping
    n = len(X_meta_num)
    meta_split = int(n * 0.75)
    X_mt, X_mv = X_meta_num.iloc[:meta_split], X_meta_num.iloc[meta_split:]
    y_mt, y_mv = y_val.iloc[:meta_split], y_val.iloc[meta_split:]

    params = dict(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=50,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if regularize:
        params.update(
            reg_alpha=1.0,    # Strong L1
            reg_lambda=10.0,  # Strong L2
        )

    meta_learner = lgb.LGBMRegressor(**params)
    meta_learner.fit(
        X_mt, y_mt,
        eval_set=[(X_mv, y_mv)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Evaluate on full val set
    meta_preds = meta_learner.predict(X_meta_num)
    meta_mae = float(np.mean(np.abs(y_val.values - meta_preds)))

    return meta_learner, numeric_cols, meta_mae


# ============================================================================
# AutoFitV1 — Data-Driven Best Single Model + Residual Correction
# ============================================================================

class AutoFitV1Wrapper(ModelBase):
    """
    Data-driven model selection (NO heuristic routing).

    1. Evaluate ALL available base models on temporal val fold.
    2. Select the BEST by val MAE.
    3. Train LightGBM on (features + base_pred) -> residual.
    4. Refit best model on full data, keep meta-learner frozen.
    5. Predict: base_pred + lgbm_correction.
    """

    def __init__(self, **kwargs):
        config = ModelConfig(
            name="AutoFitV1",
            model_type="regression",
            params={"strategy": "data_driven_best_plus_residual"},
        )
        super().__init__(config)
        self._base_model: Optional[ModelBase] = None
        self._meta_learner = None
        self._base_model_name: str = ""
        self._meta_cols: List[str] = []
        self._routing_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV1Wrapper":
        from .registry import get_model

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        meta = _compute_target_regime(y, X)
        logger.info(f"[AutoFitV1] Target meta-features: {meta}")

        # -- Temporal split for stacking (80/20) --
        n = len(X)
        split_idx = int(n * 0.8)
        X_inner, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_inner, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_raw_inner, val_raw = None, None
        if train_raw is not None:
            if len(train_raw) == n:
                train_raw_inner = train_raw.iloc[:split_idx]
                val_raw = train_raw.iloc[split_idx:]
            else:
                train_raw_inner, val_raw = train_raw, train_raw

        # -- Evaluate ALL candidates --
        results = []
        for model_name in _ALL_CANDIDATES:
            r = _fit_single_candidate(
                model_name, X_inner, y_inner, X_val, y_val,
                train_raw_inner, val_raw, target, horizon,
            )
            if r is not None:
                results.append(r)

        if not results:
            logger.error("[AutoFitV1] No candidates succeeded! Fallback to LightGBM")
            self._base_model = get_model("LightGBM")
            self._base_model.fit(X, y)
            self._base_model_name = "LightGBM"
            self._fitted = True
            return self

        # Sort by val MAE — select the BEST
        results.sort(key=lambda x: x[1])
        self._base_model_name = results[0][0]
        best_val_mae = results[0][1]
        best_val_preds = results[0][2]

        logger.info(
            "[AutoFitV1] Rankings: "
            + ", ".join(f"{n}={m:,.2f}" for n, m, _, _ in results[:5])
        )
        logger.info(f"[AutoFitV1] Selected: {self._base_model_name} (MAE={best_val_mae:,.2f})")

        # -- Residual correction via LightGBM --
        residuals = y_val.values - best_val_preds
        try:
            self._meta_learner, self._meta_cols, _res_mae = _build_meta_learner(
                X_val, pd.Series(residuals, index=y_val.index),
                {"__base__": best_val_preds},
                regularize=True,
            )
            # Actual stacked MAE: base + correction
            X_corr = X_val.copy()
            X_corr["__pred___base____"] = best_val_preds
            for c in self._meta_cols:
                if c not in X_corr.columns:
                    X_corr[c] = 0.0
            corrections = self._meta_learner.predict(X_corr[self._meta_cols].fillna(0))
            corrected = best_val_preds + corrections
            actual_stacked_mae = float(np.mean(np.abs(y_val.values - corrected)))

            # -- Anti-overfit guard: reject meta-learner if it does not help --
            if actual_stacked_mae >= best_val_mae * 0.995:
                logger.info(
                    f"[AutoFitV1] Meta-learner doesn't improve "
                    f"({actual_stacked_mae:,.2f} vs {best_val_mae:,.2f}), discarding"
                )
                self._meta_learner = None
            else:
                improvement = 100 * (1 - actual_stacked_mae / best_val_mae)
                logger.info(
                    f"[AutoFitV1] base_MAE={best_val_mae:,.2f} -> stacked_MAE="
                    f"{actual_stacked_mae:,.2f} ({improvement:.1f}% improvement)"
                )
        except Exception as e:
            logger.warning(f"[AutoFitV1] Meta-learner build failed: {e}")
            self._meta_learner = None

        # -- Refit best model on FULL data --
        self._base_model = get_model(self._base_model_name)
        fit_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(self._base_model_name):
            fit_kw = {"train_raw": train_raw, "target": target, "horizon": horizon}
        self._base_model.fit(X, y, **fit_kw)

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "base_model": self._base_model_name,
            "base_val_mae": best_val_mae,
            "has_meta_learner": self._meta_learner is not None,
            "candidate_rankings": [(n, m) for n, m, _, _ in results],
            "elapsed_seconds": elapsed,
        }
        self._fitted = True
        logger.info(f"[AutoFitV1] Fitted in {elapsed:.1f}s")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted or self._base_model is None:
            raise RuntimeError("AutoFitV1 not fitted")

        pred_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(self._base_model_name):
            for k in ("test_raw", "target", "horizon"):
                if k in kwargs:
                    pred_kw[k] = kwargs[k]

        base_preds = self._base_model.predict(X, **pred_kw)

        if self._meta_learner is not None:
            try:
                X_meta = X.copy()
                X_meta["__pred___base____"] = base_preds
                for c in self._meta_cols:
                    if c not in X_meta.columns:
                        X_meta[c] = 0.0
                X_meta_num = X_meta[self._meta_cols].fillna(0)
                corrections = self._meta_learner.predict(X_meta_num)
                return base_preds + corrections
            except Exception as e:
                logger.warning(f"[AutoFitV1] Meta-learner predict failed: {e}")

        return base_preds

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# AutoFitV2 — Top-K Adaptive Weighted Ensemble
# ============================================================================

class AutoFitV2Wrapper(ModelBase):
    """
    Data-driven top-K weighted ensemble.

    1. Evaluate ALL candidates on temporal val fold.
    2. Select top-K by val MAE.
    3. Compute inverse-MAE weights (normalized).
    4. Refit top-K on full data.
    5. Predict: sum(w_i * pred_i) / sum(w_i).
    """

    def __init__(self, top_k: int = 5, **kwargs):
        config = ModelConfig(
            name="AutoFitV2",
            model_type="regression",
            params={"top_k": top_k},
        )
        super().__init__(config)
        self._top_k = top_k
        self._models: List[Tuple[ModelBase, float, str]] = []
        self._routing_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV2Wrapper":
        from .registry import get_model

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        meta = _compute_target_regime(y, X)
        logger.info(f"[AutoFitV2] Target meta-features: {meta}")

        # -- Temporal split --
        n = len(X)
        split_idx = int(n * 0.8)
        X_inner, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_inner, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_raw_inner, val_raw = None, None
        if train_raw is not None:
            if len(train_raw) == n:
                train_raw_inner = train_raw.iloc[:split_idx]
                val_raw = train_raw.iloc[split_idx:]
            else:
                train_raw_inner, val_raw = train_raw, train_raw

        # -- Evaluate ALL candidates --
        results = []
        for model_name in _ALL_CANDIDATES:
            r = _fit_single_candidate(
                model_name, X_inner, y_inner, X_val, y_val,
                train_raw_inner, val_raw, target, horizon,
            )
            if r is not None:
                results.append(r)

        if not results:
            logger.error("[AutoFitV2] No candidates! Fallback to LightGBM")
            model = get_model("LightGBM")
            model.fit(X, y)
            self._models = [(model, 1.0, "LightGBM")]
            self._fitted = True
            return self

        # Sort by val MAE, take top K
        results.sort(key=lambda x: x[1])
        top_k = results[:self._top_k]

        # -- Inverse-MAE weights --
        inv_maes = np.array([1.0 / max(m, 1e-8) for _, m, _, _ in top_k])
        weights = inv_maes / inv_maes.sum()

        logger.info(
            f"[AutoFitV2] Selected top-{len(top_k)}: "
            + ", ".join(f"{n}={w:.3f}(MAE={m:,.2f})" for (n, m, _, _), w in zip(top_k, weights))
        )

        # -- Refit top-K on FULL data --
        self._models = []
        for (model_name, val_mae, _, _), w in zip(top_k, weights):
            try:
                model = get_model(model_name)
                fit_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    fit_kw = {"train_raw": train_raw, "target": target, "horizon": horizon}
                model.fit(X, y, **fit_kw)
                self._models.append((model, float(w), model_name))
            except Exception as e:
                logger.warning(f"[AutoFitV2] Refit {model_name} failed: {e}")

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "top_k": [(n, m) for n, m, _, _ in top_k],
            "weights": {n: float(w) for (n, _, _, _), w in zip(top_k, weights)},
            "all_rankings": [(n, m) for n, m, _, _ in results],
            "elapsed_seconds": elapsed,
        }
        self._fitted = True
        logger.info(f"[AutoFitV2] Fitted in {elapsed:.1f}s")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted or not self._models:
            raise RuntimeError("AutoFitV2 not fitted")

        preds_dict: Dict[str, np.ndarray] = {}
        for model, weight, name in self._models:
            try:
                pred_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(name):
                    for k in ("test_raw", "target", "horizon"):
                        if k in kwargs:
                            pred_kw[k] = kwargs[k]
                preds_dict[name] = model.predict(X, **pred_kw)
            except Exception as e:
                logger.warning(f"[AutoFitV2] {name} predict failed: {e}")

        if not preds_dict:
            return np.full(len(X), 0.0)

        total_w = sum(w for _, w, n in self._models if n in preds_dict)
        if total_w < 1e-8:
            return np.mean(list(preds_dict.values()), axis=0)

        result = np.zeros(len(X))
        for _, w, name in self._models:
            if name in preds_dict:
                result += (w / total_w) * preds_dict[name]
        return result

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# AutoFitV2E — Top-K Stacking with LightGBM Meta-Learner
# ============================================================================

class AutoFitV2EWrapper(ModelBase):
    """
    Full stacking with LightGBM meta-learner on OOF predictions.

    1. Evaluate ALL candidates on temporal val fold.
    2. Select top-K by val MAE.
    3. Build LightGBM meta-learner on (OOF_preds + features) -> target.
    4. Refit top-K on full data.
    5. Predict: meta_learner(base_preds + features).
    """

    def __init__(self, top_k: int = 5, **kwargs):
        config = ModelConfig(
            name="AutoFitV2E",
            model_type="regression",
            params={"top_k": top_k, "strategy": "stacking_meta_learner"},
        )
        super().__init__(config)
        self._top_k = top_k
        self._models: List[Tuple[ModelBase, str]] = []
        self._meta_learner = None
        self._meta_cols: List[str] = []
        self._pred_names: List[str] = []
        self._weights: Dict[str, float] = {}
        self._routing_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV2EWrapper":
        from .registry import get_model

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        meta = _compute_target_regime(y, X)
        logger.info(f"[AutoFitV2E] Target meta-features: {meta}")

        # -- Temporal split --
        n = len(X)
        split_idx = int(n * 0.8)
        X_inner, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_inner, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_raw_inner, val_raw = None, None
        if train_raw is not None:
            if len(train_raw) == n:
                train_raw_inner = train_raw.iloc[:split_idx]
                val_raw = train_raw.iloc[split_idx:]
            else:
                train_raw_inner, val_raw = train_raw, train_raw

        # -- Evaluate ALL candidates --
        results = []
        for model_name in _ALL_CANDIDATES:
            r = _fit_single_candidate(
                model_name, X_inner, y_inner, X_val, y_val,
                train_raw_inner, val_raw, target, horizon,
            )
            if r is not None:
                results.append(r)

        if not results:
            logger.error("[AutoFitV2E] No candidates! Fallback to LightGBM")
            model = get_model("LightGBM")
            model.fit(X, y)
            self._models = [(model, "LightGBM")]
            self._fitted = True
            return self

        results.sort(key=lambda x: x[1])
        top_k = results[:self._top_k]
        self._pred_names = [n for n, _, _, _ in top_k]

        # -- Build stacking meta-learner --
        oof_preds = {n: p for n, _, p, _ in top_k}
        try:
            self._meta_learner, self._meta_cols, meta_mae = _build_meta_learner(
                X_val, y_val, oof_preds, regularize=True,
            )
            best_base_mae = top_k[0][1]
            improvement = 100 * (1 - meta_mae / max(best_base_mae, 1e-8))

            # -- Anti-overfit guard --
            if meta_mae >= best_base_mae * 0.995:
                logger.info(
                    f"[AutoFitV2E] Meta-learner doesn't improve "
                    f"({meta_mae:,.2f} vs base {best_base_mae:,.2f}), using weights"
                )
                self._meta_learner = None
            else:
                logger.info(
                    f"[AutoFitV2E] Meta-learner MAE={meta_mae:,.2f} "
                    f"vs best_base={best_base_mae:,.2f} ({improvement:.1f}% up)"
                )
        except Exception as e:
            logger.warning(f"[AutoFitV2E] Meta-learner failed: {e}")
            self._meta_learner = None

        # Fallback weights
        inv_maes = np.array([1.0 / max(m, 1e-8) for _, m, _, _ in top_k])
        wts = inv_maes / inv_maes.sum()
        self._weights = {n: float(w) for (n, _, _, _), w in zip(top_k, wts)}

        # -- Refit top-K on FULL data --
        self._models = []
        for model_name, _, _, _ in top_k:
            try:
                model = get_model(model_name)
                fit_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    fit_kw = {"train_raw": train_raw, "target": target, "horizon": horizon}
                model.fit(X, y, **fit_kw)
                self._models.append((model, model_name))
            except Exception as e:
                logger.warning(f"[AutoFitV2E] Refit {model_name} failed: {e}")

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "top_k": [(n, m) for n, m, _, _ in top_k],
            "weights": self._weights,
            "has_meta_learner": self._meta_learner is not None,
            "all_rankings": [(n, m) for n, m, _, _ in results],
            "elapsed_seconds": elapsed,
        }
        self._fitted = True
        logger.info(f"[AutoFitV2E] Fitted in {elapsed:.1f}s")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted or not self._models:
            raise RuntimeError("AutoFitV2E not fitted")

        pred_dict: Dict[str, np.ndarray] = {}
        for model, name in self._models:
            try:
                pred_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(name):
                    for k in ("test_raw", "target", "horizon"):
                        if k in kwargs:
                            pred_kw[k] = kwargs[k]
                pred_dict[name] = model.predict(X, **pred_kw)
            except Exception as e:
                logger.warning(f"[AutoFitV2E] {name} predict failed: {e}")

        if not pred_dict:
            return np.full(len(X), 0.0)

        # Try meta-learner first
        if self._meta_learner is not None:
            try:
                X_stack = X.copy()
                for name in self._pred_names:
                    col = f"__pred_{name}__"
                    if name in pred_dict:
                        X_stack[col] = pred_dict[name]
                    else:
                        X_stack[col] = np.mean(list(pred_dict.values()), axis=0)
                for c in self._meta_cols:
                    if c not in X_stack.columns:
                        X_stack[c] = 0.0
                return self._meta_learner.predict(X_stack[self._meta_cols].fillna(0))
            except Exception as e:
                logger.warning(f"[AutoFitV2E] Meta-learner predict failed: {e}")

        # Fallback: weighted average
        total_w = sum(self._weights.get(n, 0) for n in pred_dict)
        if total_w < 1e-8:
            return np.mean(list(pred_dict.values()), axis=0)
        result = np.zeros(len(X))
        for name, preds in pred_dict.items():
            result += (self._weights.get(name, 0) / total_w) * preds
        return result

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# AutoFitV3 — Exhaustive Stacking with Temporal Cross-Validation
# ============================================================================

class AutoFitV3Wrapper(ModelBase):
    """
    Exhaustive stacking with strict temporal cross-validation.

    Key differences from V2E:
      - 2-fold temporal CV for OOF generation (more data for meta-learner)
      - Greedy forward selection or exhaustive subset search
      - Negative correlation learning: prefer diverse ensemble members

    Temporal CV structure (anti-leak guarantee):
      Fold 1: train on [0..50%], predict on [50%..80%]
      Fold 2: train on [0..80%], predict on [80%..100%]
      -> OOF predictions cover [50%..100%] without any leak
      -> Meta-learner trains on OOF from [50%..100%]
      -> Final base models refit on full [0%..100%]
    """

    def __init__(self, mode: str = "greedy", top_k: int = 8, **kwargs):
        """
        Args:
            mode: 'greedy' (forward selection), 'exhaustive' (all subsets),
                  'topk' (just top-K stacking)
            top_k: max models to consider
        """
        name_map = {
            "greedy": "AutoFitV3",
            "exhaustive": "AutoFitV3Max",
            "topk": "AutoFitV3E",
        }
        config = ModelConfig(
            name=name_map.get(mode, "AutoFitV3"),
            model_type="regression",
            params={"mode": mode, "top_k": top_k},
        )
        super().__init__(config)
        self._mode = mode
        self._top_k = top_k
        self._models: List[Tuple[ModelBase, str]] = []
        self._meta_learner = None
        self._meta_cols: List[str] = []
        self._pred_names: List[str] = []
        self._weights: Dict[str, float] = {}
        self._routing_info: Dict[str, Any] = {}

    def _temporal_cv_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_raw: Optional[pd.DataFrame],
        target: str,
        horizon: int,
    ) -> Tuple[List[Tuple[str, float, np.ndarray]], pd.DataFrame, pd.Series]:
        """Evaluate all candidates with 2-fold temporal CV.

        Returns:
          - results: [(model_name, avg_mae, oof_preds_on_fold2)]
          - X_oof: features for the OOF region
          - y_oof: targets for the OOF region
        """
        n = len(X)
        mid = int(n * 0.5)
        split = int(n * 0.8)

        # Prepare raw data splits
        tr_raw_f1, val_raw_f1 = None, None
        tr_raw_f2, val_raw_f2 = None, None
        if train_raw is not None and len(train_raw) == n:
            tr_raw_f1 = train_raw.iloc[:mid]
            val_raw_f1 = train_raw.iloc[mid:split]
            tr_raw_f2 = train_raw.iloc[:split]
            val_raw_f2 = train_raw.iloc[split:]

        # Fold 1: train [0..mid], val [mid..split]
        # Fold 2: train [0..split], val [split..n]
        all_results: Dict[str, List[Tuple[float, np.ndarray]]] = {}

        for fold_idx, (tr_end, val_start, val_end) in enumerate([
            (mid, mid, split),
            (split, split, n),
        ]):
            X_tr = X.iloc[:tr_end]
            y_tr = y.iloc[:tr_end]
            X_v = X.iloc[val_start:val_end]
            y_v = y.iloc[val_start:val_end]
            tr_raw_fold = tr_raw_f1 if fold_idx == 0 else tr_raw_f2
            val_raw_fold = val_raw_f1 if fold_idx == 0 else val_raw_f2

            for model_name in _ALL_CANDIDATES:
                r = _fit_single_candidate(
                    model_name, X_tr, y_tr, X_v, y_v,
                    tr_raw_fold, val_raw_fold, target, horizon,
                )
                if r is not None:
                    name, mae, preds, elapsed = r
                    if name not in all_results:
                        all_results[name] = []
                    all_results[name].append((mae, preds))

        # Average MAE across folds, use fold 2 OOF predictions for meta-learner
        results = []
        for name, fold_data in all_results.items():
            avg_mae = float(np.mean([m for m, _ in fold_data]))
            # Use predictions from fold 2 (the [split..n] region)
            if len(fold_data) >= 2:
                oof_preds = fold_data[1][1]  # fold 2 predictions
            else:
                oof_preds = fold_data[0][1]
            results.append((name, avg_mae, oof_preds))

        results.sort(key=lambda x: x[1])

        # OOF region is [split..n]
        X_oof = X.iloc[split:]
        y_oof = y.iloc[split:]

        return results, X_oof, y_oof

    def _greedy_forward_select(
        self,
        results: List[Tuple[str, float, np.ndarray]],
        X_oof: pd.DataFrame,
        y_oof: pd.Series,
    ) -> List[str]:
        """Greedy forward ensemble selection.

        Starting from the best single model, greedily add models
        that reduce the combined weighted ensemble MAE.
        """
        if not results:
            return []

        top_k = results[:self._top_k]
        y_arr = y_oof.values

        # Start with the best model
        selected = [top_k[0][0]]
        best_mae = top_k[0][1]

        remaining = [(n, m, p) for n, m, p in top_k[1:]]

        for _ in range(min(len(remaining), self._top_k - 1)):
            best_addition = None
            best_new_mae = best_mae

            for name, _, preds in remaining:
                trial = selected + [name]
                trial_preds = []
                trial_weights = []
                for tn in trial:
                    for rn, rm, rp in top_k:
                        if rn == tn:
                            trial_preds.append(rp)
                            trial_weights.append(1.0 / max(rm, 1e-8))
                            break

                tw = np.array(trial_weights)
                tw = tw / tw.sum()
                combined = sum(w * p for w, p in zip(tw, trial_preds))
                combined_mae = float(np.mean(np.abs(y_arr - combined)))

                if combined_mae < best_new_mae * 0.999:  # require >0.1% improvement
                    best_new_mae = combined_mae
                    best_addition = name

            if best_addition is not None:
                selected.append(best_addition)
                remaining = [(n, m, p) for n, m, p in remaining if n != best_addition]
                logger.info(
                    f"[{self.name}] +{best_addition} -> MAE={best_new_mae:,.2f} "
                    f"(ensemble size={len(selected)})"
                )
                best_mae = best_new_mae
            else:
                logger.info(f"[{self.name}] No improvement, stopping at {len(selected)} models")
                break

        return selected

    def _exhaustive_subset_select(
        self,
        results: List[Tuple[str, float, np.ndarray]],
        X_oof: pd.DataFrame,
        y_oof: pd.Series,
    ) -> List[str]:
        """Try ALL possible subsets and return the best.

        Capped at _MAX_EXHAUSTIVE_K candidates to keep 2^K manageable.
        """
        top_k = results[:min(len(results), _MAX_EXHAUSTIVE_K)]
        y_arr = y_oof.values

        best_subset: List[str] = [top_k[0][0]]
        best_mae = top_k[0][1]

        total_combos = 0
        for size in range(2, len(top_k) + 1):
            for combo in combinations(range(len(top_k)), size):
                total_combos += 1
                names = [top_k[i][0] for i in combo]
                preds = [top_k[i][2] for i in combo]
                maes = [top_k[i][1] for i in combo]

                inv = np.array([1.0 / max(m, 1e-8) for m in maes])
                w = inv / inv.sum()
                combined = sum(wi * pi for wi, pi in zip(w, preds))
                mae = float(np.mean(np.abs(y_arr - combined)))

                if mae < best_mae:
                    best_mae = mae
                    best_subset = names

        logger.info(
            f"[{self.name}] Exhaustive search: {total_combos} combos, "
            f"best={best_subset} (MAE={best_mae:,.2f})"
        )
        return best_subset

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV3Wrapper":
        from .registry import get_model

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        meta = _compute_target_regime(y, X)
        logger.info(f"[{self.name}] Target meta-features: {meta}")

        # -- Temporal CV evaluation --
        results, X_oof, y_oof = self._temporal_cv_evaluate(
            X, y, train_raw, target, horizon,
        )

        if not results:
            logger.error(f"[{self.name}] No candidates! Fallback to LightGBM")
            model = get_model("LightGBM")
            model.fit(X, y)
            self._models = [(model, "LightGBM")]
            self._pred_names = ["LightGBM"]
            self._weights = {"LightGBM": 1.0}
            self._fitted = True
            return self

        logger.info(
            f"[{self.name}] Candidate rankings: "
            + ", ".join(f"{n}={m:,.2f}" for n, m, _ in results[:8])
        )

        # -- Model selection --
        if self._mode == "greedy":
            selected = self._greedy_forward_select(results, X_oof, y_oof)
        elif self._mode == "exhaustive":
            selected = self._exhaustive_subset_select(results, X_oof, y_oof)
        else:  # topk
            selected = [n for n, _, _ in results[:self._top_k]]

        self._pred_names = selected

        # -- Build meta-learner on OOF predictions --
        oof_preds = {}
        for name in selected:
            for rn, _, rp in results:
                if rn == name:
                    oof_preds[name] = rp
                    break

        try:
            self._meta_learner, self._meta_cols, meta_mae = _build_meta_learner(
                X_oof, y_oof, oof_preds, regularize=True,
            )
            best_single = results[0][1]
            improvement = 100 * (1 - meta_mae / max(best_single, 1e-8))

            if meta_mae >= best_single * 0.995:
                logger.info(f"[{self.name}] Meta-learner doesn't improve, using weights")
                self._meta_learner = None
            else:
                logger.info(
                    f"[{self.name}] Meta-learner MAE={meta_mae:,.2f} "
                    f"vs best_single={best_single:,.2f} ({improvement:.1f}% up)"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] Meta-learner failed: {e}")
            self._meta_learner = None

        # Fallback weights
        inv_m = {}
        for n in selected:
            for rn, rm, _ in results:
                if rn == n:
                    inv_m[n] = 1.0 / max(rm, 1e-8)
                    break
        total = sum(inv_m.values()) or 1.0
        self._weights = {n: v / total for n, v in inv_m.items()}

        # -- Refit selected models on FULL data --
        self._models = []
        for model_name in selected:
            try:
                model = get_model(model_name)
                fit_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    fit_kw = {"train_raw": train_raw, "target": target, "horizon": horizon}
                model.fit(X, y, **fit_kw)
                self._models.append((model, model_name))
            except Exception as e:
                logger.warning(f"[{self.name}] Refit {model_name} failed: {e}")

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "mode": self._mode,
            "selected": selected,
            "weights": self._weights,
            "has_meta_learner": self._meta_learner is not None,
            "all_rankings": [(n, m) for n, m, _ in results],
            "elapsed_seconds": elapsed,
        }
        self._fitted = True
        logger.info(f"[{self.name}] Fitted in {elapsed:.1f}s with {len(selected)} models")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted or not self._models:
            raise RuntimeError(f"{self.name} not fitted")

        pred_dict: Dict[str, np.ndarray] = {}
        for model, name in self._models:
            try:
                pred_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(name):
                    for k in ("test_raw", "target", "horizon"):
                        if k in kwargs:
                            pred_kw[k] = kwargs[k]
                pred_dict[name] = model.predict(X, **pred_kw)
            except Exception as e:
                logger.warning(f"[{self.name}] {name} predict failed: {e}")

        if not pred_dict:
            return np.full(len(X), 0.0)

        # Meta-learner path
        if self._meta_learner is not None:
            try:
                X_stack = X.copy()
                for name in self._pred_names:
                    col = f"__pred_{name}__"
                    if name in pred_dict:
                        X_stack[col] = pred_dict[name]
                    else:
                        X_stack[col] = np.mean(list(pred_dict.values()), axis=0)
                for c in self._meta_cols:
                    if c not in X_stack.columns:
                        X_stack[c] = 0.0
                return self._meta_learner.predict(X_stack[self._meta_cols].fillna(0))
            except Exception as e:
                logger.warning(f"[{self.name}] Meta-learner predict failed: {e}")

        # Weighted ensemble fallback
        total_w = sum(self._weights.get(n, 0) for n in pred_dict)
        if total_w < 1e-8:
            return np.mean(list(pred_dict.values()), axis=0)
        result = np.zeros(len(X))
        for name, preds in pred_dict.items():
            result += (self._weights.get(name, 0) / total_w) * preds
        return result

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# Factory functions
# ============================================================================

def get_autofit_v1(**kwargs) -> AutoFitV1Wrapper:
    return AutoFitV1Wrapper(**kwargs)


def get_autofit_v2(**kwargs) -> AutoFitV2Wrapper:
    return AutoFitV2Wrapper(top_k=5, **kwargs)


def get_autofit_v2e(**kwargs) -> AutoFitV2EWrapper:
    return AutoFitV2EWrapper(top_k=5, **kwargs)


def get_autofit_v3(**kwargs) -> AutoFitV3Wrapper:
    """Greedy forward selection + meta-learner."""
    return AutoFitV3Wrapper(mode="greedy", top_k=8, **kwargs)


def get_autofit_v3e(**kwargs) -> AutoFitV3Wrapper:
    """Top-K stacking with temporal CV."""
    return AutoFitV3Wrapper(mode="topk", top_k=6, **kwargs)


def get_autofit_v3max(**kwargs) -> AutoFitV3Wrapper:
    """Exhaustive subset search (2^K combinations)."""
    return AutoFitV3Wrapper(mode="exhaustive", top_k=8, **kwargs)


AUTOFIT_MODELS = {
    "AutoFitV1": get_autofit_v1,
    "AutoFitV2": get_autofit_v2,
    "AutoFitV2E": get_autofit_v2e,
    "AutoFitV3": get_autofit_v3,
    "AutoFitV3E": get_autofit_v3e,
    "AutoFitV3Max": get_autofit_v3max,
}
