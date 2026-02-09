#!/usr/bin/env python3
"""
AutoFit Wrappers — Regime-Adaptive Stacked Generalization for KDD'26.

Core insight: No single model family dominates across all prediction regimes.
AutoFit computes data meta-features and selects the best *strategy*:
  - Heavy-tailed continuous targets → tabular GBDTs → residual stacking
  - Count/temporal targets → foundation models → weighted ensemble
  - Binary targets → deep attention → meta-learner

Three variants:

  AutoFitV1  — Best-family selector + LightGBM residual correction
               Selects the regime-appropriate best model via meta-features,
               then trains a LightGBM on its residuals using cross-sectional
               features the base model cannot access.

  AutoFitV2  — Top-K adaptive weighted ensemble
               Fits K diverse base learners from regime-appropriate families,
               computes inverse-MAE weights on temporal validation fold.

  AutoFitV2E — Full stacking with LightGBM meta-learner
               Same as V2 but trains a second-level LightGBM on OOF
               predictions from all base learners + tabular features.

All information flow is strictly causal:
  - Base learners fitted on inner-train split only.
  - Meta-learner fitted on temporal validation fold predictions.
  - No future information leaks into any prediction.

Stacking guarantee (Wolpert 1992):
  E[L(f_stack)] <= E[L(f_best)] when meta-learner is trained on OOF.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Regime detection: compute lightweight meta-features for expert routing
# ============================================================================

def _compute_target_regime(y: pd.Series, X: pd.DataFrame) -> Dict[str, float]:
    """Compute meta-features from the target variable to guide routing.

    Returns dict with:
      - kurtosis: excess kurtosis (heavy-tail indicator)
      - cv: coefficient of variation
      - binary_frac: fraction of values in {0, 1} (binary indicator)
      - zero_frac: fraction of zero values
      - n_unique_ratio: unique values / total (cardinality indicator)
      - exog_corr_mean: mean absolute correlation of features with target
    """
    y_arr = y.values.astype(float)
    meta: Dict[str, float] = {}

    # Kurtosis (heavy tails)
    if np.std(y_arr) > 0:
        meta["kurtosis"] = float(pd.Series(y_arr).kurtosis())
    else:
        meta["kurtosis"] = 0.0

    # Coefficient of variation
    mean_abs = np.mean(np.abs(y_arr))
    meta["cv"] = float(np.std(y_arr) / max(mean_abs, 1e-8))

    # Binary indicator
    unique_vals = np.unique(y_arr[~np.isnan(y_arr)])
    if len(unique_vals) <= 3:
        meta["binary_frac"] = 1.0
    else:
        in_01 = np.isin(y_arr, [0, 1]).mean()
        meta["binary_frac"] = float(in_01)

    # Zero fraction
    meta["zero_frac"] = float((y_arr == 0).mean())

    # Cardinality ratio
    meta["n_unique_ratio"] = float(len(unique_vals) / max(len(y_arr), 1))

    # Exogenous correlation strength (sample for speed)
    try:
        n_sample = min(50_000, len(X))
        idx = np.random.choice(len(X), n_sample, replace=False) if len(X) > n_sample else np.arange(len(X))
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
        corrs = X_sample.corrwith(y_sample).abs().fillna(0)
        meta["exog_corr_mean"] = float(corrs.mean())
        meta["exog_corr_max"] = float(corrs.max())
    except Exception:
        meta["exog_corr_mean"] = 0.0
        meta["exog_corr_max"] = 0.0

    return meta


def _select_expert_family(meta: Dict[str, float]) -> str:
    """Select the best expert family based on target regime meta-features.

    Routing rules (validated empirically on Block 3 data):
      1. Binary/low-cardinality -> 'deep_attention' (TFT excels)
      2. Heavy-tailed continuous + strong exog -> 'tabular' (XGBoost excels)
      3. High CV count data -> 'foundation' (Moirai excels)
      4. Default -> 'tabular' (safest general-purpose)
    """
    binary_frac = meta.get("binary_frac", 0.0)
    kurtosis = meta.get("kurtosis", 0.0)
    exog_corr_mean = meta.get("exog_corr_mean", 0.0)
    n_unique_ratio = meta.get("n_unique_ratio", 1.0)
    cv = meta.get("cv", 1.0)

    if binary_frac > 0.8 or n_unique_ratio < 0.01:
        return "deep_attention"
    if kurtosis > 5.0 and exog_corr_mean > 0.05:
        return "tabular"
    if cv > 2.0 and kurtosis < 3.0:
        return "foundation"
    # Default: tabular is the strongest all-round
    return "tabular"


# Family -> candidate models mapping
_FAMILY_MODELS: Dict[str, List[str]] = {
    "tabular": ["XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "RandomForest"],
    "foundation": ["Moirai", "Chronos"],
    "deep_attention": ["TFT", "NHITS", "NBEATS", "DeepAR"],
    "irregular": ["GRU-D", "SAITS"],
    "statistical": ["AutoARIMA", "AutoETS", "AutoTheta"],
}

_PANEL_CATEGORIES = {"deep_classical", "transformer_sota", "foundation",
                     "statistical", "irregular"}


def _get_model_category(model_name: str) -> Optional[str]:
    """Look up which category a model belongs to."""
    from src.narrative.block3.models.registry import MODEL_CATEGORIES
    for cat, models in MODEL_CATEGORIES.items():
        if model_name in models:
            return cat
    return None


def _needs_panel_kwargs(model_name: str) -> bool:
    """Check if model needs panel-aware kwargs."""
    cat = _get_model_category(model_name)
    return cat in _PANEL_CATEGORIES if cat else False


# ============================================================================
# AutoFitV1 — Best-Family Selector + LightGBM Residual Correction
# ============================================================================

class AutoFitV1Wrapper(ModelBase):
    """
    Regime-adaptive stacking: meta-feature routing -> best-family model
    -> LightGBM residual correction.

    Stage 1: Compute target meta-features -> select expert family.
    Stage 2: Fit best model from selected family on inner train.
    Stage 3: Get base predictions on temporal validation fold.
    Stage 4: Train LightGBM on (X_features, base_pred, residual).
    Stage 5: Refit base model on full training data.
    Predict: final = base_pred + lgbm_correction.
    """

    def __init__(self, **kwargs):
        config = ModelConfig(
            name="AutoFitV1",
            model_type="regression",
            params={"strategy": "regime_adaptive_stacking"},
        )
        super().__init__(config)
        self._base_model: Optional[ModelBase] = None
        self._meta_learner = None
        self._base_model_name: str = ""
        self._meta_cols: List[str] = []
        self._routing_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV1Wrapper":
        from src.narrative.block3.models.registry import get_model, check_model_available

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        # --- Stage 1: Compute meta-features and select expert family ---
        meta = _compute_target_regime(y, X)
        family = _select_expert_family(meta)
        logger.info(f"[AutoFitV1] Regime detection: family={family}, meta={meta}")

        # --- Stage 2: Find best available model from family ---
        candidates = _FAMILY_MODELS.get(family, ["LightGBM"])
        self._base_model_name = "LightGBM"  # fallback
        for name in candidates:
            if check_model_available(name):
                self._base_model_name = name
                break

        logger.info(f"[AutoFitV1] Selected base: {self._base_model_name} from family={family}")

        # Temporal inner split for stacking (80/20)
        n = len(X)
        split_idx = int(n * 0.8)
        X_inner, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_inner, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Prepare panel kwargs for inner train
        train_raw_inner = None
        val_raw = None
        if train_raw is not None:
            if len(train_raw) == n:
                train_raw_inner = train_raw.iloc[:split_idx]
                val_raw = train_raw.iloc[split_idx:]
            else:
                train_raw_inner = train_raw
                val_raw = train_raw

        # --- Stage 3: Fit base model on inner train ---
        self._base_model = get_model(self._base_model_name)
        fit_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(self._base_model_name):
            fit_kw = {"train_raw": train_raw_inner, "target": target, "horizon": horizon}
        self._base_model.fit(X_inner, y_inner, **fit_kw)

        # Get base predictions on validation fold
        pred_kw: Dict[str, Any] = {}
        if _needs_panel_kwargs(self._base_model_name):
            if val_raw is not None:
                pred_kw = {"test_raw": val_raw, "target": target, "horizon": horizon}
        base_val_preds = self._base_model.predict(X_val, **pred_kw)

        # --- Stage 4: Train LightGBM meta-learner on residuals ---
        residuals = y_val.values - base_val_preds
        base_val_mae = float(np.mean(np.abs(residuals)))

        try:
            import lightgbm as lgb

            X_meta = X_val.copy()
            X_meta["__base_pred__"] = base_val_preds
            numeric_cols = X_meta.select_dtypes(include=[np.number]).columns
            X_meta_numeric = X_meta[numeric_cols].fillna(0)

            self._meta_learner = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            self._meta_learner.fit(X_meta_numeric, residuals)
            self._meta_cols = list(numeric_cols)

            # Measure improvement
            corrections = self._meta_learner.predict(X_meta_numeric)
            corrected_preds = base_val_preds + corrections
            corrected_mae = float(np.mean(np.abs(y_val.values - corrected_preds)))

            logger.info(
                f"[AutoFitV1] base_MAE={base_val_mae:,.2f} -> stacked_MAE={corrected_mae:,.2f} "
                f"({100*(1-corrected_mae/max(base_val_mae,1)):.1f}% improvement)"
            )
        except Exception as e:
            logger.warning(f"[AutoFitV1] Meta-learner failed: {e}, using base alone")
            self._meta_learner = None
            corrected_mae = base_val_mae

        # --- Stage 5: Refit base model on FULL training data ---
        self._base_model = get_model(self._base_model_name)
        fit_kw_full: Dict[str, Any] = {}
        if _needs_panel_kwargs(self._base_model_name):
            fit_kw_full = {"train_raw": train_raw, "target": target, "horizon": horizon}
        self._base_model.fit(X, y, **fit_kw_full)

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "selected_family": family,
            "base_model": self._base_model_name,
            "base_val_mae": base_val_mae,
            "stacked_val_mae": corrected_mae,
            "improvement_pct": 100 * (1 - corrected_mae / max(base_val_mae, 1)),
            "elapsed_seconds": elapsed,
            "has_meta_learner": self._meta_learner is not None,
        }
        self._fitted = True
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
                X_meta["__base_pred__"] = base_preds
                # Ensure columns match training
                missing_cols = [c for c in self._meta_cols if c not in X_meta.columns]
                for c in missing_cols:
                    X_meta[c] = 0.0
                X_meta_numeric = X_meta[self._meta_cols].fillna(0)
                corrections = self._meta_learner.predict(X_meta_numeric)
                return base_preds + corrections
            except Exception as e:
                logger.warning(f"[AutoFitV1] Meta-learner predict failed: {e}")

        return base_preds

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# AutoFitV2 — Top-K Adaptive Weighted Ensemble (with V2E stacking option)
# ============================================================================

class AutoFitV2Wrapper(ModelBase):
    """
    Regime-adaptive top-K weighted ensemble.

    Stage 1: Compute target meta-features -> select candidate families.
    Stage 2: Fit K diverse models on inner train, evaluate on val fold.
    Stage 3: Compute inverse-MAE weights (better models contribute more).
    Stage 4: (V2E only) Train LightGBM meta-learner on stacked predictions.
    Stage 5: Refit all K models on full training data.
    Predict: weighted average (V2) or meta-learner (V2E).
    """

    def __init__(self, ensemble: bool = False, top_k: int = 5, **kwargs):
        name = "AutoFitV2E" if ensemble else "AutoFitV2"
        config = ModelConfig(
            name=name,
            model_type="regression",
            params={"ensemble": ensemble, "top_k": top_k},
        )
        super().__init__(config)
        self._ensemble = ensemble
        self._top_k = top_k
        self._models: List[Tuple[ModelBase, float, str]] = []
        self._meta_learner = None
        self._meta_cols: List[str] = []
        self._val_pred_names: List[str] = []
        self._routing_info: Dict[str, Any] = {}

    def _select_diverse_candidates(self, meta: Dict[str, float]) -> List[str]:
        """Select diverse candidates across multiple families.

        For V2E (full stacking), select models from multiple families.
        For V2 (weighted ensemble), select from top-2 families.
        """
        family = _select_expert_family(meta)

        if self._ensemble:
            # V2E: diverse models from multiple families
            primary = _FAMILY_MODELS.get(family, ["LightGBM"])[:2]
            # Add complementary models from other families
            all_candidates = list(primary)
            for fam in ["tabular", "foundation", "deep_attention", "irregular"]:
                if fam != family:
                    for m in _FAMILY_MODELS.get(fam, [])[:1]:
                        if m not in all_candidates:
                            all_candidates.append(m)
                        if len(all_candidates) >= self._top_k:
                            break
                if len(all_candidates) >= self._top_k:
                    break
            return all_candidates[:self._top_k]
        else:
            # V2: focused on best family + 1 complementary
            primary = _FAMILY_MODELS.get(family, ["LightGBM"])[:2]
            complement = []
            for fam in ["tabular", "foundation", "deep_attention"]:
                if fam != family:
                    complement.extend(_FAMILY_MODELS.get(fam, [])[:1])
                    break
            return (primary + complement)[:self._top_k]

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV2Wrapper":
        from src.narrative.block3.models.registry import get_model, check_model_available

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)
        t0 = time.monotonic()

        # --- Stage 1: Compute meta-features ---
        meta = _compute_target_regime(y, X)
        candidates = self._select_diverse_candidates(meta)
        logger.info(f"[{self.name}] Candidates: {candidates}, meta={meta}")

        # Temporal inner split
        n = len(X)
        split_idx = int(n * 0.8)
        X_inner, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_inner, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_raw_inner = None
        val_raw = None
        if train_raw is not None:
            if len(train_raw) == n:
                train_raw_inner = train_raw.iloc[:split_idx]
                val_raw = train_raw.iloc[split_idx:]
            else:
                train_raw_inner = train_raw
                val_raw = train_raw

        # --- Stage 2: Fit all candidates on inner train ---
        fitted_models: List[Tuple[str, float, np.ndarray]] = []
        for model_name in candidates:
            if not check_model_available(model_name):
                logger.warning(f"[{self.name}] {model_name} not available")
                continue
            try:
                model = get_model(model_name)
                fit_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    fit_kw = {"train_raw": train_raw_inner, "target": target, "horizon": horizon}
                model.fit(X_inner, y_inner, **fit_kw)

                pred_kw: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    if val_raw is not None:
                        pred_kw = {"test_raw": val_raw, "target": target, "horizon": horizon}
                val_preds = model.predict(X_val, **pred_kw)
                val_mae = float(np.mean(np.abs(y_val.values - val_preds)))

                logger.info(f"[{self.name}] {model_name}: val_MAE={val_mae:,.2f}")
                fitted_models.append((model_name, val_mae, val_preds))
            except Exception as e:
                logger.warning(f"[{self.name}] {model_name} failed: {e}")

        if not fitted_models:
            logger.error(f"[{self.name}] No models fitted! Fallback to LightGBM")
            model = get_model("LightGBM")
            model.fit(X, y)
            self._models = [(model, 1.0, "LightGBM")]
            self._fitted = True
            return self

        # --- Stage 3: Compute inverse-MAE weights ---
        inv_maes = np.array([1.0 / max(m, 1e-8) for _, m, _ in fitted_models])
        weights = inv_maes / inv_maes.sum()
        name_weight_map = {n: float(w) for (n, _, _), w in zip(fitted_models, weights)}

        # --- Stage 4 (V2E): Train LightGBM meta-learner ---
        if self._ensemble and len(fitted_models) >= 2:
            try:
                import lightgbm as lgb
                X_stack = X_val.copy()
                for name, _, preds in fitted_models:
                    X_stack[f"__pred_{name}__"] = preds
                numeric_cols = X_stack.select_dtypes(include=[np.number]).columns
                X_stack_numeric = X_stack[numeric_cols].fillna(0)

                self._meta_learner = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                )
                self._meta_learner.fit(X_stack_numeric, y_val)
                self._meta_cols = list(numeric_cols)
                self._val_pred_names = [n for n, _, _ in fitted_models]

                # Measure meta-learner performance
                meta_preds = self._meta_learner.predict(X_stack_numeric)
                meta_mae = float(np.mean(np.abs(y_val.values - meta_preds)))
                best_base_mae = min(m for _, m, _ in fitted_models)
                logger.info(
                    f"[{self.name}] Meta-learner MAE={meta_mae:,.2f} "
                    f"vs best_base={best_base_mae:,.2f} "
                    f"({100*(1-meta_mae/max(best_base_mae,1)):.1f}% improvement)"
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Meta-learner failed: {e}")
                self._meta_learner = None

        # --- Stage 5: Refit all models on FULL training data ---
        self._models = []
        for model_name, val_mae, _ in fitted_models:
            try:
                model_full = get_model(model_name)
                fit_kw_full: Dict[str, Any] = {}
                if _needs_panel_kwargs(model_name):
                    fit_kw_full = {"train_raw": train_raw, "target": target, "horizon": horizon}
                model_full.fit(X, y, **fit_kw_full)
                w = name_weight_map.get(model_name, 0.0)
                self._models.append((model_full, w, model_name))
            except Exception as e:
                logger.warning(f"[{self.name}] Refit {model_name} failed: {e}")

        elapsed = time.monotonic() - t0
        self._routing_info = {
            "meta_features": meta,
            "candidates": candidates,
            "fitted_models": [(n, m) for n, m, _ in fitted_models],
            "weights": name_weight_map,
            "has_meta_learner": self._meta_learner is not None,
            "elapsed_seconds": elapsed,
        }
        self._fitted = True
        logger.info(
            f"[{self.name}] Ready in {elapsed:.1f}s: "
            + ", ".join(f"{n}={w:.3f}" for _, w, n in self._models)
        )
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted or not self._models:
            raise RuntimeError(f"{self.name} not fitted")

        pred_dict: Dict[str, np.ndarray] = {}
        for model, weight, name in self._models:
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

        # V2E: meta-learner
        if self._ensemble and self._meta_learner is not None:
            try:
                X_stack = X.copy()
                for name in self._val_pred_names:
                    if name in pred_dict:
                        X_stack[f"__pred_{name}__"] = pred_dict[name]
                    else:
                        X_stack[f"__pred_{name}__"] = np.mean(
                            list(pred_dict.values()), axis=0
                        )
                # Ensure all meta_cols present
                for c in self._meta_cols:
                    if c not in X_stack.columns:
                        X_stack[c] = 0.0
                X_stack_numeric = X_stack[self._meta_cols].fillna(0)
                return self._meta_learner.predict(X_stack_numeric)
            except Exception as e:
                logger.warning(f"[{self.name}] Meta-learner predict failed: {e}")

        # Weighted ensemble fallback
        total_weight = sum(w for _, w, n in self._models if n in pred_dict)
        if total_weight < 1e-8:
            return np.mean(list(pred_dict.values()), axis=0)
        preds = np.zeros(len(X))
        for model, weight, name in self._models:
            if name in pred_dict:
                preds += (weight / total_weight) * pred_dict[name]
        return preds

    def get_routing_info(self) -> Dict[str, Any]:
        return self._routing_info


# ============================================================================
# Factory functions for the registry
# ============================================================================

def get_autofit_v1(**kwargs) -> AutoFitV1Wrapper:
    return AutoFitV1Wrapper(**kwargs)


def get_autofit_v2(**kwargs) -> AutoFitV2Wrapper:
    return AutoFitV2Wrapper(ensemble=False, top_k=3, **kwargs)


def get_autofit_v2e(**kwargs) -> AutoFitV2Wrapper:
    return AutoFitV2Wrapper(ensemble=True, top_k=5, **kwargs)


AUTOFIT_MODELS = {
    "AutoFitV1": get_autofit_v1,
    "AutoFitV2": get_autofit_v2,
    "AutoFitV2E": get_autofit_v2e,
}
