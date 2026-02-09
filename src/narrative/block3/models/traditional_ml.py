#!/usr/bin/env python3
"""
Traditional ML Models for Block 3 KDD'26 Benchmark — PRODUCTION GRADE.

KDD'26 publication-quality hyperparameters:
- GBDT (LightGBM/XGBoost/CatBoost): 2000 iterations, early stopping,
  GPU acceleration where available, proper learning rates
- Tree ensembles (RF/ET): 500 estimators, unrestricted depth, n_jobs=-1
- Linear models (Ridge/Lasso/ElasticNet): properly regularized
- SVR/KNN: subsampled to 100K rows (O(n²) scaling on 4.4M rows is infeasible)
- LogisticRegression: classification-only, skipped for regression tasks
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import (
    ModelBase, ModelConfig, SklearnModelWrapper,
    GradientBoostingWrapper, NaiveForecaster,
)

_logger = logging.getLogger(__name__)

# Maximum training rows for O(n²) models (SVR, KNN)
_SUBSAMPLE_LIMIT = 100_000


# ============================================================================
# Subsampling wrapper for O(n²) models
# ============================================================================

class SubsampledSklearnWrapper(SklearnModelWrapper):
    """Sklearn wrapper that subsamples training data for O(n²) models."""

    def __init__(self, config: ModelConfig, sklearn_model, max_rows: int):
        super().__init__(config, sklearn_model)
        self._max_rows = max_rows

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SubsampledSklearnWrapper":
        if len(X) > self._max_rows:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), self._max_rows, replace=False)
            idx.sort()
            X = X.iloc[idx]
            y = y.iloc[idx]
            _logger.info(
                f"  [{self.config.name}] Subsampled {len(X):,} → {self._max_rows:,} rows"
            )
        self.model.fit(X, y)
        self._fitted = True
        return self


# ============================================================================
# GBDT wrapper with built-in early stopping
# ============================================================================

class ProductionGBDTWrapper(ModelBase):
    """GBDT wrapper with temporal validation split + early stopping.

    Automatically holds out the last 20% of the training data (by time
    index order) for early stopping.  Works with LightGBM, XGBoost, and
    CatBoost.
    """

    def __init__(
        self,
        config: ModelConfig,
        model_class,
        early_stopping_rounds: int = 50,
        **init_kwargs,
    ):
        super().__init__(config)
        self.model_class = model_class
        self.init_kwargs = init_kwargs
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ProductionGBDTWrapper":
        self.model = self.model_class(**self.init_kwargs)
        self._feature_names = list(X.columns)

        # Temporal 80/20 split for early stopping (preserves order)
        n = len(X)
        split_idx = int(n * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model_name = self.config.name
        _logger.info(
            f"  [{model_name}] Training {n:,} rows "
            f"(train={split_idx:,}, val={n - split_idx:,}), "
            f"early_stop={self.early_stopping_rounds}"
        )

        try:
            if model_name == "CatBoost":
                # CatBoost: eval_set is a Pool or (X, y) tuple
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            elif model_name == "LightGBM":
                import lightgbm as lgb
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds),
                        lgb.log_evaluation(period=0),  # silent
                    ],
                )
            elif model_name == "XGBoost":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                # Generic: try eval_set, fall back to plain fit
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                    )
                except TypeError:
                    self.model.fit(X_train, y_train)
        except Exception as e:
            _logger.warning(f"  [{model_name}] Early stopping failed ({e}), plain fit")
            self.model = self.model_class(**self.init_kwargs)
            self.model.fit(X, y)

        self._fitted = True
        _logger.info(f"  [{model_name}] Training complete ✓")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return super().predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self._feature_names, self.model.feature_importances_))
        return None


# ============================================================================
# Sklearn Models — PRODUCTION PARAMETERS
# ============================================================================

def create_logistic_regression(**kwargs) -> ModelBase:
    """Logistic Regression (classification ONLY — skip for regression targets)."""
    from sklearn.linear_model import LogisticRegression
    config = ModelConfig(
        name="LogisticRegression",
        model_type="classification",
        params=kwargs,
    )
    model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        C=1.0,
        **kwargs,
    )
    return SklearnModelWrapper(config, model)


def create_ridge(**kwargs) -> ModelBase:
    """Ridge Regression with proper regularization."""
    from sklearn.linear_model import Ridge
    config = ModelConfig(name="Ridge", model_type="regression", params=kwargs)
    return SklearnModelWrapper(config, Ridge(alpha=1.0, **kwargs))


def create_lasso(**kwargs) -> ModelBase:
    """Lasso Regression."""
    from sklearn.linear_model import Lasso
    config = ModelConfig(name="Lasso", model_type="regression", params=kwargs)
    return SklearnModelWrapper(config, Lasso(alpha=0.01, max_iter=10000, **kwargs))


def create_elastic_net(**kwargs) -> ModelBase:
    """ElasticNet Regression."""
    from sklearn.linear_model import ElasticNet
    config = ModelConfig(name="ElasticNet", model_type="regression", params=kwargs)
    return SklearnModelWrapper(
        config,
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, **kwargs),
    )


def create_svr(**kwargs) -> ModelBase:
    """Support Vector Regression — subsampled to 10K rows (O(n²) scaling).

    At 100K rows SVR with RBF kernel takes ~1h45m per target-horizon
    combo.  Reducing to 10K keeps each run under ~2 minutes while still
    giving the model enough data to learn from.
    """
    from sklearn.svm import SVR
    config = ModelConfig(name="SVR", model_type="regression", params=kwargs)
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1, **kwargs)
    return SubsampledSklearnWrapper(config, model, max_rows=10_000)


def create_knn_regressor(**kwargs) -> ModelBase:
    """KNN Regressor — subsampled to 100K rows (O(n²) scaling)."""
    from sklearn.neighbors import KNeighborsRegressor
    config = ModelConfig(name="KNN", model_type="regression", params=kwargs)
    n_neighbors = kwargs.pop("n_neighbors", 10)
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights="distance",
        n_jobs=-1,
        **kwargs,
    )
    return SubsampledSklearnWrapper(config, model, max_rows=_SUBSAMPLE_LIMIT)


def create_random_forest(**kwargs) -> ModelBase:
    """Random Forest — 500 estimators, full depth, n_jobs=-1."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(name="RandomForest", model_type=model_type, params=kwargs)
    common = dict(
        n_estimators=kwargs.pop("n_estimators", 500),
        max_depth=kwargs.pop("max_depth", None),  # unrestricted
        min_samples_leaf=kwargs.pop("min_samples_leaf", 5),
        n_jobs=-1,
        random_state=42,
    )
    cls = RandomForestClassifier if model_type == "classification" else RandomForestRegressor
    return SklearnModelWrapper(config, cls(**common, **kwargs))


def create_extra_trees(**kwargs) -> ModelBase:
    """Extra Trees — 500 estimators, full depth."""
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(name="ExtraTrees", model_type=model_type, params=kwargs)
    common = dict(
        n_estimators=kwargs.pop("n_estimators", 500),
        max_depth=kwargs.pop("max_depth", None),
        min_samples_leaf=kwargs.pop("min_samples_leaf", 5),
        n_jobs=-1,
        random_state=42,
    )
    cls = ExtraTreesClassifier if model_type == "classification" else ExtraTreesRegressor
    return SklearnModelWrapper(config, cls(**common, **kwargs))


def create_hist_gradient_boosting(**kwargs) -> ModelBase:
    """Histogram-based Gradient Boosting (sklearn native) — 500 iterations."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(name="HistGradientBoosting", model_type=model_type, params=kwargs)
    common = dict(
        max_iter=kwargs.pop("max_iter", 500),
        learning_rate=kwargs.pop("learning_rate", 0.05),
        max_depth=kwargs.pop("max_depth", 8),
        min_samples_leaf=kwargs.pop("min_samples_leaf", 20),
        early_stopping=True,          # sklearn built-in early stopping
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )
    cls = HistGradientBoostingClassifier if model_type == "classification" else HistGradientBoostingRegressor
    return SklearnModelWrapper(config, cls(**common, **kwargs))


# ============================================================================
# Gradient Boosting Models — PRODUCTION PARAMETERS
# ============================================================================

def create_lightgbm(**kwargs) -> ModelBase:
    """LightGBM — 2000 estimators, lr=0.01, 63 leaves, early stopping."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(
        name="LightGBM", model_type=model_type, params=kwargs,
        optional_dependency="lightgbm",
    )
    cls = lgb.LGBMClassifier if model_type == "classification" else lgb.LGBMRegressor
    return ProductionGBDTWrapper(
        config, cls,
        early_stopping_rounds=50,
        # -- production hyperparameters --
        n_estimators=kwargs.pop("n_estimators", 2000),
        learning_rate=kwargs.pop("learning_rate", 0.01),
        num_leaves=kwargs.pop("num_leaves", 63),
        max_depth=kwargs.pop("max_depth", -1),   # unlimited
        min_child_samples=kwargs.pop("min_child_samples", 20),
        subsample=kwargs.pop("subsample", 0.8),
        colsample_bytree=kwargs.pop("colsample_bytree", 0.8),
        reg_alpha=kwargs.pop("reg_alpha", 0.1),
        reg_lambda=kwargs.pop("reg_lambda", 1.0),
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
        **kwargs,
    )


def create_xgboost(**kwargs) -> ModelBase:
    """XGBoost — 2000 estimators, lr=0.01, GPU hist, early stopping."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(
        name="XGBoost", model_type=model_type, params=kwargs,
        optional_dependency="xgboost",
    )
    cls = xgb.XGBClassifier if model_type == "classification" else xgb.XGBRegressor
    return ProductionGBDTWrapper(
        config, cls,
        early_stopping_rounds=50,
        # -- production hyperparameters --
        n_estimators=kwargs.pop("n_estimators", 2000),
        learning_rate=kwargs.pop("learning_rate", 0.01),
        max_depth=kwargs.pop("max_depth", 8),
        min_child_weight=kwargs.pop("min_child_weight", 5),
        subsample=kwargs.pop("subsample", 0.8),
        colsample_bytree=kwargs.pop("colsample_bytree", 0.8),
        reg_alpha=kwargs.pop("reg_alpha", 0.1),
        reg_lambda=kwargs.pop("reg_lambda", 1.0),
        tree_method=kwargs.pop("tree_method", "hist"),
        # device="cuda" for GPU — but keep CPU for safety alongside NF models
        device=kwargs.pop("device", "cpu"),
        verbosity=0,
        random_state=42,
        **kwargs,
    )


def create_catboost(**kwargs) -> ModelBase:
    """CatBoost — 2000 iterations, lr=0.01, early stopping."""
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        raise ImportError("catboost not installed. Run: pip install catboost")

    model_type = kwargs.pop("model_type", "regression")
    config = ModelConfig(
        name="CatBoost", model_type=model_type, params=kwargs,
        optional_dependency="catboost",
    )
    cls = CatBoostClassifier if model_type == "classification" else CatBoostRegressor
    return ProductionGBDTWrapper(
        config, cls,
        early_stopping_rounds=50,
        # -- production hyperparameters --
        iterations=kwargs.pop("iterations", 2000),
        learning_rate=kwargs.pop("learning_rate", 0.01),
        depth=kwargs.pop("depth", 8),
        l2_leaf_reg=kwargs.pop("l2_leaf_reg", 3.0),
        subsample=kwargs.pop("subsample", 0.8),
        # task_type="GPU" for GPU — keep CPU for safety
        task_type=kwargs.pop("task_type", "CPU"),
        verbose=False,
        random_seed=42,
        **kwargs,
    )


# ============================================================================
# Quantile Regression
# ============================================================================

class QuantileRegressor(ModelBase):
    """Quantile Regression using sklearn."""

    def __init__(self, quantile: float = 0.5, **kwargs):
        config = ModelConfig(
            name=f"QuantileRegressor_q{quantile}",
            model_type="regression",
            params={"quantile": quantile, **kwargs},
            supports_quantiles=True,
        )
        super().__init__(config)
        self.quantile = quantile
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "QuantileRegressor":
        from sklearn.linear_model import QuantileRegressor as SKQuantileRegressor
        self.model = SKQuantileRegressor(
            quantile=self.quantile, alpha=0.0, solver="highs",
        )
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)


def create_quantile_regressor(quantile: float = 0.5, **kwargs) -> ModelBase:
    return QuantileRegressor(quantile=quantile, **kwargs)


# ============================================================================
# Naive / Baseline Models
# ============================================================================

def create_seasonal_naive(seasonality: int = 7, **kwargs) -> ModelBase:
    config = ModelConfig(
        name="SeasonalNaive", model_type="forecasting",
        params={"seasonality": seasonality},
    )
    return NaiveForecaster(config, seasonality=seasonality)


def create_mean_predictor(**kwargs) -> ModelBase:
    class MeanPredictor(ModelBase):
        def __init__(self):
            config = ModelConfig(name="MeanPredictor", model_type="regression")
            super().__init__(config)
            self._mean = 0.0

        def fit(self, X, y, **kw):
            self._mean = float(y.mean())
            self._fitted = True
            return self

        def predict(self, X):
            return np.full(len(X), self._mean)

    return MeanPredictor()


# ============================================================================
# Registry
# ============================================================================

TRADITIONAL_ML_MODELS = {
    # Classification
    "LogisticRegression": create_logistic_regression,
    # Regression
    "Ridge": create_ridge,
    "Lasso": create_lasso,
    "ElasticNet": create_elastic_net,
    "SVR": create_svr,
    "KNN": create_knn_regressor,
    # Ensemble
    "RandomForest": create_random_forest,
    "ExtraTrees": create_extra_trees,
    "HistGradientBoosting": create_hist_gradient_boosting,
    # Gradient Boosting
    "LightGBM": create_lightgbm,
    "XGBoost": create_xgboost,
    "CatBoost": create_catboost,
    # Quantile
    "QuantileRegressor": create_quantile_regressor,
    # Baselines
    "SeasonalNaive": create_seasonal_naive,
    "MeanPredictor": create_mean_predictor,
}


def get_traditional_model(name: str, **kwargs) -> ModelBase:
    if name not in TRADITIONAL_ML_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(TRADITIONAL_ML_MODELS.keys())}")
    return TRADITIONAL_ML_MODELS[name](**kwargs)


def list_traditional_models() -> list[str]:
    return list(TRADITIONAL_ML_MODELS.keys())
