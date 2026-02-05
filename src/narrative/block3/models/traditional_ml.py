#!/usr/bin/env python3
"""
Traditional ML Models for Block 3.

Comprehensive collection of sklearn-compatible models:
- Classification: LogisticRegression, RandomForest, HistGradientBoosting, etc.
- Regression: Ridge, Lasso, ElasticNet, SVR, KNN, etc.
- Gradient Boosting: LightGBM, XGBoost, CatBoost
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import (
    ModelBase, ModelConfig, SklearnModelWrapper, GradientBoostingWrapper, NaiveForecaster
)

# ============================================================================
# Sklearn Models
# ============================================================================

def create_logistic_regression(**kwargs) -> ModelBase:
    """Logistic Regression for classification."""
    from sklearn.linear_model import LogisticRegression
    
    config = ModelConfig(
        name="LogisticRegression",
        model_type="classification",
        params=kwargs,
    )
    
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        **kwargs
    )
    
    return SklearnModelWrapper(config, model)


def create_ridge(**kwargs) -> ModelBase:
    """Ridge Regression."""
    from sklearn.linear_model import Ridge
    
    config = ModelConfig(
        name="Ridge",
        model_type="regression",
        params=kwargs,
    )
    
    return SklearnModelWrapper(config, Ridge(**kwargs))


def create_lasso(**kwargs) -> ModelBase:
    """Lasso Regression."""
    from sklearn.linear_model import Lasso
    
    config = ModelConfig(
        name="Lasso",
        model_type="regression",
        params=kwargs,
    )
    
    return SklearnModelWrapper(config, Lasso(max_iter=5000, **kwargs))


def create_elastic_net(**kwargs) -> ModelBase:
    """ElasticNet Regression."""
    from sklearn.linear_model import ElasticNet
    
    config = ModelConfig(
        name="ElasticNet",
        model_type="regression",
        params=kwargs,
    )
    
    return SklearnModelWrapper(config, ElasticNet(max_iter=5000, **kwargs))


def create_svr(**kwargs) -> ModelBase:
    """Support Vector Regression."""
    from sklearn.svm import SVR
    
    config = ModelConfig(
        name="SVR",
        model_type="regression",
        params=kwargs,
    )
    
    return SklearnModelWrapper(config, SVR(**kwargs))


def create_knn_regressor(**kwargs) -> ModelBase:
    """K-Nearest Neighbors Regressor."""
    from sklearn.neighbors import KNeighborsRegressor
    
    config = ModelConfig(
        name="KNN",
        model_type="regression",
        params=kwargs,
    )
    
    n_neighbors = kwargs.pop("n_neighbors", 5)
    return SklearnModelWrapper(config, KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs))


def create_random_forest(**kwargs) -> ModelBase:
    """Random Forest (classification or regression based on task)."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="RandomForest",
        model_type=model_type,
        params=kwargs,
    )
    
    n_estimators = kwargs.pop("n_estimators", 100)
    max_depth = kwargs.pop("max_depth", None)
    n_jobs = kwargs.pop("n_jobs", -1)
    
    if model_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            **kwargs
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            **kwargs
        )
    
    return SklearnModelWrapper(config, model)


def create_extra_trees(**kwargs) -> ModelBase:
    """Extra Trees (Extremely Randomized Trees)."""
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="ExtraTrees",
        model_type=model_type,
        params=kwargs,
    )
    
    n_estimators = kwargs.pop("n_estimators", 100)
    n_jobs = kwargs.pop("n_jobs", -1)
    
    if model_type == "classification":
        model = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=n_jobs, **kwargs)
    else:
        model = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=n_jobs, **kwargs)
    
    return SklearnModelWrapper(config, model)


def create_hist_gradient_boosting(**kwargs) -> ModelBase:
    """Histogram-based Gradient Boosting (sklearn native)."""
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="HistGradientBoosting",
        model_type=model_type,
        params=kwargs,
    )
    
    max_iter = kwargs.pop("max_iter", 100)
    learning_rate = kwargs.pop("learning_rate", 0.1)
    
    if model_type == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            **kwargs
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            **kwargs
        )
    
    return SklearnModelWrapper(config, model)


# ============================================================================
# Gradient Boosting Models (optional dependencies)
# ============================================================================

def create_lightgbm(**kwargs) -> ModelBase:
    """LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="LightGBM",
        model_type=model_type,
        params=kwargs,
        optional_dependency="lightgbm",
    )
    
    n_estimators = kwargs.pop("n_estimators", 100)
    learning_rate = kwargs.pop("learning_rate", 0.1)
    num_leaves = kwargs.pop("num_leaves", 31)
    verbosity = kwargs.pop("verbosity", -1)
    
    if model_type == "classification":
        model_class = lgb.LGBMClassifier
    else:
        model_class = lgb.LGBMRegressor
    
    return GradientBoostingWrapper(
        config,
        model_class,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        verbosity=verbosity,
        **kwargs
    )


def create_xgboost(**kwargs) -> ModelBase:
    """XGBoost model."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="XGBoost",
        model_type=model_type,
        params=kwargs,
        optional_dependency="xgboost",
    )
    
    n_estimators = kwargs.pop("n_estimators", 100)
    learning_rate = kwargs.pop("learning_rate", 0.1)
    max_depth = kwargs.pop("max_depth", 6)
    verbosity = kwargs.pop("verbosity", 0)
    
    if model_type == "classification":
        model_class = xgb.XGBClassifier
    else:
        model_class = xgb.XGBRegressor
    
    return GradientBoostingWrapper(
        config,
        model_class,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        verbosity=verbosity,
        **kwargs
    )


def create_catboost(**kwargs) -> ModelBase:
    """CatBoost model."""
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        raise ImportError("catboost not installed. Run: pip install catboost")
    
    model_type = kwargs.pop("model_type", "regression")
    
    config = ModelConfig(
        name="CatBoost",
        model_type=model_type,
        params=kwargs,
        optional_dependency="catboost",
    )
    
    iterations = kwargs.pop("iterations", 100)
    learning_rate = kwargs.pop("learning_rate", 0.1)
    depth = kwargs.pop("depth", 6)
    verbose = kwargs.pop("verbose", False)
    
    if model_type == "classification":
        model_class = CatBoostClassifier
    else:
        model_class = CatBoostRegressor
    
    return GradientBoostingWrapper(
        config,
        model_class,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        verbose=verbose,
        **kwargs
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
            quantile=self.quantile,
            alpha=0.0,  # No regularization
            solver="highs",
        )
        self.model.fit(X, y)
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def create_quantile_regressor(quantile: float = 0.5, **kwargs) -> ModelBase:
    """Create quantile regressor."""
    return QuantileRegressor(quantile=quantile, **kwargs)


# ============================================================================
# Naive/Baseline Models
# ============================================================================

def create_seasonal_naive(seasonality: int = 7, **kwargs) -> ModelBase:
    """Seasonal Naive forecaster."""
    config = ModelConfig(
        name="SeasonalNaive",
        model_type="forecasting",
        params={"seasonality": seasonality},
    )
    return NaiveForecaster(config, seasonality=seasonality)


def create_mean_predictor(**kwargs) -> ModelBase:
    """Mean predictor baseline."""
    
    class MeanPredictor(ModelBase):
        def __init__(self):
            config = ModelConfig(name="MeanPredictor", model_type="regression")
            super().__init__(config)
            self._mean = 0
        
        def fit(self, X, y, **kw):
            self._mean = y.mean()
            self._fitted = True
            return self
        
        def predict(self, X):
            return np.full(len(X), self._mean)
    
    return MeanPredictor()


# ============================================================================
# Factory Functions
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
    
    # Gradient Boosting (optional)
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
    """Get a traditional ML model by name."""
    if name not in TRADITIONAL_ML_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(TRADITIONAL_ML_MODELS.keys())}")
    
    return TRADITIONAL_ML_MODELS[name](**kwargs)


def list_traditional_models() -> list[str]:
    """List all available traditional ML models."""
    return list(TRADITIONAL_ML_MODELS.keys())
