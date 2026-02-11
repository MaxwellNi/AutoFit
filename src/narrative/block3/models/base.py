#!/usr/bin/env python3
"""
Block 3 Model Base Class.

Unified interface for all models: fit, predict, predict_quantiles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_type: str  # classification, regression, forecasting, ranking
    params: Dict[str, Any] = field(default_factory=dict)
    supports_quantiles: bool = False
    supports_probabilistic: bool = False
    supports_missing: bool = False  # Handles missing values natively
    requires_sequences: bool = False
    optional_dependency: Optional[str] = None  # Package name if optional


class ModelBase(ABC):
    """
    Abstract base class for all Block 3 models.
    
    All models must implement:
    - fit(X, y): Train the model
    - predict(X): Make point predictions
    
    Optional methods:
    - predict_quantiles(X, quantiles): Quantile predictions
    - predict_proba(X): Probability predictions (classification)
    - predict_distribution(X): Full distribution (probabilistic)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.model = None
        self._fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ModelBase":
        """
        Fit the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional arguments (e.g., sample_weights, eval_set)
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make point predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Predictions array
        """
        pass
    
    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> np.ndarray:
        """
        Make quantile predictions.
        
        Args:
            X: Feature DataFrame
            quantiles: List of quantiles to predict
        
        Returns:
            Array of shape (n_samples, n_quantiles)
        """
        if not self.config.supports_quantiles:
            # Fallback: return point prediction for all quantiles
            preds = self.predict(X)
            return np.column_stack([preds] * len(quantiles))
        
        raise NotImplementedError("Subclass must implement predict_quantiles")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Returns:
            Array of shape (n_samples, n_classes)
        """
        if self.config.model_type != "classification":
            raise ValueError("predict_proba only for classification models")
        
        # Default: use sigmoid of predictions
        preds = self.predict(X)
        probs = 1 / (1 + np.exp(-preds))
        return np.column_stack([1 - probs, probs])
    
    def predict_distribution(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Sample from predictive distribution (probabilistic models).
        
        Returns:
            Array of shape (n_samples_data, n_samples)
        """
        if not self.config.supports_probabilistic:
            # Fallback: add noise to point predictions
            preds = self.predict(X)
            noise = np.random.normal(0, np.std(preds) * 0.1, (len(X), n_samples))
            return preds[:, None] + noise
        
        raise NotImplementedError("Subclass must implement predict_distribution")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available."""
        return None
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted


class SklearnModelWrapper(ModelBase):
    """Wrapper for sklearn-compatible models."""
    
    def __init__(self, config: ModelConfig, sklearn_model):
        super().__init__(config)
        self.model = sklearn_model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SklearnModelWrapper":
        self._feature_names = list(X.columns)
        self.model.fit(X, y)
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return super().predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(
                getattr(self, "_feature_names", []),
                self.model.feature_importances_
            ))
        if hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return dict(zip(
                getattr(self, "_feature_names", []),
                np.abs(coef)
            ))
        return None


class GradientBoostingWrapper(ModelBase):
    """Wrapper for gradient boosting models (LightGBM, XGBoost, CatBoost)."""
    
    def __init__(self, config: ModelConfig, model_class, **init_kwargs):
        super().__init__(config)
        self.model_class = model_class
        self.init_kwargs = init_kwargs
        self.model = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "GradientBoostingWrapper":
        self.model = self.model_class(**self.init_kwargs)
        
        # Handle eval_set if provided
        eval_set = kwargs.pop("eval_set", None)
        
        if eval_set is not None:
            self.model.fit(X, y, eval_set=eval_set, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        
        self._fitted = True
        self._feature_names = list(X.columns)
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


class NaiveForecaster(ModelBase):
    """Simple naive/seasonal naive forecaster."""
    
    def __init__(self, config: ModelConfig, seasonality: int = 7):
        super().__init__(config)
        self.seasonality = seasonality
        self._last_values = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "NaiveForecaster":
        # Store last value(s) for naive prediction
        self._last_values = y.iloc[-self.seasonality:].values
        self._mean = y.mean()
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        n = len(X)
        if self._last_values is not None and len(self._last_values) > 0:
            # Repeat seasonal pattern with proper phase alignment.
            # self._last_values contains the last `seasonality` training values,
            # so test index 0 should map to phase offset 0 (i.e. the value that
            # would follow the last training observation).
            preds = np.tile(self._last_values, (n // self.seasonality + 1))[:n]
        else:
            preds = np.full(n, self._mean)
        return preds
