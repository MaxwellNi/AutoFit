#!/usr/bin/env python3
"""
Statistical Forecasting Models for Block 3.

Uses Nixtla's StatsForecast for speed and completeness:
- AutoARIMA, ETS, Theta, TBATS, MSTL, SeasonalNaive, etc.

Optional dependency: statsforecast
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig


class StatsForecastWrapper(ModelBase):
    """
    Wrapper for StatsForecast models.
    
    Converts panel data to StatsForecast format and back.
    """
    
    def __init__(self, config: ModelConfig, model_name: str, **model_kwargs):
        super().__init__(config)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self._sf = None
        self._fitted_model = None
    
    def _check_dependency(self):
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import (
                AutoARIMA, ETS, Theta, MSTL, SeasonalNaive,
                Naive, HistoricAverage, WindowAverage
            )
            return True
        except ImportError:
            return False
    
    def _get_model_class(self):
        from statsforecast.models import (
            AutoARIMA, ETS, Theta, MSTL, SeasonalNaive,
            Naive, HistoricAverage, WindowAverage
        )
        
        models = {
            "AutoARIMA": AutoARIMA,
            "ETS": ETS,
            "Theta": Theta,
            "MSTL": MSTL,
            "SeasonalNaive": SeasonalNaive,
            "Naive": Naive,
            "HistoricAverage": HistoricAverage,
            "WindowAverage": WindowAverage,
        }
        
        if self.model_name not in models:
            raise ValueError(f"Unknown StatsForecast model: {self.model_name}")
        
        return models[self.model_name]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StatsForecastWrapper":
        """
        Fit using StatsForecast.
        
        Expects X to have 'unique_id' and 'ds' columns, or will create them.
        """
        if not self._check_dependency():
            raise ImportError("statsforecast not installed. Run: pip install statsforecast")
        
        from statsforecast import StatsForecast
        
        # Prepare data in StatsForecast format
        df = pd.DataFrame({
            "unique_id": kwargs.get("unique_id", ["series_0"] * len(y)),
            "ds": kwargs.get("ds", pd.date_range(start="2020-01-01", periods=len(y), freq="D")),
            "y": y.values,
        })
        
        # Create model
        model_class = self._get_model_class()
        
        # Handle model-specific parameters
        if self.model_name == "SeasonalNaive":
            model = model_class(season_length=self.model_kwargs.get("season_length", 7))
        elif self.model_name == "MSTL":
            model = model_class(season_length=self.model_kwargs.get("season_length", [7, 30]))
        elif self.model_name == "WindowAverage":
            model = model_class(window_size=self.model_kwargs.get("window_size", 7))
        else:
            model = model_class(**self.model_kwargs)
        
        self._sf = StatsForecast(
            models=[model],
            freq="D",
            n_jobs=1,
        )
        
        self._sf.fit(df)
        self._last_y = y.values
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict future values.
        
        For StatsForecast, X determines the horizon.
        """
        if not self._fitted:
            raise ValueError("Model not fitted")
        
        h = len(X)
        
        try:
            forecasts = self._sf.predict(h=h)
            # Extract predictions (column name is model name)
            pred_col = forecasts.columns[-1]  # Last column is prediction
            return forecasts[pred_col].values
        except Exception:
            # Fallback: repeat last value
            return np.full(h, self._last_y[-1] if len(self._last_y) > 0 else 0)


def create_auto_arima(**kwargs) -> ModelBase:
    """AutoARIMA model."""
    config = ModelConfig(
        name="AutoARIMA",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="statsforecast",
    )
    return StatsForecastWrapper(config, "AutoARIMA", **kwargs)


def create_ets(**kwargs) -> ModelBase:
    """ETS (Exponential Smoothing) model."""
    config = ModelConfig(
        name="ETS",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="statsforecast",
    )
    return StatsForecastWrapper(config, "ETS", **kwargs)


def create_theta(**kwargs) -> ModelBase:
    """Theta model."""
    config = ModelConfig(
        name="Theta",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="statsforecast",
    )
    return StatsForecastWrapper(config, "Theta", **kwargs)


def create_mstl(**kwargs) -> ModelBase:
    """MSTL (Multiple Seasonal-Trend decomposition using Loess) model."""
    config = ModelConfig(
        name="MSTL",
        model_type="forecasting",
        params=kwargs,
        optional_dependency="statsforecast",
    )
    return StatsForecastWrapper(config, "MSTL", **kwargs)


def create_statsforecast_seasonal_naive(season_length: int = 7, **kwargs) -> ModelBase:
    """StatsForecast SeasonalNaive."""
    config = ModelConfig(
        name="SF_SeasonalNaive",
        model_type="forecasting",
        params={"season_length": season_length, **kwargs},
        optional_dependency="statsforecast",
    )
    return StatsForecastWrapper(config, "SeasonalNaive", season_length=season_length)


STATISTICAL_MODELS = {
    "AutoARIMA": create_auto_arima,
    "ETS": create_ets,
    "Theta": create_theta,
    "MSTL": create_mstl,
    "SF_SeasonalNaive": create_statsforecast_seasonal_naive,
}


def get_statistical_model(name: str, **kwargs) -> ModelBase:
    """Get a statistical model by name."""
    if name not in STATISTICAL_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(STATISTICAL_MODELS.keys())}")
    
    return STATISTICAL_MODELS[name](**kwargs)


def list_statistical_models() -> list[str]:
    """List all available statistical models."""
    return list(STATISTICAL_MODELS.keys())


def check_statsforecast_available() -> bool:
    """Check if statsforecast is installed."""
    try:
        import statsforecast
        return True
    except ImportError:
        return False
