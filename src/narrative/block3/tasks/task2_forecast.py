#!/usr/bin/env python3
"""
Task 2: Trajectory Forecasting.

Multi-horizon time series forecasting with quantile/probabilistic outputs.

Targets (actual freeze columns):
- funding_raised_usd: Cumulative funding trajectory
- investors_count: Investor count trajectory  
- delta_funding_raised: Daily funding inflow

Legacy mappings:
- total_amount_sold -> funding_raised_usd
- number_investors -> investors_count

Anti-leakage: Context window uses only past values; prediction horizons are strictly future.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base import TaskBase, TaskConfig, SplitConfig


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return 2.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])


def mase(y_true, y_pred, y_train, seasonality=7):
    """Mean Absolute Scaled Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_train = np.array(y_train)
    
    # Compute naive seasonal forecast error
    if len(y_train) <= seasonality:
        naive_error = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    
    if naive_error == 0:
        return np.nan
    
    return np.mean(np.abs(y_true - y_pred)) / naive_error


def pinball_loss(y_true, y_pred, quantile=0.5):
    """Pinball loss for quantile regression."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = y_true - y_pred
    return np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))


def crps_empirical(y_true, y_samples):
    """
    Continuous Ranked Probability Score (empirical).
    
    y_samples: array of shape (n_samples, n_predictions) for probabilistic forecasts.
    """
    y_true = np.array(y_true)
    y_samples = np.array(y_samples)
    
    if y_samples.ndim == 1:
        # Point forecast - use absolute error as proxy
        return np.mean(np.abs(y_true - y_samples))
    
    # Empirical CRPS
    n_samples = y_samples.shape[1]
    crps_values = []
    
    for i, y in enumerate(y_true):
        samples = y_samples[i]
        # CRPS = E|Y - y| - 0.5 * E|Y - Y'|
        term1 = np.mean(np.abs(samples - y))
        term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        crps_values.append(term1 - term2)
    
    return np.mean(crps_values)


def interval_coverage(y_true, y_lower, y_upper):
    """Coverage of prediction intervals."""
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(covered)


TASK2_DESCRIPTION = "Multi-horizon trajectory forecasting with quantile outputs"


class Task2TrajectoryForecasting(TaskBase):
    """
    Task 2: Trajectory Forecasting.
    
    Build sequences with context length L and forecast horizons H.
    Supports point forecasts and probabilistic outputs.
    """
    
    description = TASK2_DESCRIPTION
    
    def __init__(self, config: TaskConfig, dataset=None):
        super().__init__(config, dataset)
        self._core_df = None
        self._text_df = None
        self._edgar_df = None
        self._y_train = None  # For MASE computation
    
    def _load_data(self):
        """Lazy load data."""
        if self._core_df is None:
            self._core_df = self.dataset.get_offers_core_daily()
            
        if self._text_df is None:
            try:
                self._text_df = self.dataset.get_offers_text()
            except Exception:
                self._text_df = pd.DataFrame()
        
        if self._edgar_df is None:
            try:
                self._edgar_df = self.dataset.get_edgar_store()
            except Exception:
                self._edgar_df = pd.DataFrame()
    
    def build_dataset(
        self,
        ablation: str,
        horizon: int,  # H = forecast horizon
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build forecasting sequences.
        
        For each entity at each time t:
        - Context: features from [t - L, t]
        - Target: value at t + H
        
        Anti-leakage: Context uses only past values.
        """
        self._load_data()
        
        H = horizon
        L = self.config.context_lengths[0] if self.config.context_lengths else 30
        
        core_df = self._core_df.copy()
        
        # Ensure date column
        date_col = "crawled_date_day"
        if date_col not in core_df.columns:
            date_col = "crawled_date"
        
        core_df[date_col] = pd.to_datetime(core_df[date_col])
        core_df = core_df.sort_values([date_col])
        
        # Build sequences for each entity
        features_list = []
        labels_list = []
        
        entities = core_df["entity_id"].unique()
        
        for entity_id in entities:
            entity_data = core_df[core_df["entity_id"] == entity_id].sort_values(date_col)
            
            if len(entity_data) < L + H:
                continue
            
            # Create sliding windows
            for i in range(L, len(entity_data) - H):
                # Context window
                context_end = i
                context_start = max(0, i - L)
                
                context_data = entity_data.iloc[context_start:context_end + 1]
                
                # Target at horizon H
                target_idx = i + H
                if target_idx >= len(entity_data):
                    break
                
                target_row = entity_data.iloc[target_idx]
                
                if target not in target_row or pd.isna(target_row[target]):
                    continue
                
                # Aggregate context features (last value + rolling stats)
                feature_row = context_data.iloc[-1].copy()
                
                # Add rolling statistics as features
                if target in context_data.columns:
                    target_history = context_data[target].dropna()
                    if len(target_history) > 0:
                        feature_row[f"{target}_mean_L{L}"] = target_history.mean()
                        feature_row[f"{target}_std_L{L}"] = target_history.std()
                        feature_row[f"{target}_last"] = target_history.iloc[-1]
                        if len(target_history) > 1:
                            feature_row[f"{target}_trend"] = target_history.iloc[-1] - target_history.iloc[0]
                
                features_list.append(feature_row)
                labels_list.append(target_row[target])
        
        if len(features_list) == 0:
            raise ValueError(f"No valid samples for Task2 with H={H}, L={L}, target={target}")
        
        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list, index=X.index)
        
        # Store y_train for MASE
        self._y_train = y.iloc[:int(len(y) * 0.7)].values
        
        # Apply ablation
        X = self._build_ablation_features(X, self._text_df, self._edgar_df, ablation)
        
        # Drop non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def get_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Temporal split for forecasting.
        
        70% train, 15% val, 15% test by temporal order.
        """
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        
        return [(train_idx, val_idx, test_idx)]
    
    def get_metrics(self) -> Dict[str, callable]:
        """
        Task 2 forecasting metrics.
        
        Point: MAE, RMSE, sMAPE, MASE
        Quantile: Pinball loss
        Probabilistic: CRPS, interval coverage
        """
        y_train = self._y_train if self._y_train is not None else np.array([])
        
        return {
            "mae": mean_absolute_error,
            "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
            "smape": smape,
            "mase": lambda y, p: mase(y, p, y_train),
            "pinball_50": lambda y, p: pinball_loss(y, p, 0.5),
            "pinball_90": lambda y, p: pinball_loss(y, p, 0.9),
        }


def create_task2_config(
    targets: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    context_lengths: Optional[List[int]] = None,
) -> TaskConfig:
    """Create default Task 2 configuration."""
    return TaskConfig(
        name="task2_forecast",
        targets=targets or ["funding_raised_usd", "investors_count"],
        horizons=horizons or [7, 14, 30, 60],  # H = forecast horizon
        context_lengths=context_lengths or [30, 60, 90],
        ablations=["core_only", "+text", "+edgar", "full"],
        split=SplitConfig(
            train_end=date(2024, 6, 30),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
        ),
        metrics=["mae", "rmse", "smape", "mase", "pinball_50", "pinball_90", "crps"],
    )
