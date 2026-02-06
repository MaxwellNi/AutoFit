#!/usr/bin/env python3
"""
Unified Preprocessing Module for Block 3.

Provides:
- Missing value handling policies per model family
- Scaling policies (per-series / global)
- Deterministic reproducibility with seed control
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# Deterministic Seed Control
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for all libraries for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def get_seed_sequence(base_seed: int, n_seeds: int) -> List[int]:
    """
    Generate a sequence of seeds from a base seed.
    
    Useful for running multiple seeds in experiments.
    
    Args:
        base_seed: Starting seed
        n_seeds: Number of seeds to generate
    
    Returns:
        List of deterministic seeds
    """
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(n_seeds)]


# =============================================================================
# Missing Value Policies
# =============================================================================

class MissingPolicy(Enum):
    """How to handle missing values."""
    KEEP = "keep"          # Keep NaN (for tree models)
    ZERO = "zero"          # Replace with 0
    MEAN = "mean"          # Replace with feature mean
    MEDIAN = "median"      # Replace with feature median
    FORWARD_FILL = "ffill" # Forward fill (for time series)
    MASK = "mask"          # Create mask column + fill


# Model family -> default missing policy
MODEL_FAMILY_MISSING_POLICY: Dict[str, MissingPolicy] = {
    # Tree models can handle NaN
    "lightgbm": MissingPolicy.KEEP,
    "xgboost": MissingPolicy.KEEP,
    "catboost": MissingPolicy.KEEP,
    "random_forest": MissingPolicy.MEDIAN,  # sklearn RF doesn't handle NaN
    "extra_trees": MissingPolicy.MEDIAN,
    "hist_gradient_boosting": MissingPolicy.KEEP,  # sklearn HGB handles NaN
    
    # Linear models need imputation
    "linear": MissingPolicy.MEAN,
    "ridge": MissingPolicy.MEAN,
    "lasso": MissingPolicy.MEAN,
    "elastic_net": MissingPolicy.MEAN,
    "logistic": MissingPolicy.MEAN,
    "svr": MissingPolicy.MEAN,
    "svc": MissingPolicy.MEAN,
    "knn": MissingPolicy.MEAN,
    
    # Deep models need masks
    "deep": MissingPolicy.MASK,
    "transformer": MissingPolicy.MASK,
    "nbeats": MissingPolicy.ZERO,
    "nhits": MissingPolicy.ZERO,
    "tft": MissingPolicy.MASK,
    "patchtst": MissingPolicy.MASK,
    
    # Statistical models
    "arima": MissingPolicy.FORWARD_FILL,
    "ets": MissingPolicy.FORWARD_FILL,
    "theta": MissingPolicy.FORWARD_FILL,
    "naive": MissingPolicy.FORWARD_FILL,
    
    # Foundation models
    "chronos": MissingPolicy.MASK,
    "timesfm": MissingPolicy.MASK,
    "moirai": MissingPolicy.MASK,
}


def get_missing_policy(model_name: str) -> MissingPolicy:
    """Get missing value policy for a model."""
    name_lower = model_name.lower()
    
    # Try exact match
    if name_lower in MODEL_FAMILY_MISSING_POLICY:
        return MODEL_FAMILY_MISSING_POLICY[name_lower]
    
    # Try prefix match
    for family, policy in MODEL_FAMILY_MISSING_POLICY.items():
        if family in name_lower:
            return policy
    
    # Default
    return MissingPolicy.MEAN


# =============================================================================
# Scaling Policies
# =============================================================================

class ScalingPolicy(Enum):
    """How to scale features."""
    NONE = "none"              # No scaling
    STANDARD = "standard"      # Zero mean, unit variance
    MINMAX = "minmax"          # Scale to [0, 1]
    ROBUST = "robust"          # Median/IQR scaling (outlier-robust)
    LOG = "log"                # Log transform (for positive values)
    PER_SERIES = "per_series"  # Scale each series independently


@dataclass
class ScalerState:
    """State of a fitted scaler."""
    policy: ScalingPolicy
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    iqr: Optional[np.ndarray] = None


class FeatureScaler:
    """
    Feature scaler with configurable policies.
    
    Supports global scaling and per-series scaling.
    """
    
    def __init__(
        self,
        policy: ScalingPolicy = ScalingPolicy.STANDARD,
        per_series: bool = False,
        series_col: Optional[str] = None,
    ):
        self.policy = policy
        self.per_series = per_series
        self.series_col = series_col
        self.state: Optional[ScalerState] = None
        self.series_states: Dict[Any, ScalerState] = {}
    
    def fit(self, X: pd.DataFrame, feature_cols: List[str]) -> "FeatureScaler":
        """Fit scaler on training data."""
        if self.per_series and self.series_col:
            # Fit per series
            for series_id, group in X.groupby(self.series_col):
                self.series_states[series_id] = self._fit_array(
                    group[feature_cols].values
                )
        else:
            # Fit globally
            self.state = self._fit_array(X[feature_cols].values)
        
        return self
    
    def _fit_array(self, arr: np.ndarray) -> ScalerState:
        """Fit scaler on numpy array."""
        state = ScalerState(policy=self.policy)
        
        if self.policy == ScalingPolicy.STANDARD:
            state.mean = np.nanmean(arr, axis=0)
            state.std = np.nanstd(arr, axis=0)
            state.std = np.where(state.std == 0, 1.0, state.std)
        
        elif self.policy == ScalingPolicy.MINMAX:
            state.min_val = np.nanmin(arr, axis=0)
            state.max_val = np.nanmax(arr, axis=0)
            # Avoid division by zero
            range_val = state.max_val - state.min_val
            state.max_val = np.where(range_val == 0, state.min_val + 1, state.max_val)
        
        elif self.policy == ScalingPolicy.ROBUST:
            state.median = np.nanmedian(arr, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            state.iqr = q75 - q25
            state.iqr = np.where(state.iqr == 0, 1.0, state.iqr)
        
        return state
    
    def transform(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        X_out = X.copy()
        
        if self.per_series and self.series_col:
            # Transform per series
            for series_id, group in X_out.groupby(self.series_col):
                if series_id in self.series_states:
                    state = self.series_states[series_id]
                else:
                    # Use global state or first series state
                    state = self.state or list(self.series_states.values())[0]
                
                idx = group.index
                X_out.loc[idx, feature_cols] = self._transform_array(
                    group[feature_cols].values, state
                )
        else:
            if self.state is None:
                raise ValueError("Scaler not fitted")
            X_out[feature_cols] = self._transform_array(
                X_out[feature_cols].values, self.state
            )
        
        return X_out
    
    def _transform_array(self, arr: np.ndarray, state: ScalerState) -> np.ndarray:
        """Transform numpy array."""
        if state.policy == ScalingPolicy.NONE:
            return arr
        
        if state.policy == ScalingPolicy.STANDARD:
            return (arr - state.mean) / state.std
        
        if state.policy == ScalingPolicy.MINMAX:
            return (arr - state.min_val) / (state.max_val - state.min_val)
        
        if state.policy == ScalingPolicy.ROBUST:
            return (arr - state.median) / state.iqr
        
        if state.policy == ScalingPolicy.LOG:
            return np.log1p(np.clip(arr, 0, None))
        
        return arr
    
    def inverse_transform(
        self,
        X: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Inverse transform to original scale."""
        X_out = X.copy()
        
        if self.state is None:
            raise ValueError("Scaler not fitted")
        
        X_out[feature_cols] = self._inverse_transform_array(
            X_out[feature_cols].values, self.state
        )
        
        return X_out
    
    def _inverse_transform_array(
        self, arr: np.ndarray, state: ScalerState
    ) -> np.ndarray:
        """Inverse transform numpy array."""
        if state.policy == ScalingPolicy.NONE:
            return arr
        
        if state.policy == ScalingPolicy.STANDARD:
            return arr * state.std + state.mean
        
        if state.policy == ScalingPolicy.MINMAX:
            return arr * (state.max_val - state.min_val) + state.min_val
        
        if state.policy == ScalingPolicy.ROBUST:
            return arr * state.iqr + state.median
        
        if state.policy == ScalingPolicy.LOG:
            return np.expm1(arr)
        
        return arr


# =============================================================================
# Unified Preprocessor
# =============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    missing_policy: MissingPolicy = MissingPolicy.MEAN
    scaling_policy: ScalingPolicy = ScalingPolicy.STANDARD
    per_series_scaling: bool = False
    series_col: Optional[str] = None
    
    # Columns to exclude from scaling (e.g., targets)
    exclude_from_scaling: List[str] = field(default_factory=list)
    
    # Columns to create masks for
    create_masks_for: List[str] = field(default_factory=list)
    
    # Random seed
    seed: int = 42


class UnifiedPreprocessor:
    """
    Unified preprocessor for Block 3 benchmark.
    
    Handles missing values, scaling, and mask creation
    in a consistent way across all models.
    """
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.scaler: Optional[FeatureScaler] = None
        self.fill_values: Dict[str, float] = {}
        self.feature_cols: List[str] = []
    
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> "UnifiedPreprocessor":
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training DataFrame
            feature_cols: Columns to preprocess (None = all numeric)
        """
        set_global_seed(self.config.seed)
        
        # Determine feature columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude specified columns
        feature_cols = [c for c in feature_cols if c not in self.config.exclude_from_scaling]
        self.feature_cols = feature_cols
        
        # Compute fill values for missing policy
        if self.config.missing_policy == MissingPolicy.MEAN:
            self.fill_values = df[feature_cols].mean().to_dict()
        elif self.config.missing_policy == MissingPolicy.MEDIAN:
            self.fill_values = df[feature_cols].median().to_dict()
        elif self.config.missing_policy == MissingPolicy.ZERO:
            self.fill_values = {c: 0.0 for c in feature_cols}
        
        # Fit scaler
        if self.config.scaling_policy != ScalingPolicy.NONE:
            self.scaler = FeatureScaler(
                policy=self.config.scaling_policy,
                per_series=self.config.per_series_scaling,
                series_col=self.config.series_col,
            )
            self.scaler.fit(df, feature_cols)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        df_out = df.copy()
        
        # Create masks if needed
        if self.config.missing_policy == MissingPolicy.MASK:
            for col in self.config.create_masks_for:
                if col in df_out.columns:
                    df_out[f"{col}_mask"] = df_out[col].notna().astype(float)
        
        # Handle missing values
        if self.config.missing_policy == MissingPolicy.KEEP:
            pass  # Keep NaN
        elif self.config.missing_policy == MissingPolicy.FORWARD_FILL:
            df_out[self.feature_cols] = df_out[self.feature_cols].ffill()
            # Fill remaining NaN with 0
            df_out[self.feature_cols] = df_out[self.feature_cols].fillna(0)
        elif self.config.missing_policy in (
            MissingPolicy.MEAN, MissingPolicy.MEDIAN, MissingPolicy.ZERO, MissingPolicy.MASK
        ):
            for col in self.feature_cols:
                if col in self.fill_values:
                    df_out[col] = df_out[col].fillna(self.fill_values[col])
        
        # Apply scaling
        if self.scaler is not None:
            df_out = self.scaler.transform(df_out, self.feature_cols)
        
        return df_out
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, feature_cols).transform(df)
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        target_col: str,
    ) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.
        
        Args:
            predictions: Model predictions
            target_col: Target column name
        
        Returns:
            Predictions in original scale
        """
        if self.scaler is None or target_col not in self.feature_cols:
            return predictions
        
        # Create temporary DataFrame for inverse transform
        df_temp = pd.DataFrame({target_col: predictions})
        df_inv = self.scaler.inverse_transform(df_temp, [target_col])
        return df_inv[target_col].values


__all__ = [
    "set_global_seed",
    "get_seed_sequence",
    "MissingPolicy",
    "get_missing_policy",
    "MODEL_FAMILY_MISSING_POLICY",
    "ScalingPolicy",
    "ScalerState",
    "FeatureScaler",
    "PreprocessConfig",
    "UnifiedPreprocessor",
]
