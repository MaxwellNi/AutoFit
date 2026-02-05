#!/usr/bin/env python3
"""
Block 3 Task Base Class.

All tasks inherit from TaskBase and implement:
- build_dataset: Construct features with anti-leakage guarantees
- get_splits: Return train/val/test splits by date
- get_metrics: Return task-specific evaluation metrics
- run: Execute the task end-to-end
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import from existing infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer


@dataclass
class SplitConfig:
    """Configuration for temporal train/val/test splits."""
    train_end: date  # Last date in training set
    val_end: date    # Last date in validation set
    test_start: date # First date in test set (for anti-leakage verification)
    
    # Optional: walk-forward parameters
    n_folds: int = 1
    gap_days: int = 0  # Gap between train and val/test to prevent leakage


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    targets: List[str]
    horizons: List[int]  # K for early snapshots or H for forecast
    context_lengths: List[int]
    ablations: List[str]  # core_only, +text, +edgar, full
    split: SplitConfig
    metrics: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from running a task."""
    task_name: str
    model_name: str
    ablation: str
    target: str
    horizon: int
    metrics: Dict[str, float]
    split_info: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "ablation": self.ablation,
            "target": self.target,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "split_info": self.split_info,
            "timestamp": self.timestamp,
        }


class TaskBase(ABC):
    """
    Abstract base class for all Block 3 tasks.
    
    Anti-leakage guarantees:
    1. build_dataset() only uses features available at cutoff_date
    2. get_splits() enforces strict temporal separation
    3. All date-based filtering is explicit and auditable
    """
    
    # Override in subclasses
    description: str = "Base task"
    
    def __init__(self, config: TaskConfig, dataset: Optional[Block3Dataset] = None):
        self.config = config
        self.dataset = dataset or Block3Dataset.from_pointer()
        self._cache: Dict[str, pd.DataFrame] = {}
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @abstractmethod
    def build_dataset(
        self,
        ablation: str,
        horizon: int,
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build features and labels for the task.
        
        Args:
            ablation: One of core_only, +text, +edgar, full
            horizon: Task-specific horizon (K for early snapshots, H for forecast)
            target: Target column name
        
        Returns:
            X: Feature DataFrame with index
            y: Target Series aligned with X
        
        Anti-leakage: Features must only use information available at the
        observation date. For forecasting, this means no future values.
        """
        pass
    
    @abstractmethod
    def get_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return train/val/test index splits.
        
        Returns list of (train_idx, val_idx, test_idx) for each fold.
        
        Anti-leakage: Train dates < Val dates < Test dates (strict).
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, callable]:
        """
        Return task-specific evaluation metrics.
        
        Returns dict of metric_name -> metric_function(y_true, y_pred).
        """
        pass
    
    def run(
        self,
        model,
        ablation: str,
        horizon: int,
        target: str,
    ) -> TaskResult:
        """
        Run the task with a given model.
        
        Args:
            model: Model with fit/predict interface
            ablation: Feature ablation setting
            horizon: Task-specific horizon
            target: Target column
        
        Returns:
            TaskResult with metrics and metadata
        """
        # Build dataset
        X, y = self.build_dataset(ablation, horizon, target)
        
        # Get splits
        splits = self.get_splits(X, y)
        
        # Run evaluation
        all_metrics = []
        split_info = {}
        
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict on test
            y_pred = model.predict(X_test)
            
            # Compute metrics
            metrics = self.get_metrics()
            fold_metrics = {}
            for name, fn in metrics.items():
                try:
                    fold_metrics[name] = float(fn(y_test, y_pred))
                except Exception as e:
                    fold_metrics[name] = np.nan
            
            all_metrics.append(fold_metrics)
            split_info[f"fold_{fold_idx}"] = {
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
            }
        
        # Aggregate metrics across folds
        agg_metrics = {}
        for name in all_metrics[0].keys():
            values = [m[name] for m in all_metrics if not np.isnan(m[name])]
            agg_metrics[name] = float(np.mean(values)) if values else np.nan
        
        return TaskResult(
            task_name=self.name,
            model_name=getattr(model, "name", model.__class__.__name__),
            ablation=ablation,
            target=target,
            horizon=horizon,
            metrics=agg_metrics,
            split_info=split_info,
        )
    
    # -------------------------------------------------------------------------
    # Utility methods for anti-leakage
    # -------------------------------------------------------------------------
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        date_col: str,
        end_date: date,
        start_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range (inclusive)."""
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not in DataFrame")
        
        dates = pd.to_datetime(df[date_col]).dt.date
        mask = dates <= end_date
        if start_date is not None:
            mask &= dates >= start_date
        
        return df[mask].copy()
    
    def _verify_no_leakage(
        self,
        feature_dates: pd.Series,
        label_dates: pd.Series,
        horizon: int,
    ) -> bool:
        """
        Verify that features don't leak future information.
        
        For each sample, feature_date + horizon <= label_date.
        """
        feature_dt = pd.to_datetime(feature_dates)
        label_dt = pd.to_datetime(label_dates)
        
        # Features must be at least `horizon` days before labels
        valid = (label_dt - feature_dt).dt.days >= horizon
        
        if not valid.all():
            n_violations = (~valid).sum()
            raise ValueError(f"Leakage detected: {n_violations} samples have future features")
        
        return True
    
    def _build_ablation_features(
        self,
        core_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame],
        edgar_df: Optional[pd.DataFrame],
        ablation: str,
    ) -> pd.DataFrame:
        """
        Build feature DataFrame based on ablation setting.
        
        Ablation settings:
        - core_only: Only offers_core features
        - +text: core + offers_text features
        - +edgar: core + edgar features
        - full: core + text + edgar
        """
        result = core_df.copy()
        
        if ablation in ["+text", "full"] and text_df is not None:
            # Join text features
            result = self.dataset.join_core_with_text(result, text_df)
        
        if ablation in ["+edgar", "full"] and edgar_df is not None:
            # Join edgar features
            result = self.dataset.join_core_with_edgar(result, edgar_df)
        
        return result
