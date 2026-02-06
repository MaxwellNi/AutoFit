#!/usr/bin/env python3
"""
Task 1: Outcome Prediction.

Predict final offering outcomes using early-K snapshot features.
This is a classification + regression + ranking task.

Targets (mapped from freeze data columns):
- is_funded: Whether offering was funded (binary classification)
- funding_raised_usd: Final amount raised in USD (regression)
- investors_count: Total number of investors (regression)

Legacy targets (auto-mapped):
- success_binary -> is_funded
- total_amount_sold -> funding_raised_usd
- number_investors -> investors_count

Anti-leakage: Only use features available at snapshot K days after launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss,
    log_loss, mean_absolute_error, mean_squared_error, ndcg_score,
    precision_recall_curve, roc_auc_score,
)
from scipy.stats import spearmanr

from .base import TaskBase, TaskConfig, SplitConfig


# Class description for registry
TASK1_DESCRIPTION = "Outcome prediction from early-K snapshots (classification + regression + ranking)"


def safe_auc(y_true, y_pred):
    """AUC with handling for edge cases."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return np.nan


def safe_prauc(y_true, y_pred):
    """PR-AUC with handling for edge cases."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, y_pred)
    except Exception:
        return np.nan


def safe_brier(y_true, y_pred):
    """Brier score with clipping."""
    try:
        y_pred_clipped = np.clip(y_pred, 0, 1)
        return brier_score_loss(y_true, y_pred_clipped)
    except Exception:
        return np.nan


def safe_logloss(y_true, y_pred):
    """Log loss with clipping."""
    try:
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred_clipped)
    except Exception:
        return np.nan


def safe_spearman(y_true, y_pred):
    """Spearman correlation."""
    try:
        return spearmanr(y_true, y_pred)[0]
    except Exception:
        return np.nan


def safe_ndcg(y_true, y_pred, k=10):
    """NDCG@K for ranking."""
    try:
        # Reshape for sklearn
        y_true = np.array(y_true).reshape(1, -1)
        y_pred = np.array(y_pred).reshape(1, -1)
        return ndcg_score(y_true, y_pred, k=k)
    except Exception:
        return np.nan


class Task1OutcomePrediction(TaskBase):
    """
    Task 1: Outcome Prediction.
    
    Build early-K snapshot panel and predict final outcomes.
    Supports classification, regression, and ranking.
    """
    
    description = TASK1_DESCRIPTION
    
    # Column name mappings from legacy to actual freeze columns
    COLUMN_MAP = {
        # Legacy -> Actual
        "total_amount_sold": "funding_raised_usd",
        "success_binary": "is_funded",
        "number_investors": "investors_count",
        # For success_binary derivation
        "total_offering_amount": "funding_goal_usd",
    }
    
    def __init__(self, config: TaskConfig, dataset=None):
        super().__init__(config, dataset)
        self._core_df = None
        self._text_df = None
        self._edgar_df = None
    
    def _map_target(self, target: str) -> str:
        """Map legacy target names to actual data columns."""
        return self.COLUMN_MAP.get(target, target)
    
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
        horizon: int,  # K = early snapshot days
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build early-K snapshot features (vectorized implementation).
        
        For each entity:
        1. Find first observation date (launch date proxy)
        2. Extract features at launch_date + K days
        3. Extract final outcome as label
        
        Anti-leakage: Features only use data up to launch_date + K.
        """
        self._load_data()
        
        K = horizon
        core_df = self._core_df.copy()
        
        # Ensure date column exists
        date_col = "crawled_date_day"
        if date_col not in core_df.columns:
            date_col = "crawled_date"
        
        core_df[date_col] = pd.to_datetime(core_df[date_col])
        
        # Vectorized: compute first/last dates per entity
        entity_stats = core_df.groupby("entity_id")[date_col].agg(["min", "max"])
        entity_stats.columns = ["first_date", "last_date"]
        entity_stats["snapshot_date"] = entity_stats["first_date"] + pd.Timedelta(days=K)
        
        # Map target name
        actual_target = self._map_target(target)
        
        # Get final row per entity (for labels)
        idx_last = core_df.groupby("entity_id")[date_col].idxmax()
        final_rows = core_df.loc[idx_last].set_index("entity_id")
        
        # Get snapshot row per entity (last row before snapshot_date)
        # Merge snapshot_date into core_df
        core_df = core_df.merge(
            entity_stats[["snapshot_date"]].reset_index(),
            on="entity_id",
            how="left"
        )
        
        # Filter to rows before snapshot date
        snapshot_mask = core_df[date_col] <= core_df["snapshot_date"]
        snapshot_df = core_df[snapshot_mask].copy()
        
        # Get last row per entity before snapshot
        idx_snapshot = snapshot_df.groupby("entity_id")[date_col].idxmax()
        feature_rows = snapshot_df.loc[idx_snapshot].set_index("entity_id")
        
        # Extract labels from final_rows
        if actual_target in final_rows.columns:
            labels = final_rows[actual_target]
        elif target == "success_binary" or actual_target == "is_funded":
            if "is_funded" in final_rows.columns:
                labels = final_rows["is_funded"]
            else:
                # Derive from funding columns
                goal_col = self._map_target("total_offering_amount")
                raised_col = self._map_target("total_amount_sold")
                if raised_col in final_rows.columns and goal_col in final_rows.columns:
                    labels = (final_rows[raised_col] >= final_rows[goal_col] * 0.5).astype(int)
                else:
                    raise ValueError(f"Cannot derive labels for target={target}")
        else:
            raise ValueError(f"Target column '{actual_target}' not found in data")
        
        # Align features and labels
        common_entities = feature_rows.index.intersection(labels.dropna().index)
        X = feature_rows.loc[common_entities]
        y = labels.loc[common_entities]
        
        # Filter out rows with NaN labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError(f"No valid samples for Task1 with K={K}, target={target}")
        
        # Apply ablation
        X = self._build_ablation_features(X, self._text_df, self._edgar_df, ablation)
        
        # Drop non-numeric columns for modeling
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
        Temporal split based on configuration.
        
        Default: 70% train, 15% val, 15% test by index order.
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
        Task 1 metrics based on target type.
        
        Classification: AUC, PR-AUC, LogLoss, Brier
        Regression: MAE, RMSE
        Ranking: NDCG@K, Spearman
        """
        target = self.config.targets[0] if self.config.targets else "funding_raised_usd"
        actual_target = self._map_target(target)
        
        if target in ["success_binary", "is_funded"] or actual_target == "is_funded":
            # Classification metrics
            return {
                "auc": safe_auc,
                "prauc": safe_prauc,
                "logloss": safe_logloss,
                "brier": safe_brier,
            }
        else:
            # Regression + ranking metrics
            return {
                "mae": mean_absolute_error,
                "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
                "spearman": safe_spearman,
                "ndcg@10": lambda y, p: safe_ndcg(y, p, k=10),
            }


def create_task1_config(
    targets: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> TaskConfig:
    """Create default Task 1 configuration."""
    return TaskConfig(
        name="task1_outcome",
        targets=targets or ["funding_raised_usd", "is_funded"],
        horizons=horizons or [7, 14, 30, 60],  # K = early snapshot days
        context_lengths=[],  # Not applicable for Task 1
        ablations=["core_only", "+text", "+edgar", "full"],
        split=SplitConfig(
            train_end=date(2024, 6, 30),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
        ),
        metrics=["auc", "prauc", "logloss", "brier", "mae", "rmse", "spearman", "ndcg@10"],
    )
