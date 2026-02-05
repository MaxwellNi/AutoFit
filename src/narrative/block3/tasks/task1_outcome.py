#!/usr/bin/env python3
"""
Task 1: Outcome Prediction.

Predict final offering outcomes using early-K snapshot features.
This is a classification + regression + ranking task.

Targets:
- success_binary: Whether offering reached funding goal
- total_amount_sold: Final amount raised (regression)
- days_to_close: Days until offering closes (regression)
- ranking: Relative performance among contemporaneous offerings

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
    
    def __init__(self, config: TaskConfig, dataset=None):
        super().__init__(config, dataset)
        self._core_df = None
        self._text_df = None
        self._edgar_df = None
    
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
        Build early-K snapshot features.
        
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
        
        # Group by entity and find first/last dates
        entity_info = core_df.groupby("entity_id").agg({
            date_col: ["min", "max"],
        }).reset_index()
        entity_info.columns = ["entity_id", "first_date", "last_date"]
        
        # Calculate snapshot date = first_date + K days
        entity_info["snapshot_date"] = entity_info["first_date"] + pd.Timedelta(days=K)
        
        # For each entity, get features at snapshot date (or closest prior)
        features_list = []
        labels_list = []
        
        for _, row in entity_info.iterrows():
            entity_id = row["entity_id"]
            snapshot_date = row["snapshot_date"]
            last_date = row["last_date"]
            
            # Get entity data up to snapshot date (anti-leakage)
            entity_data = core_df[
                (core_df["entity_id"] == entity_id) &
                (core_df[date_col] <= snapshot_date)
            ]
            
            if len(entity_data) == 0:
                continue
            
            # Use last available observation as features
            feature_row = entity_data.iloc[-1].copy()
            
            # Get final outcome (last observation)
            final_data = core_df[core_df["entity_id"] == entity_id]
            if len(final_data) == 0:
                continue
            
            final_row = final_data.iloc[-1]
            
            # Extract label
            if target in final_row:
                label = final_row[target]
            elif target == "success_binary":
                # Derive from total_amount_sold vs goal
                if "total_amount_sold" in final_row and "total_offering_amount" in final_row:
                    label = 1 if final_row["total_amount_sold"] >= final_row["total_offering_amount"] * 0.5 else 0
                else:
                    continue
            else:
                continue
            
            if pd.isna(label):
                continue
            
            features_list.append(feature_row)
            labels_list.append(label)
        
        if len(features_list) == 0:
            raise ValueError(f"No valid samples for Task1 with K={K}, target={target}")
        
        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list, index=X.index)
        
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
        target = self.config.targets[0] if self.config.targets else "total_amount_sold"
        
        if target in ["success_binary"]:
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
        targets=targets or ["total_amount_sold", "success_binary"],
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
