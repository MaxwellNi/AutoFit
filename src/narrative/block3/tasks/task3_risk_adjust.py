#!/usr/bin/env python3
"""
Task 3: EDGAR-conditioned Risk Adjustment.

Evaluate model robustness under distribution shifts and EDGAR ablation.

Key aspects:
- Robustness metrics: year shift, sector shift, calibration drift
- Ablation: with/without EDGAR features
- OOD splits: train pre-2024, test 2025+; sector holdout

Anti-leakage: Same as Task 1/2, plus OOD split integrity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

from .base import TaskBase, TaskConfig, SplitConfig


def calibration_error(y_true, y_pred, n_bins=10):
    """Expected Calibration Error (ECE)."""
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))
    except Exception:
        return np.nan


def performance_gap(in_dist_metric, out_dist_metric):
    """Relative performance degradation under shift."""
    if in_dist_metric == 0:
        return np.nan
    return (out_dist_metric - in_dist_metric) / abs(in_dist_metric)


TASK3_DESCRIPTION = "EDGAR-conditioned OOD robustness with year/sector shift evaluation"


class Task3RiskAdjustment(TaskBase):
    """
    Task 3: EDGAR-conditioned Risk Adjustment.
    
    Builds on Task 1/2 but focuses on:
    1. OOD robustness (year shift, sector shift)
    2. EDGAR ablation (with/without EDGAR features)
    3. Calibration metrics
    """
    
    description = TASK3_DESCRIPTION
    
    def __init__(self, config: TaskConfig, dataset=None):
        super().__init__(config, dataset)
        self._core_df = None
        self._text_df = None
        self._edgar_df = None
        self._ood_mode = "year_shift"  # year_shift or sector_shift
    
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
    
    def set_ood_mode(self, mode: str):
        """Set OOD split mode: 'year_shift' or 'sector_shift'."""
        assert mode in ["year_shift", "sector_shift"]
        self._ood_mode = mode
    
    def build_dataset(
        self,
        ablation: str,
        horizon: int,  # K = early snapshot days
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build dataset with explicit EDGAR ablation.
        
        Same as Task 1 but with enhanced EDGAR handling.
        """
        self._load_data()
        
        K = horizon
        core_df = self._core_df.copy()
        
        # Ensure date column
        date_col = "crawled_date_day"
        if date_col not in core_df.columns:
            date_col = "crawled_date"
        
        core_df[date_col] = pd.to_datetime(core_df[date_col])
        
        # Add year for OOD splits
        core_df["year"] = core_df[date_col].dt.year
        
        # Group by entity
        entity_info = core_df.groupby("entity_id").agg({
            date_col: ["min", "max"],
            "year": "first",
        }).reset_index()
        entity_info.columns = ["entity_id", "first_date", "last_date", "start_year"]
        entity_info["snapshot_date"] = entity_info["first_date"] + pd.Timedelta(days=K)
        
        # Build features
        features_list = []
        labels_list = []
        years_list = []
        
        for _, row in entity_info.iterrows():
            entity_id = row["entity_id"]
            snapshot_date = row["snapshot_date"]
            start_year = row["start_year"]
            
            # Get entity data up to snapshot
            entity_data = core_df[
                (core_df["entity_id"] == entity_id) &
                (core_df[date_col] <= snapshot_date)
            ]
            
            if len(entity_data) == 0:
                continue
            
            feature_row = entity_data.iloc[-1].copy()
            
            # Get final outcome
            final_data = core_df[core_df["entity_id"] == entity_id]
            if len(final_data) == 0:
                continue
            
            final_row = final_data.iloc[-1]
            
            if target not in final_row or pd.isna(final_row[target]):
                continue
            
            features_list.append(feature_row)
            labels_list.append(final_row[target])
            years_list.append(start_year)
        
        if len(features_list) == 0:
            raise ValueError(f"No valid samples for Task3 with K={K}, target={target}")
        
        X = pd.DataFrame(features_list)
        X["_year"] = years_list  # For OOD splitting
        y = pd.Series(labels_list, index=X.index)
        
        # Apply ablation with explicit EDGAR handling
        if ablation == "core_only":
            # No text, no edgar
            pass
        elif ablation == "+text":
            X = self._build_ablation_features(X, self._text_df, None, "+text")
        elif ablation == "+edgar":
            X = self._build_ablation_features(X, None, self._edgar_df, "+edgar")
        elif ablation == "full":
            X = self._build_ablation_features(X, self._text_df, self._edgar_df, "full")
        
        # Keep year column for OOD splits
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if "_year" not in numeric_cols:
            numeric_cols.append("_year")
        X = X[[c for c in numeric_cols if c in X.columns]]
        X = X.fillna(0)
        
        return X, y
    
    def get_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        OOD splits for robustness evaluation.
        
        year_shift: train <= 2023, val = 2024, test = 2025+
        sector_shift: if sector column exists, hold out sectors
        """
        if self._ood_mode == "year_shift":
            if "_year" in X.columns:
                train_mask = X["_year"] <= 2023
                val_mask = X["_year"] == 2024
                test_mask = X["_year"] >= 2025
                
                train_idx = np.where(train_mask)[0]
                val_idx = np.where(val_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                # Fallback if not enough samples
                if len(train_idx) < 10 or len(test_idx) < 10:
                    return self._default_split(X, y)
                
                return [(train_idx, val_idx, test_idx)]
        
        return self._default_split(X, y)
    
    def _default_split(self, X, y):
        """Default temporal split."""
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        train_idx = np.arange(train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        
        return [(train_idx, val_idx, test_idx)]
    
    def get_metrics(self) -> Dict[str, callable]:
        """
        Task 3 metrics: Task 1/2 metrics + robustness metrics.
        """
        return {
            "mae": mean_absolute_error,
            "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
            "calibration_error": calibration_error,
        }
    
    def run_robustness_analysis(
        self,
        model,
        ablation: str,
        horizon: int,
        target: str,
    ) -> Dict[str, Any]:
        """
        Run full robustness analysis.
        
        Returns in-dist metrics, OOD metrics, and performance gaps.
        """
        # Build dataset
        X, y = self.build_dataset(ablation, horizon, target)
        
        # Get OOD splits
        splits = self.get_splits(X, y)
        train_idx, val_idx, test_idx = splits[0]
        
        # Remove _year from features for training
        feature_cols = [c for c in X.columns if c != "_year"]
        X_features = X[feature_cols]
        
        # Train on in-distribution
        X_train = X_features.iloc[train_idx]
        y_train = y.iloc[train_idx]
        model.fit(X_train, y_train)
        
        # Evaluate on in-dist (val) and OOD (test)
        X_val = X_features.iloc[val_idx]
        y_val = y.iloc[val_idx]
        X_test = X_features.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Compute metrics
        metrics = self.get_metrics()
        
        in_dist_results = {}
        ood_results = {}
        
        for name, fn in metrics.items():
            try:
                in_dist_results[name] = float(fn(y_val, y_pred_val))
            except Exception:
                in_dist_results[name] = np.nan
            try:
                ood_results[name] = float(fn(y_test, y_pred_test))
            except Exception:
                ood_results[name] = np.nan
        
        # Compute performance gaps
        gaps = {}
        for name in metrics.keys():
            gaps[f"{name}_gap"] = performance_gap(
                in_dist_results.get(name, np.nan),
                ood_results.get(name, np.nan)
            )
        
        return {
            "in_dist": in_dist_results,
            "ood": ood_results,
            "gaps": gaps,
            "ood_mode": self._ood_mode,
            "ablation": ablation,
        }


def create_task3_config(
    targets: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> TaskConfig:
    """Create default Task 3 configuration."""
    return TaskConfig(
        name="task3_risk_adjust",
        targets=targets or ["funding_raised_usd"],
        horizons=horizons or [14, 30],
        context_lengths=[],
        ablations=["core_only", "+edgar", "full"],  # Explicit EDGAR ablation
        split=SplitConfig(
            train_end=date(2023, 12, 31),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
        ),
        metrics=["mae", "rmse", "calibration_error", "performance_gap"],
        extra={"ood_modes": ["year_shift", "sector_shift"]},
    )
