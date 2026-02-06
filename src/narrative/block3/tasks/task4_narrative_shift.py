#!/usr/bin/env python3
"""
Task 4: Narrative Shift & GenAI Era Effect.

Quasi-causal analysis of how narrative changes (NBI/NCI indices) affect outcomes,
with particular focus on the GenAI era (2023+).

Key aspects:
- DiD estimates: pre/post 2023 cohorts
- NBI/NCI as mediators
- Placebo tests with alternative split years
- Bootstrap confidence intervals

Anti-leakage: Same as Task 1, plus proper causal identification assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from .base import TaskBase, TaskConfig, SplitConfig


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Input data
        statistic_fn: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
    
    Returns:
        (point_estimate, lower, upper)
    """
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Point estimate
    point_est = statistic_fn(data)
    
    # Bootstrap samples
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_stats.append(statistic_fn(data[idx]))
    
    boot_stats = np.array(boot_stats)
    
    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return point_est, lower, upper


def difference_in_differences(y_pre_treat, y_post_treat, y_pre_control, y_post_control):
    """
    Compute Difference-in-Differences estimate.
    
    DiD = (E[Y|post,treat] - E[Y|pre,treat]) - (E[Y|post,control] - E[Y|pre,control])
    """
    treat_diff = np.mean(y_post_treat) - np.mean(y_pre_treat)
    control_diff = np.mean(y_post_control) - np.mean(y_pre_control)
    
    return treat_diff - control_diff


TASK4_DESCRIPTION = "Narrative Shift & GenAI Era effect via DiD + mediation analysis"


class Task4NarrativeShift(TaskBase):
    """
    Task 4: Narrative Shift & GenAI Era Effect.
    
    Analyze causal effect of narrative changes on outcomes.
    Uses DiD design with GenAI era (2023+) as treatment.
    """
    
    description = TASK4_DESCRIPTION
    
    def __init__(self, config: TaskConfig, dataset=None):
        super().__init__(config, dataset)
        self._core_df = None
        self._text_df = None
        self._edgar_df = None
        self._treatment_year = 2023  # GenAI era starts
    
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
    
    def _compute_nbi_nci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Narrative Behavior Index (NBI) and Narrative Content Index (NCI).
        
        NBI: Behavioral signals from text (sentiment, readability, etc.)
        NCI: Content signals from text (topics, risk mentions, etc.)
        
        If text features not available, returns zeros.
        """
        result = df.copy()
        
        # Placeholder NBI/NCI computation
        # In practice, this would use NLP features from offers_text
        
        # NBI: behavioral proxy (length, update frequency)
        if "text_length" in df.columns:
            result["nbi"] = (df["text_length"] - df["text_length"].mean()) / (df["text_length"].std() + 1e-8)
        else:
            result["nbi"] = 0.0
        
        # NCI: content proxy (risk mentions, positive terms)
        if "risk_mention_count" in df.columns:
            result["nci"] = (df["risk_mention_count"] - df["risk_mention_count"].mean()) / (df["risk_mention_count"].std() + 1e-8)
        else:
            result["nci"] = 0.0
        
        return result
    
    def build_dataset(
        self,
        ablation: str,
        horizon: int,  # K = early snapshot days (not used directly here)
        target: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build dataset for DiD analysis.
        
        Adds treatment indicator (post-2023) and NBI/NCI mediators.
        """
        self._load_data()
        
        core_df = self._core_df.copy()
        
        # Ensure date column
        date_col = "crawled_date_day"
        if date_col not in core_df.columns:
            date_col = "crawled_date"
        
        core_df[date_col] = pd.to_datetime(core_df[date_col])
        core_df["year"] = core_df[date_col].dt.year
        
        # Treatment indicator: GenAI era
        core_df["post_treatment"] = (core_df["year"] >= self._treatment_year).astype(int)
        
        # Group by entity for outcome-level analysis
        entity_outcomes = core_df.groupby("entity_id").agg({
            target: "last",
            "year": "first",
            "post_treatment": "first",
            date_col: "first",
        }).reset_index()
        
        entity_outcomes = entity_outcomes.dropna(subset=[target])
        
        # Compute NBI/NCI
        entity_outcomes = self._compute_nbi_nci(entity_outcomes)
        
        # Apply ablation to get features
        X = entity_outcomes.copy()
        
        if ablation in ["+text", "full"] and len(self._text_df) > 0:
            # Merge text features
            X = self.dataset.join_core_with_text(X, self._text_df)
        
        if ablation in ["+edgar", "full"] and len(self._edgar_df) > 0:
            X = self.dataset.join_core_with_edgar(X, self._edgar_df)
        
        y = X[target].copy()
        
        # Keep key columns for DiD analysis
        key_cols = ["entity_id", "year", "post_treatment", "nbi", "nci"]
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = list(set(key_cols + numeric_cols) & set(X.columns))
        
        X = X[keep_cols].fillna(0)
        
        return X, y
    
    def get_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split for DiD: pre-treatment, post-treatment.
        
        For causal analysis, we use the full dataset but split by treatment status.
        """
        if "post_treatment" in X.columns:
            pre_idx = np.where(X["post_treatment"] == 0)[0]
            post_idx = np.where(X["post_treatment"] == 1)[0]
            
            # For model training, use a subset
            n_pre = len(pre_idx)
            n_post = len(post_idx)
            
            train_idx = pre_idx[:int(n_pre * 0.7)]
            val_idx = pre_idx[int(n_pre * 0.7):]
            test_idx = post_idx
            
            return [(train_idx, val_idx, test_idx)]
        
        # Fallback
        n = len(X)
        return [(np.arange(int(n*0.7)), np.arange(int(n*0.7), int(n*0.85)), np.arange(int(n*0.85), n))]
    
    def get_metrics(self) -> Dict[str, callable]:
        """
        Task 4 metrics: regression metrics + DiD-specific.
        """
        return {
            "mae": mean_absolute_error,
            "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
            "r2": r2_score,
        }
    
    def run_did_analysis(
        self,
        ablation: str,
        target: str,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run full Difference-in-Differences analysis.
        
        Returns DiD estimate, bootstrap CI, and placebo test results.
        """
        X, y = self.build_dataset(ablation, 0, target)
        
        if "post_treatment" not in X.columns:
            return {"error": "post_treatment column not found"}
        
        # Split into pre/post groups
        pre_mask = X["post_treatment"] == 0
        post_mask = X["post_treatment"] == 1
        
        y_pre = y[pre_mask].values
        y_post = y[post_mask].values
        
        if len(y_pre) == 0 or len(y_post) == 0:
            return {"error": "Insufficient data for DiD"}
        
        # Simple DiD (no control group - use mean shift)
        pre_mean = np.mean(y_pre)
        post_mean = np.mean(y_post)
        did_estimate = post_mean - pre_mean
        
        # Bootstrap CI for DiD estimate
        def did_stat(data):
            # Combine pre and post, resample, compute mean diff
            n_pre = len(y_pre)
            return np.mean(data[n_pre:]) - np.mean(data[:n_pre])
        
        combined = np.concatenate([y_pre, y_post])
        _, ci_lower, ci_upper = bootstrap_ci(combined, did_stat, n_bootstrap)
        
        # Placebo test: use 2022 as fake treatment year
        placebo_results = self._run_placebo_test(X, y, target, placebo_year=2022)
        
        # Mediation analysis: NBI/NCI as mediators
        mediation_results = self._run_mediation_analysis(X, y)
        
        return {
            "did_estimate": did_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "n_pre": len(y_pre),
            "n_post": len(y_post),
            "placebo": placebo_results,
            "mediation": mediation_results,
            "treatment_year": self._treatment_year,
            "ablation": ablation,
            "target": target,
        }
    
    def _run_placebo_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target: str,
        placebo_year: int = 2022,
    ) -> Dict[str, Any]:
        """Run placebo test with fake treatment year."""
        if "year" not in X.columns:
            return {"error": "year column not found"}
        
        # Create placebo treatment
        placebo_treat = (X["year"] >= placebo_year) & (X["year"] < self._treatment_year)
        placebo_pre = X["year"] < placebo_year
        
        if placebo_treat.sum() == 0 or placebo_pre.sum() == 0:
            return {"error": "Insufficient data for placebo"}
        
        y_placebo_pre = y[placebo_pre].values
        y_placebo_treat = y[placebo_treat].values
        
        placebo_did = np.mean(y_placebo_treat) - np.mean(y_placebo_pre)
        
        return {
            "placebo_year": placebo_year,
            "placebo_did": placebo_did,
            "n_pre": len(y_placebo_pre),
            "n_treat": len(y_placebo_treat),
        }
    
    def _run_mediation_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Run mediation analysis with NBI/NCI.
        
        Tests if narrative indices mediate the treatment effect.
        """
        if "nbi" not in X.columns or "nci" not in X.columns:
            return {"error": "NBI/NCI not computed"}
        
        if "post_treatment" not in X.columns:
            return {"error": "post_treatment not found"}
        
        try:
            # Total effect: Y ~ treatment
            X_treat = X[["post_treatment"]].values
            model_total = LinearRegression().fit(X_treat, y)
            total_effect = model_total.coef_[0]
            
            # Direct effect: Y ~ treatment + NBI + NCI
            X_full = X[["post_treatment", "nbi", "nci"]].values
            model_direct = LinearRegression().fit(X_full, y)
            direct_effect = model_direct.coef_[0]
            
            # Indirect effect (mediated through NBI/NCI)
            indirect_effect = total_effect - direct_effect
            
            # Proportion mediated
            if total_effect != 0:
                prop_mediated = indirect_effect / total_effect
            else:
                prop_mediated = 0
            
            return {
                "total_effect": float(total_effect),
                "direct_effect": float(direct_effect),
                "indirect_effect": float(indirect_effect),
                "proportion_mediated": float(prop_mediated),
                "nbi_coef": float(model_direct.coef_[1]),
                "nci_coef": float(model_direct.coef_[2]),
            }
        except Exception as e:
            return {"error": str(e)}


def create_task4_config(
    targets: Optional[List[str]] = None,
) -> TaskConfig:
    """Create default Task 4 configuration."""
    return TaskConfig(
        name="task4_narrative_shift",
        targets=targets or ["funding_raised_usd", "investors_count"],
        horizons=[0],  # Not applicable
        context_lengths=[],
        ablations=["core_only", "+text", "full"],
        split=SplitConfig(
            train_end=date(2022, 12, 31),  # Pre-GenAI
            val_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),   # GenAI era
        ),
        metrics=["did_estimate", "placebo_did", "proportion_mediated"],
        extra={
            "treatment_year": 2023,
            "placebo_years": [2021, 2022],
            "mediators": ["nbi", "nci"],
        },
    )
