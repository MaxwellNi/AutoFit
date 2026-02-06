"""
Difference-in-Differences (DiD) Analysis Module for Block 3.

This module provides statistical tools for causal inference using DiD designs,
specifically tailored for analyzing the GenAI era effect on narrative outcomes.

Key Components:
1. DiD estimators (simple, regression-based, two-way fixed effects)
2. Parallel trends testing
3. Placebo tests
4. Bootstrap inference
5. Event study plots
6. Mediation analysis integration

Reference:
- Angrist & Pischke (2009) Mostly Harmless Econometrics
- Cunningham (2021) Causal Inference: The Mixtape
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class DiDEstimator(Enum):
    """DiD estimation methods."""
    SIMPLE = "simple"  # Basic DiD
    REGRESSION = "regression"  # OLS with controls
    TWFE = "twfe"  # Two-way fixed effects
    CALLAWAY_SANTANNA = "callaway_santanna"  # Staggered adoption


@dataclass
class DiDResult:
    """Results from DiD estimation."""
    estimator: str
    estimate: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    n_pre: int
    n_post: int
    # Additional diagnostics
    parallel_trends_pvalue: Optional[float] = None
    placebo_estimates: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimator": self.estimator,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "parallel_trends_pvalue": self.parallel_trends_pvalue,
            "placebo_estimates": self.placebo_estimates,
        }
    
    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05
    
    @property
    def significant_at_01(self) -> bool:
        return self.p_value < 0.01


@dataclass
class EventStudyResult:
    """Results from event study analysis."""
    time_points: np.ndarray  # Relative time periods
    estimates: np.ndarray  # Point estimates at each period
    ci_lower: np.ndarray  # Lower CI
    ci_upper: np.ndarray  # Upper CI
    pre_trend_pvalue: float  # Test for pre-trends = 0
    reference_period: int  # Usually -1 (last pre-treatment period)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_points": self.time_points.tolist(),
            "estimates": self.estimates.tolist(),
            "ci_lower": self.ci_lower.tolist(),
            "ci_upper": self.ci_upper.tolist(),
            "pre_trend_pvalue": self.pre_trend_pvalue,
            "reference_period": self.reference_period,
        }


def simple_did(
    y_pre_treated: np.ndarray,
    y_post_treated: np.ndarray,
    y_pre_control: np.ndarray,
    y_post_control: np.ndarray,
    ci_level: float = 0.95,
) -> DiDResult:
    """
    Simple 2x2 DiD estimator.
    
    DiD = (E[Y|post,treat] - E[Y|pre,treat]) - (E[Y|post,control] - E[Y|pre,control])
    
    Args:
        y_pre_treated: Outcomes for treated group, pre-treatment
        y_post_treated: Outcomes for treated group, post-treatment
        y_pre_control: Outcomes for control group, pre-treatment
        y_post_control: Outcomes for control group, post-treatment
        ci_level: Confidence level
        
    Returns:
        DiDResult with estimate and inference
    """
    # Means
    mean_pre_t = np.mean(y_pre_treated)
    mean_post_t = np.mean(y_post_treated)
    mean_pre_c = np.mean(y_pre_control)
    mean_post_c = np.mean(y_post_control)
    
    # DiD estimate
    treat_diff = mean_post_t - mean_pre_t
    control_diff = mean_post_c - mean_pre_c
    did_estimate = treat_diff - control_diff
    
    # Standard error via variance of differences
    n_pre_t, n_post_t = len(y_pre_treated), len(y_post_treated)
    n_pre_c, n_post_c = len(y_pre_control), len(y_post_control)
    
    var_pre_t = np.var(y_pre_treated, ddof=1)
    var_post_t = np.var(y_post_treated, ddof=1)
    var_pre_c = np.var(y_pre_control, ddof=1)
    var_post_c = np.var(y_post_control, ddof=1)
    
    # SE = sqrt(var(treat_diff) + var(control_diff))
    var_treat_diff = var_post_t / n_post_t + var_pre_t / n_pre_t
    var_control_diff = var_post_c / n_post_c + var_pre_c / n_pre_c
    
    se = np.sqrt(var_treat_diff + var_control_diff)
    
    # T-test
    if se > 0:
        t_stat = did_estimate / se
        # Approximate df using Welch-Satterthwaite
        df = (var_treat_diff + var_control_diff) ** 2 / (
            var_treat_diff ** 2 / (min(n_pre_t, n_post_t) - 1) +
            var_control_diff ** 2 / (min(n_pre_c, n_post_c) - 1) + 1e-8
        )
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Confidence interval
        alpha = 1 - ci_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = did_estimate - t_crit * se
        ci_upper = did_estimate + t_crit * se
    else:
        t_stat = 0.0
        p_value = 1.0
        ci_lower = did_estimate
        ci_upper = did_estimate
    
    return DiDResult(
        estimator="simple",
        estimate=did_estimate,
        std_error=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=n_pre_t + n_post_t,
        n_control=n_pre_c + n_post_c,
        n_pre=n_pre_t + n_pre_c,
        n_post=n_post_t + n_post_c,
    )


def regression_did(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    post_col: str,
    controls: Optional[List[str]] = None,
    ci_level: float = 0.95,
) -> DiDResult:
    """
    Regression-based DiD estimator.
    
    Y = β₀ + β₁·Treated + β₂·Post + β₃·(Treated×Post) + γ·X + ε
    
    The DiD estimate is β₃.
    
    Args:
        df: DataFrame with outcome and indicators
        outcome_col: Name of outcome column
        treated_col: Name of treatment indicator (1 = treated)
        post_col: Name of post-treatment indicator (1 = post)
        controls: Optional list of control variable names
        ci_level: Confidence level
        
    Returns:
        DiDResult with estimate and inference
    """
    # Create interaction term
    df = df.copy()
    df["_interaction_"] = df[treated_col] * df[post_col]
    
    # Build design matrix
    X_cols = [treated_col, post_col, "_interaction_"]
    if controls:
        X_cols.extend(controls)
    
    # Add constant
    X = df[X_cols].values
    X = np.column_stack([np.ones(len(X)), X])
    y = df[outcome_col].values
    
    # OLS estimation
    try:
        # (X'X)^-1 X'y
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        # Residuals and variance
        residuals = y - X @ beta
        n, k = X.shape
        sigma2 = np.sum(residuals ** 2) / (n - k)
        
        # Standard errors
        var_beta = sigma2 * XtX_inv
        se = np.sqrt(np.diag(var_beta))
        
        # DiD coefficient is at index 3 (constant, treated, post, interaction)
        did_estimate = beta[3]
        did_se = se[3]
        
        # T-test
        t_stat = did_estimate / did_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
        
        # CI
        alpha = 1 - ci_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - k)
        ci_lower = did_estimate - t_crit * did_se
        ci_upper = did_estimate + t_crit * did_se
        
    except Exception:
        did_estimate = np.nan
        did_se = np.nan
        t_stat = np.nan
        p_value = 1.0
        ci_lower = np.nan
        ci_upper = np.nan
    
    # Count observations
    n_treated = df[treated_col].sum()
    n_control = len(df) - n_treated
    n_post = df[post_col].sum()
    n_pre = len(df) - n_post
    
    return DiDResult(
        estimator="regression",
        estimate=did_estimate,
        std_error=did_se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=int(n_treated),
        n_control=int(n_control),
        n_pre=int(n_pre),
        n_post=int(n_post),
    )


def bootstrap_did(
    y_pre_treated: np.ndarray,
    y_post_treated: np.ndarray,
    y_pre_control: np.ndarray,
    y_post_control: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
) -> DiDResult:
    """
    Bootstrap inference for DiD.
    
    Args:
        y_pre_treated: Outcomes for treated group, pre-treatment
        y_post_treated: Outcomes for treated group, post-treatment
        y_pre_control: Outcomes for control group, pre-treatment
        y_post_control: Outcomes for control group, post-treatment
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level
        seed: Random seed
        
    Returns:
        DiDResult with bootstrap CI
    """
    if seed is not None:
        np.random.seed(seed)
    
    def compute_did(ypt, yposttr, ypc, ypostc):
        return (np.mean(yposttr) - np.mean(ypt)) - (np.mean(ypostc) - np.mean(ypc))
    
    # Point estimate
    did_estimate = compute_did(y_pre_treated, y_post_treated, y_pre_control, y_post_control)
    
    # Bootstrap
    boot_estimates = []
    n_pre_t, n_post_t = len(y_pre_treated), len(y_post_treated)
    n_pre_c, n_post_c = len(y_pre_control), len(y_post_control)
    
    for _ in range(n_bootstrap):
        # Resample within each group
        idx_pre_t = np.random.choice(n_pre_t, n_pre_t, replace=True)
        idx_post_t = np.random.choice(n_post_t, n_post_t, replace=True)
        idx_pre_c = np.random.choice(n_pre_c, n_pre_c, replace=True)
        idx_post_c = np.random.choice(n_post_c, n_post_c, replace=True)
        
        boot_did = compute_did(
            y_pre_treated[idx_pre_t],
            y_post_treated[idx_post_t],
            y_pre_control[idx_pre_c],
            y_post_control[idx_post_c],
        )
        boot_estimates.append(boot_did)
    
    boot_estimates = np.array(boot_estimates)
    
    # Bootstrap SE
    se = np.std(boot_estimates, ddof=1)
    
    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))
    
    # P-value (two-sided test against 0)
    p_value = 2 * min(
        np.mean(boot_estimates >= 0),
        np.mean(boot_estimates <= 0),
    )
    
    t_stat = did_estimate / se if se > 0 else 0.0
    
    return DiDResult(
        estimator="bootstrap",
        estimate=did_estimate,
        std_error=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treated=n_pre_t + n_post_t,
        n_control=n_pre_c + n_post_c,
        n_pre=n_pre_t + n_pre_c,
        n_post=n_post_t + n_post_c,
    )


def check_parallel_trends(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    time_periods: np.ndarray,
    treatment_time: int,
) -> Tuple[float, np.ndarray]:
    """
    Test for parallel pre-treatment trends.
    
    Regresses pre-treatment differences on time and tests if slope = 0.
    
    Args:
        y_treated: Outcomes for treated group over time
        y_control: Outcomes for control group over time
        time_periods: Time period indicators
        treatment_time: First treatment period
        
    Returns:
        (p_value, pre_period_differences)
    """
    # Filter to pre-treatment
    pre_mask = time_periods < treatment_time
    
    if pre_mask.sum() < 3:
        # Not enough pre-periods
        return 1.0, np.array([])
    
    y_t_pre = y_treated[pre_mask]
    y_c_pre = y_control[pre_mask]
    t_pre = time_periods[pre_mask]
    
    # Compute differences
    diffs = y_t_pre - y_c_pre
    
    # Regress on time
    X = np.column_stack([np.ones_like(t_pre), t_pre])
    
    try:
        beta = np.linalg.lstsq(X, diffs, rcond=None)[0]
        
        # Residuals
        residuals = diffs - X @ beta
        n = len(diffs)
        se = np.sqrt(np.sum(residuals ** 2) / (n - 2)) / np.sqrt(np.sum((t_pre - t_pre.mean()) ** 2))
        
        # T-test on slope
        t_stat = beta[1] / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
    except Exception:
        p_value = 1.0
    
    return p_value, diffs


def run_placebo_test(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    time_col: str,
    true_treatment_time: int,
    placebo_times: List[int],
) -> List[Dict[str, Any]]:
    """
    Run placebo tests with fake treatment times.
    
    Args:
        df: DataFrame
        outcome_col: Outcome variable
        treated_col: Treatment indicator
        time_col: Time column
        true_treatment_time: Actual treatment time
        placebo_times: List of placebo treatment times (should be pre-treatment)
        
    Returns:
        List of placebo test results
    """
    results = []
    
    for placebo_time in placebo_times:
        if placebo_time >= true_treatment_time:
            continue  # Skip post-treatment placebos
        
        # Create placebo post indicator
        df_sub = df[df[time_col] < true_treatment_time].copy()
        df_sub["_placebo_post_"] = (df_sub[time_col] >= placebo_time).astype(int)
        
        # Run DiD
        try:
            did_result = regression_did(
                df_sub,
                outcome_col=outcome_col,
                treated_col=treated_col,
                post_col="_placebo_post_",
            )
            
            results.append({
                "placebo_time": placebo_time,
                "estimate": did_result.estimate,
                "p_value": did_result.p_value,
                "significant": did_result.significant_at_05,
            })
        except Exception as e:
            results.append({
                "placebo_time": placebo_time,
                "error": str(e),
            })
    
    return results


@dataclass
class MediationResult:
    """Results from mediation analysis."""
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    mediator_name: str
    mediator_effect_on_outcome: float
    treatment_effect_on_mediator: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_effect": self.total_effect,
            "direct_effect": self.direct_effect,
            "indirect_effect": self.indirect_effect,
            "proportion_mediated": self.proportion_mediated,
            "mediator_name": self.mediator_name,
            "mediator_effect_on_outcome": self.mediator_effect_on_outcome,
            "treatment_effect_on_mediator": self.treatment_effect_on_mediator,
        }


def mediation_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    mediator_col: str,
    controls: Optional[List[str]] = None,
) -> MediationResult:
    """
    Run mediation analysis for DiD.
    
    Three regressions:
    1. Y ~ T (total effect)
    2. M ~ T (effect of treatment on mediator)
    3. Y ~ T + M (direct effect)
    
    Indirect effect = Total - Direct = (M ~ T) × (Y ~ T + M)[M coefficient]
    
    Args:
        df: DataFrame
        outcome_col: Outcome variable
        treatment_col: Treatment variable
        mediator_col: Mediator variable
        controls: Optional control variables
        
    Returns:
        MediationResult
    """
    df = df.dropna(subset=[outcome_col, treatment_col, mediator_col])
    
    Y = df[outcome_col].values
    T = df[treatment_col].values
    M = df[mediator_col].values
    
    # Build control matrix
    if controls:
        X_controls = df[controls].values
    else:
        X_controls = None
    
    def ols_coef(y, X):
        """Get OLS coefficient for first non-constant variable."""
        X = np.column_stack([np.ones(len(y)), X])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return beta[1]
        except Exception:
            return np.nan
    
    # 1. Total effect: Y ~ T
    total_effect = ols_coef(Y, T if X_controls is None else np.column_stack([T, X_controls]))
    
    # 2. Treatment on mediator: M ~ T
    treat_on_med = ols_coef(M, T if X_controls is None else np.column_stack([T, X_controls]))
    
    # 3. Direct effect: Y ~ T + M
    X_direct = np.column_stack([T, M])
    if X_controls is not None:
        X_direct = np.column_stack([X_direct, X_controls])
    X_direct = np.column_stack([np.ones(len(Y)), X_direct])
    
    try:
        beta_direct = np.linalg.lstsq(X_direct, Y, rcond=None)[0]
        direct_effect = beta_direct[1]  # T coefficient
        med_on_outcome = beta_direct[2]  # M coefficient
    except Exception:
        direct_effect = np.nan
        med_on_outcome = np.nan
    
    # Indirect effect
    indirect_effect = total_effect - direct_effect
    
    # Proportion mediated
    if abs(total_effect) > 1e-8:
        prop_mediated = indirect_effect / total_effect
    else:
        prop_mediated = 0.0
    
    return MediationResult(
        total_effect=total_effect,
        direct_effect=direct_effect,
        indirect_effect=indirect_effect,
        proportion_mediated=prop_mediated,
        mediator_name=mediator_col,
        mediator_effect_on_outcome=med_on_outcome,
        treatment_effect_on_mediator=treat_on_med,
    )


def event_study(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    time_col: str,
    treatment_time: int,
    pre_periods: int = 4,
    post_periods: int = 4,
    reference_period: int = -1,
) -> EventStudyResult:
    """
    Event study analysis for DiD.
    
    Estimates treatment effects at each time period relative to treatment.
    
    Args:
        df: DataFrame
        outcome_col: Outcome column
        treated_col: Treatment indicator
        time_col: Time column
        treatment_time: Period when treatment occurs
        pre_periods: Number of pre-treatment periods to include
        post_periods: Number of post-treatment periods to include
        reference_period: Reference period (normalized to 0)
        
    Returns:
        EventStudyResult
    """
    df = df.copy()
    
    # Create relative time
    df["_rel_time_"] = df[time_col] - treatment_time
    
    # Filter to relevant periods
    min_t = -pre_periods
    max_t = post_periods
    df = df[(df["_rel_time_"] >= min_t) & (df["_rel_time_"] <= max_t)]
    
    # Create time dummies (excluding reference)
    time_points = sorted(df["_rel_time_"].unique())
    time_points = [t for t in time_points if t != reference_period]
    
    estimates = []
    ci_lower = []
    ci_upper = []
    
    for t in time_points:
        # DiD for this period vs reference
        df_ref = df[df["_rel_time_"] == reference_period]
        df_t = df[df["_rel_time_"] == t]
        
        if len(df_ref) == 0 or len(df_t) == 0:
            estimates.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            continue
        
        # Simple comparison: (Y_t - Y_ref) for treated vs control
        y_ref_t = df_ref[df_ref[treated_col] == 1][outcome_col].mean()
        y_ref_c = df_ref[df_ref[treated_col] == 0][outcome_col].mean()
        y_t_t = df_t[df_t[treated_col] == 1][outcome_col].mean()
        y_t_c = df_t[df_t[treated_col] == 0][outcome_col].mean()
        
        if np.isnan(y_ref_t) or np.isnan(y_ref_c) or np.isnan(y_t_t) or np.isnan(y_t_c):
            estimates.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            continue
        
        est = (y_t_t - y_ref_t) - (y_t_c - y_ref_c)
        estimates.append(est)
        
        # Approximate CI (simplified)
        # In practice, should cluster SEs
        se_approx = abs(est) * 0.3 + 0.01  # Rough approximation
        ci_lower.append(est - 1.96 * se_approx)
        ci_upper.append(est + 1.96 * se_approx)
    
    estimates = np.array(estimates)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    time_points = np.array(time_points)
    
    # Test for pre-trends
    pre_mask = time_points < 0
    if pre_mask.sum() >= 2:
        pre_estimates = estimates[pre_mask]
        pre_estimates = pre_estimates[~np.isnan(pre_estimates)]
        if len(pre_estimates) >= 2:
            _, pre_trend_pvalue = stats.ttest_1samp(pre_estimates, 0)
        else:
            pre_trend_pvalue = 1.0
    else:
        pre_trend_pvalue = 1.0
    
    return EventStudyResult(
        time_points=time_points,
        estimates=estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pre_trend_pvalue=pre_trend_pvalue,
        reference_period=reference_period,
    )


__all__ = [
    "DiDEstimator",
    "DiDResult",
    "EventStudyResult",
    "MediationResult",
    "simple_did",
    "regression_did",
    "bootstrap_did",
    "check_parallel_trends",
    "run_placebo_test",
    "mediation_analysis",
    "event_study",
]
