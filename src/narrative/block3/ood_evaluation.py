#!/usr/bin/env python3
"""
OOD (Out-of-Distribution) Evaluation Module for Block 3.

Provides:
- Year-shift splits (train on pre-2024, test on 2024+)
- Sector-shift splits (train on tech, test on biotech)
- Significance tests (paired t-test, Wilcoxon, bootstrap)
- OOD degradation metrics

This module implements Task 3: EDGAR-conditioned OOD Robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class OODShiftType(Enum):
    """Type of distribution shift."""
    YEAR_SHIFT = "year_shift"           # Temporal shift
    SECTOR_SHIFT = "sector_shift"       # Industry/sector shift
    SIZE_SHIFT = "size_shift"           # Company size shift
    COVARIATE_SHIFT = "covariate_shift" # Feature distribution shift


@dataclass
class OODSplitConfig:
    """Configuration for OOD split."""
    shift_type: OODShiftType
    train_condition: Dict[str, Any]
    test_condition: Dict[str, Any]
    description: str = ""


@dataclass
class OODSplitResult:
    """Result of OOD split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    n_train: int
    n_val: int
    n_test: int
    shift_type: OODShiftType
    description: str


# =============================================================================
# OOD Split Functions
# =============================================================================

def create_year_shift_split(
    df: pd.DataFrame,
    date_col: str = "crawled_date_day",
    train_end_year: int = 2023,
    val_year: int = 2024,
    test_start_year: int = 2025,
) -> OODSplitResult:
    """
    Create year-shift OOD split.
    
    Train on data up to train_end_year, validate on val_year,
    test on test_start_year+.
    
    Args:
        df: DataFrame with date column
        date_col: Column containing dates
        train_end_year: Last year of training data
        val_year: Validation year
        test_start_year: First year of test data
    
    Returns:
        OODSplitResult with indices
    """
    dates = pd.to_datetime(df[date_col])
    years = dates.dt.year
    
    train_mask = years <= train_end_year
    val_mask = years == val_year
    test_mask = years >= test_start_year
    
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    
    return OODSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        shift_type=OODShiftType.YEAR_SHIFT,
        description=f"Train: ≤{train_end_year}, Val: {val_year}, Test: ≥{test_start_year}",
    )


def create_sector_shift_split(
    df: pd.DataFrame,
    sector_col: str = "industry",
    train_sectors: List[str] = None,
    test_sectors: List[str] = None,
    val_frac: float = 0.2,
    seed: int = 42,
) -> OODSplitResult:
    """
    Create sector-shift OOD split.
    
    Train on train_sectors, test on test_sectors (different industries).
    
    Args:
        df: DataFrame with sector column
        sector_col: Column containing sector/industry
        train_sectors: Sectors for training (None = auto-detect majority)
        test_sectors: Sectors for testing (None = auto-detect minority)
        val_frac: Fraction of training data for validation
        seed: Random seed
    
    Returns:
        OODSplitResult with indices
    """
    rng = np.random.RandomState(seed)
    
    # Auto-detect sectors if not provided
    if train_sectors is None or test_sectors is None:
        sector_counts = df[sector_col].value_counts()
        all_sectors = sector_counts.index.tolist()
        
        # Use top sectors for training, bottom for testing
        n_train_sectors = max(1, len(all_sectors) // 2)
        train_sectors = all_sectors[:n_train_sectors]
        test_sectors = all_sectors[n_train_sectors:]
    
    train_test_mask = df[sector_col].isin(train_sectors)
    test_mask = df[sector_col].isin(test_sectors)
    
    train_pool = np.where(train_test_mask)[0]
    
    # Split train pool into train/val
    rng.shuffle(train_pool)
    val_size = int(len(train_pool) * val_frac)
    val_idx = train_pool[:val_size]
    train_idx = train_pool[val_size:]
    
    test_idx = np.where(test_mask)[0]
    
    return OODSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        shift_type=OODShiftType.SECTOR_SHIFT,
        description=f"Train sectors: {train_sectors}, Test sectors: {test_sectors}",
    )


def create_size_shift_split(
    df: pd.DataFrame,
    size_col: str = "funding_goal_usd",
    train_percentile: Tuple[float, float] = (0, 75),
    test_percentile: Tuple[float, float] = (75, 100),
    val_frac: float = 0.2,
    seed: int = 42,
) -> OODSplitResult:
    """
    Create size-shift OOD split (train on small, test on large).
    
    Args:
        df: DataFrame with size column
        size_col: Column containing size metric
        train_percentile: Percentile range for training (low, high)
        test_percentile: Percentile range for testing (low, high)
        val_frac: Fraction of training data for validation
        seed: Random seed
    
    Returns:
        OODSplitResult with indices
    """
    rng = np.random.RandomState(seed)
    
    sizes = df[size_col].values
    
    train_low = np.percentile(sizes, train_percentile[0])
    train_high = np.percentile(sizes, train_percentile[1])
    test_low = np.percentile(sizes, test_percentile[0])
    test_high = np.percentile(sizes, test_percentile[1])
    
    train_test_mask = (sizes >= train_low) & (sizes <= train_high)
    test_mask = (sizes >= test_low) & (sizes <= test_high)
    
    train_pool = np.where(train_test_mask)[0]
    
    # Split train pool into train/val
    rng.shuffle(train_pool)
    val_size = int(len(train_pool) * val_frac)
    val_idx = train_pool[:val_size]
    train_idx = train_pool[val_size:]
    
    test_idx = np.where(test_mask)[0]
    
    return OODSplitResult(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        shift_type=OODShiftType.SIZE_SHIFT,
        description=f"Train: P{train_percentile[0]}-P{train_percentile[1]}, Test: P{test_percentile[0]}-P{test_percentile[1]}",
    )


# =============================================================================
# Significance Tests
# =============================================================================

@dataclass
class SignificanceResult:
    """Result of significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant_at_05: bool
    significant_at_01: bool
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def paired_t_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """
    Paired t-test for comparing two models.
    
    Args:
        errors_a: Errors from model A
        errors_b: Errors from model B
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        SignificanceResult
    """
    stat, p_value = stats.ttest_rel(errors_a, errors_b, alternative=alternative)
    
    # Effect size (Cohen's d)
    diff = errors_a - errors_b
    std_diff = np.std(diff, ddof=1)
    if std_diff > 0:
        effect_size = np.mean(diff) / std_diff
    else:
        # Zero variance: identical errors -> no effect
        effect_size = 0.0
    
    return SignificanceResult(
        test_name="Paired t-test",
        statistic=stat,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        effect_size=effect_size,
    )


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = "two-sided",
) -> SignificanceResult:
    """
    Wilcoxon signed-rank test (non-parametric).
    
    Args:
        errors_a: Errors from model A
        errors_b: Errors from model B
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        SignificanceResult
    """
    # Remove zero differences
    diff = errors_a - errors_b
    nonzero_mask = diff != 0
    
    if np.sum(nonzero_mask) < 10:
        return SignificanceResult(
            test_name="Wilcoxon signed-rank",
            statistic=np.nan,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
        )
    
    stat, p_value = stats.wilcoxon(
        errors_a[nonzero_mask],
        errors_b[nonzero_mask],
        alternative=alternative,
    )
    
    return SignificanceResult(
        test_name="Wilcoxon signed-rank",
        statistic=stat,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
    )


def bootstrap_comparison(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> SignificanceResult:
    """
    Bootstrap test for comparing two models.
    
    Args:
        errors_a: Errors from model A
        errors_b: Errors from model B
        metric_fn: Function to compute metric from errors
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI
        seed: Random seed
    
    Returns:
        SignificanceResult with CI for difference
    """
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    
    # Point estimate
    diff_point = metric_fn(errors_a) - metric_fn(errors_b)
    
    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = metric_fn(errors_a[idx]) - metric_fn(errors_b[idx])
        diffs.append(diff)
    
    diffs = np.array(diffs)
    
    # CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    
    # P-value (two-sided)
    p_value = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
    
    return SignificanceResult(
        test_name=f"Bootstrap ({n_bootstrap} samples)",
        statistic=diff_point,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> SignificanceResult:
    """
    Diebold-Mariano test for forecast comparison.
    
    Args:
        errors_a: Forecast errors from model A
        errors_b: Forecast errors from model B
        h: Forecast horizon
        power: Loss function power (1=MAE, 2=MSE)
    
    Returns:
        SignificanceResult
    """
    # Loss differential
    d = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    
    # Mean and variance of d
    d_mean = np.mean(d)
    T = len(d)
    
    # Autocovariance
    gamma_0 = np.var(d, ddof=1)
    gamma = [np.cov(d[:-k], d[k:])[0, 1] if k > 0 else gamma_0 for k in range(h)]
    
    # Long-run variance
    var_d = gamma_0 + 2 * sum(gamma[1:h])
    
    if var_d <= 0:
        return SignificanceResult(
            test_name="Diebold-Mariano",
            statistic=np.nan,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
        )
    
    # DM statistic
    dm_stat = d_mean / np.sqrt(var_d / T)
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return SignificanceResult(
        test_name="Diebold-Mariano",
        statistic=dm_stat,
        p_value=p_value,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
    )


# =============================================================================
# OOD Degradation Metrics
# =============================================================================

@dataclass
class OODDegradation:
    """Metrics quantifying OOD performance degradation."""
    iid_metric: float
    ood_metric: float
    absolute_degradation: float
    relative_degradation_pct: float
    metric_name: str


def compute_ood_degradation(
    iid_metric: float,
    ood_metric: float,
    metric_name: str = "MAE",
    higher_is_better: bool = False,
) -> OODDegradation:
    """
    Compute OOD degradation metrics.
    
    Args:
        iid_metric: Metric on in-distribution test set
        ood_metric: Metric on OOD test set
        metric_name: Name of the metric
        higher_is_better: Whether higher metric is better
    
    Returns:
        OODDegradation with absolute and relative degradation
    """
    if higher_is_better:
        # Higher is better -> degradation = iid - ood (positive = worse OOD)
        absolute_degradation = iid_metric - ood_metric
    else:
        # Lower is better -> degradation = ood - iid (positive = worse OOD)
        absolute_degradation = ood_metric - iid_metric
    
    # Relative degradation
    if abs(iid_metric) > 1e-8:
        relative_degradation_pct = (absolute_degradation / abs(iid_metric)) * 100
    else:
        relative_degradation_pct = 0.0
    
    return OODDegradation(
        iid_metric=iid_metric,
        ood_metric=ood_metric,
        absolute_degradation=absolute_degradation,
        relative_degradation_pct=relative_degradation_pct,
        metric_name=metric_name,
    )


def compute_ood_robustness_score(
    degradations: List[OODDegradation],
    threshold_pct: float = 20.0,
) -> float:
    """
    Compute aggregate OOD robustness score.
    
    Score is 1.0 if degradation <= threshold, decreasing linearly to 0.
    
    Args:
        degradations: List of OOD degradation results
        threshold_pct: Threshold for acceptable degradation (%)
    
    Returns:
        Robustness score between 0 and 1
    """
    scores = []
    for deg in degradations:
        # Clamp degradation to [0, 2*threshold]
        clamped = max(0, min(deg.relative_degradation_pct, 2 * threshold_pct))
        score = max(0, 1 - clamped / (2 * threshold_pct))
        scores.append(score)
    
    return float(np.mean(scores))


__all__ = [
    # Enums and configs
    "OODShiftType",
    "OODSplitConfig",
    "OODSplitResult",
    # Split functions
    "create_year_shift_split",
    "create_sector_shift_split",
    "create_size_shift_split",
    # Significance tests
    "SignificanceResult",
    "paired_t_test",
    "wilcoxon_test",
    "bootstrap_comparison",
    "diebold_mariano_test",
    # Degradation metrics
    "OODDegradation",
    "compute_ood_degradation",
    "compute_ood_robustness_score",
]
