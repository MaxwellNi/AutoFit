#!/usr/bin/env python3
"""
Unified Metrics Module for Block 3.

Provides:
- Standard regression/classification metrics
- Bootstrap confidence intervals
- Proper CRPS for probabilistic forecasts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Metric Functions
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> float:
    """Precision for binary classification."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    pp = np.sum(y_pred == pos_label)
    return float(tp / pp) if pp > 0 else 0.0


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> float:
    """Recall for binary classification."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    p = np.sum(y_true == pos_label)
    return float(tp / p) if p > 0 else 0.0


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> float:
    """F1 score for binary classification."""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area Under ROC Curve (simple Mann-Whitney U approximation)."""
    # Sort by probability
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    
    # Calculate AUC
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Rank sum method
    ranks = np.arange(1, len(y_sorted) + 1)
    pos_rank_sum = np.sum(ranks[y_sorted == 1])
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(1 - auc)  # Adjust for descending order


# =============================================================================
# Probabilistic Metrics
# =============================================================================

def crps_sample(
    y_true: np.ndarray,
    y_samples: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score (sample-based).
    
    Args:
        y_true: True values, shape (n,)
        y_samples: Predicted samples, shape (n, n_samples)
    
    Returns:
        Mean CRPS across all observations
    """
    n, n_samples = y_samples.shape
    crps_values = []
    
    for i in range(n):
        samples = y_samples[i]
        obs = y_true[i]
        
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        term1 = np.mean(np.abs(samples - obs))
        term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
        crps_values.append(term1 - 0.5 * term2)
    
    return float(np.mean(crps_values))


def crps_gaussian(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score (Gaussian).
    
    Args:
        y_true: True values, shape (n,)
        y_mean: Predicted means, shape (n,)
        y_std: Predicted standard deviations, shape (n,)
    
    Returns:
        Mean CRPS across all observations
    """
    from scipy.stats import norm
    
    z = (y_true - y_mean) / (y_std + 1e-8)
    crps = y_std * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps))


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Pinball loss for quantile regression.
    
    Args:
        y_true: True values
        y_pred: Predicted quantile values
        quantile: Quantile level (0-1)
    
    Returns:
        Mean pinball loss
    """
    diff = y_true - y_pred
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Coverage of prediction intervals.
    
    Args:
        y_true: True values
        y_lower: Lower bound of interval
        y_upper: Upper bound of interval
    
    Returns:
        Fraction of observations within intervals
    """
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(in_interval))


def interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Mean width of prediction intervals."""
    return float(np.mean(y_upper - y_lower))


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    confidence_level: float


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_fn: Metric function (y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
    
    Returns:
        BootstrapCI with point estimate and confidence interval
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)
    
    # Bootstrap samples
    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_value = metric_fn(y_true[idx], y_pred[idx])
        bootstrap_values.append(boot_value)
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
    
    # Standard error
    std_error = np.std(bootstrap_values)
    
    return BootstrapCI(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


# =============================================================================
# Metric Registry
# =============================================================================

class MetricType(Enum):
    """Type of metric."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PROBABILISTIC = "probabilistic"


@dataclass
class MetricSpec:
    """Specification for a metric."""
    name: str
    fn: Callable
    metric_type: MetricType
    lower_is_better: bool = True
    requires_proba: bool = False


METRIC_REGISTRY: Dict[str, MetricSpec] = {
    # Regression metrics
    "rmse": MetricSpec("RMSE", rmse, MetricType.REGRESSION, lower_is_better=True),
    "mae": MetricSpec("MAE", mae, MetricType.REGRESSION, lower_is_better=True),
    "mape": MetricSpec("MAPE", mape, MetricType.REGRESSION, lower_is_better=True),
    "smape": MetricSpec("sMAPE", smape, MetricType.REGRESSION, lower_is_better=True),
    "mse": MetricSpec("MSE", mse, MetricType.REGRESSION, lower_is_better=True),
    "r2": MetricSpec("R²", r2_score, MetricType.REGRESSION, lower_is_better=False),
    
    # Classification metrics
    "accuracy": MetricSpec(
        "Accuracy", accuracy, MetricType.CLASSIFICATION, lower_is_better=False
    ),
    "precision": MetricSpec(
        "Precision", precision, MetricType.CLASSIFICATION, lower_is_better=False
    ),
    "recall": MetricSpec(
        "Recall", recall, MetricType.CLASSIFICATION, lower_is_better=False
    ),
    "f1": MetricSpec(
        "F1", f1_score, MetricType.CLASSIFICATION, lower_is_better=False
    ),
    "auroc": MetricSpec(
        "AUROC", auroc, MetricType.CLASSIFICATION,
        lower_is_better=False, requires_proba=True
    ),
    
    # Probabilistic metrics
    "crps": MetricSpec(
        "CRPS", crps_sample, MetricType.PROBABILISTIC, lower_is_better=True
    ),
    "crps_gaussian": MetricSpec(
        "CRPS (Gaussian)", crps_gaussian, MetricType.PROBABILISTIC, lower_is_better=True
    ),
}


def get_metrics_for_task(task_type: str) -> List[str]:
    """Get appropriate metrics for a task type."""
    if task_type in ("forecast", "outcome"):
        return ["rmse", "mae", "mape", "smape"]
    elif task_type == "classification":
        return ["accuracy", "precision", "recall", "f1", "auroc"]
    elif task_type == "probabilistic":
        return ["crps", "coverage", "interval_width"]
    else:
        return ["rmse", "mae"]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: List[str],
    with_ci: bool = False,
    ci_seed: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute multiple metrics at once.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_names: List of metric names to compute
        with_ci: Whether to compute bootstrap CIs
        ci_seed: Random seed for bootstrap
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dict mapping metric name to value (or BootstrapCI if with_ci=True)
    """
    results = {}
    
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            continue
        
        spec = METRIC_REGISTRY[name]
        
        if with_ci:
            ci = bootstrap_ci(
                y_true, y_pred, spec.fn,
                n_bootstrap=n_bootstrap, seed=ci_seed
            )
            results[name] = ci
        else:
            results[name] = spec.fn(y_true, y_pred)
    
    return results


def format_metric_with_ci(metric_name: str, ci: BootstrapCI) -> str:
    """Format metric with CI for display."""
    spec = METRIC_REGISTRY.get(metric_name)
    if spec is None:
        name = metric_name
    else:
        name = spec.name
    
    return f"{name}: {ci.point_estimate:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"


__all__ = [
    # Basic metrics
    "rmse",
    "mae",
    "mape",
    "smape",
    "mse",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "auroc",
    # Probabilistic
    "crps_sample",
    "crps_gaussian",
    "pinball_loss",
    "coverage",
    "interval_width",
    # Bootstrap
    "BootstrapCI",
    "bootstrap_ci",
    # Registry
    "MetricType",
    "MetricSpec",
    "METRIC_REGISTRY",
    "get_metrics_for_task",
    "compute_all_metrics",
    "format_metric_with_ci",
]
