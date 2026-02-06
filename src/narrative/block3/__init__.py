"""
Block 3 Benchmark Module.

Provides unified interfaces for:
- Task protocols with leakage validation
- Preprocessing (missing values, scaling)
- Metrics with bootstrap confidence intervals
"""
from . import tasks
from . import models

# Protocol
from .protocol import (
    TaskType,
    FeatureAvailability,
    SplitPolicy,
    TaskProtocol,
    LeakageValidator,
    get_protocol,
    TASK_PROTOCOLS,
)

# Preprocessing
from .preprocessing import (
    set_global_seed,
    get_seed_sequence,
    MissingPolicy,
    get_missing_policy,
    ScalingPolicy,
    FeatureScaler,
    PreprocessConfig,
    UnifiedPreprocessor,
)

# Metrics
from .metrics import (
    rmse,
    mae,
    mape,
    smape,
    mse,
    r2_score,
    accuracy,
    precision,
    recall,
    f1_score,
    auroc,
    crps_sample,
    BootstrapCI,
    bootstrap_ci,
    METRIC_REGISTRY,
    get_metrics_for_task,
    compute_all_metrics,
)

__all__ = [
    # Submodules
    "tasks",
    "models",
    # Protocol
    "TaskType",
    "FeatureAvailability",
    "SplitPolicy",
    "TaskProtocol",
    "LeakageValidator",
    "get_protocol",
    "TASK_PROTOCOLS",
    # Preprocessing
    "set_global_seed",
    "get_seed_sequence",
    "MissingPolicy",
    "get_missing_policy",
    "ScalingPolicy",
    "FeatureScaler",
    "PreprocessConfig",
    "UnifiedPreprocessor",
    # Metrics
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
    "crps_sample",
    "BootstrapCI",
    "bootstrap_ci",
    "METRIC_REGISTRY",
    "get_metrics_for_task",
    "compute_all_metrics",
]
