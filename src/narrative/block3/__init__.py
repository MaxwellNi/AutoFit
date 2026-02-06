"""
Block 3 Benchmark Module.

Provides unified interfaces for:
- Task protocols with leakage validation
- Preprocessing (missing values, scaling)
- Metrics with bootstrap confidence intervals
- OOD evaluation with significance tests
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

# OOD Evaluation
from .ood_evaluation import (
    OODShiftType,
    create_year_shift_split,
    create_sector_shift_split,
    create_size_shift_split,
    SignificanceResult,
    paired_t_test,
    wilcoxon_test,
    bootstrap_comparison,
    diebold_mariano_test,
    OODDegradation,
    compute_ood_degradation,
    compute_ood_robustness_score,
)

# DiD Analysis
from .did_analysis import (
    DiDEstimator,
    DiDResult,
    EventStudyResult,
    MediationResult,
    simple_did,
    regression_did,
    bootstrap_did,
    check_parallel_trends,
    run_placebo_test,
    mediation_analysis,
    event_study,
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
    # OOD Evaluation
    "OODShiftType",
    "create_year_shift_split",
    "create_sector_shift_split",
    "create_size_shift_split",
    "SignificanceResult",
    "paired_t_test",
    "wilcoxon_test",
    "bootstrap_comparison",
    "diebold_mariano_test",
    "OODDegradation",
    "compute_ood_degradation",
    "compute_ood_robustness_score",
    # DiD Analysis
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
