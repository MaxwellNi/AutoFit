#!/usr/bin/env python3
"""
Block 3 Task Protocol.

Defines the unified protocol for all benchmark tasks with:
- Input availability rules (static, early-K snapshots, optional edgar/text)
- Target definitions and metrics
- Anti-leakage validation
- Split policies

This module enforces the "no leakage" constraint across all tasks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class FeatureAvailability(Enum):
    """When a feature is available relative to observation time."""
    STATIC = "static"          # Known at entity creation (e.g., industry, location)
    EARLY_K = "early_k"        # Available after K days from launch
    HISTORICAL = "historical"  # All history up to observation time
    EDGAR_SYNC = "edgar_sync"  # EDGAR filing sync (may lag)
    TEXT = "text"              # Text features from offering materials
    DERIVED = "derived"        # Computed from other features (must validate sources)


class TaskType(Enum):
    """Type of prediction task."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    FORECASTING = "forecasting"
    CAUSAL = "causal"  # For DiD/mediation


@dataclass
class FeatureSpec:
    """Specification for a feature column."""
    name: str
    availability: FeatureAvailability
    source: str  # "offers_core", "edgar", "text", "derived"
    description: str = ""
    # For derived features, list source columns
    derived_from: List[str] = field(default_factory=list)


@dataclass
class TargetSpec:
    """Specification for a target variable."""
    name: str
    task_type: TaskType
    actual_column: str  # Actual column name in data
    description: str = ""
    # For classification: class labels
    classes: Optional[List[Any]] = None
    # For regression: expected range
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class MetricSpec:
    """Specification for an evaluation metric."""
    name: str
    function: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool
    requires_proba: bool = False  # Needs probability outputs
    description: str = ""


@dataclass
class SplitPolicy:
    """Policy for train/val/test splits."""
    # Temporal split boundaries
    train_end: Optional[date] = None
    val_end: Optional[date] = None
    test_start: Optional[date] = None
    
    # Gap between splits (anti-leakage)
    gap_days: int = 0
    
    # For K-fold or bootstrapping
    n_folds: int = 1
    random_seed: int = 42
    
    # For stratification
    stratify_by: Optional[str] = None


@dataclass
class TaskProtocol:
    """
    Complete protocol for a benchmark task.
    
    This defines everything needed to run a task reproducibly:
    - What features are available and when
    - What targets to predict
    - How to split data
    - How to evaluate
    """
    name: str
    description: str
    task_types: List[TaskType]
    
    # Feature availability
    allowed_features: Dict[str, FeatureSpec] = field(default_factory=dict)
    
    # Targets
    targets: Dict[str, TargetSpec] = field(default_factory=dict)
    
    # Horizons (interpretation depends on task)
    horizons: List[int] = field(default_factory=list)
    
    # Context lengths (for forecasting)
    context_lengths: List[int] = field(default_factory=list)
    
    # Ablation configurations
    ablations: List[str] = field(default_factory=list)
    
    # Split policy
    split_policy: SplitPolicy = field(default_factory=SplitPolicy)
    
    # Evaluation metrics
    metrics: Dict[str, MetricSpec] = field(default_factory=dict)
    
    # Additional constraints
    min_samples_per_entity: int = 1
    require_contiguous_dates: bool = False


class LeakageValidator:
    """
    Validates that features don't leak future information.
    
    Core principle: Features at observation time t can only use
    information from time <= t.
    """
    
    def __init__(self, protocol: TaskProtocol):
        self.protocol = protocol
        self.violations: List[Dict[str, Any]] = []
    
    def validate_feature_availability(
        self,
        df: pd.DataFrame,
        observation_date_col: str,
        feature_cols: List[str],
    ) -> bool:
        """
        Check that feature columns respect availability rules.
        
        Args:
            df: DataFrame with features
            observation_date_col: Column containing observation dates
            feature_cols: Columns to validate
        
        Returns:
            True if no leakage detected
        """
        self.violations = []
        
        for col in feature_cols:
            if col not in self.protocol.allowed_features:
                # Unknown feature - flag for review
                self.violations.append({
                    "type": "unknown_feature",
                    "column": col,
                    "message": f"Feature '{col}' not in allowed features list"
                })
                continue
            
            spec = self.protocol.allowed_features[col]
            
            # Check derived features recursively
            if spec.availability == FeatureAvailability.DERIVED:
                for source_col in spec.derived_from:
                    if source_col not in self.protocol.allowed_features:
                        self.violations.append({
                            "type": "derived_from_unknown",
                            "column": col,
                            "source": source_col,
                            "message": f"Derived feature '{col}' uses unknown source '{source_col}'"
                        })
        
        return len(self.violations) == 0
    
    def validate_temporal_split(
        self,
        train_dates: pd.Series,
        val_dates: pd.Series,
        test_dates: pd.Series,
    ) -> bool:
        """
        Validate that splits respect temporal ordering.
        
        Args:
            train_dates: Dates in training set
            val_dates: Dates in validation set
            test_dates: Dates in test set
        
        Returns:
            True if no temporal leakage
        """
        train_max = train_dates.max()
        val_min = val_dates.min()
        val_max = val_dates.max()
        test_min = test_dates.min()
        
        gap_days = self.protocol.split_policy.gap_days
        
        # Check train < val
        if train_max >= val_min:
            self.violations.append({
                "type": "temporal_overlap",
                "sets": ("train", "val"),
                "message": f"Train max ({train_max}) >= Val min ({val_min})"
            })
        
        # Check val < test
        if val_max >= test_min:
            self.violations.append({
                "type": "temporal_overlap",
                "sets": ("val", "test"),
                "message": f"Val max ({val_max}) >= Test min ({test_min})"
            })
        
        # Check gap
        if gap_days > 0:
            train_val_gap = (val_min - train_max).days
            if train_val_gap < gap_days:
                self.violations.append({
                    "type": "insufficient_gap",
                    "sets": ("train", "val"),
                    "actual_gap": train_val_gap,
                    "required_gap": gap_days,
                })
            
            val_test_gap = (test_min - val_max).days
            if val_test_gap < gap_days:
                self.violations.append({
                    "type": "insufficient_gap",
                    "sets": ("val", "test"),
                    "actual_gap": val_test_gap,
                    "required_gap": gap_days,
                })
        
        return len(self.violations) == 0
    
    def validate_target_not_in_features(
        self,
        feature_cols: List[str],
        target_cols: List[str],
    ) -> bool:
        """Ensure target columns are not accidentally used as features."""
        overlap = set(feature_cols) & set(target_cols)
        
        if overlap:
            self.violations.append({
                "type": "target_in_features",
                "columns": list(overlap),
                "message": f"Target columns used as features: {overlap}"
            })
        
        return len(overlap) == 0
    
    def get_report(self) -> str:
        """Generate human-readable validation report."""
        if not self.violations:
            return "✅ No leakage violations detected"
        
        lines = ["❌ Leakage violations detected:", ""]
        for v in self.violations:
            lines.append(f"  - [{v['type']}] {v.get('message', str(v))}")
        
        return "\n".join(lines)


# =============================================================================
# Standard Task Protocols
# =============================================================================

def create_task1_protocol() -> TaskProtocol:
    """Create protocol for Task 1: Outcome Prediction."""
    return TaskProtocol(
        name="task1_outcome",
        description="Predict final offering outcomes from early-K snapshot features",
        task_types=[TaskType.CLASSIFICATION, TaskType.REGRESSION, TaskType.RANKING],
        targets={
            "funding_raised_usd": TargetSpec(
                name="funding_raised_usd",
                task_type=TaskType.REGRESSION,
                actual_column="funding_raised_usd",
                description="Final amount raised in USD",
                min_value=0,
            ),
            "is_funded": TargetSpec(
                name="is_funded",
                task_type=TaskType.CLASSIFICATION,
                actual_column="is_funded",
                description="Whether offering reached funding goal",
                classes=[0, 1],
            ),
            "investors_count": TargetSpec(
                name="investors_count",
                task_type=TaskType.REGRESSION,
                actual_column="investors_count",
                description="Total number of investors",
                min_value=0,
            ),
        },
        horizons=[7, 14, 30, 60],  # K = days after launch
        ablations=["core_only", "+text", "+edgar", "full"],
        split_policy=SplitPolicy(
            train_end=date(2024, 6, 30),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            gap_days=7,
        ),
    )


def create_task2_protocol() -> TaskProtocol:
    """Create protocol for Task 2: Trajectory Forecasting."""
    return TaskProtocol(
        name="task2_forecast",
        description="Multi-horizon trajectory forecasting with quantile outputs",
        task_types=[TaskType.FORECASTING],
        targets={
            "funding_raised_usd": TargetSpec(
                name="funding_raised_usd",
                task_type=TaskType.FORECASTING,
                actual_column="funding_raised_usd",
                description="Cumulative funding trajectory",
            ),
            "investors_count": TargetSpec(
                name="investors_count",
                task_type=TaskType.FORECASTING,
                actual_column="investors_count",
                description="Investor count trajectory",
            ),
        },
        horizons=[7, 14, 30, 60],  # H = forecast horizon
        context_lengths=[30, 60, 90],
        ablations=["core_only", "+text", "+edgar", "full"],
        split_policy=SplitPolicy(
            train_end=date(2024, 6, 30),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            gap_days=0,  # For forecasting, gap is implicit in horizon
        ),
    )


def create_task3_protocol() -> TaskProtocol:
    """Create protocol for Task 3: EDGAR-conditioned OOD Robustness."""
    return TaskProtocol(
        name="task3_risk_adjust",
        description="EDGAR-conditioned OOD robustness with year/sector shift evaluation",
        task_types=[TaskType.REGRESSION],
        targets={
            "funding_raised_usd": TargetSpec(
                name="funding_raised_usd",
                task_type=TaskType.REGRESSION,
                actual_column="funding_raised_usd",
                description="Target for OOD evaluation",
            ),
        },
        horizons=[14, 30],
        ablations=["core_only", "+edgar", "full"],  # Explicit EDGAR ablation
        split_policy=SplitPolicy(
            train_end=date(2023, 12, 31),  # Pre-2024 for year shift
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),  # OOD test
            gap_days=0,
        ),
    )


def create_task4_protocol() -> TaskProtocol:
    """Create protocol for Task 4: Narrative Shift & GenAI Era Effect."""
    return TaskProtocol(
        name="task4_narrative_shift",
        description="Narrative Shift & GenAI Era effect via DiD + mediation analysis",
        task_types=[TaskType.CAUSAL],
        targets={
            "funding_raised_usd": TargetSpec(
                name="funding_raised_usd",
                task_type=TaskType.CAUSAL,
                actual_column="funding_raised_usd",
                description="Outcome for DiD analysis",
            ),
            "investors_count": TargetSpec(
                name="investors_count",
                task_type=TaskType.CAUSAL,
                actual_column="investors_count",
                description="Alternative outcome",
            ),
        },
        horizons=[0],  # Not applicable
        ablations=["core_only", "+text", "full"],
        split_policy=SplitPolicy(
            train_end=date(2022, 12, 31),  # Pre-GenAI
            val_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),  # GenAI era
        ),
    )


# Registry
TASK_PROTOCOLS: Dict[str, Callable[[], TaskProtocol]] = {
    "task1_outcome": create_task1_protocol,
    "task2_forecast": create_task2_protocol,
    "task3_risk_adjust": create_task3_protocol,
    "task4_narrative_shift": create_task4_protocol,
}


def get_protocol(task_name: str) -> TaskProtocol:
    """Get protocol for a task."""
    if task_name not in TASK_PROTOCOLS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_PROTOCOLS[task_name]()


__all__ = [
    "FeatureAvailability",
    "TaskType",
    "FeatureSpec",
    "TargetSpec",
    "MetricSpec",
    "SplitPolicy",
    "TaskProtocol",
    "LeakageValidator",
    "get_protocol",
    "TASK_PROTOCOLS",
]
