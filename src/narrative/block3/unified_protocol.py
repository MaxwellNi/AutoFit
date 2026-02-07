#!/usr/bin/env python3
"""
Block 3 Unified Protocol Module.

Shared helpers for Task1/2/3:
- Early-K snapshot extraction (K snapshots, NOT K days)
- Strict temporal split policy
- Leakage guard validation

CRITICAL: This is the SINGLE source of truth for:
1. Early-K definition: K = first K snapshots per entity
2. Label time t0 = last snapshot in early window
3. All features must be available at t0 (no future joins)
4. Strict temporal splits: train_end < val_end < test_end
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TemporalSplitConfig:
    """Configuration for strict temporal splits."""
    train_end: date
    val_end: date
    test_end: date
    embargo_days: int = 7  # Gap between splits
    
    def __post_init__(self):
        # Convert strings to dates if needed
        if isinstance(self.train_end, str):
            self.train_end = datetime.strptime(self.train_end, "%Y-%m-%d").date()
        if isinstance(self.val_end, str):
            self.val_end = datetime.strptime(self.val_end, "%Y-%m-%d").date()
        if isinstance(self.test_end, str):
            self.test_end = datetime.strptime(self.test_end, "%Y-%m-%d").date()
        
        # Validate ordering
        if not (self.train_end < self.val_end < self.test_end):
            raise ValueError(
                f"Invalid temporal order: train_end={self.train_end} < "
                f"val_end={self.val_end} < test_end={self.test_end} must hold"
            )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TemporalSplitConfig":
        """Load split config from YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        split_cfg = data.get("split", {})
        return cls(
            train_end=split_cfg["train_end"],
            val_end=split_cfg["val_end"],
            test_end=split_cfg["test_end"],
            embargo_days=split_cfg.get("embargo_days", 7),
        )


@dataclass 
class EarlyKConfig:
    """Configuration for early-K snapshot extraction."""
    k: int  # Number of snapshots (NOT days)
    date_col: str = "crawled_date_day"
    entity_col: str = "entity_id"
    sort_ascending: bool = True
    
    def __post_init__(self):
        if self.k < 1:
            raise ValueError(f"K must be >= 1, got {self.k}")


@dataclass
class SplitStats:
    """Statistics about a data split."""
    n_rows: int
    n_entities: int
    date_min: date
    date_max: date
    n_unique_dates: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_entities": self.n_entities,
            "date_min": str(self.date_min),
            "date_max": str(self.date_max),
            "n_unique_dates": self.n_unique_dates,
        }


# =============================================================================
# Early-K Extraction (CRITICAL - K snapshots, NOT K days)
# =============================================================================

def extract_early_k_snapshots(
    df: pd.DataFrame,
    k: int,
    entity_col: str = "entity_id",
    date_col: str = "crawled_date_day",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract first K snapshots per entity for early-K feature construction.
    
    CRITICAL: K is the number of SNAPSHOTS (rows), NOT days.
    This ensures consistent feature availability across entities.
    
    Args:
        df: Panel DataFrame sorted by entity and date
        k: Number of snapshots to include (first K rows per entity)
        entity_col: Column name for entity identifier
        date_col: Column name for snapshot date
        feature_cols: Columns to include (default: all numeric)
    
    Returns:
        X: Features from first K snapshots (aggregated per entity)
        label_time: Series mapping entity_id -> t0 (last snapshot date in early window)
    
    Example:
        If entity E has snapshots at [Jan 1, Jan 3, Jan 5, Jan 8] and K=3:
        - Uses snapshots from Jan 1, Jan 3, Jan 5
        - label_time[E] = Jan 5 (the last snapshot in early window)
        - Features are aggregated from these 3 snapshots
    """
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")
    
    if entity_col not in df.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in DataFrame")
    
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Ensure date column is proper datetime for sorting
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by entity and date
    df = df.sort_values([entity_col, date_col])
    
    # Take first K snapshots per entity
    early_k_df = df.groupby(entity_col).head(k)
    
    # Compute label time t0 = last snapshot in early window per entity
    label_time = early_k_df.groupby(entity_col)[date_col].max()
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.bool_]]
        # Exclude entity/date columns
        feature_cols = [c for c in feature_cols if c not in [entity_col, date_col]]
    
    # Aggregate features per entity (mean for simplicity, can be extended)
    agg_funcs = {col: ['mean', 'min', 'max', 'std', 'last'] for col in feature_cols if col in early_k_df.columns}
    if not agg_funcs:
        logger.warning("No numeric feature columns found for aggregation")
        X = early_k_df.groupby(entity_col).size().to_frame(name="n_snapshots")
    else:
        X = early_k_df.groupby(entity_col).agg(agg_funcs)
        # Flatten multi-level columns
        X.columns = ['_'.join(col).strip() for col in X.columns.values]
    
    # Add count of snapshots used (for audit)
    X["n_snapshots_used"] = early_k_df.groupby(entity_col).size()
    
    logger.info(
        f"Extracted early-K={k} snapshots: {len(X)} entities, "
        f"label_time range: [{label_time.min()}, {label_time.max()}]"
    )
    
    return X, label_time


def get_entity_label_time(
    df: pd.DataFrame,
    k: int,
    entity_col: str = "entity_id",
    date_col: str = "crawled_date_day",
) -> pd.Series:
    """
    Get label time t0 for each entity based on K snapshots.
    
    t0 = the date of the K-th snapshot for each entity.
    
    Args:
        df: Panel DataFrame
        k: Number of early snapshots
        entity_col: Entity column name
        date_col: Date column name
    
    Returns:
        Series mapping entity_id -> t0
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([entity_col, date_col])
    
    # Get the K-th snapshot date per entity
    early_k = df.groupby(entity_col).head(k)
    label_time = early_k.groupby(entity_col)[date_col].max()
    
    return label_time


# =============================================================================
# Temporal Splits (CRITICAL - Strict time ordering, NO random split)
# =============================================================================

def apply_temporal_split(
    df: pd.DataFrame,
    config: TemporalSplitConfig,
    date_col: str = "crawled_date_day",
    entity_col: str = "entity_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, SplitStats]]:
    """
    Apply strict temporal train/val/test split.
    
    CRITICAL: This is the ONLY split function for all tasks.
    - train: date <= train_end
    - val: train_end + embargo < date <= val_end
    - test: val_end + embargo < date <= test_end
    
    Args:
        df: DataFrame with date column
        config: Temporal split configuration
        date_col: Date column name
        entity_col: Entity column for stats
    
    Returns:
        train, val, test DataFrames
        stats dict with split statistics
    
    Raises:
        RuntimeError: If temporal ordering is violated
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Compute boundaries
    train_end = pd.Timestamp(config.train_end)
    val_start = train_end + pd.Timedelta(days=config.embargo_days)
    val_end = pd.Timestamp(config.val_end)
    test_start = val_end + pd.Timedelta(days=config.embargo_days)
    test_end = pd.Timestamp(config.test_end)
    
    # Split
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > val_start) & (df[date_col] <= val_end)]
    test = df[(df[date_col] > test_start) & (df[date_col] <= test_end)]
    
    # Compute statistics
    stats = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        if len(split_df) > 0:
            dates = split_df[date_col]
            stats[name] = SplitStats(
                n_rows=len(split_df),
                n_entities=split_df[entity_col].nunique(),
                date_min=dates.min().date(),
                date_max=dates.max().date(),
                n_unique_dates=dates.nunique(),
            )
        else:
            stats[name] = SplitStats(
                n_rows=0,
                n_entities=0,
                date_min=config.train_end if name == "train" else config.val_end,
                date_max=config.train_end if name == "train" else config.val_end,
                n_unique_dates=0,
            )
    
    # Validate strict ordering
    _validate_temporal_ordering(train, val, test, date_col)
    
    logger.info(
        f"Temporal split: train={stats['train'].n_rows} ({stats['train'].n_entities} entities), "
        f"val={stats['val'].n_rows} ({stats['val'].n_entities} entities), "
        f"test={stats['test'].n_rows} ({stats['test'].n_entities} entities)"
    )
    
    return train, val, test, stats


def _validate_temporal_ordering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str,
) -> None:
    """
    Validate strict temporal ordering: max(train) < min(val) < min(test).
    
    Raises:
        RuntimeError: If ordering is violated
    """
    if len(train) > 0 and len(val) > 0:
        train_max = train[date_col].max()
        val_min = val[date_col].min()
        if train_max >= val_min:
            raise RuntimeError(
                f"Temporal leakage: train max date ({train_max}) >= val min date ({val_min})"
            )
    
    if len(val) > 0 and len(test) > 0:
        val_max = val[date_col].max()
        test_min = test[date_col].min()
        if val_max >= test_min:
            raise RuntimeError(
                f"Temporal leakage: val max date ({val_max}) >= test min date ({test_min})"
            )


# =============================================================================
# Leakage Guard (CRITICAL - Fail-fast validation)
# =============================================================================

class LeakageGuard:
    """
    Validates that features don't leak future information.
    
    Core principle: For any row used for prediction, all features must
    have been available at or before the label cutoff date.
    """
    
    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, raise RuntimeError on violation. If False, log warning.
        """
        self.strict = strict
        self.violations: List[Dict[str, Any]] = []
    
    def validate_feature_dates(
        self,
        df: pd.DataFrame,
        feature_date_col: str,
        label_cutoff_col: str,
    ) -> bool:
        """
        Validate that feature_date <= label_cutoff for all rows.
        
        Args:
            df: DataFrame with feature and cutoff dates
            feature_date_col: Column with feature availability date
            label_cutoff_col: Column with label cutoff date
        
        Returns:
            True if no violations
        
        Raises:
            RuntimeError: If strict=True and violations found
        """
        self.violations = []
        
        feature_dates = pd.to_datetime(df[feature_date_col])
        cutoff_dates = pd.to_datetime(df[label_cutoff_col])
        
        # Find violations
        mask = feature_dates > cutoff_dates
        n_violations = mask.sum()
        
        if n_violations > 0:
            violation_df = df[mask][[feature_date_col, label_cutoff_col]].head(10)
            
            self.violations.append({
                "type": "future_feature",
                "n_violations": int(n_violations),
                "pct_violations": float(n_violations / len(df) * 100),
                "examples": violation_df.to_dict("records"),
            })
            
            msg = (
                f"LEAKAGE DETECTED: {n_violations} rows ({n_violations/len(df)*100:.2f}%) have "
                f"feature_date > label_cutoff. Examples:\n{violation_df.to_string()}"
            )
            
            if self.strict:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
        
        return len(self.violations) == 0
    
    def validate_no_future_joins(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_date_col: str,
        right_date_col: str,
        join_keys: List[str],
    ) -> bool:
        """
        Validate that a join doesn't introduce future information.
        
        The right DataFrame's date must be <= left DataFrame's date for valid joins.
        
        Args:
            left_df: Primary DataFrame
            right_df: DataFrame being joined
            left_date_col: Date column in left
            right_date_col: Date column in right  
            join_keys: Columns used for joining
        
        Returns:
            True if no leakage detected
        """
        # Sample check: for each join key combination, verify date ordering
        left_dates = left_df.groupby(join_keys)[left_date_col].min()
        right_dates = right_df.groupby(join_keys)[right_date_col].max()
        
        # Align indices
        common_keys = left_dates.index.intersection(right_dates.index)
        
        if len(common_keys) == 0:
            return True
        
        left_aligned = left_dates.loc[common_keys]
        right_aligned = right_dates.loc[common_keys]
        
        mask = pd.to_datetime(right_aligned) > pd.to_datetime(left_aligned)
        n_violations = mask.sum()
        
        if n_violations > 0:
            self.violations.append({
                "type": "future_join",
                "n_violations": int(n_violations),
                "join_keys": join_keys,
            })
            
            msg = f"LEAKAGE DETECTED: {n_violations} join key combinations have right_date > left_date"
            
            if self.strict:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
        
        return len(self.violations) == 0


def validate_no_leakage(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_date_col: str,
    label_date_col: str,
    entity_col: str = "entity_id",
) -> None:
    """
    Convenience function to validate no leakage between features and labels.
    
    For each entity, asserts that all feature dates are <= the label cutoff date.
    
    Args:
        features_df: Features DataFrame with dates
        labels_df: Labels DataFrame with cutoff dates
        feature_date_col: Feature date column
        label_date_col: Label cutoff date column
        entity_col: Entity identifier column
    
    Raises:
        RuntimeError: If leakage detected
    """
    # Merge to align
    merged = features_df.merge(
        labels_df[[entity_col, label_date_col]].drop_duplicates(),
        on=entity_col,
        how="left",
    )
    
    guard = LeakageGuard(strict=True)
    guard.validate_feature_dates(merged, feature_date_col, label_date_col)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_block3_tasks_config(path: Union[str, Path] = "configs/block3_tasks.yaml") -> Dict[str, Any]:
    """Load the block3_tasks.yaml configuration."""
    path = Path(path)
    if not path.exists():
        # Try relative to repo root
        alt_path = Path(__file__).parent.parent.parent.parent / path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Config not found: {path}")
    
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    "TemporalSplitConfig",
    "EarlyKConfig",
    "SplitStats",
    
    # Early-K extraction
    "extract_early_k_snapshots",
    "get_entity_label_time",
    
    # Temporal splits
    "apply_temporal_split",
    
    # Leakage guard
    "LeakageGuard",
    "validate_no_leakage",
    
    # Config loading
    "load_block3_tasks_config",
]
