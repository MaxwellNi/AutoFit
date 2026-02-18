#!/usr/bin/env python3
"""
Block 3 Benchmark Harness (SLURM-Sharded Version).

Unified training/prediction/evaluation interface for all baseline models.
Designed for parallel execution across SLURM jobs.

CLI Options:
    --task {task1_outcome,task2_forecast,task3_risk_adjust}
    --category {statistical,ml_tabular,deep_classical,transformer_sota,foundation,irregular}
    --models comma-separated allowlist (default: all in category)
    --model-kwargs-json / --model-kwargs-file for per-model hyperparameter overrides
    --ablation {core_only,core_text,core_edgar,full}
    --preset {smoke,quick,standard,full}
    --output-dir (required for non-smoke)
    --seed (default 42)
    --max-entities / --max-rows for smoke/quick
    --num-workers for dataloaders
    --list-models
    --list-tasks

Output per shard:
    MANIFEST.json - run metadata
    metrics.json - evaluation metrics
    predictions.parquet - model predictions

Usage:
    # List modes
    python scripts/run_block3_benchmark.py --list-models
    python scripts/run_block3_benchmark.py --list-tasks
    
    # Smoke test
    python scripts/run_block3_benchmark.py --preset smoke --task task1_outcome --category ml_tabular
    
    # Full shard
    python scripts/run_block3_benchmark.py \\
        --task task1_outcome \\
        --category ml_tabular \\
        --ablation core_only \\
        --preset full \\
        --output-dir runs/benchmarks/block3_20260203_225620/task1/ml_tabular/core_only
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Per-model timeout ────────────────────────────────────────────────────────
# SVR with RBF kernel on 100K rows takes ~1h45m per target.  With 8-12
# targets per shard this blows past the 12h SLURM limit.  Reduce
# SVR subsample to 10K rows to keep it under ~2 minutes per combo.
_MODEL_TIMEOUT_SECONDS = int(os.environ.get("B3_MODEL_TIMEOUT", 1200))  # 20 min

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer
from src.narrative.block3.unified_protocol import (
    TemporalSplitConfig,
    extract_early_k_snapshots,
    apply_temporal_split,
    LeakageGuard,
    load_block3_tasks_config,
)
from src.narrative.block3.models import (
    get_model,
    list_models,
    list_all_models,
    get_models_by_category,
    check_model_available,
    MODEL_CATEGORIES,
)
from src.narrative.block3.tasks import (
    get_task,
    list_tasks,
    TaskBase,
)
from src.narrative.block3.preprocessing import set_global_seed
from src.narrative.block3.metrics import (
    rmse as metric_rmse,
    mae as metric_mae,
    mape as metric_mape,
    smape as metric_smape,
    bootstrap_ci,
    compute_all_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

GLOBAL_SEED = 42
TASK_NAMES = ["task1_outcome", "task2_forecast", "task3_risk_adjust"]
CATEGORY_NAMES = ["statistical", "ml_tabular", "deep_classical", "transformer_sota", "foundation", "irregular", "autofit"]
ABLATION_NAMES = ["core_only", "core_text", "core_edgar", "full"]
PRESET_NAMES = ["smoke", "quick", "standard", "full"]

# Global reference for SIGTERM handler
_ACTIVE_SHARD: Optional["BenchmarkShard"] = None

def _sigterm_handler(signum, frame):
    """Save partial results when SLURM sends SIGTERM before OOM/timeout kill."""
    logger.warning(f"Received signal {signum}; saving partial results before exit...")
    if _ACTIVE_SHARD is not None:
        _ACTIVE_SHARD.manifest.status = "partial_timeout"
        _ACTIVE_SHARD.manifest.finished_at = datetime.now(timezone.utc).isoformat()
        _ACTIVE_SHARD._save_outputs(partial=False)
        logger.warning(f"Saved {len(_ACTIVE_SHARD.metrics)} metric records before exit")
    sys.exit(128 + signum)

import signal
signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGUSR1, _sigterm_handler)  # SLURM pre-emption


@dataclass
class PresetConfig:
    """Preset configuration for different run sizes."""
    max_entities: Optional[int]
    max_rows: Optional[int]
    horizons: List[int]
    k_values: List[int]
    models_per_category: Optional[int]
    n_bootstrap: int
    
    @classmethod
    def get_preset(cls, name: str) -> "PresetConfig":
        presets = {
            "smoke": cls(
                max_entities=100,
                max_rows=1000,
                horizons=[7],
                k_values=[7],
                models_per_category=1,
                n_bootstrap=0,
            ),
            "quick": cls(
                max_entities=500,
                max_rows=10000,
                horizons=[7, 14],
                k_values=[14, 30],
                models_per_category=2,
                n_bootstrap=100,
            ),
            "standard": cls(
                max_entities=None,
                max_rows=None,
                horizons=[7, 14, 30],
                k_values=[7, 14, 30],
                models_per_category=None,
                n_bootstrap=500,
            ),
            "full": cls(
                max_entities=None,
                max_rows=None,
                horizons=[1, 7, 14, 30],
                k_values=[7, 14, 30, 60, 90],
                models_per_category=None,
                n_bootstrap=1000,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name]


@dataclass
class ShardManifest:
    """Manifest for a benchmark shard run."""
    task: str
    category: str
    ablation: str
    models: List[str]
    preset: str
    seed: int
    git_hash: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    error: Optional[str] = None
    n_models_run: int = 0
    n_models_failed: int = 0
    slurm_job_id: Optional[str] = None
    hostname: Optional[str] = None
    model_kwargs_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    """Metrics for a single model run."""
    model_name: str
    category: str
    task: str
    ablation: str
    horizon: int
    target: str
    split: str  # train/val/test
    
    # Core metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    mase: Optional[float] = None
    
    # Bootstrap CI
    mae_ci_lower: Optional[float] = None
    mae_ci_upper: Optional[float] = None
    
    # Probabilistic metrics
    crps: Optional[float] = None
    quantile_loss: Optional[float] = None
    
    # Metadata
    n_samples: int = 0
    effective_eval_rows: int = 0
    prediction_coverage_ratio: float = 1.0
    n_missing_predictions: int = 0
    fallback_fraction: float = 0.0
    fairness_pass: bool = True
    lane_clip_rate: float = 0.0
    inverse_transform_guard_hits: int = 0
    anchor_models_used: List[str] = field(default_factory=list)
    policy_action_id: Optional[str] = None
    oof_guard_triggered: bool = False
    train_time_seconds: Optional[float] = None
    inference_time_seconds: Optional[float] = None
    seed: int = GLOBAL_SEED
    git_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkShard:
    """
    Single benchmark shard for SLURM parallelization.
    
    Each shard runs one (task, category, ablation) combination.
    """
    
    def __init__(
        self,
        task: str,
        category: str,
        ablation: str,
        models: Optional[List[str]] = None,
        preset: str = "standard",
        output_dir: Optional[Path] = None,
        seed: int = GLOBAL_SEED,
        max_entities: Optional[int] = None,
        max_rows: Optional[int] = None,
        num_workers: int = 4,
        model_kwargs_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.task = task
        self.category = category
        self.ablation = ablation
        self.preset = preset
        self.output_dir = output_dir
        self.seed = seed
        self.num_workers = num_workers
        
        # Load preset config
        self.preset_config = PresetConfig.get_preset(preset)
        self.max_entities = max_entities or self.preset_config.max_entities
        self.max_rows = max_rows or self.preset_config.max_rows
        self.model_kwargs_overrides = model_kwargs_overrides or {}
        
        # Determine models to run
        if models:
            self.models = models
        else:
            category_models = MODEL_CATEGORIES.get(category, [])
            if self.preset_config.models_per_category:
                self.models = category_models[:self.preset_config.models_per_category]
            else:
                self.models = category_models
        
        # Get git hash
        self.git_hash = self._get_git_hash()
        
        # Load dataset and task configs
        self.tasks_config = load_block3_tasks_config()
        self.dataset = Block3Dataset.from_pointer()
        
        # Setup output
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.metrics: List[ModelMetrics] = []
        self.predictions: List[pd.DataFrame] = []
        
        # Initialize manifest
        self.manifest = ShardManifest(
            task=task,
            category=category,
            ablation=ablation,
            models=self.models,
            preset=preset,
            seed=seed,
            git_hash=self.git_hash,
            started_at=datetime.now(timezone.utc).isoformat(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            hostname=os.environ.get("HOSTNAME", os.uname().nodename),
            model_kwargs_overrides=self.model_kwargs_overrides,
        )
        
        set_global_seed(seed)
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_ablation_config(self) -> Dict[str, bool]:
        """Get ablation configuration flags."""
        ablations = self.tasks_config.get("ablations", {})
        abl_config = ablations.get(self.ablation, {})
        return {
            "use_text": abl_config.get("use_text", False),
            "use_edgar": abl_config.get("use_edgar", False),
        }
    
    def _load_data(self, target: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for the task."""
        ablation_cfg = self._get_ablation_config()
        
        # Load core data
        df = self.dataset.get_offers_core_daily()
        
        # Apply max_entities/max_rows limits
        if self.max_entities:
            unique_entities = df["entity_id"].unique()[:self.max_entities]
            df = df[df["entity_id"].isin(unique_entities)]
        
        # NOTE: max_rows truncation moved AFTER temporal split to avoid
        # biasing toward early rows and potentially removing all test data.
        # See _load_data post-split truncation below.
        
        # Add text features
        if ablation_cfg.get("use_text", False):
            try:
                df = self.dataset.join_core_with_text(df)
            except Exception as e:
                logger.warning(f"Could not add text features: {e}")
        
        # Add EDGAR features (as-of join for better coverage)
        if ablation_cfg.get("use_edgar", False):
            try:
                edgar_df = self.dataset.get_edgar_store()
                # Ensure datetime types for merge_asof — strip timezone to
                # avoid "incompatible merge keys" between tz-naive and tz-aware
                df["crawled_date_day"] = pd.to_datetime(
                    df["crawled_date_day"], utc=True
                ).dt.tz_convert(None)
                edgar_df = edgar_df.copy()
                edgar_df["crawled_date_day"] = pd.to_datetime(
                    edgar_df["crawled_date_day"], utc=True
                ).dt.tz_convert(None)

                n_with_cik = int(df["cik"].notna().sum())
                edgar_feature_cols = [c for c in edgar_df.columns
                                     if c not in ("cik", "crawled_date_day")]

                if n_with_cik > 0 and len(edgar_feature_cols) > 0:
                    # Split into rows with/without CIK
                    df_cik = df[df["cik"].notna()].sort_values("crawled_date_day").copy()
                    df_no_cik = df[df["cik"].isna()].copy()
                    edgar_sorted = edgar_df.sort_values("crawled_date_day")

                    # As-of join: most recent EDGAR filing within 90 days
                    # direction="backward" prevents future-information leakage
                    df_merged = pd.merge_asof(
                        df_cik,
                        edgar_sorted,
                        by="cik",
                        on="crawled_date_day",
                        direction="backward",
                        tolerance=pd.Timedelta("90D"),
                        suffixes=("", "_edgar"),
                    )
                    # Rejoin rows without CIK (EDGAR cols will be NaN)
                    for ec in edgar_feature_cols:
                        if ec not in df_no_cik.columns:
                            df_no_cik[ec] = np.nan
                    df = pd.concat([df_merged, df_no_cik], ignore_index=True)

                    match_rate = df[edgar_feature_cols[0]].notna().mean()
                    logger.info(
                        f"  EDGAR as-of join: {n_with_cik:,}/{len(df):,} rows with CIK, "
                        f"match_rate={match_rate:.1%} on {len(edgar_feature_cols)} features"
                    )
                else:
                    logger.warning(
                        f"  EDGAR join skipped: {n_with_cik} rows with CIK, "
                        f"{len(edgar_feature_cols)} EDGAR features"
                    )
            except Exception as e:
                logger.warning(f"Could not add edgar features: {e}")
                import traceback as _tb; logger.debug(_tb.format_exc())
        
        # Apply temporal split
        split_cfg = self.tasks_config.get("split", {})
        temporal_config = TemporalSplitConfig(
            train_end=split_cfg.get("train_end", "2025-06-30"),
            val_end=split_cfg.get("val_end", "2025-09-30"),
            test_end=split_cfg.get("test_end", "2025-12-31"),
            embargo_days=split_cfg.get("embargo_days", 7),
        )
        
        train, val, test, stats = apply_temporal_split(df, temporal_config)

        # Apply max_rows limit AFTER temporal split — truncate training set
        # only (never truncate test set, which would bias evaluation)
        if self.max_rows:
            if len(train) > self.max_rows:
                train = train.tail(self.max_rows)  # keep most recent rows
                logger.info(f"  max_rows: truncated train to {len(train):,}")

        logger.info(
            f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}"
        )
        
        return train, val, test
    
    # ------------------------------------------------------------------
    # Leakage-safe drop sets: when predicting one target, all
    # co-determined / synonym / derivative columns MUST be removed.
    # ------------------------------------------------------------------
    _TARGET_LEAK_GROUPS: Dict[str, set] = {
        "funding_raised_usd": {
            "funding_raised_usd", "funding_raised",          # same value, diff currency
            "is_funded",                                      # derived: raised >= goal
            "investors_count", "non_national_investors",      # co-determined
        },
        "investors_count": {
            "investors_count", "non_national_investors",
            "funding_raised_usd", "funding_raised",
            "is_funded",
        },
        "is_funded": {
            "is_funded",
            "funding_raised_usd", "funding_raised",
            "investors_count", "non_national_investors",
        },
        "funding_goal_usd": {
            "funding_goal_usd", "funding_goal",
            "funding_goal_maximum", "funding_goal_maximum_usd",
        },
    }

    # Columns that must ALWAYS be dropped from features (identifiers / dates)
    _ALWAYS_DROP: set = {
        "entity_id", "crawled_date_day", "cik", "date",
        "offer_id", "snapshot_ts", "crawled_date", "processed_datetime",
    }

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target: str,
        horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for a model.

        CRITICAL changes for KDD'26 correctness:
        1. Drop ALL co-determined / synonym columns for the current target
           (see _TARGET_LEAK_GROUPS) — prevents target-synonym leakage.
        2. Use dropna() on target instead of fillna(0) — NaN ≠ zero funding.
        3. Synchronise X and y indices after dropping NaN targets.
        """
        # --- 1. Drop NaN targets (NOT fill with 0) ---
        valid_mask = df[target].notna()
        df_clean = df[valid_mask].copy()

        if len(df_clean) == 0:
            logger.warning(f"No valid target values for {target}")
            return pd.DataFrame(), pd.Series(dtype=np.float64)

        # --- 2. Build leakage-safe drop set ---
        leak_group = self._TARGET_LEAK_GROUPS.get(target, {target})
        drop_cols = self._ALWAYS_DROP | leak_group

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in drop_cols]

        n_leaked_dropped = len(leak_group) - 1  # minus target itself
        if n_leaked_dropped > 0:
            actually_present = [c for c in (leak_group - {target}) if c in df_clean.columns]
            if actually_present:
                logger.info(
                    f"  [LEAKAGE GUARD] Dropped {len(actually_present)} "
                    f"co-determined cols for target={target}: {actually_present}"
                )

        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target]

        return X, y
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 0,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Filter NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return metrics
        
        metrics["mae"] = float(metric_mae(y_true, y_pred))
        metrics["rmse"] = float(metric_rmse(y_true, y_pred))
        metrics["smape"] = float(metric_smape(y_true, y_pred))
        
        # MAPE with zero handling
        nonzero = y_true != 0
        if nonzero.sum() > 0:
            metrics["mape"] = float(metric_mape(y_true[nonzero], y_pred[nonzero]))
        
        # Bootstrap CI
        if n_bootstrap > 0 and len(y_true) >= 10:
            try:
                ci = bootstrap_ci(y_true, y_pred, metric_mae, n_bootstrap=n_bootstrap, seed=self.seed)
                metrics["mae_ci_lower"] = ci.ci_lower
                metrics["mae_ci_upper"] = ci.ci_upper
            except Exception:
                pass
        
        return metrics
    
    # Classification-only models: skip for regression targets
    _CLASSIFICATION_ONLY_MODELS = {"LogisticRegression"}
    _CLASSIFICATION_TARGETS = {"is_funded"}
    _MODEL_TIMEOUT_SECONDS = 30 * 60  # 30 minutes max per model

    @staticmethod
    def _extract_fallback_fraction(model: Any) -> float:
        """Best-effort fallback usage extraction for fairness metadata."""
        try:
            if hasattr(model, "get_routing_info"):
                info = model.get_routing_info()
                if isinstance(info, dict):
                    for key in ("fallback_fraction", "fallback_rate", "fallback_ratio"):
                        if key in info:
                            val = float(info[key])
                            return min(max(val, 0.0), 1.0)
        except Exception:
            pass

        # Common wrappers expose a boolean fallback mode.
        if hasattr(model, "_use_fallback"):
            try:
                return 1.0 if bool(getattr(model, "_use_fallback")) else 0.0
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _extract_routing_signals(model: Any) -> Dict[str, Any]:
        """Best-effort extraction of model routing diagnostics for metrics schema."""
        signals: Dict[str, Any] = {
            "lane_clip_rate": 0.0,
            "inverse_transform_guard_hits": 0,
            "anchor_models_used": [],
            "policy_action_id": None,
            "oof_guard_triggered": False,
        }
        try:
            if hasattr(model, "get_routing_info"):
                info = model.get_routing_info()
                if isinstance(info, dict):
                    signals["lane_clip_rate"] = float(info.get("lane_clip_rate", 0.0) or 0.0)
                    signals["inverse_transform_guard_hits"] = int(
                        info.get("inverse_transform_guard_hits", 0) or 0
                    )
                    anchors = info.get("anchor_models_used", info.get("anchor_set", []))
                    if isinstance(anchors, list):
                        signals["anchor_models_used"] = [str(x) for x in anchors]
                    signals["policy_action_id"] = (
                        None if info.get("policy_action_id") is None
                        else str(info.get("policy_action_id"))
                    )
                    signals["oof_guard_triggered"] = bool(info.get("oof_guard_triggered", False))
        except Exception:
            pass
        return signals

    def run_model(
        self,
        model_name: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        horizon: int,
    ) -> Optional[ModelMetrics]:
        """Run a single model and return metrics."""
        logger.info(f"Running {model_name}...")

        # Skip classification-only models for regression targets
        if model_name in self._CLASSIFICATION_ONLY_MODELS and target not in self._CLASSIFICATION_TARGETS:
            logger.info(f"  Skipping {model_name} (classification-only) for regression target={target}")
            return None

        # Skip regression-only linear models for binary classification targets
        _REGRESSION_ONLY = {"Ridge", "Lasso", "ElasticNet", "SVR", "QuantileRegressor"}
        if model_name in _REGRESSION_ONLY and target in self._CLASSIFICATION_TARGETS:
            logger.info(f"  Skipping {model_name} (regression-only) for classification target={target}")
            return None

        if not check_model_available(model_name):
            logger.warning(f"Model {model_name} not available, skipping")
            return None

        try:
            # Prepare features
            X_train, y_train = self._prepare_features(train, target, horizon)
            X_test, y_test = self._prepare_features(test, target, horizon)

            if len(X_train) < 10 or len(X_test) < 10:
                logger.warning(f"Insufficient data for {model_name}")
                return None

            # Get model
            model_kwargs_overrides = getattr(self, "model_kwargs_overrides", {})
            model_kwargs = model_kwargs_overrides.get(model_name, {})
            if model_kwargs:
                logger.info(f"  Applying model kwargs for {model_name}: {model_kwargs}")
            model = get_model(model_name, **model_kwargs)

            # Train - pass raw DataFrames to ALL panel-aware model categories
            train_start = time.time()
            fit_kwargs = {}
            if self.category in ("deep_classical", "transformer_sota", "foundation",
                                 "statistical", "irregular", "autofit"):
                fit_kwargs["train_raw"] = train
                fit_kwargs["target"] = target
                fit_kwargs["horizon"] = horizon
            model.fit(X_train, y_train, **fit_kwargs)
            train_time = time.time() - train_start

            # Predict — pass entity mapping for panel-aware categories
            infer_start = time.time()
            predict_kwargs = {}
            if self.category in ("deep_classical", "transformer_sota", "foundation",
                                 "statistical", "irregular", "autofit"):
                predict_kwargs["test_raw"] = test
                predict_kwargs["target"] = target
                predict_kwargs["horizon"] = horizon
            y_pred = np.asarray(model.predict(X_test, **predict_kwargs), dtype=float)
            infer_time = time.time() - infer_start

            # Fairness guard 1: strict length match
            if len(y_pred) != len(y_test):
                raise RuntimeError(
                    f"[FAIRNESS GUARD] len(y_pred)={len(y_pred)} "
                    f"!= len(y_test)={len(y_test)} for {model_name}"
                )

            # Fairness guard 2: prediction coverage threshold
            pred_finite = np.isfinite(y_pred)
            prediction_coverage_ratio = float(pred_finite.mean()) if len(y_pred) else 0.0
            n_missing_predictions = int((~pred_finite).sum())
            if prediction_coverage_ratio < 0.98:
                raise RuntimeError(
                    f"[FAIRNESS GUARD] prediction_coverage_ratio="
                    f"{prediction_coverage_ratio:.4f} < 0.98 for {model_name}"
                )
            fallback_fraction = self._extract_fallback_fraction(model)
            routing_signals = self._extract_routing_signals(model)

            # ---- CONSTANT-PREDICTION GUARD ----
            if len(y_pred) > 1 and np.std(y_pred) == 0.0:
                logger.warning(
                    f"  ⚠ CONSTANT-PREDICTION detected for {model_name} "
                    f"(target={target}, h={horizon}): all {len(y_pred)} "
                    f"predictions = {y_pred[0]:.4f}"
                )

            # Compute metrics
            n_bootstrap = self.preset_config.n_bootstrap
            metrics_dict = self._compute_metrics(y_test.values, y_pred, n_bootstrap)
            effective_eval_rows = int(np.sum(np.isfinite(y_test.values) & np.isfinite(y_pred)))

            result = ModelMetrics(
                model_name=model_name,
                category=self.category,
                task=self.task,
                ablation=self.ablation,
                horizon=horizon,
                target=target,
                split="test",
                n_samples=len(X_test),
                effective_eval_rows=effective_eval_rows,
                prediction_coverage_ratio=prediction_coverage_ratio,
                n_missing_predictions=n_missing_predictions,
                fallback_fraction=fallback_fraction,
                fairness_pass=True,
                lane_clip_rate=float(routing_signals["lane_clip_rate"]),
                inverse_transform_guard_hits=int(routing_signals["inverse_transform_guard_hits"]),
                anchor_models_used=list(routing_signals["anchor_models_used"]),
                policy_action_id=routing_signals["policy_action_id"],
                oof_guard_triggered=bool(routing_signals["oof_guard_triggered"]),
                train_time_seconds=train_time,
                inference_time_seconds=infer_time,
                seed=self.seed,
                git_hash=self.git_hash,
                **metrics_dict,
            )

            # Store predictions
            pred_df = pd.DataFrame({
                "y_true": y_test.values,
                "y_pred": y_pred,
                "model": model_name,
                "task": self.task,
                "ablation": self.ablation,
                "horizon": horizon,
                "target": target,
            })
            self.predictions.append(pred_df)

            mae_val = metrics_dict.get("mae")
            rmse_val = metrics_dict.get("rmse")
            mae_txt = f"{mae_val:.4f}" if isinstance(mae_val, (int, float, np.floating)) else "N/A"
            rmse_txt = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float, np.floating)) else "N/A"
            logger.info(
                f"  {model_name}: MAE={mae_txt}, RMSE={rmse_txt}, "
                f"coverage={prediction_coverage_ratio:.3f}, fallback={fallback_fraction:.3f}"
            )
            return result

        except Exception as e:
            if isinstance(e, RuntimeError) and str(e).startswith("[FAIRNESS GUARD]"):
                # Hard fail to keep benchmark comparisons fair.
                raise
            logger.error(f"Error running {model_name}: {e}")
            logger.debug(traceback.format_exc())
            return None
        finally:
            # Eagerly release model and intermediate arrays
            try:
                del model
            except (NameError, UnboundLocalError):
                pass
            try:
                del X_train, y_train, X_test, y_test
            except (NameError, UnboundLocalError):
                pass
            try:
                del y_pred, pred_df
            except (NameError, UnboundLocalError):
                pass
            gc.collect()
    
    def run(self) -> bool:
        """Run the benchmark shard."""
        global _ACTIVE_SHARD
        _ACTIVE_SHARD = self
        logger.info("=" * 80)
        logger.info(f"Block 3 Benchmark Shard")
        logger.info(f"  Task: {self.task}")
        logger.info(f"  Category: {self.category}")
        logger.info(f"  Ablation: {self.ablation}")
        logger.info(f"  Models: {self.models}")
        logger.info(f"  Preset: {self.preset}")
        logger.info("=" * 80)

        # Resume support: load existing partial metrics
        done_combos: set = set()  # (model_name, target, horizon) already done
        if self.output_dir:
            metrics_path = self.output_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    existing = json.loads(metrics_path.read_text())
                    if isinstance(existing, list):
                        for rec in existing:
                            key = (rec.get("model_name",""), rec.get("target",""), rec.get("horizon",0))
                            done_combos.add(key)
                            # Reconstruct ModelMetrics from saved dict
                            fields = {k for k in ModelMetrics.__dataclass_fields__}
                            filtered = {k: v for k, v in rec.items() if k in fields}
                            try:
                                self.metrics.append(ModelMetrics(**filtered))
                            except TypeError:
                                pass  # skip malformed records
                        logger.info(f"  [resume] Loaded {len(done_combos)} existing metric records")
                except Exception as e:
                    logger.warning(f"  [resume] Could not load existing metrics: {e}")
        
        try:
            # Get task config
            task_config = self.tasks_config.get("tasks", {}).get(self.task, {})
            targets = task_config.get("targets", [{"name": "funding_raised_usd"}])
            if isinstance(targets[0], dict):
                targets = [t["name"] for t in targets]
            horizons = self.preset_config.horizons
            
            # Run each combination
            for target in targets:  # ALL targets for KDD'26 full paper
                # Cross-sectional models produce identical results for all
                # horizons (features are horizon-independent).  Run only the
                # first horizon to avoid wasted SLURM compute.
                run_horizons = horizons
                if self.category == "ml_tabular":
                    run_horizons = [horizons[0]]
                    logger.info(
                        f"  Cross-sectional category {self.category}: "
                        f"single horizon {run_horizons[0]} (feature-independent)"
                    )

                for horizon in run_horizons:
                    logger.info(f"\nTarget: {target}, Horizon: {horizon}")
                    
                    # Load data
                    train, val, test = self._load_data(target, horizon)
                    
                    # Run each model
                    for model_name in self.models:
                        # Skip already-completed combos (resume support)
                        if (model_name, target, horizon) in done_combos:
                            logger.info(f"  [resume] Skipping {model_name} (already done for target={target}, h={horizon})")
                            continue
                        result = self.run_model(model_name, train, val, test, target, horizon)
                        if result:
                            self.metrics.append(result)
                            self.manifest.n_models_run += 1
                        else:
                            self.manifest.n_models_failed += 1
                        # Explicit GC after each model to prevent OOM
                        gc.collect()
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except (ImportError, RuntimeError):
                            pass
                    
                    # Incremental save after each target-horizon combo
                    self._save_outputs(partial=True)
            
            self.manifest.status = "completed"
            self.manifest.finished_at = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            self.manifest.status = "failed"
            self.manifest.error = str(e)
            self.manifest.finished_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"Shard failed: {e}")
            logger.debug(traceback.format_exc())
            return False
        
        # Save outputs
        self._save_outputs()
        
        return True
    
    def _save_outputs(self, partial: bool = False):
        """Save shard outputs. Called incrementally and at final completion."""
        if not self.output_dir:
            logger.warning("No output directory specified, skipping save")
            return
        
        # Save manifest
        manifest_path = self.output_dir / "MANIFEST.json"
        manifest_dict = self.manifest.to_dict()
        if partial:
            manifest_dict["status"] = "partial"
        manifest_path.write_text(
            json.dumps(manifest_dict, indent=2),
            encoding="utf-8",
        )
        if not partial:
            logger.info(f"Saved manifest to {manifest_path}")
        
        # Save metrics
        if self.metrics:
            metrics_records = [m.to_dict() for m in self.metrics]
            metrics_path = self.output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics_records, indent=2),
                encoding="utf-8",
            )
            if not partial:
                logger.info(f"Saved metrics to {metrics_path}")
            else:
                logger.info(f"  [incremental] Saved {len(metrics_records)} metric records")
        
        # Save predictions (only on final save to avoid large IO)
        if not partial and self.predictions:
            pred_df = pd.concat(self.predictions, ignore_index=True)
            pred_path = self.output_dir / "predictions.parquet"
            pred_df.to_parquet(pred_path)
            logger.info(f"Saved predictions to {pred_path}")


# =============================================================================
# CLI
# =============================================================================

def list_available_models():
    """Print available models by category."""
    print("Available Models by Category:")
    print("=" * 60)
    for cat, models in MODEL_CATEGORIES.items():
        print(f"\n[{cat}]")
        for m in models:
            avail = "✓" if check_model_available(m) else "✗"
            print(f"  [{avail}] {m}")


def list_available_tasks():
    """Print available tasks."""
    print("Available Tasks:")
    print("=" * 60)
    for task_name in TASK_NAMES:
        print(f"\n{task_name}")
        try:
            config = load_block3_tasks_config()
            task_cfg = config.get("tasks", {}).get(task_name, {})
            print(f"  Description: {task_cfg.get('description', 'N/A')}")
            print(f"  Targets: {task_cfg.get('targets', [])}")
            print(f"  Horizons: {task_cfg.get('horizons', [])}")
        except Exception as e:
            print(f"  Error loading config: {e}")


def _load_model_kwargs_overrides(
    json_str: Optional[str],
    json_file: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    """Load per-model kwargs overrides from JSON string/file."""
    if json_str and json_file:
        raise ValueError("Use either --model-kwargs-json or --model-kwargs-file, not both.")

    raw: Dict[str, Any] = {}
    if json_file:
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif json_str:
        raw = json.loads(json_str)

    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Model kwargs overrides must be a JSON object (dict).")

    out: Dict[str, Dict[str, Any]] = {}
    for model_name, kwargs in raw.items():
        if not isinstance(model_name, str):
            raise ValueError("Model name keys in overrides must be strings.")
        if not isinstance(kwargs, dict):
            raise ValueError(f"Override for model '{model_name}' must be a JSON object.")
        out[model_name] = kwargs
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Block 3 Benchmark Harness (SLURM-Sharded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # List modes
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    
    # Shard specification
    parser.add_argument("--task", choices=TASK_NAMES, help="Task to run")
    parser.add_argument("--category", choices=CATEGORY_NAMES, help="Model category")
    parser.add_argument("--models", type=str, help="Comma-separated model names (default: all in category)")
    parser.add_argument(
        "--model-kwargs-json",
        type=str,
        help="JSON string mapping model_name -> kwargs, e.g. '{\"LightGBM\":{\"n_estimators\":3000}}'",
    )
    parser.add_argument(
        "--model-kwargs-file",
        type=Path,
        help="Path to JSON file mapping model_name -> kwargs",
    )
    parser.add_argument("--ablation", choices=ABLATION_NAMES, default="core_only", help="Ablation config")
    
    # Run configuration
    parser.add_argument("--preset", choices=PRESET_NAMES, default="standard", help="Preset configuration")
    parser.add_argument("--output-dir", type=Path, help="Output directory (required for non-smoke)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed")
    
    # Data limits
    parser.add_argument("--max-entities", type=int, help="Maximum number of entities")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Other
    parser.add_argument("--verify-first", action="store_true", default=True, help="Verify freeze before running")
    parser.add_argument("--no-verify-first", dest="verify_first", action="store_false", help="Skip freeze verification")
    
    args = parser.parse_args()
    
    # Handle list modes
    if args.list_models:
        list_available_models()
        return
    
    if args.list_tasks:
        list_available_tasks()
        return
    
    # Validate required args for actual runs
    if not args.task:
        parser.error("--task is required for benchmark runs")
    
    if not args.category:
        parser.error("--category is required for benchmark runs")
    
    if args.preset != "smoke" and not args.output_dir:
        parser.error("--output-dir is required for non-smoke runs")
    
    # Parse models list
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    model_kwargs_overrides: Dict[str, Dict[str, Any]] = {}
    if args.model_kwargs_json or args.model_kwargs_file:
        try:
            model_kwargs_overrides = _load_model_kwargs_overrides(
                args.model_kwargs_json,
                args.model_kwargs_file,
            )
            logger.info(
                f"Loaded model kwargs overrides for {len(model_kwargs_overrides)} model(s): "
                f"{list(model_kwargs_overrides.keys())}"
            )
        except Exception as e:
            parser.error(f"Invalid model kwargs overrides: {e}")
    
    # Verify freeze gates first
    if args.verify_first:
        logger.info("Verifying freeze gates...")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "block3_verify_freeze.py")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Freeze verification FAILED")
            logger.error(result.stdout)
            logger.error(result.stderr)
            sys.exit(1)
        logger.info("Freeze verification PASSED")
    
    # Create and run shard
    shard = BenchmarkShard(
        task=args.task,
        category=args.category,
        ablation=args.ablation,
        models=models,
        preset=args.preset,
        output_dir=args.output_dir,
        seed=args.seed,
        max_entities=args.max_entities,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
        model_kwargs_overrides=model_kwargs_overrides,
    )
    
    success = shard.run()
    
    if not success:
        sys.exit(1)
    
    logger.info("Shard completed successfully")


if __name__ == "__main__":
    main()
