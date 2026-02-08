#!/usr/bin/env python3
"""
Block 3 Benchmark Harness (SLURM-Sharded Version).

Unified training/prediction/evaluation interface for all baseline models.
Designed for parallel execution across SLURM jobs.

CLI Options:
    --task {task1_outcome,task2_forecast,task3_risk_adjust}
    --category {statistical,ml_tabular,deep_classical,transformer_sota,foundation,irregular}
    --models comma-separated allowlist (default: all in category)
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
CATEGORY_NAMES = ["statistical", "ml_tabular", "deep_classical", "transformer_sota", "foundation", "irregular"]
ABLATION_NAMES = ["core_only", "core_text", "core_edgar", "full"]
PRESET_NAMES = ["smoke", "quick", "standard", "full"]


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
        
        if self.max_rows and len(df) > self.max_rows:
            df = df.head(self.max_rows)
        
        # Add text features
        if ablation_cfg.get("use_text", False):
            try:
                df = self.dataset.join_core_with_text(df)
            except Exception as e:
                logger.warning(f"Could not add text features: {e}")
        
        # Add EDGAR features
        if ablation_cfg.get("use_edgar", False):
            try:
                edgar_df = self.dataset.get_edgar_store()
                df = df.merge(
                    edgar_df,
                    on=["cik", "crawled_date_day"],
                    how="left",
                    suffixes=("", "_edgar"),
                )
            except Exception as e:
                logger.warning(f"Could not add edgar features: {e}")
        
        # Apply temporal split
        split_cfg = self.tasks_config.get("split", {})
        temporal_config = TemporalSplitConfig(
            train_end=split_cfg.get("train_end", "2025-06-30"),
            val_end=split_cfg.get("val_end", "2025-09-30"),
            test_end=split_cfg.get("test_end", "2025-12-31"),
            embargo_days=split_cfg.get("embargo_days", 7),
        )
        
        train, val, test, stats = apply_temporal_split(df, temporal_config)
        
        logger.info(
            f"Data loaded: train={len(train):,}, val={len(val):,}, test={len(test):,}"
        )
        
        return train, val, test
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        target: str,
        horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for a model."""
        # Get numeric feature columns
        drop_cols = {"entity_id", "crawled_date_day", "cik", target, "date"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in drop_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[target].fillna(0)
        
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
            model = get_model(model_name)
            
            # Train - pass raw DataFrames to ALL panel-aware model categories
            train_start = time.time()
            fit_kwargs = {}
            if self.category in ("deep_classical", "transformer_sota", "foundation",
                                 "statistical", "irregular"):
                fit_kwargs["train_raw"] = train
                fit_kwargs["target"] = target
                fit_kwargs["horizon"] = horizon
            model.fit(X_train, y_train, **fit_kwargs)
            train_time = time.time() - train_start
            
            # Predict
            infer_start = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - infer_start
            
            # Compute metrics
            n_bootstrap = self.preset_config.n_bootstrap
            metrics_dict = self._compute_metrics(y_test.values, y_pred, n_bootstrap)
            
            result = ModelMetrics(
                model_name=model_name,
                category=self.category,
                task=self.task,
                ablation=self.ablation,
                horizon=horizon,
                target=target,
                split="test",
                n_samples=len(X_test),
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
            
            logger.info(f"  {model_name}: MAE={metrics_dict.get('mae', 'N/A'):.4f}, RMSE={metrics_dict.get('rmse', 'N/A'):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running {model_name}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def run(self) -> bool:
        """Run the benchmark shard."""
        logger.info("=" * 80)
        logger.info(f"Block 3 Benchmark Shard")
        logger.info(f"  Task: {self.task}")
        logger.info(f"  Category: {self.category}")
        logger.info(f"  Ablation: {self.ablation}")
        logger.info(f"  Models: {self.models}")
        logger.info(f"  Preset: {self.preset}")
        logger.info("=" * 80)
        
        try:
            # Get task config
            task_config = self.tasks_config.get("tasks", {}).get(self.task, {})
            targets = task_config.get("targets", [{"name": "funding_raised_usd"}])
            if isinstance(targets[0], dict):
                targets = [t["name"] for t in targets]
            horizons = self.preset_config.horizons
            
            # Run each combination
            for target in targets[:1]:  # Primary target only for sharding
                for horizon in horizons:
                    logger.info(f"\nTarget: {target}, Horizon: {horizon}")
                    
                    # Load data
                    train, val, test = self._load_data(target, horizon)
                    
                    # Run each model
                    for model_name in self.models:
                        result = self.run_model(model_name, train, val, test, target, horizon)
                        if result:
                            self.metrics.append(result)
                            self.manifest.n_models_run += 1
                        else:
                            self.manifest.n_models_failed += 1
            
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
    
    def _save_outputs(self):
        """Save shard outputs."""
        if not self.output_dir:
            logger.warning("No output directory specified, skipping save")
            return
        
        # Save manifest
        manifest_path = self.output_dir / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Save metrics
        if self.metrics:
            metrics_records = [m.to_dict() for m in self.metrics]
            metrics_path = self.output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics_records, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Saved metrics to {metrics_path}")
        
        # Save predictions
        if self.predictions:
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
    )
    
    success = shard.run()
    
    if not success:
        sys.exit(1)
    
    logger.info("Shard completed successfully")


if __name__ == "__main__":
    main()
