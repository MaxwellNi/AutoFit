#!/usr/bin/env python3
"""
Block 3 Benchmark Harness.

Unified training/prediction/evaluation interface for all baseline models.
Includes support for:
- Multiple baseline categories (statistical, ML, deep, transformer, foundation)
- Ablation configurations (core-only, +text, +edgar, +multiscale)
- Comprehensive metrics (point, probabilistic, coverage)
- Leaderboard generation
- Task-based benchmarking (Task1-4)

Usage:
    python scripts/run_block3_benchmark.py --config configs/block3.yaml
    python scripts/run_block3_benchmark.py --smoke-test
    python scripts/run_block3_benchmark.py --task outcome --models LightGBM XGBoost
    python scripts/run_block3_benchmark.py --list-models
    python scripts/run_block3_benchmark.py --list-tasks
    python scripts/run_block3_benchmark.py --preset smoke_test  # Leaderboard + skipped table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.data_preprocessing.block3_dataset import Block3Dataset
from src.narrative.block3.models import (
    get_model,
    list_models,
    get_preset_models,
    check_model_available,
    MODEL_CATEGORIES,
)
from src.narrative.block3.tasks import (
    get_task,
    list_tasks,
    TaskBase,
)
# Import new unified modules
from src.narrative.block3.preprocessing import (
    set_global_seed,
    get_seed_sequence,
    MissingPolicy,
    get_missing_policy,
    ScalingPolicy,
    PreprocessConfig,
    UnifiedPreprocessor,
)
from src.narrative.block3.metrics import (
    rmse as metric_rmse,
    mae as metric_mae,
    mape as metric_mape,
    smape as metric_smape,
    bootstrap_ci,
    METRIC_REGISTRY,
    compute_all_metrics,
)
from src.narrative.block3.protocol import (
    get_protocol,
    LeakageValidator,
    TASK_PROTOCOLS,
)

# Global seed for reproducibility
GLOBAL_SEED = 42


@dataclass
class BenchmarkConfig:
    """Parsed block3.yaml configuration."""
    data: Dict[str, Any]
    targets: Dict[str, List[str]]
    horizons: List[int]
    context_lengths: List[int]
    split: Dict[str, Any]
    metrics: Dict[str, Any]
    baselines: Dict[str, Any]
    ablations: Dict[str, Any]
    autofit: Dict[str, Any]
    training: Dict[str, Any]
    output: Dict[str, Any]
    smoke_test: Dict[str, Any]
    
    @classmethod
    def load(cls, path: Path) -> "BenchmarkConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class SkippedModel:
    """Info about a model that was skipped."""
    model_name: str
    model_category: str
    skip_reason: str
    details: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_category": self.model_category,
            "skip_reason": self.skip_reason,
            "details": self.details,
        }


@dataclass
class ModelResult:
    """Result from a single model run."""
    model_name: str
    model_category: str
    ablation: str
    horizon: int
    context_length: int
    target: str
    
    # Point metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    smape: Optional[float] = None
    wape: Optional[float] = None
    mase: Optional[float] = None
    
    # Bootstrap CI for primary metric (MAE)
    mae_ci_lower: Optional[float] = None
    mae_ci_upper: Optional[float] = None
    
    # Probabilistic metrics
    quantile_loss: Optional[float] = None
    crps: Optional[float] = None
    
    # Coverage
    coverage_10: Optional[float] = None
    coverage_50: Optional[float] = None
    coverage_90: Optional[float] = None
    
    # Meta
    train_time_seconds: Optional[float] = None
    inference_time_seconds: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    n_parameters: Optional[int] = None
    seed: int = GLOBAL_SEED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_category": self.model_category,
            "ablation": self.ablation,
            "horizon": self.horizon,
            "context_length": self.context_length,
            "target": self.target,
            "mae": self.mae,
            "rmse": self.rmse,
            "smape": self.smape,
            "wape": self.wape,
            "mase": self.mase,
            "mae_ci_lower": self.mae_ci_lower,
            "mae_ci_upper": self.mae_ci_upper,
            "quantile_loss": self.quantile_loss,
            "crps": self.crps,
            "coverage_10": self.coverage_10,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
            "train_time_seconds": self.train_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "peak_memory_gb": self.peak_memory_gb,
            "n_parameters": self.n_parameters,
            "seed": self.seed,
        }


class BenchmarkHarness:
    """Main benchmark orchestrator."""
    
    def __init__(
        self,
        config: BenchmarkConfig,
        dataset: Block3Dataset,
        seed: int = GLOBAL_SEED,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
    ):
        self.config = config
        self.dataset = dataset
        self.seed = seed
        self.compute_ci = compute_ci
        self.n_bootstrap = n_bootstrap
        self.results: List[ModelResult] = []
        self.skipped_models: List[SkippedModel] = []
        
        # Set global seed for reproducibility
        set_global_seed(seed)
        
        # Setup output directory
        self.output_dir = Path(config.output["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _record_skipped(
        self,
        model_name: str,
        model_category: str,
        reason: str,
        details: str = "",
    ):
        """Record a skipped model."""
        self.skipped_models.append(SkippedModel(
            model_name=model_name,
            model_category=model_category,
            skip_reason=reason,
            details=details,
        ))
    
    def prepare_data(
        self,
        target: str,
        horizon: int,
        context_length: int,
        ablation_config: Dict[str, bool],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/val/test splits with specified features."""
        # Load core data
        df = self.dataset.get_offers_core_daily()
        
        # Add features based on ablation
        if ablation_config.get("use_text", False):
            try:
                df = self.dataset.join_core_with_text(df)
            except Exception as e:
                print(f"Warning: Could not add text features: {e}")
        
        if ablation_config.get("use_edgar", False):
            try:
                edgar_df = self.dataset.get_edgar_store()
                df = df.merge(
                    edgar_df,
                    on=["cik", "crawled_date_day"],
                    how="left",
                    suffixes=("", "_edgar"),
                )
            except Exception as e:
                print(f"Warning: Could not add edgar features: {e}")
        
        # Parse dates
        df["date"] = pd.to_datetime(df["crawled_date_day"])
        
        # Split by time
        split_cfg = self.config.split
        train_end = pd.to_datetime(split_cfg["train_end"])
        val_end = pd.to_datetime(split_cfg["val_end"])
        test_end = pd.to_datetime(split_cfg["test_end"])
        embargo = pd.Timedelta(days=split_cfg.get("embargo_days", 7))
        
        train = df[df["date"] <= train_end]
        val = df[(df["date"] > train_end + embargo) & (df["date"] <= val_end)]
        test = df[(df["date"] > val_end + embargo) & (df["date"] <= test_end)]
        
        return train, val, test
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_quantiles: Optional[Dict[float, np.ndarray]] = None,
        with_ci: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute all metrics using unified metrics module.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            y_pred_quantiles: Optional quantile predictions
            with_ci: Whether to compute bootstrap confidence intervals
        
        Returns:
            Dictionary of metrics (float or BootstrapCI if with_ci=True)
        """
        metrics = {}
        
        # Filter NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return metrics
        
        # Use unified metrics module
        metrics["mae"] = metric_mae(y_true_clean, y_pred_clean)
        metrics["rmse"] = metric_rmse(y_true_clean, y_pred_clean)
        metrics["smape"] = metric_smape(y_true_clean, y_pred_clean)
        
        # WAPE
        if np.sum(np.abs(y_true_clean)) > 0:
            metrics["wape"] = float(np.sum(np.abs(y_true_clean - y_pred_clean)) / np.sum(np.abs(y_true_clean)) * 100)
        
        # MASE (assuming seasonal naive baseline with period 1)
        naive_mae = np.mean(np.abs(np.diff(y_true_clean))) if len(y_true_clean) > 1 else 1.0
        if naive_mae > 0:
            metrics["mase"] = metrics["mae"] / naive_mae
        
        # Bootstrap CI for MAE
        if with_ci and len(y_true_clean) >= 10:
            mae_ci = bootstrap_ci(
                y_true_clean, y_pred_clean, metric_mae,
                n_bootstrap=self.n_bootstrap, seed=self.seed
            )
            metrics["mae_ci_lower"] = mae_ci.ci_lower
            metrics["mae_ci_upper"] = mae_ci.ci_upper
        
        # Probabilistic metrics (if quantiles provided)
        if y_pred_quantiles:
            # Quantile loss
            ql_sum = 0
            for q, y_q in y_pred_quantiles.items():
                y_q = y_q[mask]
                errors = y_true_clean - y_q
                ql = np.where(errors > 0, q * errors, (q - 1) * errors)
                ql_sum += np.mean(ql)
            metrics["quantile_loss"] = float(ql_sum / len(y_pred_quantiles))
            
            # Coverage
            for q in [0.1, 0.5, 0.9]:
                if q in y_pred_quantiles:
                    y_q = y_pred_quantiles[q][mask]
                    if q == 0.5:
                        # P50 should be close to median
                        metrics[f"coverage_{int(q*100)}"] = float(np.mean(y_true_clean <= y_q))
                    else:
                        metrics[f"coverage_{int(q*100)}"] = float(np.mean(y_true_clean <= y_q))
        
        return metrics
    
    def run_statistical_baseline(
        self,
        model_config: Dict[str, Any],
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        horizon: int,
    ) -> ModelResult:
        """Run a statistical baseline model."""
        model_name = model_config["name"]
        print(f"  Running {model_name}...")
        
        start_time = time.time()
        
        # Simple implementation: compute naive/seasonal naive forecast
        # Full implementation would use statsforecast library
        
        result = ModelResult(
            model_name=model_name,
            model_category="statistical",
            ablation="core_only",
            horizon=horizon,
            context_length=0,
            target=target,
        )
        
        try:
            # Seasonal naive baseline
            if model_name == "SeasonalNaive":
                season = model_config.get("params", {}).get("season_length", 7)
                
                # Group by entity and compute forecast
                all_true = []
                all_pred = []
                
                for entity_id, group in test.groupby("entity_id"):
                    group = group.sort_values("crawled_date_day")
                    y = group[target].values
                    
                    # Naive forecast: use value from season_length days ago
                    for i in range(len(y)):
                        if i >= season:
                            all_pred.append(y[i - season])
                            all_true.append(y[i])
                
                if all_true:
                    metrics = self.compute_metrics(np.array(all_true), np.array(all_pred))
                    result.mae = metrics.get("mae")
                    result.rmse = metrics.get("rmse")
                    result.smape = metrics.get("smape")
                    result.wape = metrics.get("wape")
                    result.mase = metrics.get("mase")
            
            else:
                # Placeholder for other statistical models
                result.mae = None
        
        except Exception as e:
            print(f"    Error: {e}")
        
        result.train_time_seconds = time.time() - start_time
        
        return result
    
    def run_ml_baseline(
        self,
        model_config: Dict[str, Any],
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        horizon: int,
        context_length: int,
    ) -> ModelResult:
        """Run an ML tabular baseline model."""
        model_name = model_config["name"]
        print(f"  Running {model_name}...")
        
        start_time = time.time()
        
        result = ModelResult(
            model_name=model_name,
            model_category="ml_tabular",
            ablation="core_only",
            horizon=horizon,
            context_length=context_length,
            target=target,
        )
        
        try:
            # Feature engineering: lag features
            feature_cols = []
            
            # Create lag features from train
            train_processed = train.copy()
            for lag in range(1, min(context_length, 30) + 1):
                train_processed[f"{target}_lag_{lag}"] = train_processed.groupby("entity_id")[target].shift(lag)
                feature_cols.append(f"{target}_lag_{lag}")
            
            # Drop rows with NaN features
            train_processed = train_processed.dropna(subset=feature_cols + [target])
            
            if len(train_processed) < 100:
                print(f"    Not enough training data after feature engineering")
                return result
            
            X_train = train_processed[feature_cols].values
            y_train = train_processed[target].values
            
            # Same for test
            test_processed = test.copy()
            for lag in range(1, min(context_length, 30) + 1):
                test_processed[f"{target}_lag_{lag}"] = test_processed.groupby("entity_id")[target].shift(lag)
            
            test_processed = test_processed.dropna(subset=feature_cols + [target])
            X_test = test_processed[feature_cols].values
            y_test = test_processed[target].values
            
            if len(X_test) == 0:
                return result
            
            # Train model
            if model_name == "LightGBM":
                try:
                    import lightgbm as lgb
                    params = model_config.get("params", {})
                    model = lgb.LGBMRegressor(**params, verbose=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    result.n_parameters = model.n_features_in_ * 100  # Rough estimate
                except ImportError:
                    print(f"    LightGBM not installed")
                    return result
            
            elif model_name == "XGBoost":
                try:
                    import xgboost as xgb
                    params = model_config.get("params", {})
                    model = xgb.XGBRegressor(**params, verbosity=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    result.n_parameters = model.n_features_in_ * 100
                except ImportError:
                    print(f"    XGBoost not installed")
                    return result
            
            else:
                return result
            
            # Compute metrics
            metrics = self.compute_metrics(y_test, y_pred)
            result.mae = metrics.get("mae")
            result.rmse = metrics.get("rmse")
            result.smape = metrics.get("smape")
            result.wape = metrics.get("wape")
            result.mase = metrics.get("mase")
        
        except Exception as e:
            print(f"    Error: {e}")
        
        result.train_time_seconds = time.time() - start_time
        
        return result
    
    def run_benchmark(
        self,
        model_names: Optional[List[str]] = None,
        ablation_names: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        targets: Optional[List[str]] = None,
    ):
        """Run the full benchmark."""
        print("=" * 80)
        print("Block 3 Benchmark")
        print("=" * 80)
        
        # Use config defaults if not specified
        if horizons is None:
            horizons = self.config.horizons
        if targets is None:
            targets = self.config.targets["primary"]
        if ablation_names is None:
            ablation_names = [a["name"] for a in self.config.ablations["feature_sets"]]
        
        context_length = self.config.context_lengths[0]  # Use first context length
        
        # Get ablation configs
        ablation_map = {a["name"]: a for a in self.config.ablations["feature_sets"]}
        
        for target in targets:
            print(f"\nTarget: {target}")
            
            for horizon in horizons:
                print(f"  Horizon: {horizon} days")
                
                for ablation_name in ablation_names:
                    print(f"    Ablation: {ablation_name}")
                    
                    ablation_cfg = ablation_map.get(ablation_name, {"use_text": False, "use_edgar": False, "use_multiscale": False})
                    
                    # Prepare data
                    train, val, test = self.prepare_data(target, horizon, context_length, ablation_cfg)
                    print(f"      Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
                    
                    # Run statistical baselines
                    for model_cfg in self.config.baselines.get("statistical", []):
                        if model_names is None or model_cfg["name"] in model_names:
                            result = self.run_statistical_baseline(model_cfg, train, val, test, target, horizon)
                            result.ablation = ablation_name
                            self.results.append(result)
                    
                    # Run ML baselines
                    for model_cfg in self.config.baselines.get("ml_tabular", []):
                        if model_names is None or model_cfg["name"] in model_names:
                            result = self.run_ml_baseline(model_cfg, train, val, test, target, horizon, context_length)
                            result.ablation = ablation_name
                            self.results.append(result)
        
        # Save results
        self.save_results()
    
    def run_smoke_test(self):
        """Run minimal smoke test."""
        print("=" * 80)
        print("Block 3 Smoke Test")
        print("=" * 80)
        
        smoke_cfg = self.config.smoke_test
        
        # Run with minimal configuration
        self.run_benchmark(
            model_names=smoke_cfg.get("baselines", ["SeasonalNaive", "LightGBM"]),
            ablation_names=["core_only"],
            horizons=smoke_cfg.get("horizons", [1, 7]),
            targets=["funding_raised_usd"],
        )
    
    def run_task_benchmark(
        self,
        task_name: str,
        model_names: Optional[List[str]] = None,
        ablation_names: Optional[List[str]] = None,
    ):
        """
        Run benchmark for a specific task.
        
        Args:
            task_name: Task identifier (outcome, forecast, risk, narrative)
            model_names: Optional list of model names to run
            ablation_names: Optional list of ablation configs
        """
        print("=" * 80)
        print(f"Block 3 Task Benchmark: {task_name}")
        print("=" * 80)
        
        # Get task
        task = get_task(task_name)
        
        # Get ablations
        if ablation_names is None:
            ablation_names = ["core_only"]
        
        # Get targets and horizons from task config
        targets = task.config.targets[:1] if task.config.targets else ["funding_raised_usd"]
        horizons = task.config.horizons[:1] if task.config.horizons else [7]
        
        # Determine models to run
        if model_names is None:
            # Use appropriate models based on task type
            if task_name in ("outcome", "task1_outcome"):
                model_names = ["Ridge", "RandomForest", "LightGBM", "XGBoost"]
            elif task_name in ("forecast", "task2_forecast"):
                model_names = ["SeasonalNaive", "Ridge", "LightGBM", "XGBoost"]
            else:
                model_names = ["LightGBM", "XGBoost"]
        
        # Run each combination
        for target in targets:
            for horizon in horizons:
                for ablation_name in ablation_names:
                    print(f"\nTarget: {target}, Horizon: {horizon}, Ablation: {ablation_name}")
                    
                    # Build dataset for this configuration
                    try:
                        X, y = task.build_dataset(
                            ablation=ablation_name,
                            horizon=horizon,
                            target=target,
                        )
                        print(f"  Dataset: {len(X):,} samples, {X.shape[1]} features")
                    except Exception as e:
                        print(f"  Error building dataset: {e}")
                        continue
                    
                    # Get splits
                    try:
                        splits = task.get_splits(X, y)
                        # Handle both list format (train_idx, val_idx, test_idx) and dict format
                        if isinstance(splits, list) and len(splits) > 0:
                            if isinstance(splits[0], tuple):
                                train_idx, val_idx, test_idx = splits[0]
                            else:
                                train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
                        elif isinstance(splits, dict):
                            train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
                        else:
                            raise ValueError(f"Unexpected splits format: {type(splits)}")
                        print(f"  Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
                    except Exception as e:
                        print(f"  Error getting splits: {e}")
                        continue
                    
                    # Run each model
                    for model_name in model_names:
                        if not check_model_available(model_name):
                            print(f"    Skipping {model_name} (not available)")
                            continue
                        
                        print(f"    Running {model_name}...")
                        
                        try:
                            model = get_model(model_name)
                            
                            # Fit on train
                            X_train = X.iloc[train_idx] if hasattr(train_idx, '__len__') else X.loc[train_idx]
                            y_train = y.iloc[train_idx] if hasattr(train_idx, '__len__') else y.loc[train_idx]
                            X_test = X.iloc[test_idx] if hasattr(test_idx, '__len__') else X.loc[test_idx]
                            y_test = y.iloc[test_idx] if hasattr(test_idx, '__len__') else y.loc[test_idx]
                            
                            import time
                            start_time = time.time()
                            model.fit(X_train, y_train)
                            train_time = time.time() - start_time
                            
                            # Predict
                            y_pred = model.predict(X_test)
                            
                            # Compute metrics
                            metrics = self.compute_metrics(y_test.values, y_pred)
                            
                            # Store result
                            model_result = ModelResult(
                                model_name=model_name,
                                model_category=self._get_model_category(model_name),
                                ablation=ablation_name,
                                horizon=horizon,
                                context_length=task.config.context_lengths[0] if task.config.context_lengths else 30,
                                target=target,
                                mae=metrics.get("mae"),
                                rmse=metrics.get("rmse"),
                                smape=metrics.get("smape"),
                                wape=metrics.get("wape"),
                                mase=metrics.get("mase"),
                                train_time_seconds=train_time,
                            )
                            self.results.append(model_result)
                            
                            print(f"      MAE={metrics.get('mae', 'N/A'):.4f}, RMSE={metrics.get('rmse', 'N/A'):.4f}")
                            
                        except Exception as e:
                            print(f"      Error: {e}")
        
        # Save results
        self.save_results()
    
    def _get_model_category(self, model_name: str) -> str:
        """Get category for a model name."""
        for cat, models in MODEL_CATEGORIES.items():
            if model_name in models:
                return cat
        return "unknown"
    
    def save_results(self):
        """Save benchmark results including skipped models table."""
        # Convert results to DataFrame
        records = [r.to_dict() for r in self.results]
        df = pd.DataFrame(records)
        
        # Convert skipped models to DataFrame
        skipped_records = [s.to_dict() for s in self.skipped_models]
        df_skipped = pd.DataFrame(skipped_records)
        
        # Handle empty results
        if len(df) == 0:
            print("\nNo results to save.")
            # Still save skipped models if any
            if len(df_skipped) > 0:
                skipped_path = self.output_dir / "skipped_models.parquet"
                df_skipped.to_parquet(skipped_path)
                print(f"Saved skipped models to {skipped_path}")
            return
        
        # Save leaderboard
        leaderboard_path = self.output_dir / self.config.output["leaderboard"]
        df.to_parquet(leaderboard_path)
        print(f"\nSaved leaderboard to {leaderboard_path}")
        
        # Save skipped models
        if len(df_skipped) > 0:
            skipped_path = self.output_dir / "skipped_models.parquet"
            df_skipped.to_parquet(skipped_path)
            print(f"Saved skipped models to {skipped_path}")
        
        # Save summary JSON
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "seed": self.seed,
            "n_results": len(self.results),
            "n_skipped": len(self.skipped_models),
            "models_run": list(df["model_name"].unique()) if "model_name" in df.columns else [],
            "models_skipped": list(df_skipped["model_name"].unique()) if len(df_skipped) > 0 else [],
            "ablations_run": list(df["ablation"].unique()) if "ablation" in df.columns else [],
            "targets_run": list(df["target"].unique()) if "target" in df.columns else [],
            "horizons_run": [int(h) for h in df["horizon"].unique()] if "horizon" in df.columns else [],
        }
        
        summary_path = self.output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path}")
        
        # Print leaderboard
        print("\n" + "=" * 80)
        print("Leaderboard (by MAE)")
        print("=" * 80)
        
        if len(df) > 0 and "mae" in df.columns:
            leaderboard = df.dropna(subset=["mae"]).sort_values("mae")
            display_cols = ["model_name", "ablation", "horizon", "target", "mae", "rmse"]
            if "mae_ci_lower" in df.columns:
                display_cols.extend(["mae_ci_lower", "mae_ci_upper"])
            print(leaderboard[display_cols].head(20).to_string(index=False))
        
        # Print skipped models table
        if len(df_skipped) > 0:
            print("\n" + "=" * 80)
            print("Skipped Models")
            print("=" * 80)
            print(df_skipped.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Block 3 Benchmark Harness")
    parser.add_argument("--config", type=Path, default=Path("configs/block3.yaml"))
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test")
    parser.add_argument("--verify-first", action="store_true", default=True, help="Verify freeze before running")
    parser.add_argument("--task", type=str, help="Run specific task (outcome, forecast, risk, narrative)")
    parser.add_argument("--models", nargs="+", help="Models to run")
    parser.add_argument("--ablations", nargs="+", help="Ablation configs to run")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--preset", choices=["smoke_test", "quick", "standard", "comprehensive"],
                        help="Use a preset model configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--compute-ci", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()
    
    # List modes (no data loading)
    if args.list_models:
        print("Available Models by Category:")
        print("=" * 60)
        for cat, models in list_models().items():
            print(f"\n{cat}:")
            for m in models:
                avail = "✓" if check_model_available(m) else "✗"
                print(f"  [{avail}] {m}")
        return
    
    if args.list_tasks:
        print("Available Tasks:")
        print("=" * 60)
        for task_name in list_tasks():
            task = get_task(task_name)
            print(f"\n{task_name}: {task.description}")
        return
    
    # Set global seed
    set_global_seed(args.seed)
    
    # Load config
    config = BenchmarkConfig.load(args.config)
    
    # Verify freeze gates first
    if args.verify_first:
        print("Verifying freeze gates...")
        result = subprocess.run(
            [sys.executable, "scripts/block3_verify_freeze.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Freeze verification FAILED:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        print("Freeze verification PASSED\n")
    
    # Load dataset
    dataset = Block3Dataset.from_pointer(Path(config.data["pointer"]))
    
    # Create harness with reproducibility settings
    harness = BenchmarkHarness(
        config,
        dataset,
        seed=args.seed,
        compute_ci=args.compute_ci,
        n_bootstrap=args.n_bootstrap,
    )
    
    # Determine models from preset or args
    model_names = args.models
    if args.preset:
        preset_models = get_preset_models(args.preset)
        model_names = [m.config.name for m in preset_models]
        print(f"Using preset '{args.preset}': {model_names}")
    
    # Run benchmark
    if args.smoke_test:
        harness.run_smoke_test()
    elif args.task:
        harness.run_task_benchmark(
            task_name=args.task,
            model_names=model_names,
            ablation_names=args.ablations,
        )
    else:
        harness.run_benchmark(
            model_names=model_names,
            ablation_names=args.ablations,
        )


if __name__ == "__main__":
    main()
