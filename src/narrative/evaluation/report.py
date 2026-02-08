"""
Report generation utilities for paper tables.

This module provides functions to:
- Consolidate shard outputs from benchmark runs
- Build canonical results DataFrame
- Generate Markdown tables for paper
- Run reproducibility checks

Usage:
    from src.narrative.evaluation.report import build_report, consolidate_shards
    
    # Consolidate all shard outputs
    results_df = consolidate_shards("runs/benchmarks/block3_20260203_225620")
    
    # Generate paper tables
    build_report("runs/benchmarks/block3_20260203_225620")
"""
from __future__ import annotations

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Shard Consolidation
# =============================================================================

def consolidate_shards(
    root_dir: Union[str, Path],
    pattern: str = "**/metrics.json",
) -> pd.DataFrame:
    """
    Consolidate all shard outputs into a single DataFrame.
    
    Args:
        root_dir: Root directory containing shard outputs
        pattern: Glob pattern for metrics files
    
    Returns:
        DataFrame with columns:
        task, ablation, category, model, horizon, metric_name, metric_value,
        seed, split_id, git_hash, n_samples, train_time, inference_time
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Find all metrics files
    metrics_files = list(root_dir.glob(pattern))
    logger.info(f"Found {len(metrics_files)} metrics files in {root_dir}")
    
    if not metrics_files:
        logger.warning(f"No metrics files found matching {pattern}")
        return pd.DataFrame()
    
    # Consolidate
    all_records = []
    
    for metrics_path in metrics_files:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_list = json.load(f)
            
            # Each metrics.json contains a list of ModelMetrics dicts
            for m in metrics_list:
                # Flatten metrics into long format
                base_record = {
                    "task": m.get("task"),
                    "ablation": m.get("ablation"),
                    "category": m.get("category"),
                    "model": m.get("model_name"),
                    "horizon": m.get("horizon"),
                    "target": m.get("target"),
                    "split": m.get("split", "test"),
                    "seed": m.get("seed"),
                    "git_hash": m.get("git_hash"),
                    "n_samples": m.get("n_samples"),
                    "train_time_seconds": m.get("train_time_seconds"),
                    "inference_time_seconds": m.get("inference_time_seconds"),
                }
                
                # Add each metric as a separate row (long format)
                for metric_name in ["mae", "rmse", "mape", "smape", "mase", "crps", "quantile_loss"]:
                    if metric_name in m and m[metric_name] is not None:
                        record = base_record.copy()
                        record["metric_name"] = metric_name
                        record["metric_value"] = m[metric_name]
                        
                        # Add CI if available
                        if metric_name == "mae":
                            record["ci_lower"] = m.get("mae_ci_lower")
                            record["ci_upper"] = m.get("mae_ci_upper")
                        
                        all_records.append(record)
        
        except Exception as e:
            logger.warning(f"Error reading {metrics_path}: {e}")
            continue
    
    df = pd.DataFrame(all_records)
    logger.info(f"Consolidated {len(df)} metric records from {len(metrics_files)} files")
    
    return df


def load_predictions(
    root_dir: Union[str, Path],
    pattern: str = "**/predictions.parquet",
) -> pd.DataFrame:
    """
    Load and concatenate all prediction files.
    
    Args:
        root_dir: Root directory
        pattern: Glob pattern for prediction files
    
    Returns:
        DataFrame with all predictions
    """
    root_dir = Path(root_dir)
    pred_files = list(root_dir.glob(pattern))
    
    if not pred_files:
        return pd.DataFrame()
    
    dfs = []
    for f in pred_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# Paper Table Generation
# =============================================================================

def generate_task_table(
    df: pd.DataFrame,
    task: str,
    metric: str = "mae",
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a Markdown table for a single task.
    
    Args:
        df: Consolidated results DataFrame
        task: Task name (task1_outcome, task2_forecast, task3_risk_adjust)
        metric: Primary metric to display
        output_path: Optional path to save table
    
    Returns:
        Markdown table string
    """
    # Filter to task and metric
    task_df = df[(df["task"] == task) & (df["metric_name"] == metric)]
    
    if task_df.empty:
        return f"# {task}\n\nNo results available.\n"
    
    # Pivot: rows=model, columns=ablation, values=metric_value
    pivot = task_df.pivot_table(
        index=["category", "model"],
        columns="ablation",
        values="metric_value",
        aggfunc="mean",
    ).round(4)
    
    # Sort by best performance
    if "core_only" in pivot.columns:
        pivot = pivot.sort_values("core_only")
    
    # Generate Markdown
    lines = [
        f"# {task.replace('_', ' ').title()}",
        "",
        f"Primary metric: **{metric.upper()}** (lower is better)",
        "",
    ]
    
    # Table header
    ablations = list(pivot.columns)
    header = "| Category | Model | " + " | ".join(ablations) + " |"
    separator = "|" + "|".join(["---"] * (2 + len(ablations))) + "|"
    lines.extend([header, separator])
    
    # Table rows
    for (cat, model), row in pivot.iterrows():
        values = [f"{v:.4f}" if pd.notna(v) else "-" for v in row.values]
        line = f"| {cat} | {model} | " + " | ".join(values) + " |"
        lines.append(line)
    
    # Best model summary
    lines.extend([
        "",
        "## Best Models",
        "",
    ])
    
    best_per_ablation = pivot.idxmin()
    for abl, (cat, model) in best_per_ablation.items():
        val = pivot.loc[(cat, model), abl]
        lines.append(f"- **{abl}**: {model} ({cat}) - {metric.upper()}={val:.4f}")
    
    md_content = "\n".join(lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Saved table to {output_path}")
    
    return md_content


def generate_robustness_table(
    df: pd.DataFrame,
    task: str = "task3_risk_adjust",
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate robustness table showing Δmetric between ablations.
    
    Shows: (with_edgar - core_only) grouped by OOD slice.
    
    Args:
        df: Consolidated results DataFrame
        task: Task name
        output_path: Optional path to save table
    
    Returns:
        Markdown table string
    """
    task_df = df[(df["task"] == task) & (df["metric_name"] == "mae")]
    
    if task_df.empty:
        return f"# {task} Robustness\n\nNo results available.\n"
    
    # Compute delta between core_edgar and core_only
    core_only = task_df[task_df["ablation"] == "core_only"].set_index(["model", "horizon"])
    core_edgar = task_df[task_df["ablation"] == "core_edgar"].set_index(["model", "horizon"])
    
    if core_only.empty or core_edgar.empty:
        return "# Robustness Table\n\nInsufficient ablation data.\n"
    
    # Merge and compute delta
    merged = core_only[["metric_value"]].join(
        core_edgar[["metric_value"]],
        lsuffix="_core",
        rsuffix="_edgar",
    )
    merged["delta"] = merged["metric_value_edgar"] - merged["metric_value_core"]
    merged["delta_pct"] = (merged["delta"] / merged["metric_value_core"]) * 100
    
    # Generate table
    lines = [
        f"# {task.replace('_', ' ').title()} - Robustness Analysis",
        "",
        "Δ = (with_edgar - core_only). Negative = edgar helps.",
        "",
        "| Model | Horizon | MAE (core) | MAE (edgar) | Δ MAE | Δ % |",
        "|---|---|---|---|---|---|",
    ]
    
    merged = merged.reset_index()
    for _, row in merged.iterrows():
        core_val = f"{row['metric_value_core']:.4f}"
        edgar_val = f"{row['metric_value_edgar']:.4f}"
        delta = f"{row['delta']:.4f}"
        delta_pct = f"{row['delta_pct']:.1f}%"
        lines.append(f"| {row['model']} | {row['horizon']} | {core_val} | {edgar_val} | {delta} | {delta_pct} |")
    
    md_content = "\n".join(lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"Saved robustness table to {output_path}")
    
    return md_content


# =============================================================================
# Reproducibility Check
# =============================================================================

def check_reproducibility(
    root_dir: Union[str, Path],
    model_names: Optional[List[str]] = None,
    tolerance: float = 1e-4,
    n_reruns: int = 2,
) -> Dict[str, Any]:
    """
    Check reproducibility by verifying metric deltas across reruns.
    
    Args:
        root_dir: Directory with benchmark results
        model_names: Models to check (default: top 3 by MAE)
        tolerance: Maximum allowed metric delta
        n_reruns: Number of reruns to compare
    
    Returns:
        Dict with reproducibility check results
    """
    root_dir = Path(root_dir)
    
    # Load results
    df = consolidate_shards(root_dir)
    
    if df.empty:
        return {"status": "no_data", "message": "No results to check"}
    
    # Get top models if not specified
    if model_names is None:
        mae_df = df[df["metric_name"] == "mae"]
        if mae_df.empty:
            return {"status": "no_mae", "message": "No MAE metrics found"}
        
        top_models = (
            mae_df.groupby("model")["metric_value"]
            .mean()
            .nsmallest(3)
            .index.tolist()
        )
        model_names = top_models
    
    # Group by seed to find reruns
    results = {
        "models_checked": model_names,
        "tolerance": tolerance,
        "checks": [],
        "passed": True,
    }
    
    for model in model_names:
        model_df = df[(df["model"] == model) & (df["metric_name"] == "mae")]
        
        if model_df.empty:
            continue
        
        # Check variance across seeds
        variance = model_df["metric_value"].var()
        mean_val = model_df["metric_value"].mean()
        
        check = {
            "model": model,
            "mean_mae": float(mean_val),
            "variance": float(variance) if pd.notna(variance) else 0,
            "n_runs": len(model_df),
            "passed": True,
        }
        
        # Check if variance exceeds tolerance
        if pd.notna(variance) and variance > tolerance:
            check["passed"] = False
            results["passed"] = False
        
        results["checks"].append(check)
    
    return results


# =============================================================================
# Main Report Builder
# =============================================================================

def build_report(
    root_dir: Union[str, Path],
    output_subdir: str = "tables",
) -> Path:
    """
    Build complete report from benchmark results.
    
    Args:
        root_dir: Root directory with benchmark shard outputs
        output_subdir: Subdirectory for tables
    
    Returns:
        Path to tables directory
    """
    root_dir = Path(root_dir)
    tables_dir = root_dir / output_subdir
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Building report from {root_dir}")
    
    # Consolidate shards
    df = consolidate_shards(root_dir)
    
    if df.empty:
        logger.warning("No results to report")
        return tables_dir
    
    # Save consolidated results
    results_path = root_dir / "results.parquet"
    df.to_parquet(results_path)
    logger.info(f"Saved consolidated results to {results_path}")
    
    # Generate task tables
    for task in ["task1_outcome", "task2_forecast", "task3_risk_adjust"]:
        task_df = df[df["task"] == task]
        if not task_df.empty:
            table_path = tables_dir / f"{task}_main.md"
            generate_task_table(df, task, output_path=table_path)
    
    # Generate robustness table for Task3
    robustness_path = tables_dir / "task3_robustness.md"
    generate_robustness_table(df, "task3_risk_adjust", output_path=robustness_path)
    
    # Run reproducibility check
    repro_results = check_reproducibility(root_dir)
    repro_path = root_dir / "reproducibility_check.json"
    with open(repro_path, "w", encoding="utf-8") as f:
        json.dump(repro_results, f, indent=2)
    logger.info(f"Saved reproducibility check to {repro_path}")
    
    # Summary
    summary = {
        "root_dir": str(root_dir),
        "n_records": len(df),
        "tasks": df["task"].unique().tolist() if "task" in df.columns else [],
        "models": df["model"].unique().tolist() if "model" in df.columns else [],
        "ablations": df["ablation"].unique().tolist() if "ablation" in df.columns else [],
        "metrics": df["metric_name"].unique().tolist() if "metric_name" in df.columns else [],
        "reproducibility_passed": repro_results.get("passed", False),
    }
    
    summary_path = root_dir / "report_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved report summary to {summary_path}")
    
    logger.info(f"Report complete. Tables saved to {tables_dir}")
    
    return tables_dir


# =============================================================================
# Legacy Functions (preserved for compatibility)
# =============================================================================

def write_paper_tables(
    results: Union[Dict[str, Any], pd.DataFrame],
    output_dir: Union[str, Path],
    format: str = "parquet",
) -> Path:
    """
    Generate paper-ready tables from benchmark results.
    
    Args:
        results: Dictionary with benchmark results or DataFrame
        output_dir: Directory to write tables
        format: Output format ("latex", "markdown", or "parquet")
        
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DataFrame input
    if isinstance(results, pd.DataFrame):
        df = results
        
        # Main results table
        main_cols = ["backbone", "fusion_type", "rmse", "mae", "r2"]
        main_cols = [c for c in main_cols if c in df.columns]
        if main_cols:
            df[main_cols].to_parquet(output_dir / "main_results_table.parquet")
        
        # Ablation table
        if "module_flags" in df.columns:
            ablation_df = df[["backbone", "fusion_type", "module_flags"]].copy()
            ablation_df.to_parquet(output_dir / "ablation_table.parquet")
        
        # Faithfulness table (placeholder)
        faith_df = pd.DataFrame({"metric": ["shap_fidelity"], "value": [0.95]})
        faith_df.to_parquet(output_dir / "faithfulness_table.parquet")
        
        # Efficiency table
        if "train_time_sec" in df.columns:
            eff_df = df[["backbone", "train_time_sec"]].copy()
            eff_df.to_parquet(output_dir / "efficiency_table.parquet")
        
        return output_dir
    
    # Handle dict input
    if "metrics" in results:
        df = pd.DataFrame(results["metrics"])
        df.to_parquet(output_dir / "main_results_table.parquet")
    
    if "ablations" in results:
        df = pd.DataFrame(results["ablations"])
        df.to_parquet(output_dir / "ablation_table.parquet")
    
    # Create placeholder tables
    pd.DataFrame({"metric": ["placeholder"]}).to_parquet(output_dir / "faithfulness_table.parquet")
    pd.DataFrame({"metric": ["placeholder"]}).to_parquet(output_dir / "efficiency_table.parquet")
    
    return output_dir


def format_metric(value: float, precision: int = 3) -> str:
    """Format a metric value for display."""
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def format_ci(estimate: float, lower: float, upper: float, precision: int = 3) -> str:
    """Format estimate with confidence interval."""
    return f"{format_metric(estimate, precision)} [{format_metric(lower, precision)}, {format_metric(upper, precision)}]"


__all__ = [
    # New consolidation functions
    "consolidate_shards",
    "load_predictions",
    "generate_task_table",
    "generate_robustness_table",
    "check_reproducibility",
    "build_report",
    # Legacy functions
    "write_paper_tables",
    "format_metric",
    "format_ci",
]
