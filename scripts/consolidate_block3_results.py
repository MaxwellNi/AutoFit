#!/usr/bin/env python3
"""
Consolidate Block 3 benchmark results and generate paper-ready tables.

This script:
1. Scans all shard outputs under input-dir
2. Validates MANIFEST.json for each shard
3. Consolidates metrics.json files
4. Generates Markdown tables for each task
5. Produces robustness analysis for Task 3

Usage:
    python consolidate_block3_results.py --input-dir runs/benchmarks/block3_standard_YYYYMMDD_HHMMSS --output-dir results/paper_tables
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.narrative.evaluation.report import (
    consolidate_shards,
    generate_task_table,
    generate_robustness_table,
    check_reproducibility,
    build_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_shards(input_dir: Path) -> tuple[list[Path], list[Path]]:
    """Validate all shards and return valid/invalid paths."""
    valid = []
    invalid = []
    
    for manifest_path in input_dir.rglob("MANIFEST.json"):
        shard_dir = manifest_path.parent
        metrics_path = shard_dir / "metrics.json"
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Check status
            if manifest.get("status") != "completed":
                logger.warning(f"Shard {shard_dir} not completed: {manifest.get('status')}")
                invalid.append(shard_dir)
                continue
            
            # Check metrics exist
            if not metrics_path.exists():
                logger.warning(f"Shard {shard_dir} missing metrics.json")
                invalid.append(shard_dir)
                continue
            
            valid.append(shard_dir)
            
        except Exception as e:
            logger.warning(f"Error validating {shard_dir}: {e}")
            invalid.append(shard_dir)
    
    return valid, invalid


def load_all_metrics(shard_dirs: list[Path]) -> pd.DataFrame:
    """Load and concatenate all metrics from valid shards."""
    all_metrics = []
    
    for shard_dir in shard_dirs:
        metrics_path = shard_dir / "metrics.json"
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            all_metrics.extend(metrics)
        except Exception as e:
            logger.warning(f"Error loading metrics from {shard_dir}: {e}")
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)


def generate_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for the consolidated results."""
    if df.empty:
        return {}
    
    return {
        "total_experiments": len(df),
        "unique_models": df["model_name"].nunique(),
        "unique_tasks": df["task"].nunique(),
        "unique_ablations": df["ablation"].nunique(),
        "tasks": df["task"].unique().tolist(),
        "ablations": df["ablation"].unique().tolist(),
        "categories": df["category"].unique().tolist(),
        "best_model_by_mae": df.loc[df["mae"].idxmin()].to_dict() if "mae" in df.columns else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate Block 3 benchmark results"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark shard outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for consolidated outputs",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.05,
        help="Variance threshold for reproducibility check (default: 5%%)",
    )
    args = parser.parse_args()
    
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scanning shards in: {input_dir}")
    
    # Validate shards
    valid_shards, invalid_shards = validate_shards(input_dir)
    logger.info(f"Valid shards: {len(valid_shards)}, Invalid shards: {len(invalid_shards)}")
    
    if not valid_shards:
        logger.error("No valid shards found!")
        sys.exit(1)
    
    # Load all metrics
    df = load_all_metrics(valid_shards)
    logger.info(f"Loaded {len(df)} metric records")
    
    if df.empty:
        logger.error("No metrics loaded!")
        sys.exit(1)
    
    # Save consolidated metrics
    metrics_csv = output_dir / "consolidated_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved consolidated metrics to: {metrics_csv}")
    
    # Generate summary statistics
    summary = generate_summary_stats(df)
    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary stats to: {summary_path}")
    
    # Generate task tables
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        table = generate_task_table(task_df, task_name=task)
        table_path = output_dir / f"{task}_main.md"
        with open(table_path, "w") as f:
            f.write(table)
        logger.info(f"Saved table to: {table_path}")
    
    # Generate robustness table for Task 3
    if "task3_risk_adjust" in df["task"].values:
        task3_df = df[df["task"] == "task3_risk_adjust"]
        robustness_table = generate_robustness_table(task3_df)
        robustness_path = output_dir / "task3_robustness.md"
        with open(robustness_path, "w") as f:
            f.write(robustness_table)
        logger.info(f"Saved robustness table to: {robustness_path}")
    
    # Check reproducibility
    repro_issues = check_reproducibility(df, variance_threshold=args.variance_threshold)
    if repro_issues:
        logger.warning(f"Reproducibility issues found: {len(repro_issues)}")
        repro_path = output_dir / "reproducibility_issues.json"
        with open(repro_path, "w") as f:
            json.dump(repro_issues, f, indent=2)
        logger.info(f"Saved reproducibility issues to: {repro_path}")
    else:
        logger.info("No reproducibility issues found")
    
    # Generate final report
    report = build_report(df, output_dir)
    report_path = output_dir / "BENCHMARK_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved final report to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"Total experiments: {summary.get('total_experiments', 0)}")
    print(f"Unique models: {summary.get('unique_models', 0)}")
    print(f"Tasks: {', '.join(summary.get('tasks', []))}")
    print(f"Ablations: {', '.join(summary.get('ablations', []))}")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
