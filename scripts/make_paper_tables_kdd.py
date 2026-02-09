#!/usr/bin/env python3
"""
KDD'26 Paper Table Generator for Block 3 Benchmark.

Reads all completed benchmark results (metrics.json files) and produces:
  - Table 1: Main results (all models × tasks × targets, MAE/RMSE/SMAPE)
  - Table 2: Per-horizon breakdown
  - Table 3: Ablation study (core_only vs core_edgar)
  - Table 4: AutoFit v2 routing analysis
  - Table 5: Category-level summary (best-in-class per category)
  - LaTeX output for paper inclusion

Usage:
    python scripts/make_paper_tables_kdd.py \\
        --results-dir runs/benchmarks/block3_20260203_225620_iris_full \\
        --output-dir docs/paper_tables
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Load all metrics.json files from the results directory."""
    all_records = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        # Extract category and ablation from path
        # Expected: results_dir/task/category/ablation/metrics.json
        rel = metrics_path.relative_to(results_dir)
        parts = list(rel.parts)
        if len(parts) < 4:
            logger.warning(f"Unexpected path structure: {metrics_path}")
            continue

        task = parts[0]
        category = parts[1]
        ablation = parts[2]

        # Load MANIFEST to get git hash and status
        manifest_path = metrics_path.parent / "MANIFEST.json"
        git_hash = "unknown"
        manifest_status = "unknown"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                git_hash = manifest.get("git_hash", "unknown")
                manifest_status = manifest.get("status", "unknown")
            except Exception:
                pass

        # Only include completed runs
        if manifest_status != "completed":
            logger.info(f"Skipping incomplete shard: {rel.parent} (status={manifest_status})")
            continue

        try:
            records = json.loads(metrics_path.read_text(encoding="utf-8"))
            for r in records:
                r["_task"] = task
                r["_category"] = category
                r["_ablation"] = ablation
                r["_git_hash"] = git_hash
            all_records.extend(records)
        except Exception as e:
            logger.warning(f"Failed to load {metrics_path}: {e}")

    if not all_records:
        logger.error("No metric records found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Loaded {len(df)} metric records from {results_dir}")

    # Standardise column names
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"]
    if "category" not in df.columns and "_category" in df.columns:
        df["category"] = df["_category"]
    if "task" not in df.columns and "_task" in df.columns:
        df["task"] = df["_task"]

    return df


# =============================================================================
# Table Builders
# =============================================================================

def table1_main_results(df: pd.DataFrame, ablation: str = "core_only") -> pd.DataFrame:
    """
    Table 1: Main results — models × targets, averaged across horizons.
    Uses core_only ablation for fair comparison.
    """
    sub = df[df["_ablation"] == ablation].copy()
    if sub.empty:
        # Try whatever ablation is available
        sub = df.copy()
        logger.warning(f"No records for ablation={ablation}, using all")

    # Group by model, category, task, target → average across horizons
    group_cols = ["model_name", "category", "task", "target"]
    agg = sub.groupby(group_cols).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        smape=("smape", "mean"),
        mape=("mape", "mean"),
        n_horizons=("horizon", "nunique"),
        train_time=("train_time_seconds", "mean"),
    ).reset_index()

    return agg.sort_values(["task", "target", "mae"])


def table2_per_horizon(df: pd.DataFrame, ablation: str = "core_only") -> pd.DataFrame:
    """
    Table 2: Per-horizon breakdown for the main target (funding_raised_usd).
    """
    sub = df[
        (df["_ablation"] == ablation) &
        (df["target"] == "funding_raised_usd")
    ].copy()

    if sub.empty:
        sub = df[df["target"] == "funding_raised_usd"].copy()

    group_cols = ["model_name", "category", "task", "horizon"]
    agg = sub.groupby(group_cols).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        smape=("smape", "mean"),
    ).reset_index()

    return agg.sort_values(["task", "horizon", "mae"])


def table3_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Ablation study — core_only vs core_edgar.
    Shows improvement (or not) from adding EDGAR features.
    """
    core = df[df["_ablation"] == "core_only"].copy()
    edgar = df[df["_ablation"] == "core_edgar"].copy()

    if core.empty or edgar.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "category", "task", "target"]

    core_agg = core.groupby(group_cols).agg(
        mae_core=("mae", "mean"),
        rmse_core=("rmse", "mean"),
    ).reset_index()

    edgar_agg = edgar.groupby(group_cols).agg(
        mae_edgar=("mae", "mean"),
        rmse_edgar=("rmse", "mean"),
    ).reset_index()

    merged = core_agg.merge(edgar_agg, on=group_cols, how="outer")
    merged["mae_improvement_pct"] = (
        (merged["mae_core"] - merged["mae_edgar"]) / merged["mae_core"] * 100
    )
    merged["rmse_improvement_pct"] = (
        (merged["rmse_core"] - merged["rmse_edgar"]) / merged["rmse_core"] * 100
    )

    return merged.sort_values(["task", "target", "mae_improvement_pct"], ascending=[True, True, False])


def table4_category_summary(df: pd.DataFrame, ablation: str = "core_only") -> pd.DataFrame:
    """
    Table 4: Best-in-class per category — one row per (category, task, target).
    """
    sub = df[df["_ablation"] == ablation].copy()
    if sub.empty:
        sub = df.copy()

    # For each (category, task, target), find the best model by MAE
    group_cols = ["category", "task", "target"]
    results = []
    for keys, grp in sub.groupby(group_cols):
        cat, task, target = keys
        avg = grp.groupby("model_name").agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            train_time=("train_time_seconds", "mean"),
        ).reset_index()
        best = avg.loc[avg["mae"].idxmin()]
        results.append({
            "category": cat,
            "task": task,
            "target": target,
            "best_model": best["model_name"],
            "mae": best["mae"],
            "rmse": best["rmse"],
            "smape": best["smape"],
            "train_time": best["train_time"],
        })

    return pd.DataFrame(results).sort_values(["task", "target", "mae"])


def table5_autofit_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 5: AutoFit v2 vs best baselines.
    Compare AutoFit models against the best model in each category.
    """
    autofit = df[df["category"] == "autofit"].copy()
    baselines = df[df["category"] != "autofit"].copy()

    if autofit.empty:
        return pd.DataFrame()

    group_cols = ["task", "target"]
    results = []
    for keys, grp in baselines.groupby(group_cols):
        task, target = keys
        # Best baseline by average MAE
        best_base = grp.groupby(["model_name", "category"]).agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
        ).reset_index()
        if best_base.empty:
            continue
        best = best_base.loc[best_base["mae"].idxmin()]

        # AutoFit results
        af_sub = autofit[(autofit["task"] == task) & (autofit["target"] == target)]
        for model_name, af_grp in af_sub.groupby("model_name"):
            af_mae = af_grp["mae"].mean()
            af_rmse = af_grp["rmse"].mean()
            results.append({
                "task": task,
                "target": target,
                "autofit_model": model_name,
                "autofit_mae": af_mae,
                "autofit_rmse": af_rmse,
                "best_baseline": best["model_name"],
                "best_baseline_cat": best["category"],
                "baseline_mae": best["mae"],
                "baseline_rmse": best["rmse"],
                "mae_vs_best_pct": (af_mae - best["mae"]) / best["mae"] * 100,
                "rmse_vs_best_pct": (af_rmse - best["rmse"]) / best["rmse"] * 100,
            })

    return pd.DataFrame(results)


# =============================================================================
# LaTeX Formatters
# =============================================================================

def _bold_best(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Bold the best value in a series."""
    if lower_is_better:
        best = series.min()
    else:
        best = series.max()
    return series.apply(lambda x: f"\\textbf{{{x:.2f}}}" if abs(x - best) < 1e-8 else f"{x:.2f}")


def to_latex_main_table(df: pd.DataFrame, task: str, target: str) -> str:
    """Generate LaTeX for Table 1 (one task × target)."""
    sub = df[(df["task"] == task) & (df["target"] == target)].copy()
    if sub.empty:
        return f"% No data for {task} / {target}\n"

    sub = sub.sort_values("mae")

    lines = []
    lines.append(f"% Table: {task} — {target}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Results for \\texttt{{{task}}} — \\texttt{{{target}}} (averaged across horizons)}}")
    lines.append(f"\\label{{tab:{task}_{target}}}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Category & Model & MAE$\downarrow$ & RMSE$\downarrow$ & SMAPE$\downarrow$ & Time (s) \\")
    lines.append(r"\midrule")

    best_mae = sub["mae"].min()
    best_rmse = sub["rmse"].min()
    best_smape = sub["smape"].min() if sub["smape"].notna().any() else None

    for _, row in sub.iterrows():
        cat = row["category"]
        model = row["model_name"].replace("_", r"\_")
        mae_str = f"\\textbf{{{row['mae']:.2f}}}" if abs(row["mae"] - best_mae) < 1e-6 else f"{row['mae']:.2f}"
        rmse_str = f"\\textbf{{{row['rmse']:.2f}}}" if abs(row["rmse"] - best_rmse) < 1e-6 else f"{row['rmse']:.2f}"
        smape_val = row.get("smape")
        if pd.notna(smape_val) and best_smape is not None:
            smape_str = f"\\textbf{{{smape_val:.2f}}}" if abs(smape_val - best_smape) < 1e-6 else f"{smape_val:.2f}"
        else:
            smape_str = "—"
        time_str = f"{row.get('train_time', 0):.1f}" if pd.notna(row.get("train_time")) else "—"

        lines.append(f"  {cat} & {model} & {mae_str} & {rmse_str} & {smape_str} & {time_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def to_latex_horizon_table(df: pd.DataFrame, task: str) -> str:
    """Generate LaTeX for per-horizon table."""
    sub = df[df["task"] == task].copy()
    if sub.empty:
        return f"% No horizon data for {task}\n"

    horizons = sorted(sub["horizon"].unique())
    models = sub.groupby("model_name")["mae"].mean().sort_values().index.tolist()

    lines = []
    lines.append(f"% Horizon breakdown: {task} — funding_raised_usd")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(f"\\caption{{Per-horizon MAE for \\texttt{{{task}}} — \\texttt{{funding\\_raised\\_usd}}}}")
    lines.append(f"\\label{{tab:{task}_horizons}}")
    h_cols = " ".join([f"c" for _ in horizons])
    lines.append(f"\\begin{{tabular}}{{ll{h_cols}}}")
    lines.append(r"\toprule")
    h_header = " & ".join([f"h={h}" for h in horizons])
    lines.append(f"Category & Model & {h_header} \\\\")
    lines.append(r"\midrule")

    for model in models[:20]:  # Top 20 models
        row_data = sub[sub["model_name"] == model]
        cat = row_data["category"].iloc[0]
        model_esc = model.replace("_", r"\_")
        vals = []
        for h in horizons:
            h_data = row_data[row_data["horizon"] == h]
            if not h_data.empty:
                vals.append(f"{h_data['mae'].iloc[0]:.2f}")
            else:
                vals.append("—")
        lines.append(f"  {cat} & {model_esc} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Markdown Summary
# =============================================================================

def generate_markdown_summary(df: pd.DataFrame) -> str:
    """Generate a Markdown summary of all results."""
    lines = []
    lines.append("# Block 3 KDD'26 Benchmark Results\n")
    lines.append(f"**Total records**: {len(df)}")
    lines.append(f"**Categories**: {sorted(df['category'].unique().tolist())}")
    lines.append(f"**Tasks**: {sorted(df['task'].unique().tolist())}")
    lines.append(f"**Models**: {df['model_name'].nunique()}")
    lines.append(f"**Targets**: {sorted(df['target'].unique().tolist())}")
    lines.append("")

    # Per-task summary
    for task in sorted(df["task"].unique()):
        lines.append(f"\n## {task}\n")
        task_df = df[df["task"] == task]

        for target in sorted(task_df["target"].unique()):
            lines.append(f"\n### Target: `{target}`\n")
            target_df = task_df[task_df["target"] == target]

            # Aggregate by model
            agg = target_df.groupby(["model_name", "category"]).agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                smape=("smape", "mean"),
                n=("mae", "count"),
            ).reset_index().sort_values("mae")

            lines.append("| Category | Model | MAE↓ | RMSE↓ | SMAPE↓ | N |")
            lines.append("|----------|-------|------|-------|--------|---|")

            for _, row in agg.head(30).iterrows():
                smape = f"{row['smape']:.2f}" if pd.notna(row["smape"]) else "—"
                lines.append(
                    f"| {row['category']} | {row['model_name']} | "
                    f"{row['mae']:.2f} | {row['rmse']:.2f} | {smape} | {int(row['n'])} |"
                )
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KDD'26 Paper Table Generator")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs/benchmarks/block3_20260203_225620_iris_full"),
        help="Root results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/paper_tables"),
        help="Output directory for tables",
    )
    parser.add_argument(
        "--ablation",
        default="core_only",
        help="Ablation to use for main tables",
    )
    args = parser.parse_args()

    # Load all metrics
    df = load_all_metrics(args.results_dir)
    if df.empty:
        logger.error("No data loaded, exiting")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1: Main results ---
    t1 = table1_main_results(df, ablation=args.ablation)
    t1.to_csv(args.output_dir / "table1_main_results.csv", index=False)
    logger.info(f"Table 1: {len(t1)} rows saved")

    # Generate LaTeX per task × target
    latex_all = []
    for task in sorted(df["task"].unique()):
        for target in sorted(df[df["task"] == task]["target"].unique()):
            latex = to_latex_main_table(t1, task, target)
            latex_all.append(latex)
    (args.output_dir / "table1_main.tex").write_text("\n".join(latex_all), encoding="utf-8")

    # --- Table 2: Per-horizon ---
    t2 = table2_per_horizon(df, ablation=args.ablation)
    t2.to_csv(args.output_dir / "table2_per_horizon.csv", index=False)
    logger.info(f"Table 2: {len(t2)} rows saved")

    latex_h = []
    for task in sorted(df["task"].unique()):
        latex_h.append(to_latex_horizon_table(t2, task))
    (args.output_dir / "table2_horizons.tex").write_text("\n".join(latex_h), encoding="utf-8")

    # --- Table 3: Ablation ---
    t3 = table3_ablation(df)
    if not t3.empty:
        t3.to_csv(args.output_dir / "table3_ablation.csv", index=False)
        logger.info(f"Table 3: {len(t3)} rows saved")
    else:
        logger.warning("Table 3 (ablation): no dual-ablation data available")

    # --- Table 4: Category summary ---
    t4 = table4_category_summary(df, ablation=args.ablation)
    t4.to_csv(args.output_dir / "table4_category_summary.csv", index=False)
    logger.info(f"Table 4: {len(t4)} rows saved")

    # --- Table 5: AutoFit analysis ---
    t5 = table5_autofit_analysis(df)
    if not t5.empty:
        t5.to_csv(args.output_dir / "table5_autofit_vs_baselines.csv", index=False)
        logger.info(f"Table 5: {len(t5)} rows saved")
    else:
        logger.warning("Table 5 (AutoFit): no autofit results yet")

    # --- Markdown summary ---
    md = generate_markdown_summary(df)
    md_path = args.output_dir / "RESULTS_SUMMARY.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Markdown summary saved to {md_path}")

    # --- Quick leaderboard to stdout ---
    print("\n" + "=" * 80)
    print("LEADERBOARD (core_only, averaged across horizons)")
    print("=" * 80)

    for task in sorted(df["task"].unique()):
        print(f"\n{'─' * 60}")
        print(f"  {task}")
        print(f"{'─' * 60}")
        t1_task = t1[t1["task"] == task]
        for target in sorted(t1_task["target"].unique()):
            print(f"\n  Target: {target}")
            t1_tt = t1_task[t1_task["target"] == target].sort_values("mae").head(10)
            print(f"  {'Rank':<5} {'Category':<18} {'Model':<25} {'MAE':<12} {'RMSE':<12} {'SMAPE':<10}")
            for rank, (_, row) in enumerate(t1_tt.iterrows(), 1):
                smape = f"{row['smape']:.2f}" if pd.notna(row["smape"]) else "—"
                print(f"  {rank:<5} {row['category']:<18} {row['model_name']:<25} {row['mae']:<12.2f} {row['rmse']:<12.2f} {smape:<10}")

    print(f"\n\nAll tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
