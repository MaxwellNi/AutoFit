#!/usr/bin/env python3
"""
Block 3 Benchmark Results Aggregator.

Scans all metrics.json files in a benchmark output directory and produces:
  1. A consolidated CSV (all_results.csv)
  2. A Markdown summary table (docs/BLOCK3_RESULTS.md)
  3. A per-category leaderboard ranked by MAE

Usage:
    python scripts/aggregate_block3_results.py [--bench-dir DIR] [--md-output PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "runs" / "benchmarks" / "block3_20260203_225620_4090_final"
DEFAULT_MD = ROOT / "docs" / "BLOCK3_RESULTS.md"


def collect_metrics(bench_dir: Path) -> pd.DataFrame:
    """Recursively find all metrics.json and concatenate."""
    records = []
    for mf in sorted(bench_dir.rglob("metrics.json")):
        try:
            with open(mf) as f:
                data = json.load(f)
            for rec in data:
                rec["_source"] = str(mf.relative_to(bench_dir))
            records.extend(data)
        except Exception as e:
            print(f"  WARN: {mf}: {e}")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


def is_fallback(row) -> bool:
    """Detect fallback results (mean-predictor) by checking for the
    characteristic MAE value of ~601864 (the global mean)."""
    mae = row.get("mae", 0)
    # Fallback MAE for funding_raised_usd is approximately 601864
    # We flag anything within 1% of this value as likely fallback
    if 595000 < mae < 610000:
        return True
    return False


def generate_markdown(df: pd.DataFrame, bench_dir: Path) -> str:
    """Generate comprehensive Markdown report."""
    lines = []
    lines.append("# Block 3 Benchmark Results")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Benchmark Dir**: `{bench_dir.name}`")
    lines.append(f"**Total Records**: {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("_No results yet._")
        return "\n".join(lines)

    # Summary stats
    n_models = df["model_name"].nunique()
    n_tasks = df["task"].nunique()
    n_categories = df["category"].nunique()
    tasks = sorted(df["task"].unique())
    categories = sorted(df["category"].unique())

    # Detect fallback
    df["is_fallback"] = df.apply(is_fallback, axis=1)
    n_fallback = df["is_fallback"].sum()
    n_real = len(df) - n_fallback

    lines.append("## Overview")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Models evaluated | {n_models} |")
    lines.append(f"| Categories | {', '.join(categories)} |")
    lines.append(f"| Tasks | {', '.join(tasks)} |")
    lines.append(f"| Total evaluations | {len(df)} |")
    lines.append(f"| Real results | {n_real} |")
    lines.append(f"| Fallback (mean) | {n_fallback} |")
    lines.append("")

    # Per-task, per-category leaderboard
    for task in tasks:
        lines.append(f"## {task}")
        lines.append("")
        task_df = df[df["task"] == task]

        for cat in sorted(task_df["category"].unique()):
            cat_df = task_df[task_df["category"] == cat]
            lines.append(f"### {cat}")
            lines.append("")

            # Group by model and horizon, show MAE
            pivot = cat_df.pivot_table(
                index="model_name",
                columns="horizon",
                values="mae",
                aggfunc="first",
            )
            pivot = pivot.sort_values(by=pivot.columns[0] if len(pivot.columns) > 0 else pivot.index)

            # Mark fallback models
            fallback_models = set(
                cat_df[cat_df["is_fallback"]]["model_name"].unique()
            )

            # Format table
            horizons = sorted(pivot.columns)
            header = "| Model | " + " | ".join(f"H={h}" for h in horizons) + " | Status |"
            sep = "|-------|" + "|".join("-------:" for _ in horizons) + "|--------|"
            lines.append(header)
            lines.append(sep)

            for model_name in pivot.index:
                status = "⚠️ fallback" if model_name in fallback_models else "✅"
                vals = []
                for h in horizons:
                    v = pivot.loc[model_name, h] if h in pivot.columns else None
                    if pd.isna(v):
                        vals.append("—")
                    elif v > 1_000_000:
                        vals.append(f"{v/1e6:.2f}M")
                    elif v > 1_000:
                        vals.append(f"{v/1e3:.1f}K")
                    else:
                        vals.append(f"{v:.2f}")
                row = f"| {model_name} | " + " | ".join(vals) + f" | {status} |"
                lines.append(row)
            lines.append("")

    # Training time summary
    lines.append("## Training Time Summary")
    lines.append("")
    if "train_time_seconds" in df.columns:
        time_df = df.groupby("model_name")["train_time_seconds"].agg(["mean", "min", "max"])
        time_df = time_df.sort_values("mean", ascending=False)
        lines.append("| Model | Avg (s) | Min (s) | Max (s) |")
        lines.append("|-------|--------:|--------:|--------:|")
        for m, row in time_df.iterrows():
            lines.append(f"| {m} | {row['mean']:.1f} | {row['min']:.1f} | {row['max']:.1f} |")
        lines.append("")

    # Completion matrix
    lines.append("## Completion Matrix")
    lines.append("")
    lines.append("Shows which task/category/ablation combinations have results.")
    lines.append("")

    completion = df.groupby(["task", "category", "ablation"])["model_name"].nunique().reset_index()
    completion.columns = ["Task", "Category", "Ablation", "Models"]
    lines.append("| Task | Category | Ablation | Models |")
    lines.append("|------|----------|----------|-------:|")
    for _, row in completion.iterrows():
        lines.append(f"| {row['Task']} | {row['Category']} | {row['Ablation']} | {row['Models']} |")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Block3 benchmark results")
    parser.add_argument("--bench-dir", type=str, default=str(DEFAULT_BENCH))
    parser.add_argument("--md-output", type=str, default=str(DEFAULT_MD))
    parser.add_argument("--csv-output", type=str, default=None)
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.exists():
        print(f"ERROR: Benchmark directory not found: {bench_dir}")
        sys.exit(1)

    print(f"Scanning: {bench_dir}")
    df = collect_metrics(bench_dir)
    print(f"Found {len(df)} metric records from {df['model_name'].nunique() if len(df) else 0} models")

    if df.empty:
        print("No results yet.")
        return

    # Save CSV
    csv_path = Path(args.csv_output) if args.csv_output else bench_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Generate and save Markdown
    md_content = generate_markdown(df, bench_dir)
    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown saved: {md_path}")

    # Print quick summary
    print("\n=== Quick Summary ===")
    if "mae" in df.columns:
        best = df.loc[df["mae"].idxmin()]
        print(f"Best MAE: {best['model_name']} ({best['category']}) — MAE={best['mae']:.2f}")
    if "is_fallback" not in df.columns:
        df["is_fallback"] = df.apply(is_fallback, axis=1)
    real = df[~df["is_fallback"]]
    fb = df[df["is_fallback"]]
    print(f"Real results: {len(real)}, Fallback: {len(fb)}")
    if not real.empty:
        print(f"Best real MAE: {real.loc[real['mae'].idxmin(), 'model_name']} — MAE={real['mae'].min():.2f}")


if __name__ == "__main__":
    main()
