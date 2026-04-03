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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.narrative.block3.autofit_status import CURRENT_AUTOFIT_BASELINE, is_retired_autofit_model

DEFAULT_BENCH = ROOT / "runs" / "benchmarks" / "block3_phase9_fair"
DEFAULT_MD = ROOT / "docs" / "BLOCK3_RESULTS.md"

# -----------------------------------------------------------------------
# Audit-validated exclusion list (Phase 9 deep re-audit, commit 3b9cab7)
# Models that MUST be excluded from leaderboard / paper tables due to
# verified experimental bugs.  Downstream tools MUST filter these out.
# -----------------------------------------------------------------------
AUDIT_EXCLUDED_MODELS: dict[str, str] = {
    # Finding A — silent context-mean fallback (foundation models)
    "Sundial": "silent context-mean fallback (Finding A)",
    "TimesFM2": "silent context-mean fallback (Finding A)",
    "LagLlama": "silent context-mean fallback (Finding A)",
    "Moirai": "silent context-mean fallback (Finding A)",
    "MoiraiLarge": "silent context-mean fallback (Finding A)",
    "Moirai2": "silent context-mean fallback (Finding A)",
    # Finding B — training crash → global mean fallback
    "AutoCES": "100% training crash fallback (Finding B)",
    "xLSTM": "100% training crash fallback (Finding B)",
    "TimeLLM": "100% training crash fallback (Finding B)",
    "StemGNN": "40/104 training crash fallback (Finding B)",
    "TimeXer": "40/104 training crash fallback (Finding B)",
    # Finding C — near-duplicates (share identity with Timer)
    "TimeMoE": "98% identical to Timer (Finding C)",
    "MOMENT": "99% identical to Timer (Finding C)",
    # Finding G — 100% constant predictions (TSLib)
    "MICN": "100% constant predictions (Finding G)",
    "MultiPatchFormer": "100% constant predictions (Finding G)",
    "TimeFilter": "100% constant predictions (Finding G)",
    # Finding H — 100% constant predictions (Phase 15 TSLib, verified 2026-03-22)
    "CFPT": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "DeformableTST": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "PathFormer": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "SEMPO": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "SparseTSF": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "TimeBridge": "100% constant predictions, 0/67 fairness pass (Finding H)",
    "TimePerceiver": "100% constant predictions, 0/67 fairness pass (Finding H)",
    # Structural failure
    "NegativeBinomialGLM": "convergence failure, 21 records only (Structural)",
}


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


def apply_comparability_filter(
    df: pd.DataFrame,
    min_coverage: float = 0.98,
    fairness_only: bool = True,
) -> pd.DataFrame:
    """Filter records to keep only fair and comparable evaluations.

    Three-layer filter:
      1. Audit exclusion — hardcoded list of models with verified bugs
      2. Fairness pass — constant-prediction guard from harness
      3. Coverage — minimum prediction coverage ratio
    """
    if df.empty:
        return df

    out = df.copy()

    # Layer 0: remove retired / invalid AutoFit-family lines from the active
    # public surface. They remain in raw artifacts for auditability only.
    if "model_name" in out.columns:
        retired_mask = out["model_name"].map(is_retired_autofit_model)
        n_retired = retired_mask.sum()
        if n_retired > 0:
            retired_models = sorted(out.loc[retired_mask, "model_name"].unique())
            print(
                f"  Archived AutoFit purge: dropped {n_retired} records from "
                f"{len(retired_models)} models: {retired_models}"
            )
        out = out[~retired_mask]

    # Layer 1: Audit-validated exclusions
    if "model_name" in out.columns:
        excluded_mask = out["model_name"].isin(AUDIT_EXCLUDED_MODELS)
        n_excluded = excluded_mask.sum()
        if n_excluded > 0:
            excluded_models = out.loc[excluded_mask, "model_name"].unique()
            print(f"  Audit exclusion: dropped {n_excluded} records from "
                  f"{len(excluded_models)} models: {list(excluded_models)}")
        out = out[~excluded_mask]

    # Layer 2: Fairness pass (constant-prediction guard)
    if fairness_only and "fairness_pass" in out.columns:
        out = out[out["fairness_pass"] == True]  # noqa: E712

    # Layer 3: Coverage
    if "prediction_coverage_ratio" in out.columns:
        cov = pd.to_numeric(out["prediction_coverage_ratio"], errors="coerce")
        out = out[cov >= float(min_coverage)]

    return out


# Target-specific fallback MAE thresholds (from data profiling).
# When a model silently returns the global mean, its MAE clusters
# near these values.  A 2 % tolerance window catches rounding drift.
_FALLBACK_MAE_RANGES: dict = {
    "funding_raised_usd": (585000, 620000),
    "investors_count":    (390, 420),
    "is_funded":          (0.30, 0.55),
    "funding_goal_usd":   (480000, 530000),
}


def is_fallback(row) -> bool:
    """Detect fallback results (mean-predictor) using target-specific
    MAE ranges derived from data profiling."""
    mae = row.get("mae", 0)
    target = row.get("target", "")
    lo, hi = _FALLBACK_MAE_RANGES.get(target, (595000, 610000))
    if lo < mae < hi:
        return True
    return False


def generate_markdown(
    df: pd.DataFrame,
    bench_dir: Path,
    raw_total: int,
    min_coverage: float,
    fairness_only: bool,
) -> str:
    """Generate comprehensive Markdown report."""
    lines = []
    lines.append("# Block 3 Benchmark Results")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Benchmark Dir**: `{bench_dir.name}`")
    lines.append(f"**Total Records (post-filter)**: {len(df)}")
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
    lines.append(
        f"> Current public AutoFit surface is clean: only `{CURRENT_AUTOFIT_BASELINE}` remains in the active environment and leaderboard outputs."
    )
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Raw records | {raw_total} |")
    lines.append(f"| Filtered records | {len(df)} |")
    lines.append(f"| Comparability filter | fairness_only={fairness_only}, min_coverage={min_coverage:.2f} |")
    lines.append(f"| Models evaluated | {n_models} |")
    lines.append(f"| Active AutoFit baseline | {CURRENT_AUTOFIT_BASELINE} |")
    lines.append(f"| Archived AutoFit lines in current surface | 0 |")
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
    parser.add_argument("--min-coverage", type=float, default=0.98)
    parser.add_argument(
        "--disable-fairness-filter",
        action="store_true",
        help="Disable fairness_pass filter (not recommended for final leaderboard).",
    )
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.exists():
        print(f"ERROR: Benchmark directory not found: {bench_dir}")
        sys.exit(1)

    print(f"Scanning: {bench_dir}")
    df_raw = collect_metrics(bench_dir)
    fairness_only = not args.disable_fairness_filter
    df = apply_comparability_filter(
        df_raw,
        min_coverage=args.min_coverage,
        fairness_only=fairness_only,
    )
    print(
        f"Found {len(df_raw)} raw metric records, "
        f"{len(df)} post-filter records "
        f"(fairness_only={fairness_only}, min_coverage={args.min_coverage:.2f})"
    )

    if df.empty:
        print("No results yet.")
        return

    # Save CSV
    csv_path = Path(args.csv_output) if args.csv_output else bench_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Generate and save Markdown
    md_content = generate_markdown(
        df,
        bench_dir,
        raw_total=len(df_raw),
        min_coverage=args.min_coverage,
        fairness_only=fairness_only,
    )
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
