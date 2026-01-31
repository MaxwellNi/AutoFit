#!/usr/bin/env python
"""
Build paper tables (main_results.csv, efficiency.csv) ONLY from benchmark_truthfulness.json.
Source of truth: real_runs from audit_benchmark_matrix_truthfulness. No phantom rows.
Aggregation: mean/std across seeds; std blank when single seed (not NaN).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _main_results_from_real_runs(real_runs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not real_runs:
        return pd.DataFrame(
            columns=[
                "exp_name",
                "backbone",
                "fusion_type",
                "module_variant",
                "use_edgar",
                "rmse_mean",
                "mae_mean",
                "mse_mean",
                "r2_mean",
                "rmse_std",
                "mae_std",
                "mse_std",
                "r2_std",
                "n_runs",
            ]
        )
    raw = pd.DataFrame(real_runs)
    metric_cols = [c for c in ["rmse", "mae", "mse", "r2"] if c in raw.columns]
    group_cols = ["exp_name", "backbone", "fusion_type", "module_variant", "use_edgar"]
    for c in group_cols:
        if c not in raw.columns:
            return pd.DataFrame(columns=group_cols + ["rmse_mean", "mae_mean", "mse_mean", "r2_mean", "rmse_std", "mae_std", "mse_std", "r2_std", "n_runs"])
    agg = raw.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            if c[1] == "" or c[0] in group_cols:
                new_cols.append(c[0])
            else:
                new_cols.append(f"{c[0]}_{c[1]}")
        else:
            new_cols.append(c)
    agg.columns = new_cols
    n_runs = raw.groupby(group_cols).size().reset_index(name="n_runs")
    agg = agg.merge(n_runs, on=group_cols)
    for col in ["rmse_std", "mae_std", "mse_std", "r2_std"]:
        if col in agg.columns:
            agg.loc[agg["n_runs"] == 1, col] = np.nan
    return agg


def _efficiency_from_real_runs(real_runs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not real_runs:
        return pd.DataFrame(
            columns=[
                "exp_name",
                "backbone",
                "fusion_type",
                "module_variant",
                "use_edgar",
                "metric_name",
                "metric_value",
                "note",
            ]
        )
    raw = pd.DataFrame(real_runs)
    group_cols = ["exp_name", "backbone", "fusion_type", "module_variant", "use_edgar"]
    rows: List[Dict[str, Any]] = []
    for metric in ["train_time_sec", "max_cuda_mem_mb"]:
        if metric not in raw.columns:
            continue
        grouped = raw.groupby(group_cols)[metric].mean().reset_index()
        for _, row in grouped.iterrows():
            val = row[metric]
            note = ""
            if pd.isna(val) and metric == "train_time_sec":
                note = "timing_not_logged"
            rows.append({
                "exp_name": row["exp_name"],
                "backbone": row["backbone"],
                "fusion_type": row["fusion_type"],
                "module_variant": row["module_variant"],
                "use_edgar": bool(row["use_edgar"]),
                "metric_name": metric,
                "metric_value": float(val) if pd.notna(val) else np.nan,
                "note": note,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper tables from benchmark_truthfulness.json only.")
    parser.add_argument("--benchmark_truthfulness_json", type=Path, default=None)
    parser.add_argument("--truthfulness_json", type=Path, default=None, help="Alias for --benchmark_truthfulness_json")
    parser.add_argument("--output_dir", type=Path, required=True, help="e.g. runs/orchestrator/STAMP/paper_tables")
    args = parser.parse_args()

    truth_path = args.truthfulness_json or args.benchmark_truthfulness_json
    if not truth_path:
        raise ValueError("Must provide --benchmark_truthfulness_json or --truthfulness_json")
    if not truth_path.exists():
        raise FileNotFoundError(f"truthfulness json not found: {truth_path}")

    data = json.loads(truth_path.read_text(encoding="utf-8"))
    real_runs = data.get("real_runs", [])
    if not real_runs:
        raise ValueError("benchmark_truthfulness.json has no real_runs")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_df = _main_results_from_real_runs(real_runs)
    eff_df = _efficiency_from_real_runs(real_runs)

    main_df.to_csv(args.output_dir / "main_results.csv", index=False)
    main_df.to_parquet(args.output_dir / "main_results.parquet", index=False)
    eff_df.to_csv(args.output_dir / "efficiency.csv", index=False)
    eff_df.to_parquet(args.output_dir / "efficiency.parquet", index=False)
    print(f"Wrote main_results and efficiency to {args.output_dir}")


if __name__ == "__main__":
    main()
