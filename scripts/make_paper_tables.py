from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _load_metrics_files(root: Path) -> List[Dict[str, Any]]:
    metrics = []
    for path in root.rglob("metrics.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_metrics_path"] = str(path)
            metrics.append(data)
        except Exception:
            continue
    return metrics


def _is_valid_run(entry: Dict[str, Any], min_results: int) -> bool:
    results = entry.get("results", [])
    if not results:
        return False
    success = [r for r in results if r.get("status") == "success"]
    if not success:
        return False
    if len(results) < min_results:
        return False
    if entry.get("cutoff_violation", 0) != 0:
        return False
    return True


def _flatten_results(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        exp_name = m.get("exp_name", "unknown")
        results = m.get("results", [])
        use_edgar_default = bool(m.get("use_edgar", False))
        for r in results:
            if r.get("status") != "success":
                continue
            rows.append(
                {
                    "exp_name": exp_name,
                    "backbone": r.get("backbone", r.get("model", "unknown")),
                    "model": r.get("model", "unknown"),
                    "fusion_type": r.get("fusion_type", "none"),
                    "module_variant": r.get("module_variant", "unknown"),
                    "seed": r.get("seed"),
                    "use_edgar": bool(r.get("use_edgar", use_edgar_default)),
                    "rmse": r.get("rmse"),
                    "mae": r.get("mae"),
                    "mse": r.get("mse"),
                    "r2": r.get("r2"),
                    "train_time_sec": r.get("train_time_sec"),
                    "max_cuda_mem_mb": r.get("max_cuda_mem_mb"),
                }
            )
    return pd.DataFrame(rows)


def _main_results(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    raw = _flatten_results(metrics)
    if raw.empty:
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
    metric_cols = [c for c in ["rmse", "mae", "mse", "r2"] if c in raw.columns]
    group_cols = ["exp_name", "backbone", "fusion_type", "module_variant", "use_edgar"]
    agg = raw.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    agg["n_runs"] = raw.groupby(group_cols).size().values
    return agg
    return pd.DataFrame(rows)


def _ablation(main_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if main_df.empty:
        return pd.DataFrame(
            columns=[
                "exp_name",
                "backbone",
                "fusion_type",
                "module_variant",
                "metric_name",
                "edgar_on",
                "edgar_off",
                "delta",
            ]
        )
    key_cols = ["exp_name", "backbone", "fusion_type", "module_variant"]
    for metric in ["rmse_mean", "mae_mean", "r2_mean"]:
        if metric not in main_df.columns:
            continue
        pivot = main_df.pivot_table(index=key_cols, columns="use_edgar", values=metric, aggfunc="mean")
        for idx, row in pivot.iterrows():
            edgar_on = row.get(True)
            edgar_off = row.get(False)
            if pd.isna(edgar_on) or pd.isna(edgar_off):
                continue
            rows.append(
                {
                    "exp_name": idx[0],
                    "backbone": idx[1],
                    "fusion_type": idx[2],
                    "module_variant": idx[3],
                    "metric_name": metric.replace("_mean", ""),
                    "edgar_on": float(edgar_on),
                    "edgar_off": float(edgar_off),
                    "delta": float(edgar_on - edgar_off),
                }
            )
    return pd.DataFrame(rows)


def _faithfulness(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        exp_name = m.get("exp_name", "unknown")
        rows.append(
            {
                "exp_name": exp_name,
                "model": "NA",
                "dataset_split": "test",
                "metric_name": "faithfulness_placeholder",
                "metric_value": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _efficiency(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    raw = _flatten_results(metrics)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "exp_name",
                "backbone",
                "fusion_type",
                "module_variant",
                "use_edgar",
                "metric_name",
                "metric_value",
            ]
        )
    group_cols = ["exp_name", "backbone", "fusion_type", "module_variant", "use_edgar"]
    rows = []
    for metric in ["train_time_sec", "max_cuda_mem_mb"]:
        if metric not in raw.columns:
            continue
        grouped = raw.groupby(group_cols)[metric].mean().reset_index()
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "exp_name": row["exp_name"],
                    "backbone": row["backbone"],
                    "fusion_type": row["fusion_type"],
                    "module_variant": row["module_variant"],
                    "use_edgar": bool(row["use_edgar"]),
                    "metric_name": metric,
                    "metric_value": float(row[metric]) if pd.notna(row[metric]) else np.nan,
                }
            )
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build parquet paper tables from benchmark metrics.json")
    parser.add_argument("--bench_root", type=Path, default=Path("runs/benchmarks"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/paper_tables"))
    parser.add_argument("--min_results", type=int, default=10)
    parser.add_argument("--include_prefix", nargs="+", default=None)
    args = parser.parse_args()

    bench_root = args.bench_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = _load_metrics_files(bench_root)
    if args.include_prefix:
        prefixes = tuple(args.include_prefix)
        metrics_all = [m for m in metrics_all if str(m.get("exp_name", "")).startswith(prefixes)]
    metrics = [m for m in metrics_all if _is_valid_run(m, args.min_results)]

    main_df = _main_results(metrics)
    ablation_df = _ablation(main_df)
    faith_df = _faithfulness(metrics)
    eff_df = _efficiency(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = output_dir / "tables"
    tex_dir.mkdir(exist_ok=True)

    main_path = output_dir / "main_results.parquet"
    ablation_path = output_dir / "ablation.parquet"
    faith_path = output_dir / "faithfulness.parquet"
    eff_path = output_dir / "efficiency.parquet"

    main_df.to_parquet(main_path, index=False)
    ablation_df.to_parquet(ablation_path, index=False)
    faith_df.to_parquet(faith_path, index=False)
    eff_df.to_parquet(eff_path, index=False)

    main_df.to_csv(output_dir / "main_results.csv", index=False)
    ablation_df.to_csv(output_dir / "ablation.csv", index=False)
    faith_df.to_csv(output_dir / "faithfulness.csv", index=False)
    eff_df.to_csv(output_dir / "efficiency.csv", index=False)

    main_df.to_latex(tex_dir / "main_results.tex", index=False)
    ablation_df.to_latex(tex_dir / "ablation.tex", index=False)
    faith_df.to_latex(tex_dir / "faithfulness.tex", index=False)
    eff_df.to_latex(tex_dir / "efficiency.tex", index=False)


if __name__ == "__main__":
    main()
