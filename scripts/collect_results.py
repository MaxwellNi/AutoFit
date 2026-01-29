from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
import yaml


def collect_results(src_runs: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "paper_tables"
    explain_dir = out_dir / "explain"
    config_dir = out_dir / "configs"
    table_dir.mkdir(exist_ok=True)
    explain_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    for run_path in src_runs.glob("**/"):
        if not run_path.is_dir():
            continue
        paper_tables = run_path / "paper_tables"
        if paper_tables.exists():
            for f in paper_tables.glob("*.parquet"):
                shutil.copy2(f, table_dir / f"{run_path.name}_{f.name}")
        explain = run_path / "explain"
        if explain.exists():
            for f in explain.glob("*"):
                if f.is_file():
                    shutil.copy2(f, explain_dir / f"{run_path.name}_{f.name}")
        best_cfg = run_path / "best_config.yaml"
        if best_cfg.exists():
            shutil.copy2(best_cfg, config_dir / f"{run_path.name}_best_config.yaml")
    return out_dir


def _safe_float(value: object) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _best_metric(results: list[dict]) -> dict[str, object]:
    best = None
    best_rmse = None
    for row in results:
        rmse = _safe_float(row.get("rmse"))
        if rmse is None:
            continue
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best = row
    if not best:
        return {"best_model_rmse": None, "best_model": None, "best_model_seed": None}
    return {
        "best_model_rmse": best_rmse,
        "best_model": best.get("model") or best.get("backbone"),
        "best_model_seed": best.get("seed"),
    }


def _collect_metrics_master(src_runs: Path, out_dir: Path) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for metrics_path in src_runs.rglob("metrics.json"):
        bench_dir = metrics_path.parent
        config_path = bench_dir / "configs" / "resolved_config.yaml"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                config = {}

        results = metrics.get("results", []) or []
        best_info = _best_metric(results)
        row = {
            "bench_dir": str(bench_dir),
            "exp_name": metrics.get("exp_name") or config.get("exp_name"),
            "timestamp": metrics.get("timestamp"),
            "metrics_path": str(metrics_path),
            "resolved_config_path": str(config_path) if config_path.exists() else None,
            "use_edgar": bool(metrics.get("use_edgar", config.get("use_edgar", False))),
            "n_rows": metrics.get("n_rows"),
            "n_entities": metrics.get("n_entities"),
            "n_features_edgar": metrics.get("n_features_edgar"),
            "edgar_valid_rate": metrics.get("edgar_valid_rate"),
            "results_count": metrics.get("results_count"),
            "total_runs": metrics.get("total_runs"),
            "best_model_rmse": best_info["best_model_rmse"],
            "best_model": best_info["best_model"],
            "best_model_seed": best_info["best_model_seed"],
            "label_horizon": config.get("label_horizon"),
            "strict_future": config.get("strict_future"),
            "label_goal_min": config.get("label_goal_min"),
            "seq_len": config.get("seq_len"),
            "min_label_delta_days": config.get("min_label_delta_days"),
            "min_ratio_delta_abs": config.get("min_ratio_delta_abs"),
            "min_ratio_delta_rel": config.get("min_ratio_delta_rel"),
            "offers_core": config.get("offers_core"),
            "edgar_features": config.get("edgar_features"),
            "selection_hash": config.get("selected_entities_hash") or config.get("sampled_entities_hash"),
            "selected_entities_count": config.get("selected_entities_count") or config.get("sampled_entities_count"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = out_dir / "metrics_master.parquet"
    csv_path = out_dir / "metrics_master.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run artifacts into one directory")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or Path("runs/collected") / datetime.now().strftime("%Y%m%d_%H%M%S")
    collect_results(args.runs_dir, out_dir)
    _collect_metrics_master(args.runs_dir, out_dir)
    print(f"Collected results into {out_dir}")


if __name__ == "__main__":
    main()
