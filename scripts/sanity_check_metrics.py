#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GOAL_RE = re.compile(r"goal(?P<val>\d+(?:\.\d+)?)", re.IGNORECASE)
HORIZON_RE = re.compile(r"(?:horizon|h)(?P<val>\d+)\b", re.IGNORECASE)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    try:
        value = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _parse_goal(text: str | None) -> float | None:
    if not text:
        return None
    match = GOAL_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group("val"))
    except (TypeError, ValueError):
        return None


def _parse_horizon(text: str | None) -> int | None:
    if not text:
        return None
    match = HORIZON_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("val"))
    except (TypeError, ValueError):
        return None


def _load_config(bench_dir: Path) -> Dict[str, Any] | None:
    cfg_path = bench_dir / "configs" / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _infer_goal_min(bench_dir: Path, metrics_df: pd.DataFrame, preds_df: pd.DataFrame) -> float | None:
    cfg = _load_config(bench_dir)
    if cfg and cfg.get("label_goal_min") is not None:
        return float(cfg.get("label_goal_min"))
    for df in (metrics_df, preds_df):
        if "label_goal_min" in df.columns:
            val = pd.to_numeric(df["label_goal_min"], errors="coerce")
            if val.notna().any():
                return float(val.dropna().iloc[0])
    exp_name = (cfg or {}).get("exp_name") or bench_dir.name
    return _parse_goal(str(exp_name))


def _infer_label_horizon(bench_dir: Path, metrics_df: pd.DataFrame, preds_df: pd.DataFrame) -> int | None:
    cfg = _load_config(bench_dir)
    if cfg and cfg.get("label_horizon") is not None:
        try:
            return int(cfg.get("label_horizon"))
        except (TypeError, ValueError):
            pass
    for df in (metrics_df, preds_df):
        if "label_horizon" in df.columns:
            val = pd.to_numeric(df["label_horizon"], errors="coerce")
            if val.notna().any():
                return int(val.dropna().iloc[0])
    exp_name = (cfg or {}).get("exp_name") or bench_dir.name
    return _parse_horizon(str(exp_name))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_true.size == 0:
        return {"rmse": None, "mae": None, "r2": None}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check metrics and baselines for benchmark runs.")
    parser.add_argument("--bench_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("runs") / f"sanity_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks: List[Dict[str, Any]] = []

    for raw_dir in args.bench_dirs:
        bench_dir = Path(raw_dir)
        metrics_path = bench_dir / "metrics.parquet"
        preds_path = bench_dir / "predictions.parquet"
        if not preds_path.exists():
            benchmarks.append(
                {
                    "bench_dir": str(bench_dir),
                    "ok": False,
                    "errors": ["missing_predictions.parquet"],
                    "warnings": [],
                }
            )
            continue

        metrics_df = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
        preds_df = pd.read_parquet(preds_path)

        train_df = None
        test_df = preds_df
        if "split" in preds_df.columns:
            test_df = preds_df[preds_df["split"] == "test"]
            train_df = preds_df[preds_df["split"] == "train"]

        entry: Dict[str, Any] = {
            "bench_dir": str(bench_dir),
            "ok": True,
            "errors": [],
            "warnings": [],
        }
        cfg = _load_config(bench_dir)
        if cfg:
            entry["exp_name"] = cfg.get("exp_name")
            entry["use_edgar"] = cfg.get("use_edgar")
            entry["limit_rows"] = cfg.get("limit_rows")
            entry["sample_strategy"] = cfg.get("sample_strategy")
            entry["sample_seed"] = cfg.get("sample_seed")
            entry["split_seed"] = cfg.get("split_seed")
            entry["seeds"] = cfg.get("seeds")

        entry["label_goal_min"] = _infer_goal_min(bench_dir, metrics_df, preds_df)
        entry["label_horizon"] = _infer_label_horizon(bench_dir, metrics_df, preds_df)

        if "y_true_raw" in test_df.columns and "y_pred_raw" in test_df.columns:
            y_true_full = test_df["y_true_raw"].to_numpy(dtype=float)
            y_pred_full = test_df["y_pred_raw"].to_numpy(dtype=float)
            entry["label_scale"] = "raw"
        else:
            y_true_full = test_df["y_true"].to_numpy(dtype=float)
            y_pred_full = test_df["y_pred"].to_numpy(dtype=float)
            entry["label_scale"] = "standardized"

        mask = np.isfinite(y_true_full) & np.isfinite(y_pred_full)
        y_true_eval = y_true_full[mask]
        y_pred_eval = y_pred_full[mask]

        entry["y_stats"] = {
            "count": int(y_true_eval.size),
            "mean": float(np.nanmean(y_true_eval)) if y_true_eval.size else None,
            "std": float(np.nanstd(y_true_eval)) if y_true_eval.size else None,
            "min": float(np.nanmin(y_true_eval)) if y_true_eval.size else None,
            "max": float(np.nanmax(y_true_eval)) if y_true_eval.size else None,
        }

        best_rmse = None
        best_model = None
        if not metrics_df.empty and "rmse" in metrics_df.columns:
            best_row = metrics_df.sort_values("rmse", ascending=True).iloc[0]
            best_rmse = float(best_row["rmse"])
            best_model = best_row.get("model") or best_row.get("backbone")
        entry["best_model"] = {"name": best_model, "rmse": best_rmse}

        if train_df is not None and len(train_df) > 0:
            train_y = (
                train_df["y_true_raw"].to_numpy(dtype=float)
                if entry["label_scale"] == "raw" and "y_true_raw" in train_df.columns
                else train_df["y_true"].to_numpy(dtype=float)
            )
            y_mean = float(np.nanmean(train_y)) if train_y.size else float("nan")
        else:
            y_mean = float(np.nanmean(y_true_eval)) if y_true_eval.size else float("nan")
        naive_pred = np.full_like(y_true_eval, y_mean)
        entry["naive_mean_baseline"] = _metrics(y_true_eval, naive_pred)

        total_rows = len(test_df)
        progress_invalid_reasons: Dict[str, int] = {}
        ratio_source = None
        if "funding_ratio_input_last" in test_df.columns:
            ratio_source = "funding_ratio_input_last"
            ratio = pd.to_numeric(test_df["funding_ratio_input_last"], errors="coerce").to_numpy(dtype=float)
            invalid_mask = ~np.isfinite(ratio)
            progress_invalid_reasons["ratio_not_finite"] = int(invalid_mask.sum())
        elif {"funding_raised_usd_input_last", "funding_goal_usd_input_last"}.issubset(test_df.columns):
            ratio_source = "funding_raised_usd_input_last/funding_goal_usd_input_last"
            raised = pd.to_numeric(test_df["funding_raised_usd_input_last"], errors="coerce").to_numpy(dtype=float)
            goal = pd.to_numeric(test_df["funding_goal_usd_input_last"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = raised / goal
            invalid_mask = ~np.isfinite(ratio)
            progress_invalid_reasons["ratio_not_finite"] = int(invalid_mask.sum())
        elif {"funding_raised_usd_last", "funding_goal_usd_last"}.issubset(test_df.columns):
            ratio_source = "funding_raised_usd_last/funding_goal_usd_last"
            raised = pd.to_numeric(test_df["funding_raised_usd_last"], errors="coerce").to_numpy(dtype=float)
            goal = pd.to_numeric(test_df["funding_goal_usd_last"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = raised / goal
            invalid_mask = ~np.isfinite(ratio)
            progress_invalid_reasons["ratio_not_finite"] = int(invalid_mask.sum())
        else:
            ratio = np.full(total_rows, np.nan, dtype=float)
            invalid_mask = np.ones(total_rows, dtype=bool)
            progress_invalid_reasons["missing_columns"] = total_rows

        fallback_value = y_mean if np.isfinite(y_mean) else 0.0
        progress_pred = ratio.copy()
        progress_pred[invalid_mask] = fallback_value
        defined_ratio = 1.0 - (float(invalid_mask.sum()) / max(1, total_rows))

        ratio_mask = np.isfinite(progress_pred) & np.isfinite(y_true_full)
        entry["naive_progress_baseline"] = _metrics(y_true_full[ratio_mask], progress_pred[ratio_mask])
        entry["naive_progress_baseline"]["defined_ratio"] = defined_ratio
        entry["naive_progress_baseline"]["invalid_reasons"] = progress_invalid_reasons
        entry["naive_progress_baseline"]["predict_ratio"] = ratio_source or "missing"
        entry["naive_progress_baseline"]["n_eval"] = int(ratio_mask.sum())

        current_ratio = ratio
        mask_lr = np.isfinite(current_ratio) & np.isfinite(y_true_full)
        label_vs_current = {
            "corr": _safe_corr(y_true_full[mask_lr], current_ratio[mask_lr]) if mask_lr.any() else None,
            "max_abs_diff": float(np.max(np.abs(y_true_full[mask_lr] - current_ratio[mask_lr]))) if mask_lr.any() else None,
            "n_finite": int(mask_lr.sum()),
        }
        entry["label_vs_current_ratio"] = label_vs_current

        if best_rmse is not None:
            baseline_min = min(
                entry["naive_mean_baseline"]["rmse"] or float("inf"),
                entry["naive_progress_baseline"]["rmse"] or float("inf"),
            )
            if np.isfinite(baseline_min) and best_rmse >= baseline_min:
                entry["warnings"].append(
                    f"best_rmse={best_rmse:.6f} not clearly better than baseline_min={baseline_min:.6f}"
                )
        if (
            entry["naive_progress_baseline"]["rmse"] is not None
            and entry["naive_progress_baseline"]["rmse"] < 1e-4
            and defined_ratio > 0.95
        ):
            entry["warnings"].append("STRONG WARNING: label may equal current ratio (check horizon/alignment)")

        benchmarks.append(entry)

    report = {
        "timestamp": timestamp,
        "benchmarks": benchmarks,
    }
    out_path = output_dir / "sanity_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
