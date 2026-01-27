#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
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


def _pick_ratio_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    if "funding_ratio_input_last" in df.columns:
        return "funding_ratio_input_last", None, "funding_ratio_input_last"
    if {"funding_raised_usd_input_last", "funding_goal_usd_input_last"}.issubset(df.columns):
        return "funding_raised_usd_input_last", "funding_goal_usd_input_last", "funding_raised_usd_input_last/funding_goal_usd_input_last"
    if {"funding_raised_usd_last", "funding_goal_usd_last"}.issubset(df.columns):
        return "funding_raised_usd_last", "funding_goal_usd_last", "funding_raised_usd_last/funding_goal_usd_last"
    return None, None, "missing"


def _dedupe_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "sample_id" in df.columns:
        return df.drop_duplicates(subset=["sample_id"]), "sample_id"
    if "model" in df.columns and not df["model"].empty:
        first_model = sorted(df["model"].dropna().unique().tolist())[0]
        return df[df["model"] == first_model], f"first_model:{first_model}"
    return df, "none"


def _load_config(bench_dir: Path) -> Dict[str, Any] | None:
    cfg_path = bench_dir / "configs" / "resolved_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _build_alignment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "ratio_source": None,
        "corr": None,
        "max_abs_diff": None,
        "median_abs_diff": None,
        "pct_abs_diff_lt_1e-4": None,
        "pct_abs_diff_lt_1e-6": None,
        "n_finite": 0,
        "missing_reason": None,
    }
    if "y_true_raw" in df.columns:
        y_true = pd.to_numeric(df["y_true_raw"], errors="coerce").to_numpy(dtype=float)
    elif "y_true" in df.columns:
        y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    else:
        output["missing_reason"] = "missing_y_true"
        return output

    col_a, col_b, source = _pick_ratio_columns(df)
    output["ratio_source"] = source
    if col_a is None:
        output["missing_reason"] = "missing_ratio_columns"
        return output

    if col_b is None:
        ratio = pd.to_numeric(df[col_a], errors="coerce").to_numpy(dtype=float)
    else:
        raised = pd.to_numeric(df[col_a], errors="coerce").to_numpy(dtype=float)
        goal = pd.to_numeric(df[col_b], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = raised / goal

    mask = np.isfinite(y_true) & np.isfinite(ratio)
    if not mask.any():
        output["missing_reason"] = "no_finite_pairs"
        return output

    diff = y_true[mask] - ratio[mask]
    abs_diff = np.abs(diff)
    output["corr"] = _safe_corr(y_true[mask], ratio[mask])
    output["max_abs_diff"] = float(np.max(abs_diff))
    output["median_abs_diff"] = float(np.median(abs_diff))
    output["pct_abs_diff_lt_1e-4"] = float(np.mean(abs_diff < 1e-4))
    output["pct_abs_diff_lt_1e-6"] = float(np.mean(abs_diff < 1e-6))
    output["n_finite"] = int(mask.sum())
    return output


def _build_time_alignment(df: pd.DataFrame, min_label_delta_days: float) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "missing_reason": None,
        "pct_label_ts_le_input_end_ts": None,
        "pct_label_idx_le_input_end_idx": None,
        "median_delta_steps": None,
        "median_delta_days": None,
        "pct_delta_days_lt_min": None,
    }
    if {"input_end_ts", "label_ts"}.issubset(df.columns):
        input_ts = pd.to_datetime(df["input_end_ts"], errors="coerce", utc=True)
        label_ts = pd.to_datetime(df["label_ts"], errors="coerce", utc=True)
        mask = input_ts.notna() & label_ts.notna()
        if mask.any():
            output["pct_label_ts_le_input_end_ts"] = float((label_ts[mask] <= input_ts[mask]).mean())
            deltas = (label_ts[mask] - input_ts[mask]).dt.total_seconds() / 86400.0
            output["median_delta_days"] = float(np.median(deltas)) if len(deltas) else None
            if min_label_delta_days > 0:
                output["pct_delta_days_lt_min"] = float((deltas < min_label_delta_days).mean())
    elif {"input_end_idx", "label_idx"}.issubset(df.columns):
        input_idx = pd.to_numeric(df["input_end_idx"], errors="coerce")
        label_idx = pd.to_numeric(df["label_idx"], errors="coerce")
        mask = input_idx.notna() & label_idx.notna()
        if mask.any():
            output["pct_label_idx_le_input_end_idx"] = float((label_idx[mask] <= input_idx[mask]).mean())
            deltas = (label_idx[mask] - input_idx[mask]).to_numpy(dtype=float)
            output["median_delta_steps"] = float(np.median(deltas)) if len(deltas) else None
    else:
        output["missing_reason"] = "missing_input_end_or_label_cols"
    return output


def _conclusion(alignment: Dict[str, Any], time_alignment: Dict[str, Any]) -> str:
    if alignment["pct_abs_diff_lt_1e-6"] is not None and alignment["pct_abs_diff_lt_1e-6"] > 0.95:
        base = "y≈ratio_from_last (pct_abs_diff_lt_1e-6>0.95)"
    elif alignment["corr"] is not None and alignment["corr"] > 0.9999:
        base = "y≈ratio_from_last (corr>0.9999)"
    else:
        base = "not_simple_y_eq_last_ratio"
    if time_alignment.get("median_delta_days") is not None:
        base += f" | median_delta_days={time_alignment['median_delta_days']}"
    if time_alignment.get("pct_delta_days_lt_min") is not None:
        base += f" pct_delta_days_lt_min={time_alignment['pct_delta_days_lt_min']}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit alignment between label and input-end ratio in predictions.")
    parser.add_argument("--bench_dirs", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--min_label_delta_days", type=float, default=0.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_path is None:
        out_dir = Path("runs") / f"sanity_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "alignment_audit.json"
    else:
        output_path = args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for raw_dir in args.bench_dirs:
        bench_dir = Path(raw_dir)
        preds_path = bench_dir / "predictions.parquet"
        item: Dict[str, Any] = {"bench_dir": str(bench_dir), "ok": True, "errors": [], "warnings": []}
        cfg = _load_config(bench_dir)
        if cfg:
            item["exp_name"] = cfg.get("exp_name")
            item["use_edgar"] = cfg.get("use_edgar")
            item["limit_rows"] = cfg.get("limit_rows")
            item["label_goal_min"] = cfg.get("label_goal_min")
            item["label_horizon"] = cfg.get("label_horizon")
            item["sample_strategy"] = cfg.get("sample_strategy")
            item["sample_seed"] = cfg.get("sample_seed")
            item["split_seed"] = cfg.get("split_seed")
            item["seeds"] = cfg.get("seeds")
        if not preds_path.exists():
            item["ok"] = False
            item["errors"].append("missing_predictions.parquet")
            results.append(item)
            continue
        preds_df = pd.read_parquet(preds_path)
        if "split" in preds_df.columns:
            preds_df = preds_df[preds_df["split"] == "test"]
        preds_df, dedupe_strategy = _dedupe_predictions(preds_df)
        item["dedupe_strategy"] = dedupe_strategy
        item["n_rows"] = int(len(preds_df))
        item["alignment"] = _build_alignment_metrics(preds_df)
        item["time_alignment"] = _build_time_alignment(preds_df, args.min_label_delta_days)
        item["conclusion"] = _conclusion(item["alignment"], item["time_alignment"])
        results.append(item)

    report = {
        "timestamp": timestamp,
        "run_count": len(results),
        "runs": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
