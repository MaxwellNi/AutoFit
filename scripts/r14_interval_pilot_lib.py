#!/usr/bin/env python3
"""Shared helpers for Round-14 interval diagnostics from landed predictions.

These helpers run diagnostic split-conformal style pilots on existing
prediction artifacts. They are not a replacement for temporal validation: the
current predictions lack stable row keys, so outputs must be treated as route
screening evidence only.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def latest_prediction_files(max_files: int = 8) -> list[Path]:
    files = sorted(ROOT.glob("runs/benchmarks/r14fcast_main_h*_co_*/predictions.parquet"))
    return files[-max_files:]


def parse_horizon(path: Path) -> int | None:
    match = re.search(r"_h(\d+)_", path.name) or re.search(r"_h(\d+)_", path.parent.name)
    return int(match.group(1)) if match else None


def load_predictions(max_files: int = 8, max_rows_per_group: int = 50000) -> pd.DataFrame:
    rows = []
    columns = ["y_true", "y_pred", "model", "task", "ablation", "horizon", "target"]
    for path in latest_prediction_files(max_files=max_files):
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            continue
        frame = frame[frame["target"].eq("funding_raised_usd")].copy()
        if frame.empty:
            continue
        if "horizon" not in frame or frame["horizon"].isna().all():
            frame["horizon"] = parse_horizon(path)
        frame["source_path"] = str(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=columns + ["source_path", "residual", "abs_residual"])
    data = pd.concat(rows, ignore_index=True)
    data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
    data = data[np.isfinite(data["y_true"]) & np.isfinite(data["y_pred"])].copy()
    data["residual"] = data["y_true"] - data["y_pred"]
    data["abs_residual"] = data["residual"].abs()

    capped = []
    for _, group in data.groupby(["model", "horizon", "target"], sort=False):
        if len(group) > max_rows_per_group:
            capped.append(group.iloc[:max_rows_per_group].copy())
        else:
            capped.append(group.copy())
    return pd.concat(capped, ignore_index=True) if capped else data.iloc[:0].copy()


def split_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = group.reset_index(drop=True).copy()
    mask = (np.arange(len(ordered)) % 2) == 0
    return ordered.loc[mask].copy(), ordered.loc[~mask].copy()


def empirical_coverage(y_true: pd.Series, lower: np.ndarray, upper: np.ndarray) -> float:
    vals = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    return float(np.mean((vals >= lower) & (vals <= upper))) if len(vals) else float("nan")


def width_mean(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower)) if len(lower) else float("nan")


def marginal_interval(cal: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.10) -> dict[str, Any]:
    q = float(np.quantile(cal["abs_residual"], 1.0 - alpha))
    lower = test["y_pred"].to_numpy(dtype=float) - q
    upper = test["y_pred"].to_numpy(dtype=float) + q
    return {"q": q, "coverage": empirical_coverage(test["y_true"], lower, upper), "width_mean": width_mean(lower, upper)}


def subgroup_yhat_interval(cal: pd.DataFrame, test: pd.DataFrame, bins: int = 5, alpha: float = 0.10) -> dict[str, Any]:
    quantiles = np.unique(np.quantile(cal["y_pred"], np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) <= 2:
        return marginal_interval(cal, test, alpha=alpha) | {"n_bins": 1, "fallback": "single_bin"}
    cal_bin = np.digitize(cal["y_pred"], quantiles[1:-1], right=False)
    test_bin = np.digitize(test["y_pred"], quantiles[1:-1], right=False)
    global_q = float(np.quantile(cal["abs_residual"], 1.0 - alpha))
    q_by_bin = {}
    for bin_id in range(len(quantiles) - 1):
        vals = cal.loc[cal_bin == bin_id, "abs_residual"]
        q_by_bin[bin_id] = float(np.quantile(vals, 1.0 - alpha)) if len(vals) >= 50 else global_q
    q = np.array([q_by_bin.get(int(bin_id), global_q) for bin_id in test_bin], dtype=float)
    pred = test["y_pred"].to_numpy(dtype=float)
    lower = pred - q
    upper = pred + q
    return {
        "n_bins": int(len(quantiles) - 1),
        "coverage": empirical_coverage(test["y_true"], lower, upper),
        "width_mean": width_mean(lower, upper),
        "q_by_bin": {str(k): v for k, v in q_by_bin.items()},
    }


def asymmetric_interval(cal: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.10) -> dict[str, Any]:
    lower_residual = (cal["y_pred"] - cal["y_true"]).clip(lower=0.0)
    upper_residual = (cal["y_true"] - cal["y_pred"]).clip(lower=0.0)
    q_lower = float(np.quantile(lower_residual, 1.0 - alpha / 2.0))
    q_upper = float(np.quantile(upper_residual, 1.0 - alpha / 2.0))
    pred = test["y_pred"].to_numpy(dtype=float)
    lower = pred - q_lower
    upper = pred + q_upper
    return {
        "q_lower": q_lower,
        "q_upper": q_upper,
        "coverage": empirical_coverage(test["y_true"], lower, upper),
        "width_mean": width_mean(lower, upper),
    }


def drift_guard_interval(
    cal: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float = 0.10,
    holdout_fraction: float = 0.20,
) -> dict[str, Any]:
    residuals = cal["abs_residual"].to_numpy(dtype=float)
    split = int(len(residuals) * (1.0 - holdout_fraction))
    split = min(max(split, 1), len(residuals) - 1)
    early = residuals[:split]
    late = residuals[split:]
    nominal_level = 1.0 - alpha
    early_q = float(np.quantile(early, nominal_level, method="higher"))
    holdout_coverage = float(np.mean(late <= early_q)) if len(late) else float("nan")
    adjusted_level = min(0.99, nominal_level + max(0.0, nominal_level - holdout_coverage))
    guarded_q = float(np.quantile(residuals, adjusted_level, method="higher"))
    pred = test["y_pred"].to_numpy(dtype=float)
    lower = pred - guarded_q
    upper = pred + guarded_q
    return {
        "nominal_level": nominal_level,
        "holdout_fraction": holdout_fraction,
        "holdout_coverage_at_nominal": holdout_coverage,
        "adjusted_quantile_level": adjusted_level,
        "q": guarded_q,
        "coverage": empirical_coverage(test["y_true"], lower, upper),
        "width_mean": width_mean(lower, upper),
    }


def gpd_interval(cal: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.10, threshold_q: float = 0.80) -> dict[str, Any]:
    abs_resid = cal["abs_residual"].to_numpy(dtype=float)
    empirical_q = float(np.quantile(abs_resid, 1.0 - alpha))
    threshold = float(np.quantile(abs_resid, threshold_q))
    exceedances = abs_resid[abs_resid > threshold] - threshold
    q = empirical_q
    fit_status = "empirical_fallback"
    if len(exceedances) >= 100:
        try:
            from scipy.stats import genpareto

            shape, loc, scale = genpareto.fit(exceedances, floc=0.0)
            tail_fraction = max(1.0 - threshold_q, 1e-9)
            tail_prob = min(max(((1.0 - alpha) - threshold_q) / tail_fraction, 0.0), 0.999)
            q = threshold + float(genpareto.ppf(tail_prob, shape, loc=loc, scale=scale))
            if not math.isfinite(q) or q <= 0.0:
                q = empirical_q
            else:
                fit_status = "gpd_fit"
        except Exception as exc:  # pragma: no cover - environment dependent
            fit_status = f"empirical_fallback:{type(exc).__name__}"
    pred = test["y_pred"].to_numpy(dtype=float)
    lower = pred - q
    upper = pred + q
    return {
        "threshold_q": threshold_q,
        "threshold": threshold,
        "q": float(q),
        "empirical_q": empirical_q,
        "fit_status": fit_status,
        "n_exceedances": int(len(exceedances)),
        "coverage": empirical_coverage(test["y_true"], lower, upper),
        "width_mean": width_mean(lower, upper),
    }


def evaluate(method: str, max_files: int = 8, max_rows_per_group: int = 50000) -> dict[str, Any]:
    data = load_predictions(max_files=max_files, max_rows_per_group=max_rows_per_group)
    rows = []
    for (model, horizon, target), group in data.groupby(["model", "horizon", "target"], sort=True):
        if len(group) < 1000:
            continue
        cal, test = split_group(group)
        marginal = marginal_interval(cal, test)
        if method == "marginal":
            candidate = marginal
        elif method == "subgroup":
            candidate = subgroup_yhat_interval(cal, test)
        elif method == "cqr_lite":
            candidate = asymmetric_interval(cal, test)
        elif method == "drift_guard":
            candidate = drift_guard_interval(cal, test)
        elif method == "gpd_evt":
            candidate = gpd_interval(cal, test)
        else:
            raise ValueError(f"Unknown method: {method}")
        rows.append({
            "model": str(model),
            "horizon": int(horizon) if pd.notna(horizon) else None,
            "target": str(target),
            "n_rows": int(len(group)),
            "n_cal": int(len(cal)),
            "n_test": int(len(test)),
            "marginal_coverage": marginal.get("coverage"),
            "marginal_width_mean": marginal.get("width_mean"),
            "candidate_coverage": candidate.get("coverage"),
            "candidate_width_mean": candidate.get("width_mean"),
            "coverage_delta": None if candidate.get("coverage") is None else float(candidate.get("coverage") - marginal.get("coverage")),
            "width_delta": None if candidate.get("width_mean") is None else float(candidate.get("width_mean") - marginal.get("width_mean")),
            "candidate_details": candidate,
        })
    coverages = [row["candidate_coverage"] for row in rows if row.get("candidate_coverage") is not None]
    deltas = [row["coverage_delta"] for row in rows if row.get("coverage_delta") is not None]
    return {
        "timestamp_cest": datetime.now().isoformat(),
        "method": method,
        "scope": "diagnostic_from_existing_predictions_not_temporal_validation",
        "n_prediction_rows_loaded": int(len(data)),
        "n_groups": int(len(rows)),
        "candidate_coverage_mean": float(np.mean(coverages)) if coverages else None,
        "coverage_delta_mean": float(np.mean(deltas)) if deltas else None,
        "rows": rows,
        "limitations": [
            "Existing prediction artifacts lack stable row keys.",
            "Calibration/test split is deterministic row-order split, not temporal validation.",
            "Use this only to decide which route deserves a real SLURM benchmark pilot.",
        ],
    }


def write_report(report: dict[str, Any], stem: str) -> tuple[Path, Path]:
    out_json = ROOT / "runs" / "audits" / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(out_md, "w") as fh:
        fh.write(f"# {stem}\n\n```json\n")
        fh.write(json.dumps(report, indent=2, default=str)[:20000])
        fh.write("\n```\n")
    return out_json, out_md


def print_summary(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    print(f"OK: {out_json}")
    print(f"OK: {out_md}")
    print(json.dumps({k: report.get(k) for k in ("method", "scope", "n_prediction_rows_loaded", "n_groups", "candidate_coverage_mean", "coverage_delta_mean", "limitations")}, indent=2, default=str))