#!/usr/bin/env python3
"""Row-key diagnostic for source-regime conformal calibration.

This script consumes landed prediction parquet files and tests whether source
regimes are useful calibration groups. It is deliberately marked diagnostic:
formal promotion still requires benchmark-harness temporal reruns.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ABLATION_TO_REGIME = {
    "core_only": "no_source",
    "core_text": "text_regime",
    "core_edgar": "edgar_regime",
    "full": "full_source_regime",
}
BASE_COLUMNS = ["y_true", "y_pred", "model", "task", "ablation", "horizon", "target"]
IDENTITY_COLUMNS = ["entity_id", "crawled_date_day", "offer_id", "cik", "source_row_index"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="runs/benchmarks/r14fcast_cqrrow*_*/predictions.parquet")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--max-rows-per-file", type=int, default=60000)
    parser.add_argument("--calibration-fraction", type=float, default=0.60)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-group-cal", type=int, default=250)
    parser.add_argument("--recency-weight-decay", type=float, default=0.9995)
    parser.add_argument("--stem", default="r14_source_regime_rowkey_conformal_diagnostic")
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _q(values: np.ndarray, level: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, min(max(level, 0.0), 1.0), method="higher"))


def _weighted_q(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    if len(values) == 0:
        return float("nan")
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not mask.any():
        return _q(values[np.isfinite(values)], level)
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = int(np.searchsorted(cdf, min(max(level, 0.0), 1.0), side="left"))
    return float(values[min(idx, len(values) - 1)])


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi))) if len(y) else float("nan")


def _width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo)) if len(lo) else float("nan")


def _safe_read(path: Path, target: str, horizon: int, max_rows: int) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except Exception:  # pragma: no cover - fallback for minimal envs
        pq = None
    if pq is None:
        frame = pd.read_parquet(path)
        return _post_read(frame, path, target, horizon, max_rows)

    parquet = pq.ParquetFile(path)
    names = list(parquet.schema.names)
    columns = [column for column in BASE_COLUMNS + IDENTITY_COLUMNS if column in names]
    chunks = []
    for batch in parquet.iter_batches(batch_size=50000, columns=columns):
        frame = batch.to_pandas()
        frame = _filter(frame, target, horizon)
        if not frame.empty:
            chunks.append(frame)
        if sum(len(chunk) for chunk in chunks) >= max_rows:
            break
    if not chunks:
        return pd.DataFrame(columns=BASE_COLUMNS + IDENTITY_COLUMNS)
    return _post_read(pd.concat(chunks, ignore_index=True), path, target, horizon, max_rows)


def _filter(frame: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    if "target" in frame.columns:
        frame = frame.loc[frame["target"].astype(str).eq(str(target))].copy()
    if "horizon" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["horizon"], errors="coerce").eq(int(horizon))].copy()
    if "ablation" in frame.columns:
        frame = frame.loc[frame["ablation"].astype(str).isin(ABLATION_TO_REGIME)].copy()
    return frame


def _post_read(frame: pd.DataFrame, path: Path, target: str, horizon: int, max_rows: int) -> pd.DataFrame:
    frame = _filter(frame, target, horizon)
    if frame.empty:
        return frame
    frame = frame.head(max_rows).copy()
    for column in BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])].copy()
    frame["source_regime"] = frame["ablation"].astype(str).map(ABLATION_TO_REGIME)
    frame["abs_residual"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["_metrics_path"] = str(path.relative_to(ROOT))
    frame["_mtime"] = os.path.getmtime(path)
    strict_cols = [column for column in ("entity_id", "crawled_date_day", "source_row_index") if column in frame.columns]
    frame["_rowkey_quality"] = "strict" if len(strict_cols) >= 3 else "fallback_row_order"
    if "source_row_index" not in frame.columns:
        frame["source_row_index"] = np.arange(len(frame), dtype=np.int64)
    key_cols = [column for column in ("task", "target", "horizon", "entity_id", "crawled_date_day", "offer_id", "source_row_index") if column in frame.columns]
    frame["row_key"] = frame[key_cols].astype(str).agg("|".join, axis=1)
    return frame


def _load(args: argparse.Namespace) -> pd.DataFrame:
    paths = [Path(path) for path in glob.glob(str(ROOT / args.glob))]
    paths = sorted(paths, key=lambda path: os.path.getmtime(path), reverse=True)[: max(1, args.max_files)]
    frames = []
    for path in paths:
        try:
            frame = _safe_read(path, args.target, args.horizon, args.max_rows_per_file)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=BASE_COLUMNS + ["source_regime", "row_key", "abs_residual", "_rowkey_quality"])
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("_mtime").drop_duplicates(
        subset=["model", "task", "target", "horizon", "ablation", "row_key"],
        keep="last",
    )
    return data.reset_index(drop=True)


def _split(data: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal_parts = []
    test_parts = []
    sort_cols = [column for column in ("crawled_date_day", "source_row_index", "row_key") if column in data.columns]
    for _, group in data.groupby(["model", "task", "target", "horizon", "ablation"], sort=False):
        group = group.sort_values(sort_cols).reset_index(drop=True)
        cut = int(len(group) * frac)
        cut = min(max(cut, 2), max(len(group) - 1, 2))
        cal_parts.append(group.iloc[:cut].copy())
        test_parts.append(group.iloc[cut:].copy())
    cal = pd.concat(cal_parts, ignore_index=True) if cal_parts else data.iloc[:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else data.iloc[:0].copy()
    return cal, test


def _interval_score(test: pd.DataFrame, q: np.ndarray) -> dict[str, Any]:
    pred = test["y_pred"].to_numpy(dtype=float)
    y = test["y_true"].to_numpy(dtype=float)
    lo = pred - q
    hi = pred + q
    return {
        "coverage": _coverage(y, lo, hi),
        "width_mean": _width(lo, hi),
        "n_test": int(len(test)),
    }


def _by_regime(test: pd.DataFrame, q: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    work = test.copy()
    work["_q"] = q
    for regime, group in work.groupby("source_regime", sort=True):
        out[str(regime)] = _interval_score(group, group["_q"].to_numpy(dtype=float))
    return out


def _marginal(cal: pd.DataFrame, test: pd.DataFrame, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    q = _q(cal["abs_residual"].to_numpy(dtype=float), 1.0 - alpha)
    q_arr = np.full(len(test), q, dtype=float)
    return q_arr, {"q": q}


def _regime_mondrian(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, min_group: int) -> tuple[np.ndarray, dict[str, Any]]:
    global_q = _q(cal["abs_residual"].to_numpy(dtype=float), 1.0 - alpha)
    q_by_regime = {}
    for regime, group in cal.groupby("source_regime", sort=True):
        q_by_regime[str(regime)] = _q(group["abs_residual"].to_numpy(dtype=float), 1.0 - alpha) if len(group) >= min_group else global_q
    q_arr = np.array([q_by_regime.get(str(regime), global_q) for regime in test["source_regime"]], dtype=float)
    return q_arr, {"global_q": global_q, "q_by_regime": q_by_regime, "min_group_cal": int(min_group)}


def _regime_yhat_bins(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, min_group: int, bins: int = 4) -> tuple[np.ndarray, dict[str, Any]]:
    global_q = _q(cal["abs_residual"].to_numpy(dtype=float), 1.0 - alpha)
    q_by_key: dict[tuple[str, int], float] = {}
    edges_by_regime: dict[str, list[float]] = {}
    for regime, group in cal.groupby("source_regime", sort=True):
        edges = np.unique(np.quantile(group["y_pred"].to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) <= 2:
            edges_by_regime[str(regime)] = []
            continue
        edges_by_regime[str(regime)] = [float(value) for value in edges]
        cal_bins = np.digitize(group["y_pred"].to_numpy(dtype=float), edges[1:-1], right=False)
        for bin_id in range(len(edges) - 1):
            vals = group.loc[cal_bins == bin_id, "abs_residual"].to_numpy(dtype=float)
            q_by_key[(str(regime), int(bin_id))] = _q(vals, 1.0 - alpha) if len(vals) >= min_group else global_q
    q_vals = []
    for _, row in test.iterrows():
        regime = str(row["source_regime"])
        edges = edges_by_regime.get(regime, [])
        if len(edges) <= 2:
            q_vals.append(global_q)
            continue
        bin_id = int(np.digitize([float(row["y_pred"])], np.asarray(edges[1:-1]), right=False)[0])
        q_vals.append(q_by_key.get((regime, bin_id), global_q))
    return np.asarray(q_vals, dtype=float), {"global_q": global_q, "q_by_regime_bin": {f"{k[0]}:{k[1]}": v for k, v in q_by_key.items()}, "edges_by_regime": edges_by_regime}


def _regime_drift_guard(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, min_group: int) -> tuple[np.ndarray, dict[str, Any]]:
    global_q = _q(cal["abs_residual"].to_numpy(dtype=float), 1.0 - alpha)
    q_by_regime = {}
    diagnostics = {}
    for regime, group in cal.groupby("source_regime", sort=True):
        vals = group["abs_residual"].to_numpy(dtype=float)
        if len(vals) < max(min_group, 10):
            q_by_regime[str(regime)] = global_q
            diagnostics[str(regime)] = {"fallback": "small_group", "n_cal": int(len(vals))}
            continue
        split = int(len(vals) * 0.80)
        split = min(max(split, 2), len(vals) - 1)
        early = vals[:split]
        late = vals[split:]
        nominal_q = _q(early, 1.0 - alpha)
        holdout_cov = float(np.mean(late <= nominal_q)) if len(late) else float("nan")
        adjusted = min(0.99, 1.0 - alpha + max(0.0, (1.0 - alpha) - holdout_cov))
        q_by_regime[str(regime)] = _q(vals, adjusted)
        diagnostics[str(regime)] = {"n_cal": int(len(vals)), "holdout_coverage": holdout_cov, "adjusted_level": adjusted}
    q_arr = np.array([q_by_regime.get(str(regime), global_q) for regime in test["source_regime"]], dtype=float)
    return q_arr, {"global_q": global_q, "q_by_regime": q_by_regime, "diagnostics": diagnostics}


def _regime_recency_weighted(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, min_group: int, decay: float) -> tuple[np.ndarray, dict[str, Any]]:
    decay = float(min(max(decay, 0.90), 1.0))
    ordered = cal.reset_index(drop=True).copy()
    global_weights = np.power(decay, np.arange(len(ordered) - 1, -1, -1, dtype=np.float64))
    global_q = _weighted_q(ordered["abs_residual"].to_numpy(dtype=float), global_weights, 1.0 - alpha)
    q_by_regime = {}
    diagnostics = {}
    for regime, group in ordered.groupby("source_regime", sort=True):
        vals = group["abs_residual"].to_numpy(dtype=float)
        if len(vals) < min_group:
            q_by_regime[str(regime)] = global_q
            diagnostics[str(regime)] = {"fallback": "small_group", "n_cal": int(len(vals))}
            continue
        weights = np.power(decay, np.arange(len(vals) - 1, -1, -1, dtype=np.float64))
        q_by_regime[str(regime)] = _weighted_q(vals, weights, 1.0 - alpha)
        diagnostics[str(regime)] = {
            "n_cal": int(len(vals)),
            "oldest_weight": float(weights[0]),
            "newest_weight": float(weights[-1]),
        }
    q_arr = np.array([q_by_regime.get(str(regime), global_q) for regime in test["source_regime"]], dtype=float)
    return q_arr, {"global_q": global_q, "q_by_regime": q_by_regime, "decay": decay, "diagnostics": diagnostics}


def _summarize_method(name: str, test: pd.DataFrame, q: np.ndarray, details: dict[str, Any]) -> dict[str, Any]:
    summary = {"method": name, **_interval_score(test, q), "details": details, "by_regime": _by_regime(test, q)}
    return summary


def _strict_overlap(data: pd.DataFrame) -> dict[str, Any]:
    if "_rowkey_quality" not in data.columns:
        return {"n_strict_rows": 0, "n_core_source_pairs": 0, "overlap_note": "row-key quality column missing"}
    strict = data.loc[data["_rowkey_quality"].eq("strict")].copy()
    if strict.empty:
        return {"n_strict_rows": 0, "n_core_source_pairs": 0, "overlap_note": "no strict row-key columns"}
    by_key = defaultdict(dict)
    for row in strict[["model", "task", "target", "horizon", "row_key", "ablation", "y_pred", "y_true"]].itertuples(index=False):
        by_key[(row.model, row.task, row.target, row.horizon, row.row_key)][row.ablation] = row
    deltas = []
    for values in by_key.values():
        core = values.get("core_only")
        if core is None:
            continue
        for ablation in ("core_text", "core_edgar", "full"):
            candidate = values.get(ablation)
            if candidate is not None:
                deltas.append(float(candidate.y_pred) - float(core.y_pred))
    return {
        "n_strict_rows": int(len(strict)),
        "n_strict_row_keys": int(len(by_key)),
        "n_core_source_pairs": int(len(deltas)),
        "prediction_delta_mean": float(statistics.mean(deltas)) if deltas else None,
        "prediction_delta_abs_mean": float(statistics.mean(abs(value) for value in deltas)) if deltas else None,
    }


def main() -> int:
    args = _parse_args()
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be in (0, 1)")
    if not (0.1 <= args.calibration_fraction <= 0.9):
        raise ValueError("--calibration-fraction must be in [0.1, 0.9]")
    data = _load(args)
    cal, test = _split(data, args.calibration_fraction) if len(data) else (data.copy(), data.copy())
    methods = []
    if len(cal) >= 2 and len(test) >= 1:
        for name, builder in (
            ("marginal", lambda: _marginal(cal, test, args.alpha)),
            ("source_regime_mondrian", lambda: _regime_mondrian(cal, test, args.alpha, args.min_group_cal)),
            ("source_regime_yhat_bin_mondrian", lambda: _regime_yhat_bins(cal, test, args.alpha, args.min_group_cal)),
            ("source_regime_recency_weighted", lambda: _regime_recency_weighted(cal, test, args.alpha, args.min_group_cal, args.recency_weight_decay)),
            ("source_regime_drift_guard", lambda: _regime_drift_guard(cal, test, args.alpha, args.min_group_cal)),
        ):
            q, details = builder()
            methods.append(_summarize_method(name, test, q, details))
    baseline = next((row for row in methods if row["method"] == "marginal"), None)
    if baseline:
        for row in methods:
            row["coverage_delta_vs_marginal"] = None if row.get("coverage") is None else row["coverage"] - baseline["coverage"]
            row["width_delta_vs_marginal"] = None if row.get("width_mean") is None else row["width_mean"] - baseline["width_mean"]
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "scope": "diagnostic row-key source-regime conformal from landed predictions",
        "config": vars(args),
        "n_rows_loaded": int(len(data)),
        "n_calibration_rows": int(len(cal)),
        "n_test_rows": int(len(test)),
        "rowkey_quality_counts": data.get("_rowkey_quality", pd.Series(dtype=str)).value_counts().to_dict(),
        "regime_counts": data.get("source_regime", pd.Series(dtype=str)).value_counts().to_dict(),
        "strict_overlap_audit": _strict_overlap(data),
        "methods": methods,
        "status": "diagnostic_completed" if methods else "not_enough_rows",
        "limitations": [
            "This is not a formal temporal benchmark rerun.",
            "Rows are capped per file to keep diagnostics tractable.",
            "Promotion requires benchmark-harness reruns with stable row keys and no test-label leakage.",
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"{args.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text("# R14 Source Regime Row-Key Conformal Diagnostic\n\n```json\n" + json.dumps(report, indent=2, default=str)[:40000] + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "n_rows_loaded": report["n_rows_loaded"],
        "n_calibration_rows": report["n_calibration_rows"],
        "n_test_rows": report["n_test_rows"],
        "rowkey_quality_counts": report["rowkey_quality_counts"],
        "methods": [{"method": row["method"], "coverage": row["coverage"], "width_mean": row["width_mean"], "coverage_delta_vs_marginal": row.get("coverage_delta_vs_marginal")} for row in methods],
        "out_json": str(out_json),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())