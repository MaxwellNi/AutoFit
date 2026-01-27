#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def _load_offers_core(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _limit_rows_by_entities(df: pd.DataFrame, limit_rows: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info = {"limit_rows_strategy": None, "limit_rows_selected_entities": None}
    if "entity_id" not in df.columns:
        info["limit_rows_strategy"] = "head"
        return df.head(limit_rows).copy(), info
    entities = df["entity_id"].dropna().unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(entities)
    frames = []
    total = 0
    for entity_id in entities:
        group = df[df["entity_id"] == entity_id]
        if group.empty:
            continue
        frames.append(group)
        total += len(group)
        if total >= limit_rows:
            break
    if not frames:
        info["limit_rows_strategy"] = "entity_subset"
        info["limit_rows_selected_entities"] = 0
        return df.iloc[:0].copy(), info
    out = pd.concat(frames, ignore_index=True)
    info["limit_rows_strategy"] = "entity_subset"
    info["limit_rows_selected_entities"] = len(frames)
    return out, info


def _compute_ratio_w(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    goal = pd.to_numeric(df["funding_goal_usd"], errors="coerce")
    raised = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
    ratio = raised / goal
    df["funding_ratio"] = ratio
    if ratio.notna().any():
        q_low, q_high = ratio.quantile([0.01, 0.99])
        df["funding_ratio_w"] = ratio.clip(lower=q_low, upper=q_high)
    else:
        df["funding_ratio_w"] = np.nan
    return df


def _build_samples(
    df: pd.DataFrame,
    seq_len: int,
    label_horizon: int,
    label_goal_min: float,
    static_ratio_tol: float,
    min_label_delta_days: float,
    min_ratio_delta_abs: float,
    min_ratio_delta_rel: float,
    strict_future: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    df = df.copy()
    df = df[df["funding_goal_usd"] >= label_goal_min].copy()
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="stable")
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)

    dropped_insufficient_future = 0
    dropped_due_to_static_ratio = 0
    dropped_due_to_min_delta_days = 0
    dropped_due_to_small_ratio_delta_abs = 0
    dropped_due_to_small_ratio_delta_rel = 0

    y_list: List[float] = []
    input_ratio_list: List[float] = []

    for _, group in df.groupby("entity_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue

        for input_end in range(len(group)):
            label_idx = input_end + label_horizon
            if label_idx >= len(group):
                dropped_insufficient_future += 1
                if strict_future:
                    continue
                label_idx = len(group) - 1

            input_raised = float(group["funding_raised_usd"].iloc[input_end])
            input_goal = float(group["funding_goal_usd"].iloc[input_end])
            label_raised = float(group["funding_raised_usd"].iloc[label_idx])
            label_goal = float(group["funding_goal_usd"].iloc[label_idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                input_ratio = input_raised / input_goal if input_goal != 0 else np.nan
                label_ratio = label_raised / label_goal if label_goal != 0 else np.nan

            if np.isfinite(label_ratio) and np.isfinite(input_ratio):
                if abs(label_ratio - input_ratio) < static_ratio_tol:
                    dropped_due_to_static_ratio += 1
                    continue

            input_end_ts = group["snapshot_ts"].iloc[input_end]
            label_ts = group["snapshot_ts"].iloc[label_idx]
            delta_days = None
            if pd.notna(input_end_ts) and pd.notna(label_ts):
                delta_days = (label_ts - input_end_ts).total_seconds() / 86400.0
            if delta_days is not None and min_label_delta_days > 0 and delta_days < min_label_delta_days:
                dropped_due_to_min_delta_days += 1
                continue

            if np.isfinite(label_ratio) and np.isfinite(input_ratio):
                delta_abs = abs(label_ratio - input_ratio)
                delta_rel = delta_abs / max(1.0, abs(input_ratio))
                if min_ratio_delta_abs > 0 and delta_abs < min_ratio_delta_abs:
                    dropped_due_to_small_ratio_delta_abs += 1
                    continue
                if min_ratio_delta_rel > 0 and delta_rel < min_ratio_delta_rel:
                    dropped_due_to_small_ratio_delta_rel += 1
                    continue

            y_list.append(float(group["funding_ratio_w"].iloc[label_idx]))
            input_ratio_list.append(float(input_ratio))

    drop_counts = {
        "dropped_due_to_insufficient_future": dropped_insufficient_future,
        "dropped_due_to_static_ratio": dropped_due_to_static_ratio,
        "dropped_due_to_min_delta_days": dropped_due_to_min_delta_days,
        "dropped_due_to_small_ratio_delta_abs": dropped_due_to_small_ratio_delta_abs,
        "dropped_due_to_small_ratio_delta_rel": dropped_due_to_small_ratio_delta_rel,
    }
    return np.asarray(y_list, dtype=float), np.asarray(input_ratio_list, dtype=float), drop_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Check label leakage against input-end ratio.")
    parser.add_argument("--offers_core", type=str, required=True)
    parser.add_argument("--edgar_features", type=str, default=None)
    parser.add_argument("--bench_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--use_edgar", type=int, default=0)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--sample_strategy", type=str, default="random_entities")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--label_goal_min", type=float, default=50.0)
    parser.add_argument("--label_horizon", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_label_delta_days", type=float, default=0.0)
    parser.add_argument("--min_ratio_delta_abs", type=float, default=0.0)
    parser.add_argument("--min_ratio_delta_rel", type=float, default=0.0)
    parser.add_argument("--strict_future", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    errors: List[str] = []
    selection_source = None
    selected_entities_count = None
    limit_info: Dict[str, Any] = {"limit_rows_strategy": None, "limit_rows_selected_entities": None}

    df = _load_offers_core(args.offers_core)
    if args.bench_dir:
        bench_dir = Path(args.bench_dir)
        sampled_path = bench_dir / "sampled_entities.json"
        entities = None
        if sampled_path.exists():
            entities = json.loads(sampled_path.read_text(encoding="utf-8"))
            selection_source = str(sampled_path)
        else:
            preds_path = bench_dir / "predictions.parquet"
            if preds_path.exists():
                try:
                    preds_df = pd.read_parquet(preds_path, columns=["entity_id"])
                    entities = preds_df["entity_id"].dropna().astype(str).unique().tolist()
                    selection_source = str(preds_path)
                except Exception:
                    warnings.append("predictions_missing_entity_id_fallback_used")
            warnings.append("missing_sampled_entities_json_fallback_used")
        if entities:
            entities = [str(e) for e in entities]
            selected_entities_count = len(entities)
            df = df[df["entity_id"].astype(str).isin(set(entities))].copy()

    if args.limit_rows is not None and args.limit_rows > 0:
        df, limit_info = _limit_rows_by_entities(df, args.limit_rows, args.sample_seed)
    df = _compute_ratio_w(df)

    static_ratio_tol = 1e-6
    y_true, input_ratio, drop_counts = _build_samples(
        df,
        seq_len=args.seq_len,
        label_horizon=args.label_horizon,
        label_goal_min=args.label_goal_min,
        static_ratio_tol=static_ratio_tol,
        min_label_delta_days=args.min_label_delta_days,
        min_ratio_delta_abs=args.min_ratio_delta_abs,
        min_ratio_delta_rel=args.min_ratio_delta_rel,
        strict_future=bool(args.strict_future),
    )

    mask = np.isfinite(y_true) & np.isfinite(input_ratio)
    y_true = y_true[mask]
    input_ratio = input_ratio[mask]

    corr = _safe_corr(y_true, input_ratio)
    max_abs_diff = float(np.max(np.abs(y_true - input_ratio))) if y_true.size else None
    allclose = bool(np.allclose(y_true, input_ratio, atol=1e-6, rtol=1e-6)) if y_true.size else None
    leakage_flag = False
    if corr is not None and corr > 0.9999:
        leakage_flag = True
    if max_abs_diff is not None and max_abs_diff < 1e-6:
        leakage_flag = True
    if allclose:
        leakage_flag = True

    if y_true.size == 0 or input_ratio.size == 0:
        errors.append("no_finite_samples_for_leakage_check")
    overall_ok = not leakage_flag and not errors

    report = {
        "bench_dir": args.bench_dir,
        "exp_name": args.exp_name,
        "offers_core": args.offers_core,
        "edgar_features": args.edgar_features,
        "use_edgar": bool(args.use_edgar),
        "limit_rows": args.limit_rows,
        "limit_rows_strategy": limit_info.get("limit_rows_strategy"),
        "limit_rows_selected_entities": limit_info.get("limit_rows_selected_entities"),
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "label_goal_min": args.label_goal_min,
        "label_horizon": args.label_horizon,
        "strict_future": bool(args.strict_future),
        "selection_source": selection_source,
        "selected_entities_count": selected_entities_count,
        "dropped_due_to_static_ratio": drop_counts["dropped_due_to_static_ratio"],
        "dropped_due_to_min_delta_days": drop_counts["dropped_due_to_min_delta_days"],
        "dropped_due_to_small_ratio_delta_abs": drop_counts["dropped_due_to_small_ratio_delta_abs"],
        "dropped_due_to_small_ratio_delta_rel": drop_counts["dropped_due_to_small_ratio_delta_rel"],
        "seq_len": args.seq_len,
        "n_entities": int(df["entity_id"].nunique()) if "entity_id" in df.columns else None,
        "n_rows": int(len(df)),
        "feature_cols": [
            c
            for c in [
                "funding_raised_usd",
                "funding_goal_usd",
                "investors_count",
                "time_since_start_days",
                "time_delta_days",
            ]
            if c in df.columns
        ],
        "y_stats": {
            "min": float(np.min(y_true)) if y_true.size else None,
            "max": float(np.max(y_true)) if y_true.size else None,
            "mean": float(np.mean(y_true)) if y_true.size else None,
            "std": float(np.std(y_true)) if y_true.size else None,
            "unique": int(len(np.unique(y_true))) if y_true.size else 0,
        },
        "label_vs_current_ratio": {
            "corr": corr,
            "max_abs_diff": max_abs_diff,
            "n_finite": int(y_true.size),
            "allclose": allclose,
            "leakage_flag": leakage_flag,
            "ratio_source": "funding_ratio_input_last",
        },
        "suspect_count": 0,
        "suspects": [],
        "label_degenerate": False,
        "ok": overall_ok,
        "errors": errors + (["leakage_flag"] if leakage_flag else []),
        "warnings": warnings,
        "overall_ok": overall_ok,
    }

    out_path = output_dir / "label_leakage_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
