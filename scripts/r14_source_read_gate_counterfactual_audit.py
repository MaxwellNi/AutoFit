#!/usr/bin/env python3
"""Audit whether read-confidence buckets predict source-vs-core gains."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ABLATIONS = ("core_text", "core_edgar", "full")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-glob", default="runs/benchmarks/r14fcast_cqrrow*_main_h30_*/predictions.parquet")
    parser.add_argument("--feature-glob", default="runs/audits/r14_source_event_state_features_*.parquet")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-rows-per-file", type=int, default=60000)
    parser.add_argument("--stem", default="r14_source_read_gate_counterfactual_audit")
    return parser.parse_args()


def _date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_prediction_rows(args: argparse.Namespace) -> pd.DataFrame:
    columns = ["model", "task", "target", "horizon", "ablation", "entity_id", "crawled_date_day", "offer_id", "source_row_index", "y_true", "y_pred"]
    frames = []
    paths = sorted((Path(path) for path in glob.glob(str(ROOT / args.prediction_glob))), key=lambda path: os.path.getmtime(path), reverse=True)
    for path in paths:
        try:
            parquet = pq.ParquetFile(path)
        except Exception:
            continue
        available = set(parquet.schema.names)
        read_cols = [column for column in columns if column in available]
        required = {"model", "task", "target", "horizon", "ablation", "entity_id", "crawled_date_day", "source_row_index", "y_true", "y_pred"}
        if not required.issubset(read_cols):
            continue
        loaded = 0
        for batch in parquet.iter_batches(batch_size=50000, columns=read_cols):
            frame = batch.to_pandas()
            frame = frame.loc[frame["target"].astype(str).eq(args.target)].copy()
            frame = frame.loc[pd.to_numeric(frame["horizon"], errors="coerce").eq(int(args.horizon))].copy()
            frame = frame.loc[frame["ablation"].astype(str).isin(("core_only", *SOURCE_ABLATIONS))].copy()
            if frame.empty:
                continue
            frame["_prediction_path"] = str(path.relative_to(ROOT))
            frame["_mtime"] = os.path.getmtime(path)
            frames.append(frame)
            loaded += len(frame)
            if loaded >= args.max_rows_per_file:
                break
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["date_key"] = _date_key(data["crawled_date_day"])
    data["entity_key"] = data["entity_id"].astype(str)
    data["row_key"] = data[["task", "target", "horizon", "entity_key", "date_key", "source_row_index"]].astype(str).agg("|".join, axis=1)
    data["y_true"] = pd.to_numeric(data["y_true"], errors="coerce")
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
    data = data[np.isfinite(data["y_true"]) & np.isfinite(data["y_pred"])].copy()
    data = data.sort_values("_mtime").drop_duplicates(["model", "task", "target", "horizon", "ablation", "row_key"], keep="last")
    data["abs_error"] = (data["y_true"] - data["y_pred"]).abs()
    return data.reset_index(drop=True)


def _latest_features(pattern: str) -> pd.DataFrame:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        return pd.DataFrame()
    path = paths[-1]
    frame = pd.read_parquet(path)
    frame["_feature_path"] = str(path.relative_to(ROOT))
    return frame


def _bucket(confidence: pd.Series) -> pd.Series:
    conf = pd.to_numeric(confidence, errors="coerce").fillna(0.0)
    return pd.Series(np.select([conf < 0.10, conf < 0.55], ["no_read", "low_confidence_read"], default="high_confidence_read"), index=confidence.index)


def _summarize(group: pd.DataFrame) -> dict[str, Any]:
    deltas = pd.to_numeric(group["delta_abs_error_source_minus_core"], errors="coerce").dropna()
    if deltas.empty:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(len(deltas)),
        "mean_delta_abs_error_source_minus_core": float(deltas.mean()),
        "median_delta_abs_error_source_minus_core": float(deltas.median()),
        "wins_source_lower_error": int((deltas < -1e-9).sum()),
        "losses_source_higher_error": int((deltas > 1e-9).sum()),
        "ties": int((deltas.abs() <= 1e-9).sum()),
        "win_rate": float((deltas < -1e-9).mean()),
        "source_abs_error_mean": float(pd.to_numeric(group["source_abs_error"], errors="coerce").mean()),
        "core_abs_error_mean": float(pd.to_numeric(group["core_abs_error"], errors="coerce").mean()),
        "read_confidence_mean": _finite(pd.to_numeric(group["funding_read_confidence"], errors="coerce").mean()),
    }


def main() -> int:
    args = _parse_args()
    preds = _read_prediction_rows(args)
    features = _latest_features(args.feature_glob)
    pairs = []
    if not preds.empty and not features.empty:
        core = preds.loc[preds["ablation"].eq("core_only")].copy()
        core = core[["model", "task", "target", "horizon", "row_key", "abs_error", "y_pred"]].rename(columns={"abs_error": "core_abs_error", "y_pred": "core_y_pred"})
        feature_cols = [
            "row_key", "ablation", "text_effective_confidence", "edgar_effective_confidence",
            "funding_read_confidence", "funding_no_read_fallback", "funding_low_confidence_read", "funding_high_confidence_read",
            "source_any_allowed", "text_novelty_proxy",
        ]
        feature_cols = [column for column in feature_cols if column in features.columns]
        features = features[feature_cols].drop_duplicates(["ablation", "row_key"], keep="last")
        for ablation in SOURCE_ABLATIONS:
            source = preds.loc[preds["ablation"].eq(ablation)].copy()
            if source.empty:
                continue
            source = source[["model", "task", "target", "horizon", "row_key", "abs_error", "y_pred"]].rename(columns={"abs_error": "source_abs_error", "y_pred": "source_y_pred"})
            merged = source.merge(core, on=["model", "task", "target", "horizon", "row_key"], how="inner", validate="one_to_one")
            merged["source_ablation"] = ablation
            merged = merged.merge(features.loc[features["ablation"].eq(ablation)].drop(columns=["ablation"]), on="row_key", how="left")
            merged["funding_read_confidence"] = pd.to_numeric(merged.get("funding_read_confidence", 0.0), errors="coerce").fillna(0.0)
            merged["read_bucket"] = _bucket(merged["funding_read_confidence"])
            merged["delta_abs_error_source_minus_core"] = merged["source_abs_error"] - merged["core_abs_error"]
            pairs.append(merged)
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    by_ablation = []
    by_bucket = []
    by_ablation_bucket = []
    if not paired.empty:
        for ablation, group in paired.groupby("source_ablation", sort=True):
            by_ablation.append({"source_ablation": ablation, **_summarize(group)})
        for bucket, group in paired.groupby("read_bucket", sort=True):
            by_bucket.append({"read_bucket": bucket, **_summarize(group)})
        for (ablation, bucket), group in paired.groupby(["source_ablation", "read_bucket"], sort=True):
            by_ablation_bucket.append({"source_ablation": ablation, "read_bucket": bucket, **_summarize(group)})
    promotable_buckets = [
        row for row in by_bucket
        if row.get("read_bucket") != "no_read"
        and (row.get("n_pairs") or 0) >= 1000
        and (row.get("win_rate") or 0.0) >= 0.55
        and (row.get("mean_delta_abs_error_source_minus_core") or 0.0) < 0.0
    ]
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if promotable_buckets else "not_passed",
        "scope": "strict row-key source-vs-core counterfactual stratified by source read confidence",
        "config": vars(args),
        "n_prediction_rows": int(len(preds)),
        "n_counterfactual_pairs": int(len(paired)),
        "feature_path": None if features.empty or "_feature_path" not in features.columns else str(features["_feature_path"].iloc[0]),
        "overall": _summarize(paired) if not paired.empty else {"n_pairs": 0},
        "by_ablation": by_ablation,
        "by_read_bucket": by_bucket,
        "by_ablation_read_bucket": by_ablation_bucket,
        "pass_rule": {
            "non_no_read_bucket_min_pairs": 1000,
            "non_no_read_bucket_min_win_rate": 0.55,
            "requires_mean_delta_abs_error_source_minus_core_below_zero": True,
            "reason": "Source can win many small rows while losing badly on large-error rows; promotion requires both count and magnitude improvement.",
        },
        "interpretation": [
            "If high-confidence buckets do not beat core_only, the current model still fails to convert source event-state telemetry into point-forecast signal.",
            "A not_passed status does not invalidate source telemetry; it blocks source-read promotion until model training/read policy changes.",
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"{args.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text("# R14 Source Read Gate Counterfactual Audit\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False, default=str)[:50000] + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "n_prediction_rows": report["n_prediction_rows"],
        "n_counterfactual_pairs": report["n_counterfactual_pairs"],
        "overall": report["overall"],
        "by_read_bucket": by_bucket,
        "out_json": str(out_json),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())