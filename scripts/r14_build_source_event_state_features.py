#!/usr/bin/env python3
"""Build strict row-key source event-state features for TESF gate experiments.

This script operationalizes source evidence as telemetry features instead of a
residual multiplier. It joins prediction row keys to text embeddings and EDGAR
features, emits a feature parquet, and audits lane-specific read/no-read
confidence. The output is an experiment input, not a promotion pass by itself.
"""

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
TEXT_7B = ROOT / "runs" / "text_embeddings_gte_qwen2_7b_pca64_20260501_sharded" / "text_embeddings.parquet"
TEXT_15B = ROOT / "runs" / "text_embeddings" / "text_embeddings.parquet"
EDGAR_DIR = ROOT / "runs" / "edgar_feature_store_full_daily_wide_20260203_225620" / "edgar_features"
ABLATIONS = ("core_only", "core_text", "core_edgar", "full")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-glob", default="runs/benchmarks/r14fcast_cqrrow*_main_h30_*/predictions.parquet")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-rows-per-ablation", type=int, default=20000)
    parser.add_argument("--text-embeddings", default=str(TEXT_7B if TEXT_7B.exists() else TEXT_15B))
    parser.add_argument("--edgar-dir", default=str(EDGAR_DIR))
    parser.add_argument("--stem", default="r14_source_event_state_features")
    return parser.parse_args()


def _date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")


def _summary(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max())}


def _prediction_rows(args: argparse.Namespace) -> pd.DataFrame:
    columns = [
        "model", "task", "target", "horizon", "ablation", "entity_id",
        "crawled_date_day", "cik", "offer_id", "source_row_index", "y_true", "y_pred",
    ]
    by_ablation: dict[str, list[pd.DataFrame]] = {ablation: [] for ablation in ABLATIONS}
    counts = {ablation: 0 for ablation in ABLATIONS}
    paths = sorted((Path(path) for path in glob.glob(str(ROOT / args.prediction_glob))), key=lambda path: os.path.getmtime(path), reverse=True)
    for path in paths:
        if all(counts[ablation] >= args.max_rows_per_ablation for ablation in ABLATIONS):
            break
        try:
            parquet = pq.ParquetFile(path)
        except Exception:
            continue
        available = set(parquet.schema.names)
        read_cols = [column for column in columns if column in available]
        required = {"entity_id", "crawled_date_day", "source_row_index", "ablation", "target", "horizon"}
        if not required.issubset(read_cols):
            continue
        for batch in parquet.iter_batches(batch_size=50000, columns=read_cols):
            frame = batch.to_pandas()
            frame = frame.loc[frame["target"].astype(str).eq(args.target)].copy()
            frame = frame.loc[pd.to_numeric(frame["horizon"], errors="coerce").eq(int(args.horizon))].copy()
            frame = frame.loc[frame["ablation"].astype(str).isin(ABLATIONS)].copy()
            if frame.empty:
                continue
            frame["_prediction_path"] = str(path.relative_to(ROOT))
            for ablation, group in frame.groupby("ablation", sort=False):
                ablation = str(ablation)
                if counts.get(ablation, 0) >= args.max_rows_per_ablation:
                    continue
                remaining = args.max_rows_per_ablation - counts[ablation]
                chunk = group.head(remaining).copy()
                by_ablation[ablation].append(chunk)
                counts[ablation] += len(chunk)
            if all(counts[ablation] >= args.max_rows_per_ablation for ablation in ABLATIONS):
                break
    frames = [frame for frames in by_ablation.values() for frame in frames]
    if not frames:
        return pd.DataFrame(columns=columns + ["date_key", "row_key"])
    data = pd.concat(frames, ignore_index=True)
    data["date_key"] = _date_key(data["crawled_date_day"])
    data["entity_key"] = data["entity_id"].astype(str)
    data["cik_key"] = data["cik"].astype(str).where(data["cik"].notna(), "") if "cik" in data.columns else ""
    data["row_key"] = data[["task", "target", "horizon", "entity_key", "date_key", "source_row_index"]].astype(str).agg("|".join, axis=1)
    return data.drop_duplicates(["ablation", "row_key"], keep="last").reset_index(drop=True)


def _scan_text_features(text_path: Path, keys: set[tuple[str, str]]) -> pd.DataFrame:
    if not text_path.exists() or not keys:
        return pd.DataFrame(columns=["entity_key", "date_key", "text_available", "text_energy"])
    parquet = pq.ParquetFile(text_path)
    names = parquet.schema.names
    emb_cols = [column for column in names if column.startswith("text_emb_")]
    read_cols = [column for column in ["entity_id", "crawled_date_day", *emb_cols] if column in names]
    rows = []
    for batch in parquet.iter_batches(batch_size=200000, columns=read_cols):
        frame = batch.to_pandas()
        frame["date_key"] = _date_key(frame["crawled_date_day"])
        frame["entity_key"] = frame["entity_id"].astype(str)
        mask = np.array([(entity, day) in keys for entity, day in zip(frame["entity_key"], frame["date_key"])], dtype=bool)
        if not mask.any():
            continue
        sub = frame.loc[mask, ["entity_key", "date_key", *emb_cols]].copy()
        vals = sub[emb_cols].to_numpy(dtype=np.float32, copy=False) if emb_cols else np.zeros((len(sub), 0), dtype=np.float32)
        energy = np.linalg.norm(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), axis=1) if vals.size else np.zeros(len(sub), dtype=np.float32)
        out = sub[["entity_key", "date_key"]].copy()
        out["text_available"] = 1.0
        out["text_energy"] = energy.astype(np.float32)
        if vals.shape[1] >= 1:
            out["text_emb_0"] = vals[:, 0]
        if vals.shape[1] >= 2:
            out["text_emb_1"] = vals[:, 1]
        rows.append(out)
        if sum(len(row) for row in rows) >= len(keys):
            break
    if not rows:
        return pd.DataFrame(columns=["entity_key", "date_key", "text_available", "text_energy"])
    text = pd.concat(rows, ignore_index=True).drop_duplicates(["entity_key", "date_key"], keep="last")
    return text


def _edgar_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.parquet"))


def _scan_edgar_features(edgar_dir: Path, keys: set[tuple[str, str]]) -> pd.DataFrame:
    files = _edgar_files(edgar_dir)
    if not files or not keys:
        return pd.DataFrame(columns=["cik_key", "date_key", "edgar_available", "edgar_energy"])
    schema = pq.ParquetFile(files[0]).schema.names
    numeric_cols = [
        column for column in schema
        if column not in {"cik", "crawled_date_day", "cutoff_ts", "edgar_filed_date"}
        and not column.endswith("_is_missing")
        and column not in {"edgar_has_filing", "edgar_valid"}
    ][:64]
    aux_cols = [column for column in ("edgar_filed_date", "edgar_has_filing", "edgar_valid") if column in schema]
    rows = []
    for file in files:
        try:
            parquet = pq.ParquetFile(file)
        except Exception:
            continue
        available = set(parquet.schema.names)
        cols = [column for column in ["cik", "crawled_date_day", *aux_cols, *numeric_cols] if column in available]
        if "cik" not in cols or "crawled_date_day" not in cols:
            continue
        for batch in parquet.iter_batches(batch_size=100000, columns=cols):
            frame = batch.to_pandas()
            frame["date_key"] = _date_key(frame["crawled_date_day"])
            frame["cik_key"] = frame["cik"].astype(str)
            mask = np.array([(cik, day) in keys for cik, day in zip(frame["cik_key"], frame["date_key"])], dtype=bool)
            if not mask.any():
                continue
            sub = frame.loc[mask].copy()
            available_numeric = [column for column in numeric_cols if column in sub.columns]
            vals = sub[available_numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False) if available_numeric else np.zeros((len(sub), 0), dtype=np.float64)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            energy = np.linalg.norm(vals, axis=1) if vals.size else np.zeros(len(sub), dtype=np.float64)
            out = sub[["cik_key", "date_key"]].copy()
            out["edgar_available"] = 1.0
            out["edgar_energy"] = energy.astype(np.float32)
            out["edgar_nonzero"] = (energy > 1e-12).astype(np.float32)
            out["edgar_valid"] = pd.to_numeric(sub.get("edgar_valid", pd.Series(1.0, index=sub.index)), errors="coerce").fillna(0.0).astype(np.float32)
            if "edgar_filed_date" in sub.columns:
                filed = pd.to_datetime(sub["edgar_filed_date"], errors="coerce", utc=True)
                current = pd.to_datetime(sub["crawled_date_day"], errors="coerce", utc=True)
                out["edgar_recency_days"] = (current - filed).dt.days.astype("float64").clip(lower=0.0, upper=9999.0).fillna(9999.0).astype(np.float32)
            else:
                out["edgar_recency_days"] = np.float32(9999.0)
            rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["cik_key", "date_key", "edgar_available", "edgar_energy"])
    edgar = pd.concat(rows, ignore_index=True).drop_duplicates(["cik_key", "date_key"], keep="last")
    return edgar


def _confidence_from_energy(values: pd.Series, available: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    active_vals = vals.loc[(available.astype(float) > 0.0) & (vals > 0.0)]
    if len(active_vals) < 10:
        return pd.Series(np.where(vals > 0.0, 0.5, 0.0), index=values.index, dtype=np.float32)
    lo = float(np.quantile(active_vals, 0.10))
    hi = float(np.quantile(active_vals, 0.90))
    if hi <= lo:
        return pd.Series(np.where(vals > 0.0, 0.5, 0.0), index=values.index, dtype=np.float32)
    score = ((vals - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)
    return score.astype(np.float32)


def _add_read_policy(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["text_confidence"] = _confidence_from_energy(out["text_energy"], out["text_available"])
    edgar_energy_conf = _confidence_from_energy(out["edgar_energy"], out["edgar_available"])
    freshness = np.exp(-pd.to_numeric(out["edgar_recency_days"], errors="coerce").fillna(9999.0).clip(lower=0.0).to_numpy(dtype=float) / 365.0)
    out["edgar_confidence"] = (edgar_energy_conf.to_numpy(dtype=float) * out["edgar_nonzero"].to_numpy(dtype=float) * out["edgar_valid"].to_numpy(dtype=float) * freshness).astype(np.float32)
    ablation = out["ablation"].astype(str)
    out["text_ablation_allowed"] = ablation.isin(["core_text", "full"]).astype(np.float32)
    out["edgar_ablation_allowed"] = ablation.isin(["core_edgar", "full"]).astype(np.float32)
    out["text_effective_confidence"] = (out["text_confidence"] * out["text_ablation_allowed"]).astype(np.float32)
    out["edgar_effective_confidence"] = (out["edgar_confidence"] * out["edgar_ablation_allowed"]).astype(np.float32)
    out["source_any_available"] = ((out["text_available"] > 0.0) | (out["edgar_available"] > 0.0)).astype(np.float32)
    out["source_any_nonzero"] = ((out["text_energy"] > 1e-12) | (out["edgar_nonzero"] > 0.0)).astype(np.float32)
    out["source_any_allowed"] = ((out["text_ablation_allowed"] > 0.0) | (out["edgar_ablation_allowed"] > 0.0)).astype(np.float32)

    horizon = pd.to_numeric(out["horizon"], errors="coerce").fillna(30).to_numpy(dtype=float)
    out["binary_read_confidence"] = np.maximum(out["text_effective_confidence"], out["edgar_effective_confidence"]).astype(np.float32)
    out["funding_read_confidence"] = (0.45 * out["text_effective_confidence"] + 0.55 * out["edgar_effective_confidence"]).astype(np.float32)
    short_weight = np.where(horizon <= 1.0, 0.75, 0.55)
    out["investors_read_confidence"] = (short_weight * out["text_effective_confidence"] + (1.0 - short_weight) * out["edgar_effective_confidence"]).astype(np.float32)
    for lane in ("binary", "funding", "investors"):
        conf = out[f"{lane}_read_confidence"]
        out[f"{lane}_no_read_fallback"] = (conf < 0.10).astype(np.float32)
        out[f"{lane}_low_confidence_read"] = ((conf >= 0.10) & (conf < 0.55)).astype(np.float32)
        out[f"{lane}_high_confidence_read"] = (conf >= 0.55).astype(np.float32)
    return out


def main() -> int:
    args = _parse_args()
    predictions = _prediction_rows(args)
    text_keys = set(zip(predictions.get("entity_key", pd.Series(dtype=str)), predictions.get("date_key", pd.Series(dtype=str))))
    edgar_candidates = predictions.loc[predictions.get("cik_key", pd.Series(dtype=str)).astype(str).ne("")].copy() if not predictions.empty else predictions.copy()
    edgar_keys = set(zip(edgar_candidates.get("cik_key", pd.Series(dtype=str)), edgar_candidates.get("date_key", pd.Series(dtype=str))))
    text = _scan_text_features(Path(args.text_embeddings), text_keys)
    edgar = _scan_edgar_features(Path(args.edgar_dir), edgar_keys)

    features = predictions.merge(text, on=["entity_key", "date_key"], how="left")
    features = features.merge(edgar, on=["cik_key", "date_key"], how="left")
    defaults = {
        "text_available": 0.0,
        "text_energy": 0.0,
        "text_emb_0": 0.0,
        "text_emb_1": 0.0,
        "edgar_available": 0.0,
        "edgar_energy": 0.0,
        "edgar_nonzero": 0.0,
        "edgar_valid": 0.0,
        "edgar_recency_days": 9999.0,
    }
    for column, value in defaults.items():
        if column not in features.columns:
            features[column] = value
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(value).astype(np.float32)
    features = _add_read_policy(features)
    features["text_novelty_proxy"] = features.sort_values(["entity_key", "date_key", "source_row_index"]).groupby("entity_key")["text_energy"].diff().abs().fillna(0.0).astype(np.float32)
    features["source_event_state_version"] = "r14_20260503"

    out_dir = ROOT / "runs" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_path = out_dir / f"{args.stem}_{suffix}.parquet"
    audit_path = out_dir / f"{args.stem}_{suffix}.json"
    md_path = audit_path.with_suffix(".md")
    export_columns = [
        "row_key", "model", "task", "target", "horizon", "ablation", "entity_id", "crawled_date_day", "cik", "offer_id", "source_row_index",
        "text_available", "text_energy", "text_novelty_proxy", "text_confidence",
        "text_ablation_allowed", "text_effective_confidence",
        "edgar_available", "edgar_energy", "edgar_nonzero", "edgar_valid", "edgar_recency_days", "edgar_confidence",
        "edgar_ablation_allowed", "edgar_effective_confidence",
        "source_any_available", "source_any_nonzero", "source_any_allowed",
        "binary_read_confidence", "binary_no_read_fallback", "binary_low_confidence_read", "binary_high_confidence_read",
        "funding_read_confidence", "funding_no_read_fallback", "funding_low_confidence_read", "funding_high_confidence_read",
        "investors_read_confidence", "investors_no_read_fallback", "investors_low_confidence_read", "investors_high_confidence_read",
        "source_event_state_version",
    ]
    export_columns = [column for column in export_columns if column in features.columns]
    features[export_columns].to_parquet(feature_path, index=False)

    by_ablation = features.groupby("ablation").agg(
        rows=("row_key", "count"),
        text_match_rate=("text_available", "mean"),
        edgar_candidate_rate=("cik_key", lambda s: float((s.astype(str) != "").mean()) if len(s) else 0.0),
        edgar_match_rate=("edgar_available", "mean"),
        funding_no_read_rate=("funding_no_read_fallback", "mean"),
        funding_high_read_rate=("funding_high_confidence_read", "mean"),
    ).reset_index().to_dict(orient="records")
    audit = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if len(features) > 0 and float(features["text_available"].mean()) >= 0.95 else "partial",
        "feature_path": str(feature_path.relative_to(ROOT)),
        "n_rows": int(len(features)),
        "config": vars(args),
        "by_ablation": by_ablation,
        "summary": {
            "text_match_rate": float(features["text_available"].mean()) if len(features) else None,
            "text_energy": _summary(features["text_energy"]),
            "edgar_candidate_rate": float((features["cik_key"].astype(str) != "").mean()) if len(features) else None,
            "edgar_match_rate": float(features["edgar_available"].mean()) if len(features) else None,
            "edgar_nonzero_rate_among_matches": float(features.loc[features["edgar_available"] > 0, "edgar_nonzero"].mean()) if bool((features["edgar_available"] > 0).any()) else None,
            "funding_no_read_rate": float(features["funding_no_read_fallback"].mean()) if len(features) else None,
            "funding_high_read_rate": float(features["funding_high_confidence_read"].mean()) if len(features) else None,
            "investors_no_read_rate": float(features["investors_no_read_fallback"].mean()) if len(features) else None,
            "binary_no_read_rate": float(features["binary_no_read_fallback"].mean()) if len(features) else None,
            "source_any_allowed_rate": float(features["source_any_allowed"].mean()) if len(features) else None,
        },
        "interpretation": [
            "This table is the source-native event-state interface for later TESF ablations.",
            "Read confidence is telemetry, not an accuracy claim; formal gates require paired reruns.",
            "A high no-read rate is acceptable when source evidence is sparse or stale; it prevents source noise from forcing predictions.",
        ],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text("# R14 Source Event-State Feature Table Audit\n\n```json\n" + json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "feature_path": str(feature_path),
        "n_rows": audit["n_rows"],
        "summary": audit["summary"],
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())