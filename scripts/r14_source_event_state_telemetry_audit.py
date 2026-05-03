#!/usr/bin/env python3
"""Audit source event-state telemetry on strict prediction row keys.

The goal is to verify whether text/EDGAR can be represented as usable event
state signals on the exact rows being evaluated: coverage, key alignment, and
nonzero source energy. This is not a forecasting pass/fail by itself.
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
TEXT_DEFAULT_7B = ROOT / "runs" / "text_embeddings_gte_qwen2_7b_pca64_20260501_sharded" / "text_embeddings.parquet"
TEXT_DEFAULT_15B = ROOT / "runs" / "text_embeddings" / "text_embeddings.parquet"
EDGAR_DEFAULT = ROOT / "runs" / "edgar_feature_store_full_daily_wide_20260203_225620" / "edgar_features"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-glob", default="runs/benchmarks/r14fcast_cqrrow*_main_h30_*/predictions.parquet")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-prediction-rows", type=int, default=100000)
    parser.add_argument("--text-embeddings", default=str(TEXT_DEFAULT_7B if TEXT_DEFAULT_7B.exists() else TEXT_DEFAULT_15B))
    parser.add_argument("--edgar-dir", default=str(EDGAR_DEFAULT))
    parser.add_argument("--stem", default="r14_source_event_state_telemetry_audit")
    return parser.parse_args()


def _date_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")


def _summary(values: list[float]) -> dict[str, float | None]:
    vals = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _prediction_rows(args: argparse.Namespace) -> pd.DataFrame:
    columns = ["model", "task", "target", "horizon", "ablation", "entity_id", "crawled_date_day", "cik", "offer_id", "source_row_index"]
    paths = sorted((Path(path) for path in glob.glob(str(ROOT / args.prediction_glob))), key=lambda path: os.path.getmtime(path), reverse=True)
    chunks = []
    for path in paths:
        if sum(len(chunk) for chunk in chunks) >= args.max_prediction_rows:
            break
        try:
            parquet = pq.ParquetFile(path)
        except Exception:
            continue
        available = set(parquet.schema.names)
        read_cols = [column for column in columns if column in available]
        if not {"entity_id", "crawled_date_day", "source_row_index"}.issubset(read_cols):
            continue
        for batch in parquet.iter_batches(batch_size=50000, columns=read_cols):
            frame = batch.to_pandas()
            frame = frame.loc[frame["target"].astype(str).eq(args.target)].copy()
            frame = frame.loc[pd.to_numeric(frame["horizon"], errors="coerce").eq(int(args.horizon))].copy()
            if frame.empty:
                continue
            frame["_prediction_path"] = str(path.relative_to(ROOT))
            chunks.append(frame)
            if sum(len(chunk) for chunk in chunks) >= args.max_prediction_rows:
                break
    if not chunks:
        return pd.DataFrame(columns=columns + ["date_key", "row_key"])
    data = pd.concat(chunks, ignore_index=True).head(args.max_prediction_rows).copy()
    data["date_key"] = _date_key(data["crawled_date_day"])
    data["entity_key"] = data["entity_id"].astype(str)
    data["cik_key"] = data["cik"].astype(str).where(data["cik"].notna(), "")
    data["row_key"] = data[["task", "target", "horizon", "entity_key", "date_key", "source_row_index"]].astype(str).agg("|".join, axis=1)
    return data.drop_duplicates("row_key", keep="last").reset_index(drop=True)


def _scan_text(text_path: Path, text_keys: set[tuple[str, str]]) -> dict[str, Any]:
    if not text_path.exists() or not text_keys:
        return {"path": str(text_path), "exists": text_path.exists(), "matched_keys": 0, "energy": _summary([])}
    parquet = pq.ParquetFile(text_path)
    names = parquet.schema.names
    emb_cols = [column for column in names if column.startswith("text_emb_")]
    read_cols = [column for column in ["entity_id", "crawled_date_day", *emb_cols] if column in names]
    matched: set[tuple[str, str]] = set()
    energy: list[float] = []
    for batch in parquet.iter_batches(batch_size=200000, columns=read_cols):
        frame = batch.to_pandas()
        frame["date_key"] = _date_key(frame["crawled_date_day"])
        frame["entity_key"] = frame["entity_id"].astype(str)
        keys = list(zip(frame["entity_key"], frame["date_key"]))
        mask = np.array([key in text_keys for key in keys], dtype=bool)
        if not mask.any():
            continue
        sub = frame.loc[mask].copy()
        matched.update(zip(sub["entity_key"], sub["date_key"]))
        if emb_cols:
            vals = sub[emb_cols].to_numpy(dtype=np.float32, copy=False)
            row_energy = np.linalg.norm(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
            energy.extend(float(value) for value in row_energy)
        if len(matched) >= len(text_keys):
            break
    return {
        "path": str(text_path),
        "exists": True,
        "embedding_columns": len(emb_cols),
        "matched_keys": int(len(matched)),
        "energy": _summary(energy),
        "zero_energy_matches": int(sum(1 for value in energy if abs(value) <= 1e-12)),
    }


def _edgar_files(edgar_dir: Path) -> list[Path]:
    if edgar_dir.is_file():
        return [edgar_dir]
    return sorted(edgar_dir.rglob("*.parquet"))


def _scan_edgar(edgar_dir: Path, edgar_keys: set[tuple[str, str]]) -> dict[str, Any]:
    files = _edgar_files(edgar_dir)
    if not files or not edgar_keys:
        return {"path": str(edgar_dir), "exists": bool(files), "matched_keys": 0, "energy": _summary([])}
    matched: set[tuple[str, str]] = set()
    energy: list[float] = []
    first_schema = pq.ParquetFile(files[0]).schema.names
    numeric_cols = [
        column for column in first_schema
        if column not in {"cik", "crawled_date_day", "cutoff_ts", "edgar_filed_date"}
        and not column.endswith("_is_missing")
        and column not in {"edgar_has_filing", "edgar_valid"}
    ][:64]
    read_cols = [column for column in ["cik", "crawled_date_day", *numeric_cols] if column in first_schema]
    for file in files:
        try:
            parquet = pq.ParquetFile(file)
        except Exception:
            continue
        available = set(parquet.schema.names)
        cols = [column for column in read_cols if column in available]
        if "cik" not in cols or "crawled_date_day" not in cols:
            continue
        for batch in parquet.iter_batches(batch_size=100000, columns=cols):
            frame = batch.to_pandas()
            frame["date_key"] = _date_key(frame["crawled_date_day"])
            frame["cik_key"] = frame["cik"].astype(str)
            keys = list(zip(frame["cik_key"], frame["date_key"]))
            mask = np.array([key in edgar_keys for key in keys], dtype=bool)
            if not mask.any():
                continue
            sub = frame.loc[mask].copy()
            matched.update(zip(sub["cik_key"], sub["date_key"]))
            available_numeric = [column for column in numeric_cols if column in sub.columns]
            if available_numeric:
                vals = sub[available_numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                row_energy = np.linalg.norm(vals, axis=1)
                energy.extend(float(value) for value in row_energy)
        if len(matched) >= len(edgar_keys):
            break
    return {
        "path": str(edgar_dir),
        "exists": True,
        "n_files": len(files),
        "numeric_columns_used": numeric_cols,
        "matched_keys": int(len(matched)),
        "energy": _summary(energy),
        "zero_energy_matches": int(sum(1 for value in energy if abs(value) <= 1e-12)),
    }


def main() -> int:
    args = _parse_args()
    predictions = _prediction_rows(args)
    text_keys = set(zip(predictions.get("entity_key", pd.Series(dtype=str)), predictions.get("date_key", pd.Series(dtype=str))))
    edgar_candidates = predictions.loc[predictions.get("cik_key", pd.Series(dtype=str)).astype(str).ne("")].copy() if not predictions.empty else predictions.copy()
    edgar_keys = set(zip(edgar_candidates.get("cik_key", pd.Series(dtype=str)), edgar_candidates.get("date_key", pd.Series(dtype=str))))
    text = _scan_text(Path(args.text_embeddings), text_keys)
    edgar = _scan_edgar(Path(args.edgar_dir), edgar_keys)

    n_rows = int(len(predictions))
    text_rate = text["matched_keys"] / max(len(text_keys), 1)
    edgar_rate = edgar["matched_keys"] / max(len(edgar_keys), 1)
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if n_rows > 0 and text_rate >= 0.95 and (not edgar_keys or edgar_rate >= 0.50) else "partial",
        "scope": "strict prediction row-key source event-state telemetry",
        "config": vars(args),
        "prediction_sample": {
            "n_rows": n_rows,
            "n_text_candidate_keys": int(len(text_keys)),
            "n_edgar_candidate_keys_with_cik": int(len(edgar_keys)),
            "ablations": predictions.get("ablation", pd.Series(dtype=str)).value_counts().to_dict(),
            "models": predictions.get("model", pd.Series(dtype=str)).value_counts().head(20).to_dict(),
        },
        "text_telemetry": {
            **text,
            "match_rate_on_prediction_keys": float(text_rate),
        },
        "edgar_telemetry": {
            **edgar,
            "match_rate_on_prediction_cik_day_keys": float(edgar_rate),
        },
        "interpretation": [
            "High key match plus nonzero energy means source can be represented as event-state telemetry.",
            "It does not prove the model reads the source correctly; source_path_activation and downstream pair audits handle that.",
            "Low EDGAR match is expected when many evaluated rows lack CIK; lane read policy must allow no-read fallback.",
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"{args.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text("# R14 Source Event-State Telemetry Audit\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "prediction_rows": n_rows,
        "text_match_rate": text_rate,
        "edgar_match_rate": edgar_rate,
        "text_energy": text.get("energy"),
        "edgar_energy": edgar.get("energy"),
        "out_json": str(out_json),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())