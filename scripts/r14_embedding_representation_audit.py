#!/usr/bin/env python3
"""Audit whether a text embedding artifact carries event-semantic signal.

This is not a downstream benchmark and must not be used to claim an embedding is
"best". It is a representation-quality audit: row alignment, numeric quality,
PCA metadata, and lightweight linear probes for event-family keywords.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "audits"

TEXT_COLS = [
    "headline",
    "title",
    "description_text",
    "company_description",
    "product_description",
    "financial_condition",
    "financial_forecasts",
    "financial_risks",
    "offering_purpose",
    "use_of_funds_text",
    "reasons_to_invest_text",
]

EVENT_KEYWORDS = {
    "financing": ["funding", "financing", "raised", "round", "series a", "series b", "investor"],
    "product_launch": ["launch", "released", "product", "platform", "commercial", "customers"],
    "regulatory": ["fda", "approval", "clearance", "regulatory", "clinical", "trial"],
    "risk_negative": ["risk", "litigation", "lawsuit", "default", "bankruptcy", "layoff", "going concern"],
    "growth_positive": ["growth", "revenue", "profit", "contract", "partnership", "expansion"],
}


def _read_head_rows(path: Path, columns: list[str], n_rows: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    chunks = []
    seen = 0
    available = set(parquet.schema.names)
    cols = [col for col in columns if col in available]
    for batch in parquet.iter_batches(columns=cols, batch_size=8192):
        frame = batch.to_pandas()
        remaining = n_rows - seen
        if remaining <= 0:
            break
        if len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        chunks.append(frame)
        seen += len(frame)
        if seen >= n_rows:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def _read_position_rows(path: Path, columns: list[str], positions: np.ndarray) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    cols = [col for col in columns if col in available]
    targets = np.sort(np.asarray(positions, dtype=np.int64))
    chunks = []
    offset = 0
    for batch in parquet.iter_batches(columns=cols, batch_size=65536):
        batch_len = batch.num_rows
        lo = int(np.searchsorted(targets, offset, side="left"))
        hi = int(np.searchsorted(targets, offset + batch_len, side="left"))
        if hi > lo:
            local_idx = targets[lo:hi] - offset
            frame = batch.to_pandas()
            chunks.append(frame.iloc[local_idx].copy())
        offset += batch_len
        if hi >= len(targets):
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _linear_probe_auc(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int32)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives < 100 or negatives < 100:
        return {"status": "too_few_labels", "positives": positives, "negatives": negatives, "auc": None}

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    keep_n = min(len(pos_idx), len(neg_idx), 20000)
    keep = np.concatenate([
        rng.choice(pos_idx, size=keep_n, replace=False),
        rng.choice(neg_idx, size=keep_n, replace=False),
    ])
    rng.shuffle(keep)
    X_bal = X[keep]
    y_bal = y[keep]
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.30,
        random_state=seed,
        stratify=y_bal,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
    clf.fit(X_train, y_train)
    score = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, score))
    return {
        "status": "ok",
        "positives": positives,
        "negatives": negatives,
        "balanced_train_per_class": int(keep_n),
        "auc": auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit text embedding representation quality")
    parser.add_argument("--embedding-dir", default="runs/text_embeddings")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--sample-mode", choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embedding_dir = (ROOT / args.embedding_dir).resolve() if not Path(args.embedding_dir).is_absolute() else Path(args.embedding_dir)
    emb_path = embedding_dir / "text_embeddings.parquet"
    meta_path = embedding_dir / "embedding_metadata.json"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding parquet: {emb_path}")

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.load(open(meta_path))

    pointer = yaml.safe_load(open(ROOT / args.pointer))
    text_path = Path(pointer["offers_text"]["dir"]) / "offers_text.parquet"

    emb_schema = pq.read_schema(emb_path)
    emb_cols = [col for col in emb_schema.names if col.startswith("text_emb_")]
    emb_meta = pq.ParquetFile(emb_path).metadata

    total_rows = min(int(emb_meta.num_rows), int(pq.ParquetFile(text_path).metadata.num_rows))
    sample_rows = min(args.sample_rows, total_rows)
    if args.sample_mode == "random":
        rng = np.random.default_rng(args.seed)
        positions = np.sort(rng.choice(total_rows, size=sample_rows, replace=False))
        emb = _read_position_rows(emb_path, ["entity_id", "crawled_date_day", *emb_cols], positions)
        text = _read_position_rows(text_path, ["entity_id", "snapshot_ts", *TEXT_COLS], positions)
    else:
        emb = _read_head_rows(emb_path, ["entity_id", "crawled_date_day", *emb_cols], sample_rows)
        text = _read_head_rows(text_path, ["entity_id", "snapshot_ts", *TEXT_COLS], len(emb))
    text["crawled_date_day"] = pd.to_datetime(text["snapshot_ts"], errors="coerce").dt.strftime("%Y-%m-%d")
    emb["crawled_date_day"] = pd.to_datetime(emb["crawled_date_day"], errors="coerce").dt.strftime("%Y-%m-%d")

    aligned = len(emb) == len(text)
    if aligned and len(emb):
        entity_equal = emb["entity_id"].astype(str).to_numpy() == text["entity_id"].astype(str).to_numpy()
        date_equal = emb["crawled_date_day"].astype(str).to_numpy() == text["crawled_date_day"].astype(str).to_numpy()
        row_alignment_rate = float(np.mean(entity_equal & date_equal))
    else:
        row_alignment_rate = 0.0

    X = emb[emb_cols].to_numpy(dtype=np.float32, copy=False)
    finite_mask = np.isfinite(X)
    row_l2 = np.linalg.norm(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    col_std = np.nanstd(X, axis=0)

    text_blob = text[[col for col in TEXT_COLS if col in text.columns]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    probes: dict[str, Any] = {}
    for event, keywords in EVENT_KEYWORDS.items():
        pattern = "|".join([kw.replace(" ", r"\s+") for kw in keywords])
        y = text_blob.str.contains(pattern, regex=True, na=False).to_numpy(dtype=np.int32)
        probes[event] = _linear_probe_auc(X, y, args.seed)
        probes[event]["hit_rate"] = float(np.mean(y)) if len(y) else None
        probes[event]["keywords"] = keywords

    ok_aucs = [probe["auc"] for probe in probes.values() if probe.get("auc") is not None]
    quality_pass = (
        emb_meta.num_rows > 0
        and len(emb_cols) > 0
        and float(np.mean(finite_mask)) >= 0.999999
        and row_alignment_rate >= 0.999
        and float(np.nanmin(col_std)) > 0.0
    )
    semantic_partial = len([auc for auc in ok_aucs if auc >= 0.65]) >= 3
    status = "partial" if quality_pass and semantic_partial else ("not_passed" if not quality_pass else "weak_partial")

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "embedding_dir": str(embedding_dir),
        "embedding_parquet": str(emb_path),
        "metadata": metadata,
        "parquet_rows": int(emb_meta.num_rows),
        "embedding_columns": len(emb_cols),
        "sample_rows": int(len(emb)),
        "sample_mode": args.sample_mode,
        "row_alignment_rate_first_sample": row_alignment_rate,
        "numeric_quality": {
            "finite_value_rate": float(np.mean(finite_mask)) if finite_mask.size else None,
            "row_l2": _summary(row_l2),
            "column_std": _summary(col_std),
            "zero_l2_rows": int(np.sum(row_l2 <= 1e-12)),
        },
        "linear_event_probes": probes,
        "interpretation_lock": [
            "This proves only representation-level event separability, not causal downstream usefulness.",
            "Embedding optimality still requires at least two artifacts and matched downstream benchmarks.",
            "If source activation remains zero, do not claim source-scaling mechanism even with good probes.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"r14_embedding_representation_audit_{suffix}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    out_md.write_text("# R14 Embedding Representation Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {out_json}")
    print(f"OK: {out_md}")
    print(json.dumps({
        "status": status,
        "embedding_columns": len(emb_cols),
        "row_alignment_rate": row_alignment_rate,
        "event_probe_aucs": {k: v.get("auc") for k, v in probes.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
