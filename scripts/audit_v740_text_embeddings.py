#!/usr/bin/env python3
"""Audit V740-relevant text embedding lineage and usage.

This script is intended to answer three concrete questions:

1. How complete is text embedding coverage on the frozen core panel?
2. What exactly is the active embedding lineage (base model, PCA, fields)?
3. On a lightweight sampled slice, do the current PCA embedding dimensions show
   strong direct association with the Block 3 targets?

The result is an audit artifact for root-cause analysis, not a benchmark run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer


TARGET_COLS = ["funding_raised_usd", "investors_count", "is_funded"]


def _rate(num: int | float, den: int | float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _normalize_scalar(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_scalar(v) for v in value]
    return value


def _load_metadata(embedding_root: Path) -> Dict[str, Any]:
    meta_path = embedding_root / "embedding_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _top_abs_correlations(df: pd.DataFrame, target: str, emb_cols: List[str], method: str, top_k: int) -> List[Dict[str, Any]]:
    valid = df[target].notna()
    if int(valid.sum()) < 10:
        return []
    work = df.loc[valid, emb_cols + [target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    results: List[Dict[str, Any]] = []
    for col in emb_cols:
        corr = work[[col, target]].corr(method=method).iloc[0, 1]
        if pd.notna(corr):
            results.append({"dimension": col, "correlation": float(corr), "abs_correlation": abs(float(corr))})
    results.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return results[:top_k]


def _sample_field_stats(text_df: pd.DataFrame, text_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in text_cols:
        if col not in text_df.columns:
            continue
        values = text_df[col].fillna("").astype(str).str.strip()
        non_empty = values != ""
        lengths = values[non_empty].str.len()
        out[col] = {
            "non_empty_rate": _rate(int(non_empty.sum()), len(values)),
            "avg_chars_non_empty": None if lengths.empty else float(lengths.mean()),
            "median_chars_non_empty": None if lengths.empty else float(lengths.median()),
        }
    return out


def _stream_sample_parquet(path: Path, columns: List[str], row_limit: int, batch_size: int = 4096) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    chunks: List[pd.DataFrame] = []
    seen = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        remaining = row_limit - seen
        if remaining <= 0:
            break
        if len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        chunks.append(frame)
        seen += len(frame)
        if seen >= row_limit:
            break
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def build_audit(pointer_path: Path, sample_rows: int, raw_text_sample_rows: int, top_k: int) -> Dict[str, Any]:
    dataset = Block3Dataset.from_pointer(pointer_path)
    pointer = FreezePointer.load(pointer_path)

    core_cols = ["entity_id", "crawled_date_day", *TARGET_COLS]
    core = dataset.get_offers_core_daily(columns=core_cols).copy()
    core = core.dropna(subset=["entity_id", "crawled_date_day"]).reset_index(drop=True)

    embedding_root = REPO_ROOT / "runs" / "text_embeddings"
    metadata = _load_metadata(embedding_root)
    emb_path = embedding_root / "text_embeddings.parquet"
    pca_dim = int(metadata.get("pca_dim", 0) or 0)
    emb_cols = [f"text_emb_{i}" for i in range(pca_dim)]

    emb_sample_cols = ["entity_id", "crawled_date_day", *emb_cols]
    emb_sample = _stream_sample_parquet(emb_path, emb_sample_cols, row_limit=sample_rows)
    sample_coverage = emb_sample[["entity_id", "crawled_date_day"]].drop_duplicates().merge(
        core[["entity_id", "crawled_date_day", *TARGET_COLS]],
        on=["entity_id", "crawled_date_day"],
        how="left",
    )
    sample_coverage = sample_coverage.merge(
        emb_sample[["entity_id", "crawled_date_day"]].drop_duplicates().assign(has_embedding=True),
        on=["entity_id", "crawled_date_day"],
        how="left",
    )
    sample_coverage["has_embedding"] = sample_coverage["has_embedding"].fillna(False).astype(bool)

    target_subset: Dict[str, Any] = {}
    for target in TARGET_COLS:
        mask = sample_coverage[target].notna()
        denom = int(mask.sum())
        target_subset[target] = {
            "eligible_rows": denom,
            "embedding_match_rate": _rate(int((sample_coverage["has_embedding"] & mask).sum()), denom),
        }

    joined = sample_coverage[sample_coverage["has_embedding"]].merge(
        emb_sample,
        on=["entity_id", "crawled_date_day"],
        how="inner",
    )

    corr_targets: Dict[str, Any] = {}
    for target in TARGET_COLS:
        corr_targets[target] = {
            "top_abs_pearson": _top_abs_correlations(joined, target, emb_cols, method="pearson", top_k=top_k),
            "top_abs_spearman": _top_abs_correlations(joined, target, emb_cols, method="spearman", top_k=top_k),
        }

    raw_text_stats: Dict[str, Any] = {}
    raw_text_error: str | None = None
    text_cols = list(metadata.get("text_columns_used", []))
    text_path = pointer.offers_text_dir / "offers_text.parquet"
    if raw_text_sample_rows > 0 and text_cols and text_path.exists():
        try:
            raw_text = _stream_sample_parquet(
                text_path,
                columns=["entity_id", *text_cols],
                row_limit=raw_text_sample_rows,
            )
            raw_text_stats = _sample_field_stats(raw_text, text_cols)
        except Exception as exc:  # pragma: no cover - defensive audit path
            raw_text_error = str(exc)

    result = {
        "pointer_path": str(pointer_path),
        "embedding_root": str(embedding_root),
        "embedding_lineage": {
            "model": metadata.get("model"),
            "raw_dim": int(metadata.get("raw_dim", 0) or 0),
            "pca_dim": pca_dim,
            "pca_explained_variance": float(metadata.get("pca_explained_variance", 0.0) or 0.0),
            "max_length": int(metadata.get("max_length", 0) or 0),
            "max_chars": int(metadata.get("max_chars", 0) or 0),
            "text_columns_used": text_cols,
        },
        "embedding_cardinality": {
            "n_total_rows": int(metadata.get("n_total_rows", 0) or 0),
            "n_unique_texts": int(metadata.get("n_unique_texts", 0) or 0),
            "n_entities": int(metadata.get("n_entities", 0) or 0),
            "unique_text_ratio": _rate(int(metadata.get("n_unique_texts", 0) or 0), int(metadata.get("n_total_rows", 0) or 0)),
        },
        "coverage": {
            "core_rows": int(len(core)),
            "core_entities": int(core["entity_id"].nunique()),
            "embedding_rows_from_metadata": int(metadata.get("n_total_rows", 0) or 0),
            "global_row_parity_vs_core": _rate(int(metadata.get("n_total_rows", 0) or 0), len(core)),
            "streamed_embedding_rows": int(len(emb_sample)),
            "sample_rows": int(len(sample_coverage)),
            "sample_rows_with_embedding": int(sample_coverage["has_embedding"].sum()),
            "sample_row_match_rate": _rate(int(sample_coverage["has_embedding"].sum()), len(sample_coverage)),
            "target_subsets": target_subset,
        },
        "sample": {
            "sampled_rows_after_join": int(len(joined)),
            "top_abs_correlations": corr_targets,
            "raw_text_field_stats": raw_text_stats,
            "raw_text_error": raw_text_error,
        },
    }
    return _normalize_scalar(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V740 text embedding lineage, coverage, and sampled signal strength")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--raw-text-sample-rows", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = build_audit(
        pointer_path=Path(args.pointer),
        sample_rows=args.sample_rows,
        raw_text_sample_rows=args.raw_text_sample_rows,
        top_k=args.top_k,
    )

    rendered = json.dumps(result, indent=2, sort_keys=False)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()