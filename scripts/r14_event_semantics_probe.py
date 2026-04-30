#!/usr/bin/env python3
"""Lightweight event-semantics probe over frozen offering text."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer


EVENT_KEYWORDS = {
    "financing": ["funding", "financing", "raised", "round", "series a", "series b", "investor"],
    "product_launch": ["launch", "released", "product", "platform", "commercial", "customers"],
    "regulatory": ["fda", "approval", "clearance", "regulatory", "clinical", "trial"],
    "risk_negative": ["risk", "litigation", "lawsuit", "default", "bankruptcy", "layoff", "going concern"],
    "growth_positive": ["growth", "revenue", "profit", "contract", "partnership", "expansion"],
}


def _stream_text(path: Path, columns: list[str], row_limit: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    chunks = []
    seen = 0
    for batch in parquet.iter_batches(columns=columns, batch_size=4096):
        frame = batch.to_pandas()
        remaining = row_limit - seen
        if remaining <= 0:
            break
        if len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        chunks.append(frame)
        seen += len(frame)
        if seen >= row_limit:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe whether raw text contains event-like categories with target association")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--sample-rows", type=int, default=100000)
    args = parser.parse_args()

    pointer = FreezePointer.load(Path(args.pointer))
    metadata_path = ROOT / "runs/text_embeddings/embedding_metadata.json"
    metadata = json.load(open(metadata_path))
    text_cols = [col for col in metadata.get("text_columns_used", [])]
    text_path = pointer.offers_text_dir / "offers_text.parquet"

    dataset = Block3Dataset.from_pointer(Path(args.pointer))
    core = dataset.get_offers_core_daily(columns=["entity_id", "crawled_date_day", "funding_raised_usd", "is_funded"]).copy()
    core["crawled_date_day"] = pd.to_datetime(core["crawled_date_day"], errors="coerce").dt.tz_localize(None).dt.normalize()

    available_cols = ["entity_id", "snapshot_ts", *text_cols]
    text = _stream_text(text_path, available_cols, args.sample_rows)
    text["crawled_date_day"] = pd.to_datetime(text["snapshot_ts"], errors="coerce").dt.tz_localize(None).dt.normalize()
    text_blob = text[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    for event, keywords in EVENT_KEYWORDS.items():
        pattern = "|".join([kw.replace(" ", r"\s+") for kw in keywords])
        text[event] = text_blob.str.contains(pattern, regex=True, na=False)

    joined = text[["entity_id", "crawled_date_day", *EVENT_KEYWORDS.keys()]].merge(
        core,
        on=["entity_id", "crawled_date_day"],
        how="left",
    )
    rows = []
    for event in EVENT_KEYWORDS:
        mask = joined[event].fillna(False).astype(bool)
        hit = joined.loc[mask]
        miss = joined.loc[~mask]
        rows.append({
            "event_family": event,
            "keywords": EVENT_KEYWORDS[event],
            "hit_rows": int(mask.sum()),
            "hit_rate": float(mask.mean()) if len(mask) else None,
            "funding_mean_hit": None if hit["funding_raised_usd"].dropna().empty else float(hit["funding_raised_usd"].mean()),
            "funding_mean_non_hit": None if miss["funding_raised_usd"].dropna().empty else float(miss["funding_raised_usd"].mean()),
            "is_funded_rate_hit": None if hit["is_funded"].dropna().empty else float(hit["is_funded"].mean()),
            "is_funded_rate_non_hit": None if miss["is_funded"].dropna().empty else float(miss["is_funded"].mean()),
        })

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "partial",
        "scope": "keyword_probe_not_embedding_semantics_proof",
        "sample_rows": int(len(text)),
        "joined_rows": int(len(joined)),
        "event_rows": rows,
        "required_for_pass": [
            "embedding-neighbor audit for event families",
            "controlled text perturbation/counterfactual",
            "downstream improvement on held-out temporal benchmark",
        ],
    }
    out_json = ROOT / "runs/audits" / f"r14_event_semantics_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    out_md.write_text("# R14 Event Semantics Probe\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {out_json}")
    print(f"OK: {out_md}")
    print(json.dumps({"status": report["status"], "sample_rows": report["sample_rows"], "event_rows": rows}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())