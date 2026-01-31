#!/usr/bin/env python
"""
Build offers_text parquet from raw offers delta (narrative columns for NBI/NCI).
Joins by entity_id + snapshot_ts. Converts array fields to joined strings.
Output: offers_text.parquet + MANIFEST.json (local, untracked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent


def _join_list(val: Any) -> str:
    """Convert list/array to newline-joined string."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if isinstance(val, (list, tuple)):
        return "\n".join(str(x) for x in val if x is not None and str(x).strip())
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            return "\n".join(str(x) for x in val if x is not None and str(x).strip())
        except (TypeError, ValueError):
            return str(val).strip() if val is not None else ""
    s = str(val).strip()
    return s if s else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offers_text from raw offers delta.")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit_rows", type=int, default=None, help="Limit rows read (for testing); omit for full build")
    args = parser.parse_args()

    try:
        from deltalake import DeltaTable
    except ImportError:
        raise RuntimeError("deltalake required; pip install deltalake")

    dt = DeltaTable(str(args.raw_offers_delta))
    delta_version = dt.version()

    scalar_cols = [
        "platform_name",
        "offer_id",
        "crawled_date_day",
        "headline",
        "title",
        "description_text",
        "company_description",
        "product_description",
        "financial_condition",
        "financial_forecasts",
        "financial_risks",
        "offering_purpose",
        "term_sheet",
        "front_video_transcript",
    ]
    array_cols = ["use_of_funds", "reasons_to_invest", "updates", "questions"]
    schema_names = [f.name for f in dt.schema().fields]
    read_scalar = [c for c in scalar_cols if c in schema_names]
    read_arrays = [c for c in array_cols if c in schema_names]
    dedup_col = "processed_datetime" if "processed_datetime" in schema_names else None
    if dedup_col:
        read_scalar.append(dedup_col)
    read_cols = read_scalar + read_arrays

    dset = dt.to_pyarrow_dataset()
    limit_rows = args.limit_rows
    frames = []
    seen = 0
    for batch in dset.scanner(columns=read_cols, batch_size=200_000).to_batches():
        frames.append(batch.to_pandas())
        seen += len(frames[-1])
        if limit_rows and seen >= limit_rows:
            break
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if limit_rows and len(df) > limit_rows:
        df = df.head(limit_rows)

    if df.empty:
        df = pd.DataFrame(columns=["entity_id", "platform_name", "offer_id", "snapshot_ts"] + scalar_cols + [f"{c}_text" for c in array_cols])
    else:
        df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
        df["snapshot_ts"] = pd.to_datetime(df["crawled_date_day"], errors="coerce", utc=True)
        for c in array_cols:
            if c in df.columns:
                df[f"{c}_text"] = df[c].apply(_join_list)
            else:
                df[f"{c}_text"] = ""
        text_cols = ["headline", "title", "description_text", "company_description", "product_description",
                     "financial_condition", "financial_forecasts", "financial_risks", "offering_purpose",
                     "term_sheet", "front_video_transcript"] + [f"{c}_text" for c in array_cols]
        out_cols = ["entity_id", "platform_name", "offer_id", "snapshot_ts"] + [c for c in text_cols if c in df.columns]
        df = df[[c for c in out_cols if c in df.columns]]
        df = df.dropna(subset=["entity_id", "snapshot_ts"])
        if dedup_col and dedup_col in df.columns:
            df = df.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
            df = df.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
        else:
            df = df.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "offers_text.parquet"
    df.to_parquet(out_path, index=False)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "raw_offers_delta_version": delta_version,
        "script_sha256": script_sha,
        "output_path": str(out_path),
        "row_count": len(df),
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
