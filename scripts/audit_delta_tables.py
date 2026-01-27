#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _sha256_text(values: List[str]) -> str:
    joined = "\n".join(values).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _open_delta(path: Path):
    try:
        from deltalake import DeltaTable
    except Exception as exc:
        raise RuntimeError("deltalake is required for Delta snapshot probing") from exc
    return DeltaTable(str(path))


def _sample_rows(dataset, columns: List[str]) -> List[Dict[str, Any]]:
    scanner = dataset.scanner(columns=columns, use_threads=True)
    try:
        table = scanner.head(5)
    except Exception:
        table = scanner.to_table().slice(0, 5)
    rows = table.to_pylist()
    return [_json_safe(row) for row in rows]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _count_fragments(dataset) -> int:
    return sum(1 for _ in dataset.get_fragments())


def _stream_column_stats(dataset, column: str) -> Tuple[int, int, int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    total = 0
    non_null = 0
    parseable = 0
    min_ts = None
    max_ts = None
    for batch in dataset.scanner(columns=[column], use_threads=True).to_batches():
        series = batch.column(0).to_pandas()
        total += len(series)
        non_null += int(series.notna().sum())
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        parseable += int(parsed.notna().sum())
        if parsed.notna().any():
            cur_min = parsed.min()
            cur_max = parsed.max()
            min_ts = cur_min if min_ts is None else min(min_ts, cur_min)
            max_ts = cur_max if max_ts is None else max(max_ts, cur_max)
    return total, non_null, parseable, min_ts, max_ts


def _select_snapshot_time_col(dataset, candidates: List[str]) -> Dict[str, Any]:
    schema_cols = set(dataset.schema.names)
    metrics = {}
    best = None
    for col in candidates:
        if col not in schema_cols:
            continue
        total, non_null, parseable, min_ts, max_ts = _stream_column_stats(dataset, col)
        non_null_rate = float(non_null / total) if total else 0.0
        parseable_rate = float(parseable / total) if total else 0.0
        metrics[col] = {
            "total_rows": total,
            "non_null_rate": non_null_rate,
            "parseable_rate": parseable_rate,
            "min_ts": str(min_ts) if min_ts is not None else None,
            "max_ts": str(max_ts) if max_ts is not None else None,
        }
        if best is None:
            best = col
        else:
            cur = metrics[col]
            best_metrics = metrics[best]
            if (cur["non_null_rate"], cur["parseable_rate"]) > (
                best_metrics["non_null_rate"],
                best_metrics["parseable_rate"],
            ):
                best = col
    return {"selected": best, "candidates": metrics}


def _dataset_counts(dataset, entity_col: str = "entity_id") -> Dict[str, Any]:
    total = 0
    entities: set = set()
    if entity_col not in dataset.schema.names:
        for batch in dataset.scanner(columns=[], use_threads=True).to_batches():
            total += batch.num_rows
        return {"n_rows": total, "n_entities": None}
    for batch in dataset.scanner(columns=[entity_col], use_threads=True).to_batches():
        series = batch.column(0).to_pandas()
        total += len(series)
        entities.update(series.dropna().astype(str).unique().tolist())
    return {"n_rows": total, "n_entities": len(entities)}


def _probe_table(path: Path, label: str, sample_cols: List[str]) -> Dict[str, Any]:
    if not (path / "_delta_log").exists():
        raise SystemExit(f"FATAL: {label} is missing _delta_log: {path}")
    dt = _open_delta(path)
    dataset = dt.to_pyarrow_dataset()
    schema_cols = dataset.schema.names
    schema_hash = _sha256_text(schema_cols)
    sample_cols = [c for c in sample_cols if c in schema_cols]
    samples = _sample_rows(dataset, sample_cols if sample_cols else schema_cols[:10])
    counts = _dataset_counts(dataset)
    return {
        "path": str(path),
        "is_delta": True,
        "version": int(dt.version()),
        "schema": schema_cols,
        "schema_hash": schema_hash,
        "active_files": len(dt.files()),
        "fragment_count": _count_fragments(dataset),
        "counts": counts,
        "sample_rows": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Delta tables for offers/edgar.")
    parser.add_argument("--offers_path", required=True)
    parser.add_argument("--edgar_path", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument(
        "--snapshot_candidates",
        type=str,
        default="crawled_date_day,crawled_date,crawled_at,snapshot_time,snapshot_date",
    )
    args = parser.parse_args()

    offers_path = Path(args.offers_path)
    edgar_path = Path(args.edgar_path)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    offers_report = _probe_table(
        offers_path,
        "offers",
        ["entity_id", "offer_id", "platform_name", "cik", "crawled_date", "snapshot_time"],
    )
    edgar_report = _probe_table(
        edgar_path,
        "edgar",
        ["cik", "filed_date", "submission_offering_data"],
    )

    candidates = [c.strip() for c in args.snapshot_candidates.split(",") if c.strip()]
    offers_dataset = _open_delta(offers_path).to_pyarrow_dataset()
    snapshot_pick = _select_snapshot_time_col(offers_dataset, candidates)
    offers_report["snapshot_time"] = snapshot_pick

    report = {
        "offers": offers_report,
        "edgar": edgar_report,
        "snapshot_time_col": snapshot_pick["selected"],
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Delta coverage report\n\n")
        f.write(f"offers_path: `{offers_path}`\n\n")
        f.write(f"edgar_path: `{edgar_path}`\n\n")
        f.write("## Offers\n\n")
        f.write(f"- delta_version: {offers_report['version']}\n")
        f.write(f"- active_files: {offers_report['active_files']}\n")
        f.write(f"- n_rows: {offers_report['counts']['n_rows']}\n")
        f.write(f"- n_entities: {offers_report['counts']['n_entities']}\n")
        f.write(f"- schema_hash: {offers_report['schema_hash']}\n\n")
        f.write("### Snapshot time selection\n\n")
        f.write(f"- selected: {snapshot_pick['selected']}\n\n")
        f.write("## EDGAR\n\n")
        f.write(f"- delta_version: {edgar_report['version']}\n")
        f.write(f"- active_files: {edgar_report['active_files']}\n")
        f.write(f"- n_rows: {edgar_report['counts']['n_rows']}\n")
        f.write(f"- n_entities: {edgar_report['counts']['n_entities']}\n")
        f.write(f"- schema_hash: {edgar_report['schema_hash']}\n")

    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()
