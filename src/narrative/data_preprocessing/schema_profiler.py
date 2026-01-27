from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

TIME_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "crawled",
    "snapshot",
    "filed",
    "filing",
    "open",
    "close",
)

KEY_CANDIDATES = [
    ["platform_name", "offer_id"],
    ["offer_id"],
    ["cik"],
    ["accession_number"],
    ["file_number"],
    ["entity_id"],
    ["link"],
]


@dataclass
class SchemaProfile:
    name: str
    path: str
    n_columns: int
    columns: List[Dict[str, str]]
    time_columns: List[str]
    key_candidates: List[List[str]]
    sample_rows: int
    null_rate: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "time_columns": self.time_columns,
            "key_candidates": self.key_candidates,
            "sample_rows": self.sample_rows,
            "null_rate": self.null_rate,
        }


def _dataset_for_path(path: Path) -> ds.Dataset:
    return ds.dataset(
        str(path),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )


def _infer_time_columns(
    schema: pa.Schema,
    partition_cols: Iterable[str],
) -> List[str]:
    time_cols: List[str] = []
    for field in schema:
        name = field.name
        lower = name.lower()
        if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type) or pa.types.is_time(field.type):
            time_cols.append(name)
        elif any(hint in lower for hint in TIME_NAME_HINTS):
            time_cols.append(name)

    for col in partition_cols:
        if col not in time_cols and any(hint in col.lower() for hint in TIME_NAME_HINTS):
            time_cols.append(col)

    return sorted(list(dict.fromkeys(time_cols)))


def _infer_key_candidates(column_names: Iterable[str]) -> List[List[str]]:
    names = set(column_names)
    found = []
    for cand in KEY_CANDIDATES:
        if all(c in names for c in cand):
            found.append(cand)
    return found


def _scan_null_rate(
    dataset: ds.Dataset,
    columns: Sequence[str],
    *,
    batch_size: int,
    max_batches: int,
) -> tuple[int, Dict[str, float]]:
    totals = {c: 0 for c in columns}
    nulls = {c: 0 for c in columns}
    rows = 0
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size, use_threads=True)

    for i, batch in enumerate(scanner.to_batches()):
        df = batch.to_pandas()
        if df.empty:
            continue
        rows += len(df)
        for col in columns:
            if col in df.columns:
                nulls[col] += int(df[col].isna().sum())
                totals[col] += len(df)
        if i + 1 >= max_batches:
            break

    null_rate = {
        col: (nulls[col] / max(1, totals[col])) if totals[col] else 0.0 for col in columns
    }
    return rows, null_rate


def profile_parquet_dataset(
    path: Path,
    *,
    name: Optional[str] = None,
    batch_size: int = 50_000,
    max_batches: int = 5,
    max_profile_cols: int = 50,
) -> SchemaProfile:
    dataset = _dataset_for_path(path)
    schema = dataset.schema
    partition_cols = []
    if dataset.partitioning is not None:
        partition_cols = list(dataset.partitioning.schema.names)

    columns = [{"name": f.name, "dtype": str(f.type)} for f in schema]
    time_cols = _infer_time_columns(schema, partition_cols)
    key_candidates = _infer_key_candidates(schema.names)

    profile_cols = list(dict.fromkeys(time_cols + [c for k in key_candidates for c in k]))
    if len(profile_cols) < max_profile_cols:
        for col in schema.names:
            if col not in profile_cols:
                profile_cols.append(col)
            if len(profile_cols) >= max_profile_cols:
                break

    sample_rows, null_rate = _scan_null_rate(
        dataset, profile_cols, batch_size=batch_size, max_batches=max_batches
    )

    return SchemaProfile(
        name=name or path.name,
        path=str(path),
        n_columns=len(schema.names),
        columns=columns,
        time_columns=time_cols,
        key_candidates=key_candidates,
        sample_rows=sample_rows,
        null_rate=null_rate,
    )


def profiles_to_frame(profiles: Sequence[SchemaProfile]) -> pd.DataFrame:
    rows = []
    for prof in profiles:
        row = {
            "name": prof.name,
            "path": prof.path,
            "n_columns": prof.n_columns,
            "time_columns": ",".join(prof.time_columns),
            "key_candidates": ";".join(["|".join(k) for k in prof.key_candidates]),
            "sample_rows": prof.sample_rows,
        }
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "SchemaProfile",
    "profile_parquet_dataset",
    "profiles_to_frame",
]
