from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - tested via functional usage
    import pyarrow as pa
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover
    pa = None
    ds = None

TimeRange = Optional[Tuple[Optional[datetime], Optional[datetime]]]

DEFAULT_OFFERS_PATH = Path("data/raw/offers")
DEFAULT_EDGAR_PATH = Path("data/raw/edgar/accessions")

OFFERS_ENV = "NARRATIVE_OFFERS_PARQUET_PATH"
EDGAR_ENV = "NARRATIVE_EDGAR_PARQUET_PATH"


def _require_pyarrow() -> None:
    if ds is None or pa is None:  # pragma: no cover
        raise ImportError("pyarrow is required for parquet scanning (pip install pyarrow)")


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def _resolve_base_dir(base_dir: Optional[Path], env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        base_dir = Path(raw)
    if base_dir is None:
        base_dir = default
    if base_dir.is_absolute():
        return base_dir
    repo_root = _find_repo_root(Path.cwd())
    return (repo_root / base_dir).resolve()


def _dataset_for_path(path: Path) -> "ds.Dataset":
    _require_pyarrow()
    return ds.dataset(
        str(path),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )


def _normalize_time_range(time_range: TimeRange) -> TimeRange:
    if time_range is None:
        return None
    start, end = time_range
    start_ts = pd.to_datetime(start, errors="coerce", utc=True) if start is not None else None
    end_ts = pd.to_datetime(end, errors="coerce", utc=True) if end is not None else None
    if start_ts is None and end_ts is None:
        return None
    return start_ts, end_ts


def _pick_time_column(names: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    name_set = set(names)
    for c in candidates:
        if c in name_set:
            return c
    return None


def _field_type(dataset: "ds.Dataset", col: str) -> Optional["pa.DataType"]:
    try:
        return dataset.schema.field(col).type
    except Exception:
        return None


def _cast_ids(values: Sequence[str], field_type: Optional["pa.DataType"]) -> Sequence[object]:
    if field_type is None:
        return [str(v) for v in values]
    if pa.types.is_integer(field_type):
        out = []
        for v in values:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out
    if pa.types.is_floating(field_type):
        out = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    return [str(v) for v in values]


def _build_offer_filter(
    dataset: "ds.Dataset",
    offer_ids: Sequence[object],
) -> Optional["ds.Expression"]:
    if not offer_ids:
        return None

    if isinstance(offer_ids[0], (tuple, list)) and len(offer_ids[0]) == 2:
        if "platform_name" not in dataset.schema.names or "offer_id" not in dataset.schema.names:
            # fallback: treat as offer_id only
            offer_ids = [x[1] for x in offer_ids]  # type: ignore[index]
        else:
            expr = None
            offer_type = _field_type(dataset, "offer_id")
            for platform, oid in offer_ids:  # type: ignore[misc]
                oid_cast = _cast_ids([oid], offer_type)
                if not oid_cast:
                    continue
                pair_expr = (ds.field("platform_name") == str(platform)) & (
                    ds.field("offer_id") == oid_cast[0]
                )
                expr = pair_expr if expr is None else (expr | pair_expr)
            return expr

    if "offer_id" not in dataset.schema.names:
        raise KeyError("offers parquet missing offer_id column")

    offer_type = _field_type(dataset, "offer_id")
    casted = _cast_ids([str(v) for v in offer_ids], offer_type)
    if not casted:
        return None
    return ds.field("offer_id").isin(casted)


def _build_id_filter(dataset: "ds.Dataset", id_col: str, ids: Sequence[object]) -> Optional["ds.Expression"]:
    if not ids:
        return None
    if id_col not in dataset.schema.names:
        raise KeyError(f"parquet missing id column: {id_col}")
    id_type = _field_type(dataset, id_col)
    casted = _cast_ids([str(v) for v in ids], id_type)
    if not casted:
        return None
    return ds.field(id_col).isin(casted)


def _build_time_filter(
    dataset: "ds.Dataset",
    time_col: Optional[str],
    time_range: TimeRange,
) -> Optional["ds.Expression"]:
    tr = _normalize_time_range(time_range)
    if time_col is None or tr is None:
        return None
    start_ts, end_ts = tr
    field_type = _field_type(dataset, time_col)
    is_string = field_type is None or pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
    expr = None
    if start_ts is not None:
        start_val = start_ts.strftime("%Y-%m-%d") if is_string else start_ts.to_pydatetime()
        expr = ds.field(time_col) >= start_val
    if end_ts is not None:
        end_val = end_ts.strftime("%Y-%m-%d") if is_string else end_ts.to_pydatetime()
        end_expr = ds.field(time_col) <= end_val
        expr = end_expr if expr is None else (expr & end_expr)
    return expr


def _apply_filters(
    dataset: "ds.Dataset",
    filters: Sequence[Optional["ds.Expression"]],
) -> Optional["ds.Expression"]:
    expr = None
    for f in filters:
        if f is None:
            continue
        expr = f if expr is None else (expr & f)
    return expr


def scan_offers_static(
    offer_ids: Sequence[object],
    *,
    base_dir: Optional[Path] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return latest snapshot row per offer_id (or platform_name, offer_id) using parquet.
    """
    df = scan_snapshots(offer_ids, time_range=None, base_dir=base_dir, columns=columns)
    if df.empty:
        return df

    keys = ["offer_id"]
    if "platform_name" in df.columns:
        keys = ["platform_name", "offer_id"]

    time_col = _pick_time_column(
        df.columns,
        ["crawled_date", "snapshot_date", "crawled_date_day", "datetime_open_offering"],
    )
    if time_col is None:
        # no time col -> drop duplicates deterministically
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.sort_values(keys + [time_col], kind="mergesort")
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def scan_snapshots(
    offer_ids: Sequence[object],
    time_range: TimeRange = None,
    *,
    base_dir: Optional[Path] = None,
    columns: Optional[Sequence[str]] = None,
    allow_all: bool = False,
    limit_rows: Optional[int] = None,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """
    Scan offers snapshots parquet with predicate pushdown for offer_ids + time_range.
    """
    if not offer_ids and not allow_all:
        return pd.DataFrame()

    base = _resolve_base_dir(base_dir, OFFERS_ENV, DEFAULT_OFFERS_PATH)
    dataset = _dataset_for_path(base)

    time_col = _pick_time_column(
        dataset.schema.names, ["crawled_date", "snapshot_date", "crawled_date_day"]
    )
    offer_filter = _build_offer_filter(dataset, list(offer_ids))
    time_filter = _build_time_filter(dataset, time_col, time_range)
    expr = _apply_filters(dataset, [offer_filter, time_filter])

    cols = None
    if columns is not None:
        cols = [c for c in columns if c in dataset.schema.names]

    if limit_rows is None:
        table = dataset.to_table(columns=cols, filter=expr)
        return table.to_pandas()

    frames = []
    seen = 0
    scanner = dataset.scanner(
        columns=cols, filter=expr, batch_size=batch_size, use_threads=True
    )
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        remaining = limit_rows - seen
        if remaining <= 0:
            break
        if len(df) > remaining:
            df = df.head(remaining)
        frames.append(df)
        seen += len(df)
        if seen >= limit_rows:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scan_edgar(
    entity_ids: Sequence[object],
    time_range: TimeRange = None,
    *,
    base_dir: Optional[Path] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Scan EDGAR accessions parquet with predicate pushdown for entity_ids + time_range.
    """
    if not entity_ids:
        return pd.DataFrame()

    base = _resolve_base_dir(base_dir, EDGAR_ENV, DEFAULT_EDGAR_PATH)
    dataset = _dataset_for_path(base)

    id_col = "cik" if "cik" in dataset.schema.names else ("entity_id" if "entity_id" in dataset.schema.names else None)
    if id_col is None:
        raise KeyError("EDGAR parquet missing cik/entity_id columns")

    time_col = _pick_time_column(dataset.schema.names, ["filed_date", "filing_date"])

    id_filter = _build_id_filter(dataset, id_col, list(entity_ids))
    time_filter = _build_time_filter(dataset, time_col, time_range)
    expr = _apply_filters(dataset, [id_filter, time_filter])

    cols = None
    if columns is not None:
        cols = [c for c in columns if c in dataset.schema.names]

    if limit_rows is None:
        table = dataset.to_table(columns=cols, filter=expr)
        return table.to_pandas()

    frames = []
    seen = 0
    scanner = dataset.scanner(
        columns=cols, filter=expr, batch_size=batch_size, use_threads=True
    )
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        remaining = limit_rows - seen
        if remaining <= 0:
            break
        if len(df) > remaining:
            df = df.head(remaining)
        frames.append(df)
        seen += len(df)
        if seen >= limit_rows:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "scan_offers_static",
    "scan_snapshots",
    "scan_edgar",
]
