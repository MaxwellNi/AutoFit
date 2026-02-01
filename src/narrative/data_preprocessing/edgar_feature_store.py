from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

EDGAR_FEATURE_COLUMNS = [
    "total_offering_amount",
    "total_amount_sold",
    "total_remaining",
    "number_non_accredited_investors",
    "total_number_already_invested",
    "minimum_investment_accepted",
    "sales_commissions",
    "finders_fees",
    "gross_proceeds_used",
]

AGG_PREFIXES = ("last_", "mean_", "ema_")


def _is_delta_path(path: Path) -> bool:
    return (path / "_delta_log").exists()


def _open_dataset(path: Path, *, logger: Optional[Any] = None, label: Optional[str] = None) -> ds.Dataset:
    if _is_delta_path(path):
        try:
            from deltalake import DeltaTable
        except Exception as exc:
            raise RuntimeError(f"deltalake required to read Delta table: {path}") from exc
        dt = DeltaTable(str(path))
        dataset = dt.to_pyarrow_dataset()
        if logger is not None:
            logger.info(
                "delta_dataset=%s version=%s active_files=%s",
                label or str(path),
                dt.version(),
                len((getattr(dt, "file_uris", None) or getattr(dt, "files", lambda: []))()),
            )
        return dataset
    dataset = ds.dataset(
        str(path),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )
    if logger is not None:
        logger.info(
            "parquet_dataset=%s fragments=%s",
            label or str(path),
            sum(1 for _ in dataset.get_fragments()),
        )
    return dataset


def _normalize_utc_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return ts.astype("datetime64[ns, UTC]")
    except Exception:
        return pd.to_datetime(ts.astype("string"), errors="coerce", utc=True)


def _to_float(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, dict):
        for key in ("dollarAmount", "value", "amount"):
            if key in value:
                return _to_float(value[key])
        return np.nan
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        return _to_float(value[0])
    try:
        s = str(value).strip().replace(",", "")
        if s == "":
            return np.nan
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _extract_offering_features(offering_payload) -> dict:
    out = {k: np.nan for k in EDGAR_FEATURE_COLUMNS}
    if not isinstance(offering_payload, dict):
        return out

    offering_data = offering_payload.get("offeringData") or offering_payload.get("offering_data") or {}
    if not isinstance(offering_data, dict):
        return out

    offering_amounts = offering_data.get("offeringSalesAmounts") or {}
    investors = offering_data.get("investors") or {}
    sales_fees = offering_data.get("salesCommissionsFindersFees") or {}
    use_of_proceeds = offering_data.get("useOfProceeds") or {}

    out["total_offering_amount"] = _to_float(offering_amounts.get("totalOfferingAmount"))
    out["total_amount_sold"] = _to_float(offering_amounts.get("totalAmountSold"))
    out["total_remaining"] = _to_float(offering_amounts.get("totalRemaining"))
    out["number_non_accredited_investors"] = _to_float(investors.get("numberNonAccreditedInvestors"))
    out["total_number_already_invested"] = _to_float(investors.get("totalNumberAlreadyInvested"))
    out["minimum_investment_accepted"] = _to_float(offering_data.get("minimumInvestmentAccepted"))

    sales_comm = sales_fees.get("salesCommissions") or {}
    finders_fees = sales_fees.get("findersFees") or {}
    out["sales_commissions"] = _to_float(sales_comm.get("dollarAmount"))
    out["finders_fees"] = _to_float(finders_fees.get("dollarAmount"))

    proceeds = use_of_proceeds.get("grossProceedsUsed") or {}
    out["gross_proceeds_used"] = _to_float(proceeds.get("dollarAmount"))

    return out


def build_edgar_feature_store_v2(
    edgar_path: Path,
    output_path: Path,
    *,
    ema_alpha: float = 0.2,
    batch_size: int = 200_000,
    limit_rows: Optional[int] = None,
    snapshots_path: Optional[Path] = None,
    snapshot_time_col: str = "crawled_date",
    id_cols: Sequence[str] = ("platform_name", "offer_id", "cik"),
    align_to_snapshots: bool = False,
    partition_by_year: bool = False,
    cutoff_col: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Path:
    """
    Build compact EDGAR feature store (filing-level aggregates).
    Optionally align to snapshots via forward-fill.
    """
    if partition_by_year and output_path.suffix:
        output_path = output_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edgar_raw = load_edgar_features_from_parquet(
        edgar_path, batch_size=batch_size, limit_rows=limit_rows, logger=logger
    )
    edgar_agg = aggregate_edgar_features(edgar_raw, ema_alpha=ema_alpha)

    if not align_to_snapshots or snapshots_path is None:
        if partition_by_year:
            output_path.mkdir(parents=True, exist_ok=True)
            edgar_agg = edgar_agg.copy()
            edgar_agg["filed_year"] = pd.to_datetime(
                edgar_agg["filed_date"], errors="coerce", utc=True
            ).dt.year
            table = pa.Table.from_pandas(edgar_agg)
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["filed_year"])
            return output_path
        edgar_agg.to_parquet(output_path, index=False)
        return output_path

    # align to snapshots (streaming)
    dataset = _open_dataset(snapshots_path, logger=logger, label="offers_snapshots")
    available_cols = set(dataset.schema.names)
    snapshot_cols = list(dict.fromkeys(list(id_cols) + [snapshot_time_col] + ([cutoff_col] if cutoff_col else [])))
    snapshot_cols = [c for c in snapshot_cols if c in available_cols]
    cutoff_col_used = cutoff_col if cutoff_col in available_cols else None
    scanner = dataset.scanner(columns=snapshot_cols, batch_size=batch_size, use_threads=True)

    writer = None
    seen = 0
    for batch in scanner.to_batches():
        snaps = batch.to_pandas()
        if limit_rows is not None:
            remaining = limit_rows - seen
            if remaining <= 0:
                break
            snaps = snaps.head(remaining)
        aligned = align_edgar_features_to_snapshots(
            edgar_agg,
            snaps,
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
            cutoff_col=cutoff_col_used,
        )
        if partition_by_year:
            aligned = aligned.copy()
            if "snapshot_year" not in aligned.columns:
                aligned["snapshot_year"] = pd.to_datetime(
                    aligned[snapshot_time_col], errors="coerce", utc=True
                ).dt.year
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if "snapshot_year" not in table.schema.names:
                table = table.append_column(
                    "snapshot_year",
                    pa.array([None] * table.num_rows),
                )
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["snapshot_year"])
        else:
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
        seen += len(snaps)
        if limit_rows is not None and seen >= limit_rows:
            break

    if writer is not None:
        writer.close()
    else:
        empty = align_edgar_features_to_snapshots(
            edgar_agg.iloc[:0],
            pd.DataFrame(columns=list(id_cols) + [snapshot_time_col]),
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
            cutoff_col=cutoff_col_used,
        )
        if partition_by_year:
            output_path.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(empty, preserve_index=False)
            if "snapshot_year" not in table.schema.names:
                table = table.append_column(
                    "snapshot_year",
                    pa.array([None] * table.num_rows),
                )
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["snapshot_year"])
        else:
            empty.to_parquet(output_path, index=False)

    return output_path


def build_edgar_feature_store_v2(
    edgar_path: Path,
    output_path: Path,
    *,
    ema_alpha: float = 0.2,
    batch_size: int = 200_000,
    limit_rows: Optional[int] = None,
    snapshots_path: Optional[Path] = None,
    snapshot_time_col: str = "crawled_date",
    id_cols: Sequence[str] = ("platform_name", "offer_id", "cik"),
    align_to_snapshots: bool = False,
    partition_by_year: bool = False,
    cutoff_col: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Path:
    """
    Build compact EDGAR feature store (filing-level aggregates).
    Optionally align to snapshots via forward-fill.
    """
    if partition_by_year and output_path.suffix:
        output_path = output_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edgar_raw = load_edgar_features_from_parquet(
        edgar_path, batch_size=batch_size, limit_rows=limit_rows, logger=logger
    )
    edgar_agg = aggregate_edgar_features(edgar_raw, ema_alpha=ema_alpha)

    if not align_to_snapshots or snapshots_path is None:
        if partition_by_year:
            output_path.mkdir(parents=True, exist_ok=True)
            edgar_agg = edgar_agg.copy()
            edgar_agg["filed_year"] = pd.to_datetime(
                edgar_agg["filed_date"], errors="coerce", utc=True
            ).dt.year
            table = pa.Table.from_pandas(edgar_agg)
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["filed_year"])
            return output_path
        edgar_agg.to_parquet(output_path, index=False)
        return output_path

    # align to snapshots (streaming)
    dataset = _open_dataset(snapshots_path, logger=logger, label="offers_snapshots")
    available_cols = set(dataset.schema.names)
    snapshot_cols = list(dict.fromkeys(list(id_cols) + [snapshot_time_col] + ([cutoff_col] if cutoff_col else [])))
    snapshot_cols = [c for c in snapshot_cols if c in available_cols]
    cutoff_col_used = cutoff_col if cutoff_col in available_cols else None
    scanner = dataset.scanner(columns=snapshot_cols, batch_size=batch_size, use_threads=True)

    writer = None
    seen = 0
    for batch in scanner.to_batches():
        snaps = batch.to_pandas()
        if limit_rows is not None:
            remaining = limit_rows - seen
            if remaining <= 0:
                break
            snaps = snaps.head(remaining)
        aligned = align_edgar_features_to_snapshots(
            edgar_agg,
            snaps,
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
            cutoff_col=cutoff_col_used,
        )
        if partition_by_year:
            aligned = aligned.copy()
            if "snapshot_year" not in aligned.columns:
                aligned["snapshot_year"] = pd.to_datetime(
                    aligned[snapshot_time_col], errors="coerce", utc=True
                ).dt.year
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if "snapshot_year" not in table.schema.names:
                table = table.append_column(
                    "snapshot_year",
                    pa.array([None] * table.num_rows),
                )
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["snapshot_year"])
        else:
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
        seen += len(snaps)
        if limit_rows is not None and seen >= limit_rows:
            break

    if writer is not None:
        writer.close()
    else:
        empty = align_edgar_features_to_snapshots(
            edgar_agg.iloc[:0],
            pd.DataFrame(columns=list(id_cols) + [snapshot_time_col]),
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
            cutoff_col=cutoff_col_used,
        )
        if partition_by_year:
            output_path.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(empty, preserve_index=False)
            if "snapshot_year" not in table.schema.names:
                table = table.append_column(
                    "snapshot_year",
                    pa.array([None] * table.num_rows),
                )
            pq.write_to_dataset(table, root_path=str(output_path), partition_cols=["snapshot_year"])
        else:
            empty.to_parquet(output_path, index=False)

    return output_path


def extract_edgar_features(edgar_df: pd.DataFrame) -> pd.DataFrame:
    """Extract compact numeric features from raw EDGAR accessions."""
    if edgar_df.empty:
        cols = ["cik", "filed_date"] + EDGAR_FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    df = edgar_df.copy()
    if "cik" not in df.columns or "filed_date" not in df.columns:
        raise KeyError("edgar_df must include cik and filed_date")

    df["cik"] = df["cik"].astype(str)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True)

    if "submission_offering_data" in df.columns:
        features = df["submission_offering_data"].apply(_extract_offering_features)
    else:
        features = pd.Series([{k: np.nan for k in EDGAR_FEATURE_COLUMNS}] * len(df))

    features_df = pd.DataFrame(features.tolist())
    out = pd.concat([df[["cik", "filed_date"]].reset_index(drop=True), features_df], axis=1)

    out = out.dropna(subset=["cik"])
    out = out.sort_values(["cik", "filed_date"], kind="mergesort")
    out = out.drop_duplicates(subset=["cik", "filed_date"], keep="last").reset_index(drop=True)
    return out


def aggregate_edgar_features(
    edgar_features: pd.DataFrame,
    *,
    ema_alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Aggregate per filing into last/mean/ema features with missing masks.
    """
    if edgar_features.empty:
        cols = ["cik", "filed_date"]
        for col in EDGAR_FEATURE_COLUMNS:
            cols.append(f"last_{col}")
            cols.append(f"mean_{col}")
            cols.append(f"ema_{col}")
            cols.append(f"last_{col}_is_missing")
        return pd.DataFrame(columns=cols)

    df = edgar_features.copy()
    df["cik"] = df["cik"].astype(str)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True)
    df = df.sort_values(["cik", "filed_date"], kind="mergesort")

    base = df[EDGAR_FEATURE_COLUMNS].copy()
    missing = base.isna().rename(columns={c: f"last_{c}_is_missing" for c in base.columns})

    last_cols = base.rename(columns={c: f"last_{c}" for c in base.columns})

    mean_cols = []
    ema_cols = []
    grouped = base.groupby(df["cik"], sort=False)
    for col in base.columns:
        mean_col = grouped[col].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        ema_col = (
            grouped[col]
            .apply(lambda s: s.ewm(alpha=ema_alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
        mean_cols.append(mean_col.rename(f"mean_{col}"))
        ema_cols.append(ema_col.rename(f"ema_{col}"))

    mean_df = pd.concat(mean_cols, axis=1)
    ema_df = pd.concat(ema_cols, axis=1)

    out = pd.concat([df[["cik", "filed_date"]], last_cols, mean_df, ema_df, missing], axis=1)
    out = out.drop_duplicates(subset=["cik", "filed_date"], keep="last").reset_index(drop=True)
    return out


def aggregate_edgar_features(
    edgar_features: pd.DataFrame,
    *,
    ema_alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Aggregate per filing into last/mean/ema features with missing masks.
    """
    if edgar_features.empty:
        cols = ["cik", "filed_date"]
        for col in EDGAR_FEATURE_COLUMNS:
            cols.append(f"last_{col}")
            cols.append(f"mean_{col}")
            cols.append(f"ema_{col}")
            cols.append(f"last_{col}_is_missing")
        return pd.DataFrame(columns=cols)

    df = edgar_features.copy()
    df["cik"] = df["cik"].astype(str)
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce", utc=True)
    df = df.sort_values(["cik", "filed_date"], kind="mergesort")

    base = df[EDGAR_FEATURE_COLUMNS].copy()
    missing = base.isna().rename(columns={c: f"last_{c}_is_missing" for c in base.columns})

    last_cols = base.rename(columns={c: f"last_{c}" for c in base.columns})

    mean_cols = []
    ema_cols = []
    grouped = base.groupby(df["cik"], sort=False)
    for col in base.columns:
        mean_col = grouped[col].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        ema_col = (
            grouped[col]
            .apply(lambda s: s.ewm(alpha=ema_alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
        mean_cols.append(mean_col.rename(f"mean_{col}"))
        ema_cols.append(ema_col.rename(f"ema_{col}"))

    mean_df = pd.concat(mean_cols, axis=1)
    ema_df = pd.concat(ema_cols, axis=1)

    out = pd.concat([df[["cik", "filed_date"]], last_cols, mean_df, ema_df, missing], axis=1)
    out = out.drop_duplicates(subset=["cik", "filed_date"], keep="last").reset_index(drop=True)
    return out


def align_edgar_features_to_snapshots(
    edgar_features: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    *,
    snapshot_time_col: str = "crawled_date",
    id_cols: Sequence[str] = ("platform_name", "offer_id", "cik"),
    cutoff_col: Optional[str] = None,
) -> pd.DataFrame:
    """Forward-fill EDGAR features onto snapshot timeline per CIK."""
    if snapshots_df.empty:
        cols = list(id_cols) + [snapshot_time_col, "cutoff_ts", "edgar_filed_date", "edgar_has_filing", "edgar_valid"]
        if edgar_features is not None and len(edgar_features.columns) > 0:
            feat_cols = [c for c in edgar_features.columns if c not in {"cik", "filed_date"}]
            cols += feat_cols
        else:
            cols += EDGAR_FEATURE_COLUMNS + [f"{c}_is_missing" for c in EDGAR_FEATURE_COLUMNS]
        return pd.DataFrame(columns=cols)
        return pd.DataFrame(columns=cols)

    snaps = snapshots_df.copy()
    if "cik" not in snaps.columns:
        raise KeyError("snapshots_df must include cik")
    if snapshot_time_col not in snaps.columns:
        raise KeyError(f"snapshots_df must include {snapshot_time_col}")

    cutoff_col_used = cutoff_col if cutoff_col in snaps.columns else None
    keep_cols = list(dict.fromkeys(list(id_cols) + [snapshot_time_col] + ([cutoff_col_used] if cutoff_col_used else [])))
    snaps = snaps[keep_cols]
    snaps["cik"] = snaps["cik"].astype(str)
    snaps[snapshot_time_col] = _normalize_utc_ts(snaps[snapshot_time_col])
    snaps = snaps.dropna(subset=[snapshot_time_col])
    snaps = snaps[~snaps["cik"].isin(["", "nan", "None", "NaN"])]
    snaps = snaps.dropna(subset=[snapshot_time_col])
    snaps = snaps[~snaps["cik"].isin(["", "nan", "None", "NaN"])]
    if cutoff_col_used:
        snaps[cutoff_col_used] = _normalize_utc_ts(snaps[cutoff_col_used])
        snaps["cutoff_ts"] = snaps[cutoff_col_used]
    else:
        snaps["cutoff_ts"] = snaps[snapshot_time_col]
    snaps = snaps.sort_values([snapshot_time_col, "cik"], kind="mergesort")
    edgar = edgar_features.copy()
    if not edgar.empty:
        edgar["cik"] = edgar["cik"].astype(str)
        edgar["filed_date"] = _normalize_utc_ts(edgar["filed_date"])
        edgar = edgar.dropna(subset=["filed_date"])
        edgar = edgar[~edgar["cik"].isin(["", "nan", "None", "NaN"])]
        edgar = edgar.sort_values(["filed_date", "cik"], kind="mergesort")
        edgar = edgar.dropna(subset=["filed_date"])
        edgar = edgar[~edgar["cik"].isin(["", "nan", "None", "NaN"])]
        edgar = edgar.sort_values(["filed_date", "cik"], kind="mergesort")
    aligned = pd.merge_asof(
        snaps,
        edgar,
        left_on=snapshot_time_col,
        right_on="filed_date",
        by="cik",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = aligned.rename(columns={"filed_date": "edgar_filed_date"})
    aligned["edgar_has_filing"] = aligned["edgar_filed_date"].notna()
    aligned["edgar_valid"] = aligned["edgar_has_filing"] & (
        pd.to_datetime(aligned["edgar_filed_date"], errors="coerce", utc=True)
        <= pd.to_datetime(aligned["cutoff_ts"], errors="coerce", utc=True)
    )

    invalid = aligned["edgar_has_filing"] & (~aligned["edgar_valid"])
    if invalid.any():
        feature_cols = [
            c for c in aligned.columns
            if c not in list(id_cols) + [snapshot_time_col, "cutoff_ts", "edgar_filed_date", "edgar_has_filing", "edgar_valid"]
        ]
        aligned.loc[invalid, feature_cols] = pd.NA
        aligned.loc[invalid, "edgar_filed_date"] = pd.NaT
        aligned.loc[invalid, "edgar_has_filing"] = False
        aligned.loc[invalid, "edgar_valid"] = False

    feature_cols = [
        c for c in aligned.columns
        if c not in list(id_cols) + [
            snapshot_time_col,
            "cutoff_ts",
            "edgar_filed_date",
            "edgar_has_filing",
            "edgar_valid",
        ]
    ]
    has_mask = any(c.endswith("_is_missing") for c in feature_cols)
    aligned["edgar_valid"] = aligned["edgar_has_filing"] & (
        pd.to_datetime(aligned["edgar_filed_date"], errors="coerce", utc=True)
        <= pd.to_datetime(aligned["cutoff_ts"], errors="coerce", utc=True)
    )

    invalid = aligned["edgar_has_filing"] & (~aligned["edgar_valid"])
    if invalid.any():
        feature_cols = [
            c for c in aligned.columns
            if c not in list(id_cols) + [snapshot_time_col, "cutoff_ts", "edgar_filed_date", "edgar_has_filing", "edgar_valid"]
        ]
        aligned.loc[invalid, feature_cols] = pd.NA
        aligned.loc[invalid, "edgar_filed_date"] = pd.NaT
        aligned.loc[invalid, "edgar_has_filing"] = False
        aligned.loc[invalid, "edgar_valid"] = False

    feature_cols = [
        c for c in aligned.columns
        if c not in list(id_cols) + [
            snapshot_time_col,
            "cutoff_ts",
            "edgar_filed_date",
            "edgar_has_filing",
            "edgar_valid",
        ]
    ]
    has_mask = any(c.endswith("_is_missing") for c in feature_cols)
    if not has_mask:
        for col in feature_cols:
            if col.endswith("_is_missing"):
                continue
            aligned[f"{col}_is_missing"] = aligned[col].isna()

    return aligned


def load_edgar_features_from_parquet(
    edgar_path: Path,
    *,
    batch_size: int = 200_000,
    limit_rows: Optional[int] = None,
    cik_filter: Optional[Sequence[str]] = None,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    dataset = _open_dataset(edgar_path, logger=logger, label="edgar_accessions")
    columns = ["cik", "filed_date", "submission_offering_data"]
    filter_expr = None
    if cik_filter and "cik" in (dataset.schema.names if hasattr(dataset.schema, "names") else [f.name for f in dataset.schema]):
        casted = pa.array([str(c) for c in cik_filter])
        filter_expr = ds.field("cik").isin(casted)
    scanner = dataset.scanner(columns=columns, batch_size=batch_size, filter=filter_expr, use_threads=True) if filter_expr is not None else dataset.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    frames = []
    seen = 0
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if limit_rows is not None:
            remaining = limit_rows - seen
            if remaining <= 0:
                break
            df = df.head(remaining)
        frames.append(extract_edgar_features(df))
        seen += len(df)
        if limit_rows is not None and seen >= limit_rows:
            break

    if not frames:
        return extract_edgar_features(pd.DataFrame(columns=columns))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["cik", "filed_date"], kind="mergesort")
    out = out.drop_duplicates(subset=["cik", "filed_date"], keep="last").reset_index(drop=True)
    return out


def build_edgar_feature_store(
    edgar_path: Path,
    snapshots_path: Path,
    output_path: Path,
    *,
    snapshot_time_col: str = "crawled_date",
    id_cols: Sequence[str] = ("platform_name", "offer_id", "cik"),
    batch_size: int = 200_000,
    limit_rows: Optional[int] = None,
    logger: Optional[Any] = None,
) -> Path:
    """Build EDGAR feature store aligned to offers snapshots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edgar_features = load_edgar_features_from_parquet(
        edgar_path, batch_size=batch_size, limit_rows=limit_rows, logger=logger
    )

    dataset = _open_dataset(snapshots_path, logger=logger, label="offers_snapshots")
    snapshot_cols = list(dict.fromkeys(list(id_cols) + [snapshot_time_col]))
    scanner = dataset.scanner(columns=snapshot_cols, use_threads=True)

    writer = None
    seen = 0
    for batch in scanner.to_batches(batch_size=batch_size):
        snaps = batch.to_pandas()
        if limit_rows is not None:
            remaining = limit_rows - seen
            if remaining <= 0:
                break
            snaps = snaps.head(remaining)
        aligned = align_edgar_features_to_snapshots(
            edgar_features,
            snaps,
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
        )
        table = pa.Table.from_pandas(aligned)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema)
        writer.write_table(table)
        seen += len(snaps)
        if limit_rows is not None and seen >= limit_rows:
            break

    if writer is not None:
        writer.close()
    else:
        empty = align_edgar_features_to_snapshots(
            edgar_features.iloc[:0],
            pd.DataFrame(columns=list(id_cols) + [snapshot_time_col]),
            snapshot_time_col=snapshot_time_col,
            id_cols=id_cols,
        )
        empty.to_parquet(output_path, index=False)

    return output_path


__all__ = [
    "EDGAR_FEATURE_COLUMNS",
    "extract_edgar_features",
    "aggregate_edgar_features",
    "align_edgar_features_to_snapshots",
    "load_edgar_features_from_parquet",
    "build_edgar_feature_store",
    "build_edgar_feature_store_v2",
]
