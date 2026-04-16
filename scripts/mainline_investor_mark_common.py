#!/usr/bin/env python3
"""Shared loading helpers for marked-investor audits and weak-label tables."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.narrative.data_preprocessing.block3_dataset import FreezePointer


DEFAULT_POINTER_PATH = REPO_ROOT / "docs" / "audits" / "FULL_SCALE_POINTER.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "analysis" / "mainline_marked_investor"


BASE_INVESTOR_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "cik",
    "crawled_date_day",
    "investors",
    "investors_count",
    "investors__json",
    "investors__len",
    "investors__hash",
    "investor_website",
    "investment_type",
    "non_national_investors",
)
MARK_PROXY_COLUMNS: tuple[str, ...] = (
    "last_number_non_accredited_investors",
    "mean_number_non_accredited_investors",
    "ema_number_non_accredited_investors",
    "last_total_number_already_invested",
    "mean_total_number_already_invested",
    "ema_total_number_already_invested",
    "last_minimum_investment_accepted",
    "mean_minimum_investment_accepted",
    "ema_minimum_investment_accepted",
    "last_total_offering_amount",
    "mean_total_offering_amount",
    "ema_total_offering_amount",
    "last_total_amount_sold",
    "mean_total_amount_sold",
    "ema_total_amount_sold",
    "last_total_remaining",
    "mean_total_remaining",
    "ema_total_remaining",
)


def resolve_pointer_path(pointer_path: str | Path | None) -> Path:
    if pointer_path is None:
        return DEFAULT_POINTER_PATH
    path = Path(pointer_path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_pointer(pointer_path: str | Path | None = None) -> FreezePointer:
    return FreezePointer.load(resolve_pointer_path(pointer_path))


def output_dir(path_like: str | Path | None = None) -> Path:
    if path_like is None:
        return DEFAULT_OUTPUT_DIR
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def offers_core_dataset_path(pointer: FreezePointer) -> Path:
    parquet_file = pointer.offers_core_daily_dir / "offers_core_daily.parquet"
    return parquet_file if parquet_file.exists() else pointer.offers_core_daily_dir


def edgar_features_dataset_path(pointer: FreezePointer) -> Path:
    parquet_dir = pointer.edgar_store_dir / "edgar_features"
    return parquet_dir if parquet_dir.exists() else pointer.edgar_store_dir


def available_core_columns(pointer: FreezePointer) -> list[str]:
    dataset = ds.dataset(str(offers_core_dataset_path(pointer)), format="parquet")
    return list(dataset.schema.names)


def available_edgar_columns(pointer: FreezePointer) -> list[str]:
    dataset = ds.dataset(str(edgar_features_dataset_path(pointer)), format="parquet")
    return list(dataset.schema.names)


def available_panel_columns(pointer: FreezePointer) -> list[str]:
    return sorted(set(available_core_columns(pointer)) | set(available_edgar_columns(pointer)))


def requested_core_columns(pointer: FreezePointer, requested: Sequence[str]) -> list[str]:
    available = set(available_core_columns(pointer))
    cols = [column for column in requested if column in available]
    for required in ("entity_id", "crawled_date_day"):
        if required in available and required not in cols:
            cols.insert(0, required)
    return cols


def requested_edgar_columns(pointer: FreezePointer, requested: Sequence[str]) -> list[str]:
    available = set(available_edgar_columns(pointer))
    cols = [column for column in requested if column in available]
    for required in ("cik", "crawled_date_day"):
        if required in available and required not in cols:
            cols.insert(0, required)
    return cols


def iter_core_batches(
    pointer: FreezePointer,
    columns: Sequence[str],
    *,
    batch_size: int = 65536,
) -> Iterable[pd.DataFrame]:
    dataset = ds.dataset(str(offers_core_dataset_path(pointer)), format="parquet")
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def _normalize_day_key(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    return normalized.fillna("")


def _normalize_cik_key(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def _load_edgar_proxy_frame(pointer: FreezePointer, requested_columns: Sequence[str]) -> pd.DataFrame:
    edgar_columns = [
        column
        for column in requested_edgar_columns(pointer, requested_columns)
        if column not in {"cik", "crawled_date_day"}
    ]
    if not edgar_columns:
        return pd.DataFrame()

    dataset = ds.dataset(str(edgar_features_dataset_path(pointer)), format="parquet")
    columns = ["cik", "crawled_date_day", *edgar_columns]
    frame = dataset.to_table(columns=columns).to_pandas()
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["_merge_cik"] = _normalize_cik_key(frame["cik"])
    frame["_merge_crawled_date_day"] = _normalize_day_key(frame["crawled_date_day"])
    frame = frame[
        frame["_merge_cik"].ne("") & frame["_merge_crawled_date_day"].ne("")
    ].copy()
    frame = frame.drop_duplicates(["_merge_cik", "_merge_crawled_date_day"], keep="last")
    return frame


def _attach_edgar_proxy_columns(
    batch: pd.DataFrame,
    edgar_proxy_frame: pd.DataFrame,
    requested_columns: Sequence[str],
) -> pd.DataFrame:
    if batch.empty or edgar_proxy_frame.empty or "cik" not in batch.columns or "crawled_date_day" not in batch.columns:
        return batch

    proxy_columns = [
        column
        for column in requested_columns
        if column in edgar_proxy_frame.columns and column not in {"cik", "crawled_date_day"}
    ]
    if not proxy_columns:
        return batch

    enriched = batch.copy()
    enriched["_merge_cik"] = _normalize_cik_key(enriched["cik"])
    enriched["_merge_crawled_date_day"] = _normalize_day_key(enriched["crawled_date_day"])
    batch_ciks = enriched["_merge_cik"]
    batch_ciks = batch_ciks[batch_ciks.ne("")].unique().tolist()
    batch_days = enriched["_merge_crawled_date_day"]
    batch_days = batch_days[batch_days.ne("")].unique().tolist()

    if not batch_ciks or not batch_days:
        for column in proxy_columns:
            if column not in enriched.columns:
                enriched[column] = np.nan
        return enriched.drop(columns=["_merge_cik", "_merge_crawled_date_day"], errors="ignore")

    lookup = edgar_proxy_frame[
        edgar_proxy_frame["_merge_cik"].isin(batch_ciks)
        & edgar_proxy_frame["_merge_crawled_date_day"].isin(batch_days)
    ]
    merge_columns = ["_merge_cik", "_merge_crawled_date_day", *proxy_columns]
    enriched = enriched.merge(lookup[merge_columns], on=["_merge_cik", "_merge_crawled_date_day"], how="left")
    return enriched.drop(columns=["_merge_cik", "_merge_crawled_date_day"], errors="ignore")


def investor_surface_mask(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    json_present = frame.get("investors__json", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    website_present = frame.get("investor_website", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    investment_type_present = frame.get("investment_type", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    list_len = pd.to_numeric(frame.get("investors__len", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    investors_count = pd.to_numeric(frame.get("investors_count", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    return json_present | website_present | investment_type_present | (list_len > 0.0) | (investors_count > 0.0)


def investor_reference_mask(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    json_present = frame.get("investors__json", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    website_present = frame.get("investor_website", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    return json_present | website_present


def investor_mark_rich_mask(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    non_national = pd.to_numeric(frame.get("non_national_investors", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    proxy_present = pd.Series(False, index=index)
    for column in MARK_PROXY_COLUMNS:
        if column in frame.columns:
            proxy_present |= pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0.0
    return investor_surface_mask(frame) | (non_national > 0.0) | proxy_present


def selection_mask(frame: pd.DataFrame, selection_mode: str) -> pd.Series:
    if selection_mode == "reference_surface":
        return investor_reference_mask(frame)
    if selection_mode == "mark_rich":
        return investor_mark_rich_mask(frame)
    return investor_surface_mask(frame)


def selection_score(frame: pd.DataFrame, selection_mode: str) -> pd.Series:
    index = frame.index
    if selection_mode == "reference_surface":
        return investor_reference_mask(frame).astype(np.float64)
    if selection_mode != "mark_rich":
        return investor_surface_mask(frame).astype(np.float64)

    investors_count = pd.to_numeric(frame.get("investors_count", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    list_len = pd.to_numeric(frame.get("investors__len", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    non_national = pd.to_numeric(frame.get("non_national_investors", pd.Series(0.0, index=index)), errors="coerce").fillna(0.0)
    investment_type_present = frame.get("investment_type", pd.Series("", index=index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    score = np.zeros(len(frame), dtype=np.float64)
    score += investor_surface_mask(frame).to_numpy(dtype=np.float64, copy=False)
    score += 2.0 * investor_reference_mask(frame).to_numpy(dtype=np.float64, copy=False)
    score += 1.5 * investment_type_present.to_numpy(dtype=np.float64, copy=False)
    score += 2.0 * (non_national > 0.0).to_numpy(dtype=np.float64, copy=False)
    score += 0.5 * (investors_count > 0.0).to_numpy(dtype=np.float64, copy=False)
    score += 0.5 * (list_len > 0.0).to_numpy(dtype=np.float64, copy=False)
    for column in MARK_PROXY_COLUMNS:
        if column in frame.columns:
            proxy_positive = pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0.0
            score += 1.0 * proxy_positive.to_numpy(dtype=np.float64, copy=False)
    return pd.Series(score, index=index, dtype=np.float64)


def load_investor_mark_panel(
    pointer_path: str | Path | None = None,
    *,
    requested_columns: Sequence[str] | None = None,
    entity_limit: int = 512,
    max_rows_per_entity: int = 16,
    max_rows: int = 50000,
    batch_size: int = 65536,
    selection_mode: str = "investor_surface",
) -> pd.DataFrame:
    pointer = load_pointer(pointer_path)
    requested = tuple(requested_columns or (BASE_INVESTOR_COLUMNS + MARK_PROXY_COLUMNS))
    columns = requested_core_columns(pointer, requested)
    if not columns:
        return pd.DataFrame(columns=list(requested))

    edgar_proxy_columns = [
        column
        for column in requested_edgar_columns(pointer, requested)
        if column not in {"cik", "crawled_date_day"}
    ]
    output_columns = list(dict.fromkeys(columns + edgar_proxy_columns))
    edgar_proxy_frame = _load_edgar_proxy_frame(pointer, requested) if edgar_proxy_columns else pd.DataFrame()

    first_pass_requested = list(BASE_INVESTOR_COLUMNS)
    if selection_mode == "mark_rich":
        first_pass_requested.extend(MARK_PROXY_COLUMNS)
    first_pass_columns = requested_core_columns(pointer, tuple(first_pass_requested))
    entity_scores: dict[str, float] = defaultdict(float)
    for batch in iter_core_batches(pointer, first_pass_columns, batch_size=batch_size):
        if batch.empty:
            continue
        if selection_mode == "mark_rich" and not edgar_proxy_frame.empty:
            batch = _attach_edgar_proxy_columns(batch, edgar_proxy_frame, MARK_PROXY_COLUMNS)
        scores = selection_score(batch, selection_mode)
        mask = scores > 0.0
        if not bool(mask.any()):
            continue
        scored = batch.loc[mask, ["entity_id"]].copy()
        scored["_selection_score"] = scores.loc[mask].to_numpy(dtype=np.float64, copy=False)
        for entity_id, value in scored.groupby("entity_id")["_selection_score"].sum().items():
            entity_scores[str(entity_id)] += float(value)

    if not entity_scores:
        return pd.DataFrame(columns=list(columns))

    selected_entities = None
    if entity_limit > 0:
        ranked = sorted(entity_scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
        selected_entities = {entity_id for entity_id, _ in ranked[:entity_limit]}

    frames: list[pd.DataFrame] = []
    for batch in iter_core_batches(pointer, columns, batch_size=batch_size):
        if batch.empty:
            continue
        batch = batch.copy()
        batch["entity_id"] = batch["entity_id"].astype(str)
        if edgar_proxy_columns:
            batch = _attach_edgar_proxy_columns(batch, edgar_proxy_frame, edgar_proxy_columns)
        mask = selection_mask(batch, selection_mode)
        if selected_entities is not None:
            mask &= batch["entity_id"].isin(selected_entities)
        if not bool(mask.any()):
            continue
        batch_columns = [column for column in output_columns if column in batch.columns]
        frames.append(batch.loc[mask, batch_columns].copy())

    if not frames:
        return pd.DataFrame(columns=output_columns)

    panel = pd.concat(frames, ignore_index=True)
    if "crawled_date_day" in panel.columns:
        panel["crawled_date_day"] = pd.to_datetime(panel["crawled_date_day"], errors="coerce")
    sort_cols = [column for column in ("entity_id", "crawled_date_day") if column in panel.columns]
    if sort_cols:
        panel = panel.sort_values(sort_cols, kind="mergesort")
    if max_rows_per_entity > 0 and "entity_id" in panel.columns:
        panel = panel.groupby("entity_id", group_keys=False).tail(max_rows_per_entity)
    if max_rows > 0 and len(panel) > max_rows:
        if "crawled_date_day" in panel.columns:
            panel = panel.sort_values(["crawled_date_day", "entity_id"], kind="mergesort").tail(max_rows)
        else:
            panel = panel.tail(max_rows)
    return panel.reset_index(drop=True)


__all__ = [
    "BASE_INVESTOR_COLUMNS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POINTER_PATH",
    "MARK_PROXY_COLUMNS",
    "available_core_columns",
    "available_edgar_columns",
    "available_panel_columns",
    "investor_mark_rich_mask",
    "investor_reference_mask",
    "investor_surface_mask",
    "load_investor_mark_panel",
    "load_pointer",
    "output_dir",
    "requested_core_columns",
    "resolve_pointer_path",
]