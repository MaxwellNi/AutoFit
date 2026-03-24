#!/usr/bin/env python3
"""
Utilities for V740 multisource alignment and dual-clock tokenization.

This module is intentionally lightweight and pre-benchmark. It does not change
current Block 3 execution paths. It provides reusable feature builders for the
future V740 event-memory path, where `core` remains a regular state stream and
`text` / `edgar` are represented as sparse event streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_RECENCY_BUCKETS: Tuple[int, ...] = (1, 7, 30, 90)
KNOWN_EDGAR_COLUMNS: Tuple[str, ...] = (
    "last_total_offering_amount",
    "last_total_amount_sold",
    "last_total_remaining",
    "last_number_non_accredited_investors",
    "last_total_number_already_invested",
    "last_minimum_investment_accepted",
    "last_sales_commissions",
    "last_finders_fees",
    "last_gross_proceeds_used",
    "mean_total_offering_amount",
    "mean_total_amount_sold",
    "mean_total_remaining",
    "mean_number_non_accredited_investors",
    "mean_total_number_already_invested",
    "mean_minimum_investment_accepted",
    "mean_sales_commissions",
    "mean_finders_fees",
    "mean_gross_proceeds_used",
    "ema_total_offering_amount",
    "ema_total_amount_sold",
    "ema_total_remaining",
    "ema_number_non_accredited_investors",
    "ema_total_number_already_invested",
    "ema_minimum_investment_accepted",
    "ema_sales_commissions",
    "ema_finders_fees",
    "ema_gross_proceeds_used",
    "last_total_offering_amount_is_missing",
    "last_total_amount_sold_is_missing",
    "last_total_remaining_is_missing",
    "last_number_non_accredited_investors_is_missing",
    "last_total_number_already_invested_is_missing",
    "last_minimum_investment_accepted_is_missing",
    "last_sales_commissions_is_missing",
    "last_finders_fees_is_missing",
    "last_gross_proceeds_used_is_missing",
    "edgar_has_filing",
    "edgar_valid",
    "snapshot_year",
)


@dataclass(frozen=True)
class DualClockConfig:
    max_events: int = 4
    recency_buckets: Tuple[int, ...] = DEFAULT_RECENCY_BUCKETS
    embedding_prefix: str = ""
    time_col: str = "crawled_date_day"
    entity_col: str = "entity_id"


def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.Series:
    return pd.to_datetime(df[time_col], errors="coerce")


def _source_presence_mask(df: pd.DataFrame, source_cols: Sequence[str]) -> pd.Series:
    if not source_cols:
        return pd.Series(False, index=df.index)
    present = df[list(source_cols)].notna()
    return present.any(axis=1)


def _safe_numeric_mean(values: np.ndarray, dim: int = 0) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    with np.errstate(invalid="ignore"):
        out = np.nanmean(arr, axis=dim)
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32, copy=False)


def _event_payload_matrix(df: pd.DataFrame, source_cols: Sequence[str]) -> np.ndarray:
    if not source_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    vals = df[list(source_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    return vals


def build_recent_event_tokens(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: Sequence[str],
    cfg: DualClockConfig,
) -> np.ndarray:
    """Build fixed-count recent event tokens for one entity.

    Token layout:
    [time_since_event_days, availability_lag_days(placeholder), presence_ratio, payload...]
    """
    if entity_df.empty:
        width = 3 + len(source_cols)
        return np.zeros((cfg.max_events, width), dtype=np.float32)

    work = entity_df.copy()
    work[cfg.time_col] = _ensure_datetime(work, cfg.time_col)
    work = work[work[cfg.time_col].notna()]
    work = work[work[cfg.time_col] <= prediction_time]
    work = work[_source_presence_mask(work, source_cols)]
    if work.empty:
        width = 3 + len(source_cols)
        return np.zeros((cfg.max_events, width), dtype=np.float32)

    work = work.sort_values(cfg.time_col, ascending=False).head(cfg.max_events)
    payload = _event_payload_matrix(work, source_cols)
    time_since = (prediction_time - work[cfg.time_col]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    availability_lag = np.zeros_like(time_since, dtype=np.float32)
    presence_ratio = np.ones_like(time_since, dtype=np.float32)
    tokens = np.concatenate([
        time_since[:, None].astype(np.float32),
        availability_lag[:, None],
        presence_ratio[:, None],
        payload,
    ], axis=1)

    if len(tokens) < cfg.max_events:
        pad = np.zeros((cfg.max_events - len(tokens), tokens.shape[1]), dtype=np.float32)
        tokens = np.vstack([tokens, pad])
    return tokens.astype(np.float32, copy=False)


def build_recency_bucket_tokens(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: Sequence[str],
    cfg: DualClockConfig,
) -> np.ndarray:
    """Build fixed-interval recency summary tokens for one entity.

    Token layout per bucket:
    [event_count, min_age, max_age, mean_age, payload_mean...]
    """
    width = 4 + len(source_cols)
    if entity_df.empty:
        return np.zeros((len(cfg.recency_buckets) + 1, width), dtype=np.float32)

    work = entity_df.copy()
    work[cfg.time_col] = _ensure_datetime(work, cfg.time_col)
    work = work[work[cfg.time_col].notna()]
    work = work[work[cfg.time_col] <= prediction_time]
    work = work[_source_presence_mask(work, source_cols)]
    if work.empty:
        return np.zeros((len(cfg.recency_buckets) + 1, width), dtype=np.float32)

    age_days = (prediction_time - work[cfg.time_col]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    payload = _event_payload_matrix(work, source_cols)

    edges = [0.0] + [float(x) for x in cfg.recency_buckets] + [float("inf")]
    rows: List[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (age_days >= left) & (age_days <= right)
        if not mask.any():
            rows.append(np.zeros((width,), dtype=np.float32))
            continue
        ages = age_days[mask].astype(np.float32, copy=False)
        payload_mean = _safe_numeric_mean(payload[mask], dim=0)
        row = np.concatenate([
            np.array([
                float(mask.sum()),
                float(ages.min()),
                float(ages.max()),
                float(ages.mean()),
            ], dtype=np.float32),
            payload_mean,
        ])
        rows.append(row.astype(np.float32, copy=False))
    return np.vstack(rows)


def build_dual_clock_memory_for_entity(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: Sequence[str],
    cfg: DualClockConfig | None = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or DualClockConfig()
    return {
        "recent_tokens": build_recent_event_tokens(entity_df, prediction_time, source_cols, cfg),
        "bucket_tokens": build_recency_bucket_tokens(entity_df, prediction_time, source_cols, cfg),
    }


def infer_text_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("text_emb_")]


def infer_edgar_columns(df: pd.DataFrame) -> List[str]:
    present = [c for c in KNOWN_EDGAR_COLUMNS if c in df.columns]
    if present:
        return present

    keywords = (
        "edgar_",
        "offering_amount",
        "amount_sold",
        "remaining",
        "non_accredited",
        "already_invested",
        "minimum_investment",
        "sales_commissions",
        "finders_fees",
        "gross_proceeds",
    )
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if any(k in c for k in keywords)]


def infer_numeric_source_columns(
    df: pd.DataFrame,
    exclude: Sequence[str],
) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in set(exclude)]


__all__ = [
    "DEFAULT_RECENCY_BUCKETS",
    "DualClockConfig",
    "build_recent_event_tokens",
    "build_recency_bucket_tokens",
    "build_dual_clock_memory_for_entity",
    "infer_text_columns",
    "infer_edgar_columns",
    "infer_numeric_source_columns",
    "KNOWN_EDGAR_COLUMNS",
]
