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
    series = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    return series.dt.tz_convert(None)


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


def _normalized_l2_shift(current: np.ndarray, previous: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    if current.size == 0 or previous.size == 0:
        return 0.0
    delta = current - previous
    denom = float(np.sqrt(max(delta.size, 1)))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(delta) / denom)


def _cosine_shift(current: np.ndarray, previous: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    if current.size == 0 or previous.size == 0:
        return 0.0
    current_norm = float(np.linalg.norm(current))
    previous_norm = float(np.linalg.norm(previous))
    if current_norm <= 1e-8 and previous_norm <= 1e-8:
        return 0.0
    if current_norm <= 1e-8 or previous_norm <= 1e-8:
        return 1.0
    cosine = float(np.dot(current, previous) / (current_norm * previous_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(1.0 - cosine)


def _dedupe_text_events(
    entity_df: pd.DataFrame,
    source_cols: Sequence[str],
    event_col: str,
    fallback_col: str,
    min_l2_shift: float,
    min_cosine_shift: float,
    refresh_days: int,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    work = entity_df.copy()
    work["_event_time"] = _ensure_event_time(work, event_col, fallback_col)
    work = work[work["_event_time"].notna()]
    work = work.sort_values("_event_time").reset_index(drop=True)
    payload = _event_payload_matrix(work, source_cols)
    if work.empty or len(payload) == 0:
        return {
            "events": work.iloc[0:0].copy(),
            "payload": np.zeros((0, len(source_cols)), dtype=np.float32),
            "gap_days": np.zeros((0,), dtype=np.float32),
            "novelty_l2": np.zeros((0,), dtype=np.float32),
            "cosine_shift": np.zeros((0,), dtype=np.float32),
        }

    keep_idx = [0]
    gap_days = [0.0]
    novelty_l2 = [0.0]
    cosine_shift = [0.0]
    last_payload = payload[0]
    last_time = work.loc[0, "_event_time"]

    for idx in range(1, len(work)):
        event_time = work.loc[idx, "_event_time"]
        gap = float((event_time - last_time).total_seconds() / 86400.0)
        l2_shift = _normalized_l2_shift(payload[idx], last_payload)
        cos_shift = _cosine_shift(payload[idx], last_payload)
        if l2_shift > min_l2_shift or cos_shift > min_cosine_shift or gap >= float(refresh_days):
            keep_idx.append(idx)
            gap_days.append(max(gap, 0.0))
            novelty_l2.append(l2_shift)
            cosine_shift.append(cos_shift)
            last_payload = payload[idx]
            last_time = event_time

    kept = work.iloc[keep_idx].reset_index(drop=True)
    kept_payload = payload[np.asarray(keep_idx, dtype=np.int64)]
    return {
        "events": kept,
        "payload": kept_payload.astype(np.float32, copy=False),
        "gap_days": np.asarray(gap_days, dtype=np.float32),
        "novelty_l2": np.asarray(novelty_l2, dtype=np.float32),
        "cosine_shift": np.asarray(cosine_shift, dtype=np.float32),
    }


def _ensure_event_time(df: pd.DataFrame, event_col: str, fallback_col: str) -> pd.Series:
    if event_col in df.columns:
        series = pd.to_datetime(df[event_col], errors="coerce", utc=True)
        series = series.dt.tz_convert(None)
        if series.notna().any():
            return series
    return _ensure_datetime(df, fallback_col)


def _ensure_availability_time(df: pd.DataFrame, availability_col: str, fallback_col: str) -> pd.Series:
    if availability_col in df.columns:
        series = pd.to_datetime(df[availability_col], errors="coerce", utc=True)
        series = series.dt.tz_convert(None)
        if series.notna().any():
            return series
    return _ensure_datetime(df, fallback_col)


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


def build_source_native_edgar_memory(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: Sequence[str],
    cfg: DualClockConfig | None = None,
    event_col: str = "edgar_filed_date",
    availability_col: str = "cutoff_ts",
) -> Dict[str, np.ndarray]:
    """Build EDGAR memory using source-native event and availability times.

    The current joined daily panel often repeats the same filing-derived state
    across several daily rows. This helper deduplicates those repeated rows by
    event time before constructing recent-event and recency-bucket memories.
    """
    cfg = cfg or DualClockConfig()
    recent_width = 3 + len(source_cols)
    bucket_width = 4 + len(source_cols)
    if entity_df.empty:
        return {
            "recent_tokens": np.zeros((cfg.max_events, recent_width), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, bucket_width), dtype=np.float32),
        }

    work = entity_df.copy()
    work["_event_time"] = _ensure_event_time(work, event_col, cfg.time_col)
    work["_availability_time"] = _ensure_availability_time(work, availability_col, cfg.time_col)
    work = work[work["_event_time"].notna() & work["_availability_time"].notna()]
    work = work[work["_availability_time"] <= prediction_time]
    work = work[_source_presence_mask(work, source_cols)]
    if work.empty:
        return {
            "recent_tokens": np.zeros((cfg.max_events, recent_width), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, bucket_width), dtype=np.float32),
        }

    # Keep the earliest availability for each filing date so repeated carried
    # rows in the daily panel do not masquerade as multiple filing events.
    work = work.sort_values(["_availability_time", "_event_time"])
    work = work.drop_duplicates(subset=["_event_time"], keep="first")

    payload = _event_payload_matrix(work, source_cols)
    event_age_days = (
        prediction_time - work["_event_time"]
    ).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    availability_lag_days = (
        work["_availability_time"] - work["_event_time"]
    ).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    presence_ratio = np.ones_like(event_age_days, dtype=np.float32)

    order = np.argsort(event_age_days)
    work = work.iloc[order]
    payload = payload[order]
    event_age_days = event_age_days[order]
    availability_lag_days = availability_lag_days[order]

    recent_idx = np.arange(min(cfg.max_events, len(work)))
    recent_tokens = np.concatenate(
        [
            event_age_days[recent_idx, None].astype(np.float32),
            availability_lag_days[recent_idx, None].astype(np.float32),
            presence_ratio[recent_idx, None].astype(np.float32),
            payload[recent_idx],
        ],
        axis=1,
    )
    if len(recent_tokens) < cfg.max_events:
        pad = np.zeros((cfg.max_events - len(recent_tokens), recent_width), dtype=np.float32)
        recent_tokens = np.vstack([recent_tokens, pad])

    edges = [0.0] + [float(x) for x in cfg.recency_buckets] + [float("inf")]
    bucket_rows: List[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (event_age_days >= left) & (event_age_days <= right)
        if not mask.any():
            bucket_rows.append(np.zeros((bucket_width,), dtype=np.float32))
            continue
        ages = event_age_days[mask].astype(np.float32, copy=False)
        payload_mean = _safe_numeric_mean(payload[mask], dim=0)
        row = np.concatenate(
            [
                np.array(
                    [
                        float(mask.sum()),
                        float(ages.min()),
                        float(ages.max()),
                        float(ages.mean()),
                    ],
                    dtype=np.float32,
                ),
                payload_mean,
            ]
        )
        bucket_rows.append(row.astype(np.float32, copy=False))

    return {
        "recent_tokens": recent_tokens.astype(np.float32, copy=False),
        "bucket_tokens": np.vstack(bucket_rows),
    }


def build_source_native_text_memory(
    entity_df: pd.DataFrame,
    prediction_time: pd.Timestamp,
    source_cols: Sequence[str],
    cfg: DualClockConfig | None = None,
    event_col: str | None = None,
    min_l2_shift: float = 1e-4,
    min_cosine_shift: float = 1e-4,
    refresh_days: int = 90,
) -> Dict[str, np.ndarray]:
    """Build text memory from semantic state changes rather than daily repeats.

    The active benchmark surface carries one embedding row per entity-day, but
    the same semantic text can repeat across many consecutive days. This helper
    treats text as an event source by retaining the first observed text state
    and later rows only when the embedding materially changes.

    Recent token layout:
    [time_since_event_days, inter_event_gap_days, presence_ratio,
     novelty_l2, cosine_shift, payload...]

    Bucket token layout:
    [event_count, min_age, max_age, mean_age, novelty_mean,
     cosine_shift_mean, payload_mean...]
    """
    cfg = cfg or DualClockConfig()
    recent_width = 5 + len(source_cols)
    bucket_width = 6 + len(source_cols)
    if entity_df.empty or not source_cols:
        return {
            "recent_tokens": np.zeros((cfg.max_events, recent_width), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, bucket_width), dtype=np.float32),
        }

    work = entity_df.copy()
    event_col = event_col or cfg.time_col
    work["_event_time"] = _ensure_event_time(work, event_col, cfg.time_col)
    work = work[work["_event_time"].notna()]
    work = work[work["_event_time"] <= prediction_time]
    work = work[_source_presence_mask(work, source_cols)]
    if work.empty:
        return {
            "recent_tokens": np.zeros((cfg.max_events, recent_width), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, bucket_width), dtype=np.float32),
        }

    deduped = _dedupe_text_events(
        work,
        source_cols=source_cols,
        event_col="_event_time",
        fallback_col=cfg.time_col,
        min_l2_shift=float(min_l2_shift),
        min_cosine_shift=float(min_cosine_shift),
        refresh_days=int(refresh_days),
    )
    events = deduped["events"]
    payload = deduped["payload"]
    gap_days = deduped["gap_days"]
    novelty_l2 = deduped["novelty_l2"]
    cosine_shift = deduped["cosine_shift"]
    if len(events) == 0:
        return {
            "recent_tokens": np.zeros((cfg.max_events, recent_width), dtype=np.float32),
            "bucket_tokens": np.zeros((len(cfg.recency_buckets) + 1, bucket_width), dtype=np.float32),
        }

    event_age_days = (
        prediction_time - events["_event_time"]
    ).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    order = np.argsort(event_age_days)
    event_age_days = event_age_days[order]
    payload = payload[order]
    gap_days = gap_days[order]
    novelty_l2 = novelty_l2[order]
    cosine_shift = cosine_shift[order]
    presence_ratio = np.ones_like(event_age_days, dtype=np.float32)

    recent_idx = np.arange(min(cfg.max_events, len(event_age_days)))
    recent_tokens = np.concatenate(
        [
            event_age_days[recent_idx, None].astype(np.float32),
            gap_days[recent_idx, None].astype(np.float32),
            presence_ratio[recent_idx, None].astype(np.float32),
            novelty_l2[recent_idx, None].astype(np.float32),
            cosine_shift[recent_idx, None].astype(np.float32),
            payload[recent_idx],
        ],
        axis=1,
    )
    if len(recent_tokens) < cfg.max_events:
        pad = np.zeros((cfg.max_events - len(recent_tokens), recent_width), dtype=np.float32)
        recent_tokens = np.vstack([recent_tokens, pad])

    edges = [0.0] + [float(x) for x in cfg.recency_buckets] + [float("inf")]
    bucket_rows: List[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (event_age_days >= left) & (event_age_days <= right)
        if not mask.any():
            bucket_rows.append(np.zeros((bucket_width,), dtype=np.float32))
            continue
        ages = event_age_days[mask].astype(np.float32, copy=False)
        payload_mean = _safe_numeric_mean(payload[mask], dim=0)
        row = np.concatenate(
            [
                np.array(
                    [
                        float(mask.sum()),
                        float(ages.min()),
                        float(ages.max()),
                        float(ages.mean()),
                        float(np.mean(novelty_l2[mask])) if mask.any() else 0.0,
                        float(np.mean(cosine_shift[mask])) if mask.any() else 0.0,
                    ],
                    dtype=np.float32,
                ),
                payload_mean,
            ]
        )
        bucket_rows.append(row.astype(np.float32, copy=False))

    return {
        "recent_tokens": recent_tokens.astype(np.float32, copy=False),
        "bucket_tokens": np.vstack(bucket_rows),
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
    "build_source_native_edgar_memory",
    "build_source_native_text_memory",
    "infer_text_columns",
    "infer_edgar_columns",
    "infer_numeric_source_columns",
    "KNOWN_EDGAR_COLUMNS",
]
