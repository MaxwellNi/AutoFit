from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd


def add_time_index(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("platform_name", "offer_id"),
    time_col: str = "crawled_date",
) -> pd.DataFrame:
    """
    Add irregular event index + time deltas per entity.
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not in df.columns")
    for k in key_cols:
        if k not in df.columns:
            raise KeyError(f"key col '{k}' not in df.columns")

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[time_col])
    d = d.sort_values(list(key_cols) + [time_col], kind="mergesort")

    d["event_idx"] = d.groupby(list(key_cols), sort=False).cumcount()
    d["time_delta_days"] = (
        d.groupby(list(key_cols), sort=False)[time_col]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
    )
    d["time_since_start_days"] = (
        d[time_col]
        - d.groupby(list(key_cols), sort=False)[time_col].transform("min")
    ).dt.total_seconds() / 86400.0
    return d


def build_cutoff_mask(
    df: pd.DataFrame,
    *,
    time_col: str = "crawled_date",
    cutoff_time: Optional[str] = None,
) -> pd.Series:
    if cutoff_time is None:
        return pd.Series([True] * len(df), index=df.index)
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not in df.columns")
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    cut = pd.to_datetime(cutoff_time, utc=True)
    return t <= cut


def apply_snapshot_cutoff(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str] = ("platform_name", "offer_id"),
    time_col: str = "crawled_date",
    max_snapshots: Optional[int] = None,
    cutoff_time: Optional[str] = None,
    use_first_k: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply cutoff by time and/or by max snapshots per entity.
    Returns (filtered_df, cutoff_mask).
    """
    d = add_time_index(df, key_cols=key_cols, time_col=time_col)
    mask = build_cutoff_mask(d, time_col=time_col, cutoff_time=cutoff_time)
    d = d[mask].copy()

    if max_snapshots is not None:
        d = d.sort_values(list(key_cols) + [time_col], kind="mergesort")
        if use_first_k:
            d = d.groupby(list(key_cols), sort=False).head(int(max_snapshots))
        else:
            d = d.groupby(list(key_cols), sort=False).tail(int(max_snapshots))

    return d, mask


__all__ = [
    "add_time_index",
    "build_cutoff_mask",
    "apply_snapshot_cutoff",
]
