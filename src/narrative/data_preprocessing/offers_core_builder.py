from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from narrative.data_preprocessing.load_data import make_offer_key
from narrative.data_preprocessing.timeline import add_time_index

SNAPSHOT_TIME_CANDIDATES = (
    "crawled_date",
    "snapshot_date",
    "crawled_date_day",
)

STATIC_COLS_DEFAULT = (
    "platform_name",
    "offer_id",
    "hash_id",
    "link",
    "cik",
    "platform_country",
    "market_tags",
    "security_type",
    "investment_type",
    "regulation",
    "datetime_open_offering",
    "datetime_close_offering",
    "date_start_final",
    "date_end_final",
)

CUTOFF_START_COLS = (
    "datetime_open_offering",
    "date_start_final",
    "open_dt",
)

CUTOFF_END_COLS = (
    "datetime_close_offering",
    "date_end_final",
    "close_dt",
)


def infer_snapshot_time_col(
    columns: Sequence[str],
    *,
    candidates: Sequence[str] = SNAPSHOT_TIME_CANDIDATES,
) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def build_static_table(
    snapshots_df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    snapshot_time_col: str,
    static_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if snapshots_df.empty:
        base_cols = list(id_cols)
        keep = list(dict.fromkeys(base_cols + list(static_cols or [])))
        return pd.DataFrame(columns=keep + ["static_snapshot_ts"])

    static_cols = static_cols or STATIC_COLS_DEFAULT
    keep_cols = [c for c in static_cols if c in snapshots_df.columns]
    keep = list(dict.fromkeys(list(id_cols) + keep_cols + [snapshot_time_col]))

    d = snapshots_df[keep].copy()
    d[snapshot_time_col] = pd.to_datetime(d[snapshot_time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[snapshot_time_col])
    d = d.sort_values(list(id_cols) + [snapshot_time_col], kind="mergesort")
    d = d.drop_duplicates(subset=list(id_cols), keep="last").reset_index(drop=True)
    d = d.rename(columns={snapshot_time_col: "static_snapshot_ts"})
    d["entity_id"] = make_offer_key(d, id_cols)
    return d


def build_offers_core_from_snapshots(
    snapshots_df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    snapshot_time_col: str,
    static_cols: Optional[Sequence[str]] = None,
    cutoff_time: Optional[str] = None,
    cutoff_mode: str = "start",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    if snapshot_time_col not in snapshots_df.columns:
        raise KeyError(f"snapshot_time_col '{snapshot_time_col}' not found in snapshots_df")
    if cutoff_mode not in {"start", "end"}:
        raise ValueError("cutoff_mode must be 'start' or 'end'")

    static_cols = static_cols or STATIC_COLS_DEFAULT
    static_cols = [c for c in static_cols if c in snapshots_df.columns]

    static_df = build_static_table(
        snapshots_df,
        id_cols=id_cols,
        snapshot_time_col=snapshot_time_col,
        static_cols=static_cols,
    )

    snap_cols = [
        c for c in snapshots_df.columns
        if c not in static_cols or c in id_cols or c == snapshot_time_col
    ]
    snaps = snapshots_df[snap_cols].copy()
    snaps["entity_id"] = make_offer_key(snaps, id_cols)
    snaps["snapshot_ts"] = pd.to_datetime(snaps[snapshot_time_col], errors="coerce", utc=True)
    snaps = snaps.dropna(subset=["snapshot_ts"])

    snaps = add_time_index(snaps, key_cols=id_cols, time_col="snapshot_ts")
    snaps = snaps.rename(columns={"event_idx": "t_index"})

    static_merge = static_df.drop(columns=["entity_id"], errors="ignore")
    core = snaps.merge(static_merge, on=list(id_cols), how="left", validate="m:1")

    cutoff_source: Optional[str] = None
    cutoff_missing_raw = 0
    cutoff_adjusted = 0
    cutoff_raw_violations = 0
    if cutoff_time:
        cutoff_raw = pd.to_datetime(cutoff_time, errors="coerce", utc=True)
        core["cutoff_ts"] = cutoff_raw
        cutoff_source = "global"
    else:
        if cutoff_mode == "start":
            candidates = CUTOFF_START_COLS
        else:
            candidates = CUTOFF_END_COLS
        for c in candidates:
            if c in core.columns:
                core["cutoff_ts"] = pd.to_datetime(core[c], errors="coerce", utc=True)
                cutoff_source = c
                break
        if cutoff_source is None:
            core["cutoff_ts"] = pd.NaT

    if cutoff_source is None:
        core["cutoff_mask"] = True
    else:
        cutoff_ts = pd.to_datetime(core["cutoff_ts"], errors="coerce", utc=True)
        cutoff_missing_raw = int(cutoff_ts.isna().sum())
        if cutoff_mode == "start":
            cutoff_raw_violations = int((cutoff_ts > core["snapshot_ts"]).sum())
            cutoff_ts = cutoff_ts.where(cutoff_ts <= core["snapshot_ts"], core["snapshot_ts"])
        else:
            cutoff_raw_violations = int((core["snapshot_ts"] > cutoff_ts).sum())
            cutoff_ts = cutoff_ts.where(core["snapshot_ts"] <= cutoff_ts, core["snapshot_ts"])
        cutoff_adjusted = int((cutoff_ts != core["cutoff_ts"]).sum())
        cutoff_ts = cutoff_ts.fillna(core["snapshot_ts"])
        core["cutoff_ts"] = cutoff_ts
        if cutoff_mode == "start":
            mask = core["snapshot_ts"] >= core["cutoff_ts"]
        else:
            mask = core["snapshot_ts"] <= core["cutoff_ts"]
        core["cutoff_mask"] = mask | core["cutoff_ts"].isna()

    info = {
        "snapshot_time_col": snapshot_time_col,
        "cutoff_source": cutoff_source,
        "cutoff_mode": cutoff_mode,
        "cutoff_missing_raw": cutoff_missing_raw,
        "cutoff_adjusted": cutoff_adjusted,
        "cutoff_raw_violations": cutoff_raw_violations,
    }
    return core, static_df, info


__all__ = [
    "SNAPSHOT_TIME_CANDIDATES",
    "STATIC_COLS_DEFAULT",
    "CUTOFF_START_COLS",
    "CUTOFF_END_COLS",
    "infer_snapshot_time_col",
    "build_static_table",
    "build_offers_core_from_snapshots",
]
