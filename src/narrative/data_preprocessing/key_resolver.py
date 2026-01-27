from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd


def _make_compound_key(df: pd.DataFrame, cols: Sequence[str], sep: str = "||") -> pd.Series:
    parts = []
    for c in cols:
        if c not in df.columns:
            parts.append(pd.Series([""], index=df.index))
        else:
            parts.append(df[c].astype(str).fillna(""))
    out = parts[0]
    for p in parts[1:]:
        out = out + sep + p
    return out


def resolve_entity_id(
    df: pd.DataFrame,
    *,
    offer_key_cols: Sequence[str] = ("platform_name", "offer_id"),
    cik_col: str = "cik",
    prefer_cik: bool = True,
) -> pd.Series:
    """
    Entity ID protocol:
      1) prefer cik when present
      2) fallback to platform_name||offer_id
      3) fallback to offer_id only
    """
    if prefer_cik and cik_col in df.columns:
        cik = df[cik_col].astype(str).fillna("")
        has_cik = cik.str.len() > 0
    else:
        cik = pd.Series([""] * len(df), index=df.index)
        has_cik = pd.Series([False] * len(df), index=df.index)

    offer_key = _make_compound_key(df, offer_key_cols)
    if offer_key_cols[-1] in df.columns:
        offer_id = df[offer_key_cols[-1]].astype(str).fillna("")
    else:
        offer_id = pd.Series([""] * len(df), index=df.index)

    entity = pd.Series([""] * len(df), index=df.index)
    entity[has_cik] = cik[has_cik]
    entity[~has_cik] = offer_key[~has_cik]
    entity = entity.where(entity.str.len() > 0, offer_id)
    return entity


def build_crosswalk(
    offers_df: pd.DataFrame,
    edgar_df: Optional[pd.DataFrame] = None,
    *,
    offer_key_cols: Sequence[str] = ("platform_name", "offer_id"),
    cik_col: str = "cik",
) -> pd.DataFrame:
    """
    Build crosswalk between offers and EDGAR using entity_id.
    """
    offers = offers_df.copy()
    offers["entity_id"] = resolve_entity_id(offers, offer_key_cols=offer_key_cols, cik_col=cik_col)

    cols = list(offer_key_cols) + ["entity_id"]
    if cik_col in offers.columns:
        cols.append(cik_col)
    offers = offers[cols].drop_duplicates()

    if edgar_df is None or cik_col not in edgar_df.columns:
        return offers.reset_index(drop=True)

    edgar = edgar_df.copy()
    edgar[cik_col] = edgar[cik_col].astype(str)
    edgar = edgar[[cik_col]].drop_duplicates()

    crosswalk = offers.merge(edgar, on=cik_col, how="left", suffixes=("", "_edgar"))
    return crosswalk.reset_index(drop=True)


@dataclass
class UniquenessCheck:
    key_cols: List[str]
    n_rows: int
    n_duplicates: int


def validate_unique_keys(df: pd.DataFrame, key_cols: Sequence[str]) -> UniquenessCheck:
    if not all(c in df.columns for c in key_cols):
        missing = [c for c in key_cols if c not in df.columns]
        raise KeyError(f"Missing key cols: {missing}")
    n_dup = int(df.duplicated(subset=list(key_cols)).sum())
    return UniquenessCheck(key_cols=list(key_cols), n_rows=int(len(df)), n_duplicates=n_dup)


def assert_unique_keys(df: pd.DataFrame, key_cols: Sequence[str]) -> None:
    check = validate_unique_keys(df, key_cols)
    if check.n_duplicates > 0:
        raise ValueError(f"Duplicate keys detected: {check.key_cols} (n={check.n_duplicates})")


__all__ = [
    "resolve_entity_id",
    "build_crosswalk",
    "validate_unique_keys",
    "assert_unique_keys",
]
