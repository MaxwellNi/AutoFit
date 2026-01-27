from __future__ import annotations

from typing import Iterable

import pandas as pd


def make_offer_key(df: pd.DataFrame, id_cols: Iterable[str]) -> pd.Series:
    cols = [c for c in id_cols if c in df.columns]
    if not cols:
        return pd.Series(["unknown"] * len(df), index=df.index)
    key = df[cols].astype(str).agg("|".join, axis=1)
    return key


def normalize_offers_schema(df: pd.DataFrame) -> pd.DataFrame:
    return df
