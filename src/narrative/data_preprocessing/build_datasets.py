from __future__ import annotations

import numpy as np
import pandas as pd


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "funding_goal_usd" in df.columns and "funding_raised_usd" in df.columns:
        goal = pd.to_numeric(df["funding_goal_usd"], errors="coerce")
        raised = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["funding_ratio"] = raised / goal
    return df


def build_offers_core(df: pd.DataFrame) -> pd.DataFrame:
    return df


def add_time_and_peer_group(df: pd.DataFrame) -> pd.DataFrame:
    return df


def filter_modelling_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df
