from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import pandas as pd


def _pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require_cols(df: pd.DataFrame, cols, dataset: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset} missing columns: {missing}")


def normalize_kickstarter(df: pd.DataFrame) -> pd.DataFrame:
    col_id = _pick_col(df, ["ID", "id"])
    col_start = _pick_col(df, ["launched", "start_date"])
    col_end = _pick_col(df, ["deadline", "end_date"])
    col_goal = _pick_col(df, ["usd_goal_real", "goal"])
    col_raised = _pick_col(df, ["usd_pledged_real", "usd_pledged", "pledged"])
    col_backers = _pick_col(df, ["backers"])
    col_category = _pick_col(df, ["main_category", "category"])
    col_country = _pick_col(df, ["country"])
    col_state = _pick_col(df, ["state"])
    col_text = _pick_col(df, ["name", "title"])

    _require_cols(
        df,
        [c for c in [col_id, col_start, col_end, col_goal, col_raised, col_backers, col_category, col_country] if c],
        "kickstarter",
    )

    out = pd.DataFrame(
        {
            "project_id": df[col_id].astype(str),
            "platform": "kickstarter",
            "start_date": pd.to_datetime(df[col_start], errors="coerce", utc=True),
            "end_date": pd.to_datetime(df[col_end], errors="coerce", utc=True),
            "funding_goal": pd.to_numeric(df[col_goal], errors="coerce"),
            "funding_raised": pd.to_numeric(df[col_raised], errors="coerce"),
            "backers": pd.to_numeric(df[col_backers], errors="coerce"),
            "category": df[col_category].astype(str),
            "country": df[col_country].astype(str),
            "state": df[col_state].astype(str) if col_state else "unknown",
            "text": df[col_text].astype(str) if col_text else "",
        }
    )
    return out


def normalize_kiva(df: pd.DataFrame) -> pd.DataFrame:
    col_id = _pick_col(df, ["id", "loan_id"])
    col_start = _pick_col(df, ["posted_time", "posted_date"])
    col_end = _pick_col(df, ["funded_time", "paid_time"])
    col_goal = _pick_col(df, ["loan_amount", "amount"])
    col_raised = _pick_col(df, ["funded_amount", "funded"])
    col_backers = _pick_col(df, ["lenders_total", "backers"])
    col_category = _pick_col(df, ["sector", "activity"])
    col_country = _pick_col(df, ["country", "country_code"])
    col_state = _pick_col(df, ["status"])
    col_text = _pick_col(df, ["name", "description"])

    _require_cols(
        df,
        [c for c in [col_id, col_start, col_goal, col_raised, col_category, col_country] if c],
        "kiva",
    )

    out = pd.DataFrame(
        {
            "project_id": df[col_id].astype(str),
            "platform": "kiva",
            "start_date": pd.to_datetime(df[col_start], errors="coerce", utc=True),
            "end_date": pd.to_datetime(df[col_end], errors="coerce", utc=True) if col_end else pd.NaT,
            "funding_goal": pd.to_numeric(df[col_goal], errors="coerce"),
            "funding_raised": pd.to_numeric(df[col_raised], errors="coerce"),
            "backers": pd.to_numeric(df[col_backers], errors="coerce") if col_backers else 0,
            "category": df[col_category].astype(str),
            "country": df[col_country].astype(str),
            "state": df[col_state].astype(str) if col_state else "unknown",
            "text": df[col_text].astype(str) if col_text else "",
        }
    )
    return out


def normalize_gofundme(df: pd.DataFrame) -> pd.DataFrame:
    col_id = _pick_col(df, ["id", "campaign_id"])
    col_start = _pick_col(df, ["created", "start_date"])
    col_end = _pick_col(df, ["updated", "end_date"])
    col_goal = _pick_col(df, ["goal", "funding_goal"])
    col_raised = _pick_col(df, ["raised", "funding_raised"])
    col_backers = _pick_col(df, ["donors", "backers"])
    col_category = _pick_col(df, ["category"])
    col_country = _pick_col(df, ["country"])
    col_state = _pick_col(df, ["state", "status"])
    col_text = _pick_col(df, ["title", "description"])

    _require_cols(
        df,
        [c for c in [col_id, col_start, col_goal, col_raised, col_category, col_country] if c],
        "gofundme",
    )

    out = pd.DataFrame(
        {
            "project_id": df[col_id].astype(str),
            "platform": "gofundme",
            "start_date": pd.to_datetime(df[col_start], errors="coerce", utc=True),
            "end_date": pd.to_datetime(df[col_end], errors="coerce", utc=True) if col_end else pd.NaT,
            "funding_goal": pd.to_numeric(df[col_goal], errors="coerce"),
            "funding_raised": pd.to_numeric(df[col_raised], errors="coerce"),
            "backers": pd.to_numeric(df[col_backers], errors="coerce") if col_backers else 0,
            "category": df[col_category].astype(str),
            "country": df[col_country].astype(str),
            "state": df[col_state].astype(str) if col_state else "unknown",
            "text": df[col_text].astype(str) if col_text else "",
        }
    )
    return out


def normalize_external_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    dataset = dataset.lower()
    if dataset == "kickstarter":
        return normalize_kickstarter(df)
    if dataset == "kiva":
        return normalize_kiva(df)
    if dataset == "gofundme":
        return normalize_gofundme(df)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_external_dataset(
    input_path: Path,
    *,
    dataset: str,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    if input_path.suffix.lower() != ".parquet":
        raise ValueError("Unsupported file type (parquet only)")
    df = pd.read_parquet(input_path)
    if limit_rows is not None:
        df = df.head(int(limit_rows))
    return normalize_external_dataset(df, dataset)


def save_external_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


__all__ = [
    "load_external_dataset",
    "save_external_parquet",
    "normalize_kickstarter",
    "normalize_kiva",
    "normalize_gofundme",
]
