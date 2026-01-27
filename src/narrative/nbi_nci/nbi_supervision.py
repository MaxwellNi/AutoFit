from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd


def _load_parquets(
    root: Path,
    *,
    text_cols: List[str],
    label_cols: List[str],
    split_col: Optional[str] = None,
    default_split: str = "train",
) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame(columns=["text", "label", "split", "source"])

    files = list(root.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["text", "label", "split", "source"])

    frames = []
    for parquet_path in files:
        df = pd.read_parquet(parquet_path)
        text_col = next((c for c in text_cols if c in df.columns), None)
        label_col = next((c for c in label_cols if c in df.columns), None)
        if text_col is None or label_col is None:
            continue
        out = pd.DataFrame(
            {
                "text": df[text_col].astype(str),
                "label": df[label_col],
                "split": df[split_col] if split_col and split_col in df.columns else default_split,
                "source": root.name,
            }
        )
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["text", "label", "split", "source"])
    return pd.concat(frames, ignore_index=True)


def _parse_label_list(series: pd.Series) -> pd.Series:
    def _parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return [x]
        return [x]

    return series.apply(_parse)


def load_goemotions(root: Path) -> pd.DataFrame:
    """
    Expected columns: text + labels (list or json string), optional split.
    """
    df = _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["labels", "label"],
        split_col="split",
    )
    if not df.empty:
        df["label"] = _parse_label_list(df["label"])
    return df


def load_financial_phrasebank(root: Path) -> pd.DataFrame:
    """
    Expected columns: text/sentence + label/sentiment.
    """
    return _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["label", "sentiment"],
        split_col="split",
    )


def load_media_frames(root: Path) -> pd.DataFrame:
    """
    Expected columns: text + frame (or label).
    """
    return _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["frame", "label"],
        split_col="split",
    )


def load_persuasion_strategies(root: Path) -> pd.DataFrame:
    """
    Expected columns: text + strategy/label.
    """
    return _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["strategy", "label"],
        split_col="split",
    )


def load_commitmentbank(root: Path) -> pd.DataFrame:
    """
    Expected columns: text + label (commitment class).
    """
    return _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["label", "commitment"],
        split_col="split",
    )


def load_bioscope(root: Path) -> pd.DataFrame:
    """
    Expected columns: text + label (certainty/negation).
    """
    return _load_parquets(
        root,
        text_cols=["text", "sentence"],
        label_cols=["label", "scope"],
        split_col="split",
    )


def load_all_supervision(root: Path) -> Dict[str, pd.DataFrame]:
    datasets = {
        "goemotions": load_goemotions(root / "goemotions"),
        "financial_phrasebank": load_financial_phrasebank(root / "financial_phrasebank"),
        "media_frames": load_media_frames(root / "media_frames"),
        "persuasion_strategies": load_persuasion_strategies(root / "persuasion_strategies"),
        "commitmentbank": load_commitmentbank(root / "commitmentbank"),
        "bioscope": load_bioscope(root / "bioscope"),
    }
    return datasets


__all__ = [
    "load_goemotions",
    "load_financial_phrasebank",
    "load_media_frames",
    "load_persuasion_strategies",
    "load_commitmentbank",
    "load_bioscope",
    "load_all_supervision",
]
