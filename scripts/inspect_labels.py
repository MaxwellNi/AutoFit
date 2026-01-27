#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.build_datasets import add_outcomes  # noqa: E402


def _setup_logger(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inspect_labels")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "inspect_labels.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _make_entity_id(df: pd.DataFrame) -> pd.Series:
    return df["platform_name"].astype(str).fillna("NA") + "||" + df["offer_id"].astype(str).fillna("NA")


def _goal_stats(goal: pd.Series) -> Dict[str, Any]:
    goal_num = pd.to_numeric(goal, errors="coerce")
    non_null_ratio = float(goal_num.notna().mean())
    gt0_ratio = float((goal_num > 0).mean())
    quantiles = goal_num.dropna().quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "non_null_ratio": non_null_ratio,
        "gt0_ratio": gt0_ratio,
        "quantiles": {str(k): float(v) for k, v in quantiles.items()},
    }


def _raised_stats(raised: pd.Series) -> Dict[str, Any]:
    raised_num = pd.to_numeric(raised, errors="coerce")
    stats = raised_num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {k: float(v) if pd.notna(v) else float("nan") for k, v in stats.items()}


def _ratio_stats(ratio: pd.Series) -> Dict[str, Any]:
    ratio_num = pd.to_numeric(ratio, errors="coerce")
    non_null_ratio = float(ratio_num.notna().mean())
    unique = int(ratio_num.nunique(dropna=True))
    stats = ratio_num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "non_null_ratio": non_null_ratio,
        "unique": unique,
        "stats": {k: float(v) if pd.notna(v) else float("nan") for k, v in stats.items()},
    }


def _sample_head(df: pd.DataFrame, limit_rows: int | None, limit_entities: int | None) -> pd.DataFrame:
    if limit_entities is not None:
        keep_ids = df["entity_id"].drop_duplicates().head(int(limit_entities)).tolist()
        df = df[df["entity_id"].isin(keep_ids)]
    if limit_rows is not None:
        df = df.head(int(limit_rows))
    return df


def _sample_random_entities(df: pd.DataFrame, limit_rows: int | None, limit_entities: int | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = df["entity_id"].drop_duplicates().tolist()
    if limit_entities is not None:
        n_pick = min(int(limit_entities), len(entity_ids))
        keep_ids = rng.choice(entity_ids, size=n_pick, replace=False).tolist()
        return df[df["entity_id"].isin(keep_ids)]
    if limit_rows is not None:
        counts = df.groupby("entity_id")["entity_id"].size()
        shuffled = rng.permutation(entity_ids)
        keep_ids = []
        total = 0
        for eid in shuffled:
            total += int(counts.get(eid, 0))
            keep_ids.append(eid)
            if total >= int(limit_rows):
                break
        return df[df["entity_id"].isin(keep_ids)]
    return df


def _label_window_stats(df: pd.DataFrame, goal_min: float | None) -> Dict[str, Any]:
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")
    labels = []
    for _, group in df.groupby("entity_id", sort=False):
        mask = group["funding_ratio_w"].notna()
        if goal_min is not None and "funding_goal_usd" in group.columns:
            goal = pd.to_numeric(group["funding_goal_usd"], errors="coerce")
            mask = mask & (goal >= goal_min)
        if not mask.any():
            continue
        labels.append(float(group.loc[mask].iloc[-1]["funding_ratio_w"]))
    if not labels:
        return {"count": 0, "non_null_ratio": 0.0, "unique": 0, "stats": {}}
    arr = np.array(labels, dtype=float)
    stats = pd.Series(arr).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "count": int(len(arr)),
        "non_null_ratio": float(np.isfinite(arr).mean()),
        "unique": int(pd.Series(arr).nunique(dropna=True)),
        "stats": {k: float(v) if pd.notna(v) else float("nan") for k, v in stats.items()},
    }


def _inspect(df: pd.DataFrame, goal_min: float | None) -> Dict[str, Any]:
    df = add_outcomes(df.copy())
    result = {
        "columns_present": {
            "funding_goal_usd": "funding_goal_usd" in df.columns,
            "funding_raised_usd": "funding_raised_usd" in df.columns,
            "funding_ratio_w": "funding_ratio_w" in df.columns,
        },
        "funding_goal_usd": _goal_stats(df.get("funding_goal_usd", pd.Series(dtype=float))),
        "funding_raised_usd": _raised_stats(df.get("funding_raised_usd", pd.Series(dtype=float))),
        "funding_ratio_w": _ratio_stats(df.get("funding_ratio_w", pd.Series(dtype=float))),
        "label_window_ratio": _label_window_stats(df, goal_min),
    }
    return result


def _conclude(head: Dict[str, Any], rand: Dict[str, Any]) -> str:
    head_ratio = head["funding_ratio_w"]["non_null_ratio"]
    rand_ratio = rand["funding_ratio_w"]["non_null_ratio"]
    head_unique = head["funding_ratio_w"]["unique"]
    rand_unique = rand["funding_ratio_w"]["unique"]
    if head_ratio < 0.1 and rand_ratio >= 0.1 and rand_unique >= 5:
        return "退化主要来自采样策略（head 截断导致标签缺失/常数）"
    if head_ratio < 0.1 and rand_ratio < 0.1:
        return "退化可能来自字段缺失/清洗/对齐问题（两种采样均缺失）"
    if head_unique < 5 and rand_unique >= 5:
        return "退化主要来自采样策略（head 截断导致标签离散度不足）"
    return "退化原因不唯一，需要进一步排查字段缺失与采样共同影响"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect label quality for funding_ratio_w.")
    parser.add_argument("--offers_core", type=Path, required=True)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--limit_entities", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--label_goal_min", type=float, default=50.0)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("runs") / f"label_inspect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir)

    df = pd.read_parquet(args.offers_core)
    for col in ("platform_name", "offer_id"):
        if col not in df.columns:
            raise KeyError(f"offers_core missing id column: {col}")
    df["entity_id"] = _make_entity_id(df)
    if "snapshot_ts" in df.columns:
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)

    head_df = _sample_head(df.copy(), args.limit_rows, args.limit_entities)
    rand_df = _sample_random_entities(df.copy(), args.limit_rows, args.limit_entities, args.sample_seed)

    full_stats = _inspect(df.copy(), args.label_goal_min)
    head_stats = _inspect(head_df, args.label_goal_min)
    rand_stats = _inspect(rand_df, args.label_goal_min)

    conclusion = _conclude(head_stats, rand_stats)
    logger.info("Conclusion: %s", conclusion)

    report = {
        "offers_core": str(args.offers_core),
        "limit_rows": args.limit_rows,
        "limit_entities": args.limit_entities,
        "sample_seed": args.sample_seed,
        "label_goal_min": args.label_goal_min,
        "full_dataset": full_stats,
        "strategy_head": head_stats,
        "strategy_random_entities": rand_stats,
        "conclusion": conclusion,
    }

    report_path = output_dir / "label_inspect.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Label inspect report saved to %s", report_path)


if __name__ == "__main__":
    main()
