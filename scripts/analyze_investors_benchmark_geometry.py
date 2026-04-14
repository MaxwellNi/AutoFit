#!/usr/bin/env python3
"""Analyze investors benchmark geometry, raw entity dynamics, and champion distribution."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_investors_source_ablation import _load_case_frames, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _load_pointer, _resolve_repo_relative
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from src.narrative.block3.models.single_model_mainline.lanes.investors_lane import _source_profile_ids
from src.narrative.block3.unified_protocol import load_block3_tasks_config


PROFILE_NAMES = {0: "none", 1: "edgar_only", 2: "text_only", 3: "mixed"}


@dataclass(frozen=True)
class GeometryCase:
    task: str
    ablation: str
    horizon: int
    target: str = "investors_count"
    max_entities: int = 64
    max_rows: int = 0

    @property
    def name(self) -> str:
        return f"{self.task}__{self.ablation}__h{self.horizon}"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sections",
        nargs="+",
        choices=("official", "raw", "champions"),
        default=("official", "raw", "champions"),
        help="Subset of report sections to run.",
    )
    ap.add_argument(
        "--sample-entities",
        type=int,
        default=64,
        help="Coverage-ranked entities sampled per official slice for source/profile geometry.",
    )
    ap.add_argument(
        "--sample-train-rows",
        type=int,
        default=0,
        help="Optional train-row downsample per official slice; 0 disables train downsample.",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on the number of official cases analyzed; 0 means all cases.",
    )
    ap.add_argument(
        "--top-k-models",
        type=int,
        default=10,
        help="How many top models to retain in champion summaries.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report.",
    )
    return ap.parse_args()


def _profile_counts(profile_ids: np.ndarray) -> Dict[str, int]:
    return {
        PROFILE_NAMES[int(profile_id)]: int(np.sum(profile_ids == profile_id))
        for profile_id in np.unique(profile_ids)
    }


def _dominant_profile(profile_counts: Dict[str, int]) -> Dict[str, Any]:
    if not profile_counts:
        return {"name": None, "share": 0.0}
    name, count = max(profile_counts.items(), key=lambda item: item[1])
    total = sum(profile_counts.values())
    return {
        "name": name,
        "share": float(count / total) if total else 0.0,
    }


def _build_source_profile_report(model: SingleModelMainlineWrapper, frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    X = frame.drop(columns=[target]) if target in frame.columns else frame
    runtime_frame = model._prepare_runtime_frame(X, raw_frame=frame)
    source_layout = model.source_memory.infer_layout(runtime_frame)
    source_frame = model.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
    source = model._build_investors_source_features(source_frame).to_numpy(dtype=np.float32, copy=False)
    profile_ids = _source_profile_ids(source)
    counts = _profile_counts(profile_ids)
    dominant = _dominant_profile(counts)
    return {
        "rows": int(len(profile_ids)),
        "profile_counts": counts,
        "dominant_profile": dominant["name"],
        "dominant_profile_share": dominant["share"],
    }


def _target_stats(frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    values = frame[target].dropna().to_numpy(dtype=np.float64, copy=False)
    if len(values) == 0:
        return {
            "rows": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "zero_rate": None,
        }
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "zero_rate": float(np.mean(values == 0.0)),
    }


def _dynamic_entity_stats(frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    valid = frame[["entity_id", target]].dropna().copy()
    if valid.empty:
        return {
            "entities": 0,
            "dynamic_entities": 0,
            "dynamic_entity_ratio": 0.0,
            "dynamic_rows": 0,
            "dynamic_row_ratio": 0.0,
            "range_mean": 0.0,
            "range_p90": 0.0,
            "range_max": 0.0,
        }
    grouped = valid.groupby("entity_id", sort=False)[target]
    entity_count = int(grouped.ngroups)
    ranges = grouped.max() - grouped.min()
    dynamic_ranges = ranges[ranges > 0.0]
    dynamic_entities = set(dynamic_ranges.index.astype(str).tolist())
    dynamic_rows = int(valid[valid["entity_id"].astype(str).isin(dynamic_entities)].shape[0])
    return {
        "entities": entity_count,
        "dynamic_entities": int(len(dynamic_entities)),
        "dynamic_entity_ratio": float(len(dynamic_entities) / entity_count) if entity_count else 0.0,
        "dynamic_rows": dynamic_rows,
        "dynamic_row_ratio": float(dynamic_rows / len(valid)) if len(valid) else 0.0,
        "range_mean": float(dynamic_ranges.mean()) if len(dynamic_ranges) else 0.0,
        "range_p90": float(dynamic_ranges.quantile(0.90)) if len(dynamic_ranges) else 0.0,
        "range_max": float(dynamic_ranges.max()) if len(dynamic_ranges) else 0.0,
    }


def _official_cases(sample_entities: int, sample_train_rows: int) -> list[GeometryCase]:
    cfg = load_block3_tasks_config().get("tasks", {})
    cases: list[GeometryCase] = []
    for task, task_cfg in cfg.items():
        targets = task_cfg.get("targets", []) or []
        if "investors_count" not in targets:
            continue
        for ablation in task_cfg.get("ablations", []) or []:
            for horizon in task_cfg.get("horizons", []) or []:
                cases.append(
                    GeometryCase(
                        task=task,
                        ablation=ablation,
                        horizon=int(horizon),
                        max_entities=sample_entities,
                        max_rows=sample_train_rows,
                    )
                )
    cases.sort(key=lambda case: (case.task, case.ablation, case.horizon))
    return cases


def _analyze_official_sample_geometry(sample_entities: int, sample_train_rows: int, max_cases: int) -> Dict[str, Any]:
    temporal_config = _make_temporal_config()
    rows: list[Dict[str, Any]] = []
    train_homogeneous = 0
    test_homogeneous = 0

    cases = _official_cases(sample_entities, sample_train_rows)
    if max_cases > 0:
        cases = cases[:max_cases]

    for case in cases:
        train, _, test = _load_case_frames(case, temporal_config)
        wrapper = SingleModelMainlineWrapper()
        train_profiles = _build_source_profile_report(wrapper, train, case.target)
        test_profiles = _build_source_profile_report(wrapper, test, case.target)
        train_dynamic = _dynamic_entity_stats(train, case.target)
        test_dynamic = _dynamic_entity_stats(test, case.target)

        if train_profiles["dominant_profile_share"] >= 0.999:
            train_homogeneous += 1
        if test_profiles["dominant_profile_share"] >= 0.999:
            test_homogeneous += 1

        rows.append(
            {
                "case": case.name,
                "task": case.task,
                "ablation": case.ablation,
                "horizon": case.horizon,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_entities": int(train["entity_id"].astype(str).nunique()),
                "test_entities": int(test["entity_id"].astype(str).nunique()),
                "train_profiles": train_profiles,
                "test_profiles": test_profiles,
                "train_dynamic": train_dynamic,
                "test_dynamic": test_dynamic,
                "train_target_stats": _target_stats(train, case.target),
                "test_target_stats": _target_stats(test, case.target),
            }
        )

    return {
        "sample_entities": sample_entities,
        "sample_train_rows": sample_train_rows,
        "cases": rows,
        "summary": {
            "n_cases": len(rows),
            "train_homogeneous_cases": train_homogeneous,
            "test_homogeneous_cases": test_homogeneous,
            "train_homogeneous_share": float(train_homogeneous / len(rows)) if rows else 0.0,
            "test_homogeneous_share": float(test_homogeneous / len(rows)) if rows else 0.0,
        },
    }


def _raw_entity_pool_summary() -> Dict[str, Any]:
    pointer = _load_pointer()
    core_path = _resolve_repo_relative(pointer["offers_core_daily"]["dir"]) / "offers_core_daily.parquet"
    temporal_config = _make_temporal_config()
    train_end = pd.Timestamp(temporal_config.train_end)
    test_start = pd.Timestamp(temporal_config.val_end) + pd.Timedelta(days=temporal_config.embargo_days)
    test_end = pd.Timestamp(temporal_config.test_end)

    stats: Dict[str, Dict[str, Any]] = {}
    parquet = pq.ParquetFile(str(core_path))
    columns = ["entity_id", "crawled_date_day", "investors_count", "cik"]
    for batch in parquet.iter_batches(columns=columns, batch_size=65536):
        batch_df = batch.to_pandas()
        batch_df["crawled_date_day"] = pd.to_datetime(batch_df["crawled_date_day"], utc=True, errors="coerce")
        batch_df["crawled_date_day"] = batch_df["crawled_date_day"].dt.tz_convert(None)
        for row in batch_df.itertuples(index=False):
            entity_id = str(row.entity_id)
            payload = stats.setdefault(
                entity_id,
                {
                    "train_count": 0,
                    "test_count": 0,
                    "test_min": None,
                    "test_max": None,
                    "has_cik": False,
                },
            )
            timestamp = getattr(row, "crawled_date_day")
            value = getattr(row, "investors_count")
            if pd.notna(getattr(row, "cik", None)):
                payload["has_cik"] = True
            if pd.isna(timestamp) or pd.isna(value):
                continue
            numeric_value = float(value)
            if timestamp <= train_end:
                payload["train_count"] += 1
            elif test_start < timestamp <= test_end:
                payload["test_count"] += 1
                payload["test_min"] = numeric_value if payload["test_min"] is None else min(payload["test_min"], numeric_value)
                payload["test_max"] = numeric_value if payload["test_max"] is None else max(payload["test_max"], numeric_value)

    qualified: list[Dict[str, Any]] = []
    qualified_with_cik: list[Dict[str, Any]] = []
    for entity_id, payload in stats.items():
        if payload["train_count"] <= 0 or payload["test_count"] <= 0:
            continue
        dynamic = (
            payload["test_min"] is not None
            and payload["test_max"] is not None
            and payload["test_max"] > payload["test_min"]
        )
        record = {
            "entity_id": entity_id,
            "train_count": int(payload["train_count"]),
            "test_count": int(payload["test_count"]),
            "total_count": int(payload["train_count"] + payload["test_count"]),
            "has_cik": bool(payload["has_cik"]),
            "dynamic_test": bool(dynamic),
            "test_range": float((payload["test_max"] or 0.0) - (payload["test_min"] or 0.0)) if dynamic else 0.0,
        }
        qualified.append(record)
        if record["has_cik"]:
            qualified_with_cik.append(record)

    qualified.sort(key=lambda item: (-item["total_count"], item["entity_id"]))
    qualified_with_cik.sort(key=lambda item: (-item["total_count"], item["entity_id"]))

    def _coverage_bias(records: list[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in (4, 8, 16, 32, 64, 128):
            top = records[:k]
            if not top:
                continue
            out[str(k)] = {
                "dynamic_entities": int(sum(1 for item in top if item["dynamic_test"])),
                "dynamic_share": float(sum(1 for item in top if item["dynamic_test"]) / len(top)),
                "mean_test_range": float(np.mean([item["test_range"] for item in top])) if top else 0.0,
            }
        return out

    dynamic_all = [item for item in qualified if item["dynamic_test"]]
    dynamic_cik = [item for item in qualified_with_cik if item["dynamic_test"]]
    dynamic_all.sort(key=lambda item: (-item["total_count"], item["entity_id"]))
    dynamic_cik.sort(key=lambda item: (-item["total_count"], item["entity_id"]))

    return {
        "qualified_entities": int(len(qualified)),
        "qualified_entities_with_cik": int(len(qualified_with_cik)),
        "dynamic_entities": int(len(dynamic_all)),
        "dynamic_entities_with_cik": int(len(dynamic_cik)),
        "dynamic_entity_share": float(len(dynamic_all) / len(qualified)) if qualified else 0.0,
        "dynamic_entity_share_with_cik": float(len(dynamic_cik) / len(qualified_with_cik)) if qualified_with_cik else 0.0,
        "top_coverage_bias_all": _coverage_bias(qualified),
        "top_coverage_bias_with_cik": _coverage_bias(qualified_with_cik),
        "top_dynamic_entities_with_cik": dynamic_cik[:10],
    }


def _benchmark_champion_summary(top_k_models: int) -> Dict[str, Any]:
    benchmark_path = REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv"
    df = pd.read_csv(benchmark_path)
    df = df[(df["split"] == "test") & (df["target"] == "investors_count") & (df["fairness_pass"] == True)].copy()
    df = df[df["mae"].notna()].copy()

    group_cols = ["task", "ablation", "horizon", "target"]
    winner_idx = df.groupby(group_cols)["mae"].idxmin()
    winners = df.loc[winner_idx, ["task", "ablation", "horizon", "model_name", "category", "mae"]].copy()
    winners = winners.sort_values(["task", "ablation", "horizon", "mae", "model_name"], kind="mergesort")

    df["mae_rank"] = df.groupby(group_cols)["mae"].rank(method="average")
    mean_rank = df.groupby("model_name")["mae_rank"].mean().sort_values()

    return {
        "n_materialized_rows": int(len(df)),
        "n_conditions": int(len(winners)),
        "winner_counts_by_model": {
            str(k): int(v)
            for k, v in winners["model_name"].value_counts().head(top_k_models).items()
        },
        "winner_counts_by_category": {
            str(k): int(v) for k, v in winners["category"].value_counts().items()
        },
        "mean_rank_top_models": {
            str(k): float(v) for k, v in mean_rank.head(top_k_models).items()
        },
        "condition_winners": winners.to_dict(orient="records"),
    }


def _build_report(args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    sections = set(args.sections)
    if "official" in sections:
        report["official_sample_geometry"] = _analyze_official_sample_geometry(
            sample_entities=args.sample_entities,
            sample_train_rows=args.sample_train_rows,
            max_cases=args.max_cases,
        )
    if "raw" in sections:
        report["raw_entity_pool"] = _raw_entity_pool_summary()
    if "champions" in sections:
        report["benchmark_champions"] = _benchmark_champion_summary(top_k_models=args.top_k_models)
    return report


def _print_console_summary(report: Dict[str, Any]) -> None:
    if "official_sample_geometry" in report:
        official = report["official_sample_geometry"]
        print("official_sample_geometry")
        print(json.dumps(official["summary"], ensure_ascii=False, indent=2))
    if "raw_entity_pool" in report:
        raw = report["raw_entity_pool"]
        print("\nraw_entity_pool")
        print(
            json.dumps(
                {
                    "qualified_entities": raw["qualified_entities"],
                    "qualified_entities_with_cik": raw["qualified_entities_with_cik"],
                    "dynamic_entities": raw["dynamic_entities"],
                    "dynamic_entities_with_cik": raw["dynamic_entities_with_cik"],
                    "dynamic_entity_share": raw["dynamic_entity_share"],
                    "dynamic_entity_share_with_cik": raw["dynamic_entity_share_with_cik"],
                    "top_coverage_bias_with_cik": raw["top_coverage_bias_with_cik"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if "benchmark_champions" in report:
        champs = report["benchmark_champions"]
        print("\nbenchmark_champions")
        print(
            json.dumps(
                {
                    "n_conditions": champs["n_conditions"],
                    "winner_counts_by_model": champs["winner_counts_by_model"],
                    "winner_counts_by_category": champs["winner_counts_by_category"],
                    "mean_rank_top_models": champs["mean_rank_top_models"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    args = _parse_args()
    report = _build_report(args)
    _print_console_summary(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()