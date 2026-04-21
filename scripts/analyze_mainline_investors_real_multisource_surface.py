#!/usr/bin/env python3
"""Research-only real-data cross-profile surface for investors lane studies."""
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

from scripts.analyze_mainline_investors_source_ablation import _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import (
    _compute_metrics,
    _downsample_preserve_time,
    _join_edgar_asof,
    _join_text_embeddings,
    _load_core_slice,
    _load_edgar_slice,
    _load_pointer,
    _load_smoke_frame,
    _load_text_slice,
    _prepare_features,
    _resolve_repo_relative,
)
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from src.narrative.block3.models.single_model_mainline.backbone import SharedTemporalBackbone, SharedTemporalBackboneSpec
from src.narrative.block3.models.single_model_mainline.barrier import TargetIsolatedBarrier
from src.narrative.block3.models.single_model_mainline.conditioning import ConditionKey, MainlineConditionEncoder
from src.narrative.block3.models.single_model_mainline.lanes.investors_lane import (
    InvestorsLaneRuntime,
    _source_profile_ids,
    _transition_signal,
)
from src.narrative.block3.models.single_model_mainline.source_memory import SourceMemoryAssembler
from src.narrative.block3.unified_protocol import apply_temporal_split


PROFILE_NAMES = {0: "none", 1: "edgar_only", 2: "text_only", 3: "mixed"}


@dataclass(frozen=True)
class RealSurfaceCase:
    task: str = "task2_forecast"
    target: str = "investors_count"
    horizon: int = 1
    ablations: tuple[str, ...] = ("core_only", "core_edgar", "core_text", "full")
    max_entities: int = 16
    entity_selection: str = "common_coverage"
    dynamic_entity_limit: int = 4
    max_rows_per_ablation: int = 1000


VARIANTS: Dict[str, Dict[str, Dict[str, bool]]] = {
    "baseline": {
        "fit": {},
        "predict": {},
    },
    "forced_source_features": {
        "fit": {"enable_source_features": True},
        "predict": {"enable_source_features": True},
    },
    "forced_source_features_plus_guard": {
        "fit": {"enable_source_features": True},
        "predict": {"enable_source_features": True, "enable_source_guard": True},
    },
    "source_read_policy": {
        "fit": {"enable_source_read_policy": True},
        "predict": {"enable_source_read_policy": True},
    },
    "source_read_policy_plus_transition": {
        "fit": {
            "enable_source_read_policy": True,
            "enable_source_transition_correction": True,
        },
        "predict": {
            "enable_source_read_policy": True,
            "enable_source_transition_correction": True,
        },
    },
    "source_read_policy_plus_guard": {
        "fit": {"enable_source_read_policy": True, "enable_source_guard": True},
        "predict": {"enable_source_read_policy": True, "enable_source_guard": True},
    },
    "source_read_policy_plus_transition_plus_guard": {
        "fit": {
            "enable_source_read_policy": True,
            "enable_source_guard": True,
            "enable_source_transition_correction": True,
        },
        "predict": {
            "enable_source_read_policy": True,
            "enable_source_guard": True,
            "enable_source_transition_correction": True,
        },
    },
    "hurdle_only": {
        "fit": {
            "enable_hurdle_head": True,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": False,
        },
        "predict": {},
    },
    "hurdle_plus_jump": {
        "fit": {
            "enable_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
        },
        "predict": {},
    },
    "hurdle_plus_sparsity": {
        "fit": {
            "enable_hurdle_head": True,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
        },
        "predict": {},
    },
    "long_horizon_contract": {
        "fit": {
            "enable_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
        },
        "predict": {},
    },
    "long_horizon_contract_plus_guard": {
        "fit": {
            "enable_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
            "enable_source_read_policy": True,
            "enable_source_guard": True,
            "enable_source_transition_correction": True,
        },
        "predict": {
            "enable_source_read_policy": True,
            "enable_source_guard": True,
            "enable_source_transition_correction": True,
        },
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="task2_forecast", help="Task name to analyze.")
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon to analyze.")
    ap.add_argument(
        "--entity-selection",
        choices=("common_coverage", "dynamic_test"),
        default="common_coverage",
        help="Entity-selection strategy for the real multisource surface.",
    )
    ap.add_argument(
        "--entity-limit",
        type=int,
        default=None,
        help="Override entity count for the selected surface strategy.",
    )
    ap.add_argument(
        "--max-rows-per-ablation",
        type=int,
        default=1000,
        help="Optional downsample cap applied to each ablation's train split.",
    )
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to save the JSON report.")
    return ap.parse_args()


def _requires_cik(ablations: tuple[str, ...]) -> bool:
    return any(ablation in {"core_edgar", "core_edgar_seed2", "full"} for ablation in ablations)


def _rank_dynamic_entities(
    stats: Dict[str, Dict[str, Any]],
    *,
    limit: int,
    require_cik: bool,
) -> list[Dict[str, Any]]:
    ranked: list[Dict[str, Any]] = []
    for entity_id, payload in stats.items():
        train_count = int(payload.get("train_count", 0))
        test_count = int(payload.get("test_count", 0))
        test_values = sorted(float(value) for value in payload.get("test_values", set()))
        has_cik = bool(payload.get("has_cik", False))
        if train_count <= 0 or test_count <= 0:
            continue
        if require_cik and not has_cik:
            continue
        if len(test_values) <= 1:
            continue
        ranked.append(
            {
                "entity_id": entity_id,
                "train_count": train_count,
                "test_count": test_count,
                "test_unique_values": len(test_values),
                "test_unique_sample": test_values[:8],
                "has_cik": has_cik,
            }
        )
    ranked.sort(
        key=lambda item: (
            -(item["train_count"] + item["test_count"]),
            -item["test_unique_values"],
            item["entity_id"],
        )
    )
    return ranked[:limit]


def _select_dynamic_entities(case: RealSurfaceCase) -> list[Dict[str, Any]]:
    pointer = _load_pointer()
    core_path = _resolve_repo_relative(pointer["offers_core_daily"]["dir"]) / "offers_core_daily.parquet"
    temporal_config = _make_temporal_config()
    train_end = pd.Timestamp(temporal_config.train_end)
    test_start = pd.Timestamp(temporal_config.val_end) + pd.Timedelta(days=temporal_config.embargo_days)
    test_end = pd.Timestamp(temporal_config.test_end)

    stats: Dict[str, Dict[str, Any]] = {}
    parquet = pq.ParquetFile(str(core_path))
    columns = ["entity_id", "crawled_date_day", case.target, "cik"]
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
                    "test_values": set(),
                    "has_cik": False,
                },
            )
            timestamp = getattr(row, "crawled_date_day")
            value = getattr(row, case.target)
            if pd.notna(getattr(row, "cik", None)):
                payload["has_cik"] = True
            if pd.isna(timestamp) or pd.isna(value):
                continue
            if timestamp <= train_end:
                payload["train_count"] += 1
            elif test_start < timestamp <= test_end:
                payload["test_count"] += 1
                if len(payload["test_values"]) < 8:
                    payload["test_values"].add(float(value))

    ranked = _rank_dynamic_entities(
        stats,
        limit=max(case.dynamic_entity_limit, 1) * 8,
        require_cik=_requires_cik(case.ablations),
    )
    selected = ranked[: case.dynamic_entity_limit]
    if len(selected) < case.dynamic_entity_limit:
        raise RuntimeError(
            f"Requested {case.dynamic_entity_limit} dynamic entities but only found {len(selected)}"
        )
    return selected


def _load_frames_for_selected_entities(
    case: RealSurfaceCase,
    entity_report: list[Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    pointer = _load_pointer()
    entity_ids = [str(item["entity_id"]) for item in entity_report]
    base = _load_core_slice(pointer, entity_ids)
    base["entity_id"] = base["entity_id"].astype(str)
    base = base[base["entity_id"].isin(entity_ids)].copy().reset_index(drop=True)

    text_frame: pd.DataFrame | None = None
    if any(ablation in {"core_text", "full"} for ablation in case.ablations):
        text_frame = _load_text_slice(entity_ids)
        text_frame["entity_id"] = text_frame["entity_id"].astype(str)

    edgar_frame: pd.DataFrame | None = None
    if _requires_cik(case.ablations):
        ciks = [str(value) for value in base["cik"].dropna().astype(str).unique().tolist()]
        if not ciks:
            raise RuntimeError("Dynamic entity selection requested EDGAR-backed ablations but no CIK values were found")
        edgar_frame = _load_edgar_slice(pointer, ciks)
        edgar_frame["cik"] = edgar_frame["cik"].astype(str)

    frames: Dict[str, pd.DataFrame] = {}
    for ablation in case.ablations:
        frame = base.copy()
        if ablation in {"core_text", "full"}:
            if text_frame is None:
                raise RuntimeError(f"Missing text frame for ablation={ablation}")
            frame = _join_text_embeddings(frame, text_frame)
        if ablation in {"core_edgar", "core_edgar_seed2", "full"}:
            if edgar_frame is None:
                raise RuntimeError(f"Missing EDGAR frame for ablation={ablation}")
            frame["cik"] = frame["cik"].astype(str)
            frame = _join_edgar_asof(frame, edgar_frame)
        frames[ablation] = frame
    return frames


def _load_case_frames(case: RealSurfaceCase) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    if case.entity_selection == "dynamic_test":
        entity_report = _select_dynamic_entities(case)
        return _load_frames_for_selected_entities(case, entity_report), {
            "strategy": case.entity_selection,
            "requested_entity_limit": case.dynamic_entity_limit,
            "selected_entities": [item["entity_id"] for item in entity_report],
            "selected_entity_report": entity_report,
        }

    raw_frames = {ablation: _load_ablation_frame(case, ablation) for ablation in case.ablations}
    raw_frames = _filter_common_entities(raw_frames)
    return raw_frames, {
        "strategy": case.entity_selection,
        "requested_entity_limit": case.max_entities,
        "selected_entities": sorted(
            set(raw_frames[case.ablations[0]]["entity_id"].astype(str).unique().tolist())
        ),
    }


def _transition_signal_summary(test_surface: Dict[str, Any]) -> Dict[str, Any]:
    profile_ids = _source_profile_ids(test_surface["source"])
    signal = _transition_signal(test_surface["aux"], test_surface["anchor"])
    abs_signal = np.abs(signal)
    summary = {
        "overall_abs_mean": float(np.mean(abs_signal)) if len(abs_signal) else 0.0,
        "overall_abs_p75": float(np.quantile(abs_signal, 0.75)) if len(abs_signal) else 0.0,
        "overall_nonzero_rows": int(np.sum(abs_signal > 1e-9)),
        "by_profile": {},
    }
    for profile_id, profile_name in PROFILE_NAMES.items():
        mask = profile_ids == profile_id
        if not mask.any():
            continue
        profile_signal = abs_signal[mask]
        summary["by_profile"][profile_name] = {
            "rows": int(mask.sum()),
            "abs_mean": float(np.mean(profile_signal)),
            "abs_p75": float(np.quantile(profile_signal, 0.75)),
            "abs_max": float(np.max(profile_signal)),
            "nonzero_rows": int(np.sum(profile_signal > 1e-9)),
        }
    return summary


def _load_ablation_frame(case: RealSurfaceCase, ablation: str) -> pd.DataFrame:
    args = argparse.Namespace(
        task=case.task,
        target=case.target,
        horizon=case.horizon,
        max_entities=case.max_entities,
        ablation=ablation,
    )
    return _load_smoke_frame(args, _make_temporal_config())


def _filter_common_entities(raw_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_entities: set[str] | None = None
    for frame in raw_frames.values():
        entity_ids = set(frame["entity_id"].astype(str).unique().tolist())
        common_entities = entity_ids if common_entities is None else (common_entities & entity_ids)
    if not common_entities:
        raise RuntimeError("No common entities remain across the requested ablations")
    common = sorted(common_entities)
    return {
        ablation: frame[frame["entity_id"].astype(str).isin(common)].copy().reset_index(drop=True)
        for ablation, frame in raw_frames.items()
    }


def _split_and_prepare(case: RealSurfaceCase, ablation: str, frame: pd.DataFrame) -> Dict[str, Any]:
    temporal_config = _make_temporal_config()
    train, _, test, _ = apply_temporal_split(frame, temporal_config)
    train = train.sort_values(["entity_id", "crawled_date_day"], kind="mergesort").reset_index(drop=True)
    test = test.sort_values(["entity_id", "crawled_date_day"], kind="mergesort").reset_index(drop=True)
    if case.max_rows_per_ablation > 0:
        train = _downsample_preserve_time(train, case.max_rows_per_ablation)

    X_train, y_train = _prepare_features(train, case.target)
    X_test, y_test = _prepare_features(test, case.target)
    train_raw = train.loc[X_train.index].copy()
    test_raw = test.loc[X_test.index].copy()
    return {
        "ablation": ablation,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def _fit_shared_encoder(
    case: RealSurfaceCase,
    surfaces: Dict[str, Dict[str, Any]],
    backbone_kwargs: Dict[str, Any] | None = None,
):
    source_memory = SourceMemoryAssembler()
    condition_encoder = MainlineConditionEncoder()
    barrier = TargetIsolatedBarrier()
    helper = SingleModelMainlineWrapper(seed=7, **(backbone_kwargs or {}))
    helper._target_name = case.target
    helper._task_name = case.task
    helper._horizon = case.horizon

    core_cols_union: list[str] = []
    core_seen: set[str] = set()
    source_state_cols: list[str] | None = None
    investors_source_cols: list[str] | None = None
    train_core_frames = []

    encoded: Dict[str, Dict[str, Any]] = {}
    for ablation, surface in surfaces.items():
        helper._ablation_name = ablation
        runtime_train = helper._prepare_runtime_frame(surface["X_train"], raw_frame=surface["train_raw"])
        runtime_test = helper._prepare_runtime_frame(surface["X_test"], raw_frame=surface["test_raw"])

        source_layout = source_memory.infer_layout(runtime_train)
        source_train = source_memory.build_runtime_features(runtime_train, layout=source_layout)
        source_test = source_memory.build_runtime_features(runtime_test, layout=source_layout)

        train_feature_cols = list(surface["X_train"].columns)
        source_edgar = set(source_layout.edgar_cols)
        source_text = set(source_layout.text_cols)
        core_cols = [col for col in train_feature_cols if col not in (source_edgar | source_text)]
        if not core_cols:
            core_cols = list(train_feature_cols)
        for col in core_cols:
            if col not in core_seen:
                core_seen.add(col)
                core_cols_union.append(col)

        history_train = helper._build_target_history_features(runtime_train, include_seed=False)
        history_seed = helper._collect_history_seed(runtime_train)
        helper._history_seed = history_seed
        history_test = helper._build_target_history_features(runtime_test, include_seed=True)

        investors_source_train = helper._build_investors_source_features(source_train)
        investors_source_test = helper._build_investors_source_features(source_test)
        investors_mark_train = helper._build_investors_mark_features(runtime_train)
        investors_mark_test = helper._build_investors_mark_features(runtime_test)
        source_state_cols = list(source_train.columns)
        investors_source_cols = list(investors_source_train.columns)

        encoded[ablation] = {
            "runtime_train": runtime_train,
            "runtime_test": runtime_test,
            "source_train": source_train,
            "source_test": source_test,
            "history_train": history_train,
            "history_test": history_test,
            "anchor_train": helper._resolve_anchor(history_train),
            "anchor_test": helper._resolve_anchor(history_test),
            "investors_source_train": investors_source_train,
            "investors_source_test": investors_source_test,
            "investors_mark_train": investors_mark_train,
            "investors_mark_test": investors_mark_test,
            "core_train": surface["X_train"].reindex(columns=core_cols, fill_value=0.0),
            "core_test": surface["X_test"].reindex(columns=core_cols, fill_value=0.0),
            "y_train": surface["y_train"],
            "y_test": surface["y_test"],
        }
        train_core_frames.append(surface["X_train"].reindex(columns=core_cols, fill_value=0.0))

    backbone_spec = SharedTemporalBackboneSpec(**(backbone_kwargs or {}))
    backbone = SharedTemporalBackbone(spec=backbone_spec, random_state=7)
    backbone.fit(
        pd.concat([frame.reindex(columns=core_cols_union, fill_value=0.0) for frame in train_core_frames], axis=0),
        feature_cols=core_cols_union,
    )
    for payload in encoded.values():
        payload["backbone_seed"] = backbone.build_context_seed(
            payload["runtime_train"],
            payload["core_train"].reindex(columns=core_cols_union, fill_value=0.0),
        )

    sample_ablation = next(iter(encoded))
    sample_condition = condition_encoder.broadcast(
        ConditionKey(task=case.task, target=case.target, horizon=case.horizon, ablation=sample_ablation),
        n_rows=1,
    )
    sample_source = encoded[sample_ablation]["source_train"].reindex(columns=source_state_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
    sample_shared = backbone.transform(
        encoded[sample_ablation]["core_train"].reindex(columns=core_cols_union, fill_value=0.0),
        context_frame=encoded[sample_ablation]["runtime_train"],
    )
    _, sample_shared = helper._refresh_event_state_card_with_shared_state(
        encoded[sample_ablation]["runtime_train"],
        shared_state=sample_shared,
        source_frame=encoded[sample_ablation]["source_train"],
        phase="train",
    )
    barrier.fit(
        shared_dim=sample_shared.shape[1],
        condition_dim=sample_condition.shape[1],
        source_dim=sample_source.shape[1],
    )

    for ablation, payload in encoded.items():
        key = ConditionKey(task=case.task, target=case.target, horizon=case.horizon, ablation=ablation)
        shared_train = backbone.transform(
            payload["core_train"].reindex(columns=core_cols_union, fill_value=0.0),
            context_frame=payload["runtime_train"],
        )
        shared_test = backbone.transform(
            payload["core_test"].reindex(columns=core_cols_union, fill_value=0.0),
            context_frame=payload["runtime_test"],
            seed_frame=payload["backbone_seed"],
        )
        event_state_train, shared_train = helper._refresh_event_state_card_with_shared_state(
            payload["runtime_train"],
            shared_state=shared_train,
            source_frame=payload["source_train"],
            phase="train",
        )
        payload["event_state_train"] = event_state_train
        payload["event_state_trunk_train"] = dict(helper._event_state_card)
        event_state_test, shared_test = helper._refresh_event_state_card_with_shared_state(
            payload["runtime_test"],
            shared_state=shared_test,
            source_frame=payload["source_test"],
            phase="test",
        )
        payload["event_state_test"] = event_state_test
        payload["event_state_trunk_test"] = dict(helper._event_state_card)
        condition_train = condition_encoder.broadcast(key, len(payload["core_train"]))
        condition_test = condition_encoder.broadcast(key, len(payload["core_test"]))
        source_state_train = payload["source_train"].reindex(columns=source_state_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
        source_state_test = payload["source_test"].reindex(columns=source_state_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
        payload["lane_train"] = barrier.split(shared_train, condition_train, source_state_train)["investors"]
        payload["lane_test"] = barrier.split(shared_test, condition_test, source_state_test)["investors"]

    return encoded, investors_source_cols or []


def _concat_surface(encoded: Dict[str, Dict[str, Any]], split: str) -> Dict[str, Any]:
    lane = []
    aux = []
    event_state = []
    anchor = []
    source = []
    mark = []
    y = []
    ablation = []
    for name, payload in encoded.items():
        lane.append(payload[f"lane_{split}"])
        aux.append(payload[f"history_{split}"].fillna(0.0).to_numpy(dtype=np.float32, copy=False))
        event_state.append(payload[f"event_state_{split}"].to_numpy(dtype=np.float32, copy=False))
        anchor.append(payload[f"anchor_{split}"])
        source.append(payload[f"investors_source_{split}"].to_numpy(dtype=np.float32, copy=False))
        mark.append(payload[f"investors_mark_{split}"].to_numpy(dtype=np.float32, copy=False))
        y.append(payload[f"y_{split}"].to_numpy(dtype=np.float64, copy=False))
        ablation.extend([name] * len(payload[f"y_{split}"]))
    return {
        "lane": np.concatenate(lane, axis=0),
        "aux": np.concatenate(aux, axis=0),
        "event_state": np.concatenate(event_state, axis=0),
        "anchor": np.concatenate(anchor, axis=0),
        "source": np.concatenate(source, axis=0),
        "mark": np.concatenate(mark, axis=0),
        "y": np.concatenate(y, axis=0),
        "ablation": np.asarray(ablation, dtype=object),
    }


def _surface_aux_features(surface: Dict[str, Any], *, enable_event_state_features: bool) -> np.ndarray:
    history_aux = np.asarray(surface["aux"], dtype=np.float32)
    if not enable_event_state_features:
        return history_aux
    event_state_aux = np.asarray(surface.get("event_state"), dtype=np.float32)
    if event_state_aux.size == 0:
        return history_aux
    if history_aux.size == 0:
        return event_state_aux
    return np.concatenate([history_aux, event_state_aux], axis=1).astype(np.float32, copy=False)


def _profile_breakdown(y_true: np.ndarray, preds: np.ndarray, profile_ids: np.ndarray, ablations: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for profile_id, profile_name in PROFILE_NAMES.items():
        mask = profile_ids == profile_id
        if not mask.any():
            continue
        out[profile_name] = {
            "rows": int(mask.sum()),
            "metrics": _compute_metrics(y_true[mask], preds[mask]),
            "target_mean": float(np.mean(y_true[mask])),
            "pred_mean": float(np.mean(preds[mask])),
            "ablations": {
                str(name): int(np.sum(ablations[mask] == name))
                for name in np.unique(ablations[mask])
            },
        }
    return out


def _run_variant(
    case: RealSurfaceCase,
    name: str,
    variant: Dict[str, Dict[str, bool]],
    train_surface: Dict[str, Any],
    test_surface: Dict[str, Any],
) -> Dict[str, Any]:
    fit_kwargs = dict(variant["fit"])
    predict_kwargs = dict(variant["predict"])
    fit_event_state = bool(fit_kwargs.pop("enable_investors_event_state_features", False))
    predict_event_state = bool(predict_kwargs.pop("enable_investors_event_state_features", fit_event_state))
    fit_mark = bool(fit_kwargs.pop("enable_investors_mark_features", False))
    predict_mark = bool(predict_kwargs.pop("enable_investors_mark_features", fit_mark))
    train_aux = _surface_aux_features(train_surface, enable_event_state_features=fit_event_state)
    test_aux = _surface_aux_features(test_surface, enable_event_state_features=predict_event_state)
    train_mark = np.asarray(train_surface.get("mark"), dtype=np.float32) if "mark" in train_surface else None
    test_mark = np.asarray(test_surface.get("mark"), dtype=np.float32) if "mark" in test_surface else None

    runtime = InvestorsLaneRuntime(random_state=7)
    runtime.fit(
        train_surface["lane"],
        train_surface["y"],
        aux_features=train_aux,
        anchor=train_surface["anchor"],
        source_features=train_surface["source"],
        mark_features=train_mark,
        horizon=case.horizon,
        anchor_blend=0.70,
        task_name=case.task,
        enable_mark_features=fit_mark,
        **fit_kwargs,
    )
    preds = runtime.predict(
        test_surface["lane"],
        aux_features=test_aux,
        anchor=test_surface["anchor"],
        source_features=test_surface["source"],
        mark_features=test_mark,
        enable_mark_features=predict_mark,
        **predict_kwargs,
    )
    profile_ids = _source_profile_ids(test_surface["source"])
    return {
        "variant": name,
        "overall_metrics": _compute_metrics(test_surface["y"], preds),
        "overall_pred_mean": float(np.mean(preds)),
        "overall_target_mean": float(np.mean(test_surface["y"])),
        "profile_metrics": _profile_breakdown(test_surface["y"], preds, profile_ids, test_surface["ablation"]),
        "learned_profile_anchor_blends": {
            PROFILE_NAMES.get(int(profile_id), str(profile_id)): float(blend)
            for profile_id, blend in runtime._source_anchor_blend_by_profile.items()
        },
        "learned_profile_anchor_blend_reliability": {
            PROFILE_NAMES.get(int(profile_id), str(profile_id)): float(weight)
            for profile_id, weight in runtime._source_anchor_blend_reliability_by_profile.items()
        },
        "learned_profile_transition_strength": {
            PROFILE_NAMES.get(int(profile_id), str(profile_id)): float(strength)
            for profile_id, strength in runtime._source_transition_strength_by_profile.items()
        },
        "learned_profile_transition_reliability": {
            PROFILE_NAMES.get(int(profile_id), str(profile_id)): float(weight)
            for profile_id, weight in runtime._source_transition_reliability_by_profile.items()
        },
        "lane_anchor_blend": float(runtime._global_anchor_blend),
        "lane_anchor_blend_reliability": float(runtime._global_anchor_blend_reliability),
        "lane_horizon_anchor_mix": float(runtime._horizon_anchor_mix),
        "lane_horizon_anchor_mix_reliability": float(runtime._horizon_anchor_mix_reliability),
        "lane_jump_strength": float(runtime._global_jump_strength),
        "lane_jump_reliability": float(runtime._global_jump_reliability),
        "lane_flags": {
            "use_hurdle_head": bool(runtime._use_hurdle_head),
            "use_count_jump": bool(runtime._use_count_jump),
            "use_sparsity_gate": bool(runtime._use_sparsity_gate),
            "use_event_state_features": bool(fit_event_state or predict_event_state),
            "use_mark_features": bool(fit_mark or predict_mark),
        },
        "event_state_aux_dim": int(test_surface["event_state"].shape[1]) if "event_state" in test_surface else 0,
        "mark_feature_dim": int(test_surface["mark"].shape[1]) if "mark" in test_surface else 0,
    }


def main() -> int:
    args = _parse_args()
    entity_limit = args.entity_limit
    case = RealSurfaceCase(
        task=args.task,
        horizon=int(args.horizon),
        entity_selection=args.entity_selection,
        max_entities=entity_limit if entity_limit is not None and args.entity_selection == "common_coverage" else 16,
        dynamic_entity_limit=entity_limit if entity_limit is not None and args.entity_selection == "dynamic_test" else 4,
        max_rows_per_ablation=args.max_rows_per_ablation,
    )
    raw_frames, selection_report = _load_case_frames(case)
    surfaces = {
        ablation: _split_and_prepare(case, ablation, frame)
        for ablation, frame in raw_frames.items()
    }
    encoded, _ = _fit_shared_encoder(case, surfaces)
    train_surface = _concat_surface(encoded, "train")
    test_surface = _concat_surface(encoded, "test")

    report = {
        "case": asdict(case),
        "entity_selection": selection_report,
        "common_entities": selection_report["selected_entities"],
        "train_rows": int(len(train_surface["y"])),
        "test_rows": int(len(test_surface["y"])),
        "train_profile_counts": {
            PROFILE_NAMES[int(profile_id)]: int(np.sum(_source_profile_ids(train_surface["source"]) == profile_id))
            for profile_id in np.unique(_source_profile_ids(train_surface["source"]))
        },
        "test_profile_counts": {
            PROFILE_NAMES[int(profile_id)]: int(np.sum(_source_profile_ids(test_surface["source"]) == profile_id))
            for profile_id in np.unique(_source_profile_ids(test_surface["source"]))
        },
        "transition_signal_summary": _transition_signal_summary(test_surface),
        "variants": {},
    }
    for name, variant in VARIANTS.items():
        report["variants"][name] = _run_variant(case, name, variant, train_surface, test_surface)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())