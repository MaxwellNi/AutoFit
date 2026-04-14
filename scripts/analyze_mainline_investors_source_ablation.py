#!/usr/bin/env python3
"""Controlled local ablation study for native mainline investors source mechanisms."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from src.narrative.block3.models.single_model_mainline.lanes.investors_lane import (
    _source_confidence,
    _source_guard_weight,
    _source_profile_ids,
)
from src.narrative.block3.unified_protocol import TemporalSplitConfig, apply_temporal_split, load_block3_tasks_config
from scripts.run_v740_alpha_smoke_slice import (
    _compute_metrics,
    _downsample_binary_preserve_time,
    _downsample_preserve_time,
    _load_smoke_frame,
    _prepare_features,
)


PROFILE_NAMES = {0: "none", 1: "edgar_only", 2: "text_only", 3: "mixed"}
DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "enable_count_hurdle_head": False,
        "enable_count_jump": False,
        "enable_count_sparsity_gate": False,
    },
    "source_features_only": {
        "enable_investors_source_features": True,
        "enable_investors_selective_source_activation": False,
    },
    "source_guard_only": {
        "enable_investors_source_features": True,
        "enable_investors_source_guard": True,
        "enable_investors_selective_source_activation": False,
    },
    "source_specialists_only": {
        "enable_investors_source_features": True,
        "enable_count_source_specialists": True,
        "enable_investors_source_guard": False,
        "enable_investors_selective_source_activation": False,
    },
    "source_specialists_plus_guard": {
        "enable_investors_source_features": True,
        "enable_count_source_specialists": True,
        "enable_investors_source_guard": True,
        "enable_investors_selective_source_activation": False,
    },
    "selective_source_path_requested": {
        "enable_investors_source_features": True,
        "enable_count_source_specialists": True,
        "enable_investors_source_guard": True,
    },
    "selective_source_read_policy_requested": {
        "enable_investors_source_read_policy": True,
    },
    "selective_source_transition_requested": {
        "enable_investors_source_read_policy": True,
        "enable_investors_transition_correction": True,
        "enable_investors_source_guard": True,
    },
}


@dataclass(frozen=True)
class SliceCase:
    task: str
    ablation: str
    target: str = "investors_count"
    horizon: int = 1
    max_entities: int = 12
    max_rows: int = 1200

    @property
    def name(self) -> str:
        return f"{self.task}__{self.ablation}__{self.target}__h{self.horizon}"


DEFAULT_CASES = (
    SliceCase(task="task1_outcome", ablation="core_only"),
    SliceCase(task="task2_forecast", ablation="core_edgar"),
    SliceCase(task="task2_forecast", ablation="full"),
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    return ap.parse_args()


def _make_temporal_config() -> TemporalSplitConfig:
    tasks_config = load_block3_tasks_config()
    split_cfg = tasks_config.get("split", {})
    return TemporalSplitConfig(
        train_end=split_cfg.get("train_end", "2025-06-30"),
        val_end=split_cfg.get("val_end", "2025-09-30"),
        test_end=split_cfg.get("test_end", "2025-12-31"),
        embargo_days=split_cfg.get("embargo_days", 7),
    )


def _load_case_frames(case: SliceCase, temporal_config: TemporalSplitConfig):
    class _Args:
        task = case.task
        ablation = case.ablation
        target = case.target
        horizon = case.horizon
        max_entities = case.max_entities
        max_rows = case.max_rows

    df = _load_smoke_frame(_Args, temporal_config)
    train, val, test, _ = apply_temporal_split(df, temporal_config)
    if case.max_rows and len(train) > case.max_rows:
        if case.target == "is_funded":
            train = _downsample_binary_preserve_time(train, case.target, case.max_rows)
        else:
            train = _downsample_preserve_time(train, case.max_rows)
    return train, val, test


def _profile_breakdown(y_true: np.ndarray, preds: np.ndarray, profile_ids: np.ndarray) -> Dict[str, Any]:
    breakdown: Dict[str, Any] = {}
    for profile_id, profile_name in PROFILE_NAMES.items():
        mask = profile_ids == profile_id
        if not mask.any():
            continue
        breakdown[profile_name] = {
            "rows": int(mask.sum()),
            "metrics": _compute_metrics(y_true[mask], preds[mask]),
            "target_mean": float(np.mean(y_true[mask])),
            "pred_mean": float(np.mean(preds[mask])),
        }
    return breakdown


def _build_source_profile_report(model: SingleModelMainlineWrapper, X_frame, raw_frame) -> Dict[str, Any]:
    runtime_frame = model._prepare_runtime_frame(X_frame, raw_frame=raw_frame)
    source_layout = model.source_memory.infer_layout(runtime_frame)
    source_frame = model.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
    investors_source = model._build_investors_source_features(source_frame)
    source_matrix = investors_source.to_numpy(dtype=np.float32, copy=False)
    profile_ids = _source_profile_ids(source_matrix)
    profile_counts = {
        PROFILE_NAMES[int(profile_id)]: int(np.sum(profile_ids == profile_id))
        for profile_id in np.unique(profile_ids)
    }
    return {
        "profile_ids": profile_ids,
        "profile_counts": profile_counts,
        "source_confidence_mean_by_profile": {
            PROFILE_NAMES[int(profile_id)]: float(np.mean(_source_confidence(source_matrix)[profile_ids == profile_id]))
            for profile_id in np.unique(profile_ids)
        },
        "source_guard_mean_by_profile": {
            PROFILE_NAMES[int(profile_id)]: float(np.mean(_source_guard_weight(source_matrix)[profile_ids == profile_id]))
            for profile_id in np.unique(profile_ids)
        },
    }


def _run_variant(case: SliceCase, variant_name: str, kwargs: Dict[str, Any], frames) -> Dict[str, Any]:
    train, val, test = frames
    X_train, y_train = _prepare_features(train, case.target)
    X_test, y_test = _prepare_features(test, case.target)

    model = SingleModelMainlineWrapper(**kwargs)
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=case.target,
        task=case.task,
        ablation=case.ablation,
        horizon=case.horizon,
    )
    preds = np.asarray(
        model.predict(
            X_test,
            test_raw=test,
            target=case.target,
            task=case.task,
            ablation=case.ablation,
            horizon=case.horizon,
        ),
        dtype=np.float64,
    )
    y_true = y_test.to_numpy(dtype=np.float64)
    test_source_report = _build_source_profile_report(model, X_test, test)
    train_source_report = _build_source_profile_report(model, X_train, train)
    history_frame = model._build_target_history_features(model._prepare_runtime_frame(X_test, raw_frame=test), include_seed=True)
    anchor = model._resolve_anchor(history_frame)
    anchor_blend = float(model.investors_lane_runtime._anchor_blend)
    regime = model.get_regime_info()
    effective_anchor_weight_by_profile = {}
    for profile_name, guard_mean in test_source_report["source_guard_mean_by_profile"].items():
        effective_anchor_weight_by_profile[profile_name] = float(
            anchor_blend + (1.0 - anchor_blend) * (1.0 - guard_mean)
        )

    return {
        "variant": variant_name,
        "model_kwargs": kwargs,
        "overall_metrics": _compute_metrics(y_true, preds),
        "anchor_metrics": _compute_metrics(y_true, anchor),
        "anchor_blend": anchor_blend,
        "investors_source_activation": regime.get("investors_source_activation", {}),
        "overall_pred_mean": float(np.mean(preds)),
        "overall_target_mean": float(np.mean(y_true)),
        "train_profile_counts": train_source_report["profile_counts"],
        "test_profile_counts": test_source_report["profile_counts"],
        "profile_metrics": _profile_breakdown(y_true, preds, test_source_report["profile_ids"]),
        "anchor_profile_metrics": _profile_breakdown(y_true, anchor, test_source_report["profile_ids"]),
        "source_confidence_mean_by_profile": test_source_report["source_confidence_mean_by_profile"],
        "source_guard_mean_by_profile": test_source_report["source_guard_mean_by_profile"],
        "effective_anchor_weight_by_profile": effective_anchor_weight_by_profile,
        "active_specialist_profiles": sorted(int(profile_id) for profile_id in model.investors_lane_runtime._positive_specialists.keys()),
    }


def main() -> int:
    args = _parse_args()
    temporal_config = _make_temporal_config()

    report: Dict[str, Any] = {
        "cases": {},
        "variants": list(DEFAULT_VARIANTS.keys()),
    }

    for case in DEFAULT_CASES:
        frames = _load_case_frames(case, temporal_config)
        case_report: Dict[str, Any] = {
            "case": asdict(case),
            "variants": {},
        }
        for variant_name, kwargs in DEFAULT_VARIANTS.items():
            case_report["variants"][variant_name] = _run_variant(case, variant_name, kwargs, frames)
        report["cases"][case.name] = case_report

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())