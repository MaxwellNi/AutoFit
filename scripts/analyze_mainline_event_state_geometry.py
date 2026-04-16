#!/usr/bin/env python3
"""Audit the mainline event-state trunk and investors-first wave candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_investors_real_multisource_surface import (  # noqa: E402
    RealSurfaceCase,
    _load_case_frames,
    _split_and_prepare,
)
from scripts.run_v740_alpha_smoke_slice import _compute_metrics  # noqa: E402
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper  # noqa: E402


DEFAULT_ABLATIONS: Tuple[str, ...] = ("core_only", "core_edgar", "core_text", "full")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="task2_forecast", help="Task to audit.")
    ap.add_argument("--target", default="investors_count", help="Target to audit.")
    ap.add_argument(
        "--horizons",
        default="1,7,30",
        help="Comma-separated horizons to audit, e.g. '1,7,30'.",
    )
    ap.add_argument(
        "--ablations",
        nargs="+",
        default=list(DEFAULT_ABLATIONS),
        help="Ablations to include in the geometry audit.",
    )
    ap.add_argument(
        "--entity-selection",
        choices=("common_coverage", "dynamic_test"),
        default="dynamic_test",
        help="Entity-selection strategy for the real-data slice.",
    )
    ap.add_argument(
        "--entity-limit",
        type=int,
        default=4,
        help="Selected entity count for the requested strategy.",
    )
    ap.add_argument(
        "--max-rows-per-ablation",
        type=int,
        default=1000,
        help="Optional train-row cap per ablation; 0 disables downsampling.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the JSON report.",
    )
    return ap.parse_args()


def _parse_horizons(spec: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(spec).split(",")]
    horizons = tuple(int(token) for token in tokens if token)
    if not horizons:
        raise ValueError("At least one horizon is required for event-state geometry audit")
    return horizons


def _variant_kwargs(horizon: int) -> Dict[str, Dict[str, Any]]:
    common_source = {
        "enable_investors_source_features": True,
        "enable_investors_source_guard": True,
    }
    variants: Dict[str, Dict[str, Any]] = {
        "legacy_baseline": {},
        "event_state_boundary_guard": {
            "variant": "mainline_event_state_boundary_guard",
        },
        "source_features_guard": dict(common_source),
        "source_read_guard": {
            **common_source,
            "enable_investors_source_read_policy": True,
        },
    }
    if horizon > 1:
        variants["source_read_transition_guard"] = {
            **common_source,
            "enable_investors_source_read_policy": True,
            "enable_investors_transition_correction": True,
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
        }
        variants["source_read_transition_guard_plus_sparsity"] = {
            **variants["source_read_transition_guard"],
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
        }
    else:
        variants["source_read_transition_guard"] = {
            **common_source,
            "enable_investors_source_read_policy": True,
            "enable_investors_transition_correction": True,
        }
    return variants


def _run_variant(
    case: RealSurfaceCase,
    ablation: str,
    surface: Dict[str, Any],
    variant_name: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    model = SingleModelMainlineWrapper(seed=7, **kwargs)
    model.fit(
        surface["X_train"],
        surface["y_train"],
        train_raw=surface["train_raw"],
        target=case.target,
        task=case.task,
        ablation=ablation,
        horizon=case.horizon,
    )
    preds = np.asarray(
        model.predict(
            surface["X_test"],
            test_raw=surface["test_raw"],
            target=case.target,
            task=case.task,
            ablation=ablation,
            horizon=case.horizon,
        ),
        dtype=np.float64,
    )
    y_true = surface["y_test"].to_numpy(dtype=np.float64, copy=False)
    regime = model.get_regime_info()
    source_regime = regime.get("investors_source_activation", {})
    return {
        "variant": variant_name,
        "model_kwargs": dict(kwargs),
        "metrics": _compute_metrics(y_true, preds),
        "pred_mean": float(np.mean(preds)) if len(preds) else 0.0,
        "target_mean": float(np.mean(y_true)) if len(y_true) else 0.0,
        "runtime_mode": str(regime.get("runtime", {}).get("runtime_mode", "unknown")),
        "event_state_trunk": regime.get("event_state_trunk", {}),
        "investors_source_activation": {
            "requested_source_path": bool(source_regime.get("requested_source_path", False)),
            "effective_source_features": bool(source_regime.get("effective_source_features", False)),
            "effective_source_guard": bool(source_regime.get("effective_source_guard", False)),
            "effective_source_read_policy": bool(source_regime.get("effective_source_read_policy", False)),
            "effective_transition_correction": bool(source_regime.get("effective_transition_correction", False)),
            "activation_reason": str(source_regime.get("activation_reason", "unknown")),
            "transition_activation_reason": str(source_regime.get("transition_activation_reason", "unknown")),
            "geometry_card": dict(source_regime.get("geometry_card", {})),
        },
    }


def _evaluate_case(case: RealSurfaceCase) -> Dict[str, Any]:
    raw_frames, entity_report = _load_case_frames(case)
    variants = _variant_kwargs(case.horizon)
    ablation_reports: Dict[str, Any] = {}
    for ablation in case.ablations:
        surface = _split_and_prepare(case, ablation, raw_frames[ablation])
        variant_reports = {
            variant_name: _run_variant(case, ablation, surface, variant_name, kwargs)
            for variant_name, kwargs in variants.items()
        }
        ablation_reports[ablation] = {
            "train_rows": int(len(surface["train_raw"])),
            "test_rows": int(len(surface["test_raw"])),
            "train_entities": int(surface["train_raw"]["entity_id"].astype(str).nunique()),
            "test_entities": int(surface["test_raw"]["entity_id"].astype(str).nunique()),
            "variants": variant_reports,
        }
    return {
        "case": {
            "task": case.task,
            "target": case.target,
            "horizon": int(case.horizon),
            "ablations": list(case.ablations),
            "entity_selection": case.entity_selection,
            "entity_limit": case.dynamic_entity_limit if case.entity_selection == "dynamic_test" else case.max_entities,
            "max_rows_per_ablation": int(case.max_rows_per_ablation),
        },
        "entity_report": entity_report,
        "ablations": ablation_reports,
    }


def _aggregate_variant_deltas(cases: Dict[str, Any], baseline_key: str = "legacy_baseline") -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    for case_name, case_report in cases.items():
        for ablation, ablation_report in case_report.get("ablations", {}).items():
            variants = ablation_report.get("variants", {})
            if baseline_key not in variants:
                continue
            baseline_mae = float(variants[baseline_key]["metrics"]["mae"])
            for variant_name, payload in variants.items():
                entry = stats.setdefault(
                    variant_name,
                    {
                        "case_count": 0,
                        "mae_delta_pct": [],
                        "dynamic_entity_share": [],
                        "joint_financing_active_share": [],
                        "source_presence_share": [],
                        "heterogeneous_source_case_count": 0,
                        "effective_transition_case_count": 0,
                        "runtime_modes": set(),
                        "labels": [],
                    },
                )
                mae = float(payload["metrics"]["mae"])
                delta_pct = 100.0 * (baseline_mae - mae) / max(baseline_mae, 1e-8)
                event_state = payload.get("event_state_trunk", {})
                persistence = event_state.get("persistence_atoms", {})
                phase = event_state.get("phase_atoms", {})
                source = event_state.get("source_atoms", {})
                source_regime = payload.get("investors_source_activation", {})

                entry["case_count"] += 1
                entry["mae_delta_pct"].append(delta_pct)
                entry["dynamic_entity_share"].append(float(persistence.get("dynamic_entity_share", 0.0)))
                entry["joint_financing_active_share"].append(float(phase.get("joint_financing_active_share", 0.0)))
                entry["source_presence_share"].append(float(source.get("source_presence_share", 0.0)))
                entry["heterogeneous_source_case_count"] += int(source.get("source_surface") == "heterogeneous")
                entry["effective_transition_case_count"] += int(
                    bool(source_regime.get("effective_transition_correction", False))
                )
                entry["runtime_modes"].add(str(payload.get("runtime_mode", "unknown")))
                entry["labels"].append(f"{case_name}::{ablation}")

    summary: Dict[str, Any] = {}
    for variant_name, entry in stats.items():
        deltas = np.asarray(entry["mae_delta_pct"], dtype=np.float64)
        summary[variant_name] = {
            "case_count": int(entry["case_count"]),
            "mean_mae_delta_pct": float(np.mean(deltas)) if len(deltas) else 0.0,
            "best_mae_delta_pct": float(np.max(deltas)) if len(deltas) else 0.0,
            "worst_mae_delta_pct": float(np.min(deltas)) if len(deltas) else 0.0,
            "positive_case_count": int(np.sum(deltas > 0.0)),
            "negative_case_count": int(np.sum(deltas < 0.0)),
            "mean_dynamic_entity_share": float(np.mean(entry["dynamic_entity_share"]))
            if entry["dynamic_entity_share"]
            else 0.0,
            "mean_joint_financing_active_share": float(np.mean(entry["joint_financing_active_share"]))
            if entry["joint_financing_active_share"]
            else 0.0,
            "mean_source_presence_share": float(np.mean(entry["source_presence_share"]))
            if entry["source_presence_share"]
            else 0.0,
            "heterogeneous_source_case_count": int(entry["heterogeneous_source_case_count"]),
            "effective_transition_case_count": int(entry["effective_transition_case_count"]),
            "runtime_modes": sorted(entry["runtime_modes"]),
            "labels": list(entry["labels"]),
        }
    return summary


def main() -> int:
    args = _parse_args()
    horizons = _parse_horizons(args.horizons)
    entity_limit = int(args.entity_limit)
    cases: Dict[str, Any] = {}
    for horizon in horizons:
        case = RealSurfaceCase(
            task=args.task,
            target=args.target,
            horizon=int(horizon),
            ablations=tuple(args.ablations),
            entity_selection=args.entity_selection,
            max_entities=entity_limit if args.entity_selection == "common_coverage" else 16,
            dynamic_entity_limit=entity_limit if args.entity_selection == "dynamic_test" else 4,
            max_rows_per_ablation=int(args.max_rows_per_ablation),
        )
        case_key = f"{case.task}__{case.target}__h{case.horizon}"
        cases[case_key] = _evaluate_case(case)

    report = {
        "task": args.task,
        "target": args.target,
        "horizons": list(horizons),
        "entity_selection": args.entity_selection,
        "entity_limit": entity_limit,
        "max_rows_per_ablation": int(args.max_rows_per_ablation),
        "cases": cases,
        "summary": _aggregate_variant_deltas(cases),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())