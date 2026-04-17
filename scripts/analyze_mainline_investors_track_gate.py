#!/usr/bin/env python3
"""Gate active mainline investors candidates on official local slices and dynamic real multisource surfaces."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_investors_source_ablation import SliceCase, _load_case_frames, _make_temporal_config
from scripts.analyze_mainline_investors_real_multisource_surface import (
    RealSurfaceCase,
    _concat_surface,
    _fit_shared_encoder,
    _load_case_frames as _load_real_case_frames,
    _run_variant as _run_dynamic_variant,
    _split_and_prepare,
)
from scripts.run_v740_alpha_smoke_slice import _compute_metrics, _prepare_features
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper


OFFICIAL_CASES = (
    SliceCase(task="task1_outcome", ablation="core_only", horizon=1),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=7),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=14),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=30),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=1),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=7),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=14),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=30),
    SliceCase(task="task2_forecast", ablation="full", horizon=1),
    SliceCase(task="task2_forecast", ablation="full", horizon=7),
    SliceCase(task="task2_forecast", ablation="full", horizon=14),
    SliceCase(task="task2_forecast", ablation="full", horizon=30),
)

OFFICIAL_MEAN_DELTA_MIN_PCT = 0.0
OFFICIAL_WORST_DELTA_MIN_PCT = -1.0
DYNAMIC_MEAN_DELTA_MIN_PCT = 0.0


TRACK_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "legacy_baseline": {
        "role": "control",
        "official_kwargs": {
            "seed": 7,
            "use_delegate": False,
            "enable_investors_horizon_contract": False,
            "enable_count_hurdle_head": False,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": False,
        },
        "dynamic_variant": {
            "fit": {},
            "predict": {},
        },
    },
    "guarded_jump": {
        "role": "process_candidate",
        "official_kwargs": {
            "seed": 7,
            "use_delegate": False,
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": False,
            },
            "predict": {},
        },
    },
    "guarded_jump_plus_sparsity": {
        "role": "process_candidate",
        "official_kwargs": {
            "seed": 7,
            "use_delegate": False,
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": True,
                "count_sparsity_gate_strength": 0.75,
            },
            "predict": {},
        },
    },
    "event_state_boundary_guard": {
        "role": "prior_global_event_state_candidate",
        "official_kwargs": {
            "variant": "mainline_event_state_boundary_guard",
            "seed": 7,
            "use_delegate": False,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": True,
                "count_sparsity_gate_strength": 0.75,
                "enable_investors_event_state_features": True,
            },
            "predict": {
                "enable_investors_event_state_features": True,
            },
        },
    },
    "selective_event_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_selective_event_state_guard",
            "seed": 7,
            "use_delegate": False,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": False,
                "enable_investors_event_state_features": True,
            },
            "predict": {
                "enable_investors_event_state_features": True,
            },
        },
    },
    "marked_investor_guard": {
        "role": "active_mark_candidate",
        "official_kwargs": {
            "variant": "mainline_marked_investor_guard",
            "seed": 7,
            "use_delegate": False,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": False,
                "enable_investors_event_state_features": True,
                "enable_investors_mark_features": True,
            },
            "predict": {
                "enable_investors_event_state_features": True,
                "enable_investors_mark_features": True,
            },
        },
    },
    "multiscale_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_multiscale_state_guard",
            "seed": 7,
            "use_delegate": False,
        },
        "dynamic_variant": {
            "fit": {
                "enable_hurdle_head": True,
                "enable_count_jump": True,
                "count_jump_strength": 0.30,
                "enable_count_sparsity_gate": False,
                "enable_investors_event_state_features": True,
            },
            "predict": {
                "enable_investors_event_state_features": True,
            },
        },
        "dynamic_backbone": {
            "enable_multiscale_temporal_state": True,
            "temporal_state_windows": (3, 7, 30),
        },
    },
    "source_policy_transition_guard": {
        "role": "demoted_source_candidate",
        "official_kwargs": {
            "seed": 7,
            "use_delegate": False,
            "enable_investors_source_read_policy": True,
            "enable_investors_source_guard": True,
            "enable_investors_transition_correction": True,
        },
        "dynamic_variant": {
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
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("full", "official", "dynamic", "merge"),
        default="full",
        help="Run the full gate, only the official panel, only the dynamic panel, or merge precomputed panel reports.",
    )
    ap.add_argument(
        "--dynamic-horizons",
        default="14",
        help="Comma-separated dynamic_test horizons to evaluate, e.g. 7,14,30.",
    )
    ap.add_argument(
        "--dynamic-entity-limit",
        type=int,
        default=4,
        help="Entity count for each dynamic_test surface.",
    )
    ap.add_argument(
        "--dynamic-max-rows-per-ablation",
        type=int,
        default=1000,
        help="Train-row cap per ablation for dynamic surfaces.",
    )
    ap.add_argument(
        "--official-input-json",
        type=Path,
        default=None,
        help="Existing official-panel JSON report, required for --mode merge.",
    )
    ap.add_argument(
        "--dynamic-input-json",
        type=Path,
        default=None,
        help="Existing dynamic-panel JSON report, required for --mode merge.",
    )
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to write the JSON report.")
    return ap.parse_args()


def _parse_horizons(spec: str) -> tuple[int, ...]:
    values = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one dynamic horizon must be provided")
    return tuple(values)


def _track_contract() -> Dict[str, Any]:
    return {
        "runtime_owner": "single_model_mainline",
        "native_runtime_required": True,
        "delegate_forbidden": True,
        "official_panel_definition": "benchmark-consistent local official slices loaded via _load_smoke_frame plus canonical temporal split from block3 task config",
        "dynamic_panel_definition": "real-data dynamic_test multisource challenge surfaces; research-only and never merged into official claims",
        "source_results_interpreted_only_with_geometry": True,
        "demoted_families": ["event_state_boundary_guard", "source_policy_transition_guard"],
        "active_generation_focus": "multiscale_state_guard",
        "promotion_rules": {
            "official_mean_mae_delta_pct_min": OFFICIAL_MEAN_DELTA_MIN_PCT,
            "official_worst_mae_delta_pct_min": OFFICIAL_WORST_DELTA_MIN_PCT,
            "dynamic_mean_mae_delta_pct_min": DYNAMIC_MEAN_DELTA_MIN_PCT,
        },
    }


def _candidate_manifest() -> Dict[str, Dict[str, Any]]:
    return {name: {"role": payload["role"]} for name, payload in TRACK_CANDIDATES.items()}


def _dynamic_backbone_key(backbone_kwargs: Dict[str, Any] | None) -> str:
    if not backbone_kwargs:
        return "{}"
    normalized: Dict[str, Any] = {}
    for key, value in backbone_kwargs.items():
        if isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True)


def _run_official_variant(case: SliceCase, candidate_name: str, kwargs: Dict[str, Any], frames) -> Dict[str, Any]:
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
    regime = model.get_regime_info()
    return {
        "candidate": candidate_name,
        "mae": float(_compute_metrics(y_test.to_numpy(dtype=np.float64), preds)["mae"]),
        "runtime_mode": regime.get("runtime", {}).get("runtime_mode", "unknown"),
        "investors_source_activation": regime.get("investors_source_activation", {}),
        "investor_mark_activation": regime.get("investor_mark_activation", {}),
        "investors_process_contract": regime.get("investors_process_contract", {}),
    }


def _build_dynamic_cases(horizons: tuple[int, ...], entity_limit: int, max_rows_per_ablation: int) -> tuple[RealSurfaceCase, ...]:
    return tuple(
        RealSurfaceCase(
            task="task2_forecast",
            horizon=int(horizon),
            entity_selection="dynamic_test",
            dynamic_entity_limit=entity_limit,
            max_rows_per_ablation=max_rows_per_ablation,
        )
        for horizon in horizons
    )


def _run_dynamic_case(case: RealSurfaceCase) -> Dict[str, Any]:
    raw_frames, selection_report = _load_real_case_frames(case)
    surfaces = {ablation: _split_and_prepare(case, ablation, frame) for ablation, frame in raw_frames.items()}
    encoded_by_backbone: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for candidate in TRACK_CANDIDATES.values():
        backbone_kwargs = candidate.get("dynamic_backbone")
        key = _dynamic_backbone_key(backbone_kwargs)
        if key in encoded_by_backbone:
            continue
        encoded, _ = _fit_shared_encoder(case, surfaces, backbone_kwargs=backbone_kwargs)
        encoded_by_backbone[key] = (_concat_surface(encoded, "train"), _concat_surface(encoded, "test"))

    variants: Dict[str, Any] = {}
    for candidate_name, candidate in TRACK_CANDIDATES.items():
        backbone_key = _dynamic_backbone_key(candidate.get("dynamic_backbone"))
        train_surface, test_surface = encoded_by_backbone[backbone_key]
        result = _run_dynamic_variant(case, candidate_name, candidate["dynamic_variant"], train_surface, test_surface)
        variants[candidate_name] = {
            "mae": float(result["overall_metrics"]["mae"]),
            "lane_horizon_anchor_mix": float(result["lane_horizon_anchor_mix"]),
            "lane_horizon_anchor_mix_reliability": float(result["lane_horizon_anchor_mix_reliability"]),
            "lane_anchor_blend": float(result["lane_anchor_blend"]),
            "lane_jump_strength": float(result["lane_jump_strength"]),
            "lane_flags": result["lane_flags"],
            "dynamic_backbone": candidate.get("dynamic_backbone", {}),
        }

    return {
        "case": asdict(case),
        "selected_entities": selection_report["selected_entities"],
        "variants": variants,
    }


def _run_official_panel(temporal_config) -> Dict[str, Any]:
    official_cases_report: Dict[str, Dict[str, Any]] = {}
    for case in OFFICIAL_CASES:
        frames = _load_case_frames(case, temporal_config)
        case_report = {
            "case": asdict(case),
            "variants": {},
        }
        for candidate_name, candidate in TRACK_CANDIDATES.items():
            case_report["variants"][candidate_name] = _run_official_variant(
                case,
                candidate_name,
                candidate["official_kwargs"],
                frames,
            )
        official_cases_report[case.name] = case_report

    official_summary = _aggregate_candidate_deltas(official_cases_report, baseline_key="legacy_baseline")
    official_alignment = _official_alignment(official_cases_report)
    return {
        "official_cases": official_cases_report,
        "official_summary": official_summary,
        "official_alignment": official_alignment,
    }


def _run_dynamic_panel(dynamic_horizons: tuple[int, ...], entity_limit: int, max_rows_per_ablation: int) -> Dict[str, Any]:
    dynamic_cases_report: Dict[str, Dict[str, Any]] = {}
    for case in _build_dynamic_cases(dynamic_horizons, entity_limit, max_rows_per_ablation):
        dynamic_cases_report[f"{case.task}__dynamic_test__h{case.horizon}"] = _run_dynamic_case(case)

    dynamic_summary = _aggregate_candidate_deltas(dynamic_cases_report, baseline_key="legacy_baseline")
    return {
        "dynamic_cases": dynamic_cases_report,
        "dynamic_summary": dynamic_summary,
    }


def _merge_panel_reports(official_report: Dict[str, Any], dynamic_report: Dict[str, Any]) -> Dict[str, Any]:
    missing_official = [
        key
        for key in ("official_cases", "official_summary", "official_alignment")
        if key not in official_report
    ]
    missing_dynamic = [key for key in ("dynamic_cases", "dynamic_summary") if key not in dynamic_report]
    if missing_official:
        raise ValueError(f"Official report is missing keys: {', '.join(missing_official)}")
    if missing_dynamic:
        raise ValueError(f"Dynamic report is missing keys: {', '.join(missing_dynamic)}")

    return {
        "track_contract": _track_contract(),
        "candidates": _candidate_manifest(),
        "official_cases": official_report["official_cases"],
        "official_summary": official_report["official_summary"],
        "official_alignment": official_report["official_alignment"],
        "dynamic_cases": dynamic_report["dynamic_cases"],
        "dynamic_summary": dynamic_report["dynamic_summary"],
        "gate_verdict": _gate_verdict(
            official_report["official_summary"],
            dynamic_report["dynamic_summary"],
            official_report["official_alignment"],
        ),
    }


def _aggregate_candidate_deltas(cases: Dict[str, Dict[str, Any]], baseline_key: str) -> Dict[str, Dict[str, Any]]:
    candidate_rows: Dict[str, list[float]] = {name: [] for name in TRACK_CANDIDATES}
    for payload in cases.values():
        baseline_mae = float(payload["variants"][baseline_key]["mae"])
        for candidate_name, candidate_payload in payload["variants"].items():
            mae = float(candidate_payload["mae"])
            delta_pct = float(100.0 * (baseline_mae - mae) / baseline_mae) if baseline_mae else 0.0
            candidate_payload["delta_pct_vs_baseline"] = delta_pct
            candidate_rows[candidate_name].append(delta_pct)

    summary: Dict[str, Dict[str, Any]] = {}
    for candidate_name, deltas in candidate_rows.items():
        array = np.asarray(deltas, dtype=np.float64)
        summary[candidate_name] = {
            "case_count": int(array.size),
            "mean_mae_delta_pct": float(np.mean(array)) if array.size else 0.0,
            "worst_mae_delta_pct": float(np.min(array)) if array.size else 0.0,
            "best_mae_delta_pct": float(np.max(array)) if array.size else 0.0,
            "positive_case_count": int(np.sum(array > 1e-9)),
            "negative_case_count": int(np.sum(array < -1e-9)),
        }
    return summary


def _official_alignment(cases: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for candidate_name in TRACK_CANDIDATES:
        runtime_modes = []
        transition_reasons: Dict[str, int] = {}
        mark_reasons: Dict[str, int] = {}
        effective_transition_count = 0
        effective_mark_count = 0
        for payload in cases.values():
            variant = payload["variants"][candidate_name]
            runtime_modes.append(str(variant.get("runtime_mode", "unknown")))
            source_activation = variant.get("investors_source_activation", {})
            reason = str(source_activation.get("transition_activation_reason", "not_applicable"))
            transition_reasons[reason] = transition_reasons.get(reason, 0) + 1
            if bool(source_activation.get("effective_transition_correction", False)):
                effective_transition_count += 1
            mark_activation = variant.get("investor_mark_activation", {})
            mark_reason = str(mark_activation.get("activation_reason", "not_applicable"))
            mark_reasons[mark_reason] = mark_reasons.get(mark_reason, 0) + 1
            if bool(mark_activation.get("effective_mark_features", False)):
                effective_mark_count += 1
        out[candidate_name] = {
            "native_runtime_all_cases": all(mode == "native" for mode in runtime_modes),
            "runtime_modes": runtime_modes,
            "effective_transition_case_count": int(effective_transition_count),
            "transition_reason_counts": transition_reasons,
            "effective_mark_case_count": int(effective_mark_count),
            "mark_reason_counts": mark_reasons,
        }
    return out


def _gate_verdict(official_summary: Dict[str, Dict[str, Any]], dynamic_summary: Dict[str, Dict[str, Any]], official_alignment: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    verdict: Dict[str, Dict[str, Any]] = {}
    for candidate_name, candidate in TRACK_CANDIDATES.items():
        official = official_summary[candidate_name]
        dynamic = dynamic_summary[candidate_name]
        alignment = official_alignment[candidate_name]
        official_pass = (
            alignment["native_runtime_all_cases"]
            and official["mean_mae_delta_pct"] >= OFFICIAL_MEAN_DELTA_MIN_PCT
            and official["worst_mae_delta_pct"] >= OFFICIAL_WORST_DELTA_MIN_PCT
        )
        dynamic_pass = dynamic["mean_mae_delta_pct"] > DYNAMIC_MEAN_DELTA_MIN_PCT
        verdict[candidate_name] = {
            "role": str(candidate["role"]),
            "official_track_pass": bool(official_pass),
            "dynamic_track_pass": bool(dynamic_pass),
            "promotable_on_current_track": bool(official_pass and dynamic_pass),
        }
    return verdict


def main() -> int:
    args = _parse_args()
    report: Dict[str, Any]
    if args.mode == "merge":
        if args.official_input_json is None or args.dynamic_input_json is None:
            raise ValueError("--mode merge requires both --official-input-json and --dynamic-input-json")
        official_report = json.loads(args.official_input_json.read_text())
        dynamic_report = json.loads(args.dynamic_input_json.read_text())
        report = _merge_panel_reports(official_report, dynamic_report)
    else:
        report = {
            "track_contract": _track_contract(),
            "candidates": _candidate_manifest(),
        }
        if args.mode in ("full", "official"):
            temporal_config = _make_temporal_config()
            report.update(_run_official_panel(temporal_config))
        if args.mode in ("full", "dynamic"):
            dynamic_horizons = _parse_horizons(args.dynamic_horizons)
            report.update(_run_dynamic_panel(dynamic_horizons, args.dynamic_entity_limit, args.dynamic_max_rows_per_ablation))
        if args.mode == "full":
            report["gate_verdict"] = _gate_verdict(
                report["official_summary"],
                report["dynamic_summary"],
                report["official_alignment"],
            )

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())