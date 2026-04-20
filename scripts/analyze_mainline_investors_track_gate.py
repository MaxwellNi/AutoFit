#!/usr/bin/env python3
"""Gate active mainline investors candidates on official local slices and dynamic real multisource surfaces."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict

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
DYNAMIC_GEOMETRY_SHIFT_AMPLIFICATION_MAX = 128.0
DYNAMIC_GEOMETRY_TEST_SHIFT_SHARE_MAX = 0.05
ACTIVE_GENERATION_FOCUS = "selective_event_state_guard"
ACTIVE_GENERATION_RUNTIME_ALIAS = "single_model_mainline_track_active_generation_focus"


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
    "process_state_feedback_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_process_state_feedback_guard",
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
            "enable_process_state_feedback": True,
            "process_state_feedback_strength": 0.02,
            "process_state_feedback_source_decay": 0.65,
            "process_state_feedback_min_horizon": 7,
            "process_state_feedback_state_weights": (0.30, 0.20, 0.0, 0.0, 0.50),
        },
    },
    "temporal_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_temporal_state_guard",
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
            "enable_temporal_state_features": True,
            "enable_spectral_state_features": False,
        },
    },
    "spectral_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_spectral_state_guard",
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
            "enable_temporal_state_features": False,
            "enable_spectral_state_features": True,
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
            "enable_temporal_state_features": True,
            "enable_spectral_state_features": True,
        },
    },
    "hawkes_financing_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_hawkes_financing_state_guard",
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
            "enable_hawkes_financing_state": True,
            "hawkes_financing_decay_halflives": (7.0, 30.0, 90.0),
            "hawkes_positive_shock_threshold": 0.5,
        },
    },
    "jump_ode_state_guard": {
        "role": "active_trunk_candidate",
        "official_kwargs": {
            "variant": "mainline_jump_ode_state_guard",
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
                "enable_intensity_baseline": True,
                "intensity_blend": 0.5,
                "enable_funding_gpd_tail": True,
                "enable_funding_cqr_interval": True,
                "funding_cqr_alpha": 0.10,
            },
            "predict": {
                "enable_investors_event_state_features": True,
                "enable_intensity_baseline": True,
            },
        },
        "dynamic_backbone": {
            "enable_jump_ode_state": True,
            "jump_ode_dims": 8,
        },
    },
    "shrinkage_gate_guard": {
        "role": "active_lane_candidate",
        "official_kwargs": {
            "variant": "mainline_shrinkage_gate_guard",
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
                "enable_intensity_baseline": True,
                "intensity_blend": 0.5,
                "enable_shrinkage_gate": True,
                "shrinkage_strength": 0.8,
            },
            "predict": {
                "enable_investors_event_state_features": True,
                "enable_intensity_baseline": True,
            },
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

FULL_TRACK_CANDIDATES = TRACK_CANDIDATES

PROCESS_STATE_FAMILIES = (
    "attention_diffusion",
    "credibility_confirmation",
    "screening_selectivity",
    "book_depth_absorption",
    "closure_conversion",
)
PROCESS_STATE_COUPLINGS = {
    "temporal_velocity": "temporal_velocity_coupling",
    "temporal_shock": "temporal_shock_coupling",
    "spectral_low_band": "spectral_low_band_coupling",
    "spectral_high_band": "spectral_high_band_coupling",
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
        "--candidates",
        default=None,
        help=(
            "Optional comma-separated candidate list to audit. "
            "The script always keeps legacy_baseline and the active generation focus for comparison."
        ),
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


def _select_track_candidates(spec: str | None) -> Dict[str, Dict[str, Any]]:
    if spec is None or not str(spec).strip():
        return FULL_TRACK_CANDIDATES

    requested: list[str] = []
    for token in str(spec).split(","):
        name = token.strip()
        if not name:
            continue
        if name not in FULL_TRACK_CANDIDATES:
            raise ValueError(f"Unknown candidate '{name}'")
        if name not in requested:
            requested.append(name)

    selected_names: list[str] = []
    for name in ("legacy_baseline", ACTIVE_GENERATION_FOCUS, *requested):
        if name not in FULL_TRACK_CANDIDATES:
            continue
        if name not in selected_names:
            selected_names.append(name)

    if len(selected_names) < 2:
        raise ValueError("Candidate filtering must retain at least baseline and one comparison candidate")

    return {name: FULL_TRACK_CANDIDATES[name] for name in selected_names}


def _track_contract() -> Dict[str, Any]:
    return {
        "runtime_owner": "single_model_mainline",
        "native_runtime_required": True,
        "delegate_forbidden": True,
        "official_panel_definition": "benchmark-consistent local official slices loaded via _load_smoke_frame plus canonical temporal split from block3 task config",
        "dynamic_panel_definition": "real-data dynamic_test multisource challenge surfaces; research-only and never merged into official claims",
        "source_results_interpreted_only_with_geometry": True,
        "demoted_families": ["event_state_boundary_guard", "source_policy_transition_guard"],
        "active_generation_focus": ACTIVE_GENERATION_FOCUS,
        "active_generation_variant": str(
            TRACK_CANDIDATES[ACTIVE_GENERATION_FOCUS]["official_kwargs"].get("variant", "mainline_alpha")
        ),
        "active_generation_runtime_alias": ACTIVE_GENERATION_RUNTIME_ALIAS,
        "promotion_rules": {
            "official_mean_mae_delta_pct_min": OFFICIAL_MEAN_DELTA_MIN_PCT,
            "official_worst_mae_delta_pct_min": OFFICIAL_WORST_DELTA_MIN_PCT,
            "dynamic_mean_mae_delta_pct_min": DYNAMIC_MEAN_DELTA_MIN_PCT,
            "dynamic_geometry_shift_amplification_max": DYNAMIC_GEOMETRY_SHIFT_AMPLIFICATION_MAX,
            "dynamic_geometry_test_shift_share_max": DYNAMIC_GEOMETRY_TEST_SHIFT_SHARE_MAX,
        },
        "active_candidate_always_included_in_filtered_runs": True,
    }


def _candidate_manifest() -> Dict[str, Dict[str, Any]]:
    return {name: {"role": payload["role"]} for name, payload in TRACK_CANDIDATES.items()}


def _candidate_variant_name(candidate_name: str) -> str | None:
    payload = TRACK_CANDIDATES.get(candidate_name, FULL_TRACK_CANDIDATES.get(candidate_name, {}))
    official_kwargs = payload.get("official_kwargs", {})
    variant = official_kwargs.get("variant")
    if variant is None:
        if candidate_name == "legacy_baseline":
            return "mainline_alpha"
        return None
    return str(variant)


def _empty_process_state_summary() -> Dict[str, Any]:
    return {
        "case_count": 0,
        "mean_scores": {family: 0.0 for family in PROCESS_STATE_FAMILIES},
        "mean_support_shares": {family: 0.0 for family in PROCESS_STATE_FAMILIES},
        "mean_couplings": {name: 0.0 for name in PROCESS_STATE_COUPLINGS},
        "mean_score_deltas_vs_baseline": {family: 0.0 for family in PROCESS_STATE_FAMILIES},
        "mean_score_shift_l1": 0.0,
        "top_positive_state": "none",
        "top_negative_state": "none",
        "multiscale_coupling_case_count": 0,
    }


def _normalize_process_state_summary(summary: Dict[str, Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for candidate_name in TRACK_CANDIDATES:
        payload = summary.get(candidate_name, {}) if isinstance(summary, dict) else {}
        empty = _empty_process_state_summary()
        normalized[candidate_name] = {
            **empty,
            **payload,
            "mean_scores": {
                **empty["mean_scores"],
                **dict(payload.get("mean_scores", {})),
            },
            "mean_support_shares": {
                **empty["mean_support_shares"],
                **dict(payload.get("mean_support_shares", {})),
            },
            "mean_couplings": {
                **empty["mean_couplings"],
                **dict(payload.get("mean_couplings", {})),
            },
            "mean_score_deltas_vs_baseline": {
                **empty["mean_score_deltas_vs_baseline"],
                **dict(payload.get("mean_score_deltas_vs_baseline", {})),
            },
        }
    return normalized


def _extract_process_state_atoms(card: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    atoms = card.get("process_state_atoms")
    if not isinstance(atoms, dict):
        return None
    return atoms


def _snapshot_from_process_state_atoms(atoms: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(atoms, dict):
        return None
    return {
        "scores": {
            family: float(atoms.get(f"{family}_score", 0.0))
            for family in PROCESS_STATE_FAMILIES
        },
        "support_shares": {
            family: float(atoms.get(f"{family}_support_share", 0.0))
            for family in PROCESS_STATE_FAMILIES
        },
        "couplings": {
            name: float(atoms.get(key, 0.0))
            for name, key in PROCESS_STATE_COUPLINGS.items()
        },
        "multiscale_coupling_enabled": bool(atoms.get("multiscale_coupling_enabled", False)),
    }


def _snapshot_from_process_state_summary(summary: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    score_map = summary.get("mean_scores")
    support_map = summary.get("mean_support_shares")
    coupling_map = summary.get("mean_couplings")
    if not isinstance(score_map, dict) or not isinstance(support_map, dict) or not isinstance(coupling_map, dict):
        return None
    return {
        "scores": {
            family: float(score_map.get(family, 0.0))
            for family in PROCESS_STATE_FAMILIES
        },
        "support_shares": {
            family: float(support_map.get(family, 0.0))
            for family in PROCESS_STATE_FAMILIES
        },
        "couplings": {
            name: float(coupling_map.get(name, 0.0))
            for name in PROCESS_STATE_COUPLINGS
        },
        "multiscale_coupling_enabled": bool(summary.get("multiscale_coupling_enabled", False)),
    }


def _dominant_process_state(score_deltas: Dict[str, float], *, prefer_positive: bool) -> str:
    if not score_deltas:
        return "none"
    items = score_deltas.items()
    family, delta = max(items, key=lambda item: item[1]) if prefer_positive else min(items, key=lambda item: item[1])
    return family if abs(float(delta)) > 1e-9 else "none"


def _aggregate_process_state_summaries(
    cases: Dict[str, Dict[str, Any]],
    *,
    baseline_key: str,
    snapshot_getter: Callable[[Dict[str, Any]], Dict[str, Any] | None],
) -> Dict[str, Dict[str, Any]]:
    buffers: Dict[str, Dict[str, Any]] = {
        candidate_name: {
            "case_count": 0,
            "scores": {family: [] for family in PROCESS_STATE_FAMILIES},
            "support_shares": {family: [] for family in PROCESS_STATE_FAMILIES},
            "couplings": {name: [] for name in PROCESS_STATE_COUPLINGS},
            "score_deltas": {family: [] for family in PROCESS_STATE_FAMILIES},
            "score_shift_l1": [],
            "multiscale_coupling_case_count": 0,
        }
        for candidate_name in TRACK_CANDIDATES
    }

    for payload in cases.values():
        variants = payload.get("variants", {})
        baseline_variant = variants.get(baseline_key)
        if not isinstance(baseline_variant, dict):
            continue
        baseline_snapshot = snapshot_getter(baseline_variant)
        if baseline_snapshot is None:
            continue
        for candidate_name in TRACK_CANDIDATES:
            variant_payload = variants.get(candidate_name)
            if not isinstance(variant_payload, dict):
                continue
            snapshot = snapshot_getter(variant_payload)
            if snapshot is None:
                continue
            buffer = buffers[candidate_name]
            buffer["case_count"] += 1
            buffer["multiscale_coupling_case_count"] += int(snapshot["multiscale_coupling_enabled"])
            shift_l1 = 0.0
            for family in PROCESS_STATE_FAMILIES:
                score = float(snapshot["scores"][family])
                support = float(snapshot["support_shares"][family])
                delta = score - float(baseline_snapshot["scores"][family])
                buffer["scores"][family].append(score)
                buffer["support_shares"][family].append(support)
                buffer["score_deltas"][family].append(delta)
                shift_l1 += abs(delta)
            for name in PROCESS_STATE_COUPLINGS:
                buffer["couplings"][name].append(float(snapshot["couplings"][name]))
            buffer["score_shift_l1"].append(shift_l1)

    summary: Dict[str, Dict[str, Any]] = {}
    for candidate_name, buffer in buffers.items():
        case_count = int(buffer["case_count"])
        if case_count <= 0:
            summary[candidate_name] = _empty_process_state_summary()
            continue
        mean_scores = {
            family: float(np.mean(buffer["scores"][family]))
            for family in PROCESS_STATE_FAMILIES
        }
        mean_support_shares = {
            family: float(np.mean(buffer["support_shares"][family]))
            for family in PROCESS_STATE_FAMILIES
        }
        mean_couplings = {
            name: float(np.mean(buffer["couplings"][name]))
            for name in PROCESS_STATE_COUPLINGS
        }
        mean_score_deltas = {
            family: float(np.mean(buffer["score_deltas"][family]))
            for family in PROCESS_STATE_FAMILIES
        }
        summary[candidate_name] = {
            "case_count": case_count,
            "mean_scores": mean_scores,
            "mean_support_shares": mean_support_shares,
            "mean_couplings": mean_couplings,
            "mean_score_deltas_vs_baseline": mean_score_deltas,
            "mean_score_shift_l1": float(np.mean(buffer["score_shift_l1"])),
            "top_positive_state": _dominant_process_state(mean_score_deltas, prefer_positive=True),
            "top_negative_state": _dominant_process_state(mean_score_deltas, prefer_positive=False),
            "multiscale_coupling_case_count": int(buffer["multiscale_coupling_case_count"]),
        }
    return summary


def _official_process_state_snapshot(variant_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    return _snapshot_from_process_state_atoms(_extract_process_state_atoms(variant_payload.get("event_state_trunk")))


def _dynamic_process_state_snapshot(variant_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    process_state = variant_payload.get("dynamic_process_state")
    if not isinstance(process_state, dict):
        return None
    return _snapshot_from_process_state_summary(process_state.get("test"))


def _aggregate_official_process_state(cases: Dict[str, Dict[str, Any]], baseline_key: str) -> Dict[str, Dict[str, Any]]:
    return _aggregate_process_state_summaries(
        cases,
        baseline_key=baseline_key,
        snapshot_getter=_official_process_state_snapshot,
    )


def _aggregate_dynamic_process_state(cases: Dict[str, Dict[str, Any]], baseline_key: str) -> Dict[str, Dict[str, Any]]:
    return _aggregate_process_state_summaries(
        cases,
        baseline_key=baseline_key,
        snapshot_getter=_dynamic_process_state_snapshot,
    )


def _surface_process_state_report(encoded: Dict[str, Dict[str, Any]], split: str) -> Dict[str, Any]:
    total_rows = 0
    score_totals = {family: 0.0 for family in PROCESS_STATE_FAMILIES}
    support_totals = {family: 0.0 for family in PROCESS_STATE_FAMILIES}
    coupling_totals = {name: 0.0 for name in PROCESS_STATE_COUPLINGS}
    multiscale_coupling_enabled = False

    for payload in encoded.values():
        atoms = _extract_process_state_atoms(payload.get(f"event_state_trunk_{split}"))
        snapshot = _snapshot_from_process_state_atoms(atoms)
        row_count = int(len(payload.get(f"y_{split}", ())))
        if snapshot is None or row_count <= 0:
            continue
        total_rows += row_count
        multiscale_coupling_enabled = multiscale_coupling_enabled or snapshot["multiscale_coupling_enabled"]
        for family in PROCESS_STATE_FAMILIES:
            score_totals[family] += row_count * float(snapshot["scores"][family])
            support_totals[family] += row_count * float(snapshot["support_shares"][family])
        for name in PROCESS_STATE_COUPLINGS:
            coupling_totals[name] += row_count * float(snapshot["couplings"][name])

    if total_rows <= 0:
        return {
            "rows": 0,
            "mean_scores": {family: 0.0 for family in PROCESS_STATE_FAMILIES},
            "mean_support_shares": {family: 0.0 for family in PROCESS_STATE_FAMILIES},
            "mean_couplings": {name: 0.0 for name in PROCESS_STATE_COUPLINGS},
            "multiscale_coupling_enabled": False,
        }
    return {
        "rows": int(total_rows),
        "mean_scores": {
            family: float(score_totals[family] / total_rows)
            for family in PROCESS_STATE_FAMILIES
        },
        "mean_support_shares": {
            family: float(support_totals[family] / total_rows)
            for family in PROCESS_STATE_FAMILIES
        },
        "mean_couplings": {
            name: float(coupling_totals[name] / total_rows)
            for name in PROCESS_STATE_COUPLINGS
        },
        "multiscale_coupling_enabled": bool(multiscale_coupling_enabled),
    }


def _maybe_official_process_state_summary(official_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    provided = official_report.get("official_process_state_summary")
    normalized = _normalize_process_state_summary(provided)
    if any(payload["case_count"] > 0 for payload in normalized.values()):
        return normalized
    cases = official_report.get("official_cases", {})
    if isinstance(cases, dict) and cases:
        computed = _aggregate_official_process_state(cases, baseline_key="legacy_baseline")
        normalized = _normalize_process_state_summary(computed)
        if any(payload["case_count"] > 0 for payload in normalized.values()):
            return normalized
    return _normalize_process_state_summary(None)


def _maybe_dynamic_process_state_summary(dynamic_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    provided = dynamic_report.get("dynamic_process_state_summary")
    normalized = _normalize_process_state_summary(provided)
    if any(payload["case_count"] > 0 for payload in normalized.values()):
        return normalized
    cases = dynamic_report.get("dynamic_cases", {})
    if isinstance(cases, dict) and cases:
        computed = _aggregate_dynamic_process_state(cases, baseline_key="legacy_baseline")
        normalized = _normalize_process_state_summary(computed)
        if any(payload["case_count"] > 0 for payload in normalized.values()):
            return normalized
    return _normalize_process_state_summary(None)


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1e-9:
        return 0.0 if abs(float(numerator)) <= 1e-9 else float("inf")
    return float(float(numerator) / float(denominator))


def _lane_geometry_stats(lane: np.ndarray) -> Dict[str, float]:
    state = np.asarray(lane, dtype=np.float64)
    if state.ndim != 2 or state.size == 0:
        return {
            "rows": 0,
            "lane_dim": 0,
            "abs_mean": 0.0,
            "p95_abs": 0.0,
            "row_l2_mean": 0.0,
            "row_l2_p90": 0.0,
        }
    abs_state = np.abs(state)
    row_l2 = np.sqrt(np.sum(np.square(state), axis=1))
    return {
        "rows": int(state.shape[0]),
        "lane_dim": int(state.shape[1]),
        "abs_mean": float(np.mean(abs_state)),
        "p95_abs": float(np.quantile(abs_state, 0.95)),
        "row_l2_mean": float(np.mean(row_l2)),
        "row_l2_p90": float(np.quantile(row_l2, 0.90)),
    }


def _dynamic_geometry_report(
    train_surface: Dict[str, Any],
    test_surface: Dict[str, Any],
    *,
    baseline_train_surface: Dict[str, Any] | None = None,
    baseline_test_surface: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    train_lane = np.asarray(train_surface["lane"], dtype=np.float64)
    test_lane = np.asarray(test_surface["lane"], dtype=np.float64)
    train_stats = _lane_geometry_stats(train_lane)
    test_stats = _lane_geometry_stats(test_lane)

    baseline_relative = {
        "shape_aligned": True,
        "train_shift_abs_mean": 0.0,
        "test_shift_abs_mean": 0.0,
        "train_shift_p95_abs": 0.0,
        "test_shift_p95_abs": 0.0,
        "train_shift_abs_mean_share": 0.0,
        "test_shift_abs_mean_share": 0.0,
        "train_shift_p95_share": 0.0,
        "test_shift_p95_share": 0.0,
        "shift_abs_mean_amplification": 0.0,
        "shift_p95_amplification": 0.0,
    }
    if baseline_train_surface is not None and baseline_test_surface is not None:
        baseline_train_lane = np.asarray(baseline_train_surface["lane"], dtype=np.float64)
        baseline_test_lane = np.asarray(baseline_test_surface["lane"], dtype=np.float64)
        if baseline_train_lane.shape != train_lane.shape or baseline_test_lane.shape != test_lane.shape:
            baseline_relative.update(
                {
                    "shape_aligned": False,
                    "train_shift_abs_mean": float("inf"),
                    "test_shift_abs_mean": float("inf"),
                    "train_shift_p95_abs": float("inf"),
                    "test_shift_p95_abs": float("inf"),
                    "train_shift_abs_mean_share": float("inf"),
                    "test_shift_abs_mean_share": float("inf"),
                    "train_shift_p95_share": float("inf"),
                    "test_shift_p95_share": float("inf"),
                    "shift_abs_mean_amplification": float("inf"),
                    "shift_p95_amplification": float("inf"),
                }
            )
        else:
            train_shift = np.abs(train_lane - baseline_train_lane)
            test_shift = np.abs(test_lane - baseline_test_lane)
            train_shift_abs_mean = float(np.mean(train_shift))
            test_shift_abs_mean = float(np.mean(test_shift))
            train_shift_p95_abs = float(np.quantile(train_shift, 0.95))
            test_shift_p95_abs = float(np.quantile(test_shift, 0.95))
            train_shift_abs_mean_share = _safe_ratio(train_shift_abs_mean, train_stats["abs_mean"])
            test_shift_abs_mean_share = _safe_ratio(test_shift_abs_mean, test_stats["abs_mean"])
            train_shift_p95_share = _safe_ratio(train_shift_p95_abs, train_stats["p95_abs"])
            test_shift_p95_share = _safe_ratio(test_shift_p95_abs, test_stats["p95_abs"])
            baseline_relative.update(
                {
                    "train_shift_abs_mean": train_shift_abs_mean,
                    "test_shift_abs_mean": test_shift_abs_mean,
                    "train_shift_p95_abs": train_shift_p95_abs,
                    "test_shift_p95_abs": test_shift_p95_abs,
                    "train_shift_abs_mean_share": train_shift_abs_mean_share,
                    "test_shift_abs_mean_share": test_shift_abs_mean_share,
                    "train_shift_p95_share": train_shift_p95_share,
                    "test_shift_p95_share": test_shift_p95_share,
                    "shift_abs_mean_amplification": _safe_ratio(
                        test_shift_abs_mean_share,
                        train_shift_abs_mean_share,
                    ),
                    "shift_p95_amplification": _safe_ratio(
                        test_shift_p95_share,
                        train_shift_p95_share,
                    ),
                }
            )

    geometry_pass = bool(
        baseline_relative["shape_aligned"]
        and baseline_relative["shift_abs_mean_amplification"] <= DYNAMIC_GEOMETRY_SHIFT_AMPLIFICATION_MAX
        and baseline_relative["test_shift_abs_mean_share"] <= DYNAMIC_GEOMETRY_TEST_SHIFT_SHARE_MAX
    )
    return {
        "train_lane": train_stats,
        "test_lane": test_stats,
        "train_test_abs_mean_ratio": _safe_ratio(test_stats["abs_mean"], train_stats["abs_mean"]),
        "train_test_p95_abs_ratio": _safe_ratio(test_stats["p95_abs"], train_stats["p95_abs"]),
        "baseline_relative": baseline_relative,
        "geometry_pass": geometry_pass,
    }


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
        "event_state_trunk": regime.get("event_state_trunk", {}),
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
    encoded_by_backbone: Dict[str, Dict[str, Any]] = {}
    for candidate in TRACK_CANDIDATES.values():
        backbone_kwargs = candidate.get("dynamic_backbone")
        key = _dynamic_backbone_key(backbone_kwargs)
        if key in encoded_by_backbone:
            continue
        encoded, _ = _fit_shared_encoder(case, surfaces, backbone_kwargs=backbone_kwargs)
        encoded_by_backbone[key] = {
            "train_surface": _concat_surface(encoded, "train"),
            "test_surface": _concat_surface(encoded, "test"),
            "dynamic_process_state": {
                "train": _surface_process_state_report(encoded, "train"),
                "test": _surface_process_state_report(encoded, "test"),
            },
        }

    baseline_backbone = encoded_by_backbone[_dynamic_backbone_key(None)]
    for payload in encoded_by_backbone.values():
        payload["dynamic_geometry"] = _dynamic_geometry_report(
            payload["train_surface"],
            payload["test_surface"],
            baseline_train_surface=baseline_backbone["train_surface"],
            baseline_test_surface=baseline_backbone["test_surface"],
        )

    variants: Dict[str, Any] = {}
    for candidate_name, candidate in TRACK_CANDIDATES.items():
        backbone_key = _dynamic_backbone_key(candidate.get("dynamic_backbone"))
        backbone_payload = encoded_by_backbone[backbone_key]
        train_surface = backbone_payload["train_surface"]
        test_surface = backbone_payload["test_surface"]
        result = _run_dynamic_variant(case, candidate_name, candidate["dynamic_variant"], train_surface, test_surface)
        variants[candidate_name] = {
            "mae": float(result["overall_metrics"]["mae"]),
            "lane_horizon_anchor_mix": float(result["lane_horizon_anchor_mix"]),
            "lane_horizon_anchor_mix_reliability": float(result["lane_horizon_anchor_mix_reliability"]),
            "lane_anchor_blend": float(result["lane_anchor_blend"]),
            "lane_jump_strength": float(result["lane_jump_strength"]),
            "lane_flags": result["lane_flags"],
            "dynamic_backbone": candidate.get("dynamic_backbone", {}),
            "dynamic_geometry": backbone_payload["dynamic_geometry"],
            "dynamic_process_state": backbone_payload["dynamic_process_state"],
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
    official_process_state_summary = _aggregate_official_process_state(
        official_cases_report,
        baseline_key="legacy_baseline",
    )
    return {
        "official_cases": official_cases_report,
        "official_summary": official_summary,
        "official_alignment": official_alignment,
        "official_process_state_summary": official_process_state_summary,
    }


def _run_dynamic_panel(dynamic_horizons: tuple[int, ...], entity_limit: int, max_rows_per_ablation: int) -> Dict[str, Any]:
    dynamic_cases_report: Dict[str, Dict[str, Any]] = {}
    for case in _build_dynamic_cases(dynamic_horizons, entity_limit, max_rows_per_ablation):
        dynamic_cases_report[f"{case.task}__dynamic_test__h{case.horizon}"] = _run_dynamic_case(case)

    dynamic_summary = _aggregate_candidate_deltas(dynamic_cases_report, baseline_key="legacy_baseline")
    dynamic_geometry_summary = _aggregate_dynamic_geometry(dynamic_cases_report)
    dynamic_process_state_summary = _aggregate_dynamic_process_state(
        dynamic_cases_report,
        baseline_key="legacy_baseline",
    )
    return {
        "dynamic_cases": dynamic_cases_report,
        "dynamic_summary": dynamic_summary,
        "dynamic_geometry_summary": dynamic_geometry_summary,
        "dynamic_process_state_summary": dynamic_process_state_summary,
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

    dynamic_geometry_summary = dynamic_report.get("dynamic_geometry_summary")
    if dynamic_geometry_summary is None:
        dynamic_geometry_summary = _aggregate_dynamic_geometry(dynamic_report["dynamic_cases"])
    official_process_state_summary = _maybe_official_process_state_summary(official_report)
    dynamic_process_state_summary = _maybe_dynamic_process_state_summary(dynamic_report)

    gate_verdict = _gate_verdict(
        official_report["official_summary"],
        dynamic_report["dynamic_summary"],
        official_report["official_alignment"],
        dynamic_geometry_summary,
        official_process_state_summary,
        dynamic_process_state_summary,
    )

    return {
        "track_contract": _track_contract(),
        "candidates": _candidate_manifest(),
        "official_cases": official_report["official_cases"],
        "official_summary": official_report["official_summary"],
        "official_alignment": official_report["official_alignment"],
        "official_process_state_summary": official_process_state_summary,
        "dynamic_cases": dynamic_report["dynamic_cases"],
        "dynamic_summary": dynamic_report["dynamic_summary"],
        "dynamic_geometry_summary": dynamic_geometry_summary,
        "dynamic_process_state_summary": dynamic_process_state_summary,
        "gate_verdict": gate_verdict,
        "promotion_recommendation": _build_promotion_recommendation(
            gate_verdict,
            official_report["official_summary"],
            dynamic_report["dynamic_summary"],
            dynamic_geometry_summary,
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


def _aggregate_dynamic_geometry(cases: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for candidate_name in TRACK_CANDIDATES:
        reports = [
            payload["variants"][candidate_name].get("dynamic_geometry", {})
            for payload in cases.values()
        ]
        case_count = len(reports)
        geometry_pass_flags = [bool(report.get("geometry_pass", False)) for report in reports]
        abs_mean_ratios = [float(report.get("train_test_abs_mean_ratio", 0.0)) for report in reports]
        p95_ratios = [float(report.get("train_test_p95_abs_ratio", 0.0)) for report in reports]
        shift_amplifications = [
            float(report.get("baseline_relative", {}).get("shift_abs_mean_amplification", 0.0))
            for report in reports
        ]
        shift_p95_amplifications = [
            float(report.get("baseline_relative", {}).get("shift_p95_amplification", 0.0))
            for report in reports
        ]
        test_shift_shares = [
            float(report.get("baseline_relative", {}).get("test_shift_abs_mean_share", 0.0))
            for report in reports
        ]
        test_shift_p95_shares = [
            float(report.get("baseline_relative", {}).get("test_shift_p95_share", 0.0))
            for report in reports
        ]
        summary[candidate_name] = {
            "case_count": int(case_count),
            "geometry_pass_case_count": int(sum(geometry_pass_flags)),
            "geometry_fail_case_count": int(case_count - sum(geometry_pass_flags)),
            "geometry_all_cases_pass": bool(all(geometry_pass_flags)) if case_count else False,
            "worst_train_test_abs_mean_ratio": float(max(abs_mean_ratios)) if abs_mean_ratios else 0.0,
            "worst_train_test_p95_abs_ratio": float(max(p95_ratios)) if p95_ratios else 0.0,
            "worst_shift_abs_mean_amplification": float(max(shift_amplifications)) if shift_amplifications else 0.0,
            "worst_shift_p95_amplification": float(max(shift_p95_amplifications)) if shift_p95_amplifications else 0.0,
            "worst_test_shift_abs_mean_share": float(max(test_shift_shares)) if test_shift_shares else 0.0,
            "worst_test_shift_p95_share": float(max(test_shift_p95_shares)) if test_shift_p95_shares else 0.0,
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


def _gate_verdict(
    official_summary: Dict[str, Dict[str, Any]],
    dynamic_summary: Dict[str, Dict[str, Any]],
    official_alignment: Dict[str, Dict[str, Any]],
    dynamic_geometry_summary: Dict[str, Dict[str, Any]],
    official_process_state_summary: Dict[str, Dict[str, Any]],
    dynamic_process_state_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    verdict: Dict[str, Dict[str, Any]] = {}
    demoted_families = set(_track_contract().get("demoted_families", ()))
    for candidate_name, candidate in TRACK_CANDIDATES.items():
        official = official_summary[candidate_name]
        dynamic = dynamic_summary[candidate_name]
        alignment = official_alignment[candidate_name]
        geometry = dynamic_geometry_summary[candidate_name]
        official_process_state = official_process_state_summary.get(candidate_name, _empty_process_state_summary())
        dynamic_process_state = dynamic_process_state_summary.get(candidate_name, _empty_process_state_summary())
        official_pass = (
            alignment["native_runtime_all_cases"]
            and official["mean_mae_delta_pct"] >= OFFICIAL_MEAN_DELTA_MIN_PCT
            and official["worst_mae_delta_pct"] >= OFFICIAL_WORST_DELTA_MIN_PCT
        )
        dynamic_metric_pass = dynamic["mean_mae_delta_pct"] > DYNAMIC_MEAN_DELTA_MIN_PCT
        dynamic_geometry_pass = geometry["geometry_all_cases_pass"]
        dynamic_pass = dynamic_metric_pass and dynamic_geometry_pass
        demotion_blocked = candidate_name in demoted_families
        verdict[candidate_name] = {
            "role": str(candidate["role"]),
            "official_track_pass": bool(official_pass),
            "dynamic_metric_pass": bool(dynamic_metric_pass),
            "dynamic_geometry_pass": bool(dynamic_geometry_pass),
            "dynamic_track_pass": bool(dynamic_pass),
            "demotion_blocked": bool(demotion_blocked),
            "promotable_on_current_track": bool(official_pass and dynamic_pass and not demotion_blocked),
            "official_primary_process_state": str(official_process_state["top_positive_state"]),
            "dynamic_primary_process_state": str(dynamic_process_state["top_positive_state"]),
            "official_process_state_shift_l1": float(official_process_state["mean_score_shift_l1"]),
            "dynamic_process_state_shift_l1": float(dynamic_process_state["mean_score_shift_l1"]),
        }
    return verdict


def _primary_gate_blocker(verdict_entry: Dict[str, Any]) -> str:
    if bool(verdict_entry.get("promotable_on_current_track", False)):
        return "none"
    if bool(verdict_entry.get("demotion_blocked", False)):
        return "demotion_blocked"
    if not bool(verdict_entry.get("official_track_pass", False)):
        return "official_track_fail"
    if not bool(verdict_entry.get("dynamic_metric_pass", False)):
        return "dynamic_metric_fail"
    if not bool(verdict_entry.get("dynamic_geometry_pass", False)):
        return "dynamic_geometry_fail"
    return "unknown_gate_fail"


def _candidate_recommendation_snapshot(
    candidate_name: str,
    gate_verdict: Dict[str, Dict[str, Any]],
    official_summary: Dict[str, Dict[str, Any]],
    dynamic_summary: Dict[str, Dict[str, Any]],
    dynamic_geometry_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    verdict_entry = gate_verdict[candidate_name]
    official = official_summary[candidate_name]
    dynamic = dynamic_summary[candidate_name]
    geometry = dynamic_geometry_summary[candidate_name]
    return {
        "candidate": candidate_name,
        "variant": _candidate_variant_name(candidate_name),
        "role": str(verdict_entry.get("role", TRACK_CANDIDATES[candidate_name]["role"])),
        "promotable_on_current_track": bool(verdict_entry.get("promotable_on_current_track", False)),
        "official_mean_mae_delta_pct": float(official.get("mean_mae_delta_pct", 0.0)),
        "dynamic_mean_mae_delta_pct": float(dynamic.get("mean_mae_delta_pct", 0.0)),
        "dynamic_geometry_all_cases_pass": bool(geometry.get("geometry_all_cases_pass", False)),
        "primary_blocker": _primary_gate_blocker(verdict_entry),
    }


def _promotion_sort_key(
    candidate_name: str,
    gate_verdict: Dict[str, Dict[str, Any]],
    official_summary: Dict[str, Dict[str, Any]],
    dynamic_summary: Dict[str, Dict[str, Any]],
    dynamic_geometry_summary: Dict[str, Dict[str, Any]],
) -> tuple[float, ...]:
    verdict_entry = gate_verdict[candidate_name]
    official = official_summary[candidate_name]
    dynamic = dynamic_summary[candidate_name]
    geometry = dynamic_geometry_summary[candidate_name]
    promotable = bool(verdict_entry.get("promotable_on_current_track", False))
    active_focus_bonus = bool(promotable and candidate_name == ACTIVE_GENERATION_FOCUS)
    return (
        float(promotable),
        float(active_focus_bonus),
        float(dynamic.get("mean_mae_delta_pct", 0.0)),
        float(official.get("mean_mae_delta_pct", 0.0)),
        -float(geometry.get("worst_shift_abs_mean_amplification", 0.0)),
        -float(geometry.get("worst_test_shift_abs_mean_share", 0.0)),
    )


def _build_promotion_recommendation(
    gate_verdict: Dict[str, Dict[str, Any]],
    official_summary: Dict[str, Dict[str, Any]],
    dynamic_summary: Dict[str, Dict[str, Any]],
    dynamic_geometry_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ranked_candidates = sorted(
        TRACK_CANDIDATES,
        key=lambda candidate_name: _promotion_sort_key(
            candidate_name,
            gate_verdict,
            official_summary,
            dynamic_summary,
            dynamic_geometry_summary,
        ),
        reverse=True,
    )
    promotable_candidates = [
        candidate_name
        for candidate_name in ranked_candidates
        if bool(gate_verdict[candidate_name].get("promotable_on_current_track", False))
    ]
    recommended_candidate = promotable_candidates[0] if promotable_candidates else None
    runner_up_candidate = promotable_candidates[1] if len(promotable_candidates) > 1 else None
    if recommended_candidate == ACTIVE_GENERATION_FOCUS:
        selection_basis = "preferred_active_generation_focus"
    elif recommended_candidate is not None:
        selection_basis = "best_promotable_metrics"
    else:
        selection_basis = "no_promotable_candidate"

    return {
        "active_generation_focus": ACTIVE_GENERATION_FOCUS,
        "active_generation_variant": _candidate_variant_name(ACTIVE_GENERATION_FOCUS),
        "active_generation_runtime_alias": ACTIVE_GENERATION_RUNTIME_ALIAS,
        "recommended_candidate": recommended_candidate,
        "recommended_variant": _candidate_variant_name(recommended_candidate) if recommended_candidate else None,
        "recommended_runtime_alias": (
            ACTIVE_GENERATION_RUNTIME_ALIAS if recommended_candidate == ACTIVE_GENERATION_FOCUS else None
        ),
        "selection_basis": selection_basis,
        "promotable_candidate_count": int(len(promotable_candidates)),
        "promotable_candidates": [
            _candidate_recommendation_snapshot(
                candidate_name,
                gate_verdict,
                official_summary,
                dynamic_summary,
                dynamic_geometry_summary,
            )
            for candidate_name in promotable_candidates
        ],
        "runner_up_candidate": runner_up_candidate,
        "runner_up_variant": _candidate_variant_name(runner_up_candidate) if runner_up_candidate else None,
        "blocked_candidates": [
            _candidate_recommendation_snapshot(
                candidate_name,
                gate_verdict,
                official_summary,
                dynamic_summary,
                dynamic_geometry_summary,
            )
            for candidate_name in ranked_candidates
            if not bool(gate_verdict[candidate_name].get("promotable_on_current_track", False))
        ],
    }


def main() -> int:
    global TRACK_CANDIDATES
    args = _parse_args()
    TRACK_CANDIDATES = _select_track_candidates(args.candidates)
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
                report["dynamic_geometry_summary"],
                report["official_process_state_summary"],
                report["dynamic_process_state_summary"],
            )
            report["promotion_recommendation"] = _build_promotion_recommendation(
                report["gate_verdict"],
                report["official_summary"],
                report["dynamic_summary"],
                report["dynamic_geometry_summary"],
            )

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())