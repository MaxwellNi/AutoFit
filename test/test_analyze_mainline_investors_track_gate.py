import pytest

from scripts.analyze_mainline_investors_track_gate import (
    _aggregate_candidate_deltas,
    _aggregate_official_process_state,
    _merge_panel_reports,
    _parse_horizons,
    TRACK_CANDIDATES,
)


def _process_state_card(**overrides):
    atoms = {
        "attention_diffusion_score": 0.10,
        "credibility_confirmation_score": 0.10,
        "screening_selectivity_score": 0.10,
        "book_depth_absorption_score": 0.10,
        "closure_conversion_score": 0.10,
        "attention_diffusion_support_share": 0.50,
        "credibility_confirmation_support_share": 0.50,
        "screening_selectivity_support_share": 0.50,
        "book_depth_absorption_support_share": 0.50,
        "closure_conversion_support_share": 0.50,
        "temporal_velocity_coupling": 0.10,
        "temporal_shock_coupling": 0.10,
        "spectral_low_band_coupling": 0.10,
        "spectral_high_band_coupling": 0.10,
        "multiscale_coupling_enabled": True,
    }
    atoms.update(overrides)
    return {"process_state_atoms": atoms}


def _process_state_summary(top_positive_state: str, shift_l1: float) -> dict:
    deltas = {
        "attention_diffusion": 0.0,
        "credibility_confirmation": 0.0,
        "screening_selectivity": 0.0,
        "book_depth_absorption": 0.0,
        "closure_conversion": 0.0,
    }
    if top_positive_state in deltas:
        deltas[top_positive_state] = shift_l1
    return {
        "case_count": 1,
        "mean_scores": {
            "attention_diffusion": 0.10,
            "credibility_confirmation": 0.10,
            "screening_selectivity": 0.10,
            "book_depth_absorption": 0.10,
            "closure_conversion": 0.10,
        },
        "mean_support_shares": {
            "attention_diffusion": 0.50,
            "credibility_confirmation": 0.50,
            "screening_selectivity": 0.50,
            "book_depth_absorption": 0.50,
            "closure_conversion": 0.50,
        },
        "mean_couplings": {
            "temporal_velocity": 0.10,
            "temporal_shock": 0.10,
            "spectral_low_band": 0.10,
            "spectral_high_band": 0.10,
        },
        "mean_score_deltas_vs_baseline": deltas,
        "mean_score_shift_l1": shift_l1,
        "top_positive_state": top_positive_state,
        "top_negative_state": "none",
        "multiscale_coupling_case_count": 1,
    }


def test_parse_horizons_trims_tokens_and_preserves_order() -> None:
    assert _parse_horizons("7, 14,30") == (7, 14, 30)


def test_parse_horizons_rejects_empty_spec() -> None:
    with pytest.raises(ValueError, match="At least one dynamic horizon"):
        _parse_horizons(" , ")


def test_aggregate_official_process_state_surfaces_dominant_shift() -> None:
    cases = {
        "task2_forecast__full__investors_count__h14": {
            "variants": {
                "legacy_baseline": {"event_state_trunk": _process_state_card()},
                "selective_event_state_guard": {
                    "event_state_trunk": _process_state_card(
                        credibility_confirmation_score=0.34,
                        closure_conversion_score=0.18,
                        temporal_velocity_coupling=0.44,
                    )
                },
                "multiscale_state_guard": {
                    "event_state_trunk": _process_state_card(
                        attention_diffusion_score=0.22,
                        closure_conversion_score=0.30,
                        spectral_high_band_coupling=0.52,
                    )
                },
            }
        }
    }

    summary = _aggregate_official_process_state(cases, baseline_key="legacy_baseline")

    assert summary["legacy_baseline"]["mean_score_shift_l1"] == 0.0
    assert summary["selective_event_state_guard"]["case_count"] == 1
    assert summary["selective_event_state_guard"]["top_positive_state"] == "credibility_confirmation"
    assert summary["selective_event_state_guard"]["mean_score_deltas_vs_baseline"]["credibility_confirmation"] == pytest.approx(0.24)
    assert summary["multiscale_state_guard"]["top_positive_state"] == "closure_conversion"
    assert summary["multiscale_state_guard"]["mean_couplings"]["spectral_high_band"] == pytest.approx(0.52)


def test_aggregate_candidate_deltas_and_merge_report_gate_candidates() -> None:
    cases = {
        "task2_forecast__core_edgar__investors_count__h14": {
            "variants": {
                "legacy_baseline": {"mae": 10.0},
                "guarded_jump": {"mae": 9.0},
                "guarded_jump_plus_sparsity": {"mae": 10.5},
                "event_state_boundary_guard": {"mae": 8.8},
                "selective_event_state_guard": {"mae": 8.7},
                "marked_investor_guard": {"mae": 8.7},
                "process_state_feedback_guard": {"mae": 8.68},
                "temporal_state_guard": {"mae": 8.6},
                "spectral_state_guard": {"mae": 8.65},
                "multiscale_state_guard": {"mae": 8.5},
                "source_policy_transition_guard": {"mae": 9.7},
                "hawkes_financing_state_guard": {"mae": 8.55},
            }
        },
        "task2_forecast__full__investors_count__h30": {
            "variants": {
                "legacy_baseline": {"mae": 8.0},
                "guarded_jump": {"mae": 7.6},
                "guarded_jump_plus_sparsity": {"mae": 8.4},
                "event_state_boundary_guard": {"mae": 7.7},
                "selective_event_state_guard": {"mae": 7.4},
                "marked_investor_guard": {"mae": 7.4},
                "process_state_feedback_guard": {"mae": 7.38},
                "temporal_state_guard": {"mae": 7.3},
                "spectral_state_guard": {"mae": 7.35},
                "multiscale_state_guard": {"mae": 7.2},
                "source_policy_transition_guard": {"mae": 8.0},
                "hawkes_financing_state_guard": {"mae": 7.25},
            }
        },
    }
    official_summary = _aggregate_candidate_deltas(cases, baseline_key="legacy_baseline")

    assert official_summary["guarded_jump"]["positive_case_count"] == 2
    assert official_summary["guarded_jump_plus_sparsity"]["negative_case_count"] == 2
    assert official_summary["event_state_boundary_guard"]["positive_case_count"] == 2
    assert official_summary["selective_event_state_guard"]["positive_case_count"] == 2
    assert official_summary["marked_investor_guard"]["positive_case_count"] == 2
    assert official_summary["process_state_feedback_guard"]["positive_case_count"] == 2
    assert official_summary["temporal_state_guard"]["positive_case_count"] == 2
    assert official_summary["spectral_state_guard"]["positive_case_count"] == 2
    assert official_summary["multiscale_state_guard"]["positive_case_count"] == 2
    assert official_summary["source_policy_transition_guard"]["positive_case_count"] == 1
    official_report = {
        "official_cases": {"dummy": {"variants": {}}},
        "official_summary": official_summary,
        "official_process_state_summary": {
            "legacy_baseline": _process_state_summary("none", 0.0),
            "guarded_jump": _process_state_summary("attention_diffusion", 0.08),
            "guarded_jump_plus_sparsity": _process_state_summary("screening_selectivity", 0.04),
            "event_state_boundary_guard": _process_state_summary("attention_diffusion", 0.11),
            "selective_event_state_guard": _process_state_summary("closure_conversion", 0.37),
            "marked_investor_guard": _process_state_summary("closure_conversion", 0.37),
            "process_state_feedback_guard": _process_state_summary("closure_conversion", 0.19),
            "temporal_state_guard": _process_state_summary("credibility_confirmation", 0.29),
            "spectral_state_guard": _process_state_summary("attention_diffusion", 0.21),
            "multiscale_state_guard": _process_state_summary("closure_conversion", 0.41),
            "source_policy_transition_guard": _process_state_summary("screening_selectivity", 0.09),
            "hawkes_financing_state_guard": _process_state_summary("attention_diffusion", 0.32),
        },
        "official_alignment": {
            "legacy_baseline": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 1},
            },
            "guarded_jump": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
            },
            "guarded_jump_plus_sparsity": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
            },
            "event_state_boundary_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
            },
            "selective_event_state_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
            },
            "marked_investor_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 2,
                "mark_reason_counts": {"mark_features_proxy_only": 2},
            },
            "process_state_feedback_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"mark_features_not_requested": 2},
            },
            "temporal_state_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"not_applicable": 2},
            },
            "spectral_state_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"not_applicable": 2},
            },
            "multiscale_state_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"not_applicable": 2},
            },
            "source_policy_transition_guard": {
                "native_runtime_all_cases": False,
                "runtime_modes": ["native", "delegate"],
                "effective_transition_case_count": 1,
                "transition_reason_counts": {"source_homogeneous_train_surface": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"not_applicable": 2},
            },
            "hawkes_financing_state_guard": {
                "native_runtime_all_cases": True,
                "runtime_modes": ["native", "native"],
                "effective_transition_case_count": 0,
                "transition_reason_counts": {"not_applicable": 2},
                "effective_mark_case_count": 0,
                "mark_reason_counts": {"not_applicable": 2},
            },
        },
    }
    dynamic_report = {
        "dynamic_cases": {"dummy": {"variants": {}}},
        "dynamic_summary": {
            "legacy_baseline": {
                "case_count": 1,
                "mean_mae_delta_pct": 0.0,
                "worst_mae_delta_pct": 0.0,
                "best_mae_delta_pct": 0.0,
                "positive_case_count": 0,
                "negative_case_count": 0,
            },
            "guarded_jump": {
                "case_count": 1,
                "mean_mae_delta_pct": 0.8,
                "worst_mae_delta_pct": 0.8,
                "best_mae_delta_pct": 0.8,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "guarded_jump_plus_sparsity": {
                "case_count": 1,
                "mean_mae_delta_pct": -0.2,
                "worst_mae_delta_pct": -0.2,
                "best_mae_delta_pct": -0.2,
                "positive_case_count": 0,
                "negative_case_count": 1,
            },
            "event_state_boundary_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.1,
                "worst_mae_delta_pct": 1.1,
                "best_mae_delta_pct": 1.1,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "selective_event_state_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.2,
                "worst_mae_delta_pct": 1.2,
                "best_mae_delta_pct": 1.2,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "marked_investor_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.2,
                "worst_mae_delta_pct": 1.2,
                "best_mae_delta_pct": 1.2,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "process_state_feedback_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.15,
                "worst_mae_delta_pct": 1.15,
                "best_mae_delta_pct": 1.15,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "temporal_state_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.3,
                "worst_mae_delta_pct": 1.3,
                "best_mae_delta_pct": 1.3,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "spectral_state_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 0.9,
                "worst_mae_delta_pct": 0.9,
                "best_mae_delta_pct": 0.9,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "multiscale_state_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.5,
                "worst_mae_delta_pct": 1.5,
                "best_mae_delta_pct": 1.5,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "source_policy_transition_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.4,
                "worst_mae_delta_pct": 1.4,
                "best_mae_delta_pct": 1.4,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
            "hawkes_financing_state_guard": {
                "case_count": 1,
                "mean_mae_delta_pct": 1.35,
                "worst_mae_delta_pct": 1.35,
                "best_mae_delta_pct": 1.35,
                "positive_case_count": 1,
                "negative_case_count": 0,
            },
        },
        "dynamic_process_state_summary": {
            "legacy_baseline": _process_state_summary("none", 0.0),
            "guarded_jump": _process_state_summary("attention_diffusion", 0.06),
            "guarded_jump_plus_sparsity": _process_state_summary("screening_selectivity", 0.03),
            "event_state_boundary_guard": _process_state_summary("attention_diffusion", 0.09),
            "selective_event_state_guard": _process_state_summary("closure_conversion", 0.33),
            "marked_investor_guard": _process_state_summary("closure_conversion", 0.33),
            "process_state_feedback_guard": _process_state_summary("closure_conversion", 0.16),
            "temporal_state_guard": _process_state_summary("credibility_confirmation", 0.25),
            "spectral_state_guard": _process_state_summary("attention_diffusion", 0.18),
            "multiscale_state_guard": _process_state_summary("closure_conversion", 0.45),
            "source_policy_transition_guard": _process_state_summary("screening_selectivity", 0.07),
            "hawkes_financing_state_guard": _process_state_summary("attention_diffusion", 0.28),
        },
        "dynamic_geometry_summary": {
            "legacy_baseline": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "guarded_jump": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "guarded_jump_plus_sparsity": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "event_state_boundary_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "selective_event_state_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "marked_investor_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
                "process_state_feedback_guard": {
                    "case_count": 1,
                    "geometry_pass_case_count": 1,
                    "geometry_fail_case_count": 0,
                    "geometry_all_cases_pass": True,
                    "worst_train_test_abs_mean_ratio": 12.0,
                    "worst_train_test_p95_abs_ratio": 18.0,
                    "worst_shift_abs_mean_amplification": 0.5,
                    "worst_shift_p95_amplification": 0.8,
                    "worst_test_shift_abs_mean_share": 0.01,
                    "worst_test_shift_p95_share": 0.01,
                },
            "temporal_state_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 12.0,
                "worst_shift_p95_amplification": 15.0,
                "worst_test_shift_abs_mean_share": 0.03,
                "worst_test_shift_p95_share": 0.04,
            },
            "spectral_state_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 18.0,
                "worst_shift_p95_amplification": 20.0,
                "worst_test_shift_abs_mean_share": 0.02,
                "worst_test_shift_p95_share": 0.03,
            },
            "multiscale_state_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 0,
                "geometry_fail_case_count": 1,
                "geometry_all_cases_pass": False,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 2048.0,
                "worst_shift_p95_amplification": 4096.0,
                "worst_test_shift_abs_mean_share": 0.12,
                "worst_test_shift_p95_share": 0.20,
            },
            "source_policy_transition_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
            "hawkes_financing_state_guard": {
                "case_count": 1,
                "geometry_pass_case_count": 1,
                "geometry_fail_case_count": 0,
                "geometry_all_cases_pass": True,
                "worst_train_test_abs_mean_ratio": 12.0,
                "worst_train_test_p95_abs_ratio": 18.0,
                "worst_shift_abs_mean_amplification": 0.0,
                "worst_shift_p95_amplification": 0.0,
                "worst_test_shift_abs_mean_share": 0.0,
                "worst_test_shift_p95_share": 0.0,
            },
        },
    }

    merged = _merge_panel_reports(official_report, dynamic_report)

    assert merged["track_contract"]["delegate_forbidden"] is True
    assert merged["track_contract"]["active_generation_focus"] == "selective_event_state_guard"
    assert merged["track_contract"]["active_generation_runtime_alias"] == "single_model_mainline_track_active_generation_focus"
    assert merged["official_process_state_summary"]["selective_event_state_guard"]["top_positive_state"] == "closure_conversion"
    assert merged["dynamic_process_state_summary"]["multiscale_state_guard"]["top_positive_state"] == "closure_conversion"
    assert merged["gate_verdict"]["guarded_jump"]["official_track_pass"] is True
    assert merged["gate_verdict"]["guarded_jump"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["guarded_jump"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["guarded_jump"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["demotion_blocked"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["promotable_on_current_track"] is False
    assert merged["gate_verdict"]["selective_event_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["selective_event_state_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["official_primary_process_state"] == "closure_conversion"
    assert merged["gate_verdict"]["selective_event_state_guard"]["dynamic_primary_process_state"] == "closure_conversion"
    assert merged["gate_verdict"]["selective_event_state_guard"]["official_process_state_shift_l1"] == pytest.approx(0.37)
    assert merged["gate_verdict"]["marked_investor_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["marked_investor_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["process_state_feedback_guard"]["official_primary_process_state"] == "closure_conversion"
    assert merged["gate_verdict"]["temporal_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["temporal_state_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["temporal_state_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["temporal_state_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["temporal_state_guard"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["temporal_state_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["spectral_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["spectral_state_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["spectral_state_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["spectral_state_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["spectral_state_guard"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["spectral_state_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["dynamic_geometry_pass"] is False
    assert merged["gate_verdict"]["multiscale_state_guard"]["dynamic_track_pass"] is False
    assert merged["gate_verdict"]["multiscale_state_guard"]["demotion_blocked"] is False
    assert merged["gate_verdict"]["multiscale_state_guard"]["promotable_on_current_track"] is False
    assert merged["gate_verdict"]["multiscale_state_guard"]["dynamic_process_state_shift_l1"] == pytest.approx(0.45)
    assert merged["gate_verdict"]["source_policy_transition_guard"]["official_track_pass"] is False
    assert merged["gate_verdict"]["source_policy_transition_guard"]["dynamic_metric_pass"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["dynamic_geometry_pass"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["demotion_blocked"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["promotable_on_current_track"] is False

    recommendation = merged["promotion_recommendation"]
    assert recommendation["active_generation_focus"] == "selective_event_state_guard"
    assert recommendation["active_generation_runtime_alias"] == "single_model_mainline_track_active_generation_focus"
    assert recommendation["recommended_candidate"] == "selective_event_state_guard"
    assert recommendation["recommended_variant"] == "mainline_selective_event_state_guard"
    assert recommendation["recommended_runtime_alias"] == "single_model_mainline_track_active_generation_focus"
    assert recommendation["selection_basis"] == "preferred_active_generation_focus"
    assert recommendation["promotable_candidate_count"] >= 1
    assert recommendation["promotable_candidates"][0]["candidate"] == "selective_event_state_guard"
    blocked_candidates = {
        row["candidate"]: row["primary_blocker"]
        for row in recommendation["blocked_candidates"]
    }
    assert blocked_candidates["event_state_boundary_guard"] == "demotion_blocked"
    assert blocked_candidates["multiscale_state_guard"] == "dynamic_geometry_fail"


def test_track_candidates_include_process_state_feedback_guard() -> None:
    candidate = TRACK_CANDIDATES["process_state_feedback_guard"]

    assert candidate["official_kwargs"]["variant"] == "mainline_process_state_feedback_guard"
    assert candidate["dynamic_backbone"]["enable_process_state_feedback"] is True
    assert candidate["dynamic_backbone"]["process_state_feedback_min_horizon"] == 7