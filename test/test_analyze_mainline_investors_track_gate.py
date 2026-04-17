import pytest

from scripts.analyze_mainline_investors_track_gate import (
    _aggregate_candidate_deltas,
    _merge_panel_reports,
    _parse_horizons,
)


def test_parse_horizons_trims_tokens_and_preserves_order() -> None:
    assert _parse_horizons("7, 14,30") == (7, 14, 30)


def test_parse_horizons_rejects_empty_spec() -> None:
    with pytest.raises(ValueError, match="At least one dynamic horizon"):
        _parse_horizons(" , ")


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
                "multiscale_state_guard": {"mae": 8.5},
                "source_policy_transition_guard": {"mae": 9.7},
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
                "multiscale_state_guard": {"mae": 7.2},
                "source_policy_transition_guard": {"mae": 8.0},
            }
        },
    }
    official_summary = _aggregate_candidate_deltas(cases, baseline_key="legacy_baseline")

    assert official_summary["guarded_jump"]["positive_case_count"] == 2
    assert official_summary["guarded_jump_plus_sparsity"]["negative_case_count"] == 2
    assert official_summary["event_state_boundary_guard"]["positive_case_count"] == 2
    assert official_summary["selective_event_state_guard"]["positive_case_count"] == 2
    assert official_summary["marked_investor_guard"]["positive_case_count"] == 2
    assert official_summary["multiscale_state_guard"]["positive_case_count"] == 2
    assert official_summary["source_policy_transition_guard"]["positive_case_count"] == 1

    official_report = {
        "official_cases": {"dummy": {"variants": {}}},
        "official_summary": official_summary,
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
        },
    }

    merged = _merge_panel_reports(official_report, dynamic_report)

    assert merged["track_contract"]["delegate_forbidden"] is True
    assert merged["track_contract"]["active_generation_focus"] == "multiscale_state_guard"
    assert merged["gate_verdict"]["guarded_jump"]["official_track_pass"] is True
    assert merged["gate_verdict"]["guarded_jump"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["guarded_jump"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["event_state_boundary_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["selective_event_state_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["marked_investor_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["official_track_pass"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["multiscale_state_guard"]["promotable_on_current_track"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["official_track_pass"] is False
    assert merged["gate_verdict"]["source_policy_transition_guard"]["dynamic_track_pass"] is True
    assert merged["gate_verdict"]["source_policy_transition_guard"]["promotable_on_current_track"] is False