import pytest

from scripts.analyze_mainline_event_state_geometry import _aggregate_variant_deltas, _parse_horizons


def test_parse_horizons_trims_tokens_and_preserves_order() -> None:
    assert _parse_horizons("1, 7,30") == (1, 7, 30)


def test_parse_horizons_rejects_empty_spec() -> None:
    with pytest.raises(ValueError, match="At least one horizon"):
        _parse_horizons(" , ")


def test_aggregate_variant_deltas_carries_geometry_summary() -> None:
    cases = {
        "task2_forecast__investors_count__h1": {
            "ablations": {
                "full": {
                    "variants": {
                        "legacy_baseline": {
                            "metrics": {"mae": 10.0},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 1.0},
                                "persistence_atoms": {"dynamic_entity_share": 0.50},
                                "source_atoms": {
                                    "source_presence_share": 0.90,
                                    "source_surface": "heterogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": False,
                            },
                        },
                        "event_state_boundary_guard": {
                            "metrics": {"mae": 7.5},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 1.0},
                                "persistence_atoms": {"dynamic_entity_share": 0.50},
                                "source_atoms": {
                                    "source_presence_share": 0.90,
                                    "source_surface": "heterogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": False,
                            },
                        },
                        "source_read_guard": {
                            "metrics": {"mae": 8.0},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 1.0},
                                "persistence_atoms": {"dynamic_entity_share": 0.50},
                                "source_atoms": {
                                    "source_presence_share": 0.90,
                                    "source_surface": "heterogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": False,
                            },
                        },
                    }
                }
            }
        },
        "task2_forecast__investors_count__h7": {
            "ablations": {
                "core_edgar": {
                    "variants": {
                        "legacy_baseline": {
                            "metrics": {"mae": 5.0},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 0.8},
                                "persistence_atoms": {"dynamic_entity_share": 0.25},
                                "source_atoms": {
                                    "source_presence_share": 0.60,
                                    "source_surface": "homogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": False,
                            },
                        },
                        "event_state_boundary_guard": {
                            "metrics": {"mae": 4.8},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 0.8},
                                "persistence_atoms": {"dynamic_entity_share": 0.25},
                                "source_atoms": {
                                    "source_presence_share": 0.60,
                                    "source_surface": "homogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": False,
                            },
                        },
                        "source_read_guard": {
                            "metrics": {"mae": 6.0},
                            "runtime_mode": "native",
                            "event_state_trunk": {
                                "phase_atoms": {"joint_financing_active_share": 0.8},
                                "persistence_atoms": {"dynamic_entity_share": 0.25},
                                "source_atoms": {
                                    "source_presence_share": 0.60,
                                    "source_surface": "homogeneous",
                                },
                            },
                            "investors_source_activation": {
                                "effective_transition_correction": True,
                            },
                        },
                    }
                }
            }
        },
    }

    summary = _aggregate_variant_deltas(cases)

    assert summary["legacy_baseline"]["case_count"] == 2
    assert summary["event_state_boundary_guard"]["positive_case_count"] == 2
    assert summary["source_read_guard"]["positive_case_count"] == 1
    assert summary["source_read_guard"]["negative_case_count"] == 1
    assert summary["source_read_guard"]["effective_transition_case_count"] == 1
    assert summary["source_read_guard"]["mean_dynamic_entity_share"] == pytest.approx(0.375)
    assert summary["source_read_guard"]["mean_source_presence_share"] == pytest.approx(0.75)
    assert summary["source_read_guard"]["heterogeneous_source_case_count"] == 1