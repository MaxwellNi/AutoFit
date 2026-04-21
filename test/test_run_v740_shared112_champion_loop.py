from __future__ import annotations

from argparse import Namespace

from scripts.run_v740_shared112_champion_loop import (
    LOCAL_MAINLINE_ALIASES,
    _display_candidate_name,
    _pick_primary_candidate_label,
    _resolve_model_specs,
)


def test_active_generation_focus_alias_is_registered() -> None:
    alias_cfg = LOCAL_MAINLINE_ALIASES["single_model_mainline_track_active_generation_focus"]

    assert alias_cfg["variant"] == "mainline_selective_event_state_guard"
    assert alias_cfg["use_delegate"] is False


def test_resolve_model_specs_supports_active_generation_focus_alias() -> None:
    specs = _resolve_model_specs(
        ["single_model_mainline_track_active_generation_focus", "incumbent"],
        {"incumbent_model": "NBEATS"},
        Namespace(),
    )

    assert specs == [
        {
            "token": "single_model_mainline_track_active_generation_focus",
            "resolved": "single_model_mainline_track_active_generation_focus",
            "label": "single_model_mainline_track_active_generation_focus",
        },
        {"token": "incumbent", "resolved": "NBEATS", "label": "incumbent__NBEATS"},
    ]


def test_pick_primary_candidate_label_prefers_active_generation_focus() -> None:
    results = [
        {"model_label": "incumbent__NBEATS"},
        {"model_label": "single_model_mainline"},
        {"model_label": "single_model_mainline_track_active_generation_focus"},
    ]

    assert _pick_primary_candidate_label(results) == "single_model_mainline_track_active_generation_focus"


def test_display_candidate_name_for_active_generation_focus() -> None:
    assert (
        _display_candidate_name("single_model_mainline_track_active_generation_focus")
        == "SingleModelMainline-ActiveGenerationFocus"
    )