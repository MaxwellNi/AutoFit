from narrative.block3.models.single_model_mainline.objectives import MainlineObjectiveSpec


def test_binary_objective_runtime_state_exposes_native_gap_honestly():
    spec = MainlineObjectiveSpec()

    state = spec.build_runtime_state(
        lane_name="binary",
        horizon=14,
        lane_guardrails=("probability_collapse",),
        switches={
            "teacher_distill": True,
            "event_head": True,
        },
    )

    assert state["runtime_stage"] == "lane_runtime_with_objective_plan"
    assert state["implemented_terms"] == ("calibration", "hazard", "event_consistency")
    assert state["guardrails"] == ("probability_collapse",)


def test_investors_objective_runtime_state_tracks_lane_ownership_and_backlog():
    spec = MainlineObjectiveSpec()

    state = spec.build_runtime_state(
        lane_name="investors",
        horizon=1,
        lane_guardrails=("shared_count_repair",),
        switches={
            "count_source_specialists": True,
            "financing_consistency": True,
        },
    )

    assert state["lane_subregime"] == "h1_occurrence_exemplar"
    assert state["implemented_terms"] == ("occurrence", "hurdle", "intensity_baseline", "jump_ode_state", "shrinkage_gate")
    assert "transition" in state["deferred_terms"]
    assert "transition" in state["enabled_but_deferred_terms"]
    assert "binary_funding_alignment" in state["enabled_but_deferred_terms"]
    assert "cross_task_stability" in state["review_terms"]
    assert "short_horizon_exemplar_blend" in state["implemented_runtime"]
    assert "h1_transition_block_by_contract" in state["implemented_runtime"]


def test_investors_long_horizon_objective_runtime_exposes_transition_subregime():
    spec = MainlineObjectiveSpec()

    state = spec.build_runtime_state(
        lane_name="investors",
        horizon=14,
        lane_guardrails=("shared_count_repair",),
        switches={
            "count_source_specialists": True,
            "investors_transition_correction": True,
        },
    )

    assert state["lane_subregime"] == "hplus_hurdle_transition"
    assert "geometry_gated_transition_correction" in state["implemented_runtime"]


def test_funding_objective_runtime_exposes_calibrated_anchor_backoff():
    spec = MainlineObjectiveSpec()

    state = spec.build_runtime_state(
        lane_name="funding",
        horizon=30,
        lane_guardrails=("source_rich_blowup",),
        switches={
            "funding_source_scaling_guard": True,
            "task_modulation": True,
        },
    )

    assert state["lane_subregime"] == "funding_anchor_residual"
    assert "calibrated_anchor_backoff" in state["implemented_runtime"]
    assert "source_scaling_guard" in state["enabled_but_deferred_terms"]
    assert "cqr_interval" in state["implemented_terms"] or "tail_guard" not in state.get("enabled_but_deferred_terms", ())