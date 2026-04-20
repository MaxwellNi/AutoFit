import numpy as np
import pandas as pd

from narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper


def _make_investors_surface(
    profiles: list[str],
    rows_per_profile: int = 12,
    *,
    dynamic_within_entity: bool = True,
) -> pd.DataFrame:
    rows = []
    current_day = pd.Timestamp("2024-01-01")
    for profile_idx, profile in enumerate(profiles):
        for row_idx in range(rows_per_profile):
            global_idx = profile_idx * rows_per_profile + row_idx
            investors_value = float(20.0 + 2.0 * profile_idx + (row_idx % 4))
            if not dynamic_within_entity:
                investors_value = float(20.0 + 2.0 * profile_idx)
            row = {
                "entity_id": f"entity_{profile_idx}_{row_idx // 4}",
                "crawled_date_day": current_day + pd.Timedelta(days=global_idx),
                "core_signal": float((row_idx % 5) - 2),
                "funding_raised_usd": float(1000.0 + 10.0 * global_idx),
                "is_funded": 1.0,
                "investors_count": investors_value,
                "last_total_amount_sold": np.nan,
                "edgar_has_filing": np.nan,
                "text_emb_0": np.nan,
                "text_emb_1": np.nan,
            }
            if profile in {"edgar_only", "mixed"}:
                row["last_total_amount_sold"] = float(5000.0 + global_idx)
                row["edgar_has_filing"] = 1.0
            if profile in {"text_only", "mixed"}:
                row["text_emb_0"] = float(0.1 * (row_idx + 1))
                row["text_emb_1"] = float(0.05 * (profile_idx + 1))
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame


def _enrich_process_state_surface(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    row_idx = np.arange(len(enriched), dtype=np.float64)
    enriched["funding_goal_usd"] = 2500.0 + 20.0 * row_idx
    enriched["last_total_amount_sold"] = np.where(
        np.isfinite(enriched["last_total_amount_sold"]),
        enriched["last_total_amount_sold"],
        900.0 + 15.0 * row_idx,
    )
    enriched["last_total_offering_amount"] = enriched["funding_goal_usd"] + 800.0
    enriched["last_total_remaining"] = np.maximum(
        enriched["last_total_offering_amount"] - enriched["last_total_amount_sold"],
        0.0,
    )
    enriched["last_minimum_investment_accepted"] = 100.0 + 5.0 * row_idx
    enriched["last_total_number_already_invested"] = enriched["investors_count"] + 2.0
    enriched["last_number_non_accredited_investors"] = np.clip(enriched["investors_count"] - 1.0, 0.0, None)
    enriched["non_national_investors"] = np.clip(enriched["investors_count"] * 0.25, 0.0, None)
    return enriched


def _fit_wrapper(frame: pd.DataFrame, **kwargs) -> SingleModelMainlineWrapper:
    horizon = int(kwargs.pop("horizon", 1))
    wrapper = SingleModelMainlineWrapper(seed=7, **kwargs)
    X = frame[["core_signal"]].copy()
    y = frame["investors_count"].copy()
    y.name = "investors_count"
    wrapper.fit(
        X,
        y,
        train_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=horizon,
    )
    return wrapper


def _make_funding_frame(n_rows: int = 24) -> pd.DataFrame:
    rows = []
    current_day = pd.Timestamp("2024-01-01")
    for idx in range(n_rows):
        rows.append(
            {
                "entity_id": f"funding_entity_{idx // 6}",
                "crawled_date_day": current_day + pd.Timedelta(days=idx),
                "core_signal": float((idx % 5) - 2),
                "funding_raised_usd": float(1000.0 + 25.0 * idx),
                "is_funded": 1.0,
                "investors_count": float(4.0 + (idx % 3)),
            }
        )
    return pd.DataFrame(rows)


def _make_binary_frame(n_rows: int = 24) -> pd.DataFrame:
    rows = []
    current_day = pd.Timestamp("2024-01-01")
    for idx in range(n_rows):
        funded = float((idx % 4) >= 2)
        rows.append(
            {
                "entity_id": f"binary_entity_{idx // 6}",
                "crawled_date_day": current_day + pd.Timedelta(days=idx),
                "core_signal": float((idx % 6) - 3),
                "funding_raised_usd": float(2500.0 + 50.0 * idx) if funded > 0.5 else 0.0,
                "is_funded": funded,
                "investors_count": float(3.0 + (idx % 4)) if funded > 0.5 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_homogeneous_source_surface_disables_requested_source_path_and_matches_baseline():
    frame = _make_investors_surface(["mixed", "mixed", "mixed"], rows_per_profile=12)

    baseline = _fit_wrapper(frame)
    requested = _fit_wrapper(
        frame,
        enable_investors_source_features=True,
        enable_count_source_specialists=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
    )

    regime = requested.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_source_path"] is True
    assert source_regime["effective_source_features"] is False
    assert source_regime["effective_source_specialists"] is False
    assert source_regime["effective_source_guard"] is False
    assert source_regime["activation_reason"] == "source_homogeneous_train_surface"
    assert source_regime["train_profile_counts"] == {"mixed": 36}
    assert source_regime["eligible_profiles"] == ["mixed"]
    assert regime["objectives"]["active_switches"]["count_source_specialists"] is False

    X = frame[["core_signal"]].copy()
    baseline_preds = baseline.predict(
        X,
        test_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=1,
    )
    requested_preds = requested.predict(
        X,
        test_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=1,
    )
    assert np.allclose(baseline_preds, requested_preds)


def test_homogeneous_surface_also_disables_requested_transition_correction():
    frame = _make_investors_surface(["mixed", "mixed", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_transition_correction"] is True
    assert source_regime["effective_source_read_policy"] is False
    assert source_regime["effective_transition_correction"] is False
    assert source_regime["activation_reason"] == "source_homogeneous_train_surface"
    assert source_regime["transition_activation_reason"] == "source_homogeneous_train_surface"
    assert regime["objectives"]["active_switches"]["investors_transition_correction"] is False


def test_heterogeneous_source_surface_enables_requested_source_path():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_features=True,
        enable_count_source_specialists=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_source_path"] is True
    assert source_regime["effective_source_features"] is True
    assert source_regime["effective_source_specialists"] is True
    assert source_regime["effective_source_guard"] is True
    assert source_regime["activation_reason"] == "multi_profile_train_surface"
    assert source_regime["train_profile_counts"] == {"edgar_only": 12, "text_only": 12, "mixed": 12}
    assert source_regime["eligible_profiles"] == ["edgar_only", "text_only", "mixed"]
    assert regime["objectives"]["active_switches"]["count_source_specialists"] is True


def test_h1_heterogeneous_surface_keeps_read_policy_and_blocks_transition_by_contract():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_transition_correction"] is True
    assert source_regime["effective_source_read_policy"] is True
    assert source_regime["effective_transition_correction"] is False
    assert source_regime["horizon_subregime"] == "h1_occurrence_exemplar"
    assert source_regime["transition_activation_reason"] == "transition_h1_blocked_by_contract"
    assert regime["objectives"]["active_switches"]["investors_transition_correction"] is False


def test_long_horizon_heterogeneous_surface_enables_transition_correction_request():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        enable_count_hurdle_head=False,
        enable_count_jump=False,
        enable_count_sparsity_gate=False,
        investors_source_activation_min_rows=8,
        horizon=7,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]
    process_regime = regime["investors_process_contract"]

    assert source_regime["requested_transition_correction"] is True
    assert source_regime["effective_source_read_policy"] is True
    assert source_regime["effective_transition_correction"] is True
    assert source_regime["horizon_subregime"] == "hplus_hurdle_transition"
    assert source_regime["transition_activation_reason"] == "multi_profile_dynamic_train_surface"
    assert regime["objectives"]["active_switches"]["investors_transition_correction"] is True
    assert process_regime["effective_count_hurdle_head"] is False
    assert process_regime["effective_count_jump"] is False
    assert process_regime["effective_count_sparsity_gate"] is False


def test_mainline_alpha_promotes_guarded_jump_by_default_on_long_horizon_investors():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(frame, horizon=7)

    regime = wrapper.get_regime_info()
    process_regime = regime["investors_process_contract"]

    assert process_regime["requested_count_hurdle_head"] is True
    assert process_regime["requested_count_jump"] is True
    assert process_regime["requested_count_sparsity_gate"] is False
    assert process_regime["effective_count_hurdle_head"] is True
    assert process_regime["effective_count_jump"] is True
    assert process_regime["effective_count_sparsity_gate"] is False
    assert np.isclose(process_regime["count_jump_strength"], 0.30)


def test_explicit_opt_out_can_restore_legacy_long_horizon_process_contract():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_count_hurdle_head=False,
        enable_count_jump=False,
        enable_count_sparsity_gate=False,
        horizon=7,
    )

    regime = wrapper.get_regime_info()
    process_regime = regime["investors_process_contract"]

    assert process_regime["requested_count_hurdle_head"] is False
    assert process_regime["requested_count_jump"] is False
    assert process_regime["requested_count_sparsity_gate"] is False
    assert process_regime["effective_count_hurdle_head"] is False
    assert process_regime["effective_count_jump"] is False
    assert process_regime["effective_count_sparsity_gate"] is False


def test_event_state_trunk_reports_audited_process_state_families() -> None:
    frame = _enrich_process_state_surface(
        _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)
    )

    wrapper = _fit_wrapper(
        frame,
        enable_investors_event_state_features=True,
        enable_multiscale_temporal_state=True,
        enable_temporal_state_features=True,
        enable_spectral_state_features=True,
    )

    process_atoms = wrapper.get_regime_info()["event_state_trunk"]["process_state_atoms"]

    assert process_atoms["schema_version"] == "process_state_v1"
    assert process_atoms["state_order"] == [
        "attention_diffusion",
        "credibility_confirmation",
        "screening_selectivity",
        "book_depth_absorption",
        "closure_conversion",
    ]
    assert process_atoms["multiscale_coupling_enabled"] is True
    assert 0.0 < process_atoms["attention_diffusion_score"] <= 1.0
    assert 0.0 < process_atoms["credibility_confirmation_score"] <= 1.0
    assert 0.0 < process_atoms["screening_selectivity_score"] <= 1.0
    assert 0.0 < process_atoms["book_depth_absorption_score"] <= 1.0
    assert 0.0 < process_atoms["closure_conversion_score"] <= 1.0
    assert process_atoms["screening_selectivity_support_share"] > 0.0
    assert process_atoms["book_depth_absorption_support_share"] > 0.0
    assert process_atoms["temporal_velocity_coupling"] >= 0.0
    assert process_atoms["spectral_high_band_coupling"] >= 0.0


def test_process_state_feedback_guard_changes_shared_trunk_on_long_horizon() -> None:
    frame = _enrich_process_state_surface(
        _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)
    )

    selective = _fit_wrapper(frame, variant="mainline_selective_event_state_guard", horizon=14)
    feedback = _fit_wrapper(frame, variant="mainline_process_state_feedback_guard", horizon=14)

    selective_trunk = selective.get_regime_info()["event_state_trunk"]
    feedback_trunk = feedback.get_regime_info()["event_state_trunk"]
    selective_shared = selective_trunk["shared_state_atoms"]
    feedback_shared = feedback_trunk["shared_state_atoms"]
    selective_process = selective_trunk["process_state_atoms"]
    feedback_process = feedback_trunk["process_state_atoms"]

    assert selective_shared["effective_process_state_feedback"] is False
    assert feedback_shared["process_state_feedback_enabled"] is True
    assert feedback_shared["effective_process_state_feedback"] is True
    assert feedback_shared["process_state_feedback_activation_reason"] == "process_state_feedback_source_attenuated"
    assert feedback_shared["process_state_feedback_gate"] > 0.0
    assert feedback_shared["shared_state_dim"] == selective_shared["shared_state_dim"]
    assert feedback_shared["process_feedback_attention_diffusion_abs_mean"] > 0.0
    assert feedback_shared["process_feedback_closure_conversion_abs_mean"] > 0.0

    shift_l1 = sum(
        abs(float(feedback_process[f"{state_name}_score"]) - float(selective_process[f"{state_name}_score"]))
        for state_name in [
            "attention_diffusion",
            "credibility_confirmation",
            "screening_selectivity",
            "book_depth_absorption",
            "closure_conversion",
        ]
    )
    assert shift_l1 > 0.0
    assert feedback_process["closure_conversion_score"] >= selective_process["closure_conversion_score"]


def test_process_state_feedback_guard_stays_blocked_on_h1() -> None:
    frame = _enrich_process_state_surface(
        _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)
    )

    feedback = _fit_wrapper(frame, variant="mainline_process_state_feedback_guard", horizon=1)
    shared_atoms = feedback.get_regime_info()["event_state_trunk"]["shared_state_atoms"]

    assert shared_atoms["process_state_feedback_enabled"] is True
    assert shared_atoms["effective_process_state_feedback"] is False
    assert shared_atoms["process_state_feedback_activation_reason"] == "process_state_feedback_horizon_blocked"
    assert np.isclose(shared_atoms["process_state_feedback_gate"], 0.0)


def test_explicit_component_flags_preserve_full_long_horizon_process_contract():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        enable_count_hurdle_head=True,
        enable_count_jump=True,
        enable_count_sparsity_gate=True,
        investors_source_activation_min_rows=8,
        horizon=7,
    )

    regime = wrapper.get_regime_info()
    process_regime = regime["investors_process_contract"]

    assert process_regime["effective_count_hurdle_head"] is True
    assert process_regime["effective_count_jump"] is True
    assert process_regime["effective_count_sparsity_gate"] is True


def test_predict_does_not_depend_on_current_test_target_values():
    frame = _make_funding_frame()
    wrapper = SingleModelMainlineWrapper(seed=11)
    X = frame[["core_signal"]].copy()
    y = frame["funding_raised_usd"].copy()
    y.name = "funding_raised_usd"
    wrapper.fit(
        X,
        y,
        train_raw=frame,
        target="funding_raised_usd",
        task="task1_outcome",
        ablation="core_only",
        horizon=14,
    )
    regime = wrapper.get_regime_info()

    test_raw_a = frame.copy()
    test_raw_b = frame.copy()
    test_raw_b["funding_raised_usd"] = test_raw_b["funding_raised_usd"] + 1_000_000.0

    preds_a = wrapper.predict(
        X,
        test_raw=test_raw_a,
        target="funding_raised_usd",
        task="task1_outcome",
        ablation="core_only",
        horizon=14,
    )
    preds_b = wrapper.predict(
        X,
        test_raw=test_raw_b,
        target="funding_raised_usd",
        task="task1_outcome",
        ablation="core_only",
        horizon=14,
    )

    assert np.allclose(preds_a, preds_b)
    assert regime["runtime"]["predict_time_current_target_masked"] is True
    assert regime["runtime"]["runtime_no_leak_contract"] == "predict_time_current_target_masked_before_history_and_anchor"
    assert regime["funding_process_contract"]["process_family"] == "anchor_plus_jump_hurdle"
    assert regime["funding_process_contract"]["lane_uses_jump_hurdle_head"] is True
    assert regime["funding_process_contract"]["lane_positive_jump_rows"] > 0


def test_binary_regime_surfaces_process_contract():
    frame = _make_binary_frame()
    wrapper = SingleModelMainlineWrapper(seed=13)
    X = frame[["core_signal"]].copy()
    y = frame["is_funded"].copy()
    y.name = "is_funded"
    wrapper.fit(
        X,
        y,
        train_raw=frame,
        target="is_funded",
        task="task1_outcome",
        ablation="core_edgar",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    process = regime["binary_process_contract"]

    assert regime["runtime"]["predict_time_current_target_masked"] is True
    assert regime["runtime"]["runtime_no_leak_contract"] == "predict_time_current_target_masked_before_history_and_anchor"
    assert process["process_family"] == "hazard_prior_plus_calibration"
    assert process["uses_neural_hazard_head"] is True
    assert process["calibration_method"] == "identity"
    assert process["selected_brier"] >= 0.0
    assert process["selected_logloss"] >= 0.0
    assert process["selected_ece"] >= 0.0
    assert np.isclose(process["constant_probability"], 0.5)
    assert np.isclose(process["train_positive_rate"], 0.5)
    assert np.isclose(process["event_rate"], 0.5)
    assert process["transition_rate"] > 0.0
    assert np.isclose(process["positive_class_weight"], 1.0)
    assert process["temperature"] >= 0.5
    assert np.isclose(process["teacher_weight"], 0.10)
    assert np.isclose(process["event_weight"], 0.15)


def test_heterogeneous_but_static_surface_keeps_read_policy_and_disables_transition():
    frame = _make_investors_surface(
        ["edgar_only", "text_only", "mixed"],
        rows_per_profile=12,
        dynamic_within_entity=False,
    )

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
        investors_transition_activation_min_rows=8,
        investors_transition_activation_min_entities=2,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]
    geometry = source_regime["geometry_card"]

    assert source_regime["effective_source_read_policy"] is True
    assert source_regime["effective_transition_correction"] is False
    assert source_regime["activation_reason"] == "multi_profile_train_surface"
    assert source_regime["horizon_subregime"] == "h1_occurrence_exemplar"
    assert source_regime["transition_activation_reason"] == "transition_h1_blocked_by_contract"
    assert geometry["profile_surface"] == "heterogeneous"
    assert geometry["transition_surface"] == "static_like"
    assert geometry["dynamic_entities"] == 0
    assert geometry["transition_nonzero_rows"] == 0


def test_long_horizon_static_surface_uses_transition_sparse_reason():
    frame = _make_investors_surface(
        ["edgar_only", "text_only", "mixed"],
        rows_per_profile=12,
        dynamic_within_entity=False,
    )

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_transition_correction=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
        investors_transition_activation_min_rows=8,
        investors_transition_activation_min_entities=2,
        horizon=7,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]
    geometry = source_regime["geometry_card"]

    assert source_regime["effective_source_read_policy"] is True
    assert source_regime["effective_transition_correction"] is False
    assert source_regime["horizon_subregime"] == "hplus_hurdle_transition"
    assert source_regime["transition_activation_reason"] == "transition_sparse_train_surface"
    assert geometry["profile_surface"] == "heterogeneous"
    assert geometry["transition_surface"] == "static_like"


def test_event_state_trunk_reports_dynamic_shared_geometry():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        enable_investors_source_read_policy=True,
        enable_investors_source_guard=True,
        investors_source_activation_min_rows=8,
    )

    trunk = wrapper.get_regime_info()["event_state_trunk"]

    assert trunk["schema_version"] == "event_state_v2"
    assert trunk["source_atoms"]["source_surface"] == "heterogeneous"
    assert "boundary_atoms" in trunk
    assert "goal_atoms" in trunk
    assert "source_flow_atoms" in trunk
    assert np.isclose(trunk["phase_atoms"]["joint_financing_active_share"], 1.0)
    assert trunk["persistence_atoms"]["dynamic_entity_share"] > 0.0
    assert trunk["shared_state_atoms"]["shared_state_dim"] == wrapper._shared_state_dim


def test_event_state_trunk_static_surface_reports_zero_dynamic_share():
    frame = _make_investors_surface(
        ["edgar_only", "text_only", "mixed"],
        rows_per_profile=12,
        dynamic_within_entity=False,
    )

    wrapper = _fit_wrapper(frame)

    trunk = wrapper.get_regime_info()["event_state_trunk"]

    assert trunk["transition_atoms"]["investor_jump_share"] == 0.0
    assert trunk["transition_atoms"]["joint_jump_share"] == 0.0
    assert trunk["persistence_atoms"]["dynamic_entity_share"] == 0.0


def test_event_state_boundary_guard_exposes_investors_event_state_features():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_event_state_boundary_guard",
    )

    regime = wrapper.get_regime_info()
    process = regime["investors_process_contract"]

    assert process["requested_event_state_features"] is True
    assert process["effective_event_state_features"] is True
    assert process["event_state_feature_count"] > 0


def test_selective_event_state_guard_blocks_h1_even_on_source_free_surface():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_selective_event_state_guard",
        horizon=1,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_event_state_features"] is True
    assert source_regime["effective_event_state_features"] is False
    assert source_regime["event_state_activation_reason"] == "event_state_h1_blocked"


def test_selective_event_state_guard_blocks_source_rich_surface():
    frame = _make_investors_surface(["edgar_only", "text_only", "mixed"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_selective_event_state_guard",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]

    assert source_regime["requested_event_state_features"] is True
    assert source_regime["effective_event_state_features"] is False
    assert source_regime["event_state_activation_reason"] == "event_state_source_rich_surface_blocked"


def test_selective_event_state_guard_enables_hplus_source_free_surface():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_selective_event_state_guard",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    source_regime = regime["investors_source_activation"]
    process = regime["investors_process_contract"]

    assert source_regime["requested_event_state_features"] is True
    assert source_regime["effective_event_state_features"] is True
    assert source_regime["event_state_activation_reason"] == "event_state_selective_gate_open"
    assert process["effective_event_state_features"] is True
    assert process["event_state_feature_count"] > 0


def test_selective_event_state_guard_can_fit_and_predict_without_feature_mismatch():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_selective_event_state_guard",
        horizon=14,
    )

    X = frame[["core_signal"]].copy()
    preds = wrapper.predict(
        X,
        test_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=14,
    )

    assert preds.shape == (len(frame),)
    assert np.all(np.isfinite(preds))


def test_multiscale_state_guard_exposes_backbone_layout_and_seeded_predict_path():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_multiscale_state_guard",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    layout = regime["state_stream"]["backbone_layout"]
    shared_atoms = regime["event_state_trunk"]["shared_state_atoms"]
    X = frame[["core_signal"]].copy()
    preds = wrapper.predict(
        X,
        test_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=14,
    )

    assert layout["uses_multiscale_temporal_state"] is True
    assert layout["temporal_state_dim"] > 0
    assert layout["spectral_state_dim"] > 0
    assert shared_atoms["temporal_state_dim"] == layout["temporal_state_dim"]
    assert shared_atoms["spectral_state_dim"] == layout["spectral_state_dim"]
    assert shared_atoms["temporal_velocity_abs_mean"] >= 0.0
    assert shared_atoms["spectral_high_band_abs_mean"] >= 0.0
    assert regime["runtime"]["backbone_seed_rows"] > 0
    assert preds.shape == (len(frame),)
    assert np.all(np.isfinite(preds))


def test_temporal_state_guard_keeps_temporal_block_and_disables_spectral_block():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_temporal_state_guard",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    layout = regime["state_stream"]["backbone_layout"]
    shared_atoms = regime["event_state_trunk"]["shared_state_atoms"]

    assert layout["uses_temporal_state_features"] is True
    assert layout["uses_spectral_state_features"] is False
    assert layout["temporal_state_dim"] > 0
    assert layout["spectral_state_dim"] == 0
    assert shared_atoms["temporal_state_dim"] == layout["temporal_state_dim"]
    assert shared_atoms["spectral_state_dim"] == 0


def test_spectral_state_guard_keeps_spectral_block_and_disables_temporal_block():
    frame = _make_investors_surface(["none", "none", "none"], rows_per_profile=12)

    wrapper = _fit_wrapper(
        frame,
        variant="mainline_spectral_state_guard",
        horizon=14,
    )

    regime = wrapper.get_regime_info()
    layout = regime["state_stream"]["backbone_layout"]
    shared_atoms = regime["event_state_trunk"]["shared_state_atoms"]

    assert layout["uses_temporal_state_features"] is False
    assert layout["uses_spectral_state_features"] is True
    assert layout["temporal_state_dim"] == 0
    assert layout["spectral_state_dim"] > 0
    assert shared_atoms["temporal_state_dim"] == 0
    assert shared_atoms["spectral_state_dim"] == layout["spectral_state_dim"]