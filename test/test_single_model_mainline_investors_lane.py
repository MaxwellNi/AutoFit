import numpy as np

from narrative.block3.models.single_model_mainline.lanes.investors_lane import (
    InvestorsLaneRuntime,
    _calibrate_horizon_anchor_mix,
    _horizon_model_anchor,
    _hurdle_activity_gate,
    _read_policy_strength,
)


def test_investors_lane_fits_source_specialists_for_mixed_source_profiles():
    runtime = InvestorsLaneRuntime(random_state=0)

    n_rows = 96
    lane_state = np.zeros((n_rows, 6), dtype=np.float32)
    aux = np.zeros((n_rows, 5), dtype=np.float32)
    anchor = np.full(n_rows, 2.0, dtype=np.float64)
    source = np.zeros((n_rows, 7), dtype=np.float32)
    target = np.full(n_rows, 2.0, dtype=np.float64)

    source[:32, 0] = 1.0
    source[:32, 2] = 1.0
    target[:32] = np.linspace(6.5, 7.5, 32)

    source[32:64, 1] = 1.0
    source[32:64, 3] = 1.0
    source[32:64, 6] = 0.1
    target[32:64] = np.linspace(4.5, 5.5, 32)

    source[64:, 0] = 1.0
    source[64:, 1] = 1.0
    source[64:, 2] = 1.0
    source[64:, 3] = 1.0
    source[64:, 6] = 0.6
    target[64:] = np.linspace(2.4, 3.0, 32)

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        enable_source_features=True,
        enable_source_specialists=True,
        horizon=1,
        task_name="task2_forecast",
    )

    assert set(runtime._positive_specialists.keys()) == {1, 2, 3}

    preds = runtime.predict(
        lane_state,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        enable_source_features=True,
        enable_source_specialists=True,
        enable_source_guard=True,
    )
    assert np.all(preds >= 0.0)
    assert preds[:32].mean() > preds[32:64].mean() > preds[64:].mean()


def test_investors_lane_core_only_path_keeps_nonnegative_predictions_without_source_features():
    runtime = InvestorsLaneRuntime(random_state=1)

    lane_state = np.zeros((24, 4), dtype=np.float32)
    aux = np.zeros((24, 5), dtype=np.float32)
    anchor = np.linspace(1.0, 3.0, 24, dtype=np.float64)
    target = anchor + 0.5

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        horizon=1,
        task_name="task1_outcome",
    )

    preds = runtime.predict(lane_state, aux_features=aux, anchor=anchor)
    assert np.all(preds >= 0.0)
    assert preds.shape == target.shape


def test_investors_lane_can_use_source_features_without_enabling_specialists():
    runtime = InvestorsLaneRuntime(random_state=2)

    lane_state = np.zeros((18, 4), dtype=np.float32)
    aux = np.zeros((18, 5), dtype=np.float32)
    anchor = np.full(18, 2.0, dtype=np.float64)
    source = np.zeros((18, 7), dtype=np.float32)
    source[:, 0] = 1.0
    source[:, 2] = 2.0
    target = np.linspace(2.0, 5.0, 18, dtype=np.float64)

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        enable_source_features=True,
        horizon=1,
        task_name="task2_forecast",
    )

    assert runtime._positive_specialists == {}

    preds = runtime.predict(
        lane_state,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        enable_source_features=True,
    )
    assert np.all(preds >= 0.0)


def test_read_policy_strength_requires_higher_reliability_for_downward_blends():
    upward = _read_policy_strength(default_blend=0.70, target_blend=0.90, reliability=0.20)
    low_rel_downward = _read_policy_strength(default_blend=0.70, target_blend=0.30, reliability=0.20)
    high_rel_downward = _read_policy_strength(default_blend=0.70, target_blend=0.30, reliability=0.55)

    assert upward == 0.20
    assert low_rel_downward == 0.0
    assert high_rel_downward > 0.0


def test_horizon_model_anchor_smooths_long_horizon_histories():
    anchor = np.array([8.0, 8.0], dtype=np.float64)
    aux = np.array(
        [
            [8.0, 4.0, 2.0, 1.0, 10.0],
            [8.0, 7.0, 6.5, 0.2, 10.0],
        ],
        dtype=np.float32,
    )

    h1_anchor = _horizon_model_anchor(anchor, aux, horizon=1, enabled=False)
    h14_anchor = _horizon_model_anchor(anchor, aux, horizon=14, enabled=True)

    assert np.allclose(h1_anchor, anchor)
    assert h14_anchor[0] < anchor[0]
    assert h14_anchor[0] > 2.0
    assert h14_anchor[1] <= anchor[1]


def test_horizon_anchor_guard_rejects_harmful_smoothed_anchor():
    raw_anchor = np.array([8.0, 8.0, 8.0], dtype=np.float64)
    smoothed_anchor = np.array([2.5, 3.0, 4.0], dtype=np.float64)
    target = np.array([8.0, 7.8, 8.2], dtype=np.float64)

    mix, reliability = _calibrate_horizon_anchor_mix(
        raw_anchor=raw_anchor,
        smoothed_anchor=smoothed_anchor,
        target=target,
        enabled=True,
    )

    assert mix == 0.0
    assert reliability == 0.0


def test_horizon_anchor_guard_can_accept_smoothed_anchor_when_it_wins():
    raw_anchor = np.array([8.0, 8.0, 8.0], dtype=np.float64)
    smoothed_anchor = np.array([5.0, 5.2, 5.4], dtype=np.float64)
    target = np.array([5.1, 5.3, 5.5], dtype=np.float64)

    mix, reliability = _calibrate_horizon_anchor_mix(
        raw_anchor=raw_anchor,
        smoothed_anchor=smoothed_anchor,
        target=target,
        enabled=True,
    )

    assert mix > 0.0
    assert reliability > 0.0


def test_hurdle_activity_gate_prefers_active_histories():
    anchor = np.array([0.5, 6.0], dtype=np.float64)
    aux = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [6.0, 5.5, 5.0, 1.2, 10.0],
        ],
        dtype=np.float32,
    )

    gate = _hurdle_activity_gate(anchor=anchor, aux_features=aux, fallback=0.3, horizon=14)

    assert gate.shape == (2,)
    assert gate[1] > gate[0]
    assert np.all(gate >= 0.0)
    assert np.all(gate <= 1.0)


def test_long_horizon_runtime_enables_hurdle_contract_without_exemplar_branch():
    runtime = InvestorsLaneRuntime(random_state=3)

    n_rows = 48
    lane_state = np.zeros((n_rows, 4), dtype=np.float32)
    aux = np.zeros((n_rows, 5), dtype=np.float32)
    anchor = np.full(n_rows, 8.0, dtype=np.float64)
    target = np.full(n_rows, 6.0, dtype=np.float64)

    aux[:24, :] = np.array([8.0, 4.0, 2.0, 1.0, 10.0], dtype=np.float32)
    aux[24:, :] = np.array([8.0, 7.0, 6.0, 0.2, 10.0], dtype=np.float32)
    target[:24] = 5.5
    target[24:] = 6.8

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        enable_hurdle_head=True,
        enable_count_jump=True,
        count_jump_strength=0.30,
        enable_count_sparsity_gate=True,
        count_sparsity_gate_strength=0.75,
        horizon=14,
        anchor_blend=0.70,
        task_name="task2_forecast",
    )

    preds = runtime.predict(lane_state, aux_features=aux, anchor=anchor)

    assert runtime._use_hurdle_head is True
    assert runtime._use_count_jump is True
    assert runtime._use_sparsity_gate is True
    assert runtime._exemplar_model is None
    assert 0.0 <= runtime._horizon_anchor_mix <= 1.0
    assert 0.0 <= runtime._horizon_anchor_mix_reliability <= 1.0
    assert 0.0 <= runtime._global_anchor_blend <= 1.0
    assert 0.0 <= runtime._global_jump_strength <= 1.0
    assert np.all(preds >= 0.0)