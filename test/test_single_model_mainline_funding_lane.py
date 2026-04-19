import numpy as np

from narrative.block3.models.single_model_mainline.lanes.funding_lane import (
    FundingLaneRuntime,
    _calibrate_anchor_residual_guard,
    _calibrate_source_scaling_guard,
    _guarded_funding_prediction,
)


def test_guarded_funding_prediction_can_fall_back_to_anchor():
    anchor = np.array([100.0, 120.0, 140.0], dtype=np.float64)
    residual = np.array([50.0, -25.0, 30.0], dtype=np.float64)

    preds = _guarded_funding_prediction(
        anchor_vec=anchor,
        residual_pred=residual,
        residual_blend=0.0,
        residual_cap=0.0,
    )

    assert np.allclose(preds, anchor)


def test_guarded_funding_prediction_supports_log_domain_compounding():
    anchor = np.array([100.0], dtype=np.float64)
    residual = np.array([np.log1p(200.0) - np.log1p(100.0)], dtype=np.float64)

    preds = _guarded_funding_prediction(
        anchor_vec=anchor,
        residual_pred=residual,
        residual_blend=1.0,
        residual_cap=np.inf,
        use_log_domain=True,
    )

    assert np.isclose(preds[0], 200.0)


def test_anchor_residual_guard_rejects_harmful_residual_path():
    anchor = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    target = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    residual_pred = np.array([40.0, -35.0, 45.0, -30.0], dtype=np.float64)
    residual_target = target - anchor

    blend, cap, anchor_mae, guarded_mae = _calibrate_anchor_residual_guard(
        anchor_vec=anchor,
        target_vec=target,
        residual_pred=residual_pred,
        residual_target=residual_target,
        anchor_dominance=10.0,
    )

    assert blend == 0.0
    assert cap == 0.0
    assert anchor_mae == 0.0
    assert guarded_mae == 0.0


def test_anchor_residual_guard_keeps_helpful_residual_path():
    anchor = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    target = np.array([120.0, 130.0, 140.0, 150.0], dtype=np.float64)
    residual_pred = np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    residual_target = target - anchor

    blend, cap, anchor_mae, guarded_mae = _calibrate_anchor_residual_guard(
        anchor_vec=anchor,
        target_vec=target,
        residual_pred=residual_pred,
        residual_target=residual_target,
        anchor_dominance=1.0,
    )

    assert blend > 0.0
    assert cap > 0.0
    assert guarded_mae < anchor_mae


def test_anchor_residual_guard_handles_sparse_jump_targets_without_zero_scale_collapse():
    anchor = np.full(11, 100.0, dtype=np.float64)
    target = np.array([100.0] * 10 + [500.0], dtype=np.float64)
    residual_pred = np.array([20.0] * 10 + [200.0], dtype=np.float64)
    residual_target = target - anchor

    blend, cap, anchor_mae, guarded_mae = _calibrate_anchor_residual_guard(
        anchor_vec=anchor,
        target_vec=target,
        residual_pred=residual_pred,
        residual_target=residual_target,
        anchor_dominance=1.0,
    )

    assert blend > 0.0
    assert cap > 0.0
    assert guarded_mae <= anchor_mae


def test_source_scaling_guard_can_shrink_source_rich_residuals():
    anchor = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    target = np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float64)
    residual_pred = np.array([50.0, 50.0, 10.0, 10.0], dtype=np.float64)
    source_scale = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

    strength, guarded_mae = _calibrate_source_scaling_guard(
        anchor_vec=anchor,
        target_vec=target,
        residual_pred=residual_pred,
        residual_blend=1.0,
        residual_cap=np.inf,
        source_scale=source_scale,
    )

    assert strength > 0.0
    assert guarded_mae < 30.0


def test_funding_lane_runtime_tracks_guard_parameters():
    runtime = FundingLaneRuntime(random_state=0)

    n_rows = 32
    lane_state = np.linspace(0.0, 1.0, n_rows * 4, dtype=np.float32).reshape(n_rows, 4)
    aux = np.linspace(1.0, 2.0, n_rows * 5, dtype=np.float32).reshape(n_rows, 5)
    anchor = np.linspace(100.0, 130.0, n_rows, dtype=np.float64)
    target = anchor + np.linspace(5.0, 15.0, n_rows, dtype=np.float64)

    runtime.fit(lane_state, target, aux_features=aux, anchor=anchor)
    preds = runtime.predict(lane_state, aux_features=aux, anchor=anchor)

    assert runtime._calibration_rows > 0
    assert runtime._anchor_calibration_mae >= runtime._guarded_calibration_mae
    assert runtime._residual_blend >= 0.0
    assert runtime._uses_jump_hurdle_head is True
    assert runtime._jump_event_rate == 1.0
    assert runtime._positive_jump_rows == n_rows
    assert runtime._positive_jump_median > 0.0
    assert np.all(preds >= 0.0)


def test_funding_lane_runtime_separates_jump_rows_from_anchor_rows():
    runtime = FundingLaneRuntime(random_state=3)

    n_rows = 64
    event = np.concatenate([np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)])
    lane_state = np.column_stack(
        [
            event,
            np.linspace(-1.0, 1.0, n_rows, dtype=np.float32),
            np.linspace(0.0, 2.0, n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    aux = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
            event,
            np.zeros(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
            np.linspace(1.0, 2.0, n_rows, dtype=np.float32),
        ]
    )
    anchor = np.full(n_rows, 100.0, dtype=np.float64)
    target = anchor + event.astype(np.float64) * 300.0

    runtime.fit(lane_state, target, aux_features=aux, anchor=anchor)
    preds = runtime.predict(lane_state, aux_features=aux, anchor=anchor)

    assert runtime._uses_jump_hurdle_head is True
    assert runtime._jump_event_rate > 0.45
    assert runtime._positive_jump_rows == 32
    assert preds[event > 0.5].mean() > preds[event < 0.5].mean() + 100.0
    assert preds[event < 0.5].mean() <= 130.0


def test_funding_lane_runtime_keeps_sparse_jump_cases_active():
    runtime = FundingLaneRuntime(random_state=5)

    n_rows = 48
    event = np.concatenate([np.zeros(42, dtype=np.float32), np.ones(6, dtype=np.float32)])
    lane_state = np.column_stack(
        [
            event,
            np.linspace(-1.0, 1.0, n_rows, dtype=np.float32),
            np.linspace(0.0, 2.0, n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    aux = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
            event,
            np.zeros(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
            np.linspace(1.0, 2.0, n_rows, dtype=np.float32),
        ]
    )
    anchor = np.full(n_rows, 100.0, dtype=np.float64)
    target = anchor + event.astype(np.float64) * 250.0

    runtime.fit(lane_state, target, aux_features=aux, anchor=anchor)
    preds = runtime.predict(lane_state, aux_features=aux, anchor=anchor)

    assert runtime._uses_jump_hurdle_head is True
    assert runtime._positive_jump_rows == 6
    assert runtime._residual_blend > 0.0
    assert preds[event > 0.5].mean() > preds[event < 0.5].mean()


def test_funding_lane_runtime_tracks_source_scaling_and_tail_focus():
    runtime = FundingLaneRuntime(random_state=9)

    n_rows = 48
    event = np.concatenate([np.zeros(24, dtype=np.float32), np.ones(24, dtype=np.float32)])
    lane_state = np.column_stack(
        [
            event,
            np.linspace(-1.0, 1.0, n_rows, dtype=np.float32),
            np.linspace(0.0, 2.0, n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    aux = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
            event,
            np.zeros(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
            np.linspace(1.0, 2.0, n_rows, dtype=np.float32),
        ]
    )
    anchor = np.full(n_rows, 100.0, dtype=np.float64)
    target = anchor + event.astype(np.float64) * np.linspace(25.0, 350.0, n_rows, dtype=np.float64)
    source_scale = np.concatenate([np.ones(24, dtype=np.float64), np.zeros(24, dtype=np.float64)])

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        source_scale=source_scale,
        use_log_domain=True,
        enable_source_scaling=True,
        tail_weight=2.0,
        tail_quantile=0.85,
    )

    assert runtime._log_domain_enabled is True
    assert runtime._source_scaling_enabled is True
    assert runtime._tail_weight == 2.0
    assert np.isclose(runtime._tail_quantile, 0.85)
    assert runtime._source_scale_strength >= 0.0
    assert runtime._source_scale_reliability >= 0.0