import numpy as np

from narrative.block3.models.single_model_mainline.lanes.funding_lane import (
    FundingLaneRuntime,
    _calibrate_anchor_residual_guard,
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