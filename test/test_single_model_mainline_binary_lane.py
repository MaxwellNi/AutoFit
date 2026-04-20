import numpy as np

from narrative.block3.models.single_model_mainline.lanes.binary_lane import (
    BinaryLaneRuntime,
)


def test_binary_lane_hazard_adapter_uses_history_transition_prior():
    runtime = BinaryLaneRuntime(random_state=5)

    n_rows = 64
    lag1_active = np.concatenate([np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)])
    lane_state = np.column_stack(
        [
            np.linspace(-1.0, 1.0, n_rows, dtype=np.float32),
            np.repeat([0.0, 1.0], repeats=32).astype(np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    aux = np.column_stack(
        [
            lag1_active,
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
            np.linspace(1.0, 2.0, n_rows, dtype=np.float32),
            np.zeros(n_rows, dtype=np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    target = np.concatenate(
        [
            np.tile([0.0, 1.0], 16),
            np.ones(32, dtype=np.float32),
        ]
    )

    runtime.fit(lane_state, target, aux_features=aux)
    preds = runtime.predict(lane_state, aux_features=aux)

    assert runtime._hazard_rows == 32
    assert runtime._transition_rate > 0.0
    assert runtime._persistence_rate > runtime._transition_rate
    assert runtime._calibrator_name == "identity"
    assert preds[lag1_active > 0.5].mean() > preds[lag1_active < 0.5].mean()


def test_neural_hazard_head_trains_end_to_end():
    """Neural hazard MLP trains with survival NLL and produces valid probabilities."""
    runtime = BinaryLaneRuntime(random_state=42)

    n_rows = 128
    lag1_active = np.concatenate([np.zeros(64, dtype=np.float32), np.ones(64, dtype=np.float32)])
    lane_state = np.column_stack(
        [
            np.linspace(-2.0, 2.0, n_rows, dtype=np.float32),
            np.repeat([0.0, 1.0], repeats=64).astype(np.float32),
            np.ones(n_rows, dtype=np.float32),
        ]
    )
    aux = np.column_stack(
        [
            lag1_active,
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
        ]
    )
    target = np.concatenate(
        [
            np.tile([0.0, 1.0], 32),
            np.ones(64, dtype=np.float32),
        ]
    )

    runtime.fit(lane_state, target, aux_features=aux, horizon=7)
    preds = runtime.predict(lane_state, aux_features=aux)

    assert runtime._model is not None
    assert runtime._fitted
    assert preds.shape == (n_rows,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))
    assert runtime._temperature >= 0.5


def test_binary_lane_constant_target_returns_constant_prediction():
    """When all labels are the same, the lane returns a constant prediction."""
    runtime = BinaryLaneRuntime(random_state=7)

    n_rows = 32
    lane_state = np.ones((n_rows, 3), dtype=np.float32)
    target = np.ones(n_rows, dtype=np.float32)

    runtime.fit(lane_state, target)
    preds = runtime.predict(lane_state)

    assert runtime._model is None
    assert np.allclose(preds, 1.0)
    assert 0.0 <= runtime._calibration_shrinkage_strength <= 0.80