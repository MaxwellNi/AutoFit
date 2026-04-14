import numpy as np

from narrative.block3.models.single_model_mainline.lanes.binary_lane import BinaryLaneRuntime


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
    assert runtime._calibrator_name in {"identity", "platt", "isotonic"}
    assert preds[lag1_active > 0.5].mean() > preds[lag1_active < 0.5].mean()