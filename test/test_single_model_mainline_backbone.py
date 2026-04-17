import numpy as np
import pandas as pd

from narrative.block3.models.single_model_mainline.backbone import SharedTemporalBackbone, SharedTemporalBackboneSpec


def _make_backbone_context(start_day: str, values: list[tuple[str, float, float]]) -> pd.DataFrame:
    rows = []
    base_day = pd.Timestamp(start_day)
    for idx, (entity_id, core_signal, core_volume) in enumerate(values):
        rows.append(
            {
                "entity_id": entity_id,
                "crawled_date_day": base_day + pd.Timedelta(days=idx),
                "core_signal": float(core_signal),
                "core_volume": float(core_volume),
            }
        )
    return pd.DataFrame(rows)


def test_multiscale_backbone_reports_temporal_and_spectral_layout_and_uses_seed_context() -> None:
    train = _make_backbone_context(
        "2024-01-01",
        [
            ("entity_a", 0.0, 1.0),
            ("entity_a", 1.0, 2.0),
            ("entity_a", 2.5, 3.0),
            ("entity_a", 4.0, 5.0),
            ("entity_b", 0.0, 2.0),
            ("entity_b", 0.5, 2.5),
            ("entity_b", 1.5, 3.0),
            ("entity_b", 3.0, 4.0),
        ],
    )
    test = _make_backbone_context(
        "2024-01-09",
        [
            ("entity_a", 5.0, 6.0),
            ("entity_a", 6.0, 7.0),
            ("entity_b", 3.5, 4.5),
            ("entity_b", 5.0, 5.5),
        ],
    )
    feature_cols = ["core_signal", "core_volume"]
    backbone = SharedTemporalBackbone(
        spec=SharedTemporalBackboneSpec(
            enable_multiscale_temporal_state=True,
            temporal_state_windows=(2, 3, 4),
            compact_state_dim=8,
        ),
        random_state=7,
    )

    shared_train = backbone.fit_transform(
        train[feature_cols],
        feature_cols=feature_cols,
        context_frame=train,
    )
    layout = backbone.describe_state_layout()
    seed = backbone.build_context_seed(train, train[feature_cols])
    shared_test = backbone.transform(
        test[feature_cols],
        context_frame=test,
        seed_frame=seed,
    )

    temporal_offset = layout["compact_state_dim"] + layout["summary_state_dim"]
    spectral_offset = temporal_offset + layout["temporal_state_dim"]
    temporal_block = shared_test[:, temporal_offset:spectral_offset]
    spectral_block = shared_test[:, spectral_offset:spectral_offset + layout["spectral_state_dim"]]

    assert shared_train.shape[0] == len(train)
    assert shared_test.shape[0] == len(test)
    assert layout["uses_multiscale_temporal_state"] is True
    assert layout["temporal_state_dim"] > 0
    assert layout["spectral_state_dim"] > 0
    assert layout["shared_state_dim"] == shared_train.shape[1]
    assert seed is not None
    assert len(seed) > 0
    assert len(seed) <= 8
    assert np.all(np.isfinite(shared_test))
    assert float(np.abs(temporal_block).sum()) > 0.0
    assert float(np.abs(spectral_block).sum()) > 0.0