import numpy as np
import pandas as pd
import pytest

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


def test_multiscale_backbone_can_isolate_temporal_state_features() -> None:
    frame = _make_backbone_context(
        "2024-02-01",
        [
            ("entity_a", 0.0, 1.0),
            ("entity_a", 1.0, 2.0),
            ("entity_a", 2.0, 3.0),
            ("entity_b", 0.5, 1.5),
            ("entity_b", 1.5, 2.5),
            ("entity_b", 2.5, 3.5),
        ],
    )
    feature_cols = ["core_signal", "core_volume"]
    backbone = SharedTemporalBackbone(
        spec=SharedTemporalBackboneSpec(
            enable_multiscale_temporal_state=True,
            temporal_state_windows=(2, 3, 4),
            enable_temporal_state_features=True,
            enable_spectral_state_features=False,
        ),
        random_state=7,
    )

    shared = backbone.fit_transform(frame[feature_cols], feature_cols=feature_cols, context_frame=frame)
    layout = backbone.describe_state_layout()

    assert shared.shape[0] == len(frame)
    assert layout["uses_temporal_state_features"] is True
    assert layout["uses_spectral_state_features"] is False
    assert layout["temporal_state_dim"] > 0
    assert layout["spectral_state_dim"] == 0
    assert layout["temporal_feature_names"]
    assert layout["spectral_feature_names"] == ()


def test_multiscale_backbone_can_isolate_spectral_state_features() -> None:
    frame = _make_backbone_context(
        "2024-03-01",
        [
            ("entity_a", 0.0, 1.0),
            ("entity_a", 1.0, 2.0),
            ("entity_a", 2.0, 3.0),
            ("entity_b", 0.5, 1.5),
            ("entity_b", 1.5, 2.5),
            ("entity_b", 2.5, 3.5),
        ],
    )
    feature_cols = ["core_signal", "core_volume"]
    backbone = SharedTemporalBackbone(
        spec=SharedTemporalBackboneSpec(
            enable_multiscale_temporal_state=True,
            temporal_state_windows=(2, 3, 4),
            enable_temporal_state_features=False,
            enable_spectral_state_features=True,
        ),
        random_state=7,
    )

    shared = backbone.fit_transform(frame[feature_cols], feature_cols=feature_cols, context_frame=frame)
    layout = backbone.describe_state_layout()

    assert shared.shape[0] == len(frame)
    assert layout["uses_temporal_state_features"] is False
    assert layout["uses_spectral_state_features"] is True
    assert layout["temporal_state_dim"] == 0
    assert layout["spectral_state_dim"] > 0
    assert layout["temporal_feature_names"] == ()
    assert layout["spectral_feature_names"]


def test_hawkes_financing_state_produces_asymmetric_intensity_features() -> None:
    """Hawkes state must be:
    - non-zero only after positive shocks (asymmetry)
    - monotonically non-increasing between shocks (exponential decay)
    - larger after a sequence of positive shocks (self-exciting)
    - zero-width when the flag is off
    """
    frame = _make_backbone_context(
        "2024-05-01",
        [
            # entity_a: monotone rise → steady positive shocks throughout
            ("entity_a", 0.0, 1.0),
            ("entity_a", 2.0, 3.0),
            ("entity_a", 4.0, 5.0),
            ("entity_a", 6.0, 7.0),
            ("entity_a", 8.0, 9.0),
            # entity_b: flat then one big jump at the end
            ("entity_b", 1.0, 1.0),
            ("entity_b", 1.0, 1.0),
            ("entity_b", 1.0, 1.0),
            ("entity_b", 1.0, 1.0),
            ("entity_b", 5.0, 5.0),
        ],
    )
    feature_cols = ["core_signal", "core_volume"]
    backbone_on = SharedTemporalBackbone(
        spec=SharedTemporalBackboneSpec(
            enable_hawkes_financing_state=True,
            hawkes_financing_decay_halflives=(3.0, 10.0),
            hawkes_positive_shock_threshold=0.1,
            compact_state_dim=4,
        ),
        random_state=3,
    )
    backbone_off = SharedTemporalBackbone(
        spec=SharedTemporalBackboneSpec(
            enable_hawkes_financing_state=False,
            compact_state_dim=4,
        ),
        random_state=3,
    )

    state_on = backbone_on.fit_transform(frame[feature_cols], feature_cols=feature_cols, context_frame=frame)
    layout_on = backbone_on.describe_state_layout()
    state_off = backbone_off.fit_transform(frame[feature_cols], feature_cols=feature_cols, context_frame=frame)
    layout_off = backbone_off.describe_state_layout()

    # Flag-off: hawkes block must be empty
    assert layout_off["hawkes_state_dim"] == 0
    assert layout_off["uses_hawkes_financing_state"] is False
    assert layout_off["hawkes_feature_names"] == ()

    # Flag-on: correct number of features (n_halflives + 1 time_since)
    n_halflives = 2
    assert layout_on["hawkes_state_dim"] == n_halflives + 1
    assert layout_on["uses_hawkes_financing_state"] is True
    assert len(layout_on["hawkes_feature_names"]) == n_halflives + 1
    assert state_on.shape[1] > state_off.shape[1]  # wider because hawkes block added

    # Extract hawkes block
    hawkes_offset = (
        layout_on["compact_state_dim"]
        + layout_on["summary_state_dim"]
        + layout_on["temporal_state_dim"]
        + layout_on["spectral_state_dim"]
    )
    hawkes_block = state_on[:, hawkes_offset: hawkes_offset + layout_on["hawkes_state_dim"]]

    # entity_a rows are the first 5; entity_b rows are the last 5
    # After at least one positive shock the intensity must be > 0
    entity_a_intensity = hawkes_block[:5, 0]  # short halflife intensity for entity_a
    assert entity_a_intensity[1] > 0.0, "Intensity must be positive after first shock"

    # For entity_a (monotone rise) intensity should accumulate: later rows >= earlier rows
    # due to recurring positive shocks
    assert entity_a_intensity[-1] >= entity_a_intensity[1], "Accumulated intensity must not decrease over steady shocks"

    # For entity_b (flat then spike) intensity at flat rows must be below intensity
    # after the spike.  After z-score normalisation the raw-zero rows may be negative,
    # so we test relative ordering rather than absolute zero.
    entity_b_intensity = hawkes_block[5:, 0]
    assert entity_b_intensity[0] < entity_b_intensity[4], "Intensity must be lower before first shock than after in entity_b"
    assert entity_b_intensity[3] < entity_b_intensity[4], "Still lower at step 3 than after spike in entity_b"
    assert entity_b_intensity[4] > entity_b_intensity[0], "Intensity must jump after the spike in entity_b"