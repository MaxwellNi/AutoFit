import pandas as pd

from narrative.block3.models.v740_alpha import _select_state_feature_cols
from narrative.block3.models.v740_multisource_features import (
    DualClockConfig,
    build_source_native_text_memory,
)


def test_source_native_text_memory_dedupes_repeated_daily_embeddings():
    df = pd.DataFrame(
        {
            "entity_id": ["e1"] * 5,
            "crawled_date_day": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-10",
                    "2025-01-11",
                ]
            ),
            "text_emb_0": [0.1, 0.1, 0.1, 0.9, 0.9],
            "text_emb_1": [0.2, 0.2, 0.2, 0.4, 0.4],
        }
    )

    memory = build_source_native_text_memory(
        df,
        prediction_time=pd.Timestamp("2025-01-12"),
        source_cols=["text_emb_0", "text_emb_1"],
        cfg=DualClockConfig(max_events=4),
    )

    recent = memory["recent_tokens"]
    bucket = memory["bucket_tokens"]
    active = recent[recent[:, 2] > 0.0]

    assert active.shape[0] == 2
    assert float(active[:, 3].max()) > 0.0
    assert float(active[:, 4].max()) > 0.0
    assert int(bucket[:, 0].sum()) == 2


def test_state_feature_selection_excludes_source_columns():
    train_raw = pd.DataFrame(
        {
            "entity_id": ["e1", "e1", "e2", "e2"] * 20,
            "crawled_date_day": pd.date_range("2025-01-01", periods=80, freq="D"),
            "funding_raised_usd": list(range(80)),
            "core_signal": [float(i % 11) for i in range(80)],
            "aux_signal": [float((i * 3) % 17) for i in range(80)],
            "text_emb_0": [float(i % 5) for i in range(80)],
            "text_emb_1": [float((i + 1) % 7) for i in range(80)],
            "last_total_offering_amount": [float(i % 13) for i in range(80)],
            "edgar_has_filing": [float(i % 2) for i in range(80)],
        }
    )

    feature_cols = _select_state_feature_cols(
        train_raw,
        target="funding_raised_usd",
        max_covariates=6,
        source_exclude=[
            "text_emb_0",
            "text_emb_1",
            "last_total_offering_amount",
            "edgar_has_filing",
        ],
    )

    assert "core_signal" in feature_cols
    assert "aux_signal" in feature_cols
    assert "text_emb_0" not in feature_cols
    assert "text_emb_1" not in feature_cols
    assert "last_total_offering_amount" not in feature_cols
    assert "edgar_has_filing" not in feature_cols