import pandas as pd

from narrative.data_preprocessing.edgar_feature_store import (
    EDGAR_FEATURE_COLUMNS,
    align_edgar_features_to_snapshots,
    aggregate_edgar_features,
    extract_edgar_features,
)


def _payload(total, sold=None):
    return {
        "offeringData": {
            "offeringSalesAmounts": {
                "totalOfferingAmount": str(total),
                "totalAmountSold": None if sold is None else str(sold),
            },
            "minimumInvestmentAccepted": "1000",
        }
    }


def test_align_edgar_features_forward_fill_and_masks():
    edgar_df = pd.DataFrame(
        {
            "cik": ["1", "1"],
            "filed_date": pd.to_datetime(["2023-01-01", "2023-02-01"], utc=True),
            "submission_offering_data": [_payload(100, sold=50), _payload(200, sold=None)],
        }
    )
    edgar_features = aggregate_edgar_features(extract_edgar_features(edgar_df))

    snapshots_df = pd.DataFrame(
        {
            "platform_name": ["p", "p", "p"],
            "offer_id": ["o", "o", "o"],
            "cik": ["1", "1", "1"],
            "crawled_date": pd.to_datetime(
                ["2022-12-31", "2023-01-15", "2023-02-15"], utc=True
            ),
        }
    )

    aligned = align_edgar_features_to_snapshots(edgar_features, snapshots_df)

    for col in EDGAR_FEATURE_COLUMNS:
        assert f"last_{col}" in aligned.columns
        assert f"mean_{col}" in aligned.columns
        assert f"ema_{col}" in aligned.columns
        assert f"last_{col}_is_missing" in aligned.columns

    first = aligned.iloc[0]
    assert bool(first["edgar_has_filing"]) is False
    assert bool(first["edgar_valid"]) is False
    assert all(first[f"last_{col}_is_missing"] for col in EDGAR_FEATURE_COLUMNS)

    mid = aligned.iloc[1]
    assert float(mid["last_total_offering_amount"]) == 100.0
    assert bool(mid["last_total_amount_sold_is_missing"]) is False
    assert bool(mid["edgar_valid"]) is True
    assert bool(mid["edgar_valid"]) is True

    last = aligned.iloc[2]
    assert float(last["last_total_offering_amount"]) == 200.0
    assert bool(last["last_total_amount_sold_is_missing"]) is True
    assert bool(last["edgar_valid"]) is True
    assert bool(last["edgar_valid"]) is True
