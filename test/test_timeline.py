import pytest
import pandas as pd

from narrative.data_preprocessing.timeline import add_time_index, build_cutoff_mask, apply_snapshot_cutoff


def test_add_time_index_irregular_deltas():
    df = pd.DataFrame(
        {
            "platform_name": ["p", "p", "p", "q"],
            "offer_id": ["1", "1", "1", "2"],
            "crawled_date": ["2023-01-01", "2023-01-03", "2023-01-10", "2023-01-02"],
        }
    )
    out = add_time_index(df)
    g = out[out["offer_id"] == "1"].sort_values("crawled_date")
    assert g["event_idx"].tolist() == [0, 1, 2]
    assert g["time_delta_days"].tolist()[1:] == [2.0, 7.0]


def test_build_cutoff_mask():
    df = pd.DataFrame(
        {
            "crawled_date": pd.to_datetime(
                ["2023-01-01", "2023-01-10"], utc=True
            )
        }
    )
    mask = build_cutoff_mask(df, cutoff_time="2023-01-05")
    assert mask.tolist() == [True, False]


def test_apply_snapshot_cutoff_max_k():
    df = pd.DataFrame(
        {
            "platform_name": ["p"] * 4,
            "offer_id": ["1"] * 4,
            "crawled_date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        }
    )
    out, mask = apply_snapshot_cutoff(df, max_snapshots=2, use_first_k=True)
    assert len(out) == 2
    assert out["event_idx"].tolist() == [0, 1]
