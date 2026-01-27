import pytest
import pandas as pd

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")

from narrative.data_preprocessing.parquet_catalog import (
    scan_offers_static,
    scan_snapshots,
    scan_edgar,
)


def _write_offers_dataset(root):
    data = pd.DataFrame(
        {
            "offer_id": ["A", "A", "B"],
            "platform_name": ["p1", "p1", "p2"],
            "crawled_date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-02"], utc=True
            ),
            "crawled_date_day": ["2023-01-01", "2023-01-02", "2023-01-02"],
            "metric": [1.0, 2.0, 3.0],
        }
    )
    table = pa.Table.from_pandas(data)
    ds.write_dataset(
        table,
        base_dir=str(root),
        format="parquet",
        partitioning=["crawled_date_day"],
        existing_data_behavior="overwrite_or_ignore",
    )


def _write_edgar_dataset(root):
    data = pd.DataFrame(
        {
            "cik": [1001, 1001, 2002],
            "filed_date": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-01-15"], utc=True
            ),
            "accession_number": ["x1", "x2", "y1"],
            "metric": [10.0, 20.0, 30.0],
        }
    )
    table = pa.Table.from_pandas(data)
    ds.write_dataset(
        table,
        base_dir=str(root),
        format="parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def test_scan_snapshots_filters_by_offer_and_time(tmp_path):
    offers_dir = tmp_path / "offers"
    _write_offers_dataset(offers_dir)

    out = scan_snapshots(
        ["A"],
        time_range=(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02 23:59:59")),
        base_dir=offers_dir,
    )
    assert set(out["offer_id"].astype(str).tolist()) == {"A"}
    assert out["crawled_date"].min() >= pd.Timestamp("2023-01-02", tz="UTC")


def test_scan_offers_static_keeps_latest(tmp_path):
    offers_dir = tmp_path / "offers"
    _write_offers_dataset(offers_dir)

    out = scan_offers_static(["A", "B"], base_dir=offers_dir)
    assert len(out) == 2
    row_a = out[out["offer_id"] == "A"].iloc[0]
    row_b = out[out["offer_id"] == "B"].iloc[0]
    assert float(row_a["metric"]) == 2.0
    assert float(row_b["metric"]) == 3.0


def test_scan_edgar_filters_by_entity_and_time(tmp_path):
    edgar_dir = tmp_path / "edgar_accessions"
    _write_edgar_dataset(edgar_dir)

    out = scan_edgar(
        ["1001"],
        time_range=(pd.Timestamp("2023-01-15"), pd.Timestamp("2023-03-01")),
        base_dir=edgar_dir,
    )
    assert set(out["cik"].astype(str).tolist()) == {"1001"}
    assert out["filed_date"].min() >= pd.Timestamp("2023-01-15", tz="UTC")
