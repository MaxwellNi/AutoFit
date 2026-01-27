import pytest

pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")
import pandas as pd

from narrative.data_preprocessing.schema_profiler import profile_parquet_dataset


def _write_dataset(root, df, partition_cols=None):
    table = pa.Table.from_pandas(df)
    ds.write_dataset(
        table,
        base_dir=str(root),
        format="parquet",
        partitioning=partition_cols,
        existing_data_behavior="overwrite_or_ignore",
    )


def test_profile_offers_schema_infers_time_and_keys(tmp_path):
    offers = tmp_path / "offers"
    df = pd.DataFrame(
        {
            "platform_name": ["p1", "p1"],
            "offer_id": ["o1", "o1"],
            "crawled_date": pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True),
            "crawled_date_day": ["2023-01-01", "2023-01-02"],
            "metric": [1.0, 2.0],
        }
    )
    _write_dataset(offers, df, partition_cols=["crawled_date_day"])

    prof = profile_parquet_dataset(offers, name="offers")
    assert "crawled_date" in prof.time_columns
    assert ["platform_name", "offer_id"] in prof.key_candidates


def test_profile_edgar_schema_infers_time_and_keys(tmp_path):
    edgar = tmp_path / "edgar"
    df = pd.DataFrame(
        {
            "cik": ["1", "2"],
            "filed_date": pd.to_datetime(["2023-01-01", "2023-02-01"], utc=True),
            "accession_number": ["x1", "x2"],
        }
    )
    _write_dataset(edgar, df)

    prof = profile_parquet_dataset(edgar, name="edgar")
    assert "filed_date" in prof.time_columns
    assert ["cik"] in prof.key_candidates
