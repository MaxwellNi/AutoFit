import pytest
import pandas as pd

from narrative.auto_fit.diagnose_dataset import diagnose_dataset


def test_diagnose_dataset_exports_files(tmp_path):
    df = pd.DataFrame(
        {
            "offer_id": ["1", "1", "1", "2", "2"],
            "crawled_date": ["2023-01-01", "2023-01-03", "2023-01-10", "2023-01-02", "2023-01-05"],
            "x1": [1.0, 2.0, 3.0, 1.0, 1.5],
            "edgar_feat": [0.1, 0.2, 0.2, 0.1, 0.3],
            "target": [1.0, 1.1, 1.2, 0.9, 1.0],
        }
    )
    out = diagnose_dataset(
        df,
        key_col="offer_id",
        time_col="crawled_date",
        value_cols=["x1"],
        target_col="target",
        output_dir=tmp_path,
    )
    assert "irregularity_score" in out
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "diagnostics_table.parquet").exists()
