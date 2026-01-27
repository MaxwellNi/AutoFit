import pandas as pd

from narrative.evaluation.report import write_paper_tables


def test_write_paper_tables(tmp_path):
    df = pd.DataFrame(
        {
            "backbone": ["dlinear", "dlinear"],
            "fusion_type": ["none", "concat"],
            "rmse": [1.0, 1.1],
            "mae": [0.5, 0.6],
            "r2": [0.1, 0.2],
            "module_flags": [{"nonstat": False}, {"nonstat": True}],
            "train_time_sec": [10.0, 12.0],
        }
    )
    out_dir = write_paper_tables(df, tmp_path)
    assert (out_dir / "main_results_table.parquet").exists()
    assert (out_dir / "ablation_table.parquet").exists()
    assert (out_dir / "faithfulness_table.parquet").exists()
    assert (out_dir / "efficiency_table.parquet").exists()
