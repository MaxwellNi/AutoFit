import pandas as pd

from narrative.nbi_nci.nbi_supervision import load_all_supervision


def test_nbi_supervision_loaders(tmp_path):
    base = tmp_path
    # goemotions
    go_dir = base / "goemotions"
    go_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": ["a", "b"],
            "labels": ["[0,1]", "[2]"],
            "split": ["train", "train"],
        }
    ).to_parquet(go_dir / "train.parquet", index=False)
    # financial phrasebank
    fp_dir = base / "financial_phrasebank"
    fp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sentence": ["x", "y"],
            "label": [0, 1],
        }
    ).to_parquet(fp_dir / "train.parquet", index=False)
    # media_frames
    mf_dir = base / "media_frames"
    mf_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": ["m"],
            "frame": ["econ"],
        }
    ).to_parquet(mf_dir / "train.parquet", index=False)

    datasets = load_all_supervision(base)
    assert "goemotions" in datasets
    assert len(datasets["goemotions"]) == 2
    assert len(datasets["financial_phrasebank"]) == 2
    assert len(datasets["media_frames"]) == 1
