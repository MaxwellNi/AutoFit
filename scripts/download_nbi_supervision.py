from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_import_datasets():
    try:
        import datasets  # type: ignore
        return datasets
    except Exception:
        return None

def _write_split(df, out_dir: Path, split: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df["split"] = split
    df.to_parquet(out_dir / f"{split}.parquet", index=False)
    df.to_parquet(out_dir / f"{split}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NBI supervision datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/nbi_supervision"))
    parser.add_argument("--phrasebank_config", type=str, default="sentences_50agree")
    args = parser.parse_args()

    datasets = _safe_import_datasets()
    if datasets is None:
        print("datasets library not installed. Skipping downloads.")
        print("Install with: pip install datasets")
        return

    # GoEmotions
    try:
        ds = datasets.load_dataset("go_emotions")
        for split in ds.keys():
            data = ds[split]
            df = data.to_pandas()[["text", "labels"]].copy()
            df["labels"] = df["labels"].apply(json.dumps)
            _write_split(df, args.output_dir / "goemotions", split)
        print("✓ goemotions downloaded")
    except Exception as e:
        print(f"⚠️ goemotions download failed: {e}")

    # Financial PhraseBank
    try:
        ds = datasets.load_dataset("financial_phrasebank", args.phrasebank_config)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        text_col = "sentence" if "sentence" in df.columns else "text"
        df = df[[text_col, "label"]].rename(columns={text_col: "text"})
        _write_split(df, args.output_dir / "financial_phrasebank", split)
        print("✓ financial_phrasebank downloaded")
    except Exception as e:
        print(f"⚠️ financial_phrasebank download failed: {e}")

    # CommitmentBank (optional)
    try:
        ds = datasets.load_dataset("commitmentbank")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        text_col = "text" if "text" in df.columns else "sentence"
        label_col = "label" if "label" in df.columns else "commitment"
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        _write_split(df, args.output_dir / "commitmentbank", split)
        print("✓ commitmentbank downloaded")
    except Exception as e:
        print(f"⚠️ commitmentbank download failed: {e}")

    # Placeholder notices for non-HF datasets
    print("Note: media_frames, persuasion_strategies, bioscope require manual download.")


if __name__ == "__main__":
    main()
