from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.external_datasets import load_external_dataset, save_external_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize external datasets to parquet (parquet input only)")
    parser.add_argument("--dataset", required=True, choices=["kickstarter", "kiva", "gofundme"])
    parser.add_argument("--input_path", required=True, type=Path)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    args = parser.parse_args()

    out_path = args.output_path or Path("data/processed/external") / f"{args.dataset}.parquet"
    df = load_external_dataset(args.input_path, dataset=args.dataset, limit_rows=args.limit_rows)
    save_external_parquet(df, out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
