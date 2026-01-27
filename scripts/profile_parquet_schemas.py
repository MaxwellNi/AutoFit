from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# Add repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from narrative.data_preprocessing.schema_profiler import profile_parquet_dataset, profiles_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile parquet schemas for offers + EDGAR")
    parser.add_argument("--offers_path", type=Path, default=Path("data/raw/offers"))
    parser.add_argument("--edgar_path", type=Path, default=Path("data/raw/edgar/accessions"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=50_000)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--max_profile_cols", type=int, default=50)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path("runs/schema_profile") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        profile_parquet_dataset(
            args.offers_path,
            name="offers_snapshots",
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            max_profile_cols=args.max_profile_cols,
        ),
        profile_parquet_dataset(
            args.edgar_path,
            name="edgar_accessions",
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            max_profile_cols=args.max_profile_cols,
        ),
    ]

    payload = {p.name: p.to_dict() for p in profiles}
    (out_dir / "schema_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    profiles_to_frame(profiles).to_parquet(out_dir / "schema_profile.parquet", index=False)

    print(f"✓ Saved schema profiles to: {out_dir}")


if __name__ == "__main__":
    main()
