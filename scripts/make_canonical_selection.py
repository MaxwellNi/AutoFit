#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical entity selection.")
    parser.add_argument("--offers_core", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.offers_core, columns=["entity_id"])
    entities = df["entity_id"].dropna().astype(str).unique().tolist()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(entities)
    count = min(args.count, len(entities))
    selected = sorted(entities[:count])

    selected_path = out_dir / "sampled_entities.json"
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    selection_hash = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    (out_dir / "selection_hash.txt").write_text(selection_hash + "\n", encoding="utf-8")

    manifest = {
        "offers_core": args.offers_core,
        "offers_core_sha256": _sha256(Path(args.offers_core)),
        "seed": args.seed,
        "count": count,
        "selection_hash": selection_hash,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()
