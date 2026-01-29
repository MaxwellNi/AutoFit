#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_selection_hash(selection_hash_path: Path) -> str:
    return selection_hash_path.read_text(encoding="utf-8").strip()


def _edgar_col_hash(edgar_dir: Path) -> str:
    files = [p for p in edgar_dir.rglob("*.parquet") if p.is_file()]
    if not files:
        raise SystemExit("FATAL: edgar_dir empty")
    rng = np.random.RandomState(42)
    sample_file = files[int(rng.randint(0, len(files)))]
    table = pq.read_table(sample_file, memory_map=True)
    if table.num_rows > 200:
        table = table.slice(0, 200)
    cols = list(table.to_pandas().columns)
    return hashlib.sha256(",".join(sorted(cols)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify key artifacts before running jobs.")
    parser.add_argument("--artifacts_json", required=True)
    parser.add_argument("--require_edgar", action="store_true")
    args = parser.parse_args()

    artifacts = json.loads(Path(args.artifacts_json).read_text(encoding="utf-8"))
    expected_head = artifacts["git_head"]
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if current_head != expected_head:
        raise SystemExit(f"FATAL: git head mismatch {current_head} != {expected_head}")

    offers_core = Path(artifacts["offers_core"])
    offers_sha = _sha256(offers_core)
    if offers_sha != artifacts["offers_core_sha256"]:
        raise SystemExit("FATAL: offers_core sha256 mismatch")

    selection_hash = _read_selection_hash(Path(artifacts["selection_hash_path"]))
    if selection_hash != artifacts["selection_hash"]:
        raise SystemExit("FATAL: selection_hash mismatch")

    if args.require_edgar:
        edgar_dir = Path(artifacts["edgar_dir"])
        files = [p for p in edgar_dir.rglob("*") if p.is_file()]
        if len(files) == 0:
            raise SystemExit("FATAL: edgar_dir empty")
        expected_col_hash = artifacts.get("edgar_col_hash")
        if expected_col_hash:
            current_col_hash = _edgar_col_hash(edgar_dir)
            if current_col_hash != expected_col_hash:
                raise SystemExit(
                    f"FATAL: edgar_col_hash mismatch {current_col_hash} != {expected_col_hash}"
                )

    print("verify_artifacts: OK")


if __name__ == "__main__":
    main()
