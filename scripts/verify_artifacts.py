#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_selection_hash(selection_hash_path: Path) -> str:
    return selection_hash_path.read_text(encoding="utf-8").strip()


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

    print("verify_artifacts: OK")


if __name__ == "__main__":
    main()
