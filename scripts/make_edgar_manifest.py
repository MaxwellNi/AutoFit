#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _du_h(path: Path) -> str:
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.strip()
    except Exception:
        return ""


def _parse_delta_versions(log_text: str) -> Dict[str, Optional[int]]:
    edgar_version = None
    offers_version = None
    edgar_match = re.search(r"delta_dataset=edgar_accessions version=(\\d+)", log_text)
    offers_match = re.search(r"delta_dataset=offers_snapshots version=(\\d+)", log_text)
    if edgar_match:
        edgar_version = int(edgar_match.group(1))
    if offers_match:
        offers_version = int(offers_match.group(1))
    return {"edgar_accessions_version": edgar_version, "offers_snapshots_version": offers_version}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MANIFEST for EDGAR feature store.")
    parser.add_argument("--edgar_dir", required=True)
    parser.add_argument("--build_log", required=True)
    parser.add_argument("--coverage_report", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cmdline", required=True)
    args = parser.parse_args()

    edgar_dir = Path(args.edgar_dir)
    build_log = Path(args.build_log)
    coverage_report = Path(args.coverage_report)
    output_path = Path(args.output_path)

    if not edgar_dir.exists():
        raise SystemExit(f"edgar_dir missing: {edgar_dir}")
    if not build_log.exists():
        raise SystemExit(f"build_log missing: {build_log}")
    if not coverage_report.exists():
        raise SystemExit(f"coverage_report missing: {coverage_report}")

    files = [p for p in edgar_dir.rglob("*") if p.is_file()]
    file_count = len(files)
    if file_count == 0:
        raise SystemExit("edgar_dir is empty")

    log_text = build_log.read_text(encoding="utf-8")
    delta_versions = _parse_delta_versions(log_text)

    coverage = json.loads(coverage_report.read_text(encoding="utf-8"))
    snapshot_time_col = coverage.get("snapshot_time_col")

    manifest = {
        "edgar_dir": str(edgar_dir),
        "file_count": file_count,
        "du_sh": _du_h(edgar_dir),
        "delta_versions": delta_versions,
        "snapshot_time_col": snapshot_time_col,
        "cmdline": args.cmdline,
        "sha256": {
            "build_edgar_features": _sha256(Path("scripts/build_edgar_features.py")),
            "edgar_feature_store": _sha256(Path("src/narrative/data_preprocessing/edgar_feature_store.py"))
            if Path("src/narrative/data_preprocessing/edgar_feature_store.py").exists()
            else None,
            "build_log": _sha256(build_log),
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hostname": socket.gethostname(),
        "python_version": subprocess.check_output(["python", "-V"], text=True).strip(),
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
