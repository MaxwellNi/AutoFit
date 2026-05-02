#!/usr/bin/env python3
"""Validate that the NeurIPS draft does not expose internal repo surfaces.

The paper can refer to an anonymized audit ledger, but the public-facing LaTeX
should not contain raw script paths, runs paths, internal audit filenames, or
local file extensions that only make sense inside this repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_ROOT = ROOT / "docs" / "paper_neurips26"

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("raw scripts path", re.compile(r"scripts/")),
    ("raw runs path", re.compile(r"runs/")),
    ("round-internal r14 identifier", re.compile(r"\br14[_-]", re.IGNORECASE)),
    ("raw json artifact filename", re.compile(r"\.json\b", re.IGNORECASE)),
    ("raw parquet artifact filename", re.compile(r"\.parquet\b", re.IGNORECASE)),
    ("method-contract implementation name", re.compile(r"event_state_method_contract", re.IGNORECASE)),
    ("timestamped internal artifact id", re.compile(r"20260[0-9]{7}")),
]


def iter_tex_files(root: Path):
    for path in sorted(root.rglob("*.tex")):
        if ".git" not in path.parts:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    args = parser.parse_args()

    failures: list[tuple[Path, int, str, str]] = []
    for path in iter_tex_files(args.paper_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    failures.append((path, line_no, label, line.strip()))

    if failures:
        for path, line_no, label, line in failures:
            rel = path.relative_to(ROOT)
            print(f"{rel}:{line_no}: {label}: {line}")
        return 1

    print(f"OK: public-surface lint passed for {args.paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
