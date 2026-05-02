#!/usr/bin/env python3
"""Validate the public-facing surface of the NeurIPS draft.

The paper can refer to an anonymized audit ledger, but the public-facing LaTeX
should not contain raw script paths, runs paths, internal audit filenames, or
local file extensions that only make sense inside this repository.

The same guard also catches unresolved cross-reference placeholders before a
PDF is circulated. A compiled PDF with ``Appendix ??'' or ``Eqs. (??)--(??)''
is an audit failure, not a cosmetic issue.
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
    ("literal unresolved placeholder", re.compile(r"\?\?")),
]

LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}")
BAD_LOG_RE = re.compile(
    r"Reference `[^']+' .* undefined|Citation `[^']+' .* undefined|"
    r"There were undefined references|There were undefined citations|"
    r"Label `[^']+' multiply defined|There were multiply-defined labels"
)


def iter_tex_files(root: Path):
    for path in sorted(root.rglob("*.tex")):
        if ".git" not in path.parts:
            yield path


def split_ref_targets(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional compiled LaTeX log to scan for undefined references/citations.",
    )
    args = parser.parse_args()

    failures: list[tuple[Path, int, str, str]] = []
    labels: dict[str, list[tuple[Path, int]]] = {}
    refs: list[tuple[Path, int, str]] = []
    for path in iter_tex_files(args.paper_root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    failures.append((path, line_no, label, line.strip()))
            for label_name in LABEL_RE.findall(line):
                labels.setdefault(label_name, []).append((path, line_no))
            for ref_group in REF_RE.findall(line):
                for ref_name in split_ref_targets(ref_group):
                    refs.append((path, line_no, ref_name))

    for label_name, occurrences in sorted(labels.items()):
        if len(occurrences) > 1:
            for path, line_no in occurrences:
                failures.append((path, line_no, "duplicate LaTeX label", label_name))

    for path, line_no, ref_name in refs:
        if ref_name not in labels:
            failures.append((path, line_no, "missing LaTeX label target", ref_name))

    log_path = args.log_path or (args.paper_root / "main.log")
    if log_path.exists():
        for line_no, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if BAD_LOG_RE.search(line):
                failures.append((log_path, line_no, "LaTeX log unresolved reference/citation", line.strip()))

    if failures:
        for path, line_no, label, line in failures:
            rel = path.relative_to(ROOT)
            print(f"{rel}:{line_no}: {label}: {line}")
        return 1

    print(f"OK: NeurIPS public-surface and reference lint passed for {args.paper_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
