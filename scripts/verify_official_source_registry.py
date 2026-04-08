#!/usr/bin/env python3
"""Validate and summarize the machine-readable official-source registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.official_sources import (  # noqa: E402
    build_official_source_rows,
    load_official_source_registry,
    summarize_official_source_registry,
    validate_official_source_registry,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--require-no-pending", action="store_true")
    return parser.parse_args()


def _markdown_lines(summary: dict, rows: List[dict], errors: List[str]) -> List[str]:
    lines = [
        "# Official Source Registry Audit",
        "",
        f"- registry: `{summary['registry_name']}` @ `{summary['registry_version']}`",
        f"- entries: `{summary['entry_count']}`",
        f"- verified: `{summary['verified_count']}`",
        f"- pending recheck: `{summary['pending_recheck_count']}`",
        f"- validation errors: `{len(errors)}`",
        "",
        "## Signals",
        "",
        "| signal | status | venue | repo/pdf/page | conservative wording | gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        urls = [value for value in (row["official_repo"], row["official_pdf"], row["official_page"]) if value]
        lines.append(
            "| `{signal}` | `{status}` | `{venue}` | `{urls}` | `{wording}` | {gap} |".format(
                signal=row["signal"],
                status=row["status"],
                venue=row["venue"],
                urls="; ".join(urls) or "-",
                wording="conservative" if row["conservative_wording"] else "open",
                gap=row["verification_gap"] or "-",
            )
        )
    if errors:
        lines.extend(["", "## Validation Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    return lines


def main() -> int:
    args = _parse_args()
    registry = load_official_source_registry(args.config)
    errors = validate_official_source_registry(registry)
    rows = build_official_source_rows(registry)
    summary = summarize_official_source_registry(registry)
    payload = {
        "summary": summary,
        "rows": rows,
        "errors": errors,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            "\n".join(_markdown_lines(summary, rows, errors)).rstrip() + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if errors:
        return 1
    if args.require_no_pending and summary["pending_recheck_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())