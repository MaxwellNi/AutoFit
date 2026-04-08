#!/usr/bin/env python3
"""Expand the repo-tracked public-pack scaffold into a runner-ready plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.public_pack.registry import (  # noqa: E402
    ensure_public_pack_directories,
    expand_public_pack_cells,
    load_public_pack_registry,
    summarize_public_pack_selection,
    validate_public_pack_roots,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pack", default="")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--verification-tier", default="")
    parser.add_argument("--enabled-only", action="store_true")
    parser.add_argument("--ensure-processed-dirs", action="store_true")
    parser.add_argument("--ensure-cache-dirs", action="store_true")
    parser.add_argument("--fail-on-missing-raw", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _markdown_lines(
    selection_summary: dict,
    root_rows: List[dict],
    cells: List[dict],
) -> List[str]:
    lines = [
        "# Public Pack Run Plan",
        "",
        f"- registry: `{selection_summary['registry_name']}` @ `{selection_summary['registry_version']}`",
        f"- families: `{selection_summary['family_count']}`",
        f"- variants: `{selection_summary['variant_count']}`",
        f"- expanded cells: `{selection_summary['cell_count']}`",
        f"- missing raw families: `{', '.join(selection_summary['missing_raw_families']) or '-'}`",
        "",
        "## Family Roots",
        "",
        "| pack | family | enabled | verification tier | raw | processed | cache |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in root_rows:
        lines.append(
            "| `{pack}` | `{family}` | {enabled} | `{verification_tier}` | `{raw_exists}` | `{processed_exists}` | `{cache_exists}` |".format(
                pack=row["pack"],
                family=row["family"],
                enabled="yes" if row["enabled"] else "no",
                verification_tier=row["verification_tier"],
                raw_exists="yes" if row["raw_exists"] else "no",
                processed_exists="yes" if row["processed_exists"] else "no",
                cache_exists="yes" if row["cache_exists"] else "no",
            )
        )

    lines.extend([
        "",
        "## Expanded Cells",
        "",
        "| pack | family | variant | ctx | pred | task type | signals |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ])
    for cell in cells:
        lines.append(
            "| `{pack}` | `{family}` | `{variant}` | {ctx} | {pred} | `{task_type}` | `{signals}` |".format(
                pack=cell["pack"],
                family=cell["family"],
                variant=cell["variant"],
                ctx=cell["context_length"] if cell["context_length"] is not None else "-",
                pred=cell["prediction_length"] if cell["prediction_length"] is not None else "-",
                task_type=cell["task_type"],
                signals=",".join(cell["official_source_signals"]) or "-",
            )
        )
    return lines


def main() -> int:
    args = _parse_args()
    registry = load_public_pack_registry(args.config)
    if args.ensure_processed_dirs or args.ensure_cache_dirs:
        root_rows = ensure_public_pack_directories(
            registry,
            pack=args.pack,
            requested_families=args.family,
            enabled_only=args.enabled_only,
            verification_tier=args.verification_tier,
            ensure_processed=args.ensure_processed_dirs,
            ensure_cache=args.ensure_cache_dirs,
        )
    else:
        root_rows = validate_public_pack_roots(
            registry,
            pack=args.pack,
            requested_families=args.family,
            enabled_only=args.enabled_only,
            verification_tier=args.verification_tier,
        )

    cells = [
        cell.to_dict()
        for cell in expand_public_pack_cells(
            registry,
            pack=args.pack,
            requested_families=args.family,
            enabled_only=args.enabled_only,
            verification_tier=args.verification_tier,
        )
    ]
    selection_summary = summarize_public_pack_selection(registry, cells, root_rows)
    payload = {
        "selection": {
            "pack": args.pack or None,
            "families": args.family,
            "verification_tier": args.verification_tier or None,
            "enabled_only": bool(args.enabled_only),
        },
        "summary": selection_summary,
        "root_rows": root_rows,
        "cells": cells,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            "\n".join(_markdown_lines(selection_summary, root_rows, cells)).rstrip() + "\n",
            encoding="utf-8",
        )

    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))

    if args.fail_on_missing_raw and selection_summary["missing_raw_families"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())