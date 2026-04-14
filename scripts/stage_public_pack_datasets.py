#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.public_pack import (  # noqa: E402
    SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES,
    load_public_pack_registry,
    resolve_public_pack_downloads,
    stage_public_pack_downloads,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage supported public-pack datasets into the configured raw roots."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-supported", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list_supported:
        print(json.dumps({"supported_families": list(SUPPORTED_PUBLIC_PACK_DOWNLOAD_FAMILIES)}, indent=2))
        return 0

    registry = load_public_pack_registry(args.config)
    downloads = resolve_public_pack_downloads(
        registry,
        requested_families=args.family,
    )

    if args.dry_run:
        rows = [download.to_dict() for download in downloads]
    else:
        rows = stage_public_pack_downloads(
            downloads,
            overwrite=args.overwrite,
            timeout=args.timeout,
        )

    summary = {
        "families": sorted({row["family"] for row in rows}),
        "variants": sorted({row["variant"] for row in rows}),
        "download_count": len(rows),
        "statuses": sorted({row.get("status", "planned") for row in rows}),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())