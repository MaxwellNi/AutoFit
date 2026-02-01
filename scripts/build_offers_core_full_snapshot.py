#!/usr/bin/env python
"""
Build full-scale offers_core_snapshot from raw offers Delta.
Wrapper: calls build_offers_core_daily_full with --output_base offers_core_snapshot.
Snapshot = one row per (entity_id, snapshot_ts_day) from raw scan.
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main() -> None:
    import argparse
    from scripts.build_offers_core_daily_full import main as _real_main
    # Inject --output_base offers_core_snapshot into sys.argv
    if "--output_base" not in " ".join(sys.argv):
        sys.argv.extend(["--output_base", "offers_core_snapshot"])
    _real_main()


if __name__ == "__main__":
    main()
