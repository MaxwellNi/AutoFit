#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single matrix entry.")
    parser.add_argument("--matrix_json", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix_json).read_text(encoding="utf-8"))
    entries = matrix.get("entries", [])
    if args.index < 0 or args.index >= len(entries):
        raise SystemExit(f"index out of range: {args.index}")
    entry = entries[args.index]
    cmd = entry["command"]

    if args.print_only:
        print(cmd)
        return

    print(f"running index={args.index} name={entry.get('name')}")
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    main()
