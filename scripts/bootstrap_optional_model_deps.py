#!/usr/bin/env python3
"""Install optional model dependencies into a user-local shared path."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.models.optional_runtime import get_optional_vendor_dir

PACKAGE_GROUPS = {
    # Install the minimal runtime set explicitly and rely on the canonical
    # insider env for numpy/pandas/matplotlib/tqdm. This avoids shadowing the
    # benchmark environment with a second scientific stack in the vendor dir.
    "prophet": ["prophet", "cmdstanpy", "holidays", "stanio", "importlib_resources"],
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group",
        action="append",
        choices=sorted(PACKAGE_GROUPS),
        default=[],
        help="Named dependency group to install. Can be passed multiple times.",
    )
    ap.add_argument(
        "--package",
        action="append",
        default=[],
        help="Additional raw package specifiers to install.",
    )
    ap.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional override for the vendor site-packages directory.",
    )
    ap.add_argument(
        "--with-deps",
        action="store_true",
        help="Allow pip to resolve transitive dependencies. Default is off to avoid shadowing core env packages.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    packages: list[str] = []
    for group in args.group:
        packages.extend(PACKAGE_GROUPS[group])
    packages.extend(args.package)
    if not packages:
        raise SystemExit("No packages requested. Use --group prophet and/or --package ...")

    target = (args.target or get_optional_vendor_dir()).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target),
    ]
    if not args.with_deps:
        cmd.append("--no-deps")
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(packages)

    print(f"[bootstrap_optional_model_deps] target={target}")
    print(f"[bootstrap_optional_model_deps] cmd={' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
