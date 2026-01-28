#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight status generator.")
    parser.add_argument("--offers_core", required=True)
    parser.add_argument("--offers_manifest", required=True)
    parser.add_argument("--use_edgar", type=int, default=0)
    parser.add_argument("--edgar_dir", default=None)
    parser.add_argument("--latest_edgar_txt", default="runs/edgar_feature_store/latest.txt")
    args = parser.parse_args()

    print(f"git_head={_run(['git', 'rev-parse', 'HEAD'])}")
    print(f"git_status_lines={_run(['bash', '-lc', 'git status --porcelain | wc -l'])}")
    print(_run(["bash", "-lc", "which python"]))
    print(_run(["python", "-V"]))

    offers_core = Path(args.offers_core)
    if offers_core.exists():
        st = offers_core.stat()
        print(f"offers_core={offers_core}")
        print(f"offers_core_size={st.st_size}")
    else:
        raise SystemExit("FATAL: offers_core missing")

    offers_manifest = Path(args.offers_manifest)
    if offers_manifest.exists():
        print("offers_core_manifest=OK")
    else:
        raise SystemExit("FATAL: offers_manifest missing")

    if int(args.use_edgar) != 1:
        print("use_edgar=0")
        return

    edgar_dir = Path(args.edgar_dir) if args.edgar_dir else None
    if edgar_dir is None:
        latest = Path(args.latest_edgar_txt)
        if not latest.exists():
            raise SystemExit("FATAL: latest edgar pointer missing")
        edgar_dir = Path("runs/edgar_feature_store") / latest.read_text().strip() / "edgar_features"
    print(f"edgar_dir={edgar_dir}")

    if not edgar_dir.exists():
        raise SystemExit("FATAL: edgar_features_dir missing")

    subprocess.check_call(
        ["python", "scripts/preflight_edgar_check.py", "--edgar_dir", str(edgar_dir)]
    )


if __name__ == "__main__":
    main()
