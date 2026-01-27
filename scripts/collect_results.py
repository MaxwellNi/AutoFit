from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil


def collect_results(src_runs: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "paper_tables"
    explain_dir = out_dir / "explain"
    config_dir = out_dir / "configs"
    table_dir.mkdir(exist_ok=True)
    explain_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    for run_path in src_runs.glob("**/"):
        if not run_path.is_dir():
            continue
        paper_tables = run_path / "paper_tables"
        if paper_tables.exists():
            for f in paper_tables.glob("*.parquet"):
                shutil.copy2(f, table_dir / f"{run_path.name}_{f.name}")
        explain = run_path / "explain"
        if explain.exists():
            for f in explain.glob("*"):
                if f.is_file():
                    shutil.copy2(f, explain_dir / f"{run_path.name}_{f.name}")
        best_cfg = run_path / "best_config.yaml"
        if best_cfg.exists():
            shutil.copy2(best_cfg, config_dir / f"{run_path.name}_best_config.yaml")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run artifacts into one directory")
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or Path("runs/collected") / datetime.now().strftime("%Y%m%d_%H%M%S")
    collect_results(args.runs_dir, out_dir)
    print(f"Collected results into {out_dir}")


if __name__ == "__main__":
    main()
