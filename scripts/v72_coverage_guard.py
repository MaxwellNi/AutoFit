#!/usr/bin/env python3
"""Build AutoFitV72 strict-missing key manifest for completion control."""

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBTASKS = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "subtasks_by_target_full.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "missing_key_manifest.csv"
DEFAULT_BENCH_GLOB = "block3_20260203_225620*"


def _priority(task: str, ablation: str) -> Tuple[int, str]:
    if task == "task1_outcome" and ablation in {"core_text", "full", "core_edgar"}:
        return 1, "P1_task1_core_text_full_core_edgar"
    if task == "task2_forecast" and ablation in {"core_only", "core_text", "full"}:
        return 2, "P2_task2_core_only_text_full"
    if task == "task3_risk_adjust" and ablation in {"core_only", "core_edgar", "full"}:
        return 3, "P3_task3_all_ablations"
    return 4, "P4_other"


def _load_expected(subtasks_csv: Path) -> List[Tuple[str, str, str, int]]:
    rows: List[Tuple[str, str, str, int]] = []
    with subtasks_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    str(row["task"]),
                    str(row["ablation"]),
                    str(row["target"]),
                    int(float(row["horizon"])),
                )
            )
    return sorted(set(rows))


def _discover_metrics(bench_glob: str) -> Iterable[Path]:
    pattern = str(ROOT / "runs" / "benchmarks" / bench_glob / "**" / "metrics.json")
    for p in glob.glob(pattern, recursive=True):
        yield Path(p)


def _load_v72_strict_keys(bench_glob: str) -> Set[Tuple[str, str, str, int]]:
    keys: Set[Tuple[str, str, str, int]] = set()
    for path in _discover_metrics(bench_glob):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("model_name") != "AutoFitV72":
                continue
            try:
                if row.get("fairness_pass") is True and float(row.get("prediction_coverage_ratio", 0.0)) >= 0.98:
                    keys.add(
                        (
                            str(row["task"]),
                            str(row["ablation"]),
                            str(row["target"]),
                            int(row["horizon"]),
                        )
                    )
            except Exception:
                continue
    return keys


def build_missing_manifest(
    expected: Sequence[Tuple[str, str, str, int]],
    v72_keys: Set[Tuple[str, str, str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task, ablation, target, horizon in expected:
        if (task, ablation, target, horizon) in v72_keys:
            continue
        priority_rank, priority_group = _priority(task, ablation)
        rows.append(
            {
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "priority_rank": priority_rank,
                "priority_group": priority_group,
                "reason": "v72_strict_missing",
            }
        )
    rows.sort(key=lambda r: (int(r["priority_rank"]), str(r["task"]), str(r["ablation"]), str(r["target"]), int(r["horizon"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AutoFitV72 strict missing-key manifest.")
    parser.add_argument("--subtasks-csv", type=Path, default=DEFAULT_SUBTASKS)
    parser.add_argument("--bench-glob", type=str, default=DEFAULT_BENCH_GLOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    expected = _load_expected(args.subtasks_csv.resolve())
    v72_keys = _load_v72_strict_keys(args.bench_glob)
    missing = build_missing_manifest(expected=expected, v72_keys=v72_keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "ablation",
                "target",
                "horizon",
                "priority_rank",
                "priority_group",
                "reason",
            ],
        )
        writer.writeheader()
        for row in missing:
            writer.writerow(row)

    coverage_ratio = float(len(expected) - len(missing)) / float(len(expected)) if expected else 0.0
    summary = {
        "expected_keys": len(expected),
        "v72_strict_keys": len(v72_keys),
        "missing_keys": len(missing),
        "coverage_ratio": coverage_ratio,
        "output": str(args.output),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
