#!/usr/bin/env python3
"""Build key-level V7.2 completion jobs from missing-key manifest."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MISSING_MANIFEST = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "missing_key_manifest.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_key_job_manifest.csv"
DEFAULT_OUTPUT_ROOT = "runs/benchmarks/block3_20260203_225620_phase7_v72_completion"
DEFAULT_MEMORY_PLAN_JSON = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_memory_plan.json"
DEFAULT_SUMMARY = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_key_job_manifest_summary.json"


@dataclass(frozen=True)
class ResourceClass:
    name: str
    partition: str
    qos: str
    mem: str
    cpus: int
    time_limit: str


RESOURCE_PROFILES: Dict[str, ResourceClass] = {
    "XL": ResourceClass(
        name="XL",
        partition="bigmem",
        qos="iris-bigmem-long",
        mem="512G",
        cpus=64,
        time_limit="3-00:00:00",
    ),
    "L": ResourceClass(
        name="L",
        partition="batch",
        qos="iris-batch-long",
        mem="160G",
        cpus=32,
        time_limit="2-00:00:00",
    ),
}


def _task_short(task: str) -> str:
    return {
        "task1_outcome": "t1",
        "task2_forecast": "t2",
        "task3_risk_adjust": "t3",
    }.get(task, "tx")


def _ablation_short(ablation: str) -> str:
    return {
        "core_only": "co",
        "core_text": "ct",
        "core_edgar": "ce",
        "full": "fu",
    }.get(ablation, "xx")


def _target_short(target: str) -> str:
    return {
        "funding_raised_usd": "fru",
        "investors_count": "ic",
        "is_funded": "if",
    }.get(target, "tg")


def _row_key(row: Dict[str, object]) -> str:
    return f"{row['task']}|{row['ablation']}|{row['target']}|h{int(row['horizon'])}"


def _load_csv_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "task": str(row["task"]),
                    "ablation": str(row["ablation"]),
                    "target": str(row["target"]),
                    "horizon": int(float(row["horizon"])),
                    "priority_rank": int(float(row.get("priority_rank", 99))),
                    "priority_group": str(row.get("priority_group", "P9_other")),
                    "reason": str(row.get("reason", "v72_strict_missing")),
                }
            )
    return rows


def _load_memory_plan(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    out: Dict[str, Dict[str, object]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key", "")).strip()
            if key:
                out[key] = row
    return out


def _default_resource_class(task: str, ablation: str) -> str:
    # Immediate hard cases from observed failures:
    #   - task3 all ablations: syntax mismatch fixed but keep on bigmem profile
    #   - OOM lineage: task1/core_text, task2/core_only, task2/core_text
    if task == "task3_risk_adjust":
        return "XL"
    if (task, ablation) in {
        ("task1_outcome", "core_text"),
        ("task2_forecast", "core_only"),
        ("task2_forecast", "core_text"),
    }:
        return "XL"
    return "L"


def _choose_resource_class(row: Dict[str, object], mem_plan: Dict[str, Dict[str, object]]) -> Tuple[str, Optional[float]]:
    key = _row_key(row)
    if key in mem_plan:
        plan_row = mem_plan[key]
        cls = str(plan_row.get("memory_class", "")).strip().upper() or "L"
        if cls not in RESOURCE_PROFILES:
            cls = "L"
        peak = plan_row.get("predicted_peak_rss_gb")
        try:
            peak_val = float(peak) if peak is not None else None
        except Exception:
            peak_val = None
        return cls, peak_val
    return _default_resource_class(str(row["task"]), str(row["ablation"])), None


def _job_name(prefix: str, row: Dict[str, object]) -> str:
    return (
        f"{prefix}_"
        f"{_task_short(str(row['task']))}_"
        f"{_ablation_short(str(row['ablation']))}_"
        f"{_target_short(str(row['target']))}_"
        f"h{int(row['horizon'])}"
    )


def build_manifest_rows(
    missing_rows: Iterable[Dict[str, object]],
    mem_plan: Dict[str, Dict[str, object]],
    output_root: str,
    job_prefix: str,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in sorted(
        missing_rows,
        key=lambda r: (
            int(r.get("priority_rank", 99)),
            str(r.get("task", "")),
            str(r.get("ablation", "")),
            str(r.get("target", "")),
            int(r.get("horizon", 0)),
        ),
    ):
        memory_class, predicted_peak = _choose_resource_class(row, mem_plan)
        profile = RESOURCE_PROFILES[memory_class]
        task = str(row["task"])
        ablation = str(row["ablation"])
        target = str(row["target"])
        horizon = int(row["horizon"])
        outdir = (
            f"{output_root}/{task}/autofit/{ablation}/{target}/h{horizon}"
        )
        rows.append(
            {
                "job_name": _job_name(job_prefix, row),
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "priority_rank": int(row.get("priority_rank", 99)),
                "priority_group": str(row.get("priority_group", "P9_other")),
                "resource_class": profile.name,
                "partition": profile.partition,
                "qos": profile.qos,
                "mem": profile.mem,
                "cpus": profile.cpus,
                "time_limit": profile.time_limit,
                "seed": seed,
                "output_dir": outdir,
                "reason": str(row.get("reason", "v72_strict_missing")),
                "predicted_peak_rss_gb": predicted_peak,
                "evidence_path": "docs/benchmarks/block3_truth_pack/missing_key_manifest.csv",
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "job_name",
        "task",
        "ablation",
        "target",
        "horizon",
        "priority_rank",
        "priority_group",
        "resource_class",
        "partition",
        "qos",
        "mem",
        "cpus",
        "time_limit",
        "seed",
        "output_dir",
        "reason",
        "predicted_peak_rss_gb",
        "evidence_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build key-level V7.2 job manifest.")
    parser.add_argument("--missing-manifest", type=Path, default=DEFAULT_MISSING_MANIFEST)
    parser.add_argument("--memory-plan-json", type=Path, default=DEFAULT_MEMORY_PLAN_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-prefix", type=str, default="p7v72k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    missing_manifest = args.missing_manifest.resolve()
    if not missing_manifest.exists():
        raise SystemExit(f"Missing input manifest: {missing_manifest}")

    missing_rows = _load_csv_rows(missing_manifest)
    mem_plan = _load_memory_plan(args.memory_plan_json.resolve() if args.memory_plan_json else None)
    rows = build_manifest_rows(
        missing_rows=missing_rows,
        mem_plan=mem_plan,
        output_root=args.output_root,
        job_prefix=args.job_prefix,
        seed=args.seed,
    )
    _write_csv(args.output.resolve(), rows)

    summary = {
        "total_jobs": len(rows),
        "resource_class_counts": {
            cls: sum(1 for r in rows if r["resource_class"] == cls)
            for cls in RESOURCE_PROFILES
        },
        "input_missing_manifest": str(missing_manifest),
        "input_memory_plan_json": str(args.memory_plan_json),
        "output_manifest": str(args.output.resolve()),
    }
    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
