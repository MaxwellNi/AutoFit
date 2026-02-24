#!/usr/bin/env python3
"""Estimate V7.2 per-key memory requirements and recommend resource classes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MISSING_MANIFEST = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "missing_key_manifest.csv"
DEFAULT_OUTPUT_JSON = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_memory_plan.json"
DEFAULT_OUTPUT_CSV = ROOT / "docs" / "benchmarks" / "block3_truth_pack" / "v72_memory_plan.csv"


@dataclass
class MissingKey:
    task: str
    ablation: str
    target: str
    horizon: int
    priority_rank: int
    priority_group: str

    @property
    def key(self) -> str:
        return f"{self.task}|{self.ablation}|{self.target}|h{self.horizon}"


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


def _safe_run(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return ""
        return proc.stdout
    except Exception:
        return ""


def _parse_missing(path: Path) -> List[MissingKey]:
    rows: List[MissingKey] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                MissingKey(
                    task=str(row["task"]),
                    ablation=str(row["ablation"]),
                    target=str(row["target"]),
                    horizon=int(float(row["horizon"])),
                    priority_rank=int(float(row.get("priority_rank", 99))),
                    priority_group=str(row.get("priority_group", "P9_other")),
                )
            )
    return rows


def _heuristic_peak_gb(task: str, ablation: str, target: str) -> float:
    # Conservative baseline with known OOM lineage up-weighting.
    base = 110.0
    if target == "investors_count":
        base += 20.0
    if task == "task3_risk_adjust":
        base += 180.0
    if (task, ablation) in {
        ("task1_outcome", "core_text"),
        ("task2_forecast", "core_only"),
        ("task2_forecast", "core_text"),
    }:
        base += 80.0
    if ablation == "full":
        base += 25.0
    if ablation == "core_edgar":
        base += 15.0
    return base


def _parse_req_mem_to_gb(req_mem: str) -> Optional[float]:
    raw = req_mem.strip()
    if not raw:
        return None
    m = re.match(r"^([0-9.]+)([KMGTP])(?:n|c)?$", raw, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "K": 1.0 / (1024 * 1024),
        "M": 1.0 / 1024,
        "G": 1.0,
        "T": 1024.0,
        "P": 1024.0 * 1024.0,
    }[unit]
    return val * mult


def _parse_max_rss_to_gb(max_rss: str) -> Optional[float]:
    return _parse_req_mem_to_gb(max_rss)


def _parse_sacct_rows(sacct_text: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in sacct_text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) != 5:
            continue
        job_name, state, max_rss, elapsed, req_mem = parts
        rows.append(
            {
                "job_name": job_name,
                "state": state,
                "max_rss_gb": _parse_max_rss_to_gb(max_rss),
                "req_mem_gb": _parse_req_mem_to_gb(req_mem),
                "elapsed": elapsed,
            }
        )
    return rows


def _match_job_to_key(job_name: str, key: MissingKey) -> bool:
    ts = _task_short(key.task)
    abl = _ablation_short(key.ablation)
    h = f"h{key.horizon}"
    tg = {
        "funding_raised_usd": "fru",
        "investors_count": "ic",
        "is_funded": "if",
    }.get(key.target, "")
    name = job_name.lower()
    if f"_{ts}_{abl}_" in name and h in name:
        if tg:
            return f"_{tg}_" in name
        return True
    # Legacy shard-level fallback
    return f"_{ts}_{abl}" in name


def _estimate_for_key(
    key: MissingKey,
    sacct_rows: List[Dict[str, object]],
) -> Tuple[float, float, float, str, List[str]]:
    heuristic = _heuristic_peak_gb(key.task, key.ablation, key.target)
    observed: List[float] = []
    sources: List[str] = []
    for row in sacct_rows:
        if _match_job_to_key(str(row.get("job_name", "")), key):
            max_rss = row.get("max_rss_gb")
            if isinstance(max_rss, (int, float)) and max_rss > 0:
                observed.append(float(max_rss))
                sources.append(f"sacct:{row.get('job_name')}")

    if observed:
        p95_like = max(observed) * 1.25
        pred = max(heuristic, p95_like)
        low = max(1.0, min(observed) * 0.9)
        high = max(pred, max(observed) * 1.4)
        reason = "observed_plus_guardband"
    else:
        pred = heuristic
        low = heuristic * 0.75
        high = heuristic * 1.30
        reason = "heuristic_only"
    return pred, low, high, reason, sources


def _memory_class_from_predicted(predicted_peak_gb: float, requested_l_gb: float = 160.0) -> str:
    # Admission guard: predicted > requested*0.8 triggers upgrade to XL.
    if predicted_peak_gb > requested_l_gb * 0.8:
        return "XL"
    return "L"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate V7.2 memory needs for missing keys.")
    parser.add_argument("--missing-manifest", type=Path, default=DEFAULT_MISSING_MANIFEST)
    parser.add_argument("--slurm-since", type=str, default="2026-02-20")
    parser.add_argument("--capture-sacct", action="store_true", default=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    missing_manifest = args.missing_manifest.resolve()
    if not missing_manifest.exists():
        raise SystemExit(f"Missing input manifest: {missing_manifest}")
    keys = _parse_missing(missing_manifest)

    sacct_rows: List[Dict[str, object]] = []
    if args.capture_sacct:
        user = os.environ.get("USER", "")
        sacct_text = _safe_run(
            [
                "sacct",
                "-u",
                user,
                "-S",
                args.slurm_since,
                "-n",
                "-X",
                "-P",
                "-o",
                "JobName,State,MaxRSS,Elapsed,ReqMem",
            ]
        )
        sacct_rows = _parse_sacct_rows(sacct_text)

    out_rows: List[Dict[str, object]] = []
    for key in keys:
        pred, low, high, reason, evidence = _estimate_for_key(key, sacct_rows)
        mem_class = _memory_class_from_predicted(predicted_peak_gb=pred)
        out_rows.append(
            {
                "key": key.key,
                "task": key.task,
                "ablation": key.ablation,
                "target": key.target,
                "horizon": key.horizon,
                "priority_rank": key.priority_rank,
                "priority_group": key.priority_group,
                "predicted_peak_rss_gb": round(pred, 3),
                "confidence_low_gb": round(low, 3),
                "confidence_high_gb": round(high, 3),
                "memory_class": mem_class,
                "admission_upgrade_triggered": bool(mem_class == "XL"),
                "estimate_reason": reason,
                "evidence_sources": evidence,
            }
        )

    out_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_since": args.slurm_since,
        "missing_manifest": str(missing_manifest),
        "rows": out_rows,
        "summary": {
            "total_keys": len(out_rows),
            "xl_count": sum(1 for r in out_rows if r["memory_class"] == "XL"),
            "l_count": sum(1 for r in out_rows if r["memory_class"] == "L"),
        },
    }

    out_json_path = args.output_json.resolve()
    out_csv_path = args.output_csv.resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    csv_fields = [
        "key",
        "task",
        "ablation",
        "target",
        "horizon",
        "priority_rank",
        "priority_group",
        "predicted_peak_rss_gb",
        "confidence_low_gb",
        "confidence_high_gb",
        "memory_class",
        "admission_upgrade_triggered",
        "estimate_reason",
        "evidence_sources",
    ]
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in out_rows:
            row_out = dict(row)
            row_out["evidence_sources"] = ";".join(row_out.get("evidence_sources", []))
            writer.writerow(row_out)

    print(json.dumps(out_json["summary"], indent=2))


if __name__ == "__main__":
    main()
