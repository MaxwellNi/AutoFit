#!/usr/bin/env python3
"""
Run an audited narrow-slice smoke matrix for V740-alpha.

This script stays outside the live benchmark harness. It simply orchestrates
multiple calls to `run_v740_alpha_smoke_slice.py`, stores per-case JSON
artifacts under docs/references/, and writes a compact Markdown summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SLICE_SCRIPT = REPO_ROOT / "scripts" / "run_v740_alpha_smoke_slice.py"


DEFAULT_CASES: List[Dict[str, Any]] = [
    {
        "name": "t1_core_edgar_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "core_edgar",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 8,
        "max_rows": 800,
    },
    {
        "name": "t1_full_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "full",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 8,
        "max_rows": 800,
    },
    {
        "name": "t1_core_only_funding_h30",
        "task": "task1_outcome",
        "ablation": "core_only",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t1_core_edgar_funding_h30",
        "task": "task1_outcome",
        "ablation": "core_edgar",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t1_core_only_investors_h14",
        "task": "task1_outcome",
        "ablation": "core_only",
        "target": "investors_count",
        "horizon": 14,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t1_full_investors_h14",
        "task": "task1_outcome",
        "ablation": "full",
        "target": "investors_count",
        "horizon": 14,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t2_core_edgar_funding_h30",
        "task": "task2_forecast",
        "ablation": "core_edgar",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t3_core_edgar_funding_h30",
        "task": "task3_risk_adjust",
        "ablation": "core_edgar",
        "target": "funding_raised_usd",
        "horizon": 30,
        "max_entities": 12,
        "max_rows": 1200,
    },
    {
        "name": "t1_core_edgar_seed2_is_funded_h14",
        "task": "task1_outcome",
        "ablation": "core_edgar_seed2",
        "target": "is_funded",
        "horizon": 14,
        "max_entities": 8,
        "max_rows": 800,
    },
    {
        "name": "t1_core_only_seed2_investors_h14",
        "task": "task1_outcome",
        "ablation": "core_only_seed2",
        "target": "investors_count",
        "horizon": 14,
        "max_entities": 12,
        "max_rows": 1200,
    },
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "v740_alpha_smoke_matrix_20260325",
    )
    ap.add_argument(
        "--summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V740_ALPHA_SMOKE_MATRIX_20260325.md",
    )
    return ap.parse_args()


def _run_case(case: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    out_json = output_dir / f"{case['name']}.json"
    cmd = [
        sys.executable,
        str(SMOKE_SLICE_SCRIPT),
        "--task", case["task"],
        "--ablation", case["ablation"],
        "--target", case["target"],
        "--horizon", str(case["horizon"]),
        "--max-entities", str(case["max_entities"]),
        "--max-rows", str(case["max_rows"]),
        "--output-json", str(out_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    result: Dict[str, Any] = {
        "name": case["name"],
        "task": case["task"],
        "ablation": case["ablation"],
        "target": case["target"],
        "horizon": case["horizon"],
        "exit_code": proc.returncode,
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-12:]) if proc.stdout else "",
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-12:]) if proc.stderr else "",
        "json_path": str(out_json),
    }
    if proc.returncode == 0 and out_json.exists():
        result["summary"] = json.loads(out_json.read_text(encoding="utf-8"))
    return result


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_summary_md(path: Path, results: List[Dict[str, Any]]) -> None:
    lines = [
        "# V740 Alpha Smoke Matrix (2026-03-25)",
        "",
        "This document is generated by `scripts/run_v740_alpha_smoke_matrix.py`.",
        "It stays outside the live benchmark line and only records narrow-slice audited smoke results.",
        "",
        "## Matrix Summary",
        "",
        "| Case | Task | Ablation | Target | H | Exit | Train | Test | Feat | Const | PredStd | MAE | BinaryRate | PosWeight |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in results:
        summary = row.get("summary", {})
        metrics = summary.get("metrics", {})
        lines.append(
            "| {name} | {task} | {ablation} | {target} | {horizon} | {exit_code} | {train_rows} | {test_rows} | "
            "{feature_count} | {constant_prediction} | {prediction_std} | {mae} | {binary_train_rate} | {binary_pos_weight} |".format(
                name=row["name"],
                task=row["task"],
                ablation=row["ablation"],
                target=row["target"],
                horizon=row["horizon"],
                exit_code=row["exit_code"],
                train_rows=summary.get("train_matrix_rows", "-"),
                test_rows=summary.get("test_matrix_rows", "-"),
                feature_count=summary.get("feature_count", "-"),
                constant_prediction=summary.get("constant_prediction", "-"),
                prediction_std=_format_metric(summary.get("prediction_std")),
                mae=_format_metric(metrics.get("mae")),
                binary_train_rate=_format_metric(summary.get("binary_train_rate")),
                binary_pos_weight=_format_metric(summary.get("binary_pos_weight")),
            )
        )

    lines.extend([
        "",
        "## Per-Case Logs",
        "",
    ])
    for row in results:
        lines.extend([
            f"### {row['name']}",
            "",
            f"- Exit code: `{row['exit_code']}`",
            f"- JSON artifact: `{row['json_path']}`",
            "",
            "```text",
            row.get("stdout_tail", "").strip() or "(no stdout)",
            "```",
        ])
        if row.get("stderr_tail"):
            lines.extend([
                "",
                "```text",
                row["stderr_tail"],
                "```",
            ])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for case in DEFAULT_CASES:
        print(
            f"[v740-matrix] running {case['name']} "
            f"({case['task']} {case['ablation']} {case['target']} h={case['horizon']})",
            flush=True,
        )
        result = _run_case(case, args.output_dir)
        results.append(result)
        print(f"[v740-matrix] finished {case['name']} exit={result['exit_code']}", flush=True)

    _write_summary_md(args.summary_md, results)
    print(f"[v740-matrix] wrote summary to {args.summary_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
