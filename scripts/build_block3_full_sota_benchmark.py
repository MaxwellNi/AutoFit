#!/usr/bin/env python3
"""Build a one-page full SOTA benchmark table for strict-comparable Block 3."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
TP_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_LEADERBOARD = TP_DIR / "condition_leaderboard.csv"
DEFAULT_SUMMARY_JSON = TP_DIR / "full_sota_104_summary.json"
DEFAULT_TABLE_CSV = TP_DIR / "full_sota_104_table.csv"
DEFAULT_MD = ROOT / "docs" / "BLOCK3_FULL_SOTA_BENCHMARK.md"
DEFAULT_TRUTH_SUMMARY = TP_DIR / "truth_pack_summary.json"


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sort_key(row: Dict[str, str]) -> tuple:
    def _to_int(v: str) -> int:
        try:
            return int(float(v))
        except Exception:
            return 0

    return (
        row.get("task", ""),
        row.get("ablation", ""),
        row.get("target", ""),
        _to_int(row.get("horizon", "0")),
    )


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _fmt_num(v: str) -> str:
    f = _to_float(v)
    if f != f:  # NaN
        return "-"
    if abs(f) >= 1000:
        return f"{f:,.3f}"
    if abs(f) >= 1:
        return f"{f:.6f}"
    return f"{f:.6f}"


def _fmt_gap(v: str) -> str:
    f = _to_float(v)
    if f != f:
        return "-"
    return f"{f:+.2f}%"


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_markdown(
    rows: List[Dict[str, str]],
    summary: Dict[str, object],
    truth_summary: Dict[str, object],
) -> str:
    strict_done = int(truth_summary.get("strict_completed_conditions", 0))
    strict_expected = int(truth_summary.get("expected_conditions", 0))
    ratio = float(truth_summary.get("strict_condition_completion_ratio", 0.0))
    bar_filled = int(round(ratio * 24))
    bar = "[" + "#" * bar_filled + "-" * (24 - bar_filled) + "]"

    lines: List[str] = []
    lines.append("# Block 3 Full SOTA Benchmark (Strict Comparable)")
    lines.append("")
    lines.append(f"> Generated UTC: {summary['generated_at_utc']}")
    lines.append(
        f"> Strict comparable completion: `{bar} {strict_done}/{strict_expected} ({ratio * 100:.1f}%)`"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value | Evidence |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| strict_completed_conditions | {strict_done}/{strict_expected} | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |"
    )
    lines.append(
        f"| table_rows | {summary['table_rows']} | `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` |"
    )
    lines.append(
        f"| champion_family_distribution | {summary['champion_family_distribution_str']} | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |"
    )
    lines.append(
        f"| top_champion_models | {summary['top_champion_models_str']} | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |"
    )
    lines.append("")
    lines.append("## Full 104-condition Champion Table")
    lines.append("")
    lines.append(
        "| task | ablation | target | horizon | champion_model | champion_family | champion_mae | best_non_autofit_model | best_non_autofit_mae | best_autofit_model | best_autofit_mae | autofit_gap_pct |"
    )
    lines.append(
        "|---|---|---|---:|---|---|---:|---|---:|---|---:|---:|"
    )

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("task", "-"),
                    r.get("ablation", "-"),
                    r.get("target", "-"),
                    r.get("horizon", "-"),
                    r.get("best_model", "-"),
                    r.get("best_category", "-"),
                    _fmt_num(r.get("best_mae", "")),
                    r.get("best_non_autofit_model", "-"),
                    _fmt_num(r.get("best_non_autofit_mae", "")),
                    r.get("best_autofit_model", "-"),
                    _fmt_num(r.get("best_autofit_mae", "")),
                    _fmt_gap(r.get("autofit_gap_pct", "")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append(
        "Source table: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full strict-comparable SOTA benchmark table."
    )
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--truth-summary", type=Path, default=DEFAULT_TRUTH_SUMMARY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    leaderboard = args.leaderboard.resolve()
    rows = _load_csv(leaderboard)
    rows = [r for r in rows if str(r.get("condition_completed", "")).lower() == "true"]
    rows.sort(key=_sort_key)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fam_counter = Counter(r.get("best_category", "unknown") for r in rows)
    model_counter = Counter(r.get("best_model", "unknown") for r in rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "leaderboard_path": str(leaderboard.relative_to(ROOT)),
        "table_csv_path": str(args.output_csv.resolve().relative_to(ROOT)),
        "markdown_path": str(args.output_md.resolve().relative_to(ROOT)),
        "table_rows": len(rows),
        "champion_family_distribution": dict(fam_counter),
        "top_champion_models": dict(model_counter.most_common(10)),
    }
    summary["champion_family_distribution_str"] = ", ".join(
        f"{k}={v}" for k, v in sorted(fam_counter.items())
    )
    summary["top_champion_models_str"] = ", ".join(
        f"{k}={v}" for k, v in model_counter.most_common(7)
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    truth_summary = _read_json(args.truth_summary.resolve())
    markdown = _build_markdown(rows, summary, truth_summary)
    args.output_md.write_text(markdown, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
