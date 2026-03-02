#!/usr/bin/env python3
"""
Build Gate-P report for AutoFit V7.2 pilot.

Outputs:
  - docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json
  - docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
RUNS_BENCH_ROOT = ROOT / "runs" / "benchmarks"
DEFAULT_BENCH_GLOB = "block3_20260203_225620*"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _resolve_bench_dirs(explicit: Sequence[Path], bench_glob: str) -> List[Path]:
    if explicit:
        out = []
        seen = set()
        for p in explicit:
            rp = p.resolve()
            if rp in seen or not rp.exists() or not rp.is_dir():
                continue
            out.append(rp)
            seen.add(rp)
        return sorted(out)

    out = []
    if RUNS_BENCH_ROOT.exists():
        for cand in sorted(RUNS_BENCH_ROOT.glob(bench_glob)):
            if not cand.is_dir():
                continue
            if cand.name.startswith("block3_preflight_"):
                continue
            if not cand.name.startswith("block3_20260203_225620"):
                continue
            out.append(cand.resolve())
    return out


def _load_rows(bench_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bdir in bench_dirs:
        for mf in sorted(bdir.rglob("metrics.json")):
            try:
                payload = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                recs = payload.get("results", []) or []
            elif isinstance(payload, list):
                recs = payload
            else:
                recs = []
            try:
                rel = str(mf.relative_to(ROOT))
            except Exception:
                rel = str(mf)
            for r in recs:
                if not isinstance(r, dict):
                    continue
                item = dict(r)
                item["_source_path"] = rel
                out.append(item)
    return out


def _condition_key(row: Dict[str, Any]) -> Optional[Tuple[str, str, str, int]]:
    task = row.get("task")
    ablation = row.get("ablation")
    target = row.get("target")
    horizon = row.get("horizon")
    h = _to_float(horizon)
    if task is None or ablation is None or target is None or h is None:
        return None
    return str(task), str(ablation), str(target), int(round(h))


def _strict_rows(rows: Iterable[Dict[str, Any]], min_coverage: float = 0.98) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        mae = _to_float(r.get("mae"))
        cov = _to_float(r.get("prediction_coverage_ratio"))
        if mae is None:
            continue
        if r.get("fairness_pass") is True and cov is not None and cov >= min_coverage:
            item = dict(r)
            item["mae"] = mae
            item["_condition_key"] = _condition_key(r)
            out.append(item)
    return out


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    arr = sorted(vals)
    n = len(arr)
    m = n // 2
    if n % 2 == 1:
        return float(arr[m])
    return float((arr[m - 1] + arr[m]) * 0.5)


def _safe_pct(num: float, den: float) -> Optional[float]:
    if den == 0.0:
        return None
    return float(num / den * 100.0)


def _build_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    strict = _strict_rows(rows)
    non_autofit_best: Dict[Tuple[str, str, str, int], float] = {}
    v7_best: Dict[Tuple[str, str, str, int], float] = {}
    v72_best: Dict[Tuple[str, str, str, int], float] = {}

    v72_raw = [r for r in rows if str(r.get("model_name")) == "AutoFitV72"]
    fairness_ok = bool(v72_raw) and all(
        r.get("fairness_pass") is True and (_to_float(r.get("prediction_coverage_ratio")) or 0.0) >= 0.98
        for r in v72_raw
    )

    for r in strict:
        key = r.get("_condition_key")
        if key is None:
            continue
        model = str(r.get("model_name", ""))
        category = str(r.get("category", ""))
        mae = float(r["mae"])
        if model == "AutoFitV7":
            old = v7_best.get(key)
            if old is None or mae < old:
                v7_best[key] = mae
        if model == "AutoFitV72":
            old = v72_best.get(key)
            if old is None or mae < old:
                v72_best[key] = mae
        if category != "autofit" and "AutoFit" not in model:
            old = non_autofit_best.get(key)
            if old is None or mae < old:
                non_autofit_best[key] = mae

    overlap = sorted(set(v7_best.keys()).intersection(v72_best.keys()).intersection(non_autofit_best.keys()))
    win_flags = []
    gnmae_v7 = []
    gnmae_v72 = []
    count_gap_v7 = []
    count_gap_v72 = []
    for key in overlap:
        base = max(non_autofit_best[key], 1e-12)
        v7 = v7_best[key]
        v72 = v72_best[key]
        win_flags.append(v72 < v7)
        gnmae_v7.append(math.log(max(v7, 1e-12) / base))
        gnmae_v72.append(math.log(max(v72, 1e-12) / base))
        if key[2] == "investors_count":
            count_gap_v7.append(v7 / base - 1.0)
            count_gap_v72.append(v72 / base - 1.0)

    gnmae_v7_mean = sum(gnmae_v7) / len(gnmae_v7) if gnmae_v7 else None
    gnmae_v72_mean = sum(gnmae_v72) / len(gnmae_v72) if gnmae_v72 else None
    gnmae_improvement_pct = None
    if gnmae_v7_mean is not None and gnmae_v72_mean is not None and abs(gnmae_v7_mean) > 1e-12:
        gnmae_improvement_pct = ((gnmae_v7_mean - gnmae_v72_mean) / abs(gnmae_v7_mean)) * 100.0

    med_gap_v7 = _median(count_gap_v7)
    med_gap_v72 = _median(count_gap_v72)
    count_gap_reduction_pct = None
    if med_gap_v7 is not None and med_gap_v72 is not None and abs(med_gap_v7) > 1e-12:
        count_gap_reduction_pct = ((med_gap_v7 - med_gap_v72) / abs(med_gap_v7)) * 100.0

    win_rate = _safe_pct(sum(1 for x in win_flags if x), len(win_flags)) if win_flags else None

    checks = {
        "fairness_pass_100": fairness_ok,
        "investors_count_gap_reduction_ge_50pct": (
            count_gap_reduction_pct is not None and count_gap_reduction_pct >= 50.0
        ),
        "global_normalized_mae_improvement_ge_8pct": (
            gnmae_improvement_pct is not None and gnmae_improvement_pct >= 8.0
        ),
    }
    overall_pass = all(checks.values())

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "strict_comparable_only",
        "bench_dirs_scanned": sorted({str(Path(r["_source_path"]).parts[2]) for r in rows if "_source_path" in r}),
        "counts": {
            "rows_total": len(rows),
            "rows_strict": len(strict),
            "v72_rows_raw": len(v72_raw),
            "overlap_keys_v7_v72_non_autofit": len(overlap),
        },
        "metrics": {
            "v72_vs_v7_win_rate_pct": win_rate,
            "global_normalized_mae_v7": gnmae_v7_mean,
            "global_normalized_mae_v72": gnmae_v72_mean,
            "global_normalized_mae_improvement_pct": gnmae_improvement_pct,
            "investors_count_median_gap_v7": med_gap_v7,
            "investors_count_median_gap_v72": med_gap_v72,
            "investors_count_gap_reduction_pct": count_gap_reduction_pct,
        },
        "thresholds": {
            "fairness_pass_100": True,
            "investors_count_gap_reduction_ge_50pct": 50.0,
            "global_normalized_mae_improvement_ge_8pct": 8.0,
        },
        "checks": checks,
        "overall_pass": overall_pass,
        "evidence": {
            "metrics": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            "full_rows": "runs/benchmarks/*/metrics.json",
        },
    }


def _render_md(report: Dict[str, Any]) -> str:
    checks = report.get("checks", {})
    metrics = report.get("metrics", {})
    counts = report.get("counts", {})
    lines = [
        "# V7.2 Pilot Gate Report",
        "",
        f"- generated_at_utc: **{report.get('generated_at_utc')}**",
        f"- scope: **{report.get('scope')}**",
        f"- overall_pass: **{report.get('overall_pass')}**",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | pass |",
            "|---|---|",
        ]
    )
    for k, v in checks.items():
        lines.append(f"| {k} | {str(bool(v)).lower()} |")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- metrics: `{report.get('evidence', {}).get('metrics')}`",
            f"- full_rows: `{report.get('evidence', {}).get('full_rows')}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build V7.2 pilot gate report.")
    p.add_argument("--bench-dir", action="append", default=[], help="Benchmark directory (repeatable).")
    p.add_argument("--bench-glob", type=str, default=DEFAULT_BENCH_GLOB, help="Glob under runs/benchmarks.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_dirs = _resolve_bench_dirs([Path(p) for p in args.bench_dir], args.bench_glob)
    rows = _load_rows(bench_dirs)
    report = _build_report(rows)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "v72_pilot_gate_report.json"
    md_path = out_dir / "v72_pilot_gate_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "overall_pass": report.get("overall_pass"),
                "json_path": str(json_path),
                "md_path": str(md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
