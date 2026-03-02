#!/usr/bin/env python3
"""Build a per-condition rank table for AutoFitV72 over the strict 104-key lattice."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
TP_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
RUNS_DIR = ROOT / "runs" / "benchmarks"
DEFAULT_BENCH_GLOB = "block3_20260203_225620*"
DEFAULT_CSV = TP_DIR / "v72_rank_104_table_latest.csv"
DEFAULT_MD = TP_DIR / "v72_rank_104_table_latest.md"
DEFAULT_JSON = TP_DIR / "v72_rank_104_summary_latest.json"


Key = Tuple[str, str, str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_int(raw: object) -> Optional[int]:
    try:
        if raw is None:
            return None
        return int(float(str(raw)))
    except Exception:
        return None


def _to_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        value = float(str(raw))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _expected_keys() -> List[Key]:
    keys: List[Key] = []
    path = TP_DIR / "condition_inventory_full.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("expected", "")).strip().lower() != "true":
                continue
            horizon = _to_int(row.get("horizon"))
            if horizon is None:
                continue
            keys.append(
                (
                    str(row.get("task", "")),
                    str(row.get("ablation", "")),
                    str(row.get("target", "")),
                    horizon,
                )
            )
    keys.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return keys


def _iter_metric_rows(bench_glob: str) -> Iterable[Dict[str, object]]:
    for bench_dir in sorted(RUNS_DIR.glob(bench_glob)):
        if not bench_dir.is_dir() or bench_dir.name.startswith("block3_preflight_"):
            continue
        for metrics_file in bench_dir.rglob("metrics.json"):
            try:
                payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                rows = payload.get("results", []) or []
            elif isinstance(payload, list):
                rows = payload
            else:
                rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                out = dict(row)
                out["_source_path"] = str(metrics_file.relative_to(ROOT))
                yield out


def _strict_row_key(row: Dict[str, object], expected: set[Key]) -> Optional[Key]:
    if str(row.get("model_name", "")) == "":
        return None
    task = str(row.get("task", ""))
    ablation = str(row.get("ablation", ""))
    target = str(row.get("target", ""))
    horizon = _to_int(row.get("horizon"))
    if horizon is None:
        return None
    key = (task, ablation, target, horizon)
    if key not in expected:
        return None
    mae = _to_float(row.get("mae"))
    cov = _to_float(row.get("prediction_coverage_ratio"))
    if mae is None or cov is None:
        return None
    if row.get("fairness_pass") is not True:
        return None
    if cov < 0.98:
        return None
    return key


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000.0:
        return f"{value:,.6f}"
    return f"{value:.6f}"


def _rank_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ranked = [r for r in rows if r["v72_status"] == "RANKED"]
    rank_values = [int(r["v72_rank"]) for r in ranked]
    gap_values = [float(r["v72_gap_pct_vs_best"]) for r in ranked]
    rank_dist = Counter(rank_values)

    by_target: Dict[str, Dict[str, object]] = {}
    for target in sorted({str(r["target"]) for r in rows}):
        subset = [r for r in rows if str(r["target"]) == target]
        ranked_subset = [r for r in subset if r["v72_status"] == "RANKED"]
        ranks = [int(r["v72_rank"]) for r in ranked_subset]
        gaps = [float(r["v72_gap_pct_vs_best"]) for r in ranked_subset]
        by_target[target] = {
            "ranked": len(ranked_subset),
            "missing": len(subset) - len(ranked_subset),
            "median_rank": (median(ranks) if ranks else None),
            "median_gap_pct_vs_best": (median(gaps) if gaps else None),
        }

    return {
        "generated_at_utc": _utc_now(),
        "expected_conditions": len(rows),
        "v72_ranked_conditions": len(ranked),
        "v72_missing_conditions": len(rows) - len(ranked),
        "rank_distribution": {str(k): v for k, v in sorted(rank_dist.items())},
        "rank1_wins": rank_dist.get(1, 0),
        "rank1_win_rate_on_ranked": (
            rank_dist.get(1, 0) / len(ranked) if ranked else None
        ),
        "median_rank_on_ranked": (median(rank_values) if rank_values else None),
        "median_gap_pct_vs_best_on_ranked": (
            median(gap_values) if gap_values else None
        ),
        "by_target": by_target,
        "csv_path": str(DEFAULT_CSV.relative_to(ROOT)),
        "md_path": str(DEFAULT_MD.relative_to(ROOT)),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "ablation",
        "target",
        "horizon",
        "v72_status",
        "v72_rank",
        "total_models",
        "v72_mae",
        "best_model",
        "best_category",
        "best_mae",
        "v72_gap_pct_vs_best",
        "evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_md(path: Path, rows: Sequence[Dict[str, object]], summary: Dict[str, object]) -> None:
    lines = [
        "# AutoFitV72 Rank Table on Full SOTA Benchmark (104 Conditions)",
        "",
        f"> Generated UTC: {summary['generated_at_utc']}",
        f"> Ranked conditions: {summary['v72_ranked_conditions']}/{summary['expected_conditions']}",
        f"> Rank-1 wins: {summary['rank1_wins']}",
        "",
        "| task | ablation | target | horizon | v72_status | v72_rank | total_models | v72_mae | best_model | best_mae | v72_gap_pct_vs_best |",
        "|---|---|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["task"]),
                    str(row["ablation"]),
                    str(row["target"]),
                    str(row["horizon"]),
                    str(row["v72_status"]),
                    str(row["v72_rank"] or "-"),
                    str(row["total_models"]),
                    str(row["v72_mae"] or "-"),
                    str(row["best_model"] or "-"),
                    str(row["best_mae"] or "-"),
                    str(row["v72_gap_pct_vs_best"] or "-"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(bench_glob: str = DEFAULT_BENCH_GLOB) -> Dict[str, object]:
    expected_keys = _expected_keys()
    expected_set = set(expected_keys)

    best_by_key_model: Dict[Tuple[Key, str], Dict[str, object]] = {}
    for row in _iter_metric_rows(bench_glob):
        key = _strict_row_key(row, expected_set)
        if key is None:
            continue
        model_name = str(row.get("model_name", ""))
        mae = _to_float(row.get("mae"))
        if mae is None:
            continue
        packed = {
            "key": key,
            "task": key[0],
            "ablation": key[1],
            "target": key[2],
            "horizon": key[3],
            "model_name": model_name,
            "category": str(row.get("category", "")),
            "mae": mae,
            "source": str(row.get("_source_path", "")),
        }
        km = (key, model_name)
        old = best_by_key_model.get(km)
        if old is None or float(packed["mae"]) < float(old["mae"]):
            best_by_key_model[km] = packed

    by_key: Dict[Key, List[Dict[str, object]]] = defaultdict(list)
    for (key, _model_name), packed in best_by_key_model.items():
        by_key[key].append(packed)

    rows: List[Dict[str, object]] = []
    for key in expected_keys:
        candidates = sorted(
            by_key.get(key, []),
            key=lambda x: (float(x["mae"]), str(x["model_name"])),
        )
        best = candidates[0] if candidates else None
        v72 = next((c for c in candidates if str(c["model_name"]) == "AutoFitV72"), None)
        if v72 is None:
            rows.append(
                {
                    "task": key[0],
                    "ablation": key[1],
                    "target": key[2],
                    "horizon": key[3],
                    "v72_status": "MISSING_STRICT_RESULT",
                    "v72_rank": "",
                    "total_models": len(candidates),
                    "v72_mae": "",
                    "best_model": (best["model_name"] if best else ""),
                    "best_category": (best["category"] if best else ""),
                    "best_mae": _fmt_num(best["mae"] if best else None),
                    "v72_gap_pct_vs_best": "",
                    "evidence": "docs/benchmarks/block3_truth_pack/condition_inventory_full.csv",
                }
            )
            continue

        rank = 1 + next(
            i for i, c in enumerate(candidates) if str(c["model_name"]) == "AutoFitV72"
        )
        best_mae = float(best["mae"]) if best is not None else None
        v72_mae = float(v72["mae"])
        gap = (
            ((v72_mae - best_mae) / best_mae) * 100.0
            if best_mae is not None and abs(best_mae) > 1e-12
            else None
        )
        rows.append(
            {
                "task": key[0],
                "ablation": key[1],
                "target": key[2],
                "horizon": key[3],
                "v72_status": "RANKED",
                "v72_rank": rank,
                "total_models": len(candidates),
                "v72_mae": _fmt_num(v72_mae),
                "best_model": (best["model_name"] if best else ""),
                "best_category": (best["category"] if best else ""),
                "best_mae": _fmt_num(best_mae),
                "v72_gap_pct_vs_best": (f"{gap:.6f}" if gap is not None else ""),
                "evidence": str(v72["source"]),
            }
        )

    summary = _rank_summary(rows)
    _write_csv(DEFAULT_CSV, rows)
    _write_md(DEFAULT_MD, rows, summary)
    DEFAULT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = build()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
