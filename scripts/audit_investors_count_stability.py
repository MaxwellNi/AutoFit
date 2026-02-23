#!/usr/bin/env python3
"""
Investors-count stability audit for Block3.

Scope:
1. Train/val/test distribution shift checks (KS/PSI/Wasserstein/quantiles)
2. Historical strict-comparable repeatability from materialized metrics
3. Count-lane guard telemetry summary for AutoFitV71/V72 records
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def _std(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    mu = _mean(vals)
    return float(math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals)))


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = max(0.0, min(1.0, q)) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _ks_stat(a: Iterable[float], b: Iterable[float]) -> float:
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = sorted(a)
    b = sorted(b)
    grid = sorted(set(a + b))
    ia = 0
    ib = 0
    max_diff = 0.0
    for g in grid:
        while ia < len(a) and a[ia] <= g:
            ia += 1
        while ib < len(b) and b[ib] <= g:
            ib += 1
        cdf_a = ia / len(a)
        cdf_b = ib / len(b)
        max_diff = max(max_diff, abs(cdf_a - cdf_b))
    return float(max_diff)


def _psi(train: Iterable[float], other: Iterable[float], bins: int = 10) -> float:
    train = [float(x) for x in train]
    other = [float(x) for x in other]
    if len(train) == 0 or len(other) == 0:
        return float("nan")
    edges = [_quantile(train, i / bins) for i in range(bins + 1)]
    edges[0] = float("-inf")
    edges[-1] = float("inf")

    def _hist(vals: List[float]) -> List[int]:
        h = [0 for _ in range(bins)]
        for v in vals:
            for i in range(bins):
                if edges[i] <= v < edges[i + 1] or (i == bins - 1 and v == edges[i + 1]):
                    h[i] += 1
                    break
        return h

    h_train = _hist(train)
    h_other = _hist(other)
    s_train = max(sum(h_train), 1)
    s_other = max(sum(h_other), 1)
    psi = 0.0
    for a_pct, b_pct in zip(h_train, h_other):
        p = max(a_pct / s_train, 1e-6)
        q = max(b_pct / s_other, 1e-6)
        psi += (q - p) * math.log(q / p)
    return float(psi)


def _wasserstein_1d(a: Iterable[float], b: Iterable[float]) -> float:
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = sorted(a)
    b = sorted(b)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    ai = [int(round(i * (len(a) - 1) / max(n - 1, 1))) for i in range(n)]
    bi = [int(round(i * (len(b) - 1) / max(n - 1, 1))) for i in range(n)]
    return float(_mean([abs(a[i] - b[j]) for i, j in zip(ai, bi)]))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_split(cfg_path: Path) -> Dict[str, Any]:
    if yaml is not None:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                "python3 -c 'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding=\"utf-8\").read())))' '%s'"
                % str(cfg_path),
            ],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
        cfg = json.loads(out)
    return cfg.get("split", {})


def _load_panel_for_target(pointer_path: Path, target: str = "investors_count") -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required for panel distribution audit")
    try:
        from src.narrative.data_preprocessing.block3_dataset import Block3Dataset  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:
        from narrative.data_preprocessing.block3_dataset import Block3Dataset  # type: ignore  # pylint: disable=import-outside-toplevel

    ds = Block3Dataset.from_pointer(pointer_path)
    core = ds.get_offers_core_daily(columns=["crawled_date_day", target])
    out = core.copy()
    out["crawled_date_day"] = pd.to_datetime(out["crawled_date_day"])
    return out


def _split_values(df: pd.DataFrame, split_cfg: Dict[str, Any], target: str = "investors_count") -> Dict[str, List[float]]:
    if pd is None:
        raise RuntimeError("pandas is required for panel split audit")
    train_end = pd.Timestamp(split_cfg.get("train_end"))
    val_end = pd.Timestamp(split_cfg.get("val_end"))
    test_end = pd.Timestamp(split_cfg.get("test_end"))
    embargo = int(split_cfg.get("embargo_days", 7))

    train_cut = train_end
    val_start = train_end + pd.Timedelta(days=embargo + 1)
    val_cut = val_end
    test_start = val_end + pd.Timedelta(days=embargo + 1)
    test_cut = test_end

    x = df[["crawled_date_day", target]].dropna().copy()
    train = x[(x["crawled_date_day"] <= train_cut)][target].to_numpy(dtype=float)
    val = x[(x["crawled_date_day"] >= val_start) & (x["crawled_date_day"] <= val_cut)][target].to_numpy(dtype=float)
    test = x[(x["crawled_date_day"] >= test_start) & (x["crawled_date_day"] <= test_cut)][target].to_numpy(dtype=float)
    return {"train": list(train), "val": list(val), "test": list(test)}


def _dist_stats(arr: Iterable[float]) -> Dict[str, Any]:
    vals = [float(x) for x in arr]
    if len(vals) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": int(len(vals)),
        "mean": _mean(vals),
        "std": _std(vals),
        "median": _quantile(vals, 0.50),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": max(vals),
    }


def _collect_metrics_records(bench_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bdir in bench_dirs:
        if not bdir.exists():
            continue
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
            for r in recs:
                if not isinstance(r, dict):
                    continue
                item = dict(r)
                item["_source_path"] = str(mf)
                rows.append(item)
    return rows


def _strict_filter(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if str(r.get("target")) != "investors_count":
            continue
        fair = r.get("fairness_pass")
        cov = _safe_float(r.get("prediction_coverage_ratio"))
        mae = _safe_float(r.get("mae"))
        if mae is None:
            continue
        if fair is True and cov is not None and cov >= 0.98:
            out.append(r)
    return out


def _repeatability_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, int, str], List[float]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get("task")),
            str(r.get("ablation")),
            str(r.get("target")),
            int(r.get("horizon")),
            str(r.get("model_name")),
        )
        mae = _safe_float(r.get("mae"))
        if mae is not None:
            grouped[key].append(mae)

    out: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        if not values:
            continue
        arr = [float(v) for v in values]
        mean = _mean(arr)
        std = _std(arr)
        cv = std / max(mean, 1e-12)
        out.append(
            {
                "task": key[0],
                "ablation": key[1],
                "target": key[2],
                "horizon": key[3],
                "model_name": key[4],
                "n_repeats": int(len(arr)),
                "mae_mean": mean,
                "mae_std": std,
                "mae_cv": cv,
                "catastrophic_count": int(sum(v > 1_000_000 for v in arr)),
            }
        )
    return out


def _guard_telemetry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    af_rows = [
        r for r in rows
        if str(r.get("model_name")) in {"AutoFitV71", "AutoFitV72"}
    ]
    if not af_rows:
        return {
            "n_rows": 0,
            "lane_clip_rate_median": None,
            "inverse_transform_guard_hits_total": 0,
            "oof_guard_triggered_count": 0,
            "policy_action_id_topk": [],
            "anchor_models_used_topk": [],
        }

    lane_clip = [float(_safe_float(r.get("lane_clip_rate")) or 0.0) for r in af_rows]
    inv_hits = int(sum(int(_safe_float(r.get("inverse_transform_guard_hits")) or 0.0) for r in af_rows))
    oof_guard = int(sum(bool(r.get("oof_guard_triggered", False)) for r in af_rows))
    policy_counts = Counter(str(r.get("policy_action_id")) for r in af_rows if r.get("policy_action_id"))
    anchor_counts = Counter()
    for r in af_rows:
        anchors = r.get("anchor_models_used", [])
        if isinstance(anchors, list):
            for a in anchors:
                anchor_counts[str(a)] += 1

    return {
        "n_rows": int(len(af_rows)),
        "lane_clip_rate_median": _quantile(lane_clip, 0.5),
        "inverse_transform_guard_hits_total": inv_hits,
        "oof_guard_triggered_count": oof_guard,
        "policy_action_id_topk": policy_counts.most_common(5),
        "anchor_models_used_topk": anchor_counts.most_common(10),
    }


def _render_md(report: Dict[str, Any]) -> str:
    pair_rows = report["distribution_shift"]["pairwise"]
    lines = [
        "# Investors Count Stability Audit",
        "",
        f"- generated_at_utc: **{report['generated_at_utc']}**",
        f"- overall_pass: **{report['overall_pass']}**",
        "",
        "## Split Distribution",
        "",
        "| split | n | mean | std | median | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in report["distribution_shift"]["stats"].items():
        lines.append(
            f"| {split} | {stats['n']} | {stats['mean']} | {stats['std']} | {stats['median']} | {stats['p95']} | {stats['p99']} | {stats['max']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Drift Metrics",
            "",
            "| pair | ks | psi | wasserstein |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['ks']} | {row['psi']} | {row['wasserstein']} |"
        )

    lines.extend(
        [
            "",
            "## Telemetry",
            "",
            f"- strict_records: **{report['strict_record_count']}**",
            f"- catastrophic_spikes: **{report['catastrophic_spikes']}**",
            f"- repeatability_groups: **{report['repeatability_group_count']}**",
            "",
            "## Gate Checks",
            "",
            "| check | pass |",
            "|---|---|",
        ]
    )
    for k, v in report["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run investors_count stability audit.")
    p.add_argument("--pointer", type=Path, default=Path("docs/audits/FULL_SCALE_POINTER.yaml"))
    p.add_argument("--config", type=Path, default=Path("configs/block3_tasks.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks/block3_truth_pack"))
    p.add_argument(
        "--bench-dir",
        action="append",
        default=[],
        help="Benchmark run dir (repeatable). Default scans block3_20260203_225620*",
    )
    return p.parse_args()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    pointer_path = _resolve_path(str(args.pointer))
    config_path = _resolve_path(str(args.config))
    output_dir = _resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bench_dir:
        bench_dirs = [_resolve_path(str(p)) for p in args.bench_dir]
    else:
        bench_dirs = [
            p.resolve()
            for p in sorted((ROOT / "runs" / "benchmarks").glob("block3_20260203_225620*"))
            if p.is_dir() and not p.name.startswith("block3_preflight_")
        ]

    split_cfg = _load_split(config_path)
    distribution_available = True
    distribution_error: str | None = None
    try:
        panel = _load_panel_for_target(pointer_path, target="investors_count")
        splits = _split_values(panel, split_cfg, target="investors_count")
        pairwise = []
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            av = splits[a]
            bv = splits[b]
            pairwise.append(
                {
                    "pair": f"{a}_vs_{b}",
                    "ks": _ks_stat(av, bv),
                    "psi": _psi(av, bv),
                    "wasserstein": _wasserstein_1d(av, bv),
                }
            )
        split_stats = {k: _dist_stats(v) for k, v in splits.items()}
    except Exception as e:
        distribution_available = False
        distribution_error = str(e)
        pairwise = [
            {"pair": "train_vs_val", "ks": None, "psi": None, "wasserstein": None},
            {"pair": "train_vs_test", "ks": None, "psi": None, "wasserstein": None},
            {"pair": "val_vs_test", "ks": None, "psi": None, "wasserstein": None},
        ]
        split_stats = {
            "train": _dist_stats([]),
            "val": _dist_stats([]),
            "test": _dist_stats([]),
        }

    records = _collect_metrics_records(bench_dirs)
    strict_rows = _strict_filter(records)
    repeatability_rows = _repeatability_table(strict_rows)
    telemetry = _guard_telemetry(strict_rows)
    catastrophic = int(sum(1 for r in strict_rows if (_safe_float(r.get("mae")) or 0.0) > 1_000_000))

    train_test_ks = _safe_float(pairwise[1]["ks"]) if len(pairwise) > 1 else None
    train_test_psi = _safe_float(pairwise[1]["psi"]) if len(pairwise) > 1 else None
    checks = {
        "distribution_available": distribution_available,
        "no_catastrophic_spike": catastrophic == 0,
        "strict_rows_present": len(strict_rows) > 0,
        "ks_train_vs_test_lt_0_25": (
            (train_test_ks is not None) and (train_test_ks < 0.25)
        ) if distribution_available else False,
        "psi_train_vs_test_lt_0_30": (
            (train_test_psi is not None) and (train_test_psi < 0.30)
        ) if distribution_available else False,
    }
    overall_pass = all(checks.values())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pointer_path": str(pointer_path),
        "config_path": str(config_path),
        "bench_dirs": [str(p) for p in bench_dirs],
        "distribution_shift": {
            "stats": split_stats,
            "pairwise": pairwise,
            "available": distribution_available,
            "error": distribution_error,
        },
        "strict_record_count": len(strict_rows),
        "catastrophic_spikes": catastrophic,
        "repeatability_group_count": len(repeatability_rows),
        "guard_telemetry": telemetry,
        "checks": checks,
        "overall_pass": overall_pass,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_json = output_dir / f"investors_count_stability_audit_{stamp}.json"
    report_md = output_dir / f"investors_count_stability_audit_{stamp}.md"
    repeat_csv = output_dir / f"investors_count_repeatability_{stamp}.csv"
    latest_json = output_dir / "investors_count_stability_audit_latest.json"
    latest_md = output_dir / "investors_count_stability_audit_latest.md"
    latest_repeat = output_dir / "investors_count_repeatability_latest.csv"

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    latest_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report_md.write_text(_render_md(report), encoding="utf-8")
    latest_md.write_text(_render_md(report), encoding="utf-8")
    if pd is not None:
        pd.DataFrame(repeatability_rows).to_csv(repeat_csv, index=False)
        pd.DataFrame(repeatability_rows).to_csv(latest_repeat, index=False)
    else:
        _write_csv(repeat_csv, repeatability_rows)
        _write_csv(latest_repeat, repeatability_rows)

    print(
        json.dumps(
            {
                "overall_pass": overall_pass,
                "report_json": str(report_json),
                "report_md": str(report_md),
                "repeatability_csv": str(repeat_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
