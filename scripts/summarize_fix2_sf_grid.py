#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return None


def _leakage_key(report: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[bool], Optional[int]]:
    limit_rows = _to_int(report.get("limit_rows"))
    label_goal_min = _to_float(report.get("label_goal_min"))
    label_horizon = _to_int(report.get("label_horizon"))
    use_edgar = bool(report.get("edgar_features"))
    return (limit_rows, label_goal_min, use_edgar, label_horizon)


def _read_leakage_reports(paths: List[Path]) -> Dict[Tuple[Optional[int], Optional[float], Optional[bool], Optional[int]], Dict[str, Any]]:
    indexed: Dict[Tuple[Optional[int], Optional[float], Optional[bool], Optional[int]], Dict[str, Any]] = {}
    for path in paths:
        data = _load_json(path)
        data["__path__"] = str(path)
        indexed[_leakage_key(data)] = data
    return indexed


def _alignment_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in report.get("runs", []):
        bench_dir = item.get("bench_dir")
        if bench_dir:
            indexed[str(bench_dir)] = item
    return indexed


def _sanity_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in report.get("benchmarks", []):
        bench_dir = item.get("bench_dir")
        if bench_dir:
            indexed[str(bench_dir)] = item
    return indexed


def _run_key(run: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[bool], Optional[int]]:
    return (
        _to_int(run.get("limit_rows")),
        _to_float(run.get("label_goal_min")),
        _to_bool(run.get("use_edgar")),
        _to_int(run.get("label_horizon")),
    )


def _format_combo(limit_rows: Optional[int], goal_min: Optional[float], use_edgar: Optional[bool]) -> str:
    return f"lr={limit_rows}|goal={goal_min}|edgar={use_edgar}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fix2_sf grid results.")
    parser.add_argument("--audit_summary", type=Path, required=True)
    parser.add_argument("--sanity_report", type=Path, required=True)
    parser.add_argument("--alignment_audit", type=Path, required=True)
    parser.add_argument("--leakage_reports", nargs="*", default=None)
    parser.add_argument("--leakage_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    leakage_paths: List[Path] = []
    if args.leakage_glob:
        leakage_paths.extend(Path(p) for p in glob.glob(args.leakage_glob))
    if args.leakage_reports:
        leakage_paths.extend(Path(p) for p in args.leakage_reports)
    leakage_paths = sorted({p for p in leakage_paths if p.exists()}, key=lambda p: str(p))

    audit_summary = _load_json(args.audit_summary)
    sanity_report = _load_json(args.sanity_report)
    alignment_report = _load_json(args.alignment_audit)

    leakage_index = _read_leakage_reports(leakage_paths)
    sanity_index = _sanity_index(sanity_report)
    alignment_index = _alignment_index(alignment_report)

    runs: List[Dict[str, Any]] = []
    for run in audit_summary.get("runs", []):
        bench_dir = str(run.get("bench_dir"))
        key = _run_key(run)
        leak = leakage_index.get(key)
        sanity = sanity_index.get(bench_dir, {})
        alignment = alignment_index.get(bench_dir, {})
        resolved_cfg = _load_yaml(Path(bench_dir) / "configs" / "resolved_config.yaml")
        dropped_static = None
        if leak is not None:
            dropped_static = leak.get("dropped_due_to_static_ratio")
        if dropped_static is None:
            dropped_static = resolved_cfg.get("dropped_due_to_static_ratio")

        warnings = sanity.get("warnings", []) or []
        strong_warning = any("STRONG WARNING" in str(w) for w in warnings)
        naive_progress_rmse = None
        if sanity.get("naive_progress_baseline"):
            naive_progress_rmse = sanity["naive_progress_baseline"].get("rmse")
        best_model_rmse = None
        if sanity.get("best_model"):
            best_model_rmse = sanity["best_model"].get("rmse")

        n_eval = None
        if sanity.get("naive_progress_baseline"):
            n_eval = sanity["naive_progress_baseline"].get("n_eval")
        if n_eval is None and sanity.get("y_stats"):
            n_eval = sanity["y_stats"].get("count")

        alignment_metrics = alignment.get("alignment", {}) if alignment else {}
        leakage_metrics = leak.get("label_vs_current_ratio", {}) if leak else {}

        run_entry = {
            "bench_dir": bench_dir,
            "limit_rows": _to_int(run.get("limit_rows")),
            "label_goal_min": _to_float(run.get("label_goal_min")),
            "use_edgar": _to_bool(run.get("use_edgar")),
            "label_horizon": _to_int(run.get("label_horizon")),
            "dropped_due_to_static_ratio": dropped_static,
            "n_eval": n_eval,
            "alignment": {
                "corr": alignment_metrics.get("corr"),
                "pct_abs_diff_lt_1e-6": alignment_metrics.get("pct_abs_diff_lt_1e-6"),
                "max_abs_diff": alignment_metrics.get("max_abs_diff"),
            },
            "leakage": {
                "corr": leakage_metrics.get("corr"),
                "allclose": leakage_metrics.get("allclose"),
                "leakage_flag": leakage_metrics.get("leakage_flag"),
            },
            "sanity_strong_warning": strong_warning,
            "best_model_rmse": best_model_rmse,
            "naive_progress_rmse": naive_progress_rmse,
            "leakage_report_path": leak.get("__path__") if leak else None,
        }
        runs.append(run_entry)

    runs_sorted = sorted(
        runs,
        key=lambda r: (
            r.get("limit_rows") or 0,
            r.get("label_goal_min") or 0,
            1 if r.get("use_edgar") else 0,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "grid_summary.json"
    md_path = args.output_dir / "grid_summary.md"

    summary = {
        "run_count": len(runs_sorted),
        "runs": runs_sorted,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    small_samples = []
    stable_runs = []
    for entry in runs_sorted:
        n_eval = entry.get("n_eval")
        strong_warning = bool(entry.get("sanity_strong_warning"))
        leakage_flag = bool(entry.get("leakage", {}).get("leakage_flag"))
        alignment_pct = entry.get("alignment", {}).get("pct_abs_diff_lt_1e-6")
        alignment_corr = entry.get("alignment", {}).get("corr")
        alignment_ok = True
        if alignment_pct is not None and alignment_pct > 0.95:
            alignment_ok = False
        if alignment_corr is not None and alignment_corr > 0.9999:
            alignment_ok = False
        combo = _format_combo(entry.get("limit_rows"), entry.get("label_goal_min"), entry.get("use_edgar"))
        if n_eval is not None and n_eval < 500:
            small_samples.append(combo)
        if (not strong_warning) and (not leakage_flag) and alignment_ok and (n_eval is not None and n_eval >= 500):
            stable_runs.append(combo)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Fix2_sf grid summary\n\n")
        f.write("| limit_rows | goal_min | use_edgar | n_eval | dropped_static | strong_warning | leakage_corr | alignment_pct_lt_1e-6 | best_rmse | naive_progress_rmse |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for entry in runs_sorted:
            f.write(
                "| {limit_rows} | {goal} | {edgar} | {n_eval} | {drop} | {warn} | {leak_corr} | {align_pct} | {best_rmse} | {naive_rmse} |\n".format(
                    limit_rows=entry.get("limit_rows"),
                    goal=entry.get("label_goal_min"),
                    edgar=entry.get("use_edgar"),
                    n_eval=entry.get("n_eval"),
                    drop=entry.get("dropped_due_to_static_ratio"),
                    warn=entry.get("sanity_strong_warning"),
                    leak_corr=entry.get("leakage", {}).get("corr"),
                    align_pct=entry.get("alignment", {}).get("pct_abs_diff_lt_1e-6"),
                    best_rmse=entry.get("best_model_rmse"),
                    naive_rmse=entry.get("naive_progress_rmse"),
                )
            )
        f.write("\n## Conclusion\n\n")
        if small_samples:
            f.write("Small-sample combos (n_eval < 500): " + ", ".join(sorted(set(small_samples))) + ".\n\n")
        else:
            f.write("All combos have n_eval >= 500.\n\n")
        if stable_runs:
            f.write("Stable combos (n_eval >= 500 and gates pass): " + ", ".join(sorted(set(stable_runs))) + ".\n\n")
        else:
            f.write("No combo meets the stability criteria (n_eval >= 500 and all gates pass).\n\n")

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
