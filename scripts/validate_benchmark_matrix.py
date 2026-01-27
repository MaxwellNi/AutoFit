from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import glob
import re


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("benchmark_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_dir(
    bench_dir: Path,
    *,
    expected_runs: int | None,
    require_exact: bool,
    require_backbones: list[str] | None,
    require_fusions: list[str] | None,
    require_module_variants: list[str] | None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"bench_dir": str(bench_dir), "ok": True, "errors": []}
    metrics_path = bench_dir / "metrics.json"
    metrics_parquet = bench_dir / "metrics.parquet"
    if not metrics_path.exists():
        report["ok"] = False
        report["errors"].append("missing metrics.json")
        return report
    if not metrics_parquet.exists():
        report["ok"] = False
        report["errors"].append("missing metrics.parquet")
        return report

    metrics = _load_metrics(metrics_path)
    results = metrics.get("results", [])
    if not results:
        report["ok"] = False
        report["errors"].append("no results in metrics.json")
        return report

    if metrics.get("cutoff_violation", 1) != 0:
        report["ok"] = False
        report["errors"].append(f"cutoff_violation={metrics.get('cutoff_violation')}")

    unique_backbones = {r.get("backbone") for r in results if r.get("backbone")}
    unique_fusions = {r.get("fusion_type") for r in results if r.get("fusion_type")}
    unique_modules = {str(r.get("module_flags")) for r in results if r.get("module_flags") is not None}
    unique_variants = {
        str(r.get("module_variant")) if r.get("module_variant") is not None else str(r.get("module_flags"))
        for r in results
        if r.get("module_variant") is not None or r.get("module_flags") is not None
    }

    if len(unique_backbones) < 3:
        report["ok"] = False
        report["errors"].append(f"unique_backbones<{3} ({len(unique_backbones)})")
    if len(unique_fusions) < 2:
        report["ok"] = False
        report["errors"].append(f"unique_fusion_types<{2} ({len(unique_fusions)})")
    if len(unique_modules) < 2:
        report["ok"] = False
        report["errors"].append(f"unique_module_configs<{2} ({len(unique_modules)})")

    if len(results) < 10:
        report["ok"] = False
        report["errors"].append(f"results_count<{10} ({len(results)})")

    if expected_runs is not None:
        if require_exact and len(results) != expected_runs:
            report["ok"] = False
            report["errors"].append(f"results_count!=expected_runs ({len(results)}!={expected_runs})")
        elif not require_exact and len(results) < expected_runs:
            report["ok"] = False
            report["errors"].append(f"results_count<expected_runs ({len(results)}<{expected_runs})")

    failed = [r for r in results if r.get("status") != "success"]
    if failed:
        report["ok"] = False
        report["errors"].append(f"failed_results={len(failed)}")

    if expected_runs is not None:
        if require_exact and len(results) != expected_runs:
            report["ok"] = False
            report["errors"].append(f"results_count!=expected_runs ({len(results)}!={expected_runs})")
        elif not require_exact and len(results) < expected_runs:
            report["ok"] = False
            report["errors"].append(f"results_count<expected_runs ({len(results)}<{expected_runs})")

    failed = [r for r in results if r.get("status") != "success"]
    if failed:
        report["ok"] = False
        report["errors"].append(f"failed_results={len(failed)}")

    use_edgar = metrics.get("use_edgar")
    if use_edgar is None:
        use_edgar = bool(results[0].get("use_edgar", False))
    report["use_edgar"] = bool(use_edgar)

    # parquet-only outputs: no csv files in benchmark dir
    csv_hits = [str(p) for p in bench_dir.rglob("*.csv")]
    if csv_hits:
        report["ok"] = False
        report["errors"].append(f"csv_outputs_found={len(csv_hits)}")
        report["csv_outputs"] = csv_hits

    if require_backbones:
        required = set(require_backbones)
        missing = required - unique_backbones
        if missing:
            report["ok"] = False
            report["errors"].append(f"missing_backbones={sorted(missing)}")
        if require_exact and unique_backbones != required:
            report["ok"] = False
            report["errors"].append("unique_backbones_not_exact_match")

    if require_fusions:
        required = set(require_fusions)
        missing = required - unique_fusions
        if missing:
            report["ok"] = False
            report["errors"].append(f"missing_fusions={sorted(missing)}")
        if require_exact and unique_fusions != required:
            report["ok"] = False
            report["errors"].append("unique_fusions_not_exact_match")

    if require_module_variants:
        required = set(require_module_variants)
        missing = required - unique_variants
        if missing:
            report["ok"] = False
            report["errors"].append(f"missing_module_variants={sorted(missing)}")
        if require_exact and unique_variants != required:
            report["ok"] = False
            report["errors"].append("unique_module_variants_not_exact_match")

    report["unique_backbones"] = sorted(unique_backbones)
    report["unique_fusions"] = sorted(unique_fusions)
    report["unique_modules"] = sorted(unique_modules)
    report["unique_module_variants"] = sorted(unique_variants)
    report["results_count"] = len(results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark matrix outputs")
    parser.add_argument("--bench_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/benchmarks_validation"))
    parser.add_argument("--expected_runs", type=int, default=None)
    parser.add_argument("--require_exact", type=int, default=0)
    parser.add_argument("--require_backbones", nargs="+", default=None)
    parser.add_argument("--require_fusions", nargs="+", default=None)
    parser.add_argument("--require_module_variants", nargs="+", default=None)
    args = parser.parse_args()

    expanded: List[Path] = []
    for pattern in args.bench_dirs:
        matches = sorted([Path(p) for p in glob.glob(pattern)])
        if not matches:
            expanded.append(Path(pattern))
            continue
        if len(matches) > 1:
            expanded.append(matches[-1])
        else:
            expanded.append(matches[0])

    deduped: List[Path] = []
    by_prefix: Dict[str, List[Path]] = {}
    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    for p in expanded:
        match = pattern.match(p.name)
        if match:
            prefix = match.group(1)
            by_prefix.setdefault(prefix, []).append(p)
        else:
            deduped.append(p)
    for prefix, paths in by_prefix.items():
        deduped.append(sorted(paths)[-1])
    deduped: List[Path] = []
    by_prefix: Dict[str, List[Path]] = {}
    pattern = re.compile(r"^(.*)_\d{8}_\d{6}$")
    for p in expanded:
        match = pattern.match(p.name)
        if match:
            prefix = match.group(1)
            by_prefix.setdefault(prefix, []).append(p)
        else:
            deduped.append(p)
    for prefix, paths in by_prefix.items():
        deduped.append(sorted(paths)[-1])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir / "logs" / "validate_benchmark_matrix.log")

    logger.info("bench_dirs=%s", [str(p) for p in deduped])

    per_dir = [
        _check_dir(
            p,
            expected_runs=args.expected_runs,
            require_exact=bool(int(args.require_exact)),
            require_backbones=args.require_backbones,
            require_fusions=args.require_fusions,
            require_module_variants=args.require_module_variants,
        )
        for p in deduped
    ]
    per_dir = [
        _check_dir(
            p,
            expected_runs=args.expected_runs,
            require_exact=bool(int(args.require_exact)),
            require_backbones=args.require_backbones,
            require_fusions=args.require_fusions,
            require_module_variants=args.require_module_variants,
        )
        for p in deduped
    ]
    has_on = any(d.get("use_edgar") is True for d in per_dir)
    has_off = any(d.get("use_edgar") is False for d in per_dir)
    overall_ok = all(d.get("ok") for d in per_dir) and has_on and has_off
    if not has_on or not has_off:
        overall_ok = False

    report = {
        "overall_ok": overall_ok,
        "has_edgar_on": has_on,
        "has_edgar_off": has_off,
        "expected_runs": args.expected_runs,
        "require_exact": bool(int(args.require_exact)),
        "require_backbones": args.require_backbones,
        "require_fusions": args.require_fusions,
        "require_module_variants": args.require_module_variants,
        "benchmarks": per_dir,
    }

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("overall_ok=%s has_edgar_on=%s has_edgar_off=%s", overall_ok, has_on, has_off)
    logger.info("report_path=%s", report_path)


if __name__ == "__main__":
    main()
