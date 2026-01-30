#!/usr/bin/env python
"""
Audit benchmark matrix truthfulness: detect actual completed runs from run artifacts,
extract metrics (rmse/mae/mse/r2, train_time_sec, peak_mem), record provenance.
Mark phantom rows (in tables but no backing run). Output benchmark_truthfulness.json/md (local, untracked).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


def _hostname() -> str:
    return os.environ.get("HOST_TAG", os.environ.get("HOSTNAME", "unknown")).replace(".", "-")[:64]


def _setup_logger(output_dir: Path, host_suffix: Optional[str] = None) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("audit_benchmark_truthfulness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    name = f"benchmark_truthfulness_{host_suffix or _hostname()}.log"
    fh = logging.FileHandler(log_dir / name, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_bench_dirs(bench_list: Path) -> List[Path]:
    if not bench_list.exists():
        return []
    lines = bench_list.read_text(encoding="utf-8").splitlines()
    return [Path(p.strip()) for p in lines if p.strip()]


def _detect_real_runs(bench_dirs: List[Path], logger: logging.Logger) -> List[Dict[str, Any]]:
    """For each bench_dir, read metrics.json (and config if present); record provenance and per-result metrics."""
    real_runs: List[Dict[str, Any]] = []
    for bench_dir in bench_dirs:
        metrics_path = bench_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning("No metrics.json: %s", bench_dir)
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse %s: %s", metrics_path, e)
            continue
        exp_name = data.get("exp_name", bench_dir.name)
        use_edgar = bool(data.get("use_edgar", False))
        status = data.get("status", "unknown")
        results = data.get("results", [])
        if not results and status != "success":
            continue
        git_head = None
        data_stamp = None
        config_path = bench_dir / "configs" / "resolved_config.yaml"
        if config_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if isinstance(cfg, dict):
                    git_head = cfg.get("git_head") or cfg.get("git_sha")
                    data_stamp = cfg.get("data_stamp") or cfg.get("offers_core_stamp")
            except Exception:
                pass
        for r in results:
            if r.get("status") != "success":
                continue
            real_runs.append({
                "exp_name": exp_name,
                "backbone": r.get("backbone", r.get("model", "unknown")),
                "fusion_type": r.get("fusion_type", "none"),
                "module_variant": r.get("module_variant", "unknown"),
                "use_edgar": use_edgar,
                "seed": r.get("seed"),
                "run_dir": str(bench_dir),
                "metrics_path": str(metrics_path),
                "rmse": r.get("rmse"),
                "mae": r.get("mae"),
                "mse": r.get("mse"),
                "r2": r.get("r2"),
                "train_time_sec": r.get("train_time_sec"),
                "max_cuda_mem_mb": r.get("max_cuda_mem_mb"),
                "git_head": git_head,
                "data_stamp": data_stamp,
            })
    return real_runs


def _phantom_rows(
    main_results_path: Path,
    real_run_keys: Set[Tuple[str, str, str, str, bool]],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Rows in main_results that have no backing run (exp_name, backbone, fusion_type, module_variant, use_edgar)."""
    phantoms: List[Dict[str, Any]] = []
    if not main_results_path.exists():
        return phantoms
    try:
        df = pd.read_csv(main_results_path)
    except Exception as e:
        logger.warning("Could not read main_results: %s", e)
        return phantoms
    key_cols = ["exp_name", "backbone", "fusion_type", "module_variant", "use_edgar"]
    for c in key_cols:
        if c not in df.columns:
            return phantoms
    for _, row in df.iterrows():
        key = (
            str(row["exp_name"]),
            str(row["backbone"]),
            str(row["fusion_type"]),
            str(row["module_variant"]),
            bool(row["use_edgar"]) if pd.notna(row["use_edgar"]) else False,
        )
        if key not in real_run_keys:
            phantoms.append(row.to_dict())
    return phantoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark matrix truthfulness (real runs vs table rows).")
    parser.add_argument("--bench_list", type=Path, required=True, help="Path to bench_dirs_all.txt")
    parser.add_argument("--output_dir", type=Path, required=True, help="Analysis output dir (e.g. runs/orchestrator/STAMP/analysis)")
    parser.add_argument("--paper_tables_dir", type=Path, default=None, help="If set, check main_results.csv for phantoms")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host = _hostname()
    logger = _setup_logger(args.output_dir, host_suffix=host)
    logger.info("=== Benchmark Truthfulness Audit Start (host=%s) ===", host)

    bench_dirs = _load_bench_dirs(args.bench_list)
    logger.info("Bench dirs from list: %d", len(bench_dirs))

    real_runs = _detect_real_runs(bench_dirs, logger)
    logger.info("Real runs (with metrics.json + success results): %d", len(real_runs))

    real_run_keys: Set[Tuple[str, str, str, str, bool]] = set()
    for r in real_runs:
        key = (
            str(r["exp_name"]),
            str(r["backbone"]),
            str(r["fusion_type"]),
            str(r["module_variant"]),
            bool(r["use_edgar"]),
        )
        real_run_keys.add(key)

    paper_dir = args.paper_tables_dir or (args.output_dir.parent / "paper_tables")
    main_results_path = paper_dir / "main_results.csv"
    phantoms = _phantom_rows(main_results_path, real_run_keys, logger)
    if phantoms:
        logger.warning("Phantom rows (in table but no backing run): %d", len(phantoms))

    metric_mapping = {
        "rmse": "metrics.json .results[].rmse",
        "mae": "metrics.json .results[].mae",
        "mse": "metrics.json .results[].mse",
        "r2": "metrics.json .results[].r2",
        "train_time_sec": "metrics.json .results[].train_time_sec (or from timing log if added)",
        "max_cuda_mem_mb": "metrics.json .results[].max_cuda_mem_mb",
    }

    report: Dict[str, Any] = {
        "real_run_count": len(real_runs),
        "bench_dirs_count": len(bench_dirs),
        "phantom_rows_count": len(phantoms),
        "phantom_rows": phantoms[:50],
        "metric_extraction_mapping": metric_mapping,
        "real_runs": real_runs,
        "real_runs_sample": real_runs[:20],
    }
    if real_runs:
        report["unique_combinations"] = len(real_run_keys)

    json_path = args.output_dir / "benchmark_truthfulness.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        "# Benchmark Truthfulness Report",
        "",
        f"- **Real runs found:** {report['real_run_count']}",
        f"- **Bench dirs in list:** {report['bench_dirs_count']}",
        f"- **Phantom rows (in main_results but no backing run):** {report['phantom_rows_count']}",
        "",
        "## Metric extraction mapping",
        "",
    ]
    for k, v in metric_mapping.items():
        md_lines.append(f"- `{k}`: {v}")
    md_lines.append("")
    md_lines.append("## Phantom rows (sample)")
    md_lines.append("")
    if phantoms:
        for i, p in enumerate(phantoms[:10]):
            md_lines.append(f"- {i+1}. {p.get('exp_name')} | {p.get('backbone')} | {p.get('fusion_type')} | {p.get('module_variant')} | use_edgar={p.get('use_edgar')}")
    else:
        md_lines.append("(none)")
    md_path = args.output_dir / "benchmark_truthfulness.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    logger.info("=== Benchmark Truthfulness Audit Complete ===")


if __name__ == "__main__":
    main()
