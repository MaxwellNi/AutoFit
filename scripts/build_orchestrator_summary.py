#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import socket

import yaml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_hashes(paths: List[Path]) -> Dict[str, str]:
    return {str(p): _sha256(p) for p in paths if p.exists()}


def _read_status_pre(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip()
    return data


def _best_rmse(metrics: Dict[str, Any]) -> float | None:
    best = None
    for row in metrics.get("results", []) or []:
        try:
            rmse = float(row.get("rmse"))
        except (TypeError, ValueError):
            continue
        if not (rmse == rmse):
            continue
        if best is None or rmse < best:
            best = rmse
    return best


def _fail(path: Path, message: str) -> None:
    path.write_text(f"# FAILURE\n\n{message}\n", encoding="utf-8")
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build orchestrator SUMMARY.md and MANIFEST_full.json")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--sanity_report", type=str, default=None)
    args = parser.parse_args()

    root = Path("/home/pni/project/repo_root")
    stamp = args.stamp
    orch = root / "runs" / "orchestrator" / stamp
    orch.mkdir(parents=True, exist_ok=True)

    artifacts_path = orch / "ARTIFACTS.json"
    bench_list_path = orch / "bench_dirs_all.txt"
    if not artifacts_path.exists():
        _fail(orch / "FAILURE.md", f"missing ARTIFACTS.json: {artifacts_path}")
    if not bench_list_path.exists():
        _fail(orch / "FAILURE.md", f"missing bench_dirs_all.txt: {bench_list_path}")

    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    offers_core = root / artifacts["offers_core"]
    if not offers_core.exists():
        _fail(orch / "FAILURE.md", f"missing offers_core: {offers_core}")
    offers_sha = _sha256(offers_core)
    if offers_sha != artifacts.get("offers_core_sha256"):
        _fail(orch / "FAILURE.md", "offers_core_sha256 mismatch vs ARTIFACTS.json")

    selection_hash_path = root / artifacts["selection_hash_path"]
    if not selection_hash_path.exists():
        _fail(orch / "FAILURE.md", f"missing selection_hash: {selection_hash_path}")
    selection_hash = selection_hash_path.read_text(encoding="utf-8").strip()
    if selection_hash != artifacts.get("selection_hash"):
        _fail(orch / "FAILURE.md", "selection_hash mismatch vs ARTIFACTS.json")

    bench_dirs = [Path(p) for p in bench_list_path.read_text(encoding="utf-8").splitlines() if p.strip()]
    if not bench_dirs:
        _fail(orch / "FAILURE.md", "bench_dirs_all.txt empty")

    sanity_report = Path(args.sanity_report) if args.sanity_report else orch / "sanity_report.json"
    sanity_by_bench: Dict[str, Dict[str, Any]] = {}
    if sanity_report.exists():
        report = json.loads(sanity_report.read_text(encoding="utf-8"))
        for entry in report.get("benchmarks", []):
            sanity_by_bench[str(entry.get("bench_dir"))] = entry

    edgar_manifest_path = root / "runs" / "edgar_feature_store" / "latest.txt"
    if not edgar_manifest_path.exists():
        _fail(orch / "FAILURE.md", "missing runs/edgar_feature_store/latest.txt")
    edgar_stamp = edgar_manifest_path.read_text(encoding="utf-8").strip()
    edgar_manifest = root / "runs" / "edgar_feature_store" / edgar_stamp / "MANIFEST.json"
    if not edgar_manifest.exists():
        _fail(orch / "FAILURE.md", f"missing EDGAR MANIFEST.json: {edgar_manifest}")
    edgar_manifest_data = json.loads(edgar_manifest.read_text(encoding="utf-8"))

    leakage_root = orch / "leakage"
    summary_rows: List[Dict[str, Any]] = []
    status_pre_entries: List[Dict[str, str]] = []
    resolved_config_hashes: Dict[str, str] = {}

    for bench_dir in bench_dirs:
        if not bench_dir.exists():
            _fail(orch / "FAILURE.md", f"missing bench_dir: {bench_dir}")

        metrics_json = bench_dir / "metrics.json"
        metrics_parquet = bench_dir / "metrics.parquet"
        config_path = bench_dir / "configs" / "resolved_config.yaml"
        if not metrics_json.exists() or not metrics_parquet.exists():
            _fail(orch / "FAILURE.md", f"missing metrics in {bench_dir}")
        if not config_path.exists():
            _fail(orch / "FAILURE.md", f"missing resolved_config.yaml in {bench_dir}")

        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        resolved_config_hashes[str(config_path)] = _sha256(config_path)

        leakage_path = leakage_root / bench_dir.name / "label_leakage_report.json"
        if not leakage_path.exists():
            _fail(orch / "FAILURE.md", f"missing leakage report: {leakage_path}")
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        if leakage.get("label_vs_current_ratio", {}).get("leakage_flag"):
            _fail(orch / "FAILURE.md", f"leakage_flag true: {leakage_path}")

        status_pre = None
        if bool(config.get("use_edgar", False)):
            status_path = bench_dir / "STATUS_PRE.txt"
            if not status_path.exists():
                _fail(orch / "FAILURE.md", f"missing STATUS_PRE.txt in {bench_dir}")
            status_pre = _read_status_pre(status_path)
            if "col_hash" not in status_pre:
                _fail(orch / "FAILURE.md", f"missing col_hash in {status_path}")
            status_pre_entries.append(status_pre)

        best_rmse = _best_rmse(metrics)
        sanity_entry = sanity_by_bench.get(str(bench_dir))
        naive_progress_rmse = None
        if sanity_entry:
            naive_progress_rmse = sanity_entry.get("naive_progress_baseline", {}).get("rmse")

        summary_rows.append(
            {
                "exp_name": config.get("exp_name"),
                "bench_dir": str(bench_dir),
                "label_horizon": config.get("label_horizon"),
                "strict_future": config.get("strict_future"),
                "use_edgar": config.get("use_edgar"),
                "best_model_rmse": best_rmse,
                "naive_progress_rmse": naive_progress_rmse,
                "leakage_flag": leakage.get("label_vs_current_ratio", {}).get("leakage_flag"),
            }
        )

    summary_lines = [
        f"# SUMMARY (stamp={stamp})",
        "",
        "## Audit Keys",
        f"- git_head: {artifacts.get('git_head')}",
        f"- offers_core_sha256: {offers_sha}",
        f"- selection_hash: {selection_hash}",
        f"- edgar_delta_versions: {edgar_manifest_data.get('delta_versions')}",
        f"- hostname: {socket.gethostname()}",
        "",
        "## Runs",
        "| exp_name | horizon | strict_future | use_edgar | best_model_rmse | naive_progress_rmse | leakage_flag | bench_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| {row['exp_name']} | {row['label_horizon']} | {row['strict_future']} | "
            f"{row['use_edgar']} | {row['best_model_rmse']} | {row['naive_progress_rmse']} | "
            f"{row['leakage_flag']} | {row['bench_dir']} |"
        )

    summary_path = orch / "SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    collected_dir = orch / "collected"
    metrics_master_paths = [
        collected_dir / "metrics_master.parquet",
        collected_dir / "metrics_master.csv",
    ]
    paper_tables_dir = orch / "paper_tables"
    table_files = list(paper_tables_dir.rglob("*.parquet")) + list(paper_tables_dir.rglob("*.csv")) + list(
        paper_tables_dir.rglob("*.tex")
    )

    manifest = {
        "stamp": stamp,
        "git_head": artifacts.get("git_head"),
        "offers_core": artifacts.get("offers_core"),
        "offers_core_sha256": offers_sha,
        "selection_hash": selection_hash,
        "edgar_manifest": str(edgar_manifest),
        "edgar_manifest_sha256": _sha256(edgar_manifest),
        "bench_dirs": [str(p) for p in bench_dirs],
        "resolved_config_sha256": resolved_config_hashes,
        "leakage_reports": [str((leakage_root / p.name / "label_leakage_report.json")) for p in bench_dirs],
        "sanity_report": str(sanity_report) if sanity_report.exists() else None,
        "status_pre": status_pre_entries,
        "summary_md": str(summary_path),
        "summary_md_sha256": _sha256(summary_path),
        "metrics_master_hashes": _collect_hashes(metrics_master_paths),
        "paper_tables_hashes": _collect_hashes(table_files),
    }

    manifest_path = orch / "MANIFEST_full.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
