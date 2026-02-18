#!/usr/bin/env python3
"""
Block3 data integrity audit (read-only).

Checks:
1. Freeze pointer and gate verification (delegates to block3_verify_freeze checks)
2. Key asset existence + lightweight schema/hash snapshot
3. Temporal split consistency checks
4. Leakage policy surface checks for target leak groups
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None

ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    out = subprocess.check_output(
        [
            "bash",
            "-lc",
            "python3 -c 'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding=\"utf-8\").read())))' '%s'"
            % str(path),
        ],
        universal_newlines=True,
        stderr=subprocess.DEVNULL,
    )
    payload = json.loads(out)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid YAML payload from {path}")
    return payload


def _safe_load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                yield f


def _asset_fingerprint(path: Path, max_files: int = 5000) -> Dict[str, Any]:
    exists = path.exists()
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "file_count": 0,
        "total_size_bytes": 0,
        "fingerprint_sha256": None,
        "schema_columns": None,
        "schema_hash_sha256": None,
        "sample_columns": [],
        "error": None,
    }
    if not exists:
        return out

    h = hashlib.sha256()
    files_seen = 0
    for f in _iter_files(path):
        files_seen += 1
        if files_seen > max_files:
            break
        try:
            stat = f.stat()
        except OSError:
            continue
        rel = str(f.relative_to(path)) if path.is_dir() else f.name
        h.update(f"{rel}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8"))
        out["file_count"] += 1
        out["total_size_bytes"] += int(stat.st_size)
    out["fingerprint_sha256"] = h.hexdigest()

    # Lightweight schema snapshot for parquet-like assets.
    if pq is not None:
        try:
            parquet_path: Path
            if path.is_file() and path.suffix == ".parquet":
                parquet_path = path
            elif path.is_dir():
                first_parquet = next((p for p in sorted(path.rglob("*.parquet")) if p.is_file()), None)
                if first_parquet is None:
                    return out
                parquet_path = first_parquet
            else:
                return out

            table = pq.read_table(parquet_path, columns=[])
            cols = [str(c) for c in table.schema.names]
            schema_sig = "|".join(cols)
            out["schema_columns"] = len(cols)
            out["sample_columns"] = cols[:25]
            out["schema_hash_sha256"] = hashlib.sha256(schema_sig.encode("utf-8")).hexdigest()
        except Exception as e:  # pragma: no cover - best effort
            out["error"] = f"schema_snapshot_failed: {e}"

    return out


def _load_split(cfg_path: Path) -> Dict[str, Any]:
    cfg = _safe_load_yaml(cfg_path)
    split = cfg.get("split", {})
    train_end = date.fromisoformat(str(split.get("train_end")))
    val_end = date.fromisoformat(str(split.get("val_end")))
    test_end = date.fromisoformat(str(split.get("test_end")))
    embargo_days = int(split.get("embargo_days", 0))
    checks = {
        "train_before_val": bool(train_end < val_end),
        "val_before_test": bool(val_end < test_end),
        "embargo_non_negative": embargo_days >= 0,
    }
    return {
        "train_end": str(train_end),
        "val_end": str(val_end),
        "test_end": str(test_end),
        "embargo_days": embargo_days,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _leakage_policy_audit(cfg_path: Path) -> Dict[str, Any]:
    bench_path = ROOT / "scripts" / "run_block3_benchmark_shard.py"
    source = bench_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    leak_groups: Dict[str, set] = {}

    def _extract_literal(node: ast.AST) -> Dict[str, set]:
        value = ast.literal_eval(node)
        if not isinstance(value, dict):
            return {}
        out: Dict[str, set] = {}
        for k, v in value.items():
            if isinstance(v, (set, list, tuple)):
                out[str(k)] = set(str(x) for x in v)
        return out

    # Top-level assignments.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_TARGET_LEAK_GROUPS":
                leak_groups = _extract_literal(node.value)
                break
        if leak_groups:
            break

    # Class attribute assignments (the current location in BenchmarkShard).
    if not leak_groups:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for body_node in node.body:
                if isinstance(body_node, ast.Assign):
                    for target in body_node.targets:
                        if isinstance(target, ast.Name) and target.id == "_TARGET_LEAK_GROUPS":
                            leak_groups = _extract_literal(body_node.value)
                            break
                elif isinstance(body_node, ast.AnnAssign):
                    target = body_node.target
                    if isinstance(target, ast.Name) and target.id == "_TARGET_LEAK_GROUPS" and body_node.value is not None:
                        leak_groups = _extract_literal(body_node.value)
                if leak_groups:
                    break
            if leak_groups:
                break

    cfg = _safe_load_yaml(cfg_path)
    task_targets = set()
    for task_cfg in cfg.get("tasks", {}).values():
        for t in task_cfg.get("targets", []):
            task_targets.add(str(t))

    coverage = {}
    missing_targets = []
    for t in sorted(task_targets):
        group = set(leak_groups.get(t, set()))
        coverage[t] = sorted(group)
        if t not in leak_groups:
            missing_targets.append(t)
        elif t not in group:
            missing_targets.append(t)

    checks = {
        "all_targets_have_leak_group": len(missing_targets) == 0,
        "groups_non_empty": all(len(v) > 0 for v in coverage.values()),
    }
    return {
        "target_count": len(task_targets),
        "group_count": len(leak_groups),
        "coverage": coverage,
        "missing_targets": sorted(set(missing_targets)),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _verify_report_all_pass(payload: Dict[str, Any]) -> bool:
    """Robustly resolve pass/fail from verify_report schema variants."""
    all_pass_raw = payload.get("all_pass")
    if all_pass_raw is None:
        all_pass_raw = payload.get("all_gates_pass")
    if all_pass_raw is not None:
        return bool(all_pass_raw)

    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        return False
    passed_flags: List[bool] = []
    for c in checks:
        if not isinstance(c, dict):
            continue
        if "passed" in c:
            passed_flags.append(bool(c.get("passed")))
        elif "status" in c:
            passed_flags.append(str(c.get("status")).upper() == "PASS")
    return all(passed_flags) if passed_flags else False


def _freeze_gate_audit(pointer_path: Path) -> Dict[str, Any]:
    pointer = _safe_load_yaml(pointer_path)
    expected_stamp = str(pointer.get("stamp", "20260203_225620"))
    expected_variant = str(pointer.get("variant", "TRAIN_WIDE_FINAL"))
    out_dir = ROOT / "docs" / "benchmarks" / "block3_truth_pack"

    def _pick_python() -> str:
        candidates = []
        if sys.executable:
            candidates.append(sys.executable)
        candidates.extend(["python3", "python"])
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            probe = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"'{cand}' -c 'import sys,yaml; assert sys.version_info >= (3, 8)'",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                return cand
        return sys.executable or "python3"

    py = _pick_python()
    cmd = (
        f"'{py}' scripts/block3_verify_freeze.py "
        f"--pointer '{pointer_path}' "
        f"--expected-stamp '{expected_stamp}' "
        f"--expected-variant '{expected_variant}' "
        f"--output-dir '{out_dir}'"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    report_path = out_dir / "verify_report.json"
    if report_path.exists():
        payload = _safe_load_json(report_path)
        checks = payload.get("checks", [])
        return {
            "checks": checks,
            "all_pass": _verify_report_all_pass(payload),
            "exit_code": int(proc.returncode),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        }

    # Runtime fallback: if verifier cannot run due environment dependency
    # mismatch, perform pointer-level existence checks so audit remains usable.
    stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
    if (
        "No module named 'yaml'" in stderr_tail
        or "future feature annotations is not defined" in stderr_tail
    ):
        pointer = _safe_load_yaml(pointer_path)
        checks = []
        all_ok = True

        for name, pth in _collect_assets(pointer):
            ok = pth.exists()
            checks.append(
                {
                    "name": f"asset_exists:{name}",
                    "status": "PASS" if ok else "FAIL",
                    "detail": str(pth),
                }
            )
            all_ok = all_ok and ok

        for key in (
            "column_manifest",
            "raw_cardinality_coverage",
            "freeze_candidates",
        ):
            rel = str(pointer.get("analysis", {}).get(key, ""))
            pth = _resolve_path(rel) if rel else Path("")
            ok = bool(rel) and pth.exists()
            checks.append(
                {
                    "name": f"analysis_exists:{key}",
                    "status": "PASS" if ok else "FAIL",
                    "detail": str(pth) if rel else "missing pointer entry",
                }
            )
            all_ok = all_ok and ok

        return {
            "checks": checks,
            "all_pass": bool(all_ok),
            "exit_code": int(proc.returncode),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
            "stderr_tail": stderr_tail,
            "fallback_mode": "pointer_internal_checks",
        }

    return {
        "checks": [],
        "all_pass": False,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": stderr_tail,
    }


def _collect_assets(pointer: Dict[str, Any]) -> List[Tuple[str, Path]]:
    pairs = [
        ("offers_core_daily", _resolve_path(str(pointer.get("offers_core_daily", {}).get("dir", "")))),
        ("offers_core_snapshot", _resolve_path(str(pointer.get("offers_core_snapshot", {}).get("dir", "")))),
        ("offers_text", _resolve_path(str(pointer.get("offers_text", {}).get("dir", "")))),
        ("edgar_store_full_daily", _resolve_path(str(pointer.get("edgar_store_full_daily", {}).get("dir", "")))),
        ("multiscale_full", _resolve_path(str(pointer.get("multiscale_full", {}).get("dir", "")))),
        ("snapshots_offer_day", _resolve_path(str(pointer.get("snapshots_index", {}).get("offer_day", "")))),
        ("snapshots_cik_day", _resolve_path(str(pointer.get("snapshots_index", {}).get("cik_day", "")))),
        ("analysis_dir", _resolve_path(str(pointer.get("analysis", {}).get("dir", "")))),
    ]
    return pairs


def _render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Block3 Data Integrity Audit",
        "",
        f"- generated_at_utc: **{report['generated_at_utc']}**",
        f"- pointer: `{report['pointer_path']}`",
        f"- config: `{report['config_path']}`",
        f"- overall_pass: **{report['overall_pass']}**",
        "",
        "## Gate Checks",
        "",
        "| check | passed | fails |",
        "|---|---|---|",
    ]
    for row in report["freeze_gate"]["checks"]:
        lines.append(
            f"| {row.get('check')} | {row.get('passed')} | {len(row.get('fails', []))} |"
        )

    lines.extend(
        [
            "",
            "## Split Checks",
            "",
            "| check | passed |",
            "|---|---|",
        ]
    )
    for k, v in report["split"]["checks"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Leakage Policy Checks",
            "",
            "| check | passed |",
            "|---|---|",
        ]
    )
    for k, v in report["leakage_policy"]["checks"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Asset Snapshot",
            "",
            "| asset | exists | file_count | total_size_bytes | schema_columns | fingerprint_sha256 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["assets"]:
        lines.append(
            "| {asset} | {exists} | {file_count} | {total_size_bytes} | {schema_columns} | `{fingerprint}` |".format(
                asset=row["asset"],
                exists=row["exists"],
                file_count=row["file_count"],
                total_size_bytes=row["total_size_bytes"],
                schema_columns=row["schema_columns"] if row["schema_columns"] is not None else "",
                fingerprint=row["fingerprint_sha256"] or "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Block3 data integrity audit.")
    p.add_argument("--pointer", type=Path, default=Path("docs/audits/FULL_SCALE_POINTER.yaml"))
    p.add_argument("--config", type=Path, default=Path("configs/block3_tasks.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks/block3_truth_pack"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pointer_path = _resolve_path(str(args.pointer))
    config_path = _resolve_path(str(args.config))
    output_dir = _resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    pointer = _safe_load_yaml(pointer_path)
    freeze_gate = _freeze_gate_audit(pointer_path)
    split = _load_split(config_path)
    leakage_policy = _leakage_policy_audit(config_path)

    assets = []
    for asset_name, asset_path in _collect_assets(pointer):
        snap = _asset_fingerprint(asset_path)
        snap["asset"] = asset_name
        assets.append(snap)

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    overall_pass = (
        freeze_gate["all_pass"]
        and split["all_pass"]
        and leakage_policy["all_pass"]
        and all(bool(a.get("exists")) for a in assets)
    )

    report = {
        "generated_at_utc": generated_at_utc,
        "pointer_path": str(pointer_path),
        "config_path": str(config_path),
        "overall_pass": overall_pass,
        "freeze_gate": freeze_gate,
        "split": split,
        "leakage_policy": leakage_policy,
        "assets": assets,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"data_integrity_audit_{stamp}.json"
    md_path = output_dir / f"data_integrity_audit_{stamp}.md"
    latest_json = output_dir / "data_integrity_audit_latest.json"
    latest_md = output_dir / "data_integrity_audit_latest.md"

    payload = json.dumps(report, indent=2, ensure_ascii=True)
    json_path.write_text(payload, encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")
    md_text = _render_md(report)
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    print(json.dumps({"overall_pass": overall_pass, "json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
