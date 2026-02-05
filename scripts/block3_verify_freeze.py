#!/usr/bin/env python3
"""
Block 3 Fail-Fast Freeze Verification.

This script verifies all gates before Block 3 modeling can proceed.
It reads the FULL_SCALE_POINTER.yaml and checks all audit artifacts for PASS status.

Exit codes:
  0 = All gates PASS, safe to proceed
  1 = One or more gates FAIL, do not proceed

Outputs:
  - verify_report.json
  - verify_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_pointer(pointer: Dict[str, Any], expected_stamp: str, expected_variant: str) -> Tuple[bool, List[str]]:
    """Verify FULL_SCALE_POINTER.yaml contents."""
    fails = []
    
    # Check stamp
    actual_stamp = pointer.get("stamp", "")
    if actual_stamp != expected_stamp:
        fails.append(f"stamp mismatch: expected '{expected_stamp}', got '{actual_stamp}'")
    
    # Check variant
    actual_variant = pointer.get("variant", "")
    if actual_variant != expected_variant:
        fails.append(f"variant mismatch: expected '{expected_variant}', got '{actual_variant}'")
    
    # Check all directories exist
    dir_keys = [
        "offers_core_daily.dir",
        "offers_core_snapshot.dir",
        "offers_text.dir",
        "edgar_store_full_daily.dir",
        "multiscale_full.dir",
        "snapshots_index.offer_day",
        "snapshots_index.cik_day",
        "analysis.dir",
    ]
    
    for key in dir_keys:
        parts = key.split(".")
        val = pointer
        for p in parts:
            val = val.get(p, {}) if isinstance(val, dict) else None
            if val is None:
                break
        if val is None or not val:
            fails.append(f"pointer key '{key}' missing or empty")
        elif isinstance(val, str) and not Path(val).exists():
            fails.append(f"path '{val}' (from {key}) does not exist")
    
    return len(fails) == 0, fails


def check_column_manifest(path: Path) -> Tuple[bool, List[str]]:
    """Verify column_manifest.json gates."""
    fails = []
    
    if not path.exists():
        return False, [f"column_manifest.json not found at {path}"]
    
    data = load_json(path)
    
    # must_keep_missing must be empty list or 0
    must_keep_missing = data.get("must_keep_missing", [])
    if isinstance(must_keep_missing, list):
        if len(must_keep_missing) > 0:
            fails.append(f"must_keep_missing has {len(must_keep_missing)} columns: {must_keep_missing[:5]}...")
    else:
        if must_keep_missing != 0:
            fails.append(f"must_keep_missing = {must_keep_missing}, expected 0")
    
    # edgar_recompute checks
    edgar = data.get("edgar_recompute", {})
    diff_count = edgar.get("diff_count", -1)
    total_compared = edgar.get("total_compared", 0)
    
    if diff_count != 0:
        fails.append(f"edgar_recompute.diff_count = {diff_count}, expected 0")
    if total_compared < 200:
        fails.append(f"edgar_recompute.total_compared = {total_compared}, expected >= 200")
    
    # gate_passed
    if not data.get("gate_passed", False):
        fails.append("gate_passed = False")
    
    return len(fails) == 0, fails


def check_raw_cardinality_coverage(path: Path) -> Tuple[bool, List[str]]:
    """Verify raw_cardinality_coverage.json gates."""
    fails = []
    
    if not path.exists():
        return False, [f"raw_cardinality_coverage.json not found at {path}"]
    
    data = load_json(path)
    
    # gate_passed must be true
    if not data.get("gate_passed", False):
        fails.append("gate_passed = False")
    
    # fail_reasons must be empty
    fail_reasons = data.get("fail_reasons", [])
    if len(fail_reasons) > 0:
        fails.append(f"fail_reasons not empty: {fail_reasons}")
    
    # snapshots_to_edgar_coverage - can be nested dict or float
    coverage_data = data.get("snapshots_to_edgar_coverage", {})
    if isinstance(coverage_data, dict):
        coverage = coverage_data.get("snapshots_to_edgar_coverage", 0.0)
    else:
        coverage = coverage_data
    if coverage < 0.9999:  # Allow tiny float tolerance
        fails.append(f"snapshots_to_edgar_coverage = {coverage}, expected 1.0")
    
    return len(fails) == 0, fails


def check_freeze_candidates(path: Path) -> Tuple[bool, List[str]]:
    """Verify freeze_candidates.json gates."""
    fails = []
    
    if not path.exists():
        return False, [f"freeze_candidates.json not found at {path}"]
    
    data = load_json(path)
    
    # Check is_full_scale summary dict
    is_full_scale = data.get("is_full_scale", {})
    required_keys = [
        "offers_core_full_daily",
        "offers_core_full_snapshot", 
        "edgar_feature_store_full_daily",
        "multiscale_full",
    ]
    
    for key in required_keys:
        val = is_full_scale.get(key)
        if val is not True:
            fails.append(f"is_full_scale[{key}] = {val}, expected True")
    
    return len(fails) == 0, fails


def check_offer_day_coverage_exact(path: Path) -> Tuple[bool, List[str]]:
    """Verify offer_day_coverage_exact.json gates."""
    fails = []
    
    if not path.exists():
        return False, [f"offer_day_coverage_exact.json not found at {path}"]
    
    data = load_json(path)
    
    # gate_passed must be true
    if not data.get("gate_passed", False):
        fails.append("gate_passed = False")
    
    # coverage_rate must be 1.0
    coverage = data.get("coverage_rate", 0.0)
    if coverage < 0.9999:  # Allow tiny float tolerance
        fails.append(f"coverage_rate = {coverage}, expected 1.0")
    
    return len(fails) == 0, fails


def main():
    parser = argparse.ArgumentParser(description="Block 3 freeze verification")
    parser.add_argument("--pointer", type=Path, default=Path("docs/audits/FULL_SCALE_POINTER.yaml"))
    parser.add_argument("--expected-stamp", default="20260203_225620")
    parser.add_argument("--expected-variant", default="TRAIN_WIDE_FINAL")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Block 3 Freeze Verification")
    print("=" * 80)
    print(f"Pointer: {args.pointer}")
    print(f"Expected stamp: {args.expected_stamp}")
    print(f"Expected variant: {args.expected_variant}")
    print()
    
    # Load pointer
    if not args.pointer.exists():
        print(f"FATAL: Pointer file not found: {args.pointer}")
        sys.exit(1)
    
    pointer = load_yaml(args.pointer)
    
    # Determine output directory
    if args.output_dir is None:
        analysis_dir = pointer.get("analysis", {}).get("dir", "")
        if analysis_dir:
            output_dir = Path(analysis_dir).parent / f"block3_{args.expected_stamp}" / "verify"
        else:
            output_dir = Path(f"runs/orchestrator/20260129_073037/block3_{args.expected_stamp}/verify")
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all checks
    results = []
    all_pass = True
    
    # 1. Pointer check
    passed, fails = check_pointer(pointer, args.expected_stamp, args.expected_variant)
    results.append({"check": "pointer", "passed": passed, "fails": fails})
    if not passed:
        all_pass = False
        print(f"[FAIL] Pointer verification:")
        for f in fails:
            print(f"  - {f}")
    else:
        print("[PASS] Pointer verification")
    
    # 2. Column manifest
    cm_path = Path(pointer.get("analysis", {}).get("column_manifest", ""))
    passed, fails = check_column_manifest(cm_path)
    results.append({"check": "column_manifest", "path": str(cm_path), "passed": passed, "fails": fails})
    if not passed:
        all_pass = False
        print(f"[FAIL] Column manifest ({cm_path}):")
        for f in fails:
            print(f"  - {f}")
    else:
        print(f"[PASS] Column manifest")
    
    # 3. Raw cardinality coverage
    rc_path = Path(pointer.get("analysis", {}).get("raw_cardinality_coverage", ""))
    passed, fails = check_raw_cardinality_coverage(rc_path)
    results.append({"check": "raw_cardinality_coverage", "path": str(rc_path), "passed": passed, "fails": fails})
    if not passed:
        all_pass = False
        print(f"[FAIL] Raw cardinality coverage ({rc_path}):")
        for f in fails:
            print(f"  - {f}")
    else:
        print(f"[PASS] Raw cardinality coverage")
    
    # 4. Freeze candidates
    fc_path = Path(pointer.get("analysis", {}).get("freeze_candidates", ""))
    passed, fails = check_freeze_candidates(fc_path)
    results.append({"check": "freeze_candidates", "path": str(fc_path), "passed": passed, "fails": fails})
    if not passed:
        all_pass = False
        print(f"[FAIL] Freeze candidates ({fc_path}):")
        for f in fails:
            print(f"  - {f}")
    else:
        print(f"[PASS] Freeze candidates")
    
    # 5. Offer day coverage exact
    analysis_dir = Path(pointer.get("analysis", {}).get("dir", ""))
    od_path = analysis_dir / "offer_day_coverage_exact.json"
    passed, fails = check_offer_day_coverage_exact(od_path)
    results.append({"check": "offer_day_coverage_exact", "path": str(od_path), "passed": passed, "fails": fails})
    if not passed:
        all_pass = False
        print(f"[FAIL] Offer day coverage exact ({od_path}):")
        for f in fails:
            print(f"  - {f}")
    else:
        print(f"[PASS] Offer day coverage exact")
    
    print()
    print("=" * 80)
    
    # Build report
    report = {
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pointer_path": str(args.pointer),
        "expected_stamp": args.expected_stamp,
        "expected_variant": args.expected_variant,
        "all_gates_pass": all_pass,
        "checks": results,
    }
    
    # Write JSON
    json_path = output_dir / "verify_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    
    # Write MD
    md_lines = [
        "# Block 3 Freeze Verification Report",
        "",
        f"**Verified at:** {report['verified_at']}",
        f"**Pointer:** `{report['pointer_path']}`",
        f"**Expected stamp:** `{report['expected_stamp']}`",
        f"**Expected variant:** `{report['expected_variant']}`",
        "",
        f"## Overall Result: {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Gate Results",
        "",
        "| Check | Path | Status | Fails |",
        "|-------|------|--------|-------|",
    ]
    for r in results:
        path = r.get("path", "N/A")
        status = "PASS" if r["passed"] else "FAIL"
        fails_str = "; ".join(r["fails"][:3]) if r["fails"] else ""
        md_lines.append(f"| {r['check']} | `{path}` | **{status}** | {fails_str} |")
    
    if not all_pass:
        md_lines.extend([
            "",
            "## Suggested Fix Actions",
            "",
            "1. Do NOT modify freeze artifacts (runs/*_wide_20260203_225620)",
            "2. If data issues exist, create a new stamp via Block 2 pipeline",
            "3. Re-run freeze verification after fixes",
            "",
        ])
    else:
        md_lines.extend([
            "",
            "## Next Steps",
            "",
            "All gates PASS. Safe to proceed with Block 3 modeling.",
            "",
            "- Run data profile: `python scripts/block3_profile_data.py`",
            "- Run benchmark harness: `python scripts/run_block3_benchmark.py`",
            "",
        ])
    
    md_path = output_dir / "verify_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    
    if all_pass:
        print("\n>>> ALL GATES PASS - Safe to proceed with Block 3 <<<")
        sys.exit(0)
    else:
        print("\n>>> VERIFICATION FAILED - Do NOT proceed with Block 3 <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
