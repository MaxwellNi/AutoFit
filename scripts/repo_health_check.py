#!/usr/bin/env python3
"""
Repository Health Check Script.

Validates:
1. All Python files compile (py_compile pass)
2. Key module imports work
3. No syntax errors in tracked code

Usage:
    python scripts/repo_health_check.py
"""
from __future__ import annotations

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to check
CHECK_DIRS = ["src", "scripts"]

# Key modules to import-test
KEY_MODULES = [
    "src.narrative.data_preprocessing.block3_dataset",
    "src.narrative.block3.tasks.registry",
    "src.narrative.block3.models.registry",
    "src.narrative.auto_fit.two_stage_selector",
    "src.narrative.auto_fit.rule_based_composer",
]


def find_python_files(base_dirs: List[str]) -> List[Path]:
    """Find all Python files in given directories."""
    py_files = []
    for dir_name in base_dirs:
        dir_path = REPO_ROOT / dir_name
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                # Skip __pycache__
                if "__pycache__" not in str(py_file):
                    py_files.append(py_file)
    return py_files


def check_compile(py_files: List[Path]) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Check that all Python files compile."""
    passed = []
    failed = []
    
    for py_file in py_files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            passed.append(py_file)
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            failed.append((py_file, error_msg))
    
    return passed, failed


def check_imports(modules: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Check that key modules can be imported."""
    # Add src to path
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Also add repo root for scripts
    repo_path = str(REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    passed = []
    failed = []
    
    for module_path in modules:
        # Convert path to module name
        module_name = module_path.replace("src.", "").replace("/", ".")
        
        try:
            # Try to import
            importlib.import_module(module_name)
            passed.append(module_path)
        except Exception as e:
            failed.append((module_path, str(e)))
    
    return passed, failed


def check_forbidden_paths() -> List[Tuple[Path, int, str]]:
    """
    Check for forbidden hard-coded runs/* paths that bypass FreezePointer.
    
    Allowed patterns:
    - FreezePointer usage
    - Comments
    - Docstrings
    - "runs/" in generic config or output paths
    """
    violations = []
    
    import re
    # Pattern to find hard-coded runs paths with timestamps
    pattern = re.compile(r'runs/[a-z_]+_\d{8}_\d{6}', re.IGNORECASE)
    
    # Exceptions - files that legitimately reference runs for output
    exception_files = {
        "FreezePointer",  # The pointer class itself
        "FULL_SCALE_POINTER.yaml",
    }
    
    for dir_name in CHECK_DIRS:
        dir_path = REPO_ROOT / dir_name
        if not dir_path.exists():
            continue
            
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    
                    # Check for hard-coded timestamp paths
                    matches = pattern.findall(line)
                    for match in matches:
                        # Allow in docstrings (simple heuristic)
                        if '"""' in line or "'''" in line:
                            continue
                        violations.append((py_file, i, line.strip()))
            except Exception:
                pass
    
    return violations


def main():
    """Run all health checks."""
    print("=" * 70)
    print("Repository Health Check")
    print("=" * 70)
    
    all_passed = True
    
    # 1. Find Python files
    print("\n[1/4] Finding Python files...")
    py_files = find_python_files(CHECK_DIRS)
    print(f"  Found {len(py_files)} Python files")
    
    # 2. Check compilation
    print("\n[2/4] Checking py_compile...")
    compile_passed, compile_failed = check_compile(py_files)
    print(f"  Passed: {len(compile_passed)}")
    print(f"  Failed: {len(compile_failed)}")
    
    if compile_failed:
        all_passed = False
        print("\n  COMPILE FAILURES:")
        for py_file, error in compile_failed:
            rel_path = py_file.relative_to(REPO_ROOT)
            print(f"    - {rel_path}")
            # Print first line of error
            error_lines = error.split("\n")
            if error_lines:
                print(f"      {error_lines[-1][:80]}")
    
    # 3. Check imports
    print("\n[3/4] Checking key module imports...")
    import_passed, import_failed = check_imports(KEY_MODULES)
    print(f"  Passed: {len(import_passed)}")
    print(f"  Failed: {len(import_failed)}")
    
    if import_failed:
        all_passed = False
        print("\n  IMPORT FAILURES:")
        for module, error in import_failed:
            print(f"    - {module}")
            print(f"      {error[:80]}")
    
    # 4. Check forbidden paths
    print("\n[4/4] Checking for forbidden hard-coded paths...")
    violations = check_forbidden_paths()
    print(f"  Violations: {len(violations)}")
    
    if violations:
        # This is a warning, not a failure (legacy code may still have these)
        print("\n  WARNING - Hard-coded runs paths found (should use FreezePointer):")
        for py_file, line_no, line in violations[:10]:  # Show first 10
            rel_path = py_file.relative_to(REPO_ROOT)
            print(f"    - {rel_path}:{line_no}")
            print(f"      {line[:60]}...")
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL HEALTH CHECKS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ HEALTH CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
