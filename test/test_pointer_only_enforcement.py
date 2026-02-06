#!/usr/bin/env python3
"""
Test: Pointer-Only Enforcement.

Validates that Block 3 code uses only FreezePointer for data access,
not hard-coded runs/*_STAMP paths.
"""
import re
import sys
from pathlib import Path

import pytest

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan
SCAN_DIRS = ["src/narrative/block3", "src/narrative/data_preprocessing"]

# Pattern for hard-coded timestamp paths
HARDCODED_PATH_PATTERN = re.compile(r'["\']runs/[a-z_]+_\d{8}_\d{6}', re.IGNORECASE)

# Allowed exceptions (file patterns that may legitimately use runs paths)
ALLOWED_FILES = {
    "block3_dataset.py",  # The dataset class uses FreezePointer internally
    "freeze_pointer.py",  # The pointer class itself
}


def find_violations() -> list:
    """Find files with hard-coded runs paths."""
    violations = []
    
    for dir_name in SCAN_DIRS:
        dir_path = REPO_ROOT / dir_name
        if not dir_path.exists():
            continue
            
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            # Skip allowed files
            if py_file.name in ALLOWED_FILES:
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    
                    # Check for hard-coded paths
                    if HARDCODED_PATH_PATTERN.search(line):
                        violations.append({
                            "file": str(py_file.relative_to(REPO_ROOT)),
                            "line": i,
                            "content": line.strip()[:80],
                        })
            except Exception as e:
                pass
    
    return violations


def test_no_hardcoded_runs_paths():
    """Block 3 code should not have hard-coded runs/*_STAMP paths."""
    violations = find_violations()
    
    if violations:
        msg = "Found hard-coded runs paths (should use FreezePointer):\n"
        for v in violations[:10]:
            msg += f"  - {v['file']}:{v['line']}: {v['content']}\n"
        if len(violations) > 10:
            msg += f"  ... and {len(violations) - 10} more\n"
        pytest.fail(msg)


def test_freeze_pointer_exists():
    """FreezePointer class should exist and be importable."""
    try:
        from narrative.data_preprocessing.block3_dataset import FreezePointer
        assert FreezePointer is not None
    except ImportError:
        pytest.fail("FreezePointer class not importable")


def test_freeze_pointer_yaml_exists():
    """FULL_SCALE_POINTER.yaml should exist."""
    pointer_path = REPO_ROOT / "docs/audits/FULL_SCALE_POINTER.yaml"
    assert pointer_path.exists(), f"Pointer YAML not found: {pointer_path}"


def test_freeze_pointer_has_required_keys():
    """FULL_SCALE_POINTER.yaml should have required keys."""
    import yaml
    
    pointer_path = REPO_ROOT / "docs/audits/FULL_SCALE_POINTER.yaml"
    
    with open(pointer_path, "r", encoding="utf-8") as f:
        pointer = yaml.safe_load(f)
    
    # Check for stamp (required)
    assert "stamp" in pointer, "Missing required key: stamp"
    
    # Check for required data directories
    required_data_keys = ["offers_core_daily", "offers_text", "edgar_store_full_daily"]
    for key in required_data_keys:
        assert key in pointer, f"Missing required data key: {key}"
        assert "dir" in pointer[key], f"Missing 'dir' in {key}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
