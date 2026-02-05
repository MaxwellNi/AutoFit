#!/usr/bin/env python3
"""
Block 3 Pointer-Only Enforcement Test.

This test ensures that Block3Dataset and benchmark code resolve all paths
ONLY via FreezePointer and do not contain hard-coded runs/* paths.

Hard Rule: No string literals like "runs/edgar_feature_store_full_daily_wide_"
or "runs/offers_core_full_daily_wide_" should appear in production code.
"""
import ast
import re
import sys
from pathlib import Path


# Patterns that indicate hard-coded paths (should NOT appear)
FORBIDDEN_PATTERNS = [
    r"runs/offers_core_full_daily_wide_\d+",
    r"runs/offers_core_full_snapshot_wide_\d+",
    r"runs/edgar_feature_store_full_daily_wide_\d+",
    r"runs/multiscale_full_wide_\d+",
    # Allow stamp in comments/docstrings but not in actual code strings
]

# Files to check for pointer-only compliance
PRODUCTION_FILES = [
    "src/narrative/data_preprocessing/block3_dataset.py",
    "src/narrative/auto_fit/rule_based_composer.py",
    "src/narrative/explainability/concept_bottleneck.py",
]

# Files that may reference paths (must use FreezePointer)
BENCHMARK_FILES = [
    "scripts/run_block3_benchmark.py",
    "scripts/block3_profile_data.py",
]


class PathViolationError(Exception):
    """Raised when hard-coded paths are found."""
    pass


def extract_string_literals(source_code: str) -> list[tuple[int, str]]:
    """Extract all string literals from Python source with line numbers."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append((getattr(node, 'lineno', 0), node.value))
        # Also check JoinedStr (f-strings) - harder to extract but check raw source
    
    return strings


def check_file_for_hardcoded_paths(filepath: Path) -> list[tuple[int, str, str]]:
    """
    Check a file for hard-coded paths.
    
    Returns list of (line_number, matched_pattern, line_content).
    """
    violations = []
    
    if not filepath.exists():
        return violations
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    
    # First pass: check raw lines (catches f-strings and comments)
    for i, line in enumerate(lines, start=1):
        # Skip comment-only lines and docstrings for path checks
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                # Check if it's in a string literal (not just a comment)
                # If it's in the actual code string, it's a violation
                if not stripped.startswith("#"):
                    violations.append((i, pattern, line.strip()))
    
    # Second pass: extract string literals and check
    strings = extract_string_literals(content)
    for lineno, s in strings:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, s):
                violations.append((lineno, pattern, s[:100]))
    
    return violations


def check_freeze_pointer_usage(filepath: Path) -> bool:
    """
    Check if a file correctly imports and uses FreezePointer.
    
    Returns True if the file uses FreezePointer for path resolution.
    """
    if not filepath.exists():
        return True  # Skip missing files
    
    content = filepath.read_text(encoding="utf-8")
    
    # Check for FreezePointer import or usage
    has_pointer = (
        "FreezePointer" in content or
        "from_pointer" in content or
        "FULL_SCALE_POINTER" in content
    )
    
    return has_pointer


def test_no_hardcoded_paths():
    """Test that production code has no hard-coded runs paths."""
    repo_root = Path(__file__).parent.parent.parent
    
    all_violations = []
    
    for relpath in PRODUCTION_FILES + BENCHMARK_FILES:
        filepath = repo_root / relpath
        violations = check_file_for_hardcoded_paths(filepath)
        
        for lineno, pattern, content in violations:
            all_violations.append(f"{relpath}:{lineno} - matches '{pattern}': {content}")
    
    if all_violations:
        msg = "Hard-coded paths found (violates pointer-only rule):\n"
        msg += "\n".join(f"  - {v}" for v in all_violations)
        raise PathViolationError(msg)


def test_freeze_pointer_used():
    """Test that key files use FreezePointer for path resolution."""
    repo_root = Path(__file__).parent.parent.parent
    
    required_files = [
        "src/narrative/data_preprocessing/block3_dataset.py",
    ]
    
    for relpath in required_files:
        filepath = repo_root / relpath
        if not check_freeze_pointer_usage(filepath):
            raise PathViolationError(
                f"{relpath} does not use FreezePointer for path resolution"
            )


def test_pointer_yaml_exists():
    """Test that the pointer YAML exists and is valid."""
    repo_root = Path(__file__).parent.parent.parent
    pointer_path = repo_root / "docs/audits/FULL_SCALE_POINTER.yaml"
    
    assert pointer_path.exists(), "FULL_SCALE_POINTER.yaml not found"
    
    import yaml
    data = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    
    required_keys = [
        "stamp", "variant", "offers_core_daily", "offers_text",
        "edgar_store_full_daily", "analysis", "snapshots_index"
    ]
    
    for key in required_keys:
        assert key in data, f"Missing required key in pointer: {key}"


def test_block3_dataset_pointer_resolution():
    """Test that Block3Dataset resolves paths via pointer."""
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    
    from narrative.data_preprocessing.block3_dataset import Block3Dataset, FreezePointer
    
    # Load pointer
    pointer = FreezePointer.load(repo_root / "docs/audits/FULL_SCALE_POINTER.yaml")
    
    # Verify pointer fields are set
    assert pointer.stamp is not None
    assert pointer.offers_core_daily_dir is not None
    assert pointer.edgar_store_dir is not None
    
    # Verify Block3Dataset uses pointer
    ds = Block3Dataset.from_pointer(repo_root / "docs/audits/FULL_SCALE_POINTER.yaml")
    assert ds.pointer.stamp == pointer.stamp


def run_all_tests():
    """Run all pointer-only compliance tests."""
    tests = [
        ("no_hardcoded_paths", test_no_hardcoded_paths),
        ("freeze_pointer_used", test_freeze_pointer_used),
        ("pointer_yaml_exists", test_pointer_yaml_exists),
        ("block3_dataset_pointer_resolution", test_block3_dataset_pointer_resolution),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("Block 3 Pointer-Only Compliance Tests")
    print("=" * 60)
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}")
            print(f"       {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
