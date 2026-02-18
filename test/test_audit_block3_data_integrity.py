#!/usr/bin/env python3
"""Unit tests for data integrity audit helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_block3_data_integrity.py"
    spec = importlib.util.spec_from_file_location("audit_block3_data_integrity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_resolve_path_under_repo():
    module = _load_module()
    p = module._resolve_path("configs/block3_tasks.yaml")
    assert p.exists()
    assert p.name == "block3_tasks.yaml"


def test_asset_fingerprint_missing_path():
    module = _load_module()
    out = module._asset_fingerprint(Path("/tmp/path_that_does_not_exist_123456"))
    assert out["exists"] is False
    assert out["file_count"] == 0
