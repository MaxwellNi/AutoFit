#!/usr/bin/env python3
"""Unit tests for investors_count stability audit helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_investors_count_stability.py"
    spec = importlib.util.spec_from_file_location("audit_investors_count_stability", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_ks_zero_for_identical_arrays():
    module = _load_module()
    arr = [1.0, 2.0, 3.0, 4.0]
    assert module._ks_stat(arr, arr) == 0.0


def test_psi_non_negative():
    module = _load_module()
    train = [1.0, 2.0, 3.0, 4.0, 5.0]
    other = [1.1, 2.1, 3.1, 4.1, 5.1]
    assert module._psi(train, other) >= 0.0


def test_guard_telemetry_empty():
    module = _load_module()
    out = module._guard_telemetry([])
    assert out["n_rows"] == 0
    assert out["oof_guard_triggered_count"] == 0
