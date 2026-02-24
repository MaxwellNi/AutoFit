#!/usr/bin/env python3
"""Tests for Block3 execution contract assertion logic."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "assert_block3_execution_contract.py"
    spec = importlib.util.spec_from_file_location("assert_block3_execution_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_audit_passes_core_doc_checks():
    module = _load_module()
    repo_root = Path(__file__).resolve().parent.parent
    audit = module.build_audit(
        repo_root=repo_root,
        entrypoint="unit-test",
        require_insider=False,
    )
    assert audit.contract_version_match is True
    assert audit.policy_hash_match is True
    assert audit.policy_text_present is True
    assert audit.required_sections_present is True


def test_forbidden_flag_fails_contract(monkeypatch):
    module = _load_module()
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("BLOCK3_ALLOW_TEST_FEEDBACK", "1")
    audit = module.build_audit(
        repo_root=repo_root,
        entrypoint="unit-test",
        require_insider=False,
    )
    assert audit.no_forbidden_flags is False
    assert "BLOCK3_ALLOW_TEST_FEEDBACK" in audit.forbidden_env_flags
    assert audit.pass_all is False
