#!/usr/bin/env python3
"""
Test: Auto-Fit Module Imports.

Validates that all auto_fit modules can be imported without errors.
"""
import sys
from pathlib import Path

import pytest

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_import_budget_search():
    """budget_search module should import."""
    from narrative.auto_fit import budget_search
    assert hasattr(budget_search, "SearchResult")
    assert hasattr(budget_search, "successive_halving")


def test_import_compose_candidates():
    """compose_candidates module should import."""
    from narrative.auto_fit import compose_candidates
    # May have limited exports depending on implementation


def test_import_diagnose_dataset():
    """diagnose_dataset module should import."""
    from narrative.auto_fit import diagnose_dataset
    # Check for key functions/classes


def test_import_leaderboard():
    """leaderboard module should import."""
    from narrative.auto_fit import leaderboard


def test_import_rule_based_composer():
    """rule_based_composer module should import."""
    from narrative.auto_fit import rule_based_composer
    assert hasattr(rule_based_composer, "RuleBasedComposer")


def test_import_two_stage_selector():
    """two_stage_selector module should import."""
    from narrative.auto_fit import two_stage_selector
    assert hasattr(two_stage_selector, "TwoStageSelector")


def test_import_auto_fit_init():
    """auto_fit __init__ should import cleanly."""
    from narrative import auto_fit
    # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
