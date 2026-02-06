#!/usr/bin/env python3
"""
Test: Task Registry.

Validates that task registry works correctly and all tasks have proper descriptions.
"""
import sys
from pathlib import Path

import pytest

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_list_tasks_returns_all_four():
    """list_tasks should return Task1-4."""
    from narrative.block3.tasks.registry import list_tasks
    
    tasks = list_tasks()
    
    assert isinstance(tasks, dict)
    assert len(tasks) >= 4, f"Expected at least 4 tasks, got {len(tasks)}"
    
    expected_tasks = [
        "task1_outcome",
        "task2_forecast",
        "task3_risk_adjust",
        "task4_narrative_shift",
    ]
    
    for task_name in expected_tasks:
        assert task_name in tasks, f"Missing task: {task_name}"


def test_all_tasks_have_descriptions():
    """All tasks should have non-empty descriptions."""
    from narrative.block3.tasks.registry import list_tasks
    
    tasks = list_tasks()
    
    for task_name, description in tasks.items():
        assert description, f"Task {task_name} has empty description"
        assert len(description) > 10, f"Task {task_name} has very short description"


def test_get_task_works_for_all():
    """get_task should work for all registered tasks."""
    from narrative.block3.tasks.registry import get_task, list_tasks
    
    for task_name in list_tasks().keys():
        task = get_task(task_name)
        assert task is not None, f"get_task returned None for {task_name}"
        assert hasattr(task, "config"), f"Task {task_name} missing config"
        assert hasattr(task, "build_dataset"), f"Task {task_name} missing build_dataset"


def test_task_aliases_work():
    """Task aliases should resolve correctly."""
    from narrative.block3.tasks.registry import get_task
    
    # Test common aliases
    aliases = {
        "outcome": "task1_outcome",
        "forecast": "task2_forecast",
        "risk": "task3_risk_adjust",
        "narrative": "task4_narrative_shift",
    }
    
    for alias, expected_name in aliases.items():
        task = get_task(alias)
        assert task.config.name == expected_name, f"Alias {alias} resolved incorrectly"


def test_task_configs_have_required_fields():
    """Task configs should have required fields."""
    from narrative.block3.tasks.registry import get_task, list_tasks
    
    required_fields = ["name", "targets", "horizons", "ablations", "metrics"]
    
    for task_name in list_tasks().keys():
        task = get_task(task_name)
        config = task.config
        
        for field in required_fields:
            assert hasattr(config, field), f"Task {task_name} config missing {field}"
            value = getattr(config, field)
            # Most fields should be non-empty lists or values
            if field in ["targets", "ablations", "metrics"]:
                assert value, f"Task {task_name} has empty {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
