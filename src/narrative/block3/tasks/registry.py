#!/usr/bin/env python3
"""
Task Registry for Block 3.

Provides unified access to all tasks via get_task(name).
"""
from __future__ import annotations

from typing import Dict, Optional, Type

from .base import TaskBase, TaskConfig
from .task1_outcome import Task1OutcomePrediction, create_task1_config
from .task2_forecast import Task2TrajectoryForecasting, create_task2_config
from .task3_risk_adjust import Task3RiskAdjustment, create_task3_config
from .task4_narrative_shift import Task4NarrativeShift, create_task4_config


# Registry of task classes and their default configs
TASK_REGISTRY: Dict[str, tuple[Type[TaskBase], callable]] = {
    "task1_outcome": (Task1OutcomePrediction, create_task1_config),
    "task2_forecast": (Task2TrajectoryForecasting, create_task2_config),
    "task3_risk_adjust": (Task3RiskAdjustment, create_task3_config),
    "task4_narrative_shift": (Task4NarrativeShift, create_task4_config),
}

# Aliases
TASK_ALIASES = {
    "outcome": "task1_outcome",
    "forecast": "task2_forecast",
    "risk": "task3_risk_adjust",
    "narrative": "task4_narrative_shift",
    "did": "task4_narrative_shift",
}


def get_task(
    name: str,
    config: Optional[TaskConfig] = None,
    dataset=None,
) -> TaskBase:
    """
    Get a task instance by name.
    
    Args:
        name: Task name or alias
        config: Optional custom config (uses default if None)
        dataset: Optional Block3Dataset instance
    
    Returns:
        TaskBase instance
    """
    # Resolve alias
    resolved_name = TASK_ALIASES.get(name, name)
    
    if resolved_name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys()) + list(TASK_ALIASES.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    
    task_class, config_fn = TASK_REGISTRY[resolved_name]
    
    if config is None:
        config = config_fn()
    
    return task_class(config, dataset)


def list_tasks() -> Dict[str, str]:
    """List all available tasks with descriptions."""
    return {
        "task1_outcome": "Outcome Prediction (classification + regression + ranking)",
        "task2_forecast": "Trajectory Forecasting (multi-horizon + quantiles)",
        "task3_risk_adjust": "EDGAR-conditioned Risk Adjustment (robustness + shift)",
        "task4_narrative_shift": "Narrative Shift & GenAI Era Effect (DiD + mediation)",
    }


def get_task_config(name: str, **kwargs) -> TaskConfig:
    """Get task config with optional overrides."""
    resolved_name = TASK_ALIASES.get(name, name)
    
    if resolved_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'")
    
    _, config_fn = TASK_REGISTRY[resolved_name]
    return config_fn(**kwargs)


__all__ = [
    "get_task",
    "list_tasks",
    "get_task_config",
    "TASK_REGISTRY",
    "Task1OutcomePrediction",
    "Task2TrajectoryForecasting",
    "Task3RiskAdjustment",
    "Task4NarrativeShift",
]
