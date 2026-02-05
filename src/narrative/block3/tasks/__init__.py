"""Block 3 Tasks Module."""
from .registry import get_task, list_tasks, get_task_config, TASK_REGISTRY
from .base import TaskBase, TaskConfig, SplitConfig, TaskResult
from .task1_outcome import Task1OutcomePrediction
from .task2_forecast import Task2TrajectoryForecasting
from .task3_risk_adjust import Task3RiskAdjustment
from .task4_narrative_shift import Task4NarrativeShift

__all__ = [
    "get_task",
    "list_tasks",
    "get_task_config",
    "TASK_REGISTRY",
    "TaskBase",
    "TaskConfig",
    "SplitConfig",
    "TaskResult",
    "Task1OutcomePrediction",
    "Task2TrajectoryForecasting",
    "Task3RiskAdjustment",
    "Task4NarrativeShift",
]
