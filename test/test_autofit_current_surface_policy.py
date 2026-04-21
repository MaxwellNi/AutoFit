#!/usr/bin/env python3
"""Regression tests for the current active AutoFit surface policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narrative.block3.autofit_status import (  # noqa: E402
    CURRENT_AUTOFIT_BASELINE,
    RETIRED_AUTOFIT_MODEL_NAMES,
    get_retired_autofit_reason,
    is_current_autofit_model,
    is_retired_autofit_model,
)
from narrative.block3.models.autofit_wrapper import (  # noqa: E402
    ACTIVE_AUTOFIT_MODELS,
    RETIRED_AUTOFIT_MODELS,
)
from narrative.block3.models.registry import get_model, list_models  # noqa: E402
from scripts.aggregate_block3_results import apply_comparability_filter  # noqa: E402


def test_current_autofit_baseline_is_v739():
    assert CURRENT_AUTOFIT_BASELINE == "AutoFitV739"
    assert is_current_autofit_model("AutoFitV739") is True
    assert is_retired_autofit_model("AutoFitV739") is False


def test_active_autofit_surface_is_v739_only():
    assert list_models()["autofit"] == ["AutoFitV739"]
    assert list(ACTIVE_AUTOFIT_MODELS.keys()) == ["AutoFitV739"]
    assert "AutoFitV739" not in RETIRED_AUTOFIT_MODELS
    assert "AutoFitV73" in RETIRED_AUTOFIT_MODELS
    assert "FusedChampion" in RETIRED_AUTOFIT_MODELS
    assert "AutoFitV734" in RETIRED_AUTOFIT_MODEL_NAMES


@pytest.mark.parametrize(
    "model_name",
    ["AutoFitV73", "FusedChampion", "NFAdaptiveChampion", "AutoFitV734"],
)
def test_retired_autofit_family_models_are_blocked_from_current_registry(model_name: str):
    with pytest.raises(ValueError, match="Retired AutoFit model blocked from current registry") as exc_info:
        get_model(model_name)

    assert get_retired_autofit_reason(model_name) in str(exc_info.value)


def test_aggregate_filter_purges_retired_autofit_rows_from_current_surface():
    frame = pd.DataFrame(
        [
            {
                "model_name": "AutoFitV739",
                "fairness_pass": True,
                "prediction_coverage_ratio": 1.0,
            },
            {
                "model_name": "AutoFitV73",
                "fairness_pass": True,
                "prediction_coverage_ratio": 1.0,
            },
            {
                "model_name": "FusedChampion",
                "fairness_pass": True,
                "prediction_coverage_ratio": 1.0,
            },
        ]
    )

    filtered = apply_comparability_filter(frame)

    assert filtered["model_name"].tolist() == ["AutoFitV739"]