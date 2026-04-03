#!/usr/bin/env python3
"""Current-status helpers for Block 3 AutoFit lines.

This module separates the only current valid AutoFit baseline from all
historical or invalid AutoFit-family lines. The archived names still exist in
source and raw benchmark artifacts for auditability, but they must not remain
in the active registry or current public leaderboard surface.
"""

from __future__ import annotations

from typing import Dict


CURRENT_AUTOFIT_BASELINE = "AutoFitV739"

# Historical and invalid AutoFit-family lines that must stay out of the active
# registry and current public leaderboard surface.
RETIRED_AUTOFIT_REASONS: Dict[str, str] = {
    "AutoFitV1": "historical early stacking iteration, retired from current environment",
    "AutoFitV2": "historical early ensemble iteration, retired from current environment",
    "AutoFitV2E": "historical early stacking iteration, retired from current environment",
    "AutoFitV3": "historical temporal-CV ensemble iteration, retired from current environment",
    "AutoFitV3E": "historical temporal-CV stacking iteration, retired from current environment",
    "AutoFitV3Max": "historical exhaustive subset-search iteration, retired from current environment",
    "AutoFitV4": "historical full-OOF stacking iteration, retired from current environment",
    "AutoFitV5": "historical quick-screen ensemble iteration, retired from current environment",
    "AutoFitV6": "historical constrained ensemble iteration, retired from current environment",
    "AutoFitV7": "historical ratio-feature ensemble iteration, retired from current environment",
    "AutoFitV71": "historical lane-adaptive research line, retired after later clean baselines landed",
    "AutoFitV72": "historical strict-key research line, retired after later clean baselines landed",
    "AutoFitV73": "historical multi-agent/oracle research line, retired after later clean baselines landed",
    "AutoFitV731": "historical champion-transfer research line, retired after later clean baselines landed",
    "AutoFitV732": "historical structural-oracle router, retired from current environment",
    "AutoFitV733": "historical NF-native adaptive champion alias, superseded by the clean V739 line",
    "FusedChampion": "historical V7.3.2 fused prototype, retired after root-cause audit",
    "NFAdaptiveChampion": "historical pre-V739 adaptive champion alias, superseded by the clean V739 line",
    "AutoFitV734": "INVALID — oracle test-set leakage",
    "AutoFitV735": "INVALID — oracle test-set leakage",
    "AutoFitV736": "INVALID — oracle test-set leakage",
    "AutoFitV737": "INVALID — oracle test-set leakage",
    "AutoFitV738": "INVALID — oracle test-set leakage",
}

RETIRED_AUTOFIT_MODEL_NAMES = tuple(sorted(RETIRED_AUTOFIT_REASONS))


def is_current_autofit_model(name: str) -> bool:
    return str(name) == CURRENT_AUTOFIT_BASELINE


def is_retired_autofit_model(name: str) -> bool:
    return str(name) in RETIRED_AUTOFIT_REASONS


def get_retired_autofit_reason(name: str) -> str:
    return RETIRED_AUTOFIT_REASONS.get(
        str(name),
        "not marked as a retired AutoFit-family model",
    )
