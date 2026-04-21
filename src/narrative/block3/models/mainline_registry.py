#!/usr/bin/env python3
"""Mainline category registration for the benchmark harness.

Exposes `SingleModelMainlineWrapper` under a small set of named variants so the
standard `scripts/run_block3_benchmark_shard.py` harness can treat it exactly
like any other registered model. This is the *total-gate* change that unlocks
head-to-head 112-cell comparisons against NBEATS / PatchTST / NHITS etc.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import ModelBase
from .single_model_mainline.wrapper import SingleModelMainlineWrapper


def _mainline_factory(variant: str):
    def _build(**kwargs: Any) -> ModelBase:
        return SingleModelMainlineWrapper(variant=variant, **kwargs)

    _build.__name__ = f"get_mainline_{variant}"
    return _build


# Only register variants that exist in MAINLINE_VARIANTS (see
# single_model_mainline/variant_profiles.py). Names surface in the public
# registry exactly as written here.
MAINLINE_MODELS: Dict[str, Any] = {
    "Mainline":                   _mainline_factory("mainline_alpha"),
    "MainlineFundingTailGuard":   _mainline_factory("mainline_funding_tailguard"),
    "MainlineBinaryCalibGuard":   _mainline_factory("mainline_binary_calibration_guard"),
    "MainlineDelegateV740Alpha":  _mainline_factory("mainline_delegate_alpha"),
    "MainlineTrunkNaked":         _mainline_factory("mainline_trunk_naked"),
}
