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
    "MainlineS5Trunk":            _mainline_factory("mainline_s5_trunk"),
    "MainlineS5Full":              _mainline_factory("mainline_s5_full"),
    "MainlineMlpFull":             _mainline_factory("mainline_mlp_full"),
    "MainlineS5FundingStripped":   _mainline_factory("mainline_s5_funding_stripped"),
    # Route D atom-isolation suite (on top of Route C stripped base).
    "MainlineS5FsLogOnly":         _mainline_factory("mainline_s5_fs_log_only"),
    "MainlineS5FsSourceOnly":      _mainline_factory("mainline_s5_fs_source_only"),
    "MainlineS5FsTailfocusLite":   _mainline_factory("mainline_s5_fs_tailfocus_lite"),
    "MainlineS5FsLogSource":       _mainline_factory("mainline_s5_fs_log_source"),
    # Route G forced-hurdle suite (bypasses short-circuit to actually run
    # the severity path on sparse-event cells like h=7 funding).
    "MainlineS5FsForcedBase":       _mainline_factory("mainline_s5_fs_forced_base"),
    "MainlineS5FsForcedLogOnly":    _mainline_factory("mainline_s5_fs_forced_logonly"),
    "MainlineS5FsForcedSource":     _mainline_factory("mainline_s5_fs_forced_source"),
    "MainlineS5FsForcedTailfocus":  _mainline_factory("mainline_s5_fs_forced_tailfocus"),
}
