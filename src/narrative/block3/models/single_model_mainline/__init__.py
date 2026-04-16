#!/usr/bin/env python3
"""Single-model mainline scaffold for the target-isolated research direction.

This package intentionally stays outside the active benchmark registry for now.
It now exposes a native local runtime for the shared-trunk plus hard
target-isolated-process mainline, while keeping audited delegate fallbacks for
compatibility checks.
"""
from .backbone import SharedTemporalBackbone, SharedTemporalBackboneSpec
from .barrier import HardTargetBarrierSpec, TargetIsolatedBarrier
from .conditioning import ConditionKey, ConditioningSchema, MainlineConditionEncoder
from .investor_mark_encoder import InvestorMarkEncoder, InvestorMarkEncoderSpec, canonicalize_investor_reference
from .lanes import BinaryLaneRuntime, BinaryLaneSpec, FundingLaneRuntime, FundingLaneSpec, InvestorsLaneRuntime, InvestorsLaneSpec
from .objectives import MainlineObjectiveSpec
from .source_memory import SourceColumnLayout, SourceMemoryAssembler, SourceMemoryBatch, SourceMemoryContract
from .variant_profiles import MAINLINE_VARIANTS, MainlineVariantProfile, build_delegate_kwargs, get_mainline_variant_profile
from .wrapper import MainlineModuleContract, SingleModelMainlineWrapper

__all__ = [
    "BinaryLaneRuntime",
    "BinaryLaneSpec",
    "ConditionKey",
    "ConditioningSchema",
    "FundingLaneRuntime",
    "FundingLaneSpec",
    "HardTargetBarrierSpec",
    "InvestorMarkEncoder",
    "InvestorMarkEncoderSpec",
    "InvestorsLaneRuntime",
    "InvestorsLaneSpec",
    "MAINLINE_VARIANTS",
    "MainlineConditionEncoder",
    "MainlineModuleContract",
    "MainlineObjectiveSpec",
    "MainlineVariantProfile",
    "SharedTemporalBackbone",
    "SharedTemporalBackboneSpec",
    "SingleModelMainlineWrapper",
    "SourceColumnLayout",
    "SourceMemoryAssembler",
    "SourceMemoryBatch",
    "SourceMemoryContract",
    "TargetIsolatedBarrier",
    "build_delegate_kwargs",
    "canonicalize_investor_reference",
    "get_mainline_variant_profile",
]