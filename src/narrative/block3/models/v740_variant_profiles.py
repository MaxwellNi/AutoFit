#!/usr/bin/env python3
"""Common local variant profiles for the V740 pre-benchmark research line.

These helpers centralize the V741-V745 boolean overrides so local scripts do
not re-encode the same architecture regimes in multiple places.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


LOCAL_V740_VARIANTS: Tuple[str, ...] = (
    "v740_alpha",
    "v741_lite",
    "v742_unified",
    "v743_factorized",
    "v744_guarded_phase",
    "v745_evidence_residual",
)


def is_local_v740_variant(name: str) -> bool:
    return name in LOCAL_V740_VARIANTS


def resolve_requested_v740_variant(token: str, flags: Mapping[str, Any]) -> str:
    if token != "v740_alpha":
        return token
    if bool(flags.get("enable_v745_evidence_residual", False)):
        return "v745_evidence_residual"
    if bool(flags.get("enable_v744_guarded_phase", False)):
        return "v744_guarded_phase"
    if bool(flags.get("enable_v743_factorized", False)):
        return "v743_factorized"
    if bool(flags.get("enable_v742_unified", False)):
        return "v742_unified"
    if bool(flags.get("enable_v741_lite", False)):
        return "v741_lite"
    return token


def apply_v740_variant_profile(variant_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(kwargs)
    if variant_name == "v740_alpha":
        return profile
    if variant_name == "v741_lite":
        profile["enable_v741_lite"] = True
        return profile
    if variant_name == "v742_unified":
        profile.update(
            {
                "enable_financing_consistency": True,
                "enable_target_routing": False,
                "enable_count_source_routing": False,
                "enable_count_source_specialists": False,
                "enable_count_hurdle_head": False,
                "enable_window_repair": True,
            }
        )
        return profile
    if variant_name == "v743_factorized":
        profile.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "enable_target_routing": False,
                "enable_count_source_routing": False,
                "enable_count_source_specialists": False,
                "enable_count_hurdle_head": False,
                "enable_window_repair": True,
            }
        )
        return profile
    if variant_name == "v744_guarded_phase":
        profile.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "enable_financing_guarded_phase": True,
                "enable_window_repair": True,
            }
        )
        return profile
    if variant_name == "v745_evidence_residual":
        profile.update(
            {
                "enable_financing_consistency": True,
                "enable_financing_factorization": True,
                "enable_financing_evidence_residual": True,
                "enable_window_repair": True,
            }
        )
        return profile
    raise ValueError(f"Unsupported local V740 variant: {variant_name}")


__all__ = [
    "LOCAL_V740_VARIANTS",
    "apply_v740_variant_profile",
    "is_local_v740_variant",
    "resolve_requested_v740_variant",
]