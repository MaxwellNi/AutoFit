#!/usr/bin/env python3
"""Mainline profiles for the target-isolated single-model scaffold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from ..v740_variant_profiles import apply_v740_variant_profile


@dataclass(frozen=True)
class MainlineVariantProfile:
    name: str
    runtime_mode: str
    delegate_variant: str | None
    description: str
    prototype_overrides: Mapping[str, Any] = field(default_factory=dict)


MAINLINE_VARIANTS: Dict[str, MainlineVariantProfile] = {
    "mainline_alpha": MainlineVariantProfile(
        name="mainline_alpha",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "First target-isolated mainline runtime. The shared trunk, sparse source "
            "memory, hard barrier, and lane estimators execute inside the mainline "
            "package while the audited V740-alpha prototype remains available as a "
            "fallback reference. The current default investors path promotes the "
            "guarded_jump process on h>1 without changing the h1 contract."
        ),
        prototype_overrides={
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
        },
    ),
    "mainline_delegate_alpha": MainlineVariantProfile(
        name="mainline_delegate_alpha",
        runtime_mode="delegate",
        delegate_variant="v740_alpha",
        description=(
            "Compatibility profile that preserves the previous audited V740-alpha "
            "delegate path for side-by-side local comparison."
        ),
        prototype_overrides={},
    ),
}


def list_mainline_variants() -> Tuple[str, ...]:
    return tuple(MAINLINE_VARIANTS.keys())


def get_mainline_variant_profile(name: str) -> MainlineVariantProfile:
    if name not in MAINLINE_VARIANTS:
        raise ValueError(f"Unsupported single-model mainline variant: {name}")
    return MAINLINE_VARIANTS[name]


def build_delegate_kwargs(name: str, extra_kwargs: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    profile = get_mainline_variant_profile(name)
    if profile.delegate_variant is None:
        raise ValueError(f"Mainline variant {name} does not define a delegate runtime")
    delegate_kwargs = apply_v740_variant_profile(profile.delegate_variant, {})
    delegate_kwargs.update(dict(profile.prototype_overrides))
    if extra_kwargs:
        delegate_kwargs.update(dict(extra_kwargs))
    return delegate_kwargs


__all__ = [
    "MAINLINE_VARIANTS",
    "MainlineVariantProfile",
    "build_delegate_kwargs",
    "get_mainline_variant_profile",
    "list_mainline_variants",
]