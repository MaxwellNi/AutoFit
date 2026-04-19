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


_MAINLINE_ALPHA_OVERRIDES: Dict[str, Any] = {
    "enable_count_hurdle_head": True,
    "enable_count_jump": True,
    "count_jump_strength": 0.30,
    "enable_count_sparsity_gate": False,
}


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
        prototype_overrides=_MAINLINE_ALPHA_OVERRIDES,
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
    "mainline_funding_tailguard": MainlineVariantProfile(
        name="mainline_funding_tailguard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Funding-focused research profile that keeps the current mainline "
            "investors contract intact while turning on log-domain jump severity, "
            "tail-weighted guard calibration, and source-side residual scaling for "
            "the hard shared112 funding slices."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_funding_log_domain": True,
            "enable_funding_source_scaling": True,
            "enable_funding_tail_focus": True,
            "funding_tail_weight": 2.0,
            "funding_tail_quantile": 0.85,
        },
    ),
    "mainline_binary_calibration_guard": MainlineVariantProfile(
        name="mainline_binary_calibration_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Binary-focused research profile that preserves the current no-leak "
            "hazard path while adding a post-calibration shrinkage gate toward the "
            "hazard prior or base rate when validation reliability is worse than the "
            "unshrunk route."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_binary_calibration_shrinkage": True,
            "binary_calibration_shrinkage_target": "auto",
        },
    ),
    "mainline_event_state_boundary_guard": MainlineVariantProfile(
        name="mainline_event_state_boundary_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Investors-first serious profile that keeps binary/funding on the "
            "current guard-rail path while promoting deeper event-state boundary, "
            "flip, goal-crossing, and source-arrival atoms into the investors "
            "auxiliary path. This profile intentionally de-prioritizes the old "
            "source-read/transition family as a primary route and instead tests "
            "whether target-isolated event-state features can move the count lane "
            "without regressing the shared trunk contract."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
            "enable_investors_event_state_features": True,
        },
    ),
    "mainline_selective_event_state_guard": MainlineVariantProfile(
        name="mainline_selective_event_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-3 investors profile that keeps the official-positive "
            "guarded-jump base route, then turns on event-state features only on "
            "the slices that survived the generation-2 verdict: h>1 and source-light "
            "surfaces. This profile is intentionally selective rather than global, "
            "so event-state geometry can help dynamic, source-light investors cells "
            "without forcing regressions on source-rich official surfaces."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_funding_gpd_tail": True,
            "enable_funding_tail_focus": True,
            "enable_funding_cqr_interval": True,
            "funding_cqr_alpha": 0.10,
        },
    ),
    "mainline_marked_investor_guard": MainlineVariantProfile(
        name="mainline_marked_investor_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-4 investors profile that preserves the selective event-state "
            "guard and adds lane-private marked-investor features derived from the "
            "investor list surface, investor websites, and offering-structure proxies. "
            "The extra mark features remain inside the investors lane rather than "
            "moving subtype semantics back into the shared trunk."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_investors_mark_features": True,
        },
    ),
    "mainline_process_state_feedback_guard": MainlineVariantProfile(
        name="mainline_process_state_feedback_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-6 investors profile that preserves the selective event-state "
            "contract while adding a bounded process-state feedback block into the "
            "shared trunk. The updater stays target-agnostic, turns off on h1, and "
            "attenuates rather than disables itself on source-rich official surfaces "
            "so the trunk can move toward closure/attention state without the geometry "
            "instability seen in the aggressive temporal/spectral variants."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_process_state_feedback": True,
            "process_state_feedback_strength": 0.02,
            "process_state_feedback_source_decay": 0.65,
            "process_state_feedback_min_horizon": 7,
            "process_state_feedback_state_weights": (0.30, 0.20, 0.0, 0.0, 0.50),
        },
    ),
    "mainline_multiscale_state_guard": MainlineVariantProfile(
        name="mainline_multiscale_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-5 investors profile that preserves the selective event-state "
            "guard while upgrading the shared trunk into a multiscale temporal-state "
            "updater. The added temporal and spectral bands stay target-agnostic, so "
            "the investors lane can read a richer financing-process state without "
            "collapsing lane-private semantics back into the trunk."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_multiscale_temporal_state": True,
            "temporal_state_windows": (3, 7, 30),
            "enable_temporal_state_features": True,
            "enable_spectral_state_features": True,
        },
    ),
    "mainline_temporal_state_guard": MainlineVariantProfile(
        name="mainline_temporal_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-5a investors profile that preserves the selective event-state "
            "guard while upgrading the shared trunk with temporal-only multiscale state "
            "features. This isolates pathwise financing dynamics from spectral bands so "
            "the gate can test whether temporal process updates alone transfer cleanly."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_multiscale_temporal_state": True,
            "temporal_state_windows": (3, 7, 30),
            "enable_temporal_state_features": True,
            "enable_spectral_state_features": False,
        },
    ),
    "mainline_spectral_state_guard": MainlineVariantProfile(
        name="mainline_spectral_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-5b investors profile that preserves the selective event-state "
            "guard while upgrading the shared trunk with spectral-only multiscale state "
            "bands. This isolates frequency-style regime information from temporal delta "
            "features so the gate can test whether spectral structure transfers cleanly."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_multiscale_temporal_state": True,
            "temporal_state_windows": (3, 7, 30),
            "enable_temporal_state_features": False,
            "enable_spectral_state_features": True,
        },
    ),
    "mainline_hawkes_financing_state_guard": MainlineVariantProfile(
        name="mainline_hawkes_financing_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-8 investors profile that upgrades the shared trunk with a "
            "Hawkes-process-inspired asymmetric event-driven intensity state. "
            "Gen-8 fixes over Gen-7: (1) per-entity velocity computation eliminates "
            "cross-entity contamination, (2) intensity clipped at 10.0 and compressed "
            "via log1p to prevent unbounded accumulation, (3) z-score normalisation "
            "and ±5σ safety clip ensure Hawkes features stay on the same scale as "
            "other trunk dimensions, (4) wrapper correctly passes Hawkes params to "
            "backbone spec. "
            "Unlike the symmetric rolling-window Gen-5 variants, this accumulates "
            "only POSITIVE financing shocks with exponential decay at three timescales "
            "(7 / 30 / 90 steps), capturing the self-exciting investor-arrival and "
            "funding-momentum dynamics documented in venture capital empirical finance "
            "(Hawkes 1971; Ait-Sahalia et al. 2014; Neural Hawkes NeurIPS 2017). "
            "The selective event-state activation guard is kept unchanged so the "
            "improved trunk state is only read on h>1 and source-light surfaces."
        ),
        prototype_overrides={
            **_MAINLINE_ALPHA_OVERRIDES,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "investors_event_state_allow_h1": False,
            "investors_event_state_max_source_presence_share": 0.0,
            "enable_hawkes_financing_state": True,
            "hawkes_financing_decay_halflives": (7.0, 30.0, 90.0),
            "hawkes_positive_shock_threshold": 0.5,
        },
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