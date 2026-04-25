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
            "enable_investors_intensity_baseline": True,
            "investors_intensity_blend": 0.5,
        },
    ),
    "mainline_shrinkage_gate_guard": MainlineVariantProfile(
        name="mainline_shrinkage_gate_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-10 (P5) investors profile that adds an adaptive shrinkage "
            "gate on top of the selective event-state guard. Instead of expanding "
            "the trunk (which failed twice: Hawkes -173%, Jump ODE -1.04%), this "
            "profile operates entirely inside the investors lane. The shrinkage "
            "gate learns per-sample optimal alpha to blend between learned "
            "prediction and anchor, reducing MSE when uncertainty is high. "
            "Inspired by James-Stein / ASH / Nash adaptive shrinkage."
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
            "enable_investors_intensity_baseline": True,
            "investors_intensity_blend": 0.5,
            "enable_investors_shrinkage_gate": True,
            "investors_shrinkage_strength": 0.8,
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
    "mainline_jump_ode_state_guard": MainlineVariantProfile(
        name="mainline_jump_ode_state_guard",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "Generation-9 investors profile that upgrades the shared trunk with a "
            "Jump ODE state evolution module (Jia & Benson, ICML 2020). "
            "Between financing events the compact state evolves via a learned linear "
            "drift (Euler discretisation); at event boundaries a mark-dependent jump "
            "correction resets the state trajectory. The resulting ODE state adds "
            "continuous-time dynamics and jump energy diagnostics to the trunk "
            "representation. This preserves the selective event-state guard, the "
            "P2 GPD tail correction, the P2.1 CQR interval, and the P3 intensity "
            "baseline while giving the investors lane access to a richer process "
            "state without requiring torchdiffeq or GPU training."
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
            "enable_investors_intensity_baseline": True,
            "investors_intensity_blend": 0.5,
            "enable_jump_ode_state": True,
            "jump_ode_dims": 8,
        },
    ),
    "mainline_trunk_naked": MainlineVariantProfile(
        name="mainline_trunk_naked",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "2026-04-21 ablation baseline: disable ALL atoms (Hawkes, GPD, CQR, "
            "count jump/hurdle, funding log-domain/source-scaling/tail-focus, "
            "investors event-state/intensity/shrinkage) so only the shared trunk "
            "+ lane ridge readout (funding_lane trunk-fallback path) + hazard MLP "
            "stays. Purpose: measure raw trunk signal quality without any atom "
            "polluting the result. This is the ONLY scientifically valid baseline "
            "for 'selective event-state trunk 路线是否有效' before atom ablation."
        ),
        prototype_overrides={
            # Atoms OFF — trunk + minimal readout only.
            "enable_count_hurdle_head": False,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": False,
            "enable_funding_anchor": True,  # keep anchor as baseline target shift
            "enable_funding_log_domain": False,
            "enable_funding_source_scaling": False,
            "enable_funding_tail_focus": False,
            "enable_funding_gpd_tail": False,
            "enable_funding_cqr_interval": False,
            "enable_log1p_target": True,  # keep: pure numerical stability, not an atom
            "enable_hawkes_financing_state": False,
            "enable_investors_event_state_features": False,
            "enable_investors_selective_event_state_activation": False,
            "enable_investors_intensity_baseline": False,
            "enable_investors_shrinkage_gate": False,
            "enable_jump_ode_state": False,
        },
    ),
    "mainline_s5_trunk": MainlineVariantProfile(
        name="mainline_s5_trunk",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "2026-04-22 path-3 trunk swap: replaces the Learnable Sparse MoE "
            "trunk with the S5-style diagonal state-space trunk "
            "(src/narrative/block3/models/single_model_mainline/state_space_trunk.py), "
            "motivated by the 2026-04-21 mainline_trunk_collapse_verdict "
            "(7/7 triples failed the admissibility gate with identical-MAE-across-h "
            "collapse signature on the MoE trunk). All atoms are disabled (same "
            "minimal baseline as mainline_trunk_naked) so that this variant "
            "measures the signal quality of the S5 trunk alone."
        ),
        prototype_overrides={
            # All atoms OFF (same as mainline_trunk_naked).
            "enable_count_hurdle_head": False,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": False,
            "enable_funding_anchor": True,
            "enable_funding_log_domain": False,
            "enable_funding_source_scaling": False,
            "enable_funding_tail_focus": False,
            "enable_funding_gpd_tail": False,
            "enable_funding_cqr_interval": False,
            "enable_log1p_target": True,
            "enable_hawkes_financing_state": False,
            "enable_investors_event_state_features": False,
            "enable_investors_selective_event_state_activation": False,
            "enable_investors_intensity_baseline": False,
            "enable_investors_shrinkage_gate": False,
            "enable_jump_ode_state": False,
            # Trunk swap.
            "enable_learnable_trunk": False,
            "enable_state_space_trunk": True,
            "s5_d_model": 128,
            "s5_d_state": 64,
            "s5_n_blocks": 3,
            "s5_dropout": 0.1,
            "s5_max_epochs": 24,
            "s5_batch_size": 512,
            "s5_device": "cpu",
        },
    ),
    "mainline_s5_full": MainlineVariantProfile(
        name="mainline_s5_full",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "2026-04-22 path-3 FULL paper-method variant: S5 diagonal "
            "state-space trunk + ALL FTE atoms enabled (hurdle, GPD tail, "
            "CQR, Hawkes financing state, source scaling, tail focus, log "
            "domain, jump ODE, investors event-state & intensity "
            "baseline, shrinkage gate). This is the concrete executable "
            "instantiation of the paper's C1 contribution (shared trunk + "
            "factorised head + L_N regulariser). Paired with "
            "mainline_s5_trunk as the trunk-only baseline: paired delta "
            "measures the added value of atoms on top of the S5 trunk."
        ),
        prototype_overrides={
            # Atoms ON — full FTE head pipeline.
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
            "enable_funding_anchor": True,
            "enable_funding_log_domain": True,
            "enable_funding_source_scaling": True,
            "enable_funding_tail_focus": True,
            "enable_funding_gpd_tail": True,
            "enable_funding_cqr_interval": True,
            "enable_log1p_target": True,
            "enable_hawkes_financing_state": True,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "enable_investors_intensity_baseline": True,
            "enable_investors_shrinkage_gate": True,
            "enable_jump_ode_state": True,
            # Trunk: S5, not MoE.
            "enable_learnable_trunk": False,
            "enable_state_space_trunk": True,
            "s5_d_model": 128,
            "s5_d_state": 64,
            "s5_n_blocks": 3,
            "s5_dropout": 0.1,
            "s5_max_epochs": 24,
            "s5_batch_size": 512,
            "s5_device": "cpu",
        },
    ),
    # `mainline_xi_unbounded` variant for the GPT-5.4 reviewer demand
    # (paired ablation: same trunk + same training, only the tail-head xi
    # activation differs between bounded `xi_max * sigma(z)` and unbounded
    # `exp(z)`). Blocked on the FTE tail-head activation switch not yet
    # being a runtime knob in the mainline lanes; track in the next
    # engineering iteration once enable_funding_gpd_tail+xi_activation are
    # exposed via the prototype-overrides surface.
    # ------------------------------------------------------------------
    # Route B — MLP-adapter fallback for the case where the S5 trunk
    # (Route A'/`mainline_s5_full`) does not close the gap on Task 2
    # funding. Same FTE atoms ON as Route A', but trunk swapped from S5
    # state-space to the (default) learnable MLP adapter. Serves as the
    # pre-committed fallback per `docs/PROJECT_STATUS_UNIFIED.md` R1.
    "mainline_mlp_full": MainlineVariantProfile(
        name="mainline_mlp_full",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "2026-04-22 Route B fallback: MLP adapter trunk + ALL FTE "
            "atoms ON (hurdle, GPD tail, CQR, Hawkes state, investors "
            "event-state, shrinkage gate, jump ODE). Activated when the "
            "S5 trunk + full-atoms variant (`mainline_s5_full`) fails "
            "to beat the NBEATS non-factorised baseline; the paired "
            "delta S5_full vs MLP_full quantifies the trunk-architecture "
            "contribution independently of the FTE head contribution."
        ),
        prototype_overrides={
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
            "enable_funding_anchor": True,
            "enable_funding_log_domain": True,
            "enable_funding_source_scaling": True,
            "enable_funding_tail_focus": True,
            "enable_funding_gpd_tail": True,
            "enable_funding_cqr_interval": True,
            "enable_log1p_target": True,
            "enable_hawkes_financing_state": True,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "enable_investors_intensity_baseline": True,
            "enable_investors_shrinkage_gate": True,
            "enable_jump_ode_state": True,
            # Trunk: MLP adapter (the default when state-space flag is off).
            "enable_learnable_trunk": True,
            "enable_state_space_trunk": False,
        },
    ),
    # ------------------------------------------------------------------
    # Route C — 2026-04-23 post-mortem. Routes A'/B both collapsed on
    # funding with identical 741255.5 MAE across 3 replicas (6-sig-digit
    # match), while Route A (naked S5 trunk) settled at 646K (≈global
    # mean, underfit, no collapse). The delta isolates the funding-
    # specific atoms as the sole collapse driver. Route C keeps the S5
    # trunk + all count/investors atoms ON, but strips funding back to
    # {anchor + log1p} only — matching Route A's funding configuration
    # while retaining the count/investors lanes that power T1 investor
    # count and T2 binary targets. Paired contrast with mainline_s5_full
    # isolates the marginal damage of the funding tail/GPD/CQR/source-
    # scaling/log-domain atoms.
    "mainline_s5_funding_stripped": MainlineVariantProfile(
        name="mainline_s5_funding_stripped",
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=(
            "2026-04-23 Route C post-collapse recovery: S5 trunk + count/"
            "investors atoms ON + funding atoms STRIPPED to anchor+log1p. "
            "Designed to falsify the hypothesis that funding-specific "
            "atoms (log_domain, source_scaling, tail_focus, gpd_tail, "
            "cqr_interval) are the sole driver of the 741255.5 MAE "
            "collapse observed in Routes A' and B across 3 replicas."
        ),
        prototype_overrides={
            # Count atoms: ON (they do not collapse; only funding does).
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
            # Funding atoms: STRIPPED to naked S5 equivalent (Route A
            # configuration that produced 646K, not 741K).
            "enable_funding_anchor": True,
            "enable_funding_log_domain": False,
            "enable_funding_source_scaling": False,
            "enable_funding_tail_focus": False,
            "enable_funding_gpd_tail": False,
            "enable_funding_cqr_interval": False,
            "enable_log1p_target": True,
            # Investors atoms: ON.
            "enable_hawkes_financing_state": True,
            "enable_investors_event_state_features": True,
            "enable_investors_selective_event_state_activation": True,
            "enable_investors_intensity_baseline": True,
            "enable_investors_shrinkage_gate": True,
            "enable_jump_ode_state": True,
            # Trunk: S5 (same as Route A'/mainline_s5_full).
            "enable_learnable_trunk": False,
            "enable_state_space_trunk": True,
            "s5_d_model": 128,
            "s5_d_state": 64,
            "s5_n_blocks": 3,
            "s5_dropout": 0.1,
            "s5_max_epochs": 24,
            "s5_batch_size": 512,
            "s5_device": "cpu",
        },
    ),
}

# ------------------------------------------------------------------
# Route D (2026-04-23 16:00 CEST) — ATOM ISOLATION SUITE.
# Route C probe (5346102 npin=639K / 5346104 cfisch=646K) confirmed the
# trunk already produces non-collapsed conditional predictions
# (std=619K ≈ y_std, 2423 unique values, range 13..31.6M). Funding_lane
# auto-enables hurdle-structured severity via `_uses_jump_hurdle_head`
# when positive_jump_rows ≥ threshold. The remaining gap to NBEATS 380K
# is tail compression, not mean collapse. These D-variants isolate ONE
# funding atom at a time on top of Route C to find which atom
# compresses the tail without inducing the 741K loss-optimal-constant
# collapse.
_ROUTE_C_BASE: Dict[str, Any] = dict(MAINLINE_VARIANTS["mainline_s5_funding_stripped"].prototype_overrides)


def _route_d(name: str, description: str, **tweaks: Any) -> MainlineVariantProfile:
    overrides = dict(_ROUTE_C_BASE)
    overrides.update(tweaks)
    return MainlineVariantProfile(
        name=name,
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=description,
        prototype_overrides=overrides,
    )


MAINLINE_VARIANTS["mainline_s5_fs_log_only"] = _route_d(
    "mainline_s5_fs_log_only",
    "Route D1: Route C + log_domain ON. Tests whether log-domain severity "
    "alone compresses the funding tail toward NBEATS 380K without the "
    "741K loss-optimal collapse seen in A'/B.",
    enable_funding_log_domain=True,
)
MAINLINE_VARIANTS["mainline_s5_fs_source_only"] = _route_d(
    "mainline_s5_fs_source_only",
    "Route D2: Route C + source_scaling ON. Tests whether EDGAR-source-"
    "conditional scaling compresses the funding tail.",
    enable_funding_source_scaling=True,
)
MAINLINE_VARIANTS["mainline_s5_fs_tailfocus_lite"] = _route_d(
    "mainline_s5_fs_tailfocus_lite",
    "Route D3: Route C + tail_focus(q=0.95, w=1.5). Lighter tail weighting "
    "than the 0.85/2.0 that caused collapse in A'/B.",
    enable_funding_tail_focus=True,
    funding_tail_quantile=0.95,
    funding_tail_weight=1.5,
)
MAINLINE_VARIANTS["mainline_s5_fs_log_source"] = _route_d(
    "mainline_s5_fs_log_source",
    "Route D4: Route C + log_domain + source_scaling pairwise combo.",
    enable_funding_log_domain=True,
    enable_funding_source_scaling=True,
)

# ----- Route G (2026-04-23): forced-hurdle variants -----
# Route D exposed that at h=7 funding the short-circuit (positive_jump_rows
# threshold + jump_target_std degeneracy) routes everything into the
# trunk-fallback ridge, so source_scaling / tail_focus atoms are never
# exercised (bit-exact MAE proved this in Round 10).  Route G flips the
# `enable_funding_forced_hurdle` flag which bypasses those gates (the
# hard n<12 gate is preserved) so the severity path actually runs and
# every atom becomes measurable.


def _route_g(name: str, description: str, **tweaks: Any) -> MainlineVariantProfile:
    overrides = dict(_ROUTE_C_BASE)
    overrides["enable_funding_forced_hurdle"] = True
    overrides.update(tweaks)
    return MainlineVariantProfile(
        name=name,
        runtime_mode="native",
        delegate_variant="v740_alpha",
        description=description,
        prototype_overrides=overrides,
    )


MAINLINE_VARIANTS["mainline_s5_fs_forced_logonly"] = _route_g(
    "mainline_s5_fs_forced_logonly",
    "Route G1: Route C + forced hurdle + log_domain. Forces the "
    "hurdle+severity head to run at h=7 funding so the log_domain atom "
    "is actually exercised.",
    enable_funding_log_domain=True,
)
MAINLINE_VARIANTS["mainline_s5_fs_forced_source"] = _route_g(
    "mainline_s5_fs_forced_source",
    "Route G2: Route C + forced hurdle + source_scaling.",
    enable_funding_source_scaling=True,
)
MAINLINE_VARIANTS["mainline_s5_fs_forced_tailfocus"] = _route_g(
    "mainline_s5_fs_forced_tailfocus",
    "Route G3: Route C + forced hurdle + tail_focus(q=0.95,w=1.5).",
    enable_funding_tail_focus=True,
    funding_tail_quantile=0.95,
    funding_tail_weight=1.5,
)
MAINLINE_VARIANTS["mainline_s5_fs_forced_base"] = _route_g(
    "mainline_s5_fs_forced_base",
    "Route G0: Route C + forced hurdle, no additional atoms. Baseline "
    "to isolate the effect of running the hurdle path itself on this cell.",
)


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