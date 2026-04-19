#!/usr/bin/env python3
"""Objective contract for the single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


def _ordered_unique(items: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class MainlineObjectiveRuntime:
    lane_name: str
    horizon: int
    lane_subregime: str
    runtime_stage: str
    lane_terms: Tuple[str, ...]
    implemented_terms: Tuple[str, ...]
    review_terms: Tuple[str, ...]
    deferred_terms: Tuple[str, ...]
    enabled_but_deferred_terms: Tuple[str, ...]
    guardrails: Tuple[str, ...]
    active_switches: Mapping[str, bool]
    implemented_runtime: Tuple[str, ...]
    requires_public_pack_graceful_degradation: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "horizon": self.horizon,
            "lane_subregime": self.lane_subregime,
            "runtime_stage": self.runtime_stage,
            "lane_terms": self.lane_terms,
            "implemented_terms": self.implemented_terms,
            "review_terms": self.review_terms,
            "deferred_terms": self.deferred_terms,
            "enabled_but_deferred_terms": self.enabled_but_deferred_terms,
            "guardrails": self.guardrails,
            "active_switches": dict(self.active_switches),
            "implemented_runtime": self.implemented_runtime,
            "requires_public_pack_graceful_degradation": self.requires_public_pack_graceful_degradation,
        }


@dataclass(frozen=True)
class MainlineObjectiveSpec:
    binary_terms: Tuple[str, ...] = ("calibration", "hazard", "event_consistency")
    funding_terms: Tuple[str, ...] = ("anchor_residual", "tail_guard", "source_scaling_guard")
    investors_terms: Tuple[str, ...] = ("occurrence", "hurdle", "transition")
    whole_model_terms: Tuple[str, ...] = ("family_balance", "cross_task_stability", "catastrophic_guard")
    consistency_terms: Tuple[str, ...] = (
        "binary_funding_alignment",
        "binary_investor_alignment",
        "lane_transition_regularization",
    )
    reliability_terms: Tuple[str, ...] = ("selective_calibration", "hard_slice_reliability", "abstention_gate")
    source_robustness_terms: Tuple[str, ...] = (
        "source_dropout",
        "delayed_source_replay",
        "counterfactual_source_ablation",
    )
    native_runtime_stage: str = "lane_runtime_with_objective_plan"
    requires_public_pack_graceful_degradation: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "binary_terms": self.binary_terms,
            "funding_terms": self.funding_terms,
            "investors_terms": self.investors_terms,
            "whole_model_terms": self.whole_model_terms,
            "consistency_terms": self.consistency_terms,
            "reliability_terms": self.reliability_terms,
            "source_robustness_terms": self.source_robustness_terms,
            "native_runtime_stage": self.native_runtime_stage,
            "requires_public_pack_graceful_degradation": self.requires_public_pack_graceful_degradation,
        }

    def runtime_plan(self) -> Dict[str, object]:
        return {
            "native_runtime_stage": self.native_runtime_stage,
            "lane_terms": {
                "binary": self.binary_terms,
                "funding": self.funding_terms,
                "investors": self.investors_terms,
            },
            "review_terms": self.whole_model_terms,
            "deferred_terms": self.consistency_terms + self.reliability_terms + self.source_robustness_terms,
            "requires_public_pack_graceful_degradation": self.requires_public_pack_graceful_degradation,
        }

    def build_runtime_state(
        self,
        lane_name: str,
        horizon: int,
        lane_guardrails: Tuple[str, ...],
        switches: Mapping[str, bool],
    ) -> Dict[str, object]:
        normalized_switches = {key: bool(value) for key, value in switches.items()}
        lane_terms = self._lane_terms(lane_name)
        implemented_terms = self._implemented_terms(lane_name)
        review_terms = self.whole_model_terms
        deferred_terms = _ordered_unique(
            tuple(term for term in lane_terms if term not in implemented_terms)
            + self.consistency_terms
            + self.reliability_terms
            + self.source_robustness_terms
        )
        enabled_but_deferred_terms = _ordered_unique(
            tuple(
                term
                for term in self._switch_requested_terms(lane_name, normalized_switches)
                if term not in implemented_terms
            )
        )
        runtime = MainlineObjectiveRuntime(
            lane_name=lane_name,
            horizon=int(horizon),
            lane_subregime=self._lane_subregime(lane_name, horizon=int(horizon)),
            runtime_stage=self.native_runtime_stage,
            lane_terms=lane_terms,
            implemented_terms=implemented_terms,
            review_terms=review_terms,
            deferred_terms=deferred_terms,
            enabled_but_deferred_terms=enabled_but_deferred_terms,
            guardrails=_ordered_unique(tuple(lane_guardrails)),
            active_switches=normalized_switches,
            implemented_runtime=self._implemented_runtime(lane_name, horizon=int(horizon)),
            requires_public_pack_graceful_degradation=self.requires_public_pack_graceful_degradation,
        )
        return runtime.as_dict()

    def _lane_terms(self, lane_name: str) -> Tuple[str, ...]:
        if lane_name == "binary":
            return self.binary_terms
        if lane_name == "funding":
            return self.funding_terms
        if lane_name == "investors":
            return self.investors_terms
        raise ValueError(f"Unsupported lane for objective runtime plan: {lane_name}")

    def _implemented_terms(self, lane_name: str) -> Tuple[str, ...]:
        if lane_name == "binary":
            return ("calibration", "hazard", "event_consistency")
        if lane_name == "funding":
            return ("anchor_residual", "tail_guard", "cqr_interval")
        if lane_name == "investors":
            return ("occurrence", "hurdle", "intensity_baseline", "jump_ode_state", "shrinkage_gate")
        raise ValueError(f"Unsupported lane for objective runtime plan: {lane_name}")

    def _implemented_runtime(self, lane_name: str, horizon: int) -> Tuple[str, ...]:
        shared_runtime = (
            "shared_backbone_condition_source_barrier",
            "lane_isolation",
            "history_feature_merge",
            "source_density_monitor",
        )
        if lane_name == "binary":
            return shared_runtime + (
                "balanced_logistic_binary_head",
                "hazard_space_calibration",
                "survival_nll_calibrator_scoring",
                "probability_output",
            )
        if lane_name == "funding":
            return shared_runtime + (
                "anchor_residual_regression",
                "calibrated_anchor_backoff",
                "gpd_tail_correction",
                "cqr_prediction_interval",
                "nonnegative_clip",
            )
        if lane_name == "investors":
            investors_runtime = (
                "occurrence_classifier",
                "intensity_baseline_blend",
                "jump_ode_state_evolution",
                "adaptive_shrinkage_gate",
                "positive_regressor",
                "anchor_blend",
            )
            if int(horizon) == 1:
                investors_runtime += (
                    "short_horizon_exemplar_blend",
                    "h1_transition_block_by_contract",
                )
            else:
                investors_runtime += ("geometry_gated_transition_correction",)
            return shared_runtime + investors_runtime
        raise ValueError(f"Unsupported lane for objective runtime plan: {lane_name}")

    def _lane_subregime(self, lane_name: str, horizon: int) -> str:
        if lane_name == "binary":
            return "binary_event_calibration"
        if lane_name == "funding":
            return "funding_anchor_residual"
        if lane_name == "investors":
            if int(horizon) == 1:
                return "h1_occurrence_exemplar"
            return "hplus_hurdle_transition"
        raise ValueError(f"Unsupported lane for objective subregime lookup: {lane_name}")

    def _switch_requested_terms(self, lane_name: str, switches: Mapping[str, bool]) -> Tuple[str, ...]:
        requested: list[str] = []
        if lane_name == "binary":
            if switches.get("teacher_distill", False):
                requested.append("hazard")
            if switches.get("event_head", False):
                requested.append("event_consistency")
        elif lane_name == "funding":
            if switches.get("funding_source_scaling_guard", False):
                requested.append("source_scaling_guard")
            if switches.get("task_modulation", False):
                requested.append("tail_guard")
        elif lane_name == "investors":
            if (
                switches.get("count_source_specialists", False)
                or switches.get("count_source_routing", False)
                or switches.get("investors_transition_correction", False)
            ):
                requested.append("transition")

        if switches.get("financing_consistency", False):
            requested.extend(self.consistency_terms)
        if switches.get("reliability_abstention", False):
            requested.extend(self.reliability_terms)
        if switches.get("counterfactual_source_ablation", False):
            requested.extend(self.source_robustness_terms)
        return tuple(requested)


__all__ = ["MainlineObjectiveRuntime", "MainlineObjectiveSpec"]