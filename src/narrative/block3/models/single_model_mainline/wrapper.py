#!/usr/bin/env python3
"""Native wrapper for the target-isolated single-model mainline scaffold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..base import ModelBase, ModelConfig
from ..v740_alpha import V740AlphaPrototypeWrapper
from .backbone import SharedTemporalBackbone, SharedTemporalBackboneSpec
from .barrier import HardTargetBarrierSpec, TargetIsolatedBarrier
from .conditioning import ConditionKey, ConditioningSchema, MainlineConditionEncoder
from .lanes.binary_lane import BinaryLaneRuntime, BinaryLaneSpec
from .lanes.funding_lane import FundingLaneRuntime, FundingLaneSpec
from .lanes.investors_lane import InvestorsLaneRuntime, InvestorsLaneSpec, _transition_signal
from .objectives import MainlineObjectiveSpec
from .source_memory import SourceMemoryAssembler, SourceMemoryContract
from .variant_profiles import build_delegate_kwargs, get_mainline_variant_profile


@dataclass(frozen=True)
class MainlineModuleContract:
    backbone: SharedTemporalBackboneSpec
    conditioning: ConditioningSchema
    source_memory: SourceMemoryContract
    barrier: HardTargetBarrierSpec
    binary_lane: BinaryLaneSpec
    funding_lane: FundingLaneSpec
    investors_lane: InvestorsLaneSpec
    objectives: MainlineObjectiveSpec

    def as_dict(self) -> Dict[str, object]:
        return {
            "backbone": self.backbone.as_dict(),
            "conditioning": self.conditioning.as_dict(),
            "source_memory": self.source_memory.as_dict(),
            "barrier": self.barrier.as_dict(),
            "binary_lane": self.binary_lane.as_dict(),
            "funding_lane": self.funding_lane.as_dict(),
            "investors_lane": self.investors_lane.as_dict(),
            "objectives": self.objectives.as_dict(),
        }


class SingleModelMainlineWrapper(ModelBase):
    """Mainline runtime owner for the TESF scaffold.

    The default path now executes inside the explicit mainline modules. The
    audited V740-alpha prototype remains available through the
    ``mainline_delegate_alpha`` compatibility profile or ``use_delegate=True``.
    """

    def __init__(self, variant: str = "mainline_alpha", **prototype_kwargs):
        self.variant = variant
        self.profile = get_mainline_variant_profile(variant)
        self.use_delegate = bool(prototype_kwargs.pop("use_delegate", self.profile.runtime_mode == "delegate"))
        seed = int(prototype_kwargs.get("seed", 42))
        config = ModelConfig(
            name="SingleModelMainlineWrapper",
            model_type="forecasting",
            params={"variant": variant, **prototype_kwargs},
            optional_dependency="torch",
        )
        super().__init__(config)
        self.prototype_kwargs = dict(prototype_kwargs)
        self.contract = MainlineModuleContract(
            backbone=SharedTemporalBackboneSpec(),
            conditioning=ConditioningSchema(),
            source_memory=SourceMemoryContract(),
            barrier=HardTargetBarrierSpec(),
            binary_lane=BinaryLaneSpec(),
            funding_lane=FundingLaneSpec(),
            investors_lane=InvestorsLaneSpec(),
            objectives=MainlineObjectiveSpec(),
        )

        self.enable_teacher_distill = bool(prototype_kwargs.get("enable_teacher_distill", True))
        self.enable_event_head = bool(prototype_kwargs.get("enable_event_head", True))
        self.enable_task_modulation = bool(prototype_kwargs.get("enable_task_modulation", True))
        self.enable_target_routing = bool(prototype_kwargs.get("enable_target_routing", True))
        self.target_route_experts = int(prototype_kwargs.get("target_route_experts", 3))
        self.enable_count_anchor = bool(prototype_kwargs.get("enable_count_anchor", True))
        self.count_anchor_strength = float(prototype_kwargs.get("count_anchor_strength", 0.70))
        self.enable_count_jump = bool(prototype_kwargs.get("enable_count_jump", False))
        self.count_jump_strength = float(prototype_kwargs.get("count_jump_strength", 0.30))
        self.enable_count_sparsity_gate = bool(prototype_kwargs.get("enable_count_sparsity_gate", False))
        self.count_sparsity_gate_strength = float(prototype_kwargs.get("count_sparsity_gate_strength", 0.75))
        self.enable_count_source_routing = bool(prototype_kwargs.get("enable_count_source_routing", True))
        self.count_route_experts = int(prototype_kwargs.get("count_route_experts", 3))
        self.count_route_floor = float(prototype_kwargs.get("count_route_floor", 0.10))
        self.count_route_entropy_strength = float(prototype_kwargs.get("count_route_entropy_strength", 0.03))
        self.count_active_loss_strength = float(prototype_kwargs.get("count_active_loss_strength", 0.08))
        self.enable_funding_anchor = bool(prototype_kwargs.get("enable_funding_anchor", True))
        self.funding_anchor_strength = float(prototype_kwargs.get("funding_anchor_strength", 0.85))
        self.enable_funding_log_domain = bool(prototype_kwargs.get("enable_funding_log_domain", False))
        self.enable_funding_source_scaling = bool(prototype_kwargs.get("enable_funding_source_scaling", False))
        self.enable_v741_lite = bool(prototype_kwargs.get("enable_v741_lite", False))
        self.enable_count_hurdle_head = bool(prototype_kwargs.get("enable_count_hurdle_head", False))
        self.enable_count_source_specialists = bool(prototype_kwargs.get("enable_count_source_specialists", False))
        self.enable_investors_source_features = bool(
            prototype_kwargs.get("enable_investors_source_features", self.enable_count_source_specialists)
        )
        self.enable_investors_source_guard = bool(
            prototype_kwargs.get("enable_investors_source_guard", self.enable_count_source_specialists)
        )
        self.enable_investors_source_read_policy = bool(
            prototype_kwargs.get("enable_investors_source_read_policy", False)
        )
        self.enable_investors_transition_correction = bool(
            prototype_kwargs.get("enable_investors_transition_correction", False)
        )
        self.enable_investors_selective_source_activation = bool(
            prototype_kwargs.get("enable_investors_selective_source_activation", True)
        )
        self.enable_investors_horizon_contract = bool(
            prototype_kwargs.get("enable_investors_horizon_contract", True)
        )
        self.enable_investors_transition_dynamic_gate = bool(
            prototype_kwargs.get("enable_investors_transition_dynamic_gate", True)
        )
        self.investors_source_activation_min_rows = int(
            prototype_kwargs.get("investors_source_activation_min_rows", 64)
        )
        self.investors_source_activation_min_share = float(
            prototype_kwargs.get("investors_source_activation_min_share", 0.10)
        )
        self.investors_transition_activation_min_rows = int(
            prototype_kwargs.get("investors_transition_activation_min_rows", 12)
        )
        self.investors_transition_activation_min_share = float(
            prototype_kwargs.get("investors_transition_activation_min_share", 0.10)
        )
        self.investors_transition_activation_min_entities = int(
            prototype_kwargs.get("investors_transition_activation_min_entities", 2)
        )
        self.enable_financing_consistency = bool(prototype_kwargs.get("enable_financing_consistency", False))
        self.financing_consistency_strength = float(prototype_kwargs.get("financing_consistency_strength", 0.0))
        self.financing_auxiliary_strength = float(prototype_kwargs.get("financing_auxiliary_strength", 0.0))
        self.financing_process_blend = float(prototype_kwargs.get("financing_process_blend", 0.0))
        self.enable_window_repair = bool(prototype_kwargs.get("enable_window_repair", False))
        self.min_window_history = int(prototype_kwargs.get("min_window_history", 0))
        self.target_windows_per_entity = int(prototype_kwargs.get("target_windows_per_entity", 0))

        self.condition_encoder = MainlineConditionEncoder(self.contract.conditioning)
        self.backbone = SharedTemporalBackbone(self.contract.backbone, random_state=seed)
        self.source_memory = SourceMemoryAssembler(contract=self.contract.source_memory)
        self.barrier = TargetIsolatedBarrier(self.contract.barrier)
        self.binary_lane_runtime = BinaryLaneRuntime(self.contract.binary_lane, random_state=seed)
        self.funding_lane_runtime = FundingLaneRuntime(self.contract.funding_lane, random_state=seed)
        self.investors_lane_runtime = InvestorsLaneRuntime(self.contract.investors_lane, random_state=seed)

        self._delegate: Optional[V740AlphaPrototypeWrapper] = None
        self._active_lane_name: str = ""
        self._active_lane_model: object | None = None
        self._train_feature_cols: list[str] = []
        self._core_feature_cols: list[str] = []
        self._source_feature_cols: list[str] = []
        self._investors_source_feature_cols: list[str] = []
        self._history_feature_cols: list[str] = []
        self._shared_state_dim = 0
        self._lane_state_dim = 0
        self._condition_key: Optional[ConditionKey] = None
        self._history_seed: Optional[pd.DataFrame] = None
        self._target_name = "funding_raised_usd"
        self._task_name = "task1_outcome"
        self._ablation_name = "core_only"
        self._horizon = 1
        self._edgar_cols: list[str] = []
        self._text_cols: list[str] = []
        self._binary_target = False
        self._nonnegative_target = False
        self._funding_target = False
        self._fallback_value = 0.0
        self._binary_train_rate = 0.0
        self._binary_event_rate = 0.0
        self._binary_transition_rate = 0.0
        self._binary_pos_weight = 1.0
        self._binary_temperature = 1.0
        self._binary_teacher_weight = 0.10
        self._binary_event_weight = 0.15
        self._teacher_logistic_mix = 0.4
        self._teacher_tree_mix = 0.6
        self._edgar_source_density = 0.0
        self._text_source_density = 0.0
        self._funding_log_domain = False
        self._funding_source_scaling = False
        self._funding_anchor_enabled = False
        self._effective_funding_anchor_strength = 0.0
        self._effective_task_modulation = self.enable_task_modulation
        self._investors_source_profile_counts: Dict[str, int] = {}
        self._investors_source_eligible_profiles: tuple[str, ...] = ()
        self._investors_source_effective_min_rows = 0
        self._investors_transition_effective_min_rows = 0
        self._investors_requested_source_path = False
        self._effective_investors_source_features = False
        self._effective_investors_source_specialists = False
        self._effective_investors_source_guard = False
        self._effective_investors_source_read_policy = False
        self._effective_investors_transition_correction = False
        self._investors_source_activation_reason = "source_switches_disabled"
        self._investors_transition_activation_reason = "transition_not_requested"
        self._investors_source_profile_entropy = 0.0
        self._investors_dynamic_entities = 0
        self._investors_dynamic_entity_share = 0.0
        self._investors_dynamic_rows = 0
        self._investors_dynamic_row_share = 0.0
        self._investors_transition_abs_mean = 0.0
        self._investors_transition_nonzero_rows = 0
        self._investors_transition_nonzero_share = 0.0
        self._investors_anchor_mae = 0.0
        self._investors_fallback_mae = 0.0
        self._investors_anchor_mae_ratio = 1.0
        self._investors_anchor_regime = "anchor_unknown"
        self._investors_profile_surface = "unknown"
        self._investors_transition_surface = "unknown"
        self._investors_horizon_subregime = "h1_occurrence_exemplar"
        self._effective_count_hurdle_head = False
        self._effective_count_jump = False
        self._effective_count_sparsity_gate = False

    def describe_contract(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "runtime_mode": self.profile.runtime_mode,
            "delegate_variant": self.profile.delegate_variant,
            "delegate_enabled": self.use_delegate,
            "description": self.profile.description,
            "modules": self.contract.as_dict(),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "SingleModelMainlineWrapper":
        self._target_name = kwargs.get("target", y.name or "funding_raised_usd")
        self._task_name = kwargs.get("task", "task1_outcome")
        self._ablation_name = kwargs.get("ablation", "core_only")
        self._horizon = int(kwargs.get("horizon", 1))
        y_arr = np.asarray(y.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
        finite = y_arr[np.isfinite(y_arr)]
        self._fallback_value = float(np.nanmedian(finite)) if finite.size else 0.0
        self._binary_target = _detect_binary(y_arr)
        self._nonnegative_target = _detect_nonnegative(y_arr)
        self._funding_target = self._target_name == "funding_raised_usd"
        self._binary_train_rate = float(np.mean(y_arr > 0.5)) if self._binary_target and y_arr.size else 0.0
        self._binary_event_rate = self._binary_train_rate
        self._binary_pos_weight = float(
            np.clip((1.0 - self._binary_train_rate) / max(self._binary_train_rate, 1e-6), 1.0, 25.0)
        ) if self._binary_target else 1.0
        self._funding_log_domain = bool(self.enable_funding_log_domain and self._funding_target)
        self._funding_source_scaling = bool(self.enable_funding_source_scaling and self._funding_target)
        self._funding_anchor_enabled = bool(self.enable_funding_anchor and self._funding_target)
        self._effective_funding_anchor_strength = self.funding_anchor_strength if self._funding_anchor_enabled else 0.0
        self._effective_task_modulation = bool(self.enable_task_modulation)
        self._condition_key = ConditionKey(
            task=self._task_name,
            target=self._target_name,
            horizon=self._horizon,
            ablation=self._ablation_name,
        )

        if self.use_delegate:
            delegate_kwargs = build_delegate_kwargs(self.variant, self.prototype_kwargs)
            self._delegate = V740AlphaPrototypeWrapper(**delegate_kwargs)
            self._delegate.fit(X, y, **kwargs)
            self.model = self._delegate
            self._sync_delegate_state()
            self._fitted = True
            return self

        runtime_frame = self._prepare_runtime_frame(X, raw_frame=kwargs.get("train_raw"))
        self._train_feature_cols = list(X.columns)
        source_layout = self.source_memory.infer_layout(runtime_frame)
        source_edgar = set(source_layout.edgar_cols)
        source_text = set(source_layout.text_cols)
        self._edgar_cols = [col for col in self._train_feature_cols if col in source_edgar]
        self._text_cols = [col for col in self._train_feature_cols if col in source_text]
        excluded = set(self._edgar_cols) | set(self._text_cols)
        self._core_feature_cols = [col for col in self._train_feature_cols if col not in excluded]
        if not self._core_feature_cols:
            self._core_feature_cols = list(self._train_feature_cols)

        shared_state = self.backbone.fit_transform(
            X.reindex(columns=self._core_feature_cols, fill_value=0.0),
            feature_cols=self._core_feature_cols,
        )
        condition_state = self.condition_encoder.broadcast(self._condition_key, len(X))
        source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
        self._source_feature_cols = list(source_frame.columns)
        source_state = source_frame.to_numpy(dtype=np.float32, copy=False)
        self._edgar_source_density = float(source_frame["edgar_active"].mean()) if "edgar_active" in source_frame else 0.0
        self._text_source_density = float(source_frame["text_active"].mean()) if "text_active" in source_frame else 0.0

        self.barrier.fit(
            shared_dim=shared_state.shape[1],
            condition_dim=condition_state.shape[1],
            source_dim=source_state.shape[1],
        )
        lane_states = self.barrier.split(shared_state, condition_state, source_state)
        self._shared_state_dim = shared_state.shape[1]
        self._active_lane_name = self._lane_name_for_target(self._target_name)
        self._lane_state_dim = lane_states[self._active_lane_name].shape[1]

        history_frame = self._build_target_history_features(runtime_frame, include_seed=False)
        self._history_feature_cols = list(history_frame.columns)
        history_matrix = history_frame.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        anchor = self._resolve_anchor(history_frame)
        investors_source_frame = self._build_investors_source_features(source_frame)
        self._investors_source_feature_cols = list(investors_source_frame.columns)
        investors_source_matrix = investors_source_frame.to_numpy(dtype=np.float32, copy=False)
        self._refresh_investors_source_activation_regime(
            investors_source_frame,
            runtime_frame=runtime_frame,
            history_frame=history_frame,
            anchor=anchor,
            y=y_arr,
        )

        if self._binary_target:
            self.binary_lane_runtime.fit(lane_states["binary"], y_arr, aux_features=history_matrix)
            self._binary_event_rate = float(self.binary_lane_runtime._event_rate)
            self._binary_transition_rate = float(self.binary_lane_runtime._transition_rate)
            self._binary_temperature = float(self.binary_lane_runtime._temperature)
            self._active_lane_model = self.binary_lane_runtime
        elif self._funding_target:
            self.funding_lane_runtime.fit(
                lane_states["funding"],
                y_arr,
                aux_features=history_matrix,
                anchor=anchor,
            )
            self._active_lane_model = self.funding_lane_runtime
        else:
            self.investors_lane_runtime.fit(
                lane_states["investors"],
                y_arr,
                aux_features=history_matrix,
                anchor=anchor,
                source_features=investors_source_matrix,
                enable_source_features=self._effective_investors_source_features,
                enable_source_specialists=self._effective_investors_source_specialists,
                enable_source_guard=self._effective_investors_source_guard,
                enable_source_read_policy=self._effective_investors_source_read_policy,
                enable_source_transition_correction=self._effective_investors_transition_correction,
                enable_hurdle_head=self._effective_count_hurdle_head,
                enable_count_jump=self._effective_count_jump,
                count_jump_strength=self.count_jump_strength,
                enable_count_sparsity_gate=self._effective_count_sparsity_gate,
                count_sparsity_gate_strength=self.count_sparsity_gate_strength,
                horizon=self._horizon,
                anchor_blend=self.count_anchor_strength if self.enable_count_anchor else 0.0,
                task_name=self._task_name,
            )
            self._active_lane_model = self.investors_lane_runtime

        self._history_seed = self._collect_history_seed(runtime_frame)
        self.model = self._active_lane_model
        self._fitted = True
        return self

    def _source_contract_state(self) -> Dict[str, object]:
        return self.source_memory.summarize_contract_from_stats(
            edgar_cols=self._edgar_cols,
            text_cols=self._text_cols,
            edgar_row_density=self._edgar_source_density,
            text_row_density=self._text_source_density,
        )

    def _objective_runtime_state(self) -> Dict[str, object]:
        lane_name = self._active_lane_name or self._lane_name_for_target(self._target_name)
        return self.contract.objectives.build_runtime_state(
            lane_name=lane_name,
            horizon=self._horizon,
            lane_guardrails=self._lane_guardrails(lane_name),
            switches={
                "teacher_distill": self.enable_teacher_distill,
                "event_head": self.enable_event_head,
                "task_modulation": self._effective_task_modulation,
                "funding_anchor": self._funding_anchor_enabled,
                "funding_source_scaling_guard": self._funding_source_scaling,
                "count_source_routing": self.enable_count_source_routing,
                "count_source_specialists": self._effective_investors_source_specialists,
                "investors_transition_correction": self._effective_investors_transition_correction,
                "financing_consistency": self.enable_financing_consistency,
                "reliability_abstention": False,
                "counterfactual_source_ablation": False,
            },
        )

    def get_regime_info(self) -> Dict[str, object]:
        key = self._condition_key or ConditionKey(
            task=self._task_name,
            target=self._target_name,
            horizon=self._horizon,
            ablation=self._ablation_name,
        )
        source_contract = self._source_contract_state()
        objective_runtime = self._objective_runtime_state()
        return {
            "task": self._task_name,
            "target": self._target_name,
            "ablation": self._ablation_name,
            "horizon": self._horizon,
            "binary_target": self._binary_target,
            "funding_target": self._funding_target,
            "teacher_distill_enabled": self.enable_teacher_distill,
            "event_head_enabled": self.enable_event_head,
            "task_mod_enabled": self._effective_task_modulation,
            "conditioning_ids": self.condition_encoder.ids(key),
            "source_stats": source_contract,
            "state_stream": {
                "feature_cols": len(self._core_feature_cols),
                "source_feature_cols": len(self._source_feature_cols),
                "shared_state_dim": self._shared_state_dim,
            },
            "barrier": {
                "active_lane": self._active_lane_name,
                "lane_state_dim": self._lane_state_dim,
                "lane_private_transition_block": self.contract.barrier.requires_lane_private_transition_block,
                "transition_block_stage": self.contract.barrier.transition_block_stage,
            },
            "objectives": objective_runtime,
            "public_pack": {
                "graceful_degradation_required": self.contract.objectives.requires_public_pack_graceful_degradation,
                "runtime_stage": self.contract.objectives.native_runtime_stage,
            },
            "runtime": {
                "variant": self.variant,
                "runtime_mode": "delegate" if self.use_delegate else "native",
                "predict_time_current_target_masked": True,
                "runtime_no_leak_contract": "predict_time_current_target_masked_before_history_and_anchor",
            },
            "binary_process_contract": {
                "process_family": "hazard_prior_plus_calibration",
                "uses_logistic_head": bool(self.binary_lane_runtime._model is not None),
                "uses_hazard_adapter": bool(self.binary_lane_runtime._uses_hazard_adapter),
                "constant_probability": float(self.binary_lane_runtime._constant_probability),
                "train_positive_rate": float(self._binary_train_rate),
                "event_rate": float(self._binary_event_rate),
                "transition_rate": float(self._binary_transition_rate),
                "persistence_rate": float(self.binary_lane_runtime._persistence_rate),
                "hazard_rows": int(self.binary_lane_runtime._hazard_rows),
                "hazard_blend": float(self.binary_lane_runtime._hazard_blend),
                "calibration_method": str(self.binary_lane_runtime._calibrator_name),
                "positive_class_weight": float(self._binary_pos_weight),
                "temperature": float(self._binary_temperature),
                "selected_brier": float(self.binary_lane_runtime._selected_metrics.get("brier", 0.0)),
                "selected_logloss": float(self.binary_lane_runtime._selected_metrics.get("logloss", 0.0)),
                "selected_ece": float(self.binary_lane_runtime._selected_metrics.get("ece", 0.0)),
                "identity_brier": float(self.binary_lane_runtime._identity_metrics.get("brier", 0.0)),
                "identity_logloss": float(self.binary_lane_runtime._identity_metrics.get("logloss", 0.0)),
                "identity_ece": float(self.binary_lane_runtime._identity_metrics.get("ece", 0.0)),
                "teacher_weight": float(self._binary_teacher_weight),
                "event_weight": float(self._binary_event_weight),
                "teacher_logistic_mix": float(self._teacher_logistic_mix),
                "teacher_tree_mix": float(self._teacher_tree_mix),
            },
            "investors_source_activation": {
                "requested_source_path": self._investors_requested_source_path,
                "selective_contract_enabled": self.enable_investors_selective_source_activation,
                "horizon_contract_enabled": self.enable_investors_horizon_contract,
                "horizon_subregime": self._investors_horizon_subregime,
                "transition_dynamic_gate_enabled": self.enable_investors_transition_dynamic_gate,
                "requested_source_features": self.enable_investors_source_features,
                "requested_source_specialists": self.enable_count_source_specialists,
                "requested_source_guard": self.enable_investors_source_guard,
                "requested_source_read_policy": self.enable_investors_source_read_policy,
                "requested_transition_correction": self.enable_investors_transition_correction,
                "effective_source_features": self._effective_investors_source_features,
                "effective_source_specialists": self._effective_investors_source_specialists,
                "effective_source_guard": self._effective_investors_source_guard,
                "effective_source_read_policy": self._effective_investors_source_read_policy,
                "effective_transition_correction": self._effective_investors_transition_correction,
                "activation_reason": self._investors_source_activation_reason,
                "transition_activation_reason": self._investors_transition_activation_reason,
                "train_profile_counts": dict(self._investors_source_profile_counts),
                "eligible_profiles": list(self._investors_source_eligible_profiles),
                "profile_entropy": self._investors_source_profile_entropy,
                "geometry_card": {
                    "profile_surface": self._investors_profile_surface,
                    "transition_surface": self._investors_transition_surface,
                    "anchor_regime": self._investors_anchor_regime,
                    "dynamic_entities": self._investors_dynamic_entities,
                    "dynamic_entity_share": self._investors_dynamic_entity_share,
                    "dynamic_rows": self._investors_dynamic_rows,
                    "dynamic_row_share": self._investors_dynamic_row_share,
                    "transition_abs_mean": self._investors_transition_abs_mean,
                    "transition_nonzero_rows": self._investors_transition_nonzero_rows,
                    "transition_nonzero_share": self._investors_transition_nonzero_share,
                    "anchor_mae": self._investors_anchor_mae,
                    "fallback_mae": self._investors_fallback_mae,
                    "anchor_mae_ratio": self._investors_anchor_mae_ratio,
                },
                "learned_profile_anchor_blends": {
                    str(profile_id): float(blend)
                    for profile_id, blend in self.investors_lane_runtime._source_anchor_blend_by_profile.items()
                },
                "learned_profile_anchor_blend_reliability": {
                    str(profile_id): float(weight)
                    for profile_id, weight in self.investors_lane_runtime._source_anchor_blend_reliability_by_profile.items()
                },
                "learned_profile_transition_strength": {
                    str(profile_id): float(strength)
                    for profile_id, strength in self.investors_lane_runtime._source_transition_strength_by_profile.items()
                },
                "learned_profile_transition_reliability": {
                    str(profile_id): float(weight)
                    for profile_id, weight in self.investors_lane_runtime._source_transition_reliability_by_profile.items()
                },
                "activation_min_rows": self.investors_source_activation_min_rows,
                "activation_min_share": self.investors_source_activation_min_share,
                "effective_min_rows": self._investors_source_effective_min_rows,
                "transition_activation_min_rows": self.investors_transition_activation_min_rows,
                "transition_activation_min_share": self.investors_transition_activation_min_share,
                "transition_activation_min_entities": self.investors_transition_activation_min_entities,
                "transition_effective_min_rows": self._investors_transition_effective_min_rows,
            },
            "investors_process_contract": {
                "horizon_subregime": self._investors_horizon_subregime,
                "effective_count_hurdle_head": self._effective_count_hurdle_head,
                "effective_count_jump": self._effective_count_jump,
                "effective_count_sparsity_gate": self._effective_count_sparsity_gate,
                "requested_count_hurdle_head": self.enable_count_hurdle_head,
                "requested_count_jump": self.enable_count_jump,
                "requested_count_sparsity_gate": self.enable_count_sparsity_gate,
                "count_jump_strength": self.count_jump_strength,
                "count_sparsity_gate_strength": self.count_sparsity_gate_strength,
                "lane_horizon_anchor_mix": float(self.investors_lane_runtime._horizon_anchor_mix),
                "lane_horizon_anchor_mix_reliability": float(
                    self.investors_lane_runtime._horizon_anchor_mix_reliability
                ),
                "lane_anchor_blend": float(self.investors_lane_runtime._global_anchor_blend),
                "lane_anchor_blend_reliability": float(self.investors_lane_runtime._global_anchor_blend_reliability),
                "lane_jump_strength": float(self.investors_lane_runtime._global_jump_strength),
                "lane_jump_reliability": float(self.investors_lane_runtime._global_jump_reliability),
            },
            "funding_process_contract": {
                "process_family": "anchor_plus_jump_hurdle",
                "anchor_enabled": self._funding_anchor_enabled,
                "funding_log_domain": self._funding_log_domain,
                "funding_source_scaling": self._funding_source_scaling,
                "lane_uses_jump_hurdle_head": bool(self.funding_lane_runtime._uses_jump_hurdle_head),
                "lane_jump_event_rate": float(self.funding_lane_runtime._jump_event_rate),
                "lane_positive_jump_rows": int(self.funding_lane_runtime._positive_jump_rows),
                "lane_positive_jump_median": float(self.funding_lane_runtime._positive_jump_median),
                "lane_jump_floor": float(self.funding_lane_runtime._jump_floor),
                "lane_residual_blend": float(self.funding_lane_runtime._residual_blend),
                "lane_process_blend": float(self.funding_lane_runtime._residual_blend),
                "lane_residual_cap": None
                if not np.isfinite(self.funding_lane_runtime._residual_cap)
                else float(self.funding_lane_runtime._residual_cap),
                "lane_process_cap": None
                if not np.isfinite(self.funding_lane_runtime._residual_cap)
                else float(self.funding_lane_runtime._residual_cap),
                "lane_anchor_calibration_mae": float(self.funding_lane_runtime._anchor_calibration_mae),
                "lane_guarded_calibration_mae": float(self.funding_lane_runtime._guarded_calibration_mae),
                "lane_anchor_dominance": float(self.funding_lane_runtime._anchor_dominance),
                "lane_calibration_rows": int(self.funding_lane_runtime._calibration_rows),
            },
        }

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("SingleModelMainlineWrapper is not fitted")

        if self.use_delegate:
            if self._delegate is None:
                raise ValueError("SingleModelMainlineWrapper delegate is not fitted")
            return self._delegate.predict(X, **kwargs)

        runtime_frame = self._prepare_runtime_frame(X, raw_frame=kwargs.get("test_raw"))
        runtime_frame = self._mask_runtime_target_for_prediction(runtime_frame)
        feature_frame = X.reindex(columns=self._train_feature_cols, fill_value=0.0)
        shared_state = self.backbone.transform(feature_frame.reindex(columns=self._core_feature_cols, fill_value=0.0))
        condition_key = ConditionKey(
            task=kwargs.get("task", self._task_name),
            target=kwargs.get("target", self._target_name),
            horizon=int(kwargs.get("horizon", self._horizon)),
            ablation=kwargs.get("ablation", self._ablation_name),
        )
        condition_state = self.condition_encoder.broadcast(condition_key, len(feature_frame))
        source_layout = self.source_memory.infer_layout(runtime_frame)
        source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
        source_state = source_frame.reindex(columns=self._source_feature_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
        lane_states = self.barrier.split(shared_state, condition_state, source_state)
        history_frame = self._build_target_history_features(runtime_frame, include_seed=True)
        history_matrix = history_frame.reindex(columns=self._history_feature_cols, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        anchor = self._resolve_anchor(history_frame)
        investors_source_frame = self._build_investors_source_features(source_frame)
        investors_source_matrix = investors_source_frame.reindex(
            columns=self._investors_source_feature_cols,
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)

        if self._active_lane_name == "binary":
            preds = self.binary_lane_runtime.predict(lane_states["binary"], aux_features=history_matrix)
            return np.clip(preds, 0.0, 1.0).astype(np.float64, copy=False)
        if self._active_lane_name == "funding":
            preds = self.funding_lane_runtime.predict(lane_states["funding"], aux_features=history_matrix, anchor=anchor)
            return np.clip(preds, 0.0, None).astype(np.float64, copy=False)
        preds = self.investors_lane_runtime.predict(
            lane_states["investors"],
            aux_features=history_matrix,
            anchor=anchor,
            source_features=investors_source_matrix,
            enable_source_features=self._effective_investors_source_features,
            enable_source_specialists=self._effective_investors_source_specialists,
            enable_source_guard=self._effective_investors_source_guard,
            enable_source_read_policy=self._effective_investors_source_read_policy,
            enable_source_transition_correction=self._effective_investors_transition_correction,
        )
        return np.clip(preds, 0.0, None).astype(np.float64, copy=False)

    def _prepare_runtime_frame(self, X: pd.DataFrame, raw_frame: pd.DataFrame | None) -> pd.DataFrame:
        if raw_frame is not None and len(raw_frame) > 0:
            if X.index.isin(raw_frame.index).all():
                runtime = raw_frame.loc[X.index].copy()
            else:
                runtime = raw_frame.reset_index(drop=True).iloc[: len(X)].copy()
                runtime.index = X.index
        else:
            runtime = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in runtime.columns:
                runtime[col] = X[col].values
        return runtime

    def _lane_name_for_target(self, target_name: str) -> str:
        if target_name == "is_funded":
            return "binary"
        if target_name == "funding_raised_usd":
            return "funding"
        return "investors"

    def _lane_guardrails(self, lane_name: str) -> tuple[str, ...]:
        if lane_name == "binary":
            return self.contract.binary_lane.guardrails
        if lane_name == "funding":
            return self.contract.funding_lane.guardrails
        if lane_name == "investors":
            return self.contract.investors_lane.guardrails
        raise ValueError(f"Unsupported lane for guardrail lookup: {lane_name}")

    def _build_target_history_features(self, frame: pd.DataFrame, include_seed: bool) -> pd.DataFrame:
        cols = ["target_lag1", "target_roll3_mean", "target_roll7_mean", "target_roll3_std", "target_history_count"]
        empty = pd.DataFrame(0.0, index=frame.index, columns=cols, dtype=np.float32)
        required = {"entity_id", self._target_name, "crawled_date_day"}
        if not required.issubset(frame.columns):
            return empty

        current = frame[["entity_id", "crawled_date_day", self._target_name]].copy()
        current["entity_id"] = current["entity_id"].astype(str)
        current["crawled_date_day"] = pd.to_datetime(current["crawled_date_day"], errors="coerce")
        current["_is_runtime_row"] = True
        current["_row_order"] = np.arange(len(current), dtype=np.int64)

        if include_seed and self._history_seed is not None and len(self._history_seed) > 0:
            seed = self._history_seed.copy()
            seed["_is_runtime_row"] = False
            seed["_row_order"] = -np.arange(len(seed), 0, -1, dtype=np.int64)
            work = pd.concat([seed, current], axis=0, ignore_index=True, sort=False)
        else:
            work = current

        work.sort_values(["entity_id", "crawled_date_day", "_row_order"], inplace=True, kind="mergesort")
        target_series = pd.to_numeric(work[self._target_name], errors="coerce")
        shifted = target_series.groupby(work["entity_id"], sort=False).shift(1)
        work["target_lag1"] = shifted
        work["target_roll3_mean"] = shifted.groupby(work["entity_id"], sort=False).transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        work["target_roll7_mean"] = shifted.groupby(work["entity_id"], sort=False).transform(
            lambda series: series.rolling(7, min_periods=1).mean()
        )
        work["target_roll3_std"] = shifted.groupby(work["entity_id"], sort=False).transform(
            lambda series: series.rolling(3, min_periods=1).std()
        )
        work["target_history_count"] = shifted.notna().groupby(work["entity_id"], sort=False).cumsum().astype(np.float32)
        runtime = work[work["_is_runtime_row"]].copy()
        runtime.sort_values("_row_order", inplace=True, kind="mergesort")
        runtime.index = frame.index
        return runtime[cols].astype(np.float32)

    def _collect_history_seed(self, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"entity_id", self._target_name, "crawled_date_day"}
        if not required.issubset(frame.columns):
            return None
        seed = frame[["entity_id", "crawled_date_day", self._target_name]].copy()
        seed["entity_id"] = seed["entity_id"].astype(str)
        seed["crawled_date_day"] = pd.to_datetime(seed["crawled_date_day"], errors="coerce")
        seed.sort_values(["entity_id", "crawled_date_day"], inplace=True, kind="mergesort")
        return seed.groupby("entity_id", sort=False).tail(7).reset_index(drop=True)

    def _resolve_anchor(self, history_frame: pd.DataFrame) -> np.ndarray:
        if history_frame.empty:
            return np.full(0, self._fallback_value, dtype=np.float64)
        anchor = history_frame.get("target_lag1", pd.Series(index=history_frame.index, dtype=np.float32)).astype(np.float64)
        for col in ("target_roll3_mean", "target_roll7_mean"):
            if col in history_frame.columns:
                anchor = anchor.where(np.isfinite(anchor), history_frame[col].astype(np.float64))
        anchor = anchor.fillna(self._fallback_value)
        return anchor.to_numpy(dtype=np.float64, copy=False)

    def _mask_runtime_target_for_prediction(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._target_name not in frame.columns:
            return frame
        masked = frame.copy()
        masked[self._target_name] = np.nan
        return masked

    def _build_investors_source_features(self, source_frame: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "edgar_active",
            "text_active",
            "edgar_recency_days",
            "text_recency_days",
            "edgar_nonzero_share",
            "text_nonzero_share",
            "text_novelty",
        ]
        return source_frame.reindex(columns=cols, fill_value=0.0).astype(np.float32)

    def _refresh_investors_source_activation_regime(
        self,
        investors_source_frame: pd.DataFrame,
        *,
        runtime_frame: pd.DataFrame,
        history_frame: pd.DataFrame,
        anchor: np.ndarray,
        y: np.ndarray,
    ) -> None:
        requested_source_path = bool(
            self.enable_investors_source_features
            or self.enable_count_source_specialists
            or self.enable_investors_source_guard
            or self.enable_investors_source_read_policy
            or self.enable_investors_transition_correction
        )
        self._investors_requested_source_path = requested_source_path
        self._investors_source_profile_counts = self._summarize_investors_source_profiles(investors_source_frame)
        self._investors_source_effective_min_rows = self._effective_investors_source_min_rows(len(investors_source_frame))
        self._investors_transition_effective_min_rows = self._effective_investors_transition_min_rows(
            len(investors_source_frame)
        )
        self._investors_source_eligible_profiles = tuple(
            name
            for name, count in self._investors_source_profile_counts.items()
            if int(count) >= self._investors_source_effective_min_rows
        )
        self._investors_source_profile_entropy = self._profile_entropy(self._investors_source_profile_counts)
        self._refresh_investors_geometry_card(runtime_frame, history_frame, anchor, y)
        self._investors_profile_surface = (
            "heterogeneous" if len(self._investors_source_eligible_profiles) >= 2 else "homogeneous"
        )
        self._investors_horizon_subregime = "h1_occurrence_exemplar" if self._horizon == 1 else "hplus_hurdle_transition"
        self._effective_count_hurdle_head = bool(self.enable_count_hurdle_head and self._horizon > 1)
        self._effective_count_jump = bool(self.enable_count_hurdle_head and self.enable_count_jump and self._horizon > 1)
        self._effective_count_sparsity_gate = bool(
            self.enable_count_hurdle_head and self.enable_count_sparsity_gate and self._horizon > 1
        )

        if not requested_source_path:
            activation_allowed = False
            reason = "source_switches_disabled"
        elif not self.enable_investors_selective_source_activation:
            activation_allowed = True
            reason = "selective_contract_disabled"
        elif len(self._investors_source_eligible_profiles) >= 2:
            activation_allowed = True
            reason = "multi_profile_train_surface"
        else:
            activation_allowed = False
            reason = "source_homogeneous_train_surface"

        self._investors_source_activation_reason = reason
        self._effective_investors_source_features = bool(activation_allowed and self.enable_investors_source_features)
        self._effective_investors_source_specialists = bool(activation_allowed and self.enable_count_source_specialists)
        self._effective_investors_source_guard = bool(activation_allowed and self.enable_investors_source_guard)
        self._effective_investors_source_read_policy = bool(activation_allowed and self.enable_investors_source_read_policy)
        transition_allowed = bool(activation_allowed and self.enable_investors_transition_correction)
        if not self.enable_investors_transition_correction:
            transition_reason = "transition_not_requested"
        elif not activation_allowed:
            transition_reason = reason
            transition_allowed = False
        elif self.enable_investors_horizon_contract and self._horizon == 1:
            transition_reason = "transition_h1_blocked_by_contract"
            transition_allowed = False
        elif self.enable_investors_transition_dynamic_gate and not self._investors_transition_surface_is_eligible():
            transition_reason = "transition_sparse_train_surface"
            transition_allowed = False
        elif not self.enable_investors_transition_dynamic_gate:
            transition_reason = "transition_dynamic_gate_disabled"
        else:
            transition_reason = "multi_profile_dynamic_train_surface"
        self._investors_transition_activation_reason = transition_reason
        self._effective_investors_transition_correction = bool(transition_allowed)

    def _effective_investors_source_min_rows(self, row_count: int) -> int:
        if row_count <= 0:
            return max(1, self.investors_source_activation_min_rows)
        share_rows = int(np.ceil(max(self.investors_source_activation_min_share, 0.0) * float(row_count)))
        return max(1, self.investors_source_activation_min_rows, share_rows)

    def _effective_investors_transition_min_rows(self, row_count: int) -> int:
        if row_count <= 0:
            return max(1, self.investors_transition_activation_min_rows)
        share_rows = int(np.ceil(max(self.investors_transition_activation_min_share, 0.0) * float(row_count)))
        return max(1, self.investors_transition_activation_min_rows, share_rows)

    def _summarize_investors_source_profiles(self, investors_source_frame: pd.DataFrame) -> Dict[str, int]:
        if investors_source_frame.empty:
            return {}
        edgar_active = investors_source_frame.get(
            "edgar_active",
            pd.Series(0.0, index=investors_source_frame.index, dtype=np.float32),
        ).to_numpy(dtype=np.float32, copy=False) >= 0.5
        text_active = investors_source_frame.get(
            "text_active",
            pd.Series(0.0, index=investors_source_frame.index, dtype=np.float32),
        ).to_numpy(dtype=np.float32, copy=False) >= 0.5
        counts = {
            "none": int(np.sum(np.logical_and(~edgar_active, ~text_active))),
            "edgar_only": int(np.sum(np.logical_and(edgar_active, ~text_active))),
            "text_only": int(np.sum(np.logical_and(~edgar_active, text_active))),
            "mixed": int(np.sum(np.logical_and(edgar_active, text_active))),
        }
        return {name: count for name, count in counts.items() if count > 0}

    def _refresh_investors_geometry_card(
        self,
        runtime_frame: pd.DataFrame,
        history_frame: pd.DataFrame,
        anchor: np.ndarray,
        y: np.ndarray,
    ) -> None:
        dynamics = self._summarize_investors_target_dynamics(runtime_frame, y)
        self._investors_dynamic_entities = int(dynamics["dynamic_entities"])
        self._investors_dynamic_entity_share = float(dynamics["dynamic_entity_share"])
        self._investors_dynamic_rows = int(dynamics["dynamic_rows"])
        self._investors_dynamic_row_share = float(dynamics["dynamic_row_share"])

        transition = self._summarize_investors_transition_signal(history_frame, anchor)
        self._investors_transition_abs_mean = float(transition["abs_mean"])
        self._investors_transition_nonzero_rows = int(transition["nonzero_rows"])
        self._investors_transition_nonzero_share = float(transition["nonzero_share"])

        anchor_summary = self._summarize_investors_anchor_regime(anchor, y)
        self._investors_anchor_mae = float(anchor_summary["anchor_mae"])
        self._investors_fallback_mae = float(anchor_summary["fallback_mae"])
        self._investors_anchor_mae_ratio = float(anchor_summary["anchor_mae_ratio"])
        self._investors_anchor_regime = str(anchor_summary["anchor_regime"])
        self._investors_transition_surface = (
            "dynamic" if self._investors_transition_surface_is_eligible() else "static_like"
        )

    def _summarize_investors_target_dynamics(self, runtime_frame: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        if runtime_frame.empty or "entity_id" not in runtime_frame.columns:
            return {
                "dynamic_entities": 0,
                "dynamic_entity_share": 0.0,
                "dynamic_rows": 0,
                "dynamic_row_share": 0.0,
            }
        if self._target_name in runtime_frame.columns:
            target_series = pd.to_numeric(runtime_frame[self._target_name], errors="coerce")
        else:
            values = np.asarray(y, dtype=np.float64).reshape(-1)
            if len(values) != len(runtime_frame):
                return {
                    "dynamic_entities": 0,
                    "dynamic_entity_share": 0.0,
                    "dynamic_rows": 0,
                    "dynamic_row_share": 0.0,
                }
            target_series = pd.Series(values, index=runtime_frame.index, dtype=np.float64)
        work = pd.DataFrame(
            {
                "entity_id": runtime_frame["entity_id"].astype(str),
                "target": target_series,
            },
            index=runtime_frame.index,
        ).dropna(subset=["target"])
        if work.empty:
            return {
                "dynamic_entities": 0,
                "dynamic_entity_share": 0.0,
                "dynamic_rows": 0,
                "dynamic_row_share": 0.0,
            }
        entity_ranges = work.groupby("entity_id", sort=False)["target"].agg(lambda series: float(series.max() - series.min()))
        dynamic_entities = entity_ranges[entity_ranges > 1e-6]
        dynamic_entity_ids = set(dynamic_entities.index.tolist())
        total_entities = int(entity_ranges.shape[0])
        dynamic_rows = int(work[work["entity_id"].isin(dynamic_entity_ids)].shape[0])
        return {
            "dynamic_entities": int(len(dynamic_entity_ids)),
            "dynamic_entity_share": float(len(dynamic_entity_ids) / total_entities) if total_entities else 0.0,
            "dynamic_rows": dynamic_rows,
            "dynamic_row_share": float(dynamic_rows / len(work)) if len(work) else 0.0,
        }

    def _summarize_investors_transition_signal(
        self,
        history_frame: pd.DataFrame,
        anchor: np.ndarray,
    ) -> Dict[str, float]:
        if history_frame.empty or len(anchor) == 0:
            return {"abs_mean": 0.0, "nonzero_rows": 0, "nonzero_share": 0.0}
        aux = history_frame.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        signal = np.abs(_transition_signal(aux, anchor))
        nonzero_rows = int(np.sum(signal > 1e-9))
        return {
            "abs_mean": float(np.mean(signal)) if len(signal) else 0.0,
            "nonzero_rows": nonzero_rows,
            "nonzero_share": float(nonzero_rows / len(signal)) if len(signal) else 0.0,
        }

    def _summarize_investors_anchor_regime(self, anchor: np.ndarray, y: np.ndarray) -> Dict[str, float | str]:
        anchor_vec = np.asarray(anchor, dtype=np.float64).reshape(-1)
        target_vec = np.asarray(y, dtype=np.float64).reshape(-1)
        mask = np.isfinite(anchor_vec) & np.isfinite(target_vec)
        if not mask.any():
            return {
                "anchor_mae": 0.0,
                "fallback_mae": 0.0,
                "anchor_mae_ratio": 1.0,
                "anchor_regime": "anchor_unknown",
            }
        anchor_mae = float(np.mean(np.abs(anchor_vec[mask] - target_vec[mask])))
        fallback_mae = float(np.mean(np.abs(np.full(int(mask.sum()), self._fallback_value) - target_vec[mask])))
        ratio = float(anchor_mae / max(fallback_mae, 1e-8)) if mask.any() else 1.0
        if ratio <= 0.35:
            regime = "anchor_dominant"
        elif ratio <= 0.75:
            regime = "anchor_helpful"
        else:
            regime = "anchor_weak"
        return {
            "anchor_mae": anchor_mae,
            "fallback_mae": fallback_mae,
            "anchor_mae_ratio": ratio,
            "anchor_regime": regime,
        }

    def _investors_transition_surface_is_eligible(self) -> bool:
        return bool(
            self._investors_dynamic_entities >= max(1, self.investors_transition_activation_min_entities)
            and self._investors_transition_nonzero_rows >= self._investors_transition_effective_min_rows
        )

    def _profile_entropy(self, counts: Dict[str, int]) -> float:
        total = float(sum(int(count) for count in counts.values()))
        if total <= 0.0 or len(counts) <= 1:
            return 0.0
        probs = np.asarray([float(count) / total for count in counts.values() if count > 0], dtype=np.float64)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        return float(entropy / np.log(len(probs))) if len(probs) > 1 else 0.0

    def _sync_delegate_state(self) -> None:
        if self._delegate is None:
            return
        for name in (
            "_edgar_cols",
            "_text_cols",
            "_binary_target",
            "_nonnegative_target",
            "_funding_target",
            "_fallback_value",
            "_binary_train_rate",
            "_binary_event_rate",
            "_binary_transition_rate",
            "_binary_pos_weight",
            "_binary_temperature",
            "_binary_teacher_weight",
            "_binary_event_weight",
            "_teacher_logistic_mix",
            "_teacher_tree_mix",
            "_edgar_source_density",
            "_text_source_density",
            "_funding_log_domain",
            "_funding_source_scaling",
            "_funding_anchor_enabled",
            "_effective_funding_anchor_strength",
            "_effective_task_modulation",
        ):
            if hasattr(self._delegate, name):
                setattr(self, name, getattr(self._delegate, name))


def _detect_binary(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    unique = np.unique(np.round(finite, 6))
    return bool(unique.size <= 2 and set(unique.tolist()).issubset({0.0, 1.0}))


def _detect_nonnegative(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.nanmin(finite) >= 0.0)


__all__ = ["MainlineModuleContract", "SingleModelMainlineWrapper"]