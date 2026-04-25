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
from .investor_mark_encoder import InvestorMarkEncoder, InvestorMarkEncoderSpec
from .lanes.binary_lane import BinaryLaneRuntime, BinaryLaneSpec
from .lanes.funding_lane import FundingLaneRuntime, FundingLaneSpec
from .lanes.investors_lane import InvestorsLaneRuntime, InvestorsLaneSpec, _transition_signal
from .learnable_trunk import LearnableTrunkAdapter
from .sequential_adapter import SequentialTrunkAdapter
from .state_space_trunk import StateSpaceTrunkAdapter
from .objectives import MainlineObjectiveSpec
from .source_memory import SourceMemoryAssembler, SourceMemoryContract
from .variant_profiles import build_delegate_kwargs, get_mainline_variant_profile


PROCESS_STATE_FAMILY_ORDER = (
    "attention_diffusion",
    "credibility_confirmation",
    "screening_selectivity",
    "book_depth_absorption",
    "closure_conversion",
)


@dataclass(frozen=True)
class MainlineModuleContract:
    backbone: SharedTemporalBackboneSpec
    conditioning: ConditioningSchema
    source_memory: SourceMemoryContract
    investor_mark_encoder: InvestorMarkEncoderSpec
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
            "investor_mark_encoder": self.investor_mark_encoder.as_dict(),
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
        resolved_kwargs = dict(self.profile.prototype_overrides)
        resolved_kwargs.update(prototype_kwargs)
        self.use_delegate = bool(resolved_kwargs.pop("use_delegate", self.profile.runtime_mode == "delegate"))
        seed = int(resolved_kwargs.get("seed", 42))
        config = ModelConfig(
            name="SingleModelMainlineWrapper",
            model_type="forecasting",
            params={"variant": variant, **resolved_kwargs},
            optional_dependency="torch",
        )
        super().__init__(config)
        self.prototype_kwargs = dict(resolved_kwargs)
        prototype_kwargs = self.prototype_kwargs
        self.contract = MainlineModuleContract(
            backbone=self._build_backbone_spec(prototype_kwargs),
            conditioning=ConditioningSchema(),
            source_memory=SourceMemoryContract(),
            investor_mark_encoder=InvestorMarkEncoderSpec(),
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
        # Log1p target transform for regression lanes: maps heavy-tailed targets to
        # a learnable manifold; predictions are restored via expm1 at inference time.
        # Enabled by default for funding_raised_usd and investors_count.
        self.enable_log1p_target = bool(prototype_kwargs.get("enable_log1p_target", True))
        self.enable_funding_source_scaling = bool(prototype_kwargs.get("enable_funding_source_scaling", False))
        self.enable_funding_tail_focus = bool(prototype_kwargs.get("enable_funding_tail_focus", False))
        self.funding_tail_weight = float(prototype_kwargs.get("funding_tail_weight", 2.0))
        self.funding_tail_quantile = float(prototype_kwargs.get("funding_tail_quantile", 0.85))
        self.enable_funding_gpd_tail = bool(prototype_kwargs.get("enable_funding_gpd_tail", False))
        self.enable_funding_cqr_interval = bool(prototype_kwargs.get("enable_funding_cqr_interval", False))
        self.funding_cqr_alpha = float(prototype_kwargs.get("funding_cqr_alpha", 0.10))
        # Route G (2026-04-23): explicit flag to bypass the short-circuit gates
        # (positive_jump_rows threshold + jump_target_std degeneracy) so the
        # severity atoms actually run even on sparse-event panels like h=7
        # funding_raised_usd. Intentionally leaves the n<12 gate intact.
        self.enable_funding_forced_hurdle = bool(
            prototype_kwargs.get("enable_funding_forced_hurdle", False)
        )
        self.enable_binary_calibration_shrinkage = bool(
            prototype_kwargs.get("enable_binary_calibration_shrinkage", False)
        )
        self.binary_calibration_shrinkage_target = str(
            prototype_kwargs.get("binary_calibration_shrinkage_target", "auto")
        )
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
        self.enable_investors_mark_features = bool(
            prototype_kwargs.get("enable_investors_mark_features", False)
        )
        self.enable_investors_intensity_baseline = bool(
            prototype_kwargs.get("enable_investors_intensity_baseline", False)
        )
        self.investors_intensity_blend = float(
            prototype_kwargs.get("investors_intensity_blend", 0.5)
        )
        self.enable_investors_shrinkage_gate = bool(
            prototype_kwargs.get("enable_investors_shrinkage_gate", False)
        )
        self.investors_shrinkage_strength = float(
            prototype_kwargs.get("investors_shrinkage_strength", 0.8)
        )
        self.enable_investors_transition_correction = bool(
            prototype_kwargs.get("enable_investors_transition_correction", False)
        )
        self.enable_investors_event_state_features = bool(
            prototype_kwargs.get("enable_investors_event_state_features", False)
        )
        self.enable_investors_selective_event_state_activation = bool(
            prototype_kwargs.get("enable_investors_selective_event_state_activation", False)
        )
        self.enable_process_state_feedback = bool(self.contract.backbone.enable_process_state_feedback)
        self.process_state_feedback_strength = float(self.contract.backbone.process_state_feedback_strength)
        self.process_state_feedback_source_decay = float(self.contract.backbone.process_state_feedback_source_decay)
        self.process_state_feedback_min_horizon = int(self.contract.backbone.process_state_feedback_min_horizon)
        self.process_state_feedback_state_weights = tuple(
            float(value) for value in self.contract.backbone.process_state_feedback_state_weights
        )
        self.process_state_feedback_max_norm_share = float(
            self.contract.backbone.process_state_feedback_max_norm_share
        )
        self.process_state_feedback_predict_scale_cap = float(
            self.contract.backbone.process_state_feedback_predict_scale_cap
        )
        self.investors_event_state_allow_h1 = bool(
            prototype_kwargs.get("investors_event_state_allow_h1", False)
        )
        self.investors_event_state_max_source_presence_share = float(
            prototype_kwargs.get("investors_event_state_max_source_presence_share", 1.0)
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
        # Learnable Sparse MoE trunk (replaces random projection backbone + barrier)
        self.enable_learnable_trunk = bool(prototype_kwargs.get("enable_learnable_trunk", False))
        self._learnable_trunk: LearnableTrunkAdapter | None = None
        if self.enable_learnable_trunk:
            self._learnable_trunk = LearnableTrunkAdapter(
                compact_dim=int(prototype_kwargs.get("learnable_compact_dim", 64)),
                n_experts=int(prototype_kwargs.get("learnable_n_experts", 6)),
                expert_dim=int(prototype_kwargs.get("learnable_expert_dim", 32)),
                top_k=int(prototype_kwargs.get("learnable_top_k", 2)),
                n_epochs=int(prototype_kwargs.get("learnable_n_epochs", 30)),
                random_state=seed,
            )
        # S5-style diagonal state-space trunk (2026-04-22 path-3 remedy
        # for mainline trunk collapse verdict). Activated by
        # `enable_state_space_trunk=True` in the variant profile; drops in
        # for `_learnable_trunk` (same fit/transform API).
        self.enable_state_space_trunk = bool(
            prototype_kwargs.get("enable_state_space_trunk", False)
        )
        self._state_space_trunk: StateSpaceTrunkAdapter | None = None
        if self.enable_state_space_trunk:
            self._state_space_trunk = StateSpaceTrunkAdapter(
                input_dim=int(prototype_kwargs.get("s5_input_dim", 64)),
                d_model=int(prototype_kwargs.get("s5_d_model", 128)),
                d_state=int(prototype_kwargs.get("s5_d_state", 64)),
                n_blocks=int(prototype_kwargs.get("s5_n_blocks", 3)),
                dropout=float(prototype_kwargs.get("s5_dropout", 0.1)),
                max_epochs=int(prototype_kwargs.get("s5_max_epochs", 32)),
                batch_size=int(prototype_kwargs.get("s5_batch_size", 512)),
                device=str(prototype_kwargs.get("s5_device", "cpu")),
            )
            # If both learnable and state-space trunks are enabled the
            # state-space trunk takes precedence (newer architecture).
            if self._learnable_trunk is not None:
                self._learnable_trunk = None
                self.enable_learnable_trunk = False
        # Sequential ED-SSM + MoE trunk (temporal-aware upgrade)
        self.enable_sequential_trunk = bool(prototype_kwargs.get("enable_sequential_trunk", False))
        self._sequential_trunk: SequentialTrunkAdapter | None = None
        if self.enable_sequential_trunk:
            self._sequential_trunk = SequentialTrunkAdapter(
                window_size=int(prototype_kwargs.get("seq_window_size", 30)),
                d_model=int(prototype_kwargs.get("seq_d_model", 64)),
                d_state=int(prototype_kwargs.get("seq_d_state", 16)),
                n_ssm_layers=int(prototype_kwargs.get("seq_n_ssm_layers", 2)),
                d_event=int(prototype_kwargs.get("seq_d_event", 0)),
                compact_dim=int(prototype_kwargs.get("learnable_compact_dim", 64)),
                n_experts=int(prototype_kwargs.get("learnable_n_experts", 6)),
                expert_dim=int(prototype_kwargs.get("learnable_expert_dim", 32)),
                top_k=int(prototype_kwargs.get("learnable_top_k", 2)),
                n_epochs=int(prototype_kwargs.get("seq_n_epochs", 20)),
                batch_size=int(prototype_kwargs.get("seq_batch_size", 512)),
                decoder_branch=str(prototype_kwargs.get("seq_decoder_branch", "legacy")),
                freeze_unified_ssm=bool(prototype_kwargs.get("seq_freeze_unified_ssm", True)),
                random_state=seed,
            )
        self.min_window_history = int(prototype_kwargs.get("min_window_history", 0))
        self.target_windows_per_entity = int(prototype_kwargs.get("target_windows_per_entity", 0))

        self.condition_encoder = MainlineConditionEncoder(self.contract.conditioning)
        self.backbone = SharedTemporalBackbone(self.contract.backbone, random_state=seed)
        self.source_memory = SourceMemoryAssembler(contract=self.contract.source_memory)
        self.investor_mark_encoder = InvestorMarkEncoder(self.contract.investor_mark_encoder)
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
        self._investors_mark_feature_cols: list[str] = []
        self._investors_event_state_feature_cols: list[str] = []
        self._history_feature_cols: list[str] = []
        self._shared_state_dim = 0
        self._lane_state_dim = 0
        self._condition_key: Optional[ConditionKey] = None
        self._history_seed: Optional[pd.DataFrame] = None
        self._backbone_seed: Optional[pd.DataFrame] = None
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
        self._use_log1p_target = False  # set in fit() for non-binary regression targets
        self._funding_source_scale_mean = 0.0
        self._funding_source_scale_max = 0.0
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
        self._effective_investors_event_state_features = False
        self._effective_investors_mark_features = False
        self._process_state_feedback_feature_cols: list[str] = []
        self._effective_process_state_feedback = False
        self._process_state_feedback_activation_reason = "process_state_feedback_not_requested"
        self._process_state_feedback_gate = 0.0
        self._process_state_feedback_source_scale = 0.0
        self._process_state_feedback_dim = 0
        self._process_state_feedback_norm_cap_mean = 1.0
        self._process_state_feedback_predict_cap_scale = 1.0
        self._process_state_feedback_reference_abs_mean = 0.0
        self._process_state_feedback_reference_row_l2_mean = 0.0
        self._process_state_feedback_reference_feature_abs_mean: np.ndarray | None = None
        self._process_state_feedback_current_feature_abs_mean: np.ndarray | None = None
        self._process_state_feedback_projection: np.ndarray | None = None
        self._process_state_feedback_projection_shared_dim = 0
        self._investors_event_state_activation_reason = "event_state_not_requested"
        self._investors_source_activation_reason = "source_switches_disabled"
        self._investors_transition_activation_reason = "transition_not_requested"
        self._investors_mark_activation_reason = "mark_features_not_requested"
        self._investors_mark_mode = "inactive"
        self._investors_mark_nonzero_share = 0.0
        self._investors_mark_coverage_share = 0.0
        self._investors_mark_raw_reference_share = 0.0
        self._investors_mark_proxy_share = 0.0
        self._investors_mark_proxy_only_share = 0.0
        self._investors_mark_summary: Dict[str, float] = {}
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
        self._event_state_card: Dict[str, object] = {}

    def _build_backbone_spec(self, prototype_kwargs: Dict[str, object]) -> SharedTemporalBackboneSpec:
        temporal_windows = prototype_kwargs.get("temporal_state_windows", (3, 7, 30))
        if isinstance(temporal_windows, (int, np.integer)):
            normalized_windows = (max(1, int(temporal_windows)),)
        else:
            normalized_windows = tuple(max(1, int(window)) for window in temporal_windows) or (3, 7, 30)
        raw_feedback_weights = prototype_kwargs.get(
            "process_state_feedback_state_weights",
            (0.25, 0.25, 0.0, 0.0, 0.5),
        )
        normalized_feedback_weights = self._normalize_process_state_feedback_weights(raw_feedback_weights)
        return SharedTemporalBackboneSpec(
            enable_multiscale_temporal_state=bool(
                prototype_kwargs.get("enable_multiscale_temporal_state", False)
            ),
            temporal_state_windows=normalized_windows,
            enable_temporal_state_features=bool(
                prototype_kwargs.get("enable_temporal_state_features", True)
            ),
            enable_spectral_state_features=bool(
                prototype_kwargs.get("enable_spectral_state_features", True)
            ),
            enable_process_state_feedback=bool(
                prototype_kwargs.get("enable_process_state_feedback", False)
            ),
            process_state_feedback_strength=float(
                np.clip(prototype_kwargs.get("process_state_feedback_strength", 0.0), 0.0, 1.0)
            ),
            process_state_feedback_source_decay=float(
                np.clip(prototype_kwargs.get("process_state_feedback_source_decay", 0.65), 0.0, 1.0)
            ),
            process_state_feedback_min_horizon=max(
                1,
                int(prototype_kwargs.get("process_state_feedback_min_horizon", 7)),
            ),
            process_state_feedback_state_weights=normalized_feedback_weights,
            process_state_feedback_max_norm_share=float(
                np.clip(prototype_kwargs.get("process_state_feedback_max_norm_share", 0.02), 0.0, 1.0)
            ),
            process_state_feedback_predict_scale_cap=float(
                max(0.0, float(prototype_kwargs.get("process_state_feedback_predict_scale_cap", 4.0)))
            ),
            enable_hawkes_financing_state=bool(
                prototype_kwargs.get("enable_hawkes_financing_state", False)
            ),
            hawkes_financing_decay_halflives=tuple(
                float(h) for h in prototype_kwargs.get(
                    "hawkes_financing_decay_halflives", (7.0, 30.0, 90.0)
                )
            ),
            hawkes_positive_shock_threshold=float(
                prototype_kwargs.get("hawkes_positive_shock_threshold", 0.5)
            ),
            enable_jump_ode_state=bool(
                prototype_kwargs.get("enable_jump_ode_state", False)
            ),
            jump_ode_dims=int(
                prototype_kwargs.get("jump_ode_dims", 8)
            ),
            normalize_mode=str(
                prototype_kwargs.get("normalize_mode", "robust")
            ),
        )

    def _normalize_process_state_feedback_weights(self, raw: object) -> tuple[float, ...]:
        if isinstance(raw, dict):
            return tuple(float(raw.get(name, 0.0)) for name in PROCESS_STATE_FAMILY_ORDER)
        if isinstance(raw, (int, float, np.number)):
            weight = float(raw)
            return tuple(weight for _ in PROCESS_STATE_FAMILY_ORDER)
        if isinstance(raw, (list, tuple)):
            seq = [float(value) for value in raw[: len(PROCESS_STATE_FAMILY_ORDER)]]
            if len(seq) < len(PROCESS_STATE_FAMILY_ORDER):
                seq.extend([0.0] * (len(PROCESS_STATE_FAMILY_ORDER) - len(seq)))
            return tuple(seq)
        return (0.25, 0.25, 0.0, 0.0, 0.5)

    def _augmented_backbone_layout(self) -> Dict[str, object]:
        layout = self.backbone.describe_state_layout()
        base_shared_dim = int(layout.get("shared_state_dim", 0))
        layout.update(
            {
                "uses_process_state_feedback": bool(self.enable_process_state_feedback),
                "effective_process_state_feedback": bool(self._effective_process_state_feedback),
                "process_state_feedback_dim": int(self._process_state_feedback_dim),
                "process_state_feedback_feature_names": tuple(self._process_state_feedback_feature_cols),
                "process_state_feedback_activation_reason": str(self._process_state_feedback_activation_reason),
                "process_state_feedback_gate": float(self._process_state_feedback_gate),
                "process_state_feedback_source_scale": float(self._process_state_feedback_source_scale),
                "process_state_feedback_norm_cap_mean": float(self._process_state_feedback_norm_cap_mean),
                "process_state_feedback_predict_cap_scale": float(self._process_state_feedback_predict_cap_scale),
                "process_state_feedback_reference_abs_mean": float(self._process_state_feedback_reference_abs_mean),
                "shared_state_dim": int(
                    self._shared_state_dim
                    if self._shared_state_dim > 0
                    else base_shared_dim + int(self._process_state_feedback_dim)
                ),
            }
        )
        return layout

    def describe_contract(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "runtime_mode": self.profile.runtime_mode,
            "delegate_variant": self.profile.delegate_variant,
            "delegate_enabled": self.use_delegate,
            "description": self.profile.description,
            "modules": self.contract.as_dict(),
        }

    def get_routing_info(self) -> Dict[str, object]:
        """Return lane-internal audit diagnostics (Route E instrumentation).

        Added 2026-04-23 Round 10 after Route D atom-isolation audit discovered
        bit-exact MAE across distinct atom configurations, which meant two of
        three atom toggles had no effect on the forward pass. These diagnostics
        expose which lane branch actually executed so future audits can
        compute toggle-effect deltas per branch.
        """
        info: Dict[str, object] = {
            "lane_family": "mainline_funding" if self._funding_target else "mainline_other",
            "lane_variant": self.variant,
        }
        runtime = getattr(self, "funding_lane_runtime", None)
        if runtime is not None and self._funding_target:
            try:
                info["lane_anchor_only_mode"] = bool(getattr(runtime, "_anchor_only_mode", False))
                info["lane_anchor_only_reason"] = str(getattr(runtime, "_anchor_only_reason", "") or "")
                info["lane_trunk_fallback_fitted"] = bool(getattr(runtime, "_trunk_fallback_fitted", False))
                info["lane_positive_jump_rows"] = int(getattr(runtime, "_positive_jump_rows", 0))
                info["lane_jump_target_std"] = float(getattr(runtime, "_jump_target_std", 0.0))
                info["lane_hurdle_engaged"] = bool(getattr(runtime, "_uses_jump_hurdle_head", False))
                info["lane_source_scale_strength"] = float(getattr(runtime, "_source_scale_strength", 0.0))
                info["lane_source_scale_reliability"] = float(getattr(runtime, "_source_scale_reliability", 0.0))
                info["lane_tail_weight_effective"] = (
                    float(self.funding_tail_weight) if self.enable_funding_tail_focus else 0.0
                )
                info["lane_jump_event_rate"] = float(getattr(runtime, "_jump_event_rate", 0.0))
                # 2026-04-24 Round-12 calibration-decision exposure. Observed
                # bit-exact MAE across (source_scaling, tail_focus) probes was
                # caused by the residual-blend grid-search rejecting the
                # severity path (residual_blend=0 → anchor-only prediction),
                # not by short-circuit routing. These fields make that
                # decision directly auditable in metrics.json + sidecar.
                info["lane_residual_blend"] = float(getattr(runtime, "_residual_blend", 0.0))
                info["lane_residual_cap"] = float(getattr(runtime, "_residual_cap", 0.0))
                info["lane_anchor_calibration_mae"] = float(getattr(runtime, "_anchor_calibration_mae", 0.0))
                info["lane_guarded_calibration_mae"] = float(getattr(runtime, "_guarded_calibration_mae", 0.0))
                info["lane_source_scaling_enabled"] = bool(getattr(runtime, "_source_scaling_enabled", False))
                info["lane_calibration_rows"] = int(getattr(runtime, "_calibration_rows", 0))
                # Round-12 L2/K2 audit gates (2026-04-24)
                info["lane_source_scale_silently_dead"] = bool(
                    getattr(runtime, "_source_scale_silently_dead", False)
                )
                info["lane_ss_fallback_active"] = bool(
                    getattr(runtime, "_ss_fallback_active", False)
                )
                info["lane_ss_fallback_env_requested_no_op"] = bool(
                    getattr(runtime, "_ss_fallback_env_requested_no_op", False)
                )
            except Exception:  # noqa: BLE001 — best-effort audit, never block fit
                pass
        return info

    def _write_lane_audit_sidecar(self) -> None:
        """Persist lane forensic state next to predictions when MAINLINE_LANE_AUDIT_DIR set."""
        import json
        import os
        audit_dir = os.environ.get("MAINLINE_LANE_AUDIT_DIR", "")
        if not audit_dir:
            return
        try:
            os.makedirs(audit_dir, exist_ok=True)
            payload = {
                "variant": self.variant,
                "task": getattr(self, "_task_name", ""),
                "target": getattr(self, "_target_name", ""),
                "horizon": int(getattr(self, "_horizon", 0)),
                "ablation": getattr(self, "_ablation_name", ""),
                "routing_info": self.get_routing_info(),
                "atom_flags": {
                    "enable_funding_log_domain": bool(self.enable_funding_log_domain),
                    "enable_funding_source_scaling": bool(self.enable_funding_source_scaling),
                    "enable_funding_tail_focus": bool(self.enable_funding_tail_focus),
                    "funding_tail_weight": float(self.funding_tail_weight),
                    "funding_tail_quantile": float(self.funding_tail_quantile),
                    "enable_funding_gpd_tail": bool(self.enable_funding_gpd_tail),
                },
            }
            fname = f"{self.variant}_{payload['task']}_{payload['target']}_h{payload['horizon']}_lane_audit.json"
            fname = fname.replace("/", "_")
            with open(os.path.join(audit_dir, fname), "w") as fh:
                json.dump(payload, fh, indent=2, default=str, sort_keys=True)
        except Exception:  # noqa: BLE001 — audit is best-effort, never fail fit
            pass

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
        # Log1p target transform: enabled for all non-binary regression targets
        # (funding_raised_usd and investors_count) to tame heavy-tailed distributions.
        self._use_log1p_target = bool(self.enable_log1p_target and not self._binary_target)
        self._effective_task_modulation = bool(self.enable_task_modulation)
        self._condition_key = ConditionKey(
            task=self._task_name,
            target=self._target_name,
            horizon=self._horizon,
            ablation=self._ablation_name,
        )
        self._history_seed = None
        self._backbone_seed = None
        self._process_state_feedback_reference_abs_mean = 0.0
        self._process_state_feedback_reference_row_l2_mean = 0.0
        self._process_state_feedback_reference_feature_abs_mean = None
        self._process_state_feedback_current_feature_abs_mean = None

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

        core_frame = X.reindex(columns=self._core_feature_cols, fill_value=0.0)

        # === Sequential ED-SSM + MoE Trunk path ===
        if self.enable_sequential_trunk and self._sequential_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            raw_df = kwargs.get("train_raw")
            entity_ids = raw_df["entity_id"].reindex(X.index).to_numpy() if raw_df is not None and "entity_id" in raw_df.columns else None
            dates = raw_df["crawled_date_day"].reindex(X.index).to_numpy() if raw_df is not None and "crawled_date_day" in raw_df.columns else None
            trunk_output = self._sequential_trunk.fit_transform(
                core_matrix, y_arr, target_name=self._target_name,
                entity_ids=entity_ids, dates=dates,
            )
            condition_state = self.condition_encoder.broadcast(self._condition_key, len(X))
            source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
            self._source_feature_cols = list(source_frame.columns)
            source_state = source_frame.to_numpy(dtype=np.float32, copy=False)
            self._edgar_source_density = float(source_frame["edgar_active"].mean()) if "edgar_active" in source_frame else 0.0
            self._text_source_density = float(source_frame["text_active"].mean()) if "text_active" in source_frame else 0.0
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            self._shared_state_dim = trunk_output.shape[1]
            self._active_lane_name = self._lane_name_for_target(self._target_name)
            self._lane_state_dim = lane_input.shape[1]
            lane_states = {self._active_lane_name: lane_input}
            event_state_frame = pd.DataFrame(index=X.index)
            self._investors_event_state_feature_cols = []
            self._effective_investors_event_state_features = False
            self._backbone_seed = None
        # === Learnable Trunk path (replaces backbone + barrier) ===
        elif self.enable_learnable_trunk and self._learnable_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            trunk_output = self._learnable_trunk.fit_transform(
                core_matrix, y_arr, target_name=self._target_name,
            )
            condition_state = self.condition_encoder.broadcast(self._condition_key, len(X))
            source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
            self._source_feature_cols = list(source_frame.columns)
            source_state = source_frame.to_numpy(dtype=np.float32, copy=False)
            self._edgar_source_density = float(source_frame["edgar_active"].mean()) if "edgar_active" in source_frame else 0.0
            self._text_source_density = float(source_frame["text_active"].mean()) if "text_active" in source_frame else 0.0
            # Combine trunk output + condition + source as lane state
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            self._shared_state_dim = trunk_output.shape[1]
            self._active_lane_name = self._lane_name_for_target(self._target_name)
            self._lane_state_dim = lane_input.shape[1]
            lane_states = {self._active_lane_name: lane_input}
            # Skip event state / process feedback for learnable trunk path
            event_state_frame = pd.DataFrame(index=X.index)
            self._investors_event_state_feature_cols = []
            self._effective_investors_event_state_features = False
            self._backbone_seed = None
        # === State-Space (S5) Trunk path (2026-04-22 path-3 remedy) ===
        elif self.enable_state_space_trunk and self._state_space_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            trunk_output = self._state_space_trunk.fit_transform(
                core_matrix, y_arr, target_name=self._target_name,
            )
            condition_state = self.condition_encoder.broadcast(self._condition_key, len(X))
            source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
            self._source_feature_cols = list(source_frame.columns)
            source_state = source_frame.to_numpy(dtype=np.float32, copy=False)
            self._edgar_source_density = float(source_frame["edgar_active"].mean()) if "edgar_active" in source_frame else 0.0
            self._text_source_density = float(source_frame["text_active"].mean()) if "text_active" in source_frame else 0.0
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            self._shared_state_dim = trunk_output.shape[1]
            self._active_lane_name = self._lane_name_for_target(self._target_name)
            self._lane_state_dim = lane_input.shape[1]
            lane_states = {self._active_lane_name: lane_input}
            event_state_frame = pd.DataFrame(index=X.index)
            self._investors_event_state_feature_cols = []
            self._effective_investors_event_state_features = False
            self._backbone_seed = None
        else:
            # === Original backbone + barrier path ===
            shared_state = self.backbone.fit_transform(
                core_frame,
                feature_cols=self._core_feature_cols,
                context_frame=runtime_frame,
            )
            self._backbone_seed = self.backbone.build_context_seed(runtime_frame, core_frame)
            condition_state = self.condition_encoder.broadcast(self._condition_key, len(X))
            source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
            self._source_feature_cols = list(source_frame.columns)
            source_state = source_frame.to_numpy(dtype=np.float32, copy=False)
            self._edgar_source_density = float(source_frame["edgar_active"].mean()) if "edgar_active" in source_frame else 0.0
            self._text_source_density = float(source_frame["text_active"].mean()) if "text_active" in source_frame else 0.0
            event_state_frame, shared_state = self._refresh_event_state_card_with_shared_state(
                runtime_frame,
                shared_state=shared_state,
                source_frame=source_frame,
                phase="train",
            )
            self._investors_event_state_feature_cols = list(event_state_frame.columns)
            self._effective_investors_event_state_features = False

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
        investors_mark_frame = self._build_investors_mark_features(runtime_frame)
        self._investors_mark_feature_cols = list(investors_mark_frame.columns)
        self._refresh_investors_mark_regime(investors_mark_frame)
        investors_mark_matrix = investors_mark_frame.to_numpy(dtype=np.float32, copy=False)
        funding_source_scale = self._build_funding_source_scale(source_frame)
        self._funding_source_scale_mean = float(np.mean(funding_source_scale)) if len(funding_source_scale) else 0.0
        self._funding_source_scale_max = float(np.max(funding_source_scale)) if len(funding_source_scale) else 0.0
        self._refresh_investors_source_activation_regime(
            investors_source_frame,
            runtime_frame=runtime_frame,
            history_frame=history_frame,
            anchor=anchor,
            y=y_arr,
        )
        investors_aux_matrix = self._compose_investors_aux_matrix(history_matrix, event_state_frame)

        # ── Log1p target + anchor transform for regression lanes ──────────────
        # Map heavy-tailed regression targets (funding, investors) to log space so
        # downstream HGBR learns on a compact, symmetrically-distributed signal.
        # anchor is also mapped so the jump residual stays in the same manifold.
        if self._use_log1p_target:
            y_lane = np.log1p(np.clip(y_arr, 0.0, None))
            anchor_lane = np.log1p(np.clip(anchor, 0.0, None))
        else:
            y_lane = y_arr
            anchor_lane = anchor

        if self._binary_target:
            self.binary_lane_runtime.fit(
                lane_states["binary"],
                y_arr,
                aux_features=history_matrix,
                enable_calibration_shrinkage=self.enable_binary_calibration_shrinkage,
                calibration_shrinkage_target=self.binary_calibration_shrinkage_target,
                horizon=self._horizon,
                teacher_probs=kwargs.get("teacher_probs"),
                kd_alpha=float(kwargs.get("kd_alpha", 0.0)),
            )
            self._binary_event_rate = float(self.binary_lane_runtime._event_rate)
            self._binary_transition_rate = float(self.binary_lane_runtime._transition_rate)
            self._binary_temperature = float(self.binary_lane_runtime._temperature)
            self._active_lane_model = self.binary_lane_runtime
        elif self._funding_target:
            self.funding_lane_runtime.fit(
                lane_states["funding"],
                y_lane,
                aux_features=history_matrix,
                anchor=anchor_lane,
                source_scale=funding_source_scale,
                use_log_domain=self._funding_log_domain,
                enable_source_scaling=self._funding_source_scaling,
                tail_weight=self.funding_tail_weight if self.enable_funding_tail_focus else 0.0,
                tail_quantile=self.funding_tail_quantile,
                enable_gpd_tail=self.enable_funding_gpd_tail,
                enable_cqr_interval=self.enable_funding_cqr_interval,
                cqr_alpha=self.funding_cqr_alpha,
                force_hurdle=self.enable_funding_forced_hurdle,
            )
            self._active_lane_model = self.funding_lane_runtime
            # Round-12 audit gate #17 (2026-04-24): flag-activation assert.
            # Warns when user sets enable_funding_source_scaling=True but the
            # lane internal _source_scaling_enabled is False (silent death).
            # This is the gate that would have caught §0r's bug one round
            # earlier. Raises only under MAINLINE_STRICT_FLAG_CHECK=1.
            import os as _os_w17
            import logging as _logging_w17
            _log_w17 = _logging_w17.getLogger("narrative.block3.mainline.wrapper")
            _ss_flag = bool(self.enable_funding_source_scaling and self._funding_target)
            _ss_real = bool(getattr(self.funding_lane_runtime, "_source_scaling_enabled", False))
            _tf_flag = bool(self.enable_funding_tail_focus and self._funding_target)
            _tf_real = float(getattr(self.funding_lane_runtime, "_tail_weight", 0.0)) > 0.0
            if (_ss_flag and not _ss_real) or (_tf_flag and not _tf_real):
                msg = (
                    f"[AUDIT #17] Flag-activation mismatch: "
                    f"source_scaling flag={_ss_flag} real={_ss_real}; "
                    f"tail_focus flag={_tf_flag} real={_tf_real}"
                )
                if _os_w17.environ.get("MAINLINE_STRICT_FLAG_CHECK", "0") in ("1", "true", "True"):
                    raise RuntimeError(msg)
                _log_w17.warning(msg)
        else:
            self.investors_lane_runtime.fit(
                lane_states["investors"],
                y_lane,
                aux_features=investors_aux_matrix,
                anchor=anchor_lane,
                source_features=investors_source_matrix,
                mark_features=investors_mark_matrix,
                enable_source_features=self._effective_investors_source_features,
                enable_source_specialists=self._effective_investors_source_specialists,
                enable_source_guard=self._effective_investors_source_guard,
                enable_source_read_policy=self._effective_investors_source_read_policy,
                enable_source_transition_correction=self._effective_investors_transition_correction,
                enable_mark_features=self._effective_investors_mark_features,
                enable_hurdle_head=self._effective_count_hurdle_head,
                enable_count_jump=self._effective_count_jump,
                count_jump_strength=self.count_jump_strength,
                enable_count_sparsity_gate=self._effective_count_sparsity_gate,
                count_sparsity_gate_strength=self.count_sparsity_gate_strength,
                horizon=self._horizon,
                anchor_blend=self.count_anchor_strength if self.enable_count_anchor else 0.0,
                task_name=self._task_name,
                enable_intensity_baseline=self.enable_investors_intensity_baseline,
                intensity_blend=self.investors_intensity_blend,
                enable_shrinkage_gate=self.enable_investors_shrinkage_gate,
                shrinkage_strength=self.investors_shrinkage_strength,
            )
            self._active_lane_model = self.investors_lane_runtime

        self._history_seed = self._collect_history_seed(runtime_frame)
        self.model = self._active_lane_model
        self._fitted = True
        # Route E instrumentation: persist lane audit sidecar if env configured.
        self._write_lane_audit_sidecar()
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
                "binary_calibration_shrinkage": self.enable_binary_calibration_shrinkage,
                "funding_anchor": self._funding_anchor_enabled,
                "funding_tail_focus": self.enable_funding_tail_focus,
                "funding_source_scaling_guard": self._funding_source_scaling,
                "count_source_routing": self.enable_count_source_routing,
                "count_source_specialists": self._effective_investors_source_specialists,
                "investors_transition_correction": self._effective_investors_transition_correction,
                "investors_event_state_features": self._effective_investors_event_state_features,
                "shared_process_state_feedback": self._effective_process_state_feedback,
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
                "event_state_schema_version": self.contract.backbone.event_state_schema_version,
                "backbone_layout": self._augmented_backbone_layout(),
            },
            "event_state_trunk": dict(self._event_state_card),
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
                "backbone_seed_rows": 0 if self._backbone_seed is None else int(len(self._backbone_seed)),
            },
            "binary_process_contract": {
                "process_family": "hazard_prior_plus_calibration",
                "uses_neural_hazard_head": bool(self.binary_lane_runtime._model is not None),
                "uses_hazard_adapter": bool(self.binary_lane_runtime._uses_hazard_adapter),
                "uses_hazard_space_calibration": bool(self.binary_lane_runtime._uses_hazard_space_calibration),
                "hazard_calibration_horizon": int(self.binary_lane_runtime._hazard_calibration_horizon),
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
                "selected_survival_nll": float(self.binary_lane_runtime._selected_metrics.get("survival_nll", 0.0)),
                "identity_survival_nll": float(self.binary_lane_runtime._identity_metrics.get("survival_nll", 0.0)),
                "calibration_shrinkage_enabled": bool(self.enable_binary_calibration_shrinkage),
                "calibration_shrinkage_target": str(self.binary_lane_runtime._calibration_shrinkage_target),
                "calibration_shrinkage_strength": float(self.binary_lane_runtime._calibration_shrinkage_strength),
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
                "requested_event_state_features": self.enable_investors_event_state_features,
                "selective_event_state_contract_enabled": self.enable_investors_selective_event_state_activation,
                "event_state_allow_h1": self.investors_event_state_allow_h1,
                "event_state_max_source_presence_share": self.investors_event_state_max_source_presence_share,
                "effective_source_features": self._effective_investors_source_features,
                "effective_source_specialists": self._effective_investors_source_specialists,
                "effective_source_guard": self._effective_investors_source_guard,
                "effective_source_read_policy": self._effective_investors_source_read_policy,
                "effective_transition_correction": self._effective_investors_transition_correction,
                "effective_event_state_features": self._effective_investors_event_state_features,
                "event_state_activation_reason": self._investors_event_state_activation_reason,
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
            "investor_mark_activation": {
                "requested_mark_features": bool(self.enable_investors_mark_features),
                "effective_mark_features": bool(self._effective_investors_mark_features),
                "activation_reason": self._investors_mark_activation_reason,
                "mark_mode": self._investors_mark_mode,
                "mark_feature_count": int(len(self._investors_mark_feature_cols)),
                "mark_nonzero_share": float(self._investors_mark_nonzero_share),
                "mark_coverage_share": float(self._investors_mark_coverage_share),
                "raw_reference_mark_share": float(self._investors_mark_raw_reference_share),
                "proxy_mark_share": float(self._investors_mark_proxy_share),
                "proxy_only_mark_share": float(self._investors_mark_proxy_only_share),
                "mark_summary": dict(self._investors_mark_summary),
            },
            "investors_process_contract": {
                "horizon_subregime": self._investors_horizon_subregime,
                "effective_count_hurdle_head": self._effective_count_hurdle_head,
                "effective_count_jump": self._effective_count_jump,
                "effective_count_sparsity_gate": self._effective_count_sparsity_gate,
                "requested_count_hurdle_head": self.enable_count_hurdle_head,
                "requested_count_jump": self.enable_count_jump,
                "requested_count_sparsity_gate": self.enable_count_sparsity_gate,
                "requested_event_state_features": self.enable_investors_event_state_features,
                "effective_event_state_features": self._effective_investors_event_state_features,
                "requested_mark_features": self.enable_investors_mark_features,
                "effective_mark_features": self._effective_investors_mark_features,
                "event_state_activation_reason": self._investors_event_state_activation_reason,
                "mark_activation_reason": self._investors_mark_activation_reason,
                "event_state_feature_count": int(len(self._investors_event_state_feature_cols)),
                "mark_feature_count": int(len(self._investors_mark_feature_cols)),
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
                **self.investors_lane_runtime.describe_intensity(),
            },
            "funding_process_contract": {
                "process_family": "anchor_plus_jump_hurdle",
                "anchor_enabled": self._funding_anchor_enabled,
                "funding_log_domain": self._funding_log_domain,
                "funding_source_scaling": self._funding_source_scaling,
                "funding_tail_focus": bool(self.enable_funding_tail_focus),
                "funding_tail_weight": float(self.funding_tail_weight),
                "funding_tail_quantile": float(self.funding_tail_quantile),
                "train_source_scale_mean": float(self._funding_source_scale_mean),
                "train_source_scale_max": float(self._funding_source_scale_max),
                "lane_uses_jump_hurdle_head": bool(self.funding_lane_runtime._uses_jump_hurdle_head),
                "lane_jump_event_rate": float(self.funding_lane_runtime._jump_event_rate),
                "lane_positive_jump_rows": int(self.funding_lane_runtime._positive_jump_rows),
                "lane_positive_jump_median": float(self.funding_lane_runtime._positive_jump_median),
                "lane_jump_floor": float(self.funding_lane_runtime._jump_floor),
                "lane_residual_blend": float(self.funding_lane_runtime._residual_blend),
                "lane_process_blend": float(self.funding_lane_runtime._residual_blend),
                "lane_source_scale_strength": float(self.funding_lane_runtime._source_scale_strength),
                "lane_source_scale_reliability": float(self.funding_lane_runtime._source_scale_reliability),
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
                "gpd_enabled": bool(self.enable_funding_gpd_tail),
                "cqr_enabled": bool(self.enable_funding_cqr_interval),
                **self.funding_lane_runtime.describe_tail(),
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
        core_frame = feature_frame.reindex(columns=self._core_feature_cols, fill_value=0.0)

        # Default: empty frame for trunk paths that skip event-state refresh.
        event_state_frame = pd.DataFrame()

        # === Sequential ED-SSM + MoE Trunk predict path ===
        if self.enable_sequential_trunk and self._sequential_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            raw_df = kwargs.get("test_raw")
            entity_ids = raw_df["entity_id"].reindex(X.index).to_numpy() if raw_df is not None and "entity_id" in raw_df.columns else None
            dates = raw_df["crawled_date_day"].reindex(X.index).to_numpy() if raw_df is not None and "crawled_date_day" in raw_df.columns else None
            trunk_output = self._sequential_trunk.transform(
                core_matrix, target_name=self._target_name,
                entity_ids=entity_ids, dates=dates,
            )
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
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            lane_states = {self._active_lane_name: lane_input}
        # === Learnable Trunk predict path ===
        elif self.enable_learnable_trunk and self._learnable_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            trunk_output = self._learnable_trunk.transform(
                core_matrix, target_name=self._target_name,
            )
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
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            lane_states = {self._active_lane_name: lane_input}
        # === State-Space (S5) Trunk predict path ===
        elif self.enable_state_space_trunk and self._state_space_trunk is not None:
            core_matrix = core_frame.to_numpy(dtype=np.float32, copy=False)
            trunk_output = self._state_space_trunk.transform(
                core_matrix, target_name=self._target_name,
            )
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
            lane_input = np.concatenate([trunk_output, condition_state, source_state], axis=1).astype(np.float32)
            lane_states = {self._active_lane_name: lane_input}
        else:
            # === Original backbone + barrier predict path ===
            shared_state = self.backbone.transform(
                core_frame,
                context_frame=runtime_frame,
                seed_frame=self._backbone_seed,
            )
            condition_key = ConditionKey(
                task=kwargs.get("task", self._task_name),
                target=kwargs.get("target", self._target_name),
                horizon=int(kwargs.get("horizon", self._horizon)),
                ablation=kwargs.get("ablation", self._ablation_name),
            )
            condition_state = self.condition_encoder.broadcast(condition_key, len(feature_frame))
            source_layout = self.source_memory.infer_layout(runtime_frame)
            source_frame = self.source_memory.build_runtime_features(runtime_frame, layout=source_layout)
            event_state_frame, shared_state = self._refresh_event_state_card_with_shared_state(
                runtime_frame,
                shared_state=shared_state,
                source_frame=source_frame,
                phase="test",
            )
            source_state = source_frame.reindex(columns=self._source_feature_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)
            lane_states = self.barrier.split(shared_state, condition_state, source_state)
        history_frame = self._build_target_history_features(runtime_frame, include_seed=True)
        history_matrix = history_frame.reindex(columns=self._history_feature_cols, fill_value=0.0).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        anchor = self._resolve_anchor(history_frame)
        # Apply log1p to anchor in predict path so the lane stays in the same
        # manifold it was trained in (log space for non-binary regression targets).
        anchor_pred = np.log1p(np.clip(anchor, 0.0, None)) if self._use_log1p_target else anchor
        investors_source_frame = self._build_investors_source_features(source_frame)
        investors_aux_matrix = self._compose_investors_aux_matrix(history_matrix, event_state_frame)
        investors_source_matrix = investors_source_frame.reindex(
            columns=self._investors_source_feature_cols,
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)
        investors_mark_matrix = self._build_investors_mark_features(runtime_frame).reindex(
            columns=self._investors_mark_feature_cols,
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)

        if self._active_lane_name == "binary":
            preds = self.binary_lane_runtime.predict(lane_states["binary"], aux_features=history_matrix)
            return np.clip(preds, 0.0, 1.0).astype(np.float64, copy=False)
        if self._active_lane_name == "funding":
            funding_source_scale = self._build_funding_source_scale(source_frame)
            preds = self.funding_lane_runtime.predict(
                lane_states["funding"],
                aux_features=history_matrix,
                anchor=anchor_pred,
                source_scale=funding_source_scale,
            )
            # Restore from log space → physical scale.
            if self._use_log1p_target:
                preds = np.expm1(np.clip(preds, 0.0, None))
            return np.clip(preds, 0.0, None).astype(np.float64, copy=False)
        preds = self.investors_lane_runtime.predict(
            lane_states["investors"],
            aux_features=investors_aux_matrix,
            anchor=anchor_pred,
            source_features=investors_source_matrix,
            mark_features=investors_mark_matrix,
            enable_source_features=self._effective_investors_source_features,
            enable_source_specialists=self._effective_investors_source_specialists,
            enable_source_guard=self._effective_investors_source_guard,
            enable_source_read_policy=self._effective_investors_source_read_policy,
            enable_source_transition_correction=self._effective_investors_transition_correction,
            enable_mark_features=self._effective_investors_mark_features,
        )
        # Restore from log space → physical scale.
        if self._use_log1p_target:
            preds = np.expm1(np.clip(preds, 0.0, None))
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

    def _build_investors_mark_features(self, runtime_frame: pd.DataFrame) -> pd.DataFrame:
        return self.investor_mark_encoder.build_mark_frame(runtime_frame).astype(np.float32)

    def _build_funding_source_scale(self, source_frame: pd.DataFrame) -> np.ndarray:
        if source_frame.empty:
            return np.zeros(0, dtype=np.float64)

        def _source_col(name: str) -> np.ndarray:
            return pd.to_numeric(
                source_frame.get(name, pd.Series(0.0, index=source_frame.index, dtype=np.float32)),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=np.float64, copy=False)

        edgar_active = np.clip(_source_col("edgar_active"), 0.0, 1.0)
        text_active = np.clip(_source_col("text_active"), 0.0, 1.0)
        edgar_nonzero = np.clip(_source_col("edgar_nonzero_share"), 0.0, 1.0)
        text_nonzero = np.clip(_source_col("text_nonzero_share"), 0.0, 1.0)
        scale = 0.35 * edgar_active + 0.35 * text_active + 0.15 * edgar_nonzero + 0.15 * text_nonzero
        return np.clip(scale, 0.0, 1.0).astype(np.float64, copy=False)

    def _refresh_event_state_card_with_shared_state(
        self,
        runtime_frame: pd.DataFrame,
        *,
        shared_state: np.ndarray,
        source_frame: pd.DataFrame,
        phase: str = "train",
    ) -> tuple[pd.DataFrame, np.ndarray]:
        panel = self._build_event_state_panel(runtime_frame)
        event_state_frame = self._build_event_state_feature_frame(panel, source_frame)
        aligned_event_state_frame = event_state_frame.reindex(runtime_frame.index, fill_value=0.0).astype(np.float32)
        adjusted_shared_state = np.asarray(shared_state, dtype=np.float32)
        self._process_state_feedback_feature_cols = (
            [f"process_feedback_{state}" for state in PROCESS_STATE_FAMILY_ORDER]
            if self.enable_process_state_feedback
            else []
        )
        self._process_state_feedback_dim = len(self._process_state_feedback_feature_cols)
        self._effective_process_state_feedback = False
        self._process_state_feedback_activation_reason = (
            "process_state_feedback_not_requested"
            if not self.enable_process_state_feedback
            else "process_state_feedback_uninitialized"
        )
        self._process_state_feedback_gate = 0.0
        self._process_state_feedback_source_scale = 0.0
        self._process_state_feedback_norm_cap_mean = 1.0
        self._process_state_feedback_predict_cap_scale = 1.0
        self._process_state_feedback_current_feature_abs_mean = None

        source_atoms = self._summarize_event_state_source_topology(source_frame)
        shared_state_atoms = self._summarize_shared_state_geometry(adjusted_shared_state)
        process_bundle = self._build_process_state_feature_bundle(
            runtime_frame,
            event_state_frame=aligned_event_state_frame,
            shared_state_atoms=shared_state_atoms,
        )
        if self.enable_process_state_feedback:
            feedback_block = self._build_process_state_feedback_block(
                process_bundle,
                source_presence_share=float(source_atoms.get("source_presence_share", 0.0)),
                base_shared_state=adjusted_shared_state,
                phase=phase,
            )
            if feedback_block.shape[1] > 0:
                self._process_state_feedback_current_feature_abs_mean = np.mean(
                    np.abs(feedback_block),
                    axis=0,
                ).astype(np.float32, copy=False)
                adjusted_shared_state = self._inject_process_state_feedback(
                    adjusted_shared_state,
                    feedback_block=feedback_block,
                )
            shared_state_atoms = self._summarize_shared_state_geometry(adjusted_shared_state)
            process_bundle = self._build_process_state_feature_bundle(
                runtime_frame,
                event_state_frame=aligned_event_state_frame,
                shared_state_atoms=shared_state_atoms,
            )
        self._event_state_card = {
            "schema_version": self.contract.backbone.event_state_schema_version,
            "atoms": list(self.contract.backbone.event_state_atoms),
            "coverage": self._summarize_event_state_coverage(panel),
            "phase_atoms": self._summarize_event_state_phase(panel),
            "boundary_atoms": self._summarize_event_state_boundaries(aligned_event_state_frame),
            "transition_atoms": self._summarize_event_state_transitions(panel),
            "persistence_atoms": self._summarize_event_state_persistence(panel),
            "goal_atoms": self._summarize_event_state_goal_progress(aligned_event_state_frame),
            "source_atoms": source_atoms,
            "source_flow_atoms": self._summarize_event_state_source_flow(aligned_event_state_frame),
            "shared_state_atoms": shared_state_atoms,
            "process_state_atoms": self._summarize_process_state_bundle(process_bundle),
        }
        return aligned_event_state_frame, adjusted_shared_state

    def _refresh_event_state_card(
        self,
        runtime_frame: pd.DataFrame,
        *,
        shared_state: np.ndarray,
        source_frame: pd.DataFrame,
        phase: str = "train",
    ) -> pd.DataFrame:
        event_state_frame, _ = self._refresh_event_state_card_with_shared_state(
            runtime_frame,
            shared_state=shared_state,
            source_frame=source_frame,
            phase=phase,
        )
        return event_state_frame

    def _build_event_state_panel(self, runtime_frame: pd.DataFrame) -> pd.DataFrame:
        panel = pd.DataFrame(index=runtime_frame.index)
        if "entity_id" in runtime_frame.columns:
            panel["entity_id"] = runtime_frame["entity_id"].astype(str)
        else:
            panel["entity_id"] = pd.Series(
                [f"row_{idx}" for idx in range(len(runtime_frame))],
                index=runtime_frame.index,
                dtype="object",
            )
        panel["crawled_date_day"] = pd.to_datetime(runtime_frame.get("crawled_date_day"), errors="coerce")
        panel["_row_order"] = np.arange(len(panel), dtype=np.int64)
        panel["funding_raised_usd"] = pd.to_numeric(
            runtime_frame.get("funding_raised_usd", pd.Series(0.0, index=runtime_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        panel["investors_count"] = pd.to_numeric(
            runtime_frame.get("investors_count", pd.Series(0.0, index=runtime_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        panel["is_funded"] = pd.to_numeric(
            runtime_frame.get("is_funded", pd.Series(0.0, index=runtime_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        panel["funding_goal_usd"] = pd.to_numeric(
            runtime_frame.get("funding_goal_usd", pd.Series(np.nan, index=runtime_frame.index)),
            errors="coerce",
        )
        panel.sort_values(["entity_id", "crawled_date_day", "_row_order"], inplace=True, kind="mergesort")
        return panel

    def _build_event_state_feature_frame(self, panel: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame(index=panel.index)

        source_aligned = source_frame.reindex(panel.index, fill_value=0.0).copy()
        work = panel.copy()
        work["funding_active"] = (work["funding_raised_usd"] > 0.0).astype(np.float32)
        work["investors_active"] = (work["investors_count"] > 0.0).astype(np.float32)
        work["funded_active"] = (work["is_funded"] > 0.5).astype(np.float32)
        work["goal_known"] = (work["funding_goal_usd"].fillna(0.0) > 0.0).astype(np.float32)
        goal_ratio = (work["funding_raised_usd"] / work["funding_goal_usd"]).replace([np.inf, -np.inf], np.nan)
        work["goal_ratio"] = goal_ratio.fillna(0.0).clip(lower=0.0, upper=4.0).astype(np.float32)
        goal_gap = (work["funding_goal_usd"].fillna(0.0) - work["funding_raised_usd"].fillna(0.0)).clip(lower=0.0)
        work["goal_gap_log"] = np.log1p(goal_gap).astype(np.float32)
        work["goal_crossed"] = ((goal_ratio >= 1.0) & (work["goal_known"] > 0.5)).astype(np.float32)

        work["funding_diff"] = work.groupby("entity_id", sort=False)["funding_raised_usd"].diff().fillna(0.0)
        work["investors_diff"] = work.groupby("entity_id", sort=False)["investors_count"].diff().fillna(0.0)
        work["funding_active_prev"] = work.groupby("entity_id", sort=False)["funding_active"].shift(1).fillna(work["funding_active"])
        work["investors_active_prev"] = work.groupby("entity_id", sort=False)["investors_active"].shift(1).fillna(work["investors_active"])
        work["funded_prev"] = work.groupby("entity_id", sort=False)["funded_active"].shift(1).fillna(work["funded_active"])
        work["goal_crossed_prev"] = work.groupby("entity_id", sort=False)["goal_crossed"].shift(1).fillna(work["goal_crossed"])

        work["funding_boundary_up"] = ((work["funding_active_prev"] <= 0.5) & (work["funding_active"] > 0.5)).astype(np.float32)
        work["funding_boundary_down"] = ((work["funding_active_prev"] > 0.5) & (work["funding_active"] <= 0.5)).astype(np.float32)
        work["investors_boundary_up"] = ((work["investors_active_prev"] <= 0.5) & (work["investors_active"] > 0.5)).astype(np.float32)
        work["investors_boundary_down"] = ((work["investors_active_prev"] > 0.5) & (work["investors_active"] <= 0.5)).astype(np.float32)
        work["funded_flip_up"] = ((work["funded_prev"] <= 0.5) & (work["funded_active"] > 0.5)).astype(np.float32)
        work["funded_flip_down"] = ((work["funded_prev"] > 0.5) & (work["funded_active"] <= 0.5)).astype(np.float32)
        work["goal_cross_up"] = ((work["goal_crossed_prev"] <= 0.5) & (work["goal_crossed"] > 0.5)).astype(np.float32)
        work["goal_cross_down"] = ((work["goal_crossed_prev"] > 0.5) & (work["goal_crossed"] <= 0.5)).astype(np.float32)

        funding_jump = work["funding_diff"].clip(lower=0.0)
        investor_jump = work["investors_diff"].clip(lower=0.0)
        work["funding_jump_log"] = np.log1p(funding_jump).astype(np.float32)
        work["investor_jump_log"] = np.log1p(investor_jump).astype(np.float32)
        work["investor_jump_flag"] = (investor_jump > 1e-6).astype(np.float32)

        edgar_active = pd.to_numeric(
            source_aligned.get("edgar_active", pd.Series(0.0, index=work.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0)
        text_active = pd.to_numeric(
            source_aligned.get("text_active", pd.Series(0.0, index=work.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0)
        work["edgar_active"] = edgar_active.astype(np.float32)
        work["text_active"] = text_active.astype(np.float32)
        work["edgar_active_prev"] = work.groupby("entity_id", sort=False)["edgar_active"].shift(1).fillna(work["edgar_active"])
        work["text_active_prev"] = work.groupby("entity_id", sort=False)["text_active"].shift(1).fillna(work["text_active"])
        work["edgar_arrival"] = ((work["edgar_active_prev"] <= 0.5) & (work["edgar_active"] > 0.5)).astype(np.float32)
        work["edgar_decay"] = ((work["edgar_active_prev"] > 0.5) & (work["edgar_active"] <= 0.5)).astype(np.float32)
        work["text_arrival"] = ((work["text_active_prev"] <= 0.5) & (work["text_active"] > 0.5)).astype(np.float32)
        work["text_decay"] = ((work["text_active_prev"] > 0.5) & (work["text_active"] <= 0.5)).astype(np.float32)
        work["source_active_count"] = (work["edgar_active"] + work["text_active"]).astype(np.float32)
        work["source_boundary_flag"] = (
            (work["edgar_arrival"] > 0.5)
            | (work["edgar_decay"] > 0.5)
            | (work["text_arrival"] > 0.5)
            | (work["text_decay"] > 0.5)
        ).astype(np.float32)

        edgar_recency = pd.to_numeric(
            source_aligned.get("edgar_recency_days", pd.Series(0.0, index=work.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        text_recency = pd.to_numeric(
            source_aligned.get("text_recency_days", pd.Series(0.0, index=work.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0)
        work["source_freshness"] = (
            0.5 / (1.0 + edgar_recency.to_numpy(dtype=np.float64))
            + 0.5 / (1.0 + text_recency.to_numpy(dtype=np.float64))
        ).astype(np.float32)

        work["phase_transition_flag"] = (
            (work["funding_boundary_up"] > 0.5)
            | (work["funding_boundary_down"] > 0.5)
            | (work["investors_boundary_up"] > 0.5)
            | (work["investors_boundary_down"] > 0.5)
            | (work["funded_flip_up"] > 0.5)
            | (work["funded_flip_down"] > 0.5)
            | (work["goal_cross_up"] > 0.5)
            | (work["goal_cross_down"] > 0.5)
            | (work["source_boundary_flag"] > 0.5)
        ).astype(np.float32)

        cols = [
            "funding_active",
            "investors_active",
            "funded_active",
            "goal_known",
            "goal_ratio",
            "goal_gap_log",
            "goal_crossed",
            "funding_boundary_up",
            "funding_boundary_down",
            "investors_boundary_up",
            "investors_boundary_down",
            "funded_flip_up",
            "funded_flip_down",
            "goal_cross_up",
            "goal_cross_down",
            "funding_jump_log",
            "investor_jump_log",
            "investor_jump_flag",
            "edgar_arrival",
            "edgar_decay",
            "text_arrival",
            "text_decay",
            "source_active_count",
            "source_boundary_flag",
            "source_freshness",
            "phase_transition_flag",
        ]
        return work[cols].astype(np.float32)

    def _compose_investors_aux_matrix(
        self,
        history_matrix: np.ndarray,
        event_state_frame: pd.DataFrame,
    ) -> np.ndarray:
        if not self._effective_investors_event_state_features:
            return history_matrix
        event_state_matrix = event_state_frame.reindex(
            columns=self._investors_event_state_feature_cols,
            fill_value=0.0,
        ).to_numpy(dtype=np.float32, copy=False)
        if history_matrix.size == 0:
            return event_state_matrix
        return np.concatenate([history_matrix, event_state_matrix], axis=1).astype(np.float32, copy=False)

    def _summarize_event_state_coverage(self, panel: pd.DataFrame) -> Dict[str, object]:
        if panel.empty:
            return {"rows": 0, "entities": 0, "mean_rows_per_entity": 0.0, "median_rows_per_entity": 0.0}
        entity_sizes = panel.groupby("entity_id", sort=False).size().astype(np.float64)
        return {
            "rows": int(len(panel)),
            "entities": int(entity_sizes.shape[0]),
            "mean_rows_per_entity": float(entity_sizes.mean()) if len(entity_sizes) else 0.0,
            "median_rows_per_entity": float(entity_sizes.median()) if len(entity_sizes) else 0.0,
        }

    def _summarize_event_state_phase(self, panel: pd.DataFrame) -> Dict[str, object]:
        if panel.empty:
            return {
                "funding_active_share": 0.0,
                "investors_active_share": 0.0,
                "funded_share": 0.0,
                "joint_financing_active_share": 0.0,
                "joint_funding_investor_share": 0.0,
                "binary_without_amount_share": 0.0,
                "goal_known_share": 0.0,
                "goal_reached_share": 0.0,
            }
        funding_active = panel["funding_raised_usd"] > 0.0
        investors_active = panel["investors_count"] > 0.0
        funded_active = panel["is_funded"] > 0.5
        goal_known = panel["funding_goal_usd"].fillna(0.0) > 0.0
        goal_ratio = (panel["funding_raised_usd"] / panel["funding_goal_usd"]).replace([np.inf, -np.inf], np.nan)
        return {
            "funding_active_share": self._mean_or_zero(funding_active),
            "investors_active_share": self._mean_or_zero(investors_active),
            "funded_share": self._mean_or_zero(funded_active),
            "joint_financing_active_share": self._mean_or_zero(funding_active | investors_active | funded_active),
            "joint_funding_investor_share": self._mean_or_zero(funding_active & investors_active),
            "binary_without_amount_share": self._mean_or_zero(funded_active & ~funding_active),
            "goal_known_share": self._mean_or_zero(goal_known),
            "goal_reached_share": self._mean_or_zero(goal_ratio.loc[goal_known] >= 1.0),
        }

    def _summarize_event_state_boundaries(self, event_state_frame: pd.DataFrame) -> Dict[str, object]:
        if event_state_frame.empty:
            return {
                "funding_boundary_up_share": 0.0,
                "funding_boundary_down_share": 0.0,
                "investors_boundary_up_share": 0.0,
                "investors_boundary_down_share": 0.0,
                "funded_flip_up_share": 0.0,
                "funded_flip_down_share": 0.0,
                "phase_transition_share": 0.0,
            }
        return {
            "funding_boundary_up_share": self._mean_or_zero(event_state_frame["funding_boundary_up"]),
            "funding_boundary_down_share": self._mean_or_zero(event_state_frame["funding_boundary_down"]),
            "investors_boundary_up_share": self._mean_or_zero(event_state_frame["investors_boundary_up"]),
            "investors_boundary_down_share": self._mean_or_zero(event_state_frame["investors_boundary_down"]),
            "funded_flip_up_share": self._mean_or_zero(event_state_frame["funded_flip_up"]),
            "funded_flip_down_share": self._mean_or_zero(event_state_frame["funded_flip_down"]),
            "phase_transition_share": self._mean_or_zero(event_state_frame["phase_transition_flag"]),
        }

    def _summarize_event_state_transitions(self, panel: pd.DataFrame) -> Dict[str, object]:
        if panel.empty:
            return {
                "valid_lag_rows": 0,
                "funding_jump_share": 0.0,
                "investor_jump_share": 0.0,
                "joint_jump_share": 0.0,
                "funded_up_share": 0.0,
                "funded_down_share": 0.0,
                "state_flip_share": 0.0,
                "funding_jump_mean": 0.0,
                "investor_jump_mean": 0.0,
            }
        work = panel.copy()
        work["funding_diff"] = work.groupby("entity_id", sort=False)["funding_raised_usd"].diff()
        work["investors_diff"] = work.groupby("entity_id", sort=False)["investors_count"].diff()
        work["funded_prev"] = work.groupby("entity_id", sort=False)["is_funded"].shift(1)
        valid = work.dropna(subset=["funding_diff", "investors_diff", "funded_prev"]).copy()
        if valid.empty:
            return {
                "valid_lag_rows": 0,
                "funding_jump_share": 0.0,
                "investor_jump_share": 0.0,
                "joint_jump_share": 0.0,
                "funded_up_share": 0.0,
                "funded_down_share": 0.0,
                "state_flip_share": 0.0,
                "funding_jump_mean": 0.0,
                "investor_jump_mean": 0.0,
            }
        funding_jump = valid["funding_diff"] > 1e-6
        investor_jump = valid["investors_diff"] > 1e-6
        funded_up = (valid["funded_prev"] <= 0.5) & (valid["is_funded"] > 0.5)
        funded_down = (valid["funded_prev"] > 0.5) & (valid["is_funded"] <= 0.5)
        return {
            "valid_lag_rows": int(len(valid)),
            "funding_jump_share": self._mean_or_zero(funding_jump),
            "investor_jump_share": self._mean_or_zero(investor_jump),
            "joint_jump_share": self._mean_or_zero(funding_jump & investor_jump),
            "funded_up_share": self._mean_or_zero(funded_up),
            "funded_down_share": self._mean_or_zero(funded_down),
            "state_flip_share": self._mean_or_zero(funding_jump | investor_jump | funded_up | funded_down),
            "funding_jump_mean": self._mean_or_zero(valid.loc[funding_jump, "funding_diff"]),
            "investor_jump_mean": self._mean_or_zero(valid.loc[investor_jump, "investors_diff"]),
        }

    def _summarize_event_state_persistence(self, panel: pd.DataFrame) -> Dict[str, object]:
        if panel.empty:
            return {
                "entity_count": 0,
                "dynamic_entity_share": 0.0,
                "funding_monotone_entity_share": 0.0,
                "investor_monotone_entity_share": 0.0,
                "funded_absorbing_entity_share": 0.0,
                "financing_active_entity_share": 0.0,
            }
        work = panel.copy()
        work["funding_active"] = work["funding_raised_usd"] > 0.0
        work["investors_active"] = work["investors_count"] > 0.0
        work["funding_diff"] = work.groupby("entity_id", sort=False)["funding_raised_usd"].diff()
        work["investors_diff"] = work.groupby("entity_id", sort=False)["investors_count"].diff()
        work["funded_prev"] = work.groupby("entity_id", sort=False)["is_funded"].shift(1)
        work["funding_active_prev"] = work.groupby("entity_id", sort=False)["funding_active"].shift(1)
        work["investors_active_prev"] = work.groupby("entity_id", sort=False)["investors_active"].shift(1)
        work["funded_down"] = ((work["funded_prev"] > 0.5) & (work["is_funded"] <= 0.5)).fillna(False)
        funded_change = (work["funded_prev"].fillna(work["is_funded"]) - work["is_funded"]).abs() > 1e-6
        funding_phase_change = work["funding_active_prev"].fillna(work["funding_active"]) != work["funding_active"]
        investors_phase_change = (
            work["investors_active_prev"].fillna(work["investors_active"]) != work["investors_active"]
        )
        work["any_change"] = (
            (work["investors_diff"].fillna(0.0).abs() > 1e-6)
            | funded_change
            | funding_phase_change
            | investors_phase_change
        )
        per_entity = pd.DataFrame(
            {
                "funding_monotone": ~work.groupby("entity_id", sort=False)["funding_diff"].apply(
                    lambda series: bool((series.dropna() < -1e-6).any())
                ),
                "investor_monotone": ~work.groupby("entity_id", sort=False)["investors_diff"].apply(
                    lambda series: bool((series.dropna() < -1e-6).any())
                ),
                "funded_absorbing": work.groupby("entity_id", sort=False)["funded_down"].sum() == 0,
                "dynamic": work.groupby("entity_id", sort=False)["any_change"].any(),
                "financing_active": work.groupby("entity_id", sort=False).apply(
                    lambda entity: bool(
                        ((entity["funding_raised_usd"] > 0.0)
                        | (entity["investors_count"] > 0.0)
                        | (entity["is_funded"] > 0.5)).any()
                    )
                ),
            }
        )
        return {
            "entity_count": int(len(per_entity)),
            "dynamic_entity_share": self._mean_or_zero(per_entity["dynamic"]),
            "funding_monotone_entity_share": self._mean_or_zero(per_entity["funding_monotone"]),
            "investor_monotone_entity_share": self._mean_or_zero(per_entity["investor_monotone"]),
            "funded_absorbing_entity_share": self._mean_or_zero(per_entity["funded_absorbing"]),
            "financing_active_entity_share": self._mean_or_zero(per_entity["financing_active"]),
        }

    def _summarize_event_state_goal_progress(self, event_state_frame: pd.DataFrame) -> Dict[str, object]:
        if event_state_frame.empty:
            return {
                "goal_cross_up_share": 0.0,
                "goal_cross_down_share": 0.0,
                "goal_crossed_share": 0.0,
                "goal_ratio_mean": 0.0,
                "goal_gap_log_mean": 0.0,
            }
        return {
            "goal_cross_up_share": self._mean_or_zero(event_state_frame["goal_cross_up"]),
            "goal_cross_down_share": self._mean_or_zero(event_state_frame["goal_cross_down"]),
            "goal_crossed_share": self._mean_or_zero(event_state_frame["goal_crossed"]),
            "goal_ratio_mean": self._mean_or_zero(event_state_frame["goal_ratio"]),
            "goal_gap_log_mean": self._mean_or_zero(event_state_frame["goal_gap_log"]),
        }

    def _summarize_event_state_source_topology(self, source_frame: pd.DataFrame) -> Dict[str, object]:
        if source_frame.empty:
            return {
                "source_presence_share": 0.0,
                "edgar_active_share": 0.0,
                "text_active_share": 0.0,
                "mixed_source_share": 0.0,
                "edgar_only_share": 0.0,
                "text_only_share": 0.0,
                "source_surface": "homogeneous",
                "source_profile_entropy": 0.0,
                "edgar_recency_mean_active": 0.0,
                "text_recency_mean_active": 0.0,
                "text_novelty_mean": 0.0,
            }
        edgar_active = source_frame.get(
            "edgar_active",
            pd.Series(0.0, index=source_frame.index, dtype=np.float32),
        ).to_numpy(dtype=np.float32, copy=False) >= 0.5
        text_active = source_frame.get(
            "text_active",
            pd.Series(0.0, index=source_frame.index, dtype=np.float32),
        ).to_numpy(dtype=np.float32, copy=False) >= 0.5
        counts = {
            "none": int(np.sum(np.logical_and(~edgar_active, ~text_active))),
            "edgar_only": int(np.sum(np.logical_and(edgar_active, ~text_active))),
            "text_only": int(np.sum(np.logical_and(~edgar_active, text_active))),
            "mixed": int(np.sum(np.logical_and(edgar_active, text_active))),
        }
        active_counts = {name: count for name, count in counts.items() if count > 0}
        edgar_recency = pd.to_numeric(
            source_frame.get("edgar_recency_days", pd.Series(0.0, index=source_frame.index)),
            errors="coerce",
        )
        text_recency = pd.to_numeric(
            source_frame.get("text_recency_days", pd.Series(0.0, index=source_frame.index)),
            errors="coerce",
        )
        text_novelty = pd.to_numeric(
            source_frame.get("text_novelty", pd.Series(0.0, index=source_frame.index)),
            errors="coerce",
        )
        active_profile_count = sum(1 for count in active_counts.values() if count > 0)
        return {
            "source_presence_share": self._mean_or_zero(edgar_active | text_active),
            "edgar_active_share": self._mean_or_zero(edgar_active),
            "text_active_share": self._mean_or_zero(text_active),
            "mixed_source_share": self._mean_or_zero(edgar_active & text_active),
            "edgar_only_share": self._mean_or_zero(edgar_active & ~text_active),
            "text_only_share": self._mean_or_zero(~edgar_active & text_active),
            "source_surface": "heterogeneous" if active_profile_count >= 2 else "homogeneous",
            "source_profile_entropy": self._profile_entropy(active_counts),
            "edgar_recency_mean_active": self._mean_or_zero(edgar_recency.loc[edgar_active]),
            "text_recency_mean_active": self._mean_or_zero(text_recency.loc[text_active]),
            "text_novelty_mean": self._mean_or_zero(text_novelty),
        }

    def _summarize_event_state_source_flow(self, event_state_frame: pd.DataFrame) -> Dict[str, object]:
        if event_state_frame.empty:
            return {
                "edgar_arrival_share": 0.0,
                "edgar_decay_share": 0.0,
                "text_arrival_share": 0.0,
                "text_decay_share": 0.0,
                "source_boundary_share": 0.0,
                "source_freshness_mean": 0.0,
                "source_active_count_mean": 0.0,
            }
        return {
            "edgar_arrival_share": self._mean_or_zero(event_state_frame["edgar_arrival"]),
            "edgar_decay_share": self._mean_or_zero(event_state_frame["edgar_decay"]),
            "text_arrival_share": self._mean_or_zero(event_state_frame["text_arrival"]),
            "text_decay_share": self._mean_or_zero(event_state_frame["text_decay"]),
            "source_boundary_share": self._mean_or_zero(event_state_frame["source_boundary_flag"]),
            "source_freshness_mean": self._mean_or_zero(event_state_frame["source_freshness"]),
            "source_active_count_mean": self._mean_or_zero(event_state_frame["source_active_count"]),
        }

    def _summarize_shared_state_geometry(self, shared_state: np.ndarray) -> Dict[str, object]:
        state = np.asarray(shared_state, dtype=np.float64)
        layout = self._augmented_backbone_layout()
        base = {
            "shared_state_dim": int(state.shape[1]) if state.ndim == 2 else int(layout.get("shared_state_dim", 0)),
            "compact_state_dim": int(layout.get("compact_state_dim", 0)),
            "summary_state_dim": int(layout.get("summary_state_dim", 0)),
            "temporal_state_dim": int(layout.get("temporal_state_dim", 0)),
            "spectral_state_dim": int(layout.get("spectral_state_dim", 0)),
            "multiscale_temporal_state_enabled": bool(layout.get("uses_multiscale_temporal_state", False)),
            "temporal_state_features_enabled": bool(layout.get("uses_temporal_state_features", False)),
            "spectral_state_features_enabled": bool(layout.get("uses_spectral_state_features", False)),
            "process_state_feedback_dim": int(layout.get("process_state_feedback_dim", 0)),
            "process_state_feedback_enabled": bool(layout.get("uses_process_state_feedback", False)),
            "effective_process_state_feedback": bool(layout.get("effective_process_state_feedback", False)),
            "process_state_feedback_activation_reason": str(
                layout.get("process_state_feedback_activation_reason", "process_state_feedback_not_requested")
            ),
            "process_state_feedback_gate": float(layout.get("process_state_feedback_gate", 0.0)),
            "process_state_feedback_source_scale": float(layout.get("process_state_feedback_source_scale", 0.0)),
            "process_state_feedback_norm_cap_mean": float(layout.get("process_state_feedback_norm_cap_mean", 1.0)),
            "process_state_feedback_predict_cap_scale": float(
                layout.get("process_state_feedback_predict_cap_scale", 1.0)
            ),
            "process_state_feedback_reference_abs_mean": float(
                layout.get("process_state_feedback_reference_abs_mean", 0.0)
            ),
        }
        if state.ndim != 2 or state.size == 0:
            return {
                **base,
                "shared_state_abs_mean": 0.0,
                "shared_state_l2_mean": 0.0,
                "shared_state_l2_p90": 0.0,
                "shared_state_row_std_mean": 0.0,
                "shared_state_nonzero_share": 0.0,
                "temporal_velocity_abs_mean": 0.0,
                "temporal_acceleration_abs_mean": 0.0,
                "temporal_shock_pressure_mean": 0.0,
                "spectral_low_band_abs_mean": 0.0,
                "spectral_mid_band_abs_mean": 0.0,
                "spectral_high_band_abs_mean": 0.0,
                **{
                    f"process_feedback_{state_name}_abs_mean": 0.0
                    for state_name in PROCESS_STATE_FAMILY_ORDER
                },
            }
        row_l2 = np.sqrt(np.sum(np.square(state), axis=1))
        row_std = np.std(state, axis=1)
        compact_dim = int(layout.get("compact_state_dim", 0))
        summary_dim = int(layout.get("summary_state_dim", 0))
        temporal_dim = int(layout.get("temporal_state_dim", 0))
        spectral_dim = int(layout.get("spectral_state_dim", 0))
        process_feedback_dim = int(layout.get("process_state_feedback_dim", 0))
        temporal_offset = compact_dim + summary_dim
        spectral_offset = temporal_offset + temporal_dim
        process_feedback_offset = spectral_offset + spectral_dim
        temporal_feature_names = tuple(layout.get("temporal_feature_names", ()))
        spectral_feature_names = tuple(layout.get("spectral_feature_names", ()))
        process_feedback_feature_names = tuple(layout.get("process_state_feedback_feature_names", ()))
        temporal_lookup = {name: idx for idx, name in enumerate(temporal_feature_names)}
        spectral_lookup = {name: idx for idx, name in enumerate(spectral_feature_names)}
        process_feedback_lookup = {name: idx for idx, name in enumerate(process_feedback_feature_names)}
        temporal_block = state[:, temporal_offset:spectral_offset] if temporal_dim > 0 else np.zeros((len(state), 0), dtype=np.float64)
        spectral_block = state[:, spectral_offset:spectral_offset + spectral_dim] if spectral_dim > 0 else np.zeros((len(state), 0), dtype=np.float64)
        process_feedback_block = (
            state[:, -process_feedback_dim:]
            if process_feedback_dim > 0 and state.shape[1] >= process_feedback_dim
            else np.zeros((len(state), 0), dtype=np.float64)
        )
        process_feedback_feature_abs_mean = self._process_state_feedback_current_feature_abs_mean

        def _block_stat(block: np.ndarray, lookup: Dict[str, int], name: str) -> float:
            idx = lookup.get(name)
            if idx is None or block.shape[1] <= idx:
                return 0.0
            return float(np.mean(np.abs(block[:, idx])))

        def _feedback_stat(state_name: str) -> float:
            if (
                process_feedback_feature_abs_mean is not None
                and process_feedback_feature_abs_mean.shape[0] == process_feedback_dim
            ):
                idx = process_feedback_lookup.get(f"process_feedback_{state_name}")
                if idx is None or idx >= process_feedback_feature_abs_mean.shape[0]:
                    return 0.0
                return float(process_feedback_feature_abs_mean[idx])
            return _block_stat(
                process_feedback_block,
                process_feedback_lookup,
                f"process_feedback_{state_name}",
            )

        return {
            **base,
            "shared_state_abs_mean": float(np.mean(np.abs(state))),
            "shared_state_l2_mean": float(np.mean(row_l2)),
            "shared_state_l2_p90": float(np.quantile(row_l2, 0.90)),
            "shared_state_row_std_mean": float(np.mean(row_std)),
            "shared_state_nonzero_share": float(np.mean(np.abs(state) > 1e-6)),
            "temporal_velocity_abs_mean": _block_stat(temporal_block, temporal_lookup, "state_velocity"),
            "temporal_acceleration_abs_mean": _block_stat(temporal_block, temporal_lookup, "state_acceleration"),
            "temporal_shock_pressure_mean": _block_stat(temporal_block, temporal_lookup, "state_shock_pressure"),
            "spectral_low_band_abs_mean": _block_stat(spectral_block, spectral_lookup, "low_band_level"),
            "spectral_mid_band_abs_mean": _block_stat(spectral_block, spectral_lookup, "mid_band_level"),
            "spectral_high_band_abs_mean": _block_stat(spectral_block, spectral_lookup, "high_band_level"),
            **{
                f"process_feedback_{state_name}_abs_mean": _feedback_stat(state_name)
                for state_name in PROCESS_STATE_FAMILY_ORDER
            },
        }

    def _resolve_process_state_feedback_gate(self, source_presence_share: float) -> tuple[float, str, float]:
        if not self.enable_process_state_feedback:
            return 0.0, "process_state_feedback_not_requested", 0.0
        if self._horizon < self.process_state_feedback_min_horizon:
            return 0.0, "process_state_feedback_horizon_blocked", 0.0
        if sum(abs(weight) for weight in self.process_state_feedback_state_weights) <= 0.0:
            return 0.0, "process_state_feedback_zero_weights", 0.0
        source_scale = float(np.clip(1.0 - self.process_state_feedback_source_decay * source_presence_share, 0.0, 1.0))
        gate = float(np.clip(self.process_state_feedback_strength * source_scale, 0.0, 1.0))
        if gate <= 0.0:
            return 0.0, "process_state_feedback_source_blocked", source_scale
        if source_presence_share > 0.0:
            return gate, "process_state_feedback_source_attenuated", source_scale
        return gate, "process_state_feedback_gate_open", source_scale

    def _store_process_state_feedback_reference(self, feedback_block: np.ndarray) -> None:
        if feedback_block.ndim != 2 or feedback_block.size <= 0:
            self._process_state_feedback_reference_abs_mean = 0.0
            self._process_state_feedback_reference_row_l2_mean = 0.0
            self._process_state_feedback_reference_feature_abs_mean = None
            return
        feature_abs_mean = np.mean(np.abs(feedback_block), axis=0).astype(np.float32, copy=False)
        row_l2 = np.sqrt(np.sum(np.square(feedback_block), axis=1))
        self._process_state_feedback_reference_feature_abs_mean = feature_abs_mean
        self._process_state_feedback_reference_abs_mean = float(np.mean(feature_abs_mean))
        self._process_state_feedback_reference_row_l2_mean = float(np.mean(row_l2)) if row_l2.size else 0.0

    def _resolve_process_state_feedback_projection(self, shared_dim: int) -> np.ndarray:
        if self._process_state_feedback_dim <= 0 or shared_dim <= 0:
            return np.zeros((self._process_state_feedback_dim, max(shared_dim, 0)), dtype=np.float32)
        if (
            self._process_state_feedback_projection is not None
            and self._process_state_feedback_projection_shared_dim == shared_dim
            and self._process_state_feedback_projection.shape == (self._process_state_feedback_dim, shared_dim)
        ):
            return self._process_state_feedback_projection

        if shared_dim < self._process_state_feedback_dim:
            projection = np.zeros((self._process_state_feedback_dim, shared_dim), dtype=np.float32)
            projection[:shared_dim, :shared_dim] = np.eye(shared_dim, dtype=np.float32)
        else:
            seed = 104729 + 97 * shared_dim + 31 * self._process_state_feedback_dim
            rng = np.random.default_rng(seed)
            basis = rng.standard_normal((shared_dim, self._process_state_feedback_dim))
            q, _ = np.linalg.qr(basis, mode="reduced")
            projection = q[:, :self._process_state_feedback_dim].T.astype(np.float32, copy=False)

        self._process_state_feedback_projection = projection
        self._process_state_feedback_projection_shared_dim = shared_dim
        return projection

    def _inject_process_state_feedback(
        self,
        shared_state: np.ndarray,
        *,
        feedback_block: np.ndarray,
    ) -> np.ndarray:
        if (
            feedback_block.ndim != 2
            or feedback_block.size <= 0
            or shared_state.ndim != 2
            or shared_state.shape[0] != feedback_block.shape[0]
            or shared_state.shape[1] <= 0
        ):
            return np.asarray(shared_state, dtype=np.float32)
        projection = self._resolve_process_state_feedback_projection(shared_state.shape[1])
        projected_feedback = feedback_block @ projection
        return (np.asarray(shared_state, dtype=np.float32) + projected_feedback).astype(
            np.float32,
            copy=False,
        )

    def _clip_process_state_feedback_by_shared_state(
        self,
        feedback_block: np.ndarray,
        *,
        base_shared_state: np.ndarray,
    ) -> np.ndarray:
        self._process_state_feedback_norm_cap_mean = 1.0
        if (
            feedback_block.ndim != 2
            or feedback_block.size <= 0
            or self.process_state_feedback_max_norm_share <= 0.0
            or base_shared_state.ndim != 2
            or base_shared_state.shape[0] != feedback_block.shape[0]
        ):
            return feedback_block
        base_row_l2 = np.sqrt(np.sum(np.square(base_shared_state.astype(np.float64, copy=False)), axis=1))
        feedback_row_l2 = np.sqrt(np.sum(np.square(feedback_block.astype(np.float64, copy=False)), axis=1))
        scales = np.ones(feedback_block.shape[0], dtype=np.float32)
        active = feedback_row_l2 > 1e-9
        if not np.any(active):
            return feedback_block
        allowed_row_l2 = self.process_state_feedback_max_norm_share * np.maximum(base_row_l2, 1e-6)
        scales[active] = np.minimum(1.0, allowed_row_l2[active] / feedback_row_l2[active]).astype(np.float32)
        self._process_state_feedback_norm_cap_mean = float(np.mean(scales)) if len(scales) else 1.0
        return (feedback_block * scales.reshape(-1, 1)).astype(np.float32, copy=False)

    def _clip_process_state_feedback_by_reference(
        self,
        feedback_block: np.ndarray,
        *,
        phase: str,
    ) -> np.ndarray:
        self._process_state_feedback_predict_cap_scale = 1.0
        if (
            phase not in {"predict", "test"}
            or feedback_block.ndim != 2
            or feedback_block.size <= 0
            or self.process_state_feedback_predict_scale_cap <= 0.0
        ):
            return feedback_block

        scale = 1.0
        current_feature_abs_mean = np.mean(np.abs(feedback_block), axis=0)
        active_features = current_feature_abs_mean > 1e-9
        if (
            self._process_state_feedback_reference_feature_abs_mean is not None
            and self._process_state_feedback_reference_feature_abs_mean.shape == current_feature_abs_mean.shape
            and np.any(active_features)
        ):
            allowed_feature_abs_mean = (
                self.process_state_feedback_predict_scale_cap
                * np.maximum(self._process_state_feedback_reference_feature_abs_mean.astype(np.float64, copy=False), 1e-6)
            )
            feature_scale = float(
                np.min(allowed_feature_abs_mean[active_features] / current_feature_abs_mean[active_features])
            )
            scale = min(scale, feature_scale)

        current_abs_mean = float(np.mean(current_feature_abs_mean)) if current_feature_abs_mean.size else 0.0
        if self._process_state_feedback_reference_abs_mean > 1e-9 and current_abs_mean > 1e-9:
            scale = min(
                scale,
                float(
                    self.process_state_feedback_predict_scale_cap
                    * self._process_state_feedback_reference_abs_mean
                    / current_abs_mean
                ),
            )

        current_row_l2 = np.sqrt(np.sum(np.square(feedback_block.astype(np.float64, copy=False)), axis=1))
        current_row_l2_mean = float(np.mean(current_row_l2)) if current_row_l2.size else 0.0
        if self._process_state_feedback_reference_row_l2_mean > 1e-9 and current_row_l2_mean > 1e-9:
            scale = min(
                scale,
                float(
                    self.process_state_feedback_predict_scale_cap
                    * self._process_state_feedback_reference_row_l2_mean
                    / current_row_l2_mean
                ),
            )

        scale = float(np.clip(scale, 0.0, 1.0))
        self._process_state_feedback_predict_cap_scale = scale
        if scale >= 0.999999:
            return feedback_block
        return (feedback_block * scale).astype(np.float32, copy=False)

    def _build_process_state_feedback_block(
        self,
        process_bundle: Dict[str, object],
        *,
        source_presence_share: float,
        base_shared_state: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        score_lookup = dict(process_bundle.get("scores", {}))
        support_lookup = dict(process_bundle.get("support_shares", {}))
        n_rows = 0
        if score_lookup:
            n_rows = len(next(iter(score_lookup.values())))
        gate, reason, source_scale = self._resolve_process_state_feedback_gate(source_presence_share)
        self._process_state_feedback_activation_reason = reason
        self._process_state_feedback_gate = float(gate)
        self._process_state_feedback_source_scale = float(source_scale)
        if not self.enable_process_state_feedback:
            self._process_state_feedback_feature_cols = []
            self._process_state_feedback_dim = 0
            self._effective_process_state_feedback = False
            return np.zeros((n_rows, 0), dtype=np.float32)

        self._process_state_feedback_feature_cols = [
            f"process_feedback_{state_name}" for state_name in PROCESS_STATE_FAMILY_ORDER
        ]
        self._process_state_feedback_dim = len(self._process_state_feedback_feature_cols)
        if n_rows <= 0:
            self._effective_process_state_feedback = False
            return np.zeros((0, self._process_state_feedback_dim), dtype=np.float32)

        weights = np.asarray(self.process_state_feedback_state_weights, dtype=np.float32).reshape(1, -1)
        support = np.asarray(
            [float(support_lookup.get(state_name, 1.0)) for state_name in PROCESS_STATE_FAMILY_ORDER],
            dtype=np.float32,
        ).reshape(1, -1)
        score_matrix = np.column_stack(
            [
                np.asarray(score_lookup.get(state_name, np.zeros(n_rows, dtype=np.float32)), dtype=np.float32)
                for state_name in PROCESS_STATE_FAMILY_ORDER
            ]
        )
        centered = np.tanh(2.0 * (score_matrix - 0.5)).astype(np.float32, copy=False)
        feedback_block = float(gate) * centered * weights * support
        feedback_block = self._clip_process_state_feedback_by_shared_state(
            feedback_block.astype(np.float32, copy=False),
            base_shared_state=base_shared_state,
        )
        feedback_block = self._clip_process_state_feedback_by_reference(
            feedback_block,
            phase=phase,
        )
        if phase in {"fit", "train"}:
            self._store_process_state_feedback_reference(feedback_block)
        self._effective_process_state_feedback = bool(np.any(np.abs(feedback_block) > 1e-6))
        return feedback_block.astype(np.float32, copy=False)

    def _build_process_state_feature_bundle(
        self,
        runtime_frame: pd.DataFrame,
        *,
        event_state_frame: pd.DataFrame,
        shared_state_atoms: Dict[str, object],
    ) -> Dict[str, object]:
        if runtime_frame.empty or event_state_frame.empty:
            zero_scores = {
                state_name: np.zeros(len(runtime_frame), dtype=np.float64)
                for state_name in PROCESS_STATE_FAMILY_ORDER
            }
            zero_support = {state_name: 0.0 for state_name in PROCESS_STATE_FAMILY_ORDER}
            zero_couplings = {
                "temporal_velocity": 0.0,
                "temporal_shock": 0.0,
                "spectral_low_band": 0.0,
                "spectral_high_band": 0.0,
            }
            return {
                "scores": zero_scores,
                "support_shares": zero_support,
                "couplings": zero_couplings,
                "multiscale_coupling_enabled": False,
            }

        investors_count = pd.to_numeric(
            runtime_frame.get("investors_count", pd.Series(0.0, index=runtime_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        minimum_investment = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_minimum_investment_accepted",
                "mean_minimum_investment_accepted",
                "ema_minimum_investment_accepted",
            ),
        )
        already_invested = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_total_number_already_invested",
                "mean_total_number_already_invested",
                "ema_total_number_already_invested",
            ),
        )
        non_accredited = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_number_non_accredited_investors",
                "mean_number_non_accredited_investors",
                "ema_number_non_accredited_investors",
            ),
        )
        non_national = self._first_available_numeric_from_frame(runtime_frame, ("non_national_investors",))
        total_offering = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_total_offering_amount",
                "mean_total_offering_amount",
                "ema_total_offering_amount",
                "funding_goal_usd",
            ),
        )
        total_sold = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_total_amount_sold",
                "mean_total_amount_sold",
                "ema_total_amount_sold",
                "funding_raised_usd",
            ),
        )
        total_remaining = self._first_available_numeric_from_frame(
            runtime_frame,
            (
                "last_total_remaining",
                "mean_total_remaining",
                "ema_total_remaining",
            ),
        )

        denom = np.maximum.reduce(
            [
                investors_count.to_numpy(dtype=np.float64, copy=False),
                already_invested.fillna(0.0).to_numpy(dtype=np.float64, copy=False),
                np.ones(len(runtime_frame), dtype=np.float64),
            ]
        )
        non_accredited_share = np.clip(
            non_accredited.fillna(0.0).to_numpy(dtype=np.float64, copy=False) / denom,
            0.0,
            1.0,
        )
        non_national_share = np.clip(
            non_national.fillna(0.0).to_numpy(dtype=np.float64, copy=False) / denom,
            0.0,
            1.0,
        )
        offering_total = np.maximum(total_offering.fillna(0.0).to_numpy(dtype=np.float64, copy=False), 1.0)
        sold_total = np.clip(total_sold.fillna(0.0).to_numpy(dtype=np.float64, copy=False), 0.0, None)
        remaining_total = np.clip(total_remaining.fillna(0.0).to_numpy(dtype=np.float64, copy=False), 0.0, None)

        min_investment_score = self._log_tanh_score(minimum_investment, scale=6.0)
        already_invested_score = self._log_tanh_score(already_invested, scale=3.5)
        sold_progress = np.clip(sold_total / offering_total, 0.0, 1.0)
        remaining_ratio = np.clip(remaining_total / offering_total, 0.0, 1.0)
        funding_jump_score = np.tanh(
            pd.to_numeric(event_state_frame["funding_jump_log"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            / 4.0
        )
        goal_ratio = np.clip(
            pd.to_numeric(event_state_frame["goal_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
            0.0,
            1.0,
        )
        source_density = np.clip(
            pd.to_numeric(event_state_frame["source_active_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            / 2.0,
            0.0,
            1.0,
        )
        source_freshness = np.clip(
            pd.to_numeric(event_state_frame["source_freshness"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
            0.0,
            1.0,
        )

        temporal_velocity = float(np.tanh(4.0 * float(shared_state_atoms.get("temporal_velocity_abs_mean", 0.0))))
        temporal_shock = float(np.tanh(4.0 * float(shared_state_atoms.get("temporal_shock_pressure_mean", 0.0))))
        spectral_low = float(np.tanh(4.0 * float(shared_state_atoms.get("spectral_low_band_abs_mean", 0.0))))
        spectral_high = float(np.tanh(4.0 * float(shared_state_atoms.get("spectral_high_band_abs_mean", 0.0))))

        feedback_boost = {
            state_name: float(
                np.tanh(6.0 * float(shared_state_atoms.get(f"process_feedback_{state_name}_abs_mean", 0.0)))
            )
            for state_name in PROCESS_STATE_FAMILY_ORDER
        }

        attention_score = np.clip(
            0.20 * pd.to_numeric(event_state_frame["investors_active"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.15 * pd.to_numeric(event_state_frame["investor_jump_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.20 * source_density
            + 0.15 * pd.to_numeric(event_state_frame["source_boundary_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.20 * source_freshness
            + 0.10 * spectral_high
            + 0.10 * feedback_boost["attention_diffusion"],
            0.0,
            1.0,
        )
        credibility_score = np.clip(
            0.20 * pd.to_numeric(event_state_frame["funding_active"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.20 * pd.to_numeric(event_state_frame["funded_active"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.15 * pd.to_numeric(event_state_frame["goal_known"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.15 * goal_ratio
            + 0.20 * source_freshness
            + 0.10 * spectral_low
            + 0.10 * feedback_boost["credibility_confirmation"],
            0.0,
            1.0,
        )
        screening_score = np.clip(
            0.30 * min_investment_score
            + 0.25 * already_invested_score
            + 0.25 * (1.0 - non_accredited_share)
            + 0.20 * non_national_share
            + 0.05 * feedback_boost["screening_selectivity"],
            0.0,
            1.0,
        )
        book_depth_score = np.clip(
            0.35 * sold_progress
            + 0.25 * (1.0 - remaining_ratio)
            + 0.20 * goal_ratio
            + 0.20 * funding_jump_score
            + 0.05 * feedback_boost["book_depth_absorption"],
            0.0,
            1.0,
        )
        closure_score = np.clip(
            0.25 * pd.to_numeric(event_state_frame["funded_active"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.20 * pd.to_numeric(event_state_frame["goal_crossed"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.15 * goal_ratio
            + 0.15 * pd.to_numeric(event_state_frame["phase_transition_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            + 0.15 * temporal_shock
            + 0.10 * temporal_velocity
            + 0.15 * feedback_boost["closure_conversion"],
            0.0,
            1.0,
        )

        screening_support = max(
            self._series_coverage_share(minimum_investment),
            self._series_coverage_share(already_invested),
            self._series_coverage_share(non_accredited),
            self._series_coverage_share(non_national),
        )
        book_depth_support = max(
            self._series_coverage_share(total_offering),
            self._series_coverage_share(total_sold),
            self._series_coverage_share(total_remaining),
        )
        multiscale_coupling_enabled = bool(
            shared_state_atoms.get("temporal_state_features_enabled", False)
            or shared_state_atoms.get("spectral_state_features_enabled", False)
            or shared_state_atoms.get("process_state_feedback_enabled", False)
        )
        return {
            "scores": {
                "attention_diffusion": attention_score,
                "credibility_confirmation": credibility_score,
                "screening_selectivity": screening_score,
                "book_depth_absorption": book_depth_score,
                "closure_conversion": closure_score,
            },
            "support_shares": {
                "attention_diffusion": 1.0,
                "credibility_confirmation": 1.0,
                "screening_selectivity": float(screening_support),
                "book_depth_absorption": float(book_depth_support),
                "closure_conversion": 1.0,
            },
            "couplings": {
                "temporal_velocity": temporal_velocity,
                "temporal_shock": temporal_shock,
                "spectral_low_band": spectral_low,
                "spectral_high_band": spectral_high,
            },
            "multiscale_coupling_enabled": multiscale_coupling_enabled,
        }

    def _summarize_process_state_bundle(self, process_bundle: Dict[str, object]) -> Dict[str, object]:
        score_lookup = dict(process_bundle.get("scores", {}))
        support_lookup = dict(process_bundle.get("support_shares", {}))
        couplings = dict(process_bundle.get("couplings", {}))
        mean_scores = {
            state_name: float(np.mean(np.asarray(score_lookup.get(state_name, np.zeros(1, dtype=np.float64)), dtype=np.float64)))
            if len(np.asarray(score_lookup.get(state_name, np.zeros(0, dtype=np.float64)))) > 0
            else 0.0
            for state_name in PROCESS_STATE_FAMILY_ORDER
        }
        return {
            "schema_version": "process_state_v1",
            "state_order": list(PROCESS_STATE_FAMILY_ORDER),
            "multiscale_coupling_enabled": bool(process_bundle.get("multiscale_coupling_enabled", False)),
            "attention_diffusion_score": mean_scores["attention_diffusion"],
            "attention_diffusion_support_share": float(support_lookup.get("attention_diffusion", 0.0)),
            "credibility_confirmation_score": mean_scores["credibility_confirmation"],
            "credibility_confirmation_support_share": float(support_lookup.get("credibility_confirmation", 0.0)),
            "screening_selectivity_score": mean_scores["screening_selectivity"],
            "screening_selectivity_support_share": float(support_lookup.get("screening_selectivity", 0.0)),
            "book_depth_absorption_score": mean_scores["book_depth_absorption"],
            "book_depth_absorption_support_share": float(support_lookup.get("book_depth_absorption", 0.0)),
            "closure_conversion_score": mean_scores["closure_conversion"],
            "closure_conversion_support_share": float(support_lookup.get("closure_conversion", 0.0)),
            "temporal_velocity_coupling": float(couplings.get("temporal_velocity", 0.0)),
            "temporal_shock_coupling": float(couplings.get("temporal_shock", 0.0)),
            "spectral_low_band_coupling": float(couplings.get("spectral_low_band", 0.0)),
            "spectral_high_band_coupling": float(couplings.get("spectral_high_band", 0.0)),
        }

    def _summarize_process_state_families(
        self,
        runtime_frame: pd.DataFrame,
        *,
        event_state_frame: pd.DataFrame,
        shared_state_atoms: Dict[str, object],
    ) -> Dict[str, object]:
        process_bundle = self._build_process_state_feature_bundle(
            runtime_frame,
            event_state_frame=event_state_frame,
            shared_state_atoms=shared_state_atoms,
        )
        return self._summarize_process_state_bundle(process_bundle)

    def _mean_or_zero(self, values: pd.Series | np.ndarray) -> float:
        series = pd.Series(values).dropna()
        if len(series) <= 0:
            return 0.0
        return float(series.astype(np.float64).mean())

    def _first_available_numeric_from_frame(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> pd.Series:
        for column in columns:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().any():
                return series.reindex(frame.index)
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)

    def _series_coverage_share(self, series: pd.Series) -> float:
        if len(series) <= 0:
            return 0.0
        return float(pd.to_numeric(series, errors="coerce").notna().mean())

    def _log_tanh_score(self, series: pd.Series, *, scale: float) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64, copy=False)
        return np.tanh(np.log1p(values) / max(scale, 1e-6))

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
        (
            self._effective_investors_event_state_features,
            self._investors_event_state_activation_reason,
        ) = self._resolve_investors_event_state_activation()

    def _refresh_investors_mark_regime(self, investors_mark_frame: pd.DataFrame) -> None:
        if not self.enable_investors_mark_features:
            self._effective_investors_mark_features = False
            self._investors_mark_activation_reason = "mark_features_not_requested"
            self._investors_mark_mode = "inactive"
        elif investors_mark_frame.empty or len(investors_mark_frame.columns) <= 0:
            self._effective_investors_mark_features = False
            self._investors_mark_activation_reason = "mark_features_unavailable"
            self._investors_mark_mode = "unavailable"
        else:
            matrix = investors_mark_frame.to_numpy(dtype=np.float32, copy=False)
            coverage = float(np.mean(np.any(np.abs(matrix) > 1e-6, axis=1))) if len(matrix) else 0.0
            self._effective_investors_mark_features = coverage > 0.0
            raw_reference_columns = [
                column
                for column in (
                    "mark_list_present",
                    "mark_website_present",
                    "mark_hash_present",
                    "mark_repeat_list_flag",
                    "mark_list_changed_flag",
                )
                if column in investors_mark_frame.columns
            ]
            proxy_columns = [column for column in investors_mark_frame.columns if column not in raw_reference_columns]
            raw_reference_matrix = (
                investors_mark_frame[raw_reference_columns].to_numpy(dtype=np.float32, copy=False)
                if raw_reference_columns
                else np.zeros((len(investors_mark_frame), 0), dtype=np.float32)
            )
            proxy_matrix = (
                investors_mark_frame[proxy_columns].to_numpy(dtype=np.float32, copy=False)
                if proxy_columns
                else np.zeros((len(investors_mark_frame), 0), dtype=np.float32)
            )
            raw_reference_share = (
                float(np.mean(np.any(np.abs(raw_reference_matrix) > 1e-6, axis=1)))
                if raw_reference_matrix.size
                else 0.0
            )
            proxy_share = (
                float(np.mean(np.any(np.abs(proxy_matrix) > 1e-6, axis=1)))
                if proxy_matrix.size
                else 0.0
            )
            proxy_only_share = (
                float(
                    np.mean(
                        np.any(np.abs(proxy_matrix) > 1e-6, axis=1)
                        & ~np.any(np.abs(raw_reference_matrix) > 1e-6, axis=1)
                    )
                )
                if proxy_matrix.size
                else 0.0
            )
            self._investors_mark_raw_reference_share = raw_reference_share
            self._investors_mark_proxy_share = proxy_share
            self._investors_mark_proxy_only_share = proxy_only_share
            if not self._effective_investors_mark_features:
                self._investors_mark_mode = "empty"
                self._investors_mark_activation_reason = "mark_features_empty"
            elif raw_reference_share > 0.0 and proxy_share > 0.0:
                self._investors_mark_mode = "hybrid"
                self._investors_mark_activation_reason = "mark_features_hybrid"
            elif raw_reference_share > 0.0:
                self._investors_mark_mode = "raw_reference_rich"
                self._investors_mark_activation_reason = "mark_features_raw_reference_rich"
            elif proxy_share > 0.0:
                self._investors_mark_mode = "proxy_only"
                self._investors_mark_activation_reason = "mark_features_proxy_only"
            else:
                self._investors_mark_mode = "empty"
                self._investors_mark_activation_reason = "mark_features_empty"

        matrix = (
            investors_mark_frame.to_numpy(dtype=np.float32, copy=False)
            if not investors_mark_frame.empty
            else np.zeros((0, 0), dtype=np.float32)
        )
        self._investors_mark_nonzero_share = float(np.mean(np.abs(matrix) > 1e-6)) if matrix.size else 0.0
        self._investors_mark_coverage_share = float(np.mean(np.any(np.abs(matrix) > 1e-6, axis=1))) if len(matrix) else 0.0
        if investors_mark_frame.empty:
            self._investors_mark_raw_reference_share = 0.0
            self._investors_mark_proxy_share = 0.0
            self._investors_mark_proxy_only_share = 0.0
        self._investors_mark_summary = {
            column: float(pd.to_numeric(investors_mark_frame[column], errors="coerce").fillna(0.0).mean())
            for column in investors_mark_frame.columns
        }

    def _resolve_investors_event_state_activation(self) -> tuple[bool, str]:
        if not self.enable_investors_event_state_features:
            return False, "event_state_not_requested"
        if len(self._investors_event_state_feature_cols) <= 0:
            return False, "event_state_features_unavailable"
        if not self.enable_investors_selective_event_state_activation:
            return True, "event_state_requested"
        if self._horizon == 1 and not self.investors_event_state_allow_h1:
            return False, "event_state_h1_blocked"
        source_presence_share = float(
            dict(self._event_state_card.get("source_atoms", {})).get("source_presence_share", 0.0)
        )
        if source_presence_share > float(self.investors_event_state_max_source_presence_share):
            return False, "event_state_source_rich_surface_blocked"
        return True, "event_state_selective_gate_open"

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