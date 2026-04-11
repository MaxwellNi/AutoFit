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
from .lanes.investors_lane import InvestorsLaneRuntime, InvestorsLaneSpec
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
        self.enable_count_jump = bool(prototype_kwargs.get("enable_count_jump", True))
        self.count_jump_strength = float(prototype_kwargs.get("count_jump_strength", 0.30))
        self.enable_count_sparsity_gate = bool(prototype_kwargs.get("enable_count_sparsity_gate", True))
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

        if self._binary_target:
            self.binary_lane_runtime.fit(lane_states["binary"], y_arr, aux_features=history_matrix)
            transition = (y_arr > 0.5) & (history_frame["target_lag1"].fillna(0.0).to_numpy(dtype=np.float64) <= 0.5)
            self._binary_transition_rate = float(np.mean(transition)) if transition.size else 0.0
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
                "count_source_specialists": self.enable_count_source_specialists,
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

        if self._active_lane_name == "binary":
            preds = self.binary_lane_runtime.predict(lane_states["binary"], aux_features=history_matrix)
            return np.clip(preds, 0.0, 1.0).astype(np.float64, copy=False)
        if self._active_lane_name == "funding":
            preds = self.funding_lane_runtime.predict(lane_states["funding"], aux_features=history_matrix, anchor=anchor)
            return np.clip(preds, 0.0, None).astype(np.float64, copy=False)
        preds = self.investors_lane_runtime.predict(lane_states["investors"], aux_features=history_matrix, anchor=anchor)
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