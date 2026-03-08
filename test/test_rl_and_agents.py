"""Tests for RL policy (contextual bandit) and multi-agent ensemble coordination."""

import numpy as np
import pandas as pd
import pytest

# ── RL Policy Tests ──


def test_state_encoder_produces_correct_dim():
    from narrative.block3.models.rl_policy import _encode_state, STATE_DIM
    state = _encode_state("count", "short", "core_only", "low")
    assert state.shape == (STATE_DIM,)
    assert np.isfinite(state).all()
    # Should have exactly 1 hot for each categorical
    assert state[:4].sum() == 1.0  # lane
    assert state[4:7].sum() == 1.0  # horizon_band
    assert state[7:11].sum() == 1.0  # ablation
    assert state[11:14].sum() == 1.0  # missingness
    assert state[-1] == 1.0  # bias


def test_state_encoder_with_meta_features():
    from narrative.block3.models.rl_policy import _encode_state, STATE_DIM
    meta = {"kurtosis": 3.5, "cv": 1.2, "zero_frac": 0.1}
    state = _encode_state("general", "mid", "full", "medium", meta_features=meta)
    assert state.shape == (STATE_DIM,)
    assert np.isfinite(state).all()


def test_state_encoder_unknown_category():
    from narrative.block3.models.rl_policy import _encode_state, STATE_DIM
    state = _encode_state("unknown_lane", "unknown_hb", "unknown_abl", "unknown_miss")
    assert state.shape == (STATE_DIM,)
    # All one-hots should be zero for unknown values
    assert state[:4].sum() == 0.0


def test_bayesian_linear_arm_update_and_predict():
    from narrative.block3.models.rl_policy import BayesianLinearArm
    arm = BayesianLinearArm(dim=5, reg_lambda=1.0)
    rng = np.random.RandomState(42)
    x = np.array([1.0, 0.5, -0.3, 0.8, 1.0])

    assert arm.n_observations == 0
    arm.update(x, reward=0.7)
    assert arm.n_observations == 1

    ucb = arm.predict_ucb(x, alpha=1.0)
    assert np.isfinite(ucb)

    ts = arm.predict_thompson(x, rng)
    assert np.isfinite(ts)


def test_factored_bandit_select_action():
    from narrative.block3.models.rl_policy import (
        FactoredContextualBandit, _encode_state, HP_SPECS
    )
    bandit = FactoredContextualBandit(seed=42)
    state = _encode_state("count", "short", "core_only", "low")

    action = bandit.select_action(state)
    assert isinstance(action, dict)
    for spec in HP_SPECS:
        assert spec.name in action
        assert action[spec.name] in spec.values


def test_factored_bandit_update_and_learn():
    from narrative.block3.models.rl_policy import (
        FactoredContextualBandit, _encode_state
    )
    bandit = FactoredContextualBandit(seed=42, min_observations=2)
    state = _encode_state("general", "mid", "full", "medium")

    # Run several update cycles
    for i in range(10):
        action = bandit.select_action(state)
        reward = -0.5 + 0.1 * (i % 3)  # varying reward
        bandit.update(state, action, reward)

    diag = bandit.get_diagnostics()
    assert diag["n_history"] == 10
    assert diag["strategy"] == "thompson"


def test_factored_bandit_serialize_deserialize():
    from narrative.block3.models.rl_policy import (
        FactoredContextualBandit, _encode_state
    )
    bandit = FactoredContextualBandit(seed=42)
    state = _encode_state("count", "short", "core_only", "low")

    for _ in range(5):
        action = bandit.select_action(state)
        bandit.update(state, action, reward=-0.3)

    serialized = bandit.serialize()
    restored = FactoredContextualBandit.deserialize(serialized)

    diag_orig = bandit.get_diagnostics()
    diag_rest = restored.get_diagnostics()
    assert diag_orig["n_history"] == diag_rest["top_k_total_obs"]  # not same key, just check non-empty
    # Check arm observations are preserved
    for spec_name in ["top_k", "moe_max_experts"]:
        assert diag_rest[f"{spec_name}_total_obs"] > 0


def test_adaptive_rl_policy_query_defaults():
    from narrative.block3.models.rl_policy import AdaptiveRLPolicy, PolicyDecision
    policy = AdaptiveRLPolicy(seed=42)
    decision = policy.query("general", "mid", "core_only", "medium")
    assert isinstance(decision, PolicyDecision)
    assert decision.top_k > 0
    assert decision.moe_max_experts > 0
    assert 0.0 <= decision.moe_temperature <= 1.0


def test_adaptive_rl_policy_heuristics():
    from narrative.block3.models.rl_policy import AdaptiveRLPolicy
    policy = AdaptiveRLPolicy(seed=42)

    # Binary lane heuristic
    decision = policy.query("binary", "mid", "core_only", "medium")
    assert decision.source == "heuristic_binary"
    assert decision.top_k == 6

    # Count + short heuristic
    decision = policy.query("count", "short", "core_only", "low")
    assert decision.source == "heuristic_count_short"
    assert decision.top_k == 8


def test_adaptive_rl_policy_observe():
    from narrative.block3.models.rl_policy import AdaptiveRLPolicy
    policy = AdaptiveRLPolicy(seed=42, enable_online_learning=True)

    # Query and observe
    decision = policy.query("general", "long", "full", "high")
    policy.observe(
        lane="general", horizon_band="long", ablation="full",
        missingness="high", action=decision.action_dict,
        reward=-0.25, meta_features={"kurtosis": 2.0}
    )

    diag = policy._bandit.get_diagnostics()
    assert diag["n_history"] >= 1


def test_compute_reward_range():
    from narrative.block3.models.rl_policy import compute_reward
    # Perfect prediction
    r = compute_reward(0.0, 10.0, 5.0)
    assert r > 0.5

    # Terrible prediction
    r = compute_reward(20.0, 10.0, 5.0)
    assert r < 0.0

    # Edge: zero naive
    r = compute_reward(1.0, 0.0, 0.5)
    assert r == 0.0

    # Edge: inf ensemble
    r = compute_reward(float("inf"), 10.0, 5.0)
    assert r == -1.0


def test_build_replay_record():
    from narrative.block3.models.rl_policy import build_replay_record
    routing_info = {
        "lane_selected": "count",
        "horizon_band": "short",
        "ablation": "core_only",
        "missingness_bucket": "low",
        "top_k": 12,
    }
    record = build_replay_record(routing_info, 5.0, 10.0, 6.0, 120.0)
    assert "reward" in record
    assert "lane" in record
    assert record["lane"] == "count"
    assert np.isfinite(record["reward"])


def test_warm_start_from_records():
    from narrative.block3.models.rl_policy import FactoredContextualBandit
    records = [
        {
            "lane": "count", "horizon_band": "short",
            "ablation": "core_only", "missingness_bucket": "low",
            "top_k": 12, "moe_max_experts": 5,
            "moe_temperature": 0.35, "blend_alpha": 0.25,
            "qs_threshold_mult": 0.90, "reward": -0.2,
        },
        {
            "lane": "general", "horizon_band": "mid",
            "ablation": "full", "missingness_bucket": "medium",
            "top_k": 8, "moe_max_experts": 3,
            "moe_temperature": 0.25, "blend_alpha": 0.15,
            "qs_threshold_mult": 0.85, "reward": -0.35,
        },
    ]
    bandit = FactoredContextualBandit(seed=42)
    n = bandit.warm_start(records)
    assert n == 2
    assert bandit.get_diagnostics()["n_history"] == 2


# ── Multi-Agent Tests ──


def test_blackboard_read_write():
    from narrative.block3.models.multi_agent_ensemble import Blackboard
    board = Blackboard()
    board.write("test/key", 42, "test_agent")
    assert board.read("test/key") == 42
    assert board.has("test/key")
    assert board.version("test/key") == 1

    board.write("test/key", 99, "test_agent")
    assert board.read("test/key") == 99
    assert board.version("test/key") == 2


def test_blackboard_audit_log():
    from narrative.block3.models.multi_agent_ensemble import Blackboard
    board = Blackboard()
    board.write("a", 1, "agent1")
    board.write("b", 2, "agent2")
    log = board.get_audit_log()
    assert len(log) == 2
    assert log[0]["agent"] == "agent1"
    assert log[1]["agent"] == "agent2"


def test_recon_agent_binary_lane():
    from narrative.block3.models.multi_agent_ensemble import ReconAgent, Blackboard
    board = Blackboard()
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1] * 10, dtype=float)
    X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    board.write("input/y_train", y)
    board.write("input/X_train", X)

    agent = ReconAgent()
    status = agent.execute(board)
    assert status == "ok"
    assert board.read("recon/lane") == "binary"
    assert board.read("recon/recommended_objective") == "binary"


def test_recon_agent_count_lane():
    from narrative.block3.models.multi_agent_ensemble import ReconAgent, Blackboard
    board = Blackboard()
    rng = np.random.RandomState(42)
    y = rng.poisson(lam=5.0, size=200).astype(float)
    X = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})
    board.write("input/y_train", y)
    board.write("input/X_train", X)

    agent = ReconAgent()
    status = agent.execute(board)
    assert status == "ok"
    assert board.read("recon/lane") == "count"


def test_recon_agent_risk_flags():
    from narrative.block3.models.multi_agent_ensemble import ReconAgent, Blackboard
    board = Blackboard()
    # Small sample with high kurtosis
    y = np.concatenate([np.zeros(40), np.array([100.0, 200.0, 300.0])])
    X = pd.DataFrame({"f1": np.random.randn(43)})
    board.write("input/y_train", y)
    board.write("input/X_train", X)

    agent = ReconAgent()
    status = agent.execute(board)
    risks = board.read("recon/risk_flags", [])
    assert "low_sample_size" in risks


def test_scout_agent_prioritizes():
    from narrative.block3.models.multi_agent_ensemble import ScoutAgent, Blackboard
    board = Blackboard()
    board.write("recon/lane", "general")
    board.write("recon/risk_flags", [])
    board.write("recon/n_samples", 1000)
    board.write("config/candidate_pool", ["LightGBM", "NBEATS", "PatchTST", "RandomForest"])
    board.write("config/category_fn", lambda n: {
        "LightGBM": "ml_tabular", "NBEATS": "deep_classical",
        "PatchTST": "transformer_sota", "RandomForest": "ml_tabular"
    }.get(n, "ml_tabular"))
    board.write("config/has_gpu", True)
    board.write("config/top_k", 4)
    board.write("config/time_budget", 1800)

    agent = ScoutAgent()
    status = agent.execute(board)
    assert status == "ok"

    prioritized = board.read("scout/prioritized_candidates")
    assert len(prioritized) > 0
    # Deep models should be prioritized with GPU
    assert prioritized[0] in {"NBEATS", "PatchTST"}


def test_scout_agent_deprioritizes_gpu_without_gpu():
    from narrative.block3.models.multi_agent_ensemble import ScoutAgent, Blackboard
    board = Blackboard()
    board.write("recon/lane", "general")
    board.write("recon/risk_flags", [])
    board.write("recon/n_samples", 1000)
    board.write("config/candidate_pool", ["LightGBM", "NBEATS", "PatchTST"])
    board.write("config/category_fn", lambda n: {
        "LightGBM": "ml_tabular", "NBEATS": "deep_classical",
        "PatchTST": "transformer_sota"
    }.get(n, "ml_tabular"))
    board.write("config/has_gpu", False)  # No GPU
    board.write("config/top_k", 4)
    board.write("config/time_budget", 600)

    agent = ScoutAgent()
    status = agent.execute(board)
    prioritized = board.read("scout/prioritized_candidates")
    # LightGBM should be first (GPU models deprioritized)
    assert prioritized[0] == "LightGBM"


def test_composer_agent_forward_select():
    from narrative.block3.models.multi_agent_ensemble import ComposerAgent, Blackboard
    board = Blackboard()
    rng = np.random.RandomState(42)
    y = rng.randn(100)

    board.write("input/y_train", y)
    board.write("recon/lane", "general")
    board.write("config/moe_max_experts", 3)
    board.write("config/moe_temperature", 0.40)
    board.write("config/blend_alpha", 0.25)
    board.write("eval/results", {
        "ModelA": {"mae": 0.8, "adj_mae": 0.82, "oof_preds": y + rng.randn(100) * 0.3, "category": "ml_tabular"},
        "ModelB": {"mae": 0.9, "adj_mae": 0.95, "oof_preds": y + rng.randn(100) * 0.5, "category": "deep_classical"},
        "ModelC": {"mae": 1.2, "adj_mae": 1.25, "oof_preds": y + rng.randn(100) * 0.8, "category": "transformer_sota"},
    })

    agent = ComposerAgent()
    status = agent.execute(board)
    assert status == "ok"

    selected = board.read("composer/selected_models")
    assert len(selected) >= 1
    assert "ModelA" in selected  # Best model should be included

    weights = board.read("composer/weights")
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


def test_critic_agent_accepts_good_ensemble():
    from narrative.block3.models.multi_agent_ensemble import CriticAgent, Blackboard
    board = Blackboard()
    rng = np.random.RandomState(42)
    y = rng.randn(100)

    board.write("input/y_train", y)
    board.write("recon/lane", "general")
    board.write("recon/risk_flags", [])
    board.write("composer/selected_models", ["ModelA", "ModelB"])
    board.write("composer/weights", {"ModelA": 0.6, "ModelB": 0.4})
    board.write("composer/ensemble_diversity", 0.3)
    board.write("eval/results", {
        "ModelA": {"mae": 0.5, "oof_preds": y + rng.randn(100) * 0.2},
        "ModelB": {"mae": 0.6, "oof_preds": y + rng.randn(100) * 0.3},
    })

    agent = CriticAgent()
    status = agent.execute(board)
    assert status == "ok"
    assert board.read("critic/verdict") in {"accept", "warn"}
    assert not board.read("critic/guard_triggered")


def test_orchestrator_full_pipeline():
    from narrative.block3.models.multi_agent_ensemble import (
        MultiAgentOrchestrator, OrchestratorConfig
    )
    rng = np.random.RandomState(42)
    y = rng.poisson(lam=5.0, size=200).astype(float)
    X = pd.DataFrame({"f1": rng.randn(200), "f2": rng.randn(200)})

    orch = MultiAgentOrchestrator(
        config=OrchestratorConfig(max_restart_attempts=1)
    )

    # Phase 1: Recon + Scout
    recon_result = orch.run_recon_phase(
        X_train=X, y_train=y,
        candidate_pool=["LightGBM", "NBEATS"],
        category_fn=lambda n: "ml_tabular" if n == "LightGBM" else "deep_classical",
        has_gpu=True, top_k=4,
    )
    assert "lane" in recon_result
    assert "prioritized_candidates" in recon_result

    # Phase 2: Compose + Critique (with mock evaluation results)
    compose_result = orch.run_compose_phase({
        "LightGBM": {"mae": 2.0, "adj_mae": 2.1, "oof_preds": y + rng.randn(200) * 0.5, "category": "ml_tabular"},
        "NBEATS": {"mae": 1.8, "adj_mae": 1.9, "oof_preds": y + rng.randn(200) * 0.4, "category": "deep_classical"},
    })
    assert "verdict" in compose_result
    assert "selected_models" in compose_result
    assert len(compose_result["selected_models"]) >= 1

    # Check telemetry
    telemetry = orch.get_full_telemetry()
    assert "timing" in telemetry


def test_v73_wrapper_has_rl_and_orchestrator():
    """V73 wrapper init should create RL policy and orchestrator."""
    from narrative.block3.models.autofit_wrapper import AutoFitV73Wrapper
    w = AutoFitV73Wrapper()
    assert w._rl_policy is not None
    assert w._orchestrator is not None
    assert w.config.name == "AutoFitV73"
    assert w.config.params["strategy"] == "v73_gpu_full_spectrum"
