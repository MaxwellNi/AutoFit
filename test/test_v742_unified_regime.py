import numpy as np
import pandas as pd
import pytest

from narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper, _torch_imports


@pytest.fixture(scope="module")
def torch_and_net():
    torch = pytest.importorskip("torch")
    _, _, net_cls = _torch_imports()
    return torch, net_cls


def _tokens(torch, batch_size: int, target_idx: int):
    task_idx = torch.full((batch_size,), 1, dtype=torch.long)
    target_ids = torch.full((batch_size,), target_idx, dtype=torch.long)
    horizon_value = torch.full((batch_size,), 7.0, dtype=torch.float32)
    ablation_idx = torch.zeros((batch_size,), dtype=torch.long)
    return task_idx, target_ids, horizon_value, ablation_idx


def test_v742_unified_forward_emits_joint_financing_outputs(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=5,
        seq_len=12,
        horizon=4,
        hidden_dim=16,
        enable_target_routing=False,
        enable_count_source_routing=False,
        enable_financing_consistency=True,
        financing_process_blend=0.65,
        funding_log_domain_enabled=True,
    )
    net.eval()

    x = torch.randn(2, 5, 12)
    out = net(x, *_tokens(torch, 2, 0))

    assert out["continuous"].shape == (2, 4)
    assert out["financing_event_logits"].shape == (2, 4)
    assert out["financing_event_prob"].shape == (2, 4)
    assert out["financing_count"].shape == (2, 4)
    assert out["financing_amount_log"].shape == (2, 4)
    assert out["financing_process_blend"].shape == (2, 4)


def test_v742_unified_wrapper_reports_single_path_financing_regime():
    wrapper = V740AlphaPrototypeWrapper(
        enable_financing_consistency=True,
        financing_consistency_strength=0.10,
        financing_auxiliary_strength=0.12,
        financing_process_blend=0.65,
        enable_target_routing=False,
        enable_count_source_routing=False,
        enable_window_repair=True,
    )
    wrapper._task_name = "task2_forecast"
    wrapper._target_name = "investors_count"
    wrapper._ablation_name = "full"
    wrapper._horizon = 7

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()
    wrapper._refresh_financing_consistency_regime(pd.DataFrame(columns=["entity_id", "funding_raised_usd", "investors_count", "is_funded"]))

    regime = wrapper.get_regime_info()

    assert regime["financing_process"]["enabled"] is True
    assert regime["financing_process"]["effective_enabled"] is True
    assert regime["financing_process"]["effective_blend"] == pytest.approx(0.65)
    assert regime["financing_process"]["single_path_bias"] is True
    assert regime["target_routing"]["effective_enabled"] is False
    assert regime["target_routing"]["count_head_type"] == "shared_financing_process_blend"


def test_v743_factorized_forward_emits_factorized_outputs(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=5,
        seq_len=12,
        horizon=4,
        hidden_dim=16,
        enable_target_routing=False,
        enable_count_source_routing=False,
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        financing_process_blend=0.20,
        funding_log_domain_enabled=True,
    )
    net.eval()

    x = torch.randn(2, 5, 12)
    out = net(x, *_tokens(torch, 2, 1))

    assert out["financing_count"].shape == (2, 4)
    assert out["financing_breadth_log"].shape == (2, 4)
    assert out["financing_breadth_anchor_log"].shape == (2, 4)
    assert out["financing_intensity_log"].shape == (2, 4)
    assert out["financing_amount_coupling"].shape == (2, 4)
    assert out["legacy_count"].shape == (2, 4)


def test_v743_factorized_wrapper_reports_factorized_regime():
    wrapper = V740AlphaPrototypeWrapper(
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        financing_consistency_strength=0.12,
        financing_auxiliary_strength=0.14,
        financing_process_blend=0.20,
        financing_scaffold_strength=0.08,
        enable_target_routing=False,
        enable_count_source_routing=False,
        enable_window_repair=True,
    )
    wrapper._task_name = "task2_forecast"
    wrapper._target_name = "investors_count"
    wrapper._ablation_name = "core_edgar"
    wrapper._horizon = 1

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()
    wrapper._refresh_financing_consistency_regime(
        pd.DataFrame(columns=["entity_id", "funding_raised_usd", "investors_count", "is_funded"])
    )

    regime = wrapper.get_regime_info()

    assert regime["financing_process"]["effective_enabled"] is True
    assert regime["financing_process"]["factorized_state_enabled"] is True
    assert regime["financing_process"]["scaffold_strength"] == pytest.approx(0.08)


def test_v744_guarded_phase_forward_emits_guard_gates(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=5,
        seq_len=12,
        horizon=4,
        hidden_dim=16,
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        enable_financing_guarded_phase=True,
        financing_process_blend=0.20,
        funding_log_domain_enabled=True,
    )
    net.eval()

    x = torch.randn(2, 5, 12)
    out = net(x, *_tokens(torch, 2, 1))

    assert out["financing_investor_gate"].shape == (2, 4)
    assert out["financing_binary_gate"].shape == (2, 4)
    assert out["financing_funding_gate"].shape == (2, 4)


def test_v744_guarded_phase_wrapper_reports_guarded_regime():
    wrapper = V740AlphaPrototypeWrapper(
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        enable_financing_guarded_phase=True,
        financing_consistency_strength=0.10,
        financing_auxiliary_strength=0.12,
        financing_process_blend=0.20,
        financing_scaffold_strength=0.08,
    )
    wrapper._task_name = "task1_outcome"
    wrapper._target_name = "is_funded"
    wrapper._ablation_name = "core_edgar"
    wrapper._horizon = 14

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()
    wrapper._refresh_financing_consistency_regime(
        pd.DataFrame(columns=["entity_id", "funding_raised_usd", "investors_count", "is_funded"])
    )

    regime = wrapper.get_regime_info()

    assert regime["financing_process"]["factorized_state_enabled"] is True
    assert regime["financing_process"]["guarded_phase_enabled"] is True
    assert regime["financing_process"]["target_loss_scale"] == pytest.approx(0.15)


def test_v745_evidence_residual_wrapper_reports_investor_regime():
    wrapper = V740AlphaPrototypeWrapper(
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        enable_financing_evidence_residual=True,
        financing_consistency_strength=0.12,
        financing_auxiliary_strength=0.14,
        financing_process_blend=0.20,
        financing_scaffold_strength=0.08,
        enable_target_routing=True,
        enable_count_source_routing=True,
    )
    wrapper._task_name = "task2_forecast"
    wrapper._target_name = "investors_count"
    wrapper._ablation_name = "core_edgar"
    wrapper._horizon = 1

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()
    wrapper._refresh_financing_consistency_regime(
        pd.DataFrame(columns=["entity_id", "funding_raised_usd", "investors_count", "is_funded"])
    )

    regime = wrapper.get_regime_info()

    assert regime["financing_process"]["evidence_residual_enabled"] is True
    assert regime["target_routing"]["effective_enabled"] is False
    assert regime["financing_process"]["target_loss_scale"] == pytest.approx(1.0)


def test_v745_evidence_residual_wrapper_zeros_noninvestor_scale():
    wrapper = V740AlphaPrototypeWrapper(
        enable_financing_consistency=True,
        enable_financing_factorization=True,
        enable_financing_evidence_residual=True,
        financing_consistency_strength=0.12,
        financing_auxiliary_strength=0.14,
        financing_process_blend=0.20,
        financing_scaffold_strength=0.08,
    )
    wrapper._task_name = "task1_outcome"
    wrapper._target_name = "is_funded"
    wrapper._ablation_name = "core_edgar"
    wrapper._horizon = 14

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()
    wrapper._refresh_financing_consistency_regime(
        pd.DataFrame(columns=["entity_id", "funding_raised_usd", "investors_count", "is_funded"])
    )

    regime = wrapper.get_regime_info()

    assert regime["financing_process"]["evidence_residual_enabled"] is True
    assert regime["financing_process"]["target_loss_scale"] == pytest.approx(0.0)


def test_v742_unified_financing_diagnostics_summary_math():
    wrapper = V740AlphaPrototypeWrapper(enable_financing_consistency=True)
    wrapper._target_name = "investors_count"
    wrapper._horizon = 2

    wrapper.collect_window_outputs = lambda raw_df: {
        "available": True,
        "window_count": 2,
        "horizon": 2,
        "edgar_active_rate": 1.0,
        "text_active_rate": 0.5,
        "financing_event_prob": np.array([[0.8, 0.4], [0.2, 0.1]], dtype=np.float64),
        "financing_count": np.array([[2.0, 1.0], [0.2, 0.0]], dtype=np.float64),
        "financing_count_positive": np.array([[2.5, 1.5], [0.4, 0.0]], dtype=np.float64),
        "financing_amount_log": np.array([[1.5, 1.0], [0.1, 0.0]], dtype=np.float64),
        "binary_true": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        "investors_true": np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        "funding_true": np.array([[4.0, 2.0], [0.0, 0.0]], dtype=np.float64),
        "primary_count": np.array([[1.8, 0.9], [0.3, 0.1]], dtype=np.float64),
        "financing_process_blend": np.full((2, 2), 0.2, dtype=np.float64),
    }

    diag = wrapper.score_financing_diagnostics(pd.DataFrame())

    assert diag["has_financing_head"] is True
    assert diag["point_count"] == 4
    assert diag["blend_strength_mean"] == pytest.approx(0.2)
    assert diag["primary_vs_financing_count_gap"] == pytest.approx(0.125)
    assert diag["coherence_mse"] >= 0.0
    assert diag["inactive_joint_mass"] >= 0.0