#!/usr/bin/env python3
"""Unit tests for the Frequency-Decoupled SSM (FD-SSM) with NCDE Modulation.

Proves:
  1. SeriesDecomp: exact decomposition, shape preservation
  2. NCDESelectiveSSMCell: shapes, gradient flow, ZOH stability, event modulation
  3. EventDrivenSSM: full pipeline shapes, return_states, variable lengths
  4. Adaptive degradation: pure-SSM mode when no event features supplied
  5. SequentialMoETrunk: end-to-end SSM -> MoE pipeline
  6. FDSSMForecaster: forecasting head shapes and gradient flow
  7. Generalization modes:
     - Data A (VC/MIMIC): sparse continuous + discrete impulse events
     - Data B (ETT/Weather): dense continuous, zero events
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.narrative.block3.models.single_model_mainline.event_driven_ssm import (
    EventDrivenSSM,
    FDSSMForecaster,
    NCDESelectiveSSMCell,
    SequentialMoETrunk,
    SeriesDecomp,
    UnifiedJumpDiffusionSSM,
)


# --- fixtures ---
@pytest.fixture
def B():
    return 8

@pytest.fixture
def L():
    return 20

@pytest.fixture
def d_cont():
    return 10

@pytest.fixture
def d_event():
    return 3

@pytest.fixture
def d_model():
    return 32

@pytest.fixture
def d_state():
    return 8


# ===================================================================
#  1. SeriesDecomp
# ===================================================================
class TestSeriesDecomp:
    def test_output_shape(self, B, L, d_cont):
        decomp = SeriesDecomp(kernel_size=5)
        x = torch.randn(B, L, d_cont)
        trend, seasonal = decomp(x)
        assert trend.shape == (B, L, d_cont)
        assert seasonal.shape == (B, L, d_cont)

    def test_exact_decomposition(self, B, L, d_cont):
        decomp = SeriesDecomp(kernel_size=5)
        x = torch.randn(B, L, d_cont)
        trend, seasonal = decomp(x)
        recon = trend + seasonal
        assert torch.allclose(x, recon, atol=1e-5), "Decomposition not exact"

    def test_causal(self):
        """Changing future values should not affect past trend."""
        decomp = SeriesDecomp(kernel_size=3)
        x = torch.randn(1, 10, 1)
        trend1, _ = decomp(x)
        x2 = x.clone()
        x2[0, 8:, :] = 999.0
        trend2, _ = decomp(x2)
        assert torch.allclose(trend1[0, :8], trend2[0, :8], atol=1e-5), \
            "Decomposition is not causal"


# ===================================================================
#  2. NCDESelectiveSSMCell
# ===================================================================
class TestNCDESelectiveSSMCell:
    def test_output_shape(self, B, d_model, d_state):
        cell = NCDESelectiveSSMCell(d_model, d_state)
        x_t = torch.randn(B, d_model)
        h_prev = torch.zeros(B, d_model, d_state)
        y_t, h_t = cell(x_t, h_prev, event_t=torch.zeros(B, 0))
        assert y_t.shape == (B, d_model), f"y_t shape: {y_t.shape}"
        assert h_t.shape == (B, d_model, d_state), f"h_t shape: {h_t.shape}"

    def test_output_shape_with_events(self, B, d_model, d_state, d_event):
        cell = NCDESelectiveSSMCell(d_model, d_state, d_event=d_event)
        x_t = torch.randn(B, d_model)
        h_prev = torch.zeros(B, d_model, d_state)
        e_t = torch.randn(B, d_event)
        y_t, h_t = cell(x_t, h_prev, event_t=e_t)
        assert y_t.shape == (B, d_model)
        assert h_t.shape == (B, d_model, d_state)

    def test_gradient_flow(self, B, d_model, d_state):
        cell = NCDESelectiveSSMCell(d_model, d_state)
        x_t = torch.randn(B, d_model, requires_grad=True)
        h_prev = torch.zeros(B, d_model, d_state, requires_grad=True)
        y_t, h_t = cell(x_t, h_prev, event_t=torch.zeros(B, 0))
        loss = y_t.sum() + h_t.sum()
        loss.backward()
        assert x_t.grad is not None
        assert x_t.grad.abs().sum() > 0, "No gradient to input"
        for name, p in cell.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gradient_flow_with_events(self, B, d_model, d_state, d_event):
        cell = NCDESelectiveSSMCell(d_model, d_state, d_event=d_event)
        x_t = torch.randn(B, d_model, requires_grad=True)
        h_prev = torch.zeros(B, d_model, d_state)
        e_t = torch.randn(B, d_event, requires_grad=True)
        y_t, h_t = cell(x_t, h_prev, event_t=e_t)
        (y_t.sum() + h_t.sum()).backward()
        assert e_t.grad is not None and e_t.grad.abs().sum() > 0, \
            "Event features have no gradient"

    def test_state_evolves(self, d_model, d_state):
        cell = NCDESelectiveSSMCell(d_model, d_state)
        x = torch.randn(1, d_model)
        h0 = torch.zeros(1, d_model, d_state)
        e0 = torch.zeros(1, 0)
        _, h1 = cell(x, h0, event_t=e0)
        _, h2 = cell(x, h1, event_t=e0)
        assert not torch.allclose(h1, h2, atol=1e-6), "State did not evolve"

    def test_diagonal_A_stability(self, d_model, d_state):
        cell = NCDESelectiveSSMCell(d_model, d_state)
        A = -torch.exp(cell.A_log)
        assert (A < 0).all(), "A eigenvalues must be strictly negative"

    def test_event_modulates_delta(self, d_model, d_state):
        """Events should change the effective step size."""
        d_event = 3
        cell = NCDESelectiveSSMCell(d_model, d_state, d_event=d_event)
        # Set event weights to non-zero for test
        with torch.no_grad():
            cell.delta_event_proj.weight.fill_(0.5)
        x_t = torch.randn(1, d_model)
        h0 = torch.zeros(1, d_model, d_state)
        e_zero = torch.zeros(1, d_event)
        e_active = torch.ones(1, d_event)
        y0, h0_out = cell(x_t, h0, event_t=e_zero)
        y1, h1_out = cell(x_t, h0, event_t=e_active)
        assert not torch.allclose(h0_out, h1_out, atol=1e-5), \
            "Events did not change hidden state (delta not modulated)"


# ===================================================================
#  3. EventDrivenSSM -- basic
# ===================================================================
class TestEventDrivenSSM:
    def test_output_shape_with_events(self, B, L, d_cont, d_event, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event, d_model, d_state, d_output=48, n_layers=2)
        x = torch.randn(B, L, d_cont + d_event)
        out = model(x)
        assert out.shape == (B, 48), f"output shape: {out.shape}"

    def test_output_shape_no_events(self, B, L, d_cont, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event=0, d_model=d_model, d_state=d_state, d_output=48)
        x = torch.randn(B, L, d_cont)
        out = model(x)
        assert out.shape == (B, 48)

    def test_gradient_flow_full(self, B, L, d_cont, d_event, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event, d_model, d_state, d_output=48, n_layers=2)
        x = torch.randn(B, L, d_cont + d_event, requires_grad=True)
        with torch.no_grad():
            x[:, 5, d_cont:] = 1.0
            x[:, 12, d_cont:] = 1.0
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_variable_lengths(self, B, d_cont, d_model, d_state):
        L = 15
        model = EventDrivenSSM(d_cont, 0, d_model, d_state, d_output=32)
        x = torch.randn(B, L, d_cont)
        lengths = torch.randint(5, L + 1, (B,))
        out = model(x, lengths=lengths)
        assert out.shape == (B, 32)
        assert not torch.isnan(out).any()


# ===================================================================
#  4. State Observability (return_states)
# ===================================================================
class TestReturnStates:
    def test_return_states_shapes(self, B, L, d_cont, d_event, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event, d_model, d_state, d_output=48, n_layers=2)
        x = torch.randn(B, L, d_cont + d_event)
        result = model(x, return_states=True)
        assert isinstance(result, dict), "return_states=True must return dict"
        assert result["output"].shape == (B, 48)
        assert result["trajectory"].shape == (B, L, d_model * d_state)
        assert result["jump_gates"].shape == (B, L)

    def test_return_states_false_is_tensor(self, B, L, d_cont, d_model, d_state):
        model = EventDrivenSSM(d_cont, 0, d_model, d_state, d_output=48)
        x = torch.randn(B, L, d_cont)
        out = model(x, return_states=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, 48)

    def test_jump_gates_zero_legacy(self, d_cont, d_model, d_state):
        """Jump gates are always zero in FD-SSM (legacy compat key)."""
        d_event = 2
        B, L = 4, 10
        model = EventDrivenSSM(d_cont, d_event, d_model, d_state, d_output=32)
        x = torch.randn(B, L, d_cont + d_event)
        x[:, 3, d_cont:] = 1.0
        x[:, 7, d_cont:] = 1.0
        result = model(x, return_states=True)
        jg = result["jump_gates"]
        assert (jg == 0).all(), "jump_gates should always be zero (legacy compat)"

    def test_trajectory_evolves(self, d_cont, d_model, d_state):
        B, L = 2, 8
        model = EventDrivenSSM(d_cont, 0, d_model, d_state, d_output=32)
        x = torch.randn(B, L, d_cont)
        result = model(x, return_states=True)
        traj = result["trajectory"]
        assert not torch.allclose(traj[:, 0], traj[:, -1], atol=1e-5), \
            "Trajectory did not evolve across time"


# ===================================================================
#  5. Adaptive Degradation
# ===================================================================
class TestAdaptiveDegradation:
    def test_model_with_events_but_no_event_input(self, B, L, d_cont, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event=3, d_model=d_model, d_state=d_state, d_output=32)
        x = torch.randn(B, L, d_cont)
        out = model(x)
        assert out.shape == (B, 32)
        assert not torch.isnan(out).any()

    def test_model_with_events_but_no_event_input_return_states(self, B, L, d_cont, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event=3, d_model=d_model, d_state=d_state, d_output=32)
        x = torch.randn(B, L, d_cont)
        result = model(x, return_states=True)
        assert result["output"].shape == (B, 32)
        assert result["jump_gates"].shape == (B, L)
        assert (result["jump_gates"] == 0).all()

    def test_gradient_still_flows_in_pure_ssm_mode(self, B, L, d_cont, d_model, d_state):
        model = EventDrivenSSM(d_cont, d_event=2, d_model=d_model, d_state=d_state, d_output=32)
        x = torch.randn(B, L, d_cont, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_zero_event_is_exactly_pure_ssm_output_and_grad(self):
        """Zero events must be algebraically identical to pure continuous SSM."""
        torch.manual_seed(123)
        B, L, d_cont = 4, 12, 6
        d_event, d_model, d_state, d_output = 3, 16, 5, 9

        model_event = UnifiedJumpDiffusionSSM(
            d_cont=d_cont,
            d_event=d_event,
            d_model=d_model,
            d_state=d_state,
            d_output=d_output,
            n_layers=2,
        ).eval()
        model_pure = UnifiedJumpDiffusionSSM(
            d_cont=d_cont,
            d_event=0,
            d_model=d_model,
            d_state=d_state,
            d_output=d_output,
            n_layers=2,
        ).eval()

        # Align shared parameters so any output difference can only come
        # from event perturbation terms.
        pure_sd = model_pure.state_dict()
        for k, v in model_event.state_dict().items():
            if k in pure_sd and pure_sd[k].shape == v.shape:
                pure_sd[k] = v.clone()
        model_pure.load_state_dict(pure_sd)

        x_cont = torch.randn(B, L, d_cont)
        x_with_zero_event = torch.cat([x_cont, torch.zeros(B, L, d_event)], dim=-1)

        x_evt = x_with_zero_event.clone().requires_grad_(True)
        x_pure = x_cont.clone().requires_grad_(True)

        y_evt = model_event(x_evt)
        y_pure = model_pure(x_pure)
        assert torch.allclose(y_evt, y_pure, atol=1e-7), "Zero-event output mismatch"

        y_evt.sum().backward()
        y_pure.sum().backward()
        assert torch.allclose(
            x_evt.grad[..., :d_cont], x_pure.grad, atol=1e-7
        ), "Zero-event gradient mismatch"


# ===================================================================
#  6. SequentialMoETrunk
# ===================================================================
class TestSequentialMoETrunk:
    @pytest.fixture
    def pipeline(self, d_cont, d_event, d_model, d_state):
        return SequentialMoETrunk(
            d_cont=d_cont, d_event=d_event,
            d_model=d_model, d_state=d_state,
            n_ssm_layers=2, d_jump=32,
            compact_dim=32, n_experts=4, expert_dim=16,
            top_k=2, n_tasks=3,
            projection_hidden=48, expert_hidden=32,
        )

    def test_output_shape(self, pipeline, B, L, d_cont, d_event):
        x = torch.randn(B, L, d_cont + d_event)
        tid = torch.zeros(B, 3); tid[:, 0] = 1.0
        result = pipeline(x, tid)
        assert result["output"].shape == (B, 16)

    def test_gradient_flow_through_pipeline(self, pipeline, B, L, d_cont, d_event):
        x = torch.randn(B, L, d_cont + d_event, requires_grad=True)
        tid = torch.zeros(B, 3); tid[:, 1] = 1.0
        result = pipeline(x, tid)
        loss = result["output"].sum() + result["load_balance_loss"]
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_return_states_through_pipeline(self, pipeline, B, L, d_cont, d_event, d_model, d_state):
        x = torch.randn(B, L, d_cont + d_event)
        tid = torch.zeros(B, 3); tid[:, 2] = 1.0
        result = pipeline(x, tid, return_states=True)
        assert "trajectory" in result
        assert result["trajectory"].shape == (B, L, d_model * d_state)
        assert "jump_gates" in result
        assert result["jump_gates"].shape == (B, L)
        assert result["output"].shape == (B, 16)

    def test_pure_ssm_mode_through_pipeline(self, d_cont, d_model, d_state):
        B, L = 4, 12
        pipe = SequentialMoETrunk(
            d_cont=d_cont, d_event=3,
            d_model=d_model, d_state=d_state,
            n_ssm_layers=1, d_jump=16,
            compact_dim=16, n_experts=2, expert_dim=8,
            top_k=1, n_tasks=3,
            projection_hidden=24, expert_hidden=16,
        )
        x = torch.randn(B, L, d_cont)
        tid = torch.zeros(B, 3); tid[:, 0] = 1.0
        result = pipe(x, tid, return_states=True)
        assert result["output"].shape == (B, 8)
        assert (result["jump_gates"] == 0).all()


# ===================================================================
#  7. FDSSMForecaster
# ===================================================================
class TestFDSSMForecaster:
    def test_output_shape(self, B):
        model = FDSSMForecaster(d_input=7, seq_len=96, pred_len=96, d_model=32)
        x = torch.randn(B, 96, 7)
        out = model(x)
        assert out.shape == (B, 96)

    def test_output_shape_individual(self, B):
        model = FDSSMForecaster(d_input=7, seq_len=96, pred_len=48, d_model=32, individual=True)
        x = torch.randn(B, 96, 7)
        out = model(x)
        assert out.shape == (B, 48)

    def test_gradient_flow(self, B):
        model = FDSSMForecaster(d_input=7, seq_len=96, pred_len=96, d_model=32)
        x = torch.randn(B, 96, 7, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_unified_forecaster_has_no_external_dlinear_branch(self):
        model = FDSSMForecaster(d_input=7, seq_len=96, pred_len=96, d_model=32)
        assert not hasattr(model, "trend_linear")
        assert not hasattr(model, "seasonal_linear")


# ===================================================================
#  8. Generalization Mode Tests
# ===================================================================
class TestGeneralizationModes:
    def test_data_a_vc_mimic_sparse_events(self):
        """Data A: sparse continuous + discrete impulse events (NCDE modulation)."""
        B, L, d_cont, d_event = 6, 30, 15, 4
        d_model, d_state = 24, 8

        model = EventDrivenSSM(
            d_cont, d_event, d_model, d_state, d_output=32, n_layers=2,
        )

        rng = np.random.default_rng(42)
        x_np = np.zeros((B, L, d_cont + d_event), dtype=np.float32)
        mask = rng.random((B, L, d_cont)) < 0.2
        x_np[:, :, :d_cont] = mask * rng.standard_normal((B, L, d_cont)).astype(np.float32)
        event_mask = rng.random((B, L, d_event)) < 0.10
        x_np[:, :, d_cont:] = event_mask.astype(np.float32)

        x = torch.from_numpy(x_np).requires_grad_(True)
        result = model(x, return_states=True)
        output = result["output"]

        assert output.shape == (B, 32)
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_data_b_ett_weather_dense_continuous(self):
        """Data B (ETT/Weather): dense continuous, zero events."""
        B, L, d_cont = 6, 50, 20
        d_model, d_state = 24, 8

        model = EventDrivenSSM(
            d_cont, d_event=4, d_model=d_model, d_state=d_state,
            d_output=32, n_layers=2,
        )

        torch.manual_seed(123)
        x = torch.randn(B, L, d_cont, requires_grad=True)

        result = model(x, return_states=True)
        output = result["output"]
        jump_gates = result["jump_gates"]

        assert output.shape == (B, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert (jump_gates == 0).all(), "Jump gates should be zero in pure SSM mode"

        out_std = output.std(dim=0).mean().item()
        assert out_std > 1e-4, f"Output is degenerate (std={out_std:.6f})"

        output.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
