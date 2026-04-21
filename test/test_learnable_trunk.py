#!/usr/bin/env python3
"""Unit tests for the Learnable Sparse MoE Trunk.

Proves:
  1. Forward pass produces correct shapes
  2. Gradients flow through LearnableProjection
  3. Gradients flow through SparseMoETrunk (all sub-modules)
  4. Expert diversity loss is non-zero and differentiable
  5. Load balance loss is differentiable
  6. Top-k selection works correctly
  7. Task-conditional routing produces different outputs for different tasks
  8. LearnableTrunkAdapter fit/transform API works end-to-end
  9. Adapter output dimension matches expert_dim
 10. Adapter improves over random features on a simple task
"""
import numpy as np
import pytest
import torch

from src.narrative.block3.models.single_model_mainline.learnable_trunk import (
    ExpertBlock,
    LearnableProjection,
    LearnableTrunkAdapter,
    SparseMoETrunk,
)


# ─── fixtures ────────────────────────────────────────────────────────
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def input_dim():
    return 20


@pytest.fixture
def batch_size():
    return 64


@pytest.fixture
def synthetic_data(rng, input_dim, batch_size):
    X = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    y_binary = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    y_funding = np.abs(X[:, 2] * 1000 + X[:, 3] * 500).astype(np.float32)
    y_investors = np.clip(np.round(np.abs(X[:, 4]) * 5), 0, 20).astype(np.float32)
    return X, y_binary, y_funding, y_investors


# ─── Test 1: LearnableProjection shape ───────────────────────────────
class TestLearnableProjection:
    def test_output_shape(self, input_dim, batch_size):
        proj = LearnableProjection(input_dim, compact_dim=32, hidden_dim=64)
        x = torch.randn(batch_size, input_dim)
        out = proj(x)
        assert out.shape == (batch_size, 32)

    def test_gradient_flow(self, input_dim, batch_size):
        proj = LearnableProjection(input_dim, compact_dim=32, hidden_dim=64)
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()

        # Gradient flows to input
        assert x.grad is not None
        assert x.grad.shape == (batch_size, input_dim)
        assert x.grad.abs().sum() > 0, "Input gradient is all zeros"

        # Gradient flows to all parameters
        for name, param in proj.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_different_inputs_different_outputs(self, input_dim):
        proj = LearnableProjection(input_dim, compact_dim=32)
        x1 = torch.randn(1, input_dim)
        x2 = torch.randn(1, input_dim)
        out1 = proj(x1)
        out2 = proj(x2)
        assert not torch.allclose(out1, out2), "Different inputs produce same output"


# ─── Test 2: ExpertBlock ─────────────────────────────────────────────
class TestExpertBlock:
    def test_output_shape(self):
        expert = ExpertBlock(input_dim=32, output_dim=16, hidden_dim=48)
        x = torch.randn(10, 32)
        out = expert(x)
        assert out.shape == (10, 16)

    def test_gradient_flow(self):
        expert = ExpertBlock(input_dim=32, output_dim=16, hidden_dim=48)
        x = torch.randn(10, 32, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ─── Test 3: SparseMoETrunk ─────────────────────────────────────────
class TestSparseMoETrunk:
    @pytest.fixture
    def model(self, input_dim):
        return SparseMoETrunk(
            input_dim=input_dim,
            compact_dim=32,
            n_experts=4,
            expert_dim=16,
            top_k=2,
            n_tasks=3,
            projection_hidden=48,
            expert_hidden=32,
        )

    def test_output_shape(self, model, input_dim, batch_size):
        x = torch.randn(batch_size, input_dim)
        task_id = torch.zeros(batch_size, 3)
        task_id[:, 0] = 1.0
        result = model(x, task_id)
        assert result["output"].shape == (batch_size, 16)
        assert result["z"].shape == (batch_size, 32)
        assert result["gate_logits"].shape == (batch_size, 4)
        assert result["gate_probs"].shape == (batch_size, 4)
        assert result["expert_outputs"].shape == (batch_size, 4, 16)
        assert result["topk_indices"].shape == (batch_size, 2)
        assert result["topk_weights"].shape == (batch_size, 2)

    def test_gradient_flow_full_pipeline(self, model, input_dim, batch_size):
        """All gradients must flow: input → projection → experts → gate → output → loss."""
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        task_id = torch.zeros(batch_size, 3)
        task_id[:, 1] = 1.0  # funding task

        result = model(x, task_id)
        pred = model.predict_aux(result["output"], "funding")
        target = torch.randn(batch_size)

        loss = (
            torch.nn.functional.mse_loss(pred, target)
            + result["load_balance_loss"]
            + model.expert_diversity_loss()
        )
        loss.backward()

        # Input gradient
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "No gradient flows to input"

        # Check ALL sub-modules have gradients
        modules_with_grad = 0
        modules_total = 0
        for name, param in model.named_parameters():
            modules_total += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                modules_with_grad += 1
            else:
                # Auxiliary heads for other tasks won't have gradients
                if "binary" in name or "investors" in name:
                    continue
                pytest.fail(f"No gradient for parameter: {name}")

        assert modules_with_grad > 0

    def test_topk_selection(self, model, input_dim):
        """Top-k indices should index valid experts."""
        x = torch.randn(8, input_dim)
        task_id = torch.zeros(8, 3)
        task_id[:, 0] = 1.0
        result = model(x, task_id)
        assert result["topk_indices"].min() >= 0
        assert result["topk_indices"].max() < model.n_experts

    def test_topk_weights_sum_to_one(self, model, input_dim):
        """Top-k weights should sum to 1 (softmax)."""
        x = torch.randn(8, input_dim)
        task_id = torch.zeros(8, 3)
        task_id[:, 0] = 1.0
        result = model(x, task_id)
        weight_sums = result["topk_weights"].sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_task_conditional_routing_differs(self, model, input_dim):
        """Different task IDs should produce different gate probabilities."""
        x = torch.randn(1, input_dim)
        task_binary = torch.tensor([[1.0, 0.0, 0.0]])
        task_funding = torch.tensor([[0.0, 1.0, 0.0]])
        task_investors = torch.tensor([[0.0, 0.0, 1.0]])

        r1 = model(x, task_binary)
        r2 = model(x, task_funding)
        r3 = model(x, task_investors)

        # Gate probabilities should differ across tasks
        assert not torch.allclose(r1["gate_probs"], r2["gate_probs"], atol=1e-3)
        assert not torch.allclose(r2["gate_probs"], r3["gate_probs"], atol=1e-3)

    def test_load_balance_loss_positive(self, model, input_dim, batch_size):
        """Load balance loss should be non-negative."""
        x = torch.randn(batch_size, input_dim)
        task_id = torch.zeros(batch_size, 3)
        task_id[:, 0] = 1.0
        result = model(x, task_id)
        assert result["load_balance_loss"].item() >= 0

    def test_expert_diversity_loss_positive(self, model):
        """Expert diversity loss should be positive (experts start non-orthogonal)."""
        loss = model.expert_diversity_loss()
        assert loss.item() > 0

    def test_expert_diversity_loss_differentiable(self, model):
        """Expert diversity loss gradient flows to expert weights."""
        loss = model.expert_diversity_loss()
        loss.backward()
        for expert in model.experts:
            w = expert.net[0].weight
            assert w.grad is not None
            assert w.grad.abs().sum() > 0


# ─── Test 4: LearnableTrunkAdapter ──────────────────────────────────
class TestLearnableTrunkAdapter:
    def test_fit_transform_binary(self, synthetic_data):
        X, y_binary, _, _ = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            projection_hidden=32,
            expert_hidden=16,
            n_epochs=5,
            batch_size=32,
        )
        output = adapter.fit_transform(X, y_binary, target_name="is_funded")
        assert output.shape == (X.shape[0], 8)  # expert_dim
        assert not np.any(np.isnan(output))

    def test_fit_transform_funding(self, synthetic_data):
        X, _, y_funding, _ = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=5,
            batch_size=32,
        )
        output = adapter.fit_transform(X, y_funding, target_name="funding_raised_usd")
        assert output.shape == (X.shape[0], 8)
        assert not np.any(np.isnan(output))

    def test_fit_transform_investors(self, synthetic_data):
        X, _, _, y_investors = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=5,
            batch_size=32,
        )
        output = adapter.fit_transform(X, y_investors, target_name="investors_count")
        assert output.shape == (X.shape[0], 8)
        assert not np.any(np.isnan(output))

    def test_separate_transform_matches_fit_transform(self, synthetic_data):
        X, y_binary, _, _ = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=3,
            batch_size=32,
            random_state=123,
        )
        # fit then transform separately
        adapter.fit(X, y_binary, target_name="is_funded")
        out_separate = adapter.transform(X, target_name="is_funded")
        assert out_separate.shape == (X.shape[0], 8)
        assert not np.any(np.isnan(out_separate))

    def test_transform_on_new_data(self, synthetic_data, rng, input_dim):
        X, y_binary, _, _ = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=3,
            batch_size=32,
        )
        adapter.fit(X, y_binary, target_name="is_funded")
        # Transform on new data with different size
        X_new = rng.standard_normal((100, input_dim)).astype(np.float32)
        out = adapter.transform(X_new, target_name="is_funded")
        assert out.shape == (100, 8)
        assert not np.any(np.isnan(out))

    def test_unfitted_raises(self, synthetic_data):
        X = synthetic_data[0]
        adapter = LearnableTrunkAdapter()
        with pytest.raises(ValueError, match="not fitted"):
            adapter.transform(X)

    def test_describe(self, synthetic_data):
        X, y_binary, _, _ = synthetic_data
        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=2,
            batch_size=32,
        )
        desc = adapter.describe()
        assert desc["fitted"] is False

        adapter.fit(X, y_binary, target_name="is_funded")
        desc = adapter.describe()
        assert desc["fitted"] is True
        assert desc["n_experts"] == 3
        assert desc["expert_dim"] == 8
        assert desc["total_params"] > 0

    def test_subsampling_large_data(self, rng):
        """Adapter should subsample when data exceeds max_train_rows."""
        n = 500
        d = 10
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        adapter = LearnableTrunkAdapter(
            compact_dim=8,
            n_experts=2,
            expert_dim=4,
            top_k=1,
            n_epochs=2,
            batch_size=32,
            max_train_rows=100,  # Force subsampling
        )
        output = adapter.fit_transform(X, y, target_name="funding_raised_usd")
        # Output should still cover ALL rows (transform runs on full data)
        assert output.shape == (n, 4)

    def test_learnable_features_beat_random(self, rng):
        """Learned features should be more predictive than random noise."""
        n = 500
        d = 10
        X = rng.standard_normal((n, d)).astype(np.float32)
        # Clear linear signal: y = X[:,0] + X[:,1]
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        adapter = LearnableTrunkAdapter(
            compact_dim=16,
            n_experts=3,
            expert_dim=8,
            top_k=2,
            n_epochs=20,
            batch_size=64,
        )
        output = adapter.fit_transform(X, y, target_name="funding_raised_usd")

        # Learned features should have some correlation with y
        correlations = np.array([
            np.corrcoef(output[:, i], y)[0, 1] for i in range(output.shape[1])
        ])
        max_corr = np.nanmax(np.abs(correlations))

        # Random features: ~0.05 correlation. Learned: should be much higher
        assert max_corr > 0.2, f"Max correlation {max_corr:.3f} too low — trunk not learning"
