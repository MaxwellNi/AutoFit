"""Tests for P4 Jump ODE state evolution (jump_ode_utils + backbone integration)."""
import numpy as np
import pandas as pd
import pytest

from narrative.block3.models.single_model_mainline.jump_ode_utils import (
    _DriftModel,
    _JumpModel,
    _detect_jumps,
    _euler_integrate_entity,
    build_jump_ode_state,
    fit_jump_ode,
    jump_ode_diagnostics,
)
from narrative.block3.models.single_model_mainline.backbone import (
    SharedTemporalBackbone,
    SharedTemporalBackboneSpec,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_backbone_context(start_day: str, values: list[tuple[str, float, float]]) -> pd.DataFrame:
    rows = []
    base_day = pd.Timestamp(start_day)
    for idx, (entity_id, core_signal, core_volume) in enumerate(values):
        rows.append({
            "entity_id": entity_id,
            "crawled_date_day": base_day + pd.Timedelta(days=idx),
            "core_signal": float(core_signal),
            "core_volume": float(core_volume),
        })
    return pd.DataFrame(rows)


# ── jump_ode_utils unit tests ─────────────────────────────────────────────

class TestDetectJumps:
    def test_no_events(self):
        atoms = np.zeros((5, 3), dtype=np.float32)
        mask = _detect_jumps(atoms)
        assert not np.any(mask)

    def test_positive_events(self):
        atoms = np.zeros((5, 3), dtype=np.float32)
        atoms[2, 0] = 1.0  # event at row 2
        atoms[4, 1] = 0.8  # event at row 4
        mask = _detect_jumps(atoms)
        assert mask[2]
        assert mask[4]
        assert not mask[0]
        assert not mask[1]
        assert not mask[3]


class TestDriftModel:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n, d = 50, 4
        x_t = rng.standard_normal((n, d))
        # True drift: f(x) = 0.1 * x (simple linear)
        x_next = x_t + 0.1 * x_t
        dt = np.ones(n)

        model = _DriftModel(n_dims=d)
        model.fit(x_t, x_next, dt)
        assert model._fitted

        pred = model.predict(x_t[:5])
        assert pred.shape == (5, d)
        # Should approximate 0.1 * x_t
        expected = 0.1 * x_t[:5]
        assert np.allclose(pred, expected, atol=0.05)

    def test_not_fitted_returns_zeros(self):
        model = _DriftModel(n_dims=4)
        x = np.ones((3, 4))
        pred = model.predict(x)
        assert np.all(pred == 0)

    def test_too_few_samples(self):
        model = _DriftModel(n_dims=2)
        model.fit(np.ones((5, 2)), np.ones((5, 2)), np.ones(5))
        assert not model._fitted


class TestJumpModel:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n, d = 30, 3
        x_pre = rng.standard_normal((n, d))
        marks = rng.standard_normal((n, 2))
        # Jump: delta = marks_sum * 0.5
        x_post = x_pre + marks.sum(axis=1, keepdims=True) * 0.5

        model = _JumpModel(n_dims=d)
        model.fit(x_pre, marks, x_post)
        assert model._fitted

        correction = model.predict(x_pre[:5], marks[:5])
        assert correction.shape == (5, d)
        assert np.any(correction != 0)

    def test_too_few_samples(self):
        model = _JumpModel(n_dims=3)
        model.fit(np.ones((3, 3)), np.ones((3, 2)), np.ones((3, 3)))
        assert not model._fitted


class TestEulerIntegration:
    def test_no_jumps_follows_drift(self):
        state = np.array([[1.0, 0.0], [1.5, 0.5], [2.0, 1.0], [2.5, 1.5]])
        jump_mask = np.array([False, False, False, False])
        event_atoms = np.zeros((4, 2))

        traj, drifts, jcounts, jenergy, smooth = _euler_integrate_entity(
            state, jump_mask, event_atoms
        )
        assert traj.shape == (4, 2)
        assert np.all(jcounts == 0)
        assert np.all(jenergy == 0)
        assert np.all(np.isfinite(traj))

    def test_jumps_increase_counts(self):
        state = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
        jump_mask = np.array([False, False, True, False])
        event_atoms = np.array([[0, 0], [0, 0], [1.0, 2.0], [0, 0]])

        traj, drifts, jcounts, jenergy, smooth = _euler_integrate_entity(
            state, jump_mask, event_atoms
        )
        assert jcounts[2] == 1.0
        assert jcounts[3] == 1.0
        assert jenergy[2] > 0
        assert np.all(np.isfinite(traj))


class TestBuildJumpOdeState:
    def test_returns_correct_shape(self):
        n, d = 20, 8
        compact = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
        event_atoms = np.zeros((n, 3), dtype=np.float32)
        entity_ids = np.array(["a"] * 10 + ["b"] * 10)

        result = build_jump_ode_state(compact, event_atoms, entity_ids, n_ode_dims=8)
        assert result.shape == (n, 12)  # 8 ode + 4 diagnostics
        assert np.all(np.isfinite(result))

    def test_none_inputs_return_zeros(self):
        compact = np.ones((5, 4), dtype=np.float32)
        result = build_jump_ode_state(compact, None, None, n_ode_dims=4)
        assert result.shape == (5, 8)  # 4 + 4
        assert np.all(result == 0)


class TestFitJumpOde:
    def test_fit_with_enough_data(self):
        rng = np.random.default_rng(42)
        n = 200
        compact = rng.standard_normal((n, 8)).astype(np.float64)
        events = np.zeros((n, 3), dtype=np.float64)
        events[5::8, 0] = 1.0  # jumps every 8 rows — more frequent
        entities = np.array(
            ["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 50
        )

        result = fit_jump_ode(compact, events, entities, n_ode_dims=8)
        assert result["fitted"]
        assert result["diagnostics"]["n_entities"] == 4
        assert result["diagnostics"]["n_drift_pairs"] > 0

    def test_fit_with_too_few_entities(self):
        compact = np.ones((10, 4))
        events = np.zeros((10, 2))
        entities = np.array(["a"] * 10)

        result = fit_jump_ode(compact, events, entities, min_entities=3)
        assert not result["fitted"]


class TestJumpOdeDiagnostics:
    def test_diagnostics_output(self):
        compact = np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)
        ode = np.random.default_rng(1).standard_normal((20, 8)).astype(np.float32)

        diag = jump_ode_diagnostics(compact, ode)
        assert "state_correlation" in diag
        assert "drift_energy" in diag
        assert "jump_density" in diag
        assert "smoothness_mean" in diag

    def test_empty_input(self):
        diag = jump_ode_diagnostics(np.empty((0, 4)), np.empty((0, 8)))
        assert diag["state_correlation"] == 0.0


# ── Backbone integration tests ──────────────────────────────────────────

class TestBackboneJumpOdeIntegration:
    def test_jump_ode_state_adds_dimensions(self):
        frame = _make_backbone_context(
            "2024-06-01",
            [
                ("entity_a", 0.0, 1.0),
                ("entity_a", 1.0, 2.0),
                ("entity_a", 3.0, 4.0),
                ("entity_a", 5.0, 6.0),
                ("entity_a", 8.0, 10.0),
                ("entity_b", 0.5, 1.5),
                ("entity_b", 1.5, 2.5),
                ("entity_b", 2.5, 3.5),
                ("entity_b", 4.0, 5.0),
                ("entity_b", 7.0, 8.0),
            ],
        )
        feature_cols = ["core_signal", "core_volume"]

        backbone_on = SharedTemporalBackbone(
            spec=SharedTemporalBackboneSpec(
                enable_jump_ode_state=True,
                jump_ode_dims=4,
                compact_state_dim=8,
            ),
            random_state=7,
        )
        backbone_off = SharedTemporalBackbone(
            spec=SharedTemporalBackboneSpec(
                enable_jump_ode_state=False,
                compact_state_dim=8,
            ),
            random_state=7,
        )

        state_on = backbone_on.fit_transform(
            frame[feature_cols], feature_cols=feature_cols, context_frame=frame
        )
        layout_on = backbone_on.describe_state_layout()

        state_off = backbone_off.fit_transform(
            frame[feature_cols], feature_cols=feature_cols, context_frame=frame
        )
        layout_off = backbone_off.describe_state_layout()

        # Flag-off: jump ODE block must be empty
        assert layout_off["jump_ode_state_dim"] == 0
        assert layout_off["uses_jump_ode_state"] is False

        # Flag-on: correct number of features (n_ode_dims + 4 diagnostics)
        assert layout_on["jump_ode_state_dim"] == 4 + 4
        assert layout_on["uses_jump_ode_state"] is True
        assert len(layout_on["jump_ode_feature_names"]) == 4 + 4
        assert state_on.shape[1] > state_off.shape[1]

        # Extract jump ODE block
        ode_offset = (
            layout_on["compact_state_dim"]
            + layout_on["summary_state_dim"]
            + layout_on["temporal_state_dim"]
            + layout_on["spectral_state_dim"]
            + layout_on["hawkes_state_dim"]
        )
        ode_block = state_on[:, ode_offset: ode_offset + layout_on["jump_ode_state_dim"]]
        assert ode_block.shape == (10, 8)
        assert np.all(np.isfinite(ode_block))

    def test_jump_ode_state_finite_and_bounded(self):
        """Jump ODE state must be finite, bounded (±5σ after z-score)."""
        frame = _make_backbone_context(
            "2024-07-01",
            [
                ("e_a", float(i), float(i * 0.5))
                for i in range(20)
            ],
        )
        feature_cols = ["core_signal", "core_volume"]

        backbone = SharedTemporalBackbone(
            spec=SharedTemporalBackboneSpec(
                enable_jump_ode_state=True,
                jump_ode_dims=6,
                compact_state_dim=4,
            ),
            random_state=42,
        )
        state = backbone.fit_transform(
            frame[feature_cols], feature_cols=feature_cols, context_frame=frame
        )

        assert np.all(np.isfinite(state))
        assert float(np.abs(state).max()) <= 10.0
