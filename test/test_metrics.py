#!/usr/bin/env python3
"""
Unit tests for Block 3 metrics module.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "/home/pni/projects/repo_root/src")

from narrative.block3.metrics import (
    rmse,
    mae,
    mape,
    smape,
    mse,
    r2_score,
    accuracy,
    precision,
    recall,
    f1_score,
    auroc,
    crps_sample,
    BootstrapCI,
    bootstrap_ci,
    METRIC_REGISTRY,
    get_metrics_for_task,
    compute_all_metrics,
)


class TestRegressionMetrics:
    """Tests for regression metrics."""
    
    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y_true, y_pred) == 0.0
    
    def test_rmse_basic(self):
        """Test RMSE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1
        assert rmse(y_true, y_pred) == 1.0
    
    def test_mae_basic(self):
        """Test MAE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1
        assert mae(y_true, y_pred) == 1.0
    
    def test_mse_basic(self):
        """Test MSE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # All off by 1
        assert mse(y_true, y_pred) == 1.0  # (1^2 + 1^2 + 1^2) / 3
    
    def test_r2_perfect(self):
        """Test R² with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y_true, y_pred) == 1.0
    
    def test_r2_mean_predictor(self):
        """Test R² with mean predictor (should be 0)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Mean
        np.testing.assert_almost_equal(r2_score(y_true, y_pred), 0.0)
    
    def test_mape_basic(self):
        """Test MAPE with known values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])  # 10% off each
        np.testing.assert_almost_equal(mape(y_true, y_pred), 10.0, decimal=5)
    
    def test_smape_symmetric(self):
        """Test sMAPE is symmetric."""
        y_true = np.array([100.0])
        y_pred = np.array([80.0])
        
        # sMAPE should give same result when swapping
        smape1 = smape(y_true, y_pred)
        smape2 = smape(y_pred, y_true)
        np.testing.assert_almost_equal(smape1, smape2)


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy(y_true, y_pred) == 1.0
    
    def test_accuracy_half(self):
        """Test accuracy with 50% correct."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1])  # 2/4 correct
        assert accuracy(y_true, y_pred) == 0.5
    
    def test_precision_basic(self):
        """Test precision with known values."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])  # TP=1, FP=1, FN=1, TN=1
        assert precision(y_true, y_pred) == 0.5  # 1 / (1 + 1)
    
    def test_recall_basic(self):
        """Test recall with known values."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])  # TP=1, FP=1, FN=1, TN=1
        assert recall(y_true, y_pred) == 0.5  # 1 / (1 + 1)
    
    def test_f1_basic(self):
        """Test F1 with equal precision and recall."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        # P = R = 0.5, so F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        assert f1_score(y_true, y_pred) == 0.5
    
    def test_auroc_perfect(self):
        """Test AUROC with perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert auroc(y_true, y_prob) == 1.0
    
    def test_auroc_random(self):
        """Test AUROC with random predictions (should be >= 0 and <= 1)."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.random.rand(8)  # Random probabilities
        # AUROC should be valid
        result = auroc(y_true, y_prob)
        assert 0.0 <= result <= 1.0


class TestProbabilisticMetrics:
    """Tests for probabilistic metrics."""
    
    def test_crps_sample_perfect(self):
        """Test CRPS with samples at true value."""
        y_true = np.array([1.0, 2.0])
        y_samples = np.array([
            [1.0, 1.0, 1.0],  # All at true value
            [2.0, 2.0, 2.0],
        ])
        assert crps_sample(y_true, y_samples) == 0.0
    
    def test_crps_sample_basic(self):
        """Test CRPS with spread samples."""
        y_true = np.array([1.0])
        y_samples = np.array([
            [0.0, 1.0, 2.0],  # Spread around true value
        ])
        crps_val = crps_sample(y_true, y_samples)
        assert crps_val > 0  # Should be positive for imperfect samples


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""
    
    def test_bootstrap_ci_basic(self):
        """Test bootstrap CI returns valid structure."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        ci = bootstrap_ci(y_true, y_pred, rmse, n_bootstrap=100, seed=42)
        
        assert isinstance(ci, BootstrapCI)
        assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
        assert ci.std_error > 0
        assert ci.n_bootstrap == 100
    
    def test_bootstrap_ci_reproducible(self):
        """Test bootstrap CI is reproducible with same seed."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        ci1 = bootstrap_ci(y_true, y_pred, rmse, n_bootstrap=100, seed=42)
        ci2 = bootstrap_ci(y_true, y_pred, rmse, n_bootstrap=100, seed=42)
        
        assert ci1.point_estimate == ci2.point_estimate
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper


class TestMetricRegistry:
    """Tests for metric registry."""
    
    def test_registry_has_standard_metrics(self):
        """Test registry contains standard metrics."""
        standard_metrics = ["rmse", "mae", "mape", "smape", "accuracy", "f1"]
        for metric in standard_metrics:
            assert metric in METRIC_REGISTRY
    
    def test_get_metrics_for_task_forecast(self):
        """Test metrics for forecast task."""
        metrics = get_metrics_for_task("forecast")
        assert "rmse" in metrics
        assert "mae" in metrics
    
    def test_get_metrics_for_task_classification(self):
        """Test metrics for classification task."""
        metrics = get_metrics_for_task("classification")
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics
    
    def test_compute_all_metrics_basic(self):
        """Test compute_all_metrics returns correct structure."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        results = compute_all_metrics(y_true, y_pred, ["rmse", "mae"])
        
        assert "rmse" in results
        assert "mae" in results
        assert isinstance(results["rmse"], float)
        assert isinstance(results["mae"], float)
    
    def test_compute_all_metrics_with_ci(self):
        """Test compute_all_metrics with bootstrap CI."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        results = compute_all_metrics(
            y_true, y_pred, ["rmse", "mae"],
            with_ci=True, n_bootstrap=100
        )
        
        assert isinstance(results["rmse"], BootstrapCI)
        assert isinstance(results["mae"], BootstrapCI)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
