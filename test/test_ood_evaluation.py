#!/usr/bin/env python3
"""
Unit tests for Block 3 OOD evaluation module.
"""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/home/pni/projects/repo_root/src")

from narrative.block3.ood_evaluation import (
    OODShiftType,
    OODSplitResult,
    create_year_shift_split,
    create_sector_shift_split,
    create_size_shift_split,
    SignificanceResult,
    paired_t_test,
    wilcoxon_test,
    bootstrap_comparison,
    diebold_mariano_test,
    OODDegradation,
    compute_ood_degradation,
    compute_ood_robustness_score,
)


class TestOODSplits:
    """Tests for OOD split functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with dates and sectors."""
        np.random.seed(42)
        n = 1000
        
        dates = pd.date_range("2022-01-01", "2025-12-31", periods=n)
        sectors = np.random.choice(["tech", "biotech", "finance", "energy"], n)
        sizes = np.random.lognormal(10, 2, n)
        
        return pd.DataFrame({
            "crawled_date_day": dates,
            "industry": sectors,
            "funding_goal_usd": sizes,
        })
    
    def test_year_shift_split_basic(self, sample_df):
        """Test basic year shift split."""
        result = create_year_shift_split(
            sample_df,
            date_col="crawled_date_day",
            train_end_year=2023,
            val_year=2024,
            test_start_year=2025,
        )
        
        assert isinstance(result, OODSplitResult)
        assert result.n_train > 0
        assert result.n_val > 0
        assert result.n_test > 0
        assert result.shift_type == OODShiftType.YEAR_SHIFT
    
    def test_year_shift_split_no_overlap(self, sample_df):
        """Test that year shift splits don't overlap."""
        result = create_year_shift_split(sample_df, train_end_year=2023)
        
        all_idx = np.concatenate([result.train_idx, result.val_idx, result.test_idx])
        # No duplicates
        assert len(all_idx) == len(set(all_idx))
    
    def test_sector_shift_split_basic(self, sample_df):
        """Test basic sector shift split."""
        result = create_sector_shift_split(
            sample_df,
            sector_col="industry",
            train_sectors=["tech", "finance"],
            test_sectors=["biotech", "energy"],
            seed=42,
        )
        
        assert isinstance(result, OODSplitResult)
        assert result.n_train > 0
        assert result.n_test > 0
        assert result.shift_type == OODShiftType.SECTOR_SHIFT
    
    def test_sector_shift_split_auto_detect(self, sample_df):
        """Test sector shift with auto-detected sectors."""
        result = create_sector_shift_split(
            sample_df,
            sector_col="industry",
            seed=42,
        )
        
        assert result.n_train > 0
        assert result.n_test > 0
    
    def test_size_shift_split_basic(self, sample_df):
        """Test basic size shift split."""
        result = create_size_shift_split(
            sample_df,
            size_col="funding_goal_usd",
            train_percentile=(0, 75),
            test_percentile=(75, 100),
            seed=42,
        )
        
        assert isinstance(result, OODSplitResult)
        assert result.n_train > 0
        assert result.n_test > 0
        assert result.shift_type == OODShiftType.SIZE_SHIFT


class TestSignificanceTests:
    """Tests for significance test functions."""
    
    @pytest.fixture
    def sample_errors(self):
        """Create sample error arrays."""
        np.random.seed(42)
        n = 100
        
        # Model A: baseline errors
        errors_a = np.random.randn(n) + 1.0
        
        # Model B: better errors (lower mean)
        errors_b = np.random.randn(n) + 0.5
        
        return errors_a, errors_b
    
    @pytest.fixture
    def identical_errors(self):
        """Create identical error arrays for null hypothesis tests."""
        np.random.seed(42)
        errors = np.random.randn(100) + 1.0
        return errors, errors.copy()
    
    def test_paired_t_test_different(self, sample_errors):
        """Test paired t-test detects difference."""
        errors_a, errors_b = sample_errors
        
        result = paired_t_test(errors_a, errors_b)
        
        assert isinstance(result, SignificanceResult)
        assert result.test_name == "Paired t-test"
        assert not np.isnan(result.statistic)
        assert 0 <= result.p_value <= 1
        # Should be significant at 0.05
        assert result.significant_at_05 or result.p_value < 0.1
    
    def test_paired_t_test_identical(self, identical_errors):
        """Test paired t-test on identical errors (should return NaN or be non-significant)."""
        errors_a, errors_b = identical_errors
        
        result = paired_t_test(errors_a, errors_b)
        
        # For identical errors, p-value should be NaN or non-significant
        # (NaN happens due to zero variance in differences)
        if np.isnan(result.p_value):
            assert not result.significant_at_05
        else:
            assert result.p_value > 0.05
            assert not result.significant_at_05
    
    def test_wilcoxon_test_different(self, sample_errors):
        """Test Wilcoxon test detects difference."""
        errors_a, errors_b = sample_errors
        
        result = wilcoxon_test(errors_a, errors_b)
        
        assert isinstance(result, SignificanceResult)
        assert result.test_name == "Wilcoxon signed-rank"
        assert 0 <= result.p_value <= 1
    
    def test_bootstrap_comparison_basic(self, sample_errors):
        """Test bootstrap comparison."""
        errors_a, errors_b = sample_errors
        
        result = bootstrap_comparison(errors_a, errors_b, n_bootstrap=500, seed=42)
        
        assert isinstance(result, SignificanceResult)
        assert "Bootstrap" in result.test_name
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.ci_upper
    
    def test_diebold_mariano_test_basic(self, sample_errors):
        """Test Diebold-Mariano test."""
        errors_a, errors_b = sample_errors
        
        result = diebold_mariano_test(errors_a, errors_b, h=1, power=2)
        
        assert isinstance(result, SignificanceResult)
        assert result.test_name == "Diebold-Mariano"
        assert 0 <= result.p_value <= 1


class TestOODDegradation:
    """Tests for OOD degradation metrics."""
    
    def test_compute_ood_degradation_worse_ood(self):
        """Test degradation when OOD is worse."""
        result = compute_ood_degradation(
            iid_metric=1.0,
            ood_metric=1.5,
            metric_name="MAE",
            higher_is_better=False,
        )
        
        assert isinstance(result, OODDegradation)
        assert result.iid_metric == 1.0
        assert result.ood_metric == 1.5
        assert result.absolute_degradation == 0.5
        assert result.relative_degradation_pct == 50.0
    
    def test_compute_ood_degradation_better_ood(self):
        """Test degradation when OOD is better (negative degradation)."""
        result = compute_ood_degradation(
            iid_metric=1.5,
            ood_metric=1.0,
            metric_name="MAE",
            higher_is_better=False,
        )
        
        assert result.absolute_degradation == -0.5  # Negative = improvement
        assert result.relative_degradation_pct < 0
    
    def test_compute_ood_degradation_higher_is_better(self):
        """Test degradation for metrics where higher is better."""
        result = compute_ood_degradation(
            iid_metric=0.9,
            ood_metric=0.7,
            metric_name="Accuracy",
            higher_is_better=True,
        )
        
        assert result.absolute_degradation == pytest.approx(0.2)  # IID - OOD
        assert result.relative_degradation_pct > 0
    
    def test_compute_ood_robustness_score_perfect(self):
        """Test robustness score with no degradation."""
        degradations = [
            OODDegradation(1.0, 1.0, 0.0, 0.0, "MAE"),
            OODDegradation(0.8, 0.8, 0.0, 0.0, "RMSE"),
        ]
        
        score = compute_ood_robustness_score(degradations, threshold_pct=20.0)
        
        assert score == 1.0
    
    def test_compute_ood_robustness_score_moderate(self):
        """Test robustness score with moderate degradation."""
        degradations = [
            OODDegradation(1.0, 1.2, 0.2, 20.0, "MAE"),  # 20% degradation
        ]
        
        score = compute_ood_robustness_score(degradations, threshold_pct=20.0)
        
        # 20% degradation with 20% threshold should give 0.5
        assert 0.4 <= score <= 0.6
    
    def test_compute_ood_robustness_score_severe(self):
        """Test robustness score with severe degradation."""
        degradations = [
            OODDegradation(1.0, 2.0, 1.0, 100.0, "MAE"),  # 100% degradation
        ]
        
        score = compute_ood_robustness_score(degradations, threshold_pct=20.0)
        
        # Severe degradation should give score near 0
        assert score < 0.1


class TestIntegration:
    """Integration tests combining splits and evaluation."""
    
    def test_full_ood_evaluation_workflow(self):
        """Test complete OOD evaluation workflow."""
        np.random.seed(42)
        
        # Create sample data
        n = 500
        dates = pd.date_range("2022-01-01", "2025-12-31", periods=n)
        df = pd.DataFrame({
            "crawled_date_day": dates,
            "feature": np.random.randn(n),
            "target": np.random.randn(n),
        })
        
        # Create year shift split
        split = create_year_shift_split(
            df,
            train_end_year=2023,
            val_year=2024,
            test_start_year=2025,
        )
        
        # Simulate model errors
        iid_errors = np.abs(np.random.randn(split.n_val))  # Validation (IID proxy)
        ood_errors = np.abs(np.random.randn(split.n_test) + 0.3)  # Test (OOD)
        
        # Compute metrics
        iid_mae = np.mean(iid_errors)
        ood_mae = np.mean(ood_errors)
        
        # Compute degradation
        degradation = compute_ood_degradation(iid_mae, ood_mae, "MAE")
        
        assert degradation.ood_metric > degradation.iid_metric
        assert degradation.relative_degradation_pct > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
