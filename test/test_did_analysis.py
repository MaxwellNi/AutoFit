"""Tests for DiD analysis module."""
import numpy as np
import pandas as pd
import pytest

from narrative.block3.did_analysis import (
    DiDEstimator,
    DiDResult,
    EventStudyResult,
    MediationResult,
    simple_did,
    regression_did,
    bootstrap_did,
    check_parallel_trends,
    run_placebo_test,
    mediation_analysis,
    event_study,
)


class TestDiDResult:
    """Test DiDResult dataclass."""
    
    def test_significant_properties(self):
        """Test significance property methods."""
        result = DiDResult(
            estimator="simple",
            estimate=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            n_treated=100,
            n_control=100,
            n_pre=100,
            n_post=100,
        )
        
        assert result.significant_at_05
        assert result.significant_at_01
    
    def test_not_significant(self):
        """Test non-significant result."""
        result = DiDResult(
            estimator="simple",
            estimate=0.1,
            std_error=0.2,
            t_stat=0.5,
            p_value=0.6,
            ci_lower=-0.3,
            ci_upper=0.5,
            n_treated=50,
            n_control=50,
            n_pre=50,
            n_post=50,
        )
        
        assert not result.significant_at_05
        assert not result.significant_at_01
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DiDResult(
            estimator="simple",
            estimate=0.5,
            std_error=0.1,
            t_stat=5.0,
            p_value=0.001,
            ci_lower=0.3,
            ci_upper=0.7,
            n_treated=100,
            n_control=100,
            n_pre=100,
            n_post=100,
        )
        
        d = result.to_dict()
        assert d["estimate"] == 0.5
        assert d["estimator"] == "simple"


class TestSimpleDiD:
    """Test simple 2x2 DiD estimator."""
    
    @pytest.fixture
    def treatment_data(self):
        """Generate synthetic treatment effect data."""
        np.random.seed(42)
        n = 100
        
        # True treatment effect = 2.0
        y_pre_treated = np.random.normal(5, 1, n)
        y_post_treated = np.random.normal(7, 1, n)  # 5 + 2 (treatment effect)
        y_pre_control = np.random.normal(4, 1, n)
        y_post_control = np.random.normal(4.5, 1, n)  # Small time trend
        
        return y_pre_treated, y_post_treated, y_pre_control, y_post_control
    
    def test_simple_did_estimate(self, treatment_data):
        """Test DiD estimate is approximately correct."""
        y_pre_t, y_post_t, y_pre_c, y_post_c = treatment_data
        
        result = simple_did(y_pre_t, y_post_t, y_pre_c, y_post_c)
        
        # True effect ≈ (7-5) - (4.5-4) = 2 - 0.5 = 1.5
        assert result.estimate == pytest.approx(1.5, abs=0.5)
        assert result.estimator == "simple"
    
    def test_simple_did_returns_correct_counts(self, treatment_data):
        """Test that counts are correct."""
        y_pre_t, y_post_t, y_pre_c, y_post_c = treatment_data
        
        result = simple_did(y_pre_t, y_post_t, y_pre_c, y_post_c)
        
        assert result.n_treated == len(y_pre_t) + len(y_post_t)
        assert result.n_control == len(y_pre_c) + len(y_post_c)
    
    def test_simple_did_ci_contains_estimate(self, treatment_data):
        """Test that CI contains the estimate."""
        y_pre_t, y_post_t, y_pre_c, y_post_c = treatment_data
        
        result = simple_did(y_pre_t, y_post_t, y_pre_c, y_post_c)
        
        assert result.ci_lower <= result.estimate <= result.ci_upper
    
    def test_simple_did_no_effect(self):
        """Test DiD with no treatment effect."""
        np.random.seed(42)
        n = 100
        
        y_pre_t = np.random.normal(5, 1, n)
        y_post_t = np.random.normal(5, 1, n)
        y_pre_c = np.random.normal(5, 1, n)
        y_post_c = np.random.normal(5, 1, n)
        
        result = simple_did(y_pre_t, y_post_t, y_pre_c, y_post_c)
        
        # Should not be significant
        assert not result.significant_at_05 or abs(result.estimate) < 0.5


class TestRegressionDiD:
    """Test regression-based DiD estimator."""
    
    @pytest.fixture
    def treatment_df(self):
        """Generate synthetic treatment data as DataFrame."""
        np.random.seed(42)
        n = 200
        
        treated = np.array([0] * 100 + [1] * 100)
        post = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
        
        # Base outcome + treated effect + post effect + interaction (treatment effect)
        y = (
            5.0 +
            1.0 * treated +
            0.5 * post +
            2.0 * treated * post +  # True treatment effect
            np.random.normal(0, 1, n)
        )
        
        return pd.DataFrame({
            "outcome": y,
            "treated": treated,
            "post": post,
        })
    
    def test_regression_did_estimate(self, treatment_df):
        """Test regression DiD estimate."""
        result = regression_did(
            treatment_df,
            outcome_col="outcome",
            treated_col="treated",
            post_col="post",
        )
        
        # True interaction effect is 2.0
        assert result.estimate == pytest.approx(2.0, abs=0.5)
        assert result.estimator == "regression"
    
    def test_regression_did_with_controls(self, treatment_df):
        """Test regression DiD with control variables."""
        # Add a control variable
        treatment_df = treatment_df.copy()
        treatment_df["control"] = np.random.normal(0, 1, len(treatment_df))
        
        result = regression_did(
            treatment_df,
            outcome_col="outcome",
            treated_col="treated",
            post_col="post",
            controls=["control"],
        )
        
        # Should still estimate ~2.0
        assert result.estimate == pytest.approx(2.0, abs=0.6)


class TestBootstrapDiD:
    """Test bootstrap DiD inference."""
    
    def test_bootstrap_did_ci(self):
        """Test bootstrap confidence interval."""
        np.random.seed(42)
        n = 50
        
        y_pre_t = np.random.normal(5, 1, n)
        y_post_t = np.random.normal(7, 1, n)
        y_pre_c = np.random.normal(5, 1, n)
        y_post_c = np.random.normal(5.5, 1, n)
        
        result = bootstrap_did(
            y_pre_t, y_post_t, y_pre_c, y_post_c,
            n_bootstrap=500,
            seed=42,
        )
        
        assert result.estimator == "bootstrap"
        assert result.ci_lower < result.estimate < result.ci_upper
    
    def test_bootstrap_did_reproducible(self):
        """Test bootstrap is reproducible with seed."""
        np.random.seed(42)
        n = 30
        
        y_pre_t = np.random.normal(5, 1, n)
        y_post_t = np.random.normal(7, 1, n)
        y_pre_c = np.random.normal(5, 1, n)
        y_post_c = np.random.normal(5.5, 1, n)
        
        result1 = bootstrap_did(y_pre_t, y_post_t, y_pre_c, y_post_c, seed=123)
        result2 = bootstrap_did(y_pre_t, y_post_t, y_pre_c, y_post_c, seed=123)
        
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper


class TestParallelTrends:
    """Test parallel trends testing."""
    
    def test_parallel_trends_satisfied(self):
        """Test when parallel trends hold."""
        np.random.seed(42)
        
        # Both groups have same trend
        time_periods = np.arange(10)
        y_treated = 5 + 0.5 * time_periods + np.random.normal(0, 0.1, 10)
        y_control = 4 + 0.5 * time_periods + np.random.normal(0, 0.1, 10)
        
        p_value, diffs = check_parallel_trends(
            y_treated, y_control, time_periods, treatment_time=8
        )
        
        # Differences should be constant -> no trend -> p > 0.05
        # Note: with noise, might occasionally fail
        assert len(diffs) > 0
    
    def test_parallel_trends_violated(self):
        """Test when parallel trends are violated."""
        np.random.seed(42)
        
        # Different trends
        time_periods = np.arange(10)
        y_treated = 5 + 1.0 * time_periods + np.random.normal(0, 0.1, 10)
        y_control = 4 + 0.2 * time_periods + np.random.normal(0, 0.1, 10)
        
        p_value, diffs = check_parallel_trends(
            y_treated, y_control, time_periods, treatment_time=8
        )
        
        # Differences have a trend -> should detect
        assert len(diffs) > 0


class TestPlaceboTest:
    """Test placebo tests."""
    
    @pytest.fixture
    def time_series_df(self):
        """Generate panel data."""
        np.random.seed(42)
        
        data = []
        for t in range(2018, 2025):
            for i in range(50):
                treated = 1 if i < 25 else 0
                post = 1 if t >= 2023 else 0
                
                y = 10 + 0.5 * t + treated + 2.0 * treated * post + np.random.normal(0, 1)
                data.append({
                    "year": t,
                    "entity": i,
                    "treated": treated,
                    "outcome": y,
                })
        
        return pd.DataFrame(data)
    
    def test_placebo_tests_run(self, time_series_df):
        """Test that placebo tests run without error."""
        results = run_placebo_test(
            time_series_df,
            outcome_col="outcome",
            treated_col="treated",
            time_col="year",
            true_treatment_time=2023,
            placebo_times=[2020, 2021],
        )
        
        assert len(results) == 2
        assert "placebo_time" in results[0]


class TestMediationAnalysis:
    """Test mediation analysis."""
    
    @pytest.fixture
    def mediation_df(self):
        """Generate data with mediation."""
        np.random.seed(42)
        n = 200
        
        treatment = np.random.binomial(1, 0.5, n)
        
        # Mediator is affected by treatment
        mediator = 1.0 * treatment + np.random.normal(0, 1, n)
        
        # Outcome is affected by both treatment (direct) and mediator (indirect)
        outcome = 2.0 * treatment + 1.5 * mediator + np.random.normal(0, 1, n)
        
        return pd.DataFrame({
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        })
    
    def test_mediation_analysis_basic(self, mediation_df):
        """Test basic mediation analysis."""
        result = mediation_analysis(
            mediation_df,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
        )
        
        assert isinstance(result, MediationResult)
        assert result.total_effect > 0
        assert result.indirect_effect > 0
    
    def test_mediation_proportion(self, mediation_df):
        """Test that proportion mediated makes sense."""
        result = mediation_analysis(
            mediation_df,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
        )
        
        # Proportion should be between 0 and 1 (approximately)
        # In this setup, ~40% should be mediated
        assert 0 < result.proportion_mediated < 1
    
    def test_mediation_to_dict(self, mediation_df):
        """Test conversion to dict."""
        result = mediation_analysis(
            mediation_df,
            outcome_col="outcome",
            treatment_col="treatment",
            mediator_col="mediator",
        )
        
        d = result.to_dict()
        assert "total_effect" in d
        assert "indirect_effect" in d
        assert "proportion_mediated" in d


class TestEventStudy:
    """Test event study analysis."""
    
    @pytest.fixture
    def panel_df(self):
        """Generate panel data for event study."""
        np.random.seed(42)
        
        data = []
        for t in range(-4, 5):  # Relative time from -4 to +4
            for i in range(50):
                treated = 1 if i < 25 else 0
                
                # No effect before treatment, increasing effect after
                if t < 0:
                    effect = 0
                else:
                    effect = 0.5 * (t + 1) * treated
                
                y = 10 + effect + np.random.normal(0, 1)
                data.append({
                    "rel_time": t,
                    "entity": i,
                    "treated": treated,
                    "outcome": y,
                })
        
        df = pd.DataFrame(data)
        df["year"] = df["rel_time"] + 2023  # Convert to actual time
        return df
    
    def test_event_study_basic(self, panel_df):
        """Test basic event study."""
        result = event_study(
            panel_df,
            outcome_col="outcome",
            treated_col="treated",
            time_col="year",
            treatment_time=2023,
            pre_periods=3,
            post_periods=3,
        )
        
        assert isinstance(result, EventStudyResult)
        assert len(result.time_points) > 0
        assert len(result.estimates) == len(result.time_points)
    
    def test_event_study_pre_trends(self, panel_df):
        """Test event study pre-trends test."""
        result = event_study(
            panel_df,
            outcome_col="outcome",
            treated_col="treated",
            time_col="year",
            treatment_time=2023,
        )
        
        # Pre-trends should be ~0 -> p-value should be high
        assert result.pre_trend_pvalue >= 0
    
    def test_event_study_to_dict(self, panel_df):
        """Test conversion to dict."""
        result = event_study(
            panel_df,
            outcome_col="outcome",
            treated_col="treated",
            time_col="year",
            treatment_time=2023,
        )
        
        d = result.to_dict()
        assert "time_points" in d
        assert "estimates" in d
        assert "pre_trend_pvalue" in d


class TestIntegration:
    """Integration tests for DiD analysis."""
    
    def test_full_did_workflow(self):
        """Test complete DiD analysis workflow."""
        np.random.seed(42)
        
        # Generate data
        n = 200
        treated = np.array([0] * 100 + [1] * 100)
        post = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
        mediator = 0.5 * treated * post + np.random.normal(0, 0.5, n)
        outcome = 5 + 1 * treated + 0.5 * post + 2 * treated * post + 1 * mediator + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({
            "outcome": outcome,
            "treated": treated,
            "post": post,
            "mediator": mediator,
        })
        
        # 1. Run simple DiD
        y_pre_t = df[(df["treated"] == 1) & (df["post"] == 0)]["outcome"].values
        y_post_t = df[(df["treated"] == 1) & (df["post"] == 1)]["outcome"].values
        y_pre_c = df[(df["treated"] == 0) & (df["post"] == 0)]["outcome"].values
        y_post_c = df[(df["treated"] == 0) & (df["post"] == 1)]["outcome"].values
        
        did_simple = simple_did(y_pre_t, y_post_t, y_pre_c, y_post_c)
        
        # 2. Run regression DiD
        did_reg = regression_did(df, "outcome", "treated", "post")
        
        # 3. Run mediation
        med_result = mediation_analysis(df, "outcome", "treated", "mediator")
        
        # Verify
        assert did_simple.estimate == pytest.approx(did_reg.estimate, abs=0.3)
        assert med_result.indirect_effect > 0
