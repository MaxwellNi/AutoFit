#!/usr/bin/env python3
"""
Unit tests for Block 3 preprocessing module.
"""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/home/pni/projects/repo_root/src")

from narrative.block3.preprocessing import (
    set_global_seed,
    get_seed_sequence,
    MissingPolicy,
    get_missing_policy,
    ScalingPolicy,
    FeatureScaler,
    PreprocessConfig,
    UnifiedPreprocessor,
)


class TestSeedControl:
    """Tests for deterministic seed control."""
    
    def test_set_global_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_global_seed(42)
        a1 = np.random.rand(10)
        
        set_global_seed(42)
        a2 = np.random.rand(10)
        
        np.testing.assert_array_equal(a1, a2)
    
    def test_get_seed_sequence_deterministic(self):
        """Test that seed sequence is deterministic."""
        seeds1 = get_seed_sequence(42, 5)
        seeds2 = get_seed_sequence(42, 5)
        
        assert seeds1 == seeds2
        assert len(seeds1) == 5
        assert all(isinstance(s, int) for s in seeds1)
    
    def test_get_seed_sequence_different_base(self):
        """Test that different base seeds produce different sequences."""
        seeds1 = get_seed_sequence(42, 5)
        seeds2 = get_seed_sequence(123, 5)
        
        assert seeds1 != seeds2


class TestMissingPolicy:
    """Tests for missing value handling."""
    
    def test_get_missing_policy_exact_match(self):
        """Test exact model name match."""
        assert get_missing_policy("lightgbm") == MissingPolicy.KEEP
        assert get_missing_policy("xgboost") == MissingPolicy.KEEP
        assert get_missing_policy("linear") == MissingPolicy.MEAN
    
    def test_get_missing_policy_prefix_match(self):
        """Test prefix-based model name match."""
        assert get_missing_policy("lightgbm_classifier") == MissingPolicy.KEEP
        assert get_missing_policy("transformer_encoder") == MissingPolicy.MASK
    
    def test_get_missing_policy_default(self):
        """Test default policy for unknown model."""
        assert get_missing_policy("unknown_model") == MissingPolicy.MEAN


class TestFeatureScaler:
    """Tests for feature scaler."""
    
    def test_standard_scaling(self):
        """Test standard (z-score) scaling."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        
        scaler = FeatureScaler(ScalingPolicy.STANDARD)
        scaler.fit(df, ["a", "b"])
        df_scaled = scaler.transform(df, ["a", "b"])
        
        # Check zero mean, unit variance
        np.testing.assert_almost_equal(df_scaled["a"].mean(), 0, decimal=5)
        np.testing.assert_almost_equal(df_scaled["a"].std(ddof=0), 1, decimal=5)
    
    def test_minmax_scaling(self):
        """Test min-max scaling."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        scaler = FeatureScaler(ScalingPolicy.MINMAX)
        scaler.fit(df, ["a"])
        df_scaled = scaler.transform(df, ["a"])
        
        # Check range [0, 1]
        assert df_scaled["a"].min() == 0.0
        assert df_scaled["a"].max() == 1.0
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        
        scaler = FeatureScaler(ScalingPolicy.STANDARD)
        scaler.fit(df, ["a"])
        df_scaled = scaler.transform(df, ["a"])
        df_inv = scaler.inverse_transform(df_scaled, ["a"])
        
        np.testing.assert_array_almost_equal(df["a"].values, df_inv["a"].values)


class TestUnifiedPreprocessor:
    """Tests for unified preprocessor."""
    
    def test_fit_transform_basic(self):
        """Test basic fit_transform."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, np.nan, 50.0],
        })
        
        config = PreprocessConfig(
            missing_policy=MissingPolicy.MEAN,
            scaling_policy=ScalingPolicy.STANDARD,
        )
        preprocessor = UnifiedPreprocessor(config)
        df_processed = preprocessor.fit_transform(df)
        
        # No NaN after processing
        assert df_processed["a"].isna().sum() == 0
        assert df_processed["b"].isna().sum() == 0
    
    def test_keep_nan_policy(self):
        """Test KEEP policy preserves NaN."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
        })
        
        config = PreprocessConfig(
            missing_policy=MissingPolicy.KEEP,
            scaling_policy=ScalingPolicy.NONE,
        )
        preprocessor = UnifiedPreprocessor(config)
        df_processed = preprocessor.fit_transform(df)
        
        # NaN should be preserved
        assert df_processed["a"].isna().sum() == 1
    
    def test_forward_fill_policy(self):
        """Test forward fill policy."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, np.nan, 4.0, 5.0],
        })
        
        config = PreprocessConfig(
            missing_policy=MissingPolicy.FORWARD_FILL,
            scaling_policy=ScalingPolicy.NONE,
        )
        preprocessor = UnifiedPreprocessor(config)
        df_processed = preprocessor.fit_transform(df)
        
        # NaN should be filled
        assert df_processed["a"].isna().sum() == 0
        assert df_processed["a"].iloc[1] == 1.0  # Forward filled
        assert df_processed["a"].iloc[2] == 1.0  # Forward filled
    
    def test_exclude_from_scaling(self):
        """Test that excluded columns are not scaled."""
        df = pd.DataFrame({
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        
        config = PreprocessConfig(
            missing_policy=MissingPolicy.KEEP,
            scaling_policy=ScalingPolicy.STANDARD,
            exclude_from_scaling=["target"],
        )
        preprocessor = UnifiedPreprocessor(config)
        df_processed = preprocessor.fit_transform(df)
        
        # Target should not be scaled
        np.testing.assert_array_equal(df["target"].values, df_processed["target"].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
