#!/usr/bin/env python3
"""
Unit tests for Block 3 model registry and models.
"""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/home/pni/projects/repo_root/src")

from narrative.block3.models.registry import (
    get_model,
    list_models,
    list_all_models,
    check_model_available,
    get_available_models,
    get_preset_models,
    get_baseline_models,
    MODEL_CATEGORIES,
    PRESET_CONFIGS,
)


class TestModelRegistry:
    """Tests for model registry."""
    
    def test_list_models_returns_dict(self):
        """Test list_models returns proper structure."""
        models = list_models()
        assert isinstance(models, dict)
        assert len(models) > 0
    
    def test_model_categories_present(self):
        """Test required categories exist."""
        models = list_models()
        expected_categories = [
            "statistical", "ml_tabular", "deep_classical",
            "transformer_sota", "foundation", "irregular_aware"
        ]
        for cat in expected_categories:
            assert cat in models, f"Missing category: {cat}"
    
    def test_list_all_models_returns_list(self):
        """Test list_all_models returns flat list."""
        all_models = list_all_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 20  # We have 33 models
    
    def test_check_model_available_known_models(self):
        """Test checking availability of known models."""
        # These should always be available (sklearn-based)
        assert check_model_available("Ridge") is True
        assert check_model_available("RandomForest") is True
        assert check_model_available("MeanPredictor") is True
    
    def test_check_model_available_unknown_model(self):
        """Test unknown model returns False."""
        assert check_model_available("NonExistentModel") is False
    
    def test_get_available_models(self):
        """Test get_available_models returns proper structure."""
        available = get_available_models()
        assert isinstance(available, dict)
        # ml_tabular should have available models
        assert len(available.get("ml_tabular", [])) > 0


class TestModelPresets:
    """Tests for model presets."""
    
    def test_preset_configs_defined(self):
        """Test preset configs are defined."""
        assert "smoke_test" in PRESET_CONFIGS
        assert "quick" in PRESET_CONFIGS
        assert "standard" in PRESET_CONFIGS
        assert "comprehensive" in PRESET_CONFIGS
    
    def test_get_preset_models_smoke_test(self):
        """Test smoke_test preset returns models."""
        models = get_preset_models("smoke_test")
        assert len(models) > 0
        assert len(models) <= 5  # Smoke test should be small
    
    def test_get_preset_models_quick(self):
        """Test quick preset returns models."""
        models = get_preset_models("quick")
        assert len(models) > 0
    
    def test_get_preset_models_invalid(self):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError):
            get_preset_models("invalid_preset")
    
    def test_get_baseline_models(self):
        """Test baseline models are returned."""
        baselines = get_baseline_models()
        assert len(baselines) >= 2
        # Should include MeanPredictor
        names = [m.config.name for m in baselines]
        assert "MeanPredictor" in names


class TestModelInstantiation:
    """Tests for instantiating models."""
    
    def test_get_model_ridge(self):
        """Test getting Ridge model."""
        model = get_model("Ridge")
        assert model is not None
        assert model.config.name == "Ridge"
    
    def test_get_model_lightgbm(self):
        """Test getting LightGBM model."""
        model = get_model("LightGBM", n_estimators=10)
        assert model is not None
        assert model.config.name == "LightGBM"
    
    def test_get_model_random_forest(self):
        """Test getting RandomForest model."""
        model = get_model("RandomForest", n_estimators=10)
        assert model is not None
        assert model.config.name == "RandomForest"
    
    def test_get_model_invalid(self):
        """Test invalid model raises error."""
        with pytest.raises(ValueError):
            get_model("NonExistentModel")


class TestModelFitPredict:
    """Tests for model fit/predict interface."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples), name="target")
        
        return X, y
    
    def test_ridge_fit_predict(self, sample_data):
        """Test Ridge model fit/predict."""
        X, y = sample_data
        
        model = get_model("Ridge")
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
    
    def test_lightgbm_fit_predict(self, sample_data):
        """Test LightGBM model fit/predict."""
        X, y = sample_data
        
        model = get_model("LightGBM", n_estimators=10, num_leaves=5)
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
    
    def test_random_forest_fit_predict(self, sample_data):
        """Test RandomForest model fit/predict."""
        X, y = sample_data
        
        model = get_model("RandomForest", n_estimators=10)
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()
    
    def test_mean_predictor_fit_predict(self, sample_data):
        """Test MeanPredictor baseline."""
        X, y = sample_data
        
        model = get_model("MeanPredictor")
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == len(X)
        # MeanPredictor should return mean of training data
        np.testing.assert_almost_equal(preds[0], y.mean(), decimal=5)
    
    def test_seasonal_naive_fit_predict(self, sample_data):
        """Test SeasonalNaive baseline."""
        X, y = sample_data
        
        model = get_model("SeasonalNaive", season_length=7)
        model.fit(X, y)
        
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestModelCategories:
    """Tests for MODEL_CATEGORIES structure."""
    
    def test_all_categories_have_models(self):
        """Test each category has at least one model."""
        for cat, models in MODEL_CATEGORIES.items():
            assert len(models) > 0, f"Category {cat} is empty"
    
    def test_model_count_reasonable(self):
        """Test we have a reasonable number of models."""
        total = sum(len(v) for v in MODEL_CATEGORIES.values())
        assert total >= 20, f"Expected at least 20 models, got {total}"
    
    def test_ml_tabular_has_required_models(self):
        """Test ml_tabular has essential models."""
        ml_models = MODEL_CATEGORIES["ml_tabular"]
        required = ["Ridge", "RandomForest", "LightGBM", "XGBoost"]
        for model in required:
            assert model in ml_models, f"Missing required model: {model}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
