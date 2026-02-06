"""Tests for NBI computation module."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from narrative.nbi_nci.nbi_computation import (
    NBIDimension,
    NBI_DIMENSION_NAMES,
    NBI_KEYWORDS,
    NBIOutput,
    KeywordNBIExtractor,
    NBIComputationModel,
    NBIAggregator,
    compute_nbi_from_text,
    create_nbi_model,
)


class TestNBIDimensions:
    """Test NBI dimension definitions."""
    
    def test_nbi_has_15_dimensions(self):
        """Verify there are exactly 15 NBI dimensions."""
        assert len(NBIDimension) == 15
        assert len(NBI_DIMENSION_NAMES) == 15
        assert len(NBI_KEYWORDS) == 15
    
    def test_dimension_names_match_enum(self):
        """Verify dimension names match enum values."""
        for dim in NBIDimension:
            assert NBI_DIMENSION_NAMES[dim.value] == dim.name.lower()
    
    def test_all_dimensions_have_keywords(self):
        """Verify all dimensions have keyword lists."""
        for name in NBI_DIMENSION_NAMES:
            assert name in NBI_KEYWORDS
            assert len(NBI_KEYWORDS[name]) > 0


class TestKeywordExtractor:
    """Test keyword-based NBI extraction."""
    
    @pytest.fixture
    def extractor(self):
        return KeywordNBIExtractor()
    
    def test_extract_returns_correct_shape(self, extractor):
        """Test that extract returns 15-dimensional array."""
        text = "This is a test text with some opportunity and growth potential."
        scores = extractor.extract(text)
        assert scores.shape == (15,)
    
    def test_extract_detects_optimism(self, extractor):
        """Test that optimism keywords are detected."""
        text = "Great opportunity for growth and success, very promising future."
        scores = extractor.extract(text)
        # Optimism is index 0
        assert scores[0] > 0
    
    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        scores = extractor.extract("")
        assert scores.shape == (15,)
        assert np.all(scores == 0)
    
    def test_batch_extract(self, extractor):
        """Test batch extraction."""
        texts = [
            "Great opportunity for growth",
            "Risk of loss and downside",
            "Neutral text here",
        ]
        scores = extractor.batch_extract(texts)
        assert scores.shape == (3, 15)
    
    def test_scores_bounded(self, extractor):
        """Test that scores are between 0 and 1."""
        text = "opportunity growth potential success promising confident positive " * 10
        scores = extractor.extract(text)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestNBIComputationModel:
    """Test neural NBI computation model."""
    
    @pytest.fixture
    def model(self):
        return NBIComputationModel(
            emb_dim=64,
            n_bias_dims=15,
            hidden_dim=32,
            n_concepts=16,
            use_moe_router=False,
        )
    
    @pytest.fixture
    def model_with_moe(self):
        return NBIComputationModel(
            emb_dim=64,
            n_bias_dims=15,
            hidden_dim=32,
            n_concepts=16,
            use_moe_router=True,
            moe_top_k=4,
        )
    
    def test_forward_2d_input(self, model):
        """Test forward pass with 2D input."""
        x = torch.randn(4, 64)
        out = model(x)
        
        assert "dim_scores" in out
        assert "confidence" in out
        assert out["dim_scores"].shape == (4, 15)
        assert out["confidence"].shape == (4, 1)
    
    def test_forward_3d_input(self, model):
        """Test forward pass with 3D input (sequence)."""
        x = torch.randn(4, 10, 64)  # [batch, seq, emb]
        out = model(x)
        
        assert out["dim_scores"].shape == (4, 15)
        assert out["confidence"].shape == (4, 1)
    
    def test_forward_with_concepts(self, model):
        """Test forward pass returning concepts."""
        x = torch.randn(4, 64)
        out = model(x, return_concepts=True)
        
        assert "concepts" in out
        assert out["concepts"].shape == (4, 16)  # n_concepts=16
    
    def test_dim_scores_bounded(self, model):
        """Test that dimension scores are bounded to [0, 1]."""
        x = torch.randn(4, 64)
        out = model(x)
        
        assert torch.all(out["dim_scores"] >= 0)
        assert torch.all(out["dim_scores"] <= 1)
    
    def test_moe_routing(self, model_with_moe):
        """Test MoE routing output."""
        x = torch.randn(4, 64)
        out = model_with_moe(x)
        
        assert "gating_diagnostics" in out
        assert "moe_losses" in out
        assert "load_balance_loss" in out["moe_losses"]
        assert "sparsity_loss" in out["moe_losses"]
    
    def test_get_dominant_dimensions(self, model):
        """Test getting dominant dimensions."""
        dim_scores = torch.tensor([
            [0.9, 0.1, 0.3, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ])
        
        dominant = model.get_dominant_dimensions(dim_scores, top_k=3)
        
        assert len(dominant) == 2
        assert len(dominant[0]) == 3
        assert "optimism" in dominant[0]  # 0.9 is highest in first sample
        assert "anchoring" in dominant[1]  # 0.9 is highest in second sample


class TestNBIAggregator:
    """Test NBI score aggregation."""
    
    @pytest.fixture
    def sample_scores(self):
        return np.array([
            [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
    
    def test_mean_aggregation(self, sample_scores):
        """Test mean aggregation."""
        agg = NBIAggregator(aggregation="mean")
        result = agg.aggregate(sample_scores)
        
        assert result.shape == (15,)
        np.testing.assert_almost_equal(result[0], 0.6)  # (0.5+0.6+0.7)/3
    
    def test_max_aggregation(self, sample_scores):
        """Test max aggregation."""
        agg = NBIAggregator(aggregation="max")
        result = agg.aggregate(sample_scores)
        
        assert result[0] == 0.7  # max of first dimension
    
    def test_last_aggregation(self, sample_scores):
        """Test last aggregation."""
        agg = NBIAggregator(aggregation="last")
        result = agg.aggregate(sample_scores)
        
        np.testing.assert_array_equal(result, sample_scores[-1])
    
    def test_weighted_aggregation(self, sample_scores):
        """Test weighted aggregation."""
        agg = NBIAggregator(aggregation="weighted")
        weights = np.array([1, 2, 3])  # More weight on recent
        result = agg.aggregate(sample_scores, weights=weights)
        
        assert result.shape == (15,)
        # First dimension: (0.5*1 + 0.6*2 + 0.7*3) / 6 = 3.8/6 ≈ 0.633
        np.testing.assert_almost_equal(result[0], 3.8 / 6, decimal=5)
    
    def test_empty_aggregation(self, ):
        """Test aggregation with empty input."""
        agg = NBIAggregator(aggregation="mean")
        result = agg.aggregate(np.array([]).reshape(0, 15))
        
        assert result.shape == (15,)
        assert np.all(result == 0)


class TestNBIOutput:
    """Test NBIOutput dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = NBIOutput(
            dimension_scores=np.array([[0.5] * 15]),
            dominant_dimensions=["optimism", "anchoring", "confirmation"],
            confidence=0.8,
        )
        
        d = output.to_dict()
        
        assert "dimension_scores" in d
        assert "dominant_dimensions" in d
        assert "confidence" in d
        assert d["confidence"] == 0.8


class TestComputeNBIFromText:
    """Test convenience function."""
    
    def test_keyword_extraction(self):
        """Test keyword-based extraction."""
        texts = ["Great opportunity for growth and success"]
        
        output = compute_nbi_from_text(texts, use_keywords=True)
        
        assert isinstance(output, NBIOutput)
        assert output.dimension_scores.shape == (1, 15)
        assert len(output.dominant_dimensions) == 3
        assert output.confidence == 0.5  # Keyword-based has lower confidence
    
    def test_neural_model_extraction(self):
        """Test neural model extraction."""
        model = create_nbi_model(emb_dim=64, use_moe=False)
        embeddings = np.random.randn(2, 64).astype(np.float32)
        
        output = compute_nbi_from_text(
            texts=["dummy", "dummy"],  # Not used when embeddings provided
            model=model,
            embeddings=embeddings,
        )
        
        assert output.dimension_scores.shape == (2, 15)
        assert output.confidence > 0


class TestCreateNBIModel:
    """Test model factory function."""
    
    def test_create_without_moe(self):
        """Test creating model without MoE."""
        model = create_nbi_model(emb_dim=128, use_moe=False)
        
        assert isinstance(model, NBIComputationModel)
        assert model.moe is None
    
    def test_create_with_moe(self):
        """Test creating model with MoE."""
        model = create_nbi_model(emb_dim=128, use_moe=True)
        
        assert isinstance(model, NBIComputationModel)
        assert model.moe is not None


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full NBI computation pipeline."""
        # 1. Create model
        model = create_nbi_model(emb_dim=64, use_moe=True)
        
        # 2. Generate embeddings (simulated)
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)
        
        # 3. Compute NBI
        with torch.no_grad():
            out = model(x)
        
        # 4. Get dominant dimensions
        dominant = model.get_dominant_dimensions(out["dim_scores"], top_k=3)
        
        # 5. Aggregate across time (using numpy)
        scores_np = out["dim_scores"].numpy()
        agg = NBIAggregator(aggregation="mean")
        entity_profile = agg.aggregate(scores_np)
        
        # Verify
        assert entity_profile.shape == (15,)
        assert len(dominant) == batch_size
        assert "moe_losses" in out
