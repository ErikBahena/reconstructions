"""
Unit tests for feature extraction.
"""

import pytest
import numpy as np
from src.reconstructions.encoding import Experience, Context
from src.reconstructions.features import (
    extract_semantic_features,
    extract_emotional_features,
    extract_temporal_features,
    extract_all_features
)


class TestSemanticExtraction:
    """Test semantic feature extraction."""
    
    def test_extract_semantic_basic(self):
        """Test basic semantic extraction."""
        text = "The sky is blue"
        embedding = extract_semantic_features(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D array
        assert embedding.shape[0] > 0  # Has dimensions
        assert embedding.dtype == np.float32
    
    def test_extract_semantic_empty(self):
        """Test semantic extraction with empty text."""
        embedding = extract_semantic_features("")
        assert embedding is None
        
        embedding = extract_semantic_features("   ")
        assert embedding is None
    
    def test_extract_semantic_different_texts(self):
        """Test that different texts produce different embeddings."""
        emb1 = extract_semantic_features("The sky is blue")
        emb2 = extract_semantic_features("I love pizza")
        
        # Should not be identical (very unlikely with real embeddings)
        assert not np.array_equal(emb1, emb2)
    
    def test_extract_semantic_deterministic(self):
        """Test that same text produces same embedding."""
        text = "Consistent text"
        emb1 = extract_semantic_features(text)
        emb2 = extract_semantic_features(text)
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(emb1, emb2)


class TestEmotionalExtraction:
    """Test emotional feature extraction."""
    
    def test_extract_emotional_basic(self):
        """Test basic emotional extraction."""
        emotional_data = {
            "valence": 0.7,
            "arousal": 0.3,
            "dominance": 0.6
        }
        
        features = extract_emotional_features(emotional_data)
        
        assert features["valence"] == 0.7
        assert features["arousal"] == 0.3
        assert features["dominance"] == 0.6
    
    def test_extract_emotional_partial(self):
        """Test emotional extraction with partial data."""
        emotional_data = {"valence": 0.8}
        
        features = extract_emotional_features(emotional_data)
        
        assert features["valence"] == 0.8
        assert features["arousal"] == 0.5  # Default
        assert features["dominance"] == 0.5  # Default
    
    def test_extract_emotional_none(self):
        """Test emotional extraction with no data."""
        features = extract_emotional_features(None)
        
        # Should return neutral values
        assert features["valence"] == 0.5
        assert features["arousal"] == 0.5
        assert features["dominance"] == 0.5
    
    def test_extract_emotional_empty(self):
        """Test emotional extraction with empty dict."""
        features = extract_emotional_features({})
        
        # Should return neutral values
        assert features["valence"] == 0.5
        assert features["arousal"] == 0.5
        assert features["dominance"] == 0.5


class TestTemporalExtraction:
    """Test temporal feature extraction."""
    
    def test_extract_temporal_basic(self):
        """Test basic temporal extraction."""
        ctx = Context(sequence_counter=42)
        
        features = extract_temporal_features(ctx)
        
        assert "absolute" in features
        assert features["absolute"] > 0  # Unix timestamp
        assert features["sequence_position"] == 42.0
        assert "context_id" in features
    
    def test_extract_temporal_sequence(self):
        """Test temporal features with different sequences."""
        ctx1 = Context(sequence_counter=1)
        ctx2 = Context(sequence_counter=2)
        
        f1 = extract_temporal_features(ctx1)
        f2 = extract_temporal_features(ctx2)
        
        assert f1["sequence_position"] == 1.0
        assert f2["sequence_position"] == 2.0


class TestAllFeatures:
    """Test combined feature extraction."""
    
    def test_extract_all_text_only(self):
        """Test feature extraction with text only."""
        exp = Experience(text="Hello world")
        ctx = Context()
        
        features = extract_all_features(exp, ctx)
        
        assert "semantic" in features
        assert "emotional" in features  # Always present (neutral if not specified)
        assert "temporal" in features  # Always present
    
    def test_extract_all_multi_modal(self):
        """Test feature extraction with multiple modalities."""
        exp = Experience(
            text="I see a tree",
            sensory={"visual": [0.1, 0.2, 0.3]},
            emotional={"valence": 0.8, "arousal": 0.4},
            motor={"action": "walking"}
        )
        ctx = Context(sequence_counter=10)
        
        features = extract_all_features(exp, ctx)
        
        assert "semantic" in features
        assert "visual" in features
        assert features["visual"] == [0.1, 0.2, 0.3]
        assert "emotional" in features
        assert features["emotional"]["valence"] == 0.8
        assert "motor" in features
        assert features["motor"]["action"] == "walking"
        assert "temporal" in features
        assert features["temporal"]["sequence_position"] == 10.0
    
    def test_extract_all_no_text(self):
        """Test feature extraction without text."""
        exp = Experience(
            sensory={"auditory": [0.5, 0.6]},
            emotional={"valence": 0.3}
        )
        ctx = Context()
        
        features = extract_all_features(exp, ctx)
        
        # Should not have semantic
        assert "semantic" not in features
        # Should have sensory
        assert "auditory" in features
        # Should have emotional
        assert "emotional" in features
        # Should have temporal
        assert "temporal" in features
