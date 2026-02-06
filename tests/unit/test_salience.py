"""
Unit tests for salience calculation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from reconstructions.store import FragmentStore
from reconstructions.core import Fragment
from reconstructions.salience import (
    calculate_encoding_salience,
    calculate_emotional_intensity,
    calculate_novelty,
    calculate_goal_relevance,
    calculate_salience_for_fragment,
    SalienceConfig,
    clamp
)


class TestClamp:
    """Test clamp utility."""
    
    def test_clamp_within_range(self):
        assert clamp(0.5) == 0.5
    
    def test_clamp_below_min(self):
        assert clamp(-0.5) == 0.0
    
    def test_clamp_above_max(self):
        assert clamp(1.5) == 1.0


class TestEncodingSalience:
    """Test main salience calculation formula."""
    
    def test_salience_default_weights(self):
        """Test salience with default weights."""
        salience = calculate_encoding_salience(
            emotional_intensity=0.8,
            novelty=0.6,
            goal_relevance=0.7,
            processing_depth=0.5
        )
        
        # Should be weighted combination
        expected = 0.35 * 0.8 + 0.30 * 0.6 + 0.25 * 0.7 + 0.10 * 0.5
        assert abs(salience - expected) < 0.001
    
    def test_salience_all_high(self):
        """Test salience when all components high."""
        salience = calculate_encoding_salience(
            emotional_intensity=1.0,
            novelty=1.0,
            goal_relevance=1.0,
            processing_depth=1.0
        )
        
        assert salience == pytest.approx(1.0)
    
    def test_salience_all_low(self):
        """Test salience when all components low."""
        salience = calculate_encoding_salience(
            emotional_intensity=0.0,
            novelty=0.0,
            goal_relevance=0.0,
            processing_depth=0.0
        )
        
        assert salience == 0.0
    
    def test_salience_custom_config(self):
        """Test salience with custom weights."""
        config = SalienceConfig()
        config.W_EMOTIONAL = 0.5
        config.W_NOVELTY = 0.5
        config.W_GOAL = 0.0
        config.W_DEPTH = 0.0
        
        salience = calculate_encoding_salience(
            emotional_intensity=0.8,
            novelty=0.2,
            goal_relevance=0.0,
            processing_depth=0.0,
            config=config
        )
        
        expected = 0.5 * 0.8 + 0.5 * 0.2
        assert abs(salience - expected) < 0.001


class TestEmotionalIntensity:
    """Test emotional intensity calculation."""
    
    def test_high_arousal(self):
        """High arousal = high intensity."""
        features = {"arousal": 0.9, "valence": 0.5}
        intensity = calculate_emotional_intensity(features)
        
        assert intensity > 0.8
    
    def test_extreme_valence(self):
        """Extreme valence adds to intensity."""
        # Very positive
        features = {"arousal": 0.3, "valence": 0.95}
        intensity1 = calculate_emotional_intensity(features)
        
        # Neutral
        features = {"arousal": 0.3, "valence": 0.5}
        intensity2 = calculate_emotional_intensity(features)
        
        assert intensity1 > intensity2
    
    def test_neutral_emotional(self):
        """Neutral emotion = moderate intensity."""
        features = {"arousal": 0.5, "valence": 0.5}
        intensity = calculate_emotional_intensity(features)
        
        assert 0.4 < intensity < 0.6


class TestNovelty:
    """Test novelty calculation."""
    
    def test_novelty_empty_store(self):
        """Empty store = everything is novel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            content = {"semantic": [0.1, 0.2, 0.3]}
            novelty = calculate_novelty(content, store)
            
            assert novelty == 1.0
            store.close()
    
    def test_novelty_similar_exists(self):
        """Similar fragment exists = low novelty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save a fragment with specific embedding
            embedding = [1.0, 0.0, 0.0]
            frag = Fragment(content={"semantic": embedding})
            store.save(frag)
            
            # Try to encode very similar content
            similar_content = {"semantic": [0.99, 0.01, 0.0]}
            novelty = calculate_novelty(similar_content, store)
            
            # Should have low novelty
            assert novelty < 0.5
            
            store.close()
    
    def test_novelty_different_content(self):
        """Different content = high novelty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save a fragment
            frag = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
            store.save(frag)
            
            # Try very different content
            different_content = {"semantic": [0.0, 1.0, 0.0]}
            novelty = calculate_novelty(different_content, store)
            
            # Should have high novelty
            assert novelty > 0.5
            
            store.close()
    
    def test_novelty_no_semantic(self):
        """No semantic content = novel (store is empty)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            content = {"visual": [0.1, 0.2]}
            novelty = calculate_novelty(content, store)
            
            # Empty store = everything novel, even without semantic
            assert novelty == 1.0
            
            store.close()


class TestGoalRelevance:
    """Test goal relevance calculation."""
    
    def test_no_goals(self):
        """No active goals = low relevance (headroom for goal-relevant fragments)."""
        content = {"semantic": "anything"}
        relevance = calculate_goal_relevance(content, [])

        assert relevance == 0.3
    
    def test_goal_match(self):
        """Matching goal = high relevance."""
        content = {"semantic": "I want to learn Python programming"}
        goals = ["learn", "programming"]
        
        relevance = calculate_goal_relevance(content, goals)
        
        # Should be boosted above 0.5
        assert relevance > 0.5
    
    def test_goal_no_match(self):
        """Non-matching goal = moderate relevance."""
        content = {"semantic": "The weather is nice today"}
        goals = ["learn", "programming"]

        relevance = calculate_goal_relevance(content, goals)

        assert relevance == 0.5

    def test_no_semantic_content(self):
        """No semantic content = low relevance."""
        content = {"visual": [0.1, 0.2]}
        goals = ["learn"]

        relevance = calculate_goal_relevance(content, goals)

        assert relevance == 0.3


class TestSalienceComplete:
    """Test complete salience calculation pipeline."""
    
    def test_high_salience_scenario(self):
        """Test scenario that should produce high salience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Novel content, high emotion, goal-relevant
            content = {"semantic": [0.5, 0.5, 0.5]}
            emotional = {"arousal": 0.9, "valence": 0.8}
            
            salience = calculate_salience_for_fragment(
                content=content,
                emotional_features=emotional,
                processing_depth=0.8,
                active_goals=["test"],
                store=store
            )
            
            # Should be high (novel + emotional + processed)
            assert salience > 0.6
            
            store.close()
    
    def test_low_salience_scenario(self):
        """Test scenario that should produce low salience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save existing similar fragment
            frag = Fragment(content={"semantic": [0.5, 0.5, 0.5]})
            store.save(frag)
            
            # Try to encode very similar, low emotion
            content = {"semantic": [0.51, 0.49, 0.5]}
            emotional = {"arousal": 0.2, "valence": 0.5}
            
            salience = calculate_salience_for_fragment(
                content=content,
                emotional_features=emotional,
                processing_depth=0.3,
                active_goals=[],
                store=store
            )
            
            # Should be lower (not novel, low emotion, shallow)
            assert salience < 0.5
            
            store.close()
