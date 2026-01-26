"""
Unit tests for Experience and Context dataclasses.
"""

import pytest
import time
from reconstructions.encoding import Experience, Context


class TestExperience:
    """Test Experience dataclass."""
    
    def test_experience_defaults(self):
        """Test Experience with default values."""
        exp = Experience()
        
        assert exp.text is None
        assert exp.sensory == {}
        assert exp.emotional is None
        assert exp.motor is None
        assert exp.source == "external"
        assert exp.tags == []
        assert exp.timestamp > 0
    
    def test_experience_with_text(self):
        """Test Experience with text content."""
        exp = Experience(text="The sky is blue")
        
        assert exp.has_text is True
        assert exp.text == "The sky is blue"
    
    def test_experience_with_sensory(self):
        """Test Experience with sensory data."""
        exp = Experience(sensory={
            "visual": [0.1, 0.2, 0.3],
            "auditory": [0.4, 0.5]
        })
        
        assert exp.has_sensory is True
        assert "visual" in exp.sensory
        assert "auditory" in exp.sensory
    
    def test_experience_with_emotional(self):
        """Test Experience with emotional data."""
        exp = Experience(emotional={
            "valence": 0.7,
            "arousal": 0.3,
            "dominance": 0.5
        })
        
        assert exp.has_emotional is True
        assert exp.emotional["valence"] == 0.7
    
    def test_experience_with_motor(self):
        """Test Experience with motor data."""
        exp = Experience(motor={
            "action": "walking",
            "speed": 1.2
        })
        
        assert exp.has_motor is True
        assert exp.motor["action"] == "walking"
    
    def test_experience_multi_modal(self):
        """Test Experience with multiple modalities."""
        exp = Experience(
            text="I'm walking outside",
            sensory={"visual": [0.1, 0.2]},
            emotional={"valence": 0.8},
            motor={"action": "walking"},
            tags=["outdoor", "exercise"]
        )
        
        assert exp.has_text is True
        assert exp.has_sensory is True
        assert exp.has_emotional is True
        assert exp.has_motor is True
        assert len(exp.tags) == 2


class TestContext:
    """Test Context dataclass."""
    
    def test_context_defaults(self):
        """Test Context with default values."""
        ctx = Context()
        
        assert ctx.id.startswith("ctx_")
        assert ctx.sequence_counter == 0
        assert ctx.active_goals == []
        assert ctx.state == {}
        assert ctx.processing_depth == 0.5
        assert ctx.variance_mode == 0.3
        assert ctx.recent_fragments == []
        assert ctx.created_at > 0
    
    def test_context_with_goals(self):
        """Test Context with active goals."""
        ctx = Context(active_goals=["learn", "explore"])
        
        assert len(ctx.active_goals) == 2
        assert "learn" in ctx.active_goals
    
    def test_context_increment_sequence(self):
        """Test sequence counter increment."""
        ctx = Context()
        
        assert ctx.sequence_counter == 0
        ctx.increment_sequence()
        assert ctx.sequence_counter == 1
        ctx.increment_sequence()
        assert ctx.sequence_counter == 2
    
    def test_context_add_recent_fragment(self):
        """Test adding recent fragments."""
        ctx = Context()
        
        ctx.add_recent_fragment("frag1")
        assert len(ctx.recent_fragments) == 1
        assert ctx.recent_fragments[0] == "frag1"
        
        ctx.add_recent_fragment("frag2")
        assert len(ctx.recent_fragments) == 2
    
    def test_context_max_recent_fragments(self):
        """Test that recent fragments are limited."""
        ctx = Context()
        
        # Add 15 fragments with max_recent=10
        for i in range(15):
            ctx.add_recent_fragment(f"frag{i}", max_recent=10)
        
        # Should only keep last 10
        assert len(ctx.recent_fragments) == 10
        assert ctx.recent_fragments[0] == "frag5"  # First 5 dropped
        assert ctx.recent_fragments[-1] == "frag14"
    
    def test_context_serialization(self):
        """Test Context serialization/deserialization."""
        original = Context(
            sequence_counter=42,
            active_goals=["goal1", "goal2"],
            state={"mode": "test"},
            processing_depth=0.8,
            variance_mode=0.2,
            recent_fragments=["frag1", "frag2"]
        )
        
        data = original.to_dict()
        restored = Context.from_dict(data)
        
        assert restored.id == original.id
        assert restored.sequence_counter == 42
        assert restored.active_goals == ["goal1", "goal2"]
        assert restored.state == {"mode": "test"}
        assert restored.processing_depth == 0.8
        assert restored.variance_mode == 0.2
        assert restored.recent_fragments == ["frag1", "frag2"]
