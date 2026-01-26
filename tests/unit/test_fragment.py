"""
Unit tests for Fragment dataclass.
"""

import pytest
import time
from reconstructions.core import Fragment


class TestFragmentCreation:
    """Test Fragment creation and default values."""
    
    def test_fragment_defaults(self):
        """Test that Fragment has sensible defaults."""
        fragment = Fragment()
        
        assert fragment.id is not None
        assert len(fragment.id) > 0
        assert fragment.created_at > 0
        assert fragment.content == {}
        assert fragment.bindings == []
        assert fragment.initial_salience == 0.5
        assert fragment.access_log == []
        assert fragment.source == "experience"
        assert fragment.tags == []
    
    def test_fragment_with_content(self):
        """Test Fragment creation with specific content."""
        content = {
            "semantic": "test memory",
            "emotional": {"valence": 0.7}
        }
        
        fragment = Fragment(
            content=content,
            initial_salience=0.8,
            source="inference"
        )
        
        assert fragment.content == content
        assert fragment.initial_salience == 0.8
        assert fragment.source == "inference"
    
    def test_fragment_unique_ids(self):
        """Test that each Fragment gets a unique ID."""
        f1 = Fragment()
        f2 = Fragment()
        
        assert f1.id != f2.id
    
    def test_fragment_timestamps(self):
        """Test that created_at timestamps are reasonable."""
        before = time.time()
        fragment = Fragment()
        after = time.time()
        
        assert before <= fragment.created_at <= after


class TestFragmentSerialization:
    """Test Fragment serialization and deserialization."""
    
    def test_to_dict(self):
        """Test Fragment.to_dict() serialization."""
        fragment = Fragment(
            content={"semantic": "test"},
            initial_salience=0.7,
            bindings=["id1", "id2"],
            tags=["important"]
        )
        
        data = fragment.to_dict()
        
        assert data["id"] == fragment.id
        assert data["created_at"] == fragment.created_at
        assert data["content"] == {"semantic": "test"}
        assert data["initial_salience"] == 0.7
        assert data["bindings"] == ["id1", "id2"]
        assert data["tags"] == ["important"]
        assert data["source"] == "experience"
        assert data["access_log"] == []
    
    def test_from_dict(self):
        """Test Fragment.from_dict() deserialization."""
        data = {
            "id": "test-id-123",
            "created_at": 1704834567.123,
            "content": {"semantic": "test memory"},
            "bindings": ["binding1"],
            "initial_salience": 0.9,
            "access_log": [1704834567.123, 1704834600.0],
            "source": "reflection",
            "tags": ["tag1", "tag2"]
        }
        
        fragment = Fragment.from_dict(data)
        
        assert fragment.id == "test-id-123"
        assert fragment.created_at == 1704834567.123
        assert fragment.content == {"semantic": "test memory"}
        assert fragment.bindings == ["binding1"]
        assert fragment.initial_salience == 0.9
        assert fragment.access_log == [1704834567.123, 1704834600.0]
        assert fragment.source == "reflection"
        assert fragment.tags == ["tag1", "tag2"]
    
    def test_roundtrip(self):
        """Test that serialization roundtrip preserves data."""
        original = Fragment(
            content={
                "semantic": "original memory",
                "emotional": {"valence": 0.5, "arousal": 0.3}
            },
            initial_salience=0.75,
            bindings=["fragment-a", "fragment-b"],
            access_log=[1704834567.0, 1704834600.0],
            source="inference",
            tags=["test", "roundtrip"]
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = Fragment.from_dict(data)
        
        # Check all fields match
        assert restored.id == original.id
        assert restored.created_at == original.created_at
        assert restored.content == original.content
        assert restored.bindings == original.bindings
        assert restored.initial_salience == original.initial_salience
        assert restored.access_log == original.access_log
        assert restored.source == original.source
        assert restored.tags == original.tags
