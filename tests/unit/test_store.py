"""
Unit tests for FragmentStore.
"""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from reconstructions.core import Fragment
from reconstructions.store import FragmentStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)


class TestFragmentStoreBasics:
    """Test basic store operations."""
    
    def test_save_and_get(self, temp_db):
        """Test saving and retrieving a fragment."""
        store = FragmentStore(temp_db)
        
        fragment = Fragment(
            content={"semantic": "test memory"},
            initial_salience=0.8
        )
        
        store.save(fragment)
        retrieved = store.get(fragment.id)
        
        assert retrieved is not None
        assert retrieved.id == fragment.id
        assert retrieved.content == fragment.content
        assert retrieved.initial_salience == 0.8
        
        store.close()
    
    def test_get_nonexistent(self, temp_db):
        """Test retrieving a fragment that doesn't exist."""
        store = FragmentStore(temp_db)
        
        result = store.get("nonexistent-id")
        
        assert result is None
        
        store.close()
    
    def test_delete(self, temp_db):
        """Test deleting a fragment."""
        store = FragmentStore(temp_db)
        
        fragment = Fragment(content={"semantic": "delete me"})
        store.save(fragment)
        
        # Verify it exists
        assert store.get(fragment.id) is not None
        
        # Delete it
        deleted = store.delete(fragment.id)
        assert deleted is True
        
        # Verify it's gone
        assert store.get(fragment.id) is None
        
        store.close()
    
    def test_delete_nonexistent(self, temp_db):
        """Test deleting a fragment that doesn't exist."""
        store = FragmentStore(temp_db)
        
        deleted = store.delete("nonexistent-id")
        assert deleted is False
        
        store.close()
    
    def test_is_empty(self, temp_db):
        """Test empty store detection."""
        store = FragmentStore(temp_db)
        
        assert store.is_empty() is True
        
        fragment = Fragment(content={"semantic": "not empty"})
        store.save(fragment)
        
        assert store.is_empty() is False
        
        store.close()


class TestFragmentStoreQueries:
    """Test query operations."""
    
    def test_find_by_time_range(self, temp_db):
        """Test finding fragments in a time range."""
        store = FragmentStore(temp_db)
        
        # Create fragments at different times
        f1 = Fragment(content={"semantic": "old"})
        f1.created_at = 1000.0
        
        f2 = Fragment(content={"semantic": "middle"})
        f2.created_at = 2000.0
        
        f3 = Fragment(content={"semantic": "new"})
        f3.created_at = 3000.0
        
        store.save(f1)
        store.save(f2)
        store.save(f3)
        
        # Query for middle time range
        results = store.find_by_time_range(1500.0, 2500.0)
        
        assert len(results) == 1
        assert results[0].id == f2.id
        
        # Query for all
        results = store.find_by_time_range(0.0, 5000.0)
        assert len(results) == 3
        
        store.close()
    
    def test_find_by_domain(self, temp_db):
        """Test finding fragments by domain."""
        store = FragmentStore(temp_db)
        
        f1 = Fragment(content={
            "semantic": "has semantic",
            "emotional": {"valence": 0.5}
        })
        
        f2 = Fragment(content={
            "semantic": "only semantic"
        })
        
        f3 = Fragment(content={
            "visual": [0.1, 0.2, 0.3]
        })
        
        store.save(f1)
        store.save(f2)
        store.save(f3)
        
        # Find fragments with "semantic" domain
        results = store.find_by_domain("semantic")
        assert len(results) == 2
        
        # Find fragments with "emotional" domain
        results = store.find_by_domain("emotional")
        assert len(results) == 1
        assert results[0].id == f1.id
        
        # Find fragments with "visual" domain
        results = store.find_by_domain("visual")
        assert len(results) == 1
        assert results[0].id == f3.id
        
        store.close()


class TestFragmentStoreVectorSearch:
    """Test vector similarity search."""
    
    def test_find_similar_semantic(self, temp_db):
        """Test finding similar fragments by embedding."""
        store = FragmentStore(temp_db)
        
        # Create fragments with embeddings
        f1 = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
        f2 = Fragment(content={"semantic": [0.9, 0.1, 0.0]})
        f3 = Fragment(content={"semantic": [0.0, 1.0, 0.0]})
        
        store.save(f1)
        store.save(f2)
        store.save(f3)
        
        # Query with embedding similar to f1
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.find_similar_semantic(query, top_k=2)
        
        assert len(results) == 2
        # f1 should be most similar
        assert results[0][0] == f1.id
        assert results[0][1] > 0.9  # High similarity
        # f2 should be second
        assert results[1][0] == f2.id
        
        store.close()
    
    def test_find_similar_empty_store(self, temp_db):
        """Test similarity search on empty store."""
        store = FragmentStore(temp_db)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.find_similar_semantic(query)
        
        assert results == []
        
        store.close()


class TestFragmentStoreAccess:
    """Test access tracking."""
    
    def test_record_access(self, temp_db):
        """Test recording access to a fragment."""
        store = FragmentStore(temp_db)
        
        fragment = Fragment(content={"semantic": "accessed"})
        store.save(fragment)
        
        assert len(fragment.access_log) == 0
        
        # Record access
        timestamp = time.time()
        store.record_access(fragment.id, timestamp)
        
        # Retrieve and check
        retrieved = store.get(fragment.id)
        assert len(retrieved.access_log) == 1
        assert retrieved.access_log[0] == timestamp
        
        store.close()


class TestFragmentStorePerformance:
    """Performance tests for FragmentStore."""

    def test_find_similar_semantic_is_fast(self, temp_db):
        """Similarity search completes in under 10ms."""
        store = FragmentStore(temp_db)

        # Add 1000 fragments with 384-dim embeddings
        for i in range(1000):
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            fragment = Fragment(
                content={"semantic": embedding.tolist()},
                initial_salience=0.5,
                source="test"
            )
            store.save(fragment)

        # Search timing
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        start = time.perf_counter()
        results = store.find_similar_semantic(query, top_k=50)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"Search took {elapsed_ms:.1f}ms"
        assert len(results) <= 50

        store.close()


class TestFragmentStoreIntegration:
    """Integration tests for FragmentStore."""

    def test_save_get_roundtrip(self, temp_db):
        """Test that save→get preserves all data."""
        store = FragmentStore(temp_db)
        
        original = Fragment(
            content={
                "semantic": "complex memory",
                "emotional": {"valence": 0.7, "arousal": 0.3},
                "temporal": {"position": 42}
            },
            bindings=["id1", "id2", "id3"],
            initial_salience=0.85,
            access_log=[1000.0, 2000.0],
            source="reflection",
            tags=["important", "test"]
        )
        
        store.save(original)
        retrieved = store.get(original.id)
        
        assert retrieved.id == original.id
        assert retrieved.created_at == original.created_at
        assert retrieved.content == original.content
        assert retrieved.bindings == original.bindings
        assert retrieved.initial_salience == original.initial_salience
        assert retrieved.access_log == original.access_log
        assert retrieved.source == original.source
        assert retrieved.tags == original.tags
        
        store.close()
    
    def test_context_manager(self, temp_db):
        """Test using store as context manager."""
        fragment = Fragment(content={"semantic": "context test"})
        
        with FragmentStore(temp_db) as store:
            store.save(fragment)
            retrieved = store.get(fragment.id)
            assert retrieved is not None
        
        # Reopen and verify persistence
        with FragmentStore(temp_db) as store:
            retrieved = store.get(fragment.id)
            assert retrieved is not None
            assert retrieved.content == fragment.content
