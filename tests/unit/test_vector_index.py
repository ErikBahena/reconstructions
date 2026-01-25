# tests/unit/test_vector_index.py
"""Tests for VectorIndex with USearch HNSW."""

import pytest
import numpy as np
import time
import tempfile
from pathlib import Path


class TestVectorIndex:
    """Tests for VectorIndex."""

    def test_add_and_search(self):
        """Add a vector, search finds it with high similarity."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()

        # Create a random normalized vector
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        index.add("test-id", vec)

        # Search should return the same vector with very high similarity
        results = index.search(vec, limit=1)

        assert len(results) == 1
        assert results[0][0] == "test-id"
        assert results[0][1] > 0.99  # Near-identical should be ~1.0

    def test_search_is_fast(self):
        """Search completes in <10ms at 10k vectors."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()

        # Add 10k random vectors
        np.random.seed(42)
        for i in range(10_000):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            index.add(f"id-{i}", vec)

        # Create query vector
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Warm up
        index.search(query, limit=10)

        # Measure search time
        start = time.perf_counter()
        results = index.search(query, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == 10
        assert elapsed_ms < 10, f"Search took {elapsed_ms:.1f}ms, expected <10ms"

    def test_persistence(self):
        """Index can be saved and loaded from disk."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"

            # Create and populate index
            index1 = VectorIndex()
            vec1 = np.random.randn(384).astype(np.float32)
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = np.random.randn(384).astype(np.float32)
            vec2 = vec2 / np.linalg.norm(vec2)

            index1.add("id-a", vec1)
            index1.add("id-b", vec2)

            # Save
            index1.save(path)

            # Load in new instance
            index2 = VectorIndex()
            index2.load(path)

            # Verify contents
            assert index2.count() == 2
            assert index2.contains("id-a")
            assert index2.contains("id-b")

            # Search should work
            results = index2.search(vec1, limit=1)
            assert results[0][0] == "id-a"
            assert results[0][1] > 0.99

    def test_remove(self):
        """Removed vectors are not returned in search."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()

        # Add two vectors
        vec1 = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        vec2 = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)

        index.add("keep", vec1)
        index.add("remove-me", vec2)

        assert index.count() == 2

        # Remove one
        index.remove("remove-me")

        assert index.count() == 1
        assert index.contains("keep")
        assert not index.contains("remove-me")

        # Search should not return removed
        results = index.search(vec2, limit=10)
        ids = [r[0] for r in results]
        assert "remove-me" not in ids

    def test_count(self):
        """Count tracks the number of vectors."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()
        assert index.count() == 0

        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        index.add("id-1", vec)
        assert index.count() == 1

        index.add("id-2", vec)
        assert index.count() == 2

        index.add("id-3", vec)
        assert index.count() == 3

    def test_update_existing_id(self):
        """Adding with existing ID updates the vector."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()

        # Add initial vector
        vec1 = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        index.add("test-id", vec1)

        # Update with different vector
        vec2 = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        index.add("test-id", vec2)

        # Count should still be 1
        assert index.count() == 1

        # Search should find the updated vector
        results = index.search(vec2, limit=1)
        assert results[0][0] == "test-id"
        assert results[0][1] > 0.99

    def test_search_empty_index(self):
        """Search on empty index returns empty list."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = index.search(query, limit=10)
        assert results == []

    def test_contains(self):
        """Contains correctly reports presence of IDs."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()
        vec = np.random.randn(384).astype(np.float32)

        assert not index.contains("missing")

        index.add("present", vec)
        assert index.contains("present")
        assert not index.contains("missing")
