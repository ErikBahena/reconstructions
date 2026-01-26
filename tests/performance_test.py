"""
Performance benchmarks for the memory system.

Run with: pytest tests/performance_test.py -v -s
"""

import pytest
import time
import tempfile
import numpy as np
from pathlib import Path


class TestPerformanceBenchmarks:
    """Performance benchmarks for speed requirements."""

    @pytest.fixture
    def memory_server(self):
        """Create a memory server for testing."""
        from reconstructions.mcp_server import MemoryServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = MemoryServer(db_path=Path(tmpdir) / "bench.db")
            yield server
            server.close()

    def test_embedding_speed(self):
        """Embedding generation under 30ms."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()

        # Warm up
        embedder.embed("warmup text")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            embedder.embed("This is a test sentence for embedding speed measurement.")
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nEmbedding: avg={avg_time:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

        assert avg_time < 30, f"Embedding too slow: {avg_time:.1f}ms"

    def test_vector_search_speed(self):
        """Vector search under 5ms at 10k scale."""
        from reconstructions.vector_index import VectorIndex

        index = VectorIndex()

        # Add 10k vectors
        print("\nAdding 10k vectors...")
        for i in range(10000):
            v = np.random.randn(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            index.add(f"id{i}", v)

        # Benchmark search
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            results = index.search(query, limit=50)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"Search @10k: avg={avg_time:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")

        assert avg_time < 5, f"Search too slow: {avg_time:.2f}ms"

    def test_memory_store_speed(self, memory_server):
        """memory_store under 50ms."""
        # Warm up
        memory_server.memory_store(text="warmup")

        times = []
        for i in range(10):
            start = time.perf_counter()
            memory_server.memory_store(
                text=f"Test memory number {i} with some content",
                emotional_valence=0.6,
                emotional_arousal=0.4
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nmemory_store: avg={avg_time:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

        assert avg_time < 50, f"Store too slow: {avg_time:.1f}ms"

    def test_memory_recall_speed(self, memory_server):
        """memory_recall under 50ms."""
        # Populate with memories
        for i in range(100):
            memory_server.memory_store(text=f"Memory about topic {i} with details")

        # Benchmark recall
        times = []
        for _ in range(10):
            start = time.perf_counter()
            memory_server.memory_recall(query="topic details", limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nmemory_recall: avg={avg_time:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

        assert avg_time < 50, f"Recall too slow: {avg_time:.1f}ms"

    def test_roundtrip_speed(self, memory_server):
        """Full store + recall roundtrip under 100ms."""
        # Warm up
        memory_server.memory_store(text="warmup")
        memory_server.memory_recall(query="warmup")

        times = []
        for i in range(10):
            start = time.perf_counter()

            # Store
            memory_server.memory_store(
                text=f"Important fact number {i}: Claude Code uses reconstructions"
            )

            # Recall
            memory_server.memory_recall(query="important fact Claude")

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        print(f"\nRoundtrip: avg={avg_time:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

        assert avg_time < 100, f"Roundtrip too slow: {avg_time:.1f}ms"
