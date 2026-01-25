# tests/unit/test_fast_embedder.py
"""Tests for fast embedding using ONNX Runtime."""

import pytest
import numpy as np
import time


class TestFastEmbedder:
    """Tests for FastEmbedder."""

    def test_embed_returns_vector(self):
        """Embedding returns a numpy array."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()
        result = embedder.embed("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)  # MiniLM dimension

    def test_embed_is_normalized(self):
        """Embedding is L2 normalized."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()
        result = embedder.embed("test text")

        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01

    def test_embed_is_fast(self):
        """Embedding completes in under 50ms."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()
        # Warm up
        embedder.embed("warmup")

        start = time.perf_counter()
        embedder.embed("test embedding speed")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Embedding took {elapsed_ms:.1f}ms"

    def test_embed_batch(self):
        """Batch embedding works."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()
        texts = ["hello", "world", "test"]
        results = embedder.embed_batch(texts)

        assert results.shape == (3, 384)

    def test_similar_texts_have_similar_embeddings(self):
        """Semantically similar texts have high cosine similarity."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()

        emb1 = embedder.embed("the cat sat on the mat")
        emb2 = embedder.embed("a cat was sitting on a rug")
        emb3 = embedder.embed("quantum physics equations")

        sim_similar = np.dot(emb1, emb2)
        sim_different = np.dot(emb1, emb3)

        assert sim_similar > sim_different

    def test_get_backend_info(self):
        """Can query which backend is being used."""
        from reconstructions.fast_embedder import FastEmbedder

        embedder = FastEmbedder()
        backend = embedder.get_backend()

        assert backend in ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
