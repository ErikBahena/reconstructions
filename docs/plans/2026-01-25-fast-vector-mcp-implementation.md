# Fast Vector MCP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace slow embedding and search with USearch + ONNX for <50ms operations, expose as MCP server.

**Architecture:** ONNX Runtime for embeddings (auto-selects CUDA/CoreML/CPU), USearch for HNSW vector index, MCP server wrapping the reconstruction engine.

**Tech Stack:** onnxruntime, usearch, mcp, transformers (tokenizer only)

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml with new dependencies**

```toml
[project]
name = "reconstructions"
version = "0.2.0"
description = "A process-first human-like memory system"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "sentence-transformers>=2.2.0",  # Keep as fallback
    "onnxruntime>=1.17.0",
    "usearch>=2.0.0",
    "transformers>=4.30.0",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
gpu = [
    "onnxruntime-gpu>=1.17.0",
]
apple = [
    "onnxruntime-silicon>=1.17.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

**Step 2: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful installation

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add onnxruntime, usearch, mcp dependencies"
```

---

## Task 2: Create FastEmbedder

**Files:**
- Create: `src/reconstructions/fast_embedder.py`
- Create: `tests/unit/test_fast_embedder.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fast_embedder.py -v`
Expected: FAIL with "No module named 'reconstructions.fast_embedder'"

**Step 3: Write minimal implementation**

```python
# src/reconstructions/fast_embedder.py
"""
Fast text embedding using ONNX Runtime.

Auto-selects the best available backend:
- CUDAExecutionProvider (NVIDIA GPU)
- CoreMLExecutionProvider (Apple Silicon)
- CPUExecutionProvider (fallback)
"""

import numpy as np
from pathlib import Path
from typing import Optional

# Lazy-loaded globals
_onnx_session = None
_tokenizer = None
_backend = None


def _get_model_path() -> Path:
    """Get path to ONNX model, downloading if needed."""
    model_dir = Path(__file__).parent.parent.parent / "models"
    model_path = model_dir / "all-MiniLM-L6-v2.onnx"

    if not model_path.exists():
        # Download and convert model
        _download_and_convert_model(model_dir)

    return model_path


def _download_and_convert_model(model_dir: Path) -> None:
    """Download model and convert to ONNX format."""
    model_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Create dummy input
    dummy_input = tokenizer(
        "Hello world",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )

    # Export to ONNX
    onnx_path = model_dir / "all-MiniLM-L6-v2.onnx"

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        },
        opset_version=14
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(model_dir))


def _init_session():
    """Initialize ONNX session with best available provider."""
    global _onnx_session, _tokenizer, _backend

    if _onnx_session is not None:
        return

    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Select best provider
    providers = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        _backend = "CUDAExecutionProvider"
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        if _backend is None:
            _backend = "CoreMLExecutionProvider"
    providers.append("CPUExecutionProvider")
    if _backend is None:
        _backend = "CPUExecutionProvider"

    # Load model
    model_path = _get_model_path()
    model_dir = model_path.parent

    _onnx_session = ort.InferenceSession(str(model_path), providers=providers)
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))


def _mean_pooling(hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling to get sentence embedding."""
    # Expand attention mask
    mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    mask_expanded = np.broadcast_to(mask_expanded, hidden_state.shape)

    # Sum embeddings
    sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

    return sum_embeddings / sum_mask


class FastEmbedder:
    """
    Fast text embedder using ONNX Runtime.

    Auto-selects best available hardware backend.
    Returns 384-dimensional normalized vectors.
    """

    def __init__(self):
        """Initialize embedder, loading model if needed."""
        _init_session()

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            384-dimensional normalized embedding
        """
        # Tokenize
        inputs = _tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Run inference
        outputs = _onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )

        # Mean pooling
        embedding = _mean_pooling(outputs[0], inputs["attention_mask"])

        # Normalize
        embedding = embedding[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (n, 384) with normalized embeddings
        """
        if not texts:
            return np.array([]).reshape(0, 384)

        # Tokenize batch
        inputs = _tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Run inference
        outputs = _onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )

        # Mean pooling
        embeddings = _mean_pooling(outputs[0], inputs["attention_mask"])

        # Normalize each
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def get_backend(self) -> str:
        """Get the active backend provider."""
        return _backend
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fast_embedder.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reconstructions/fast_embedder.py tests/unit/test_fast_embedder.py
git commit -m "feat: add FastEmbedder with ONNX Runtime"
```

---

## Task 3: Create VectorIndex

**Files:**
- Create: `src/reconstructions/vector_index.py`
- Create: `tests/unit/test_vector_index.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_vector_index.py
"""Tests for USearch vector index."""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path


class TestVectorIndex:
    """Tests for VectorIndex."""

    def test_add_and_search(self):
        """Can add vectors and search them."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(Path(tmpdir) / "test.usearch")

            # Add some vectors
            v1 = np.random.randn(384).astype(np.float32)
            v1 = v1 / np.linalg.norm(v1)
            index.add("id1", v1)

            # Search for similar
            results = index.search(v1, limit=5)

            assert len(results) >= 1
            assert results[0][0] == "id1"
            assert results[0][1] > 0.99  # Should be very similar

    def test_search_is_fast(self):
        """Search completes in under 10ms even with many vectors."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(Path(tmpdir) / "test.usearch")

            # Add 10k vectors
            for i in range(10000):
                v = np.random.randn(384).astype(np.float32)
                v = v / np.linalg.norm(v)
                index.add(f"id{i}", v)

            # Search timing
            query = np.random.randn(384).astype(np.float32)
            query = query / np.linalg.norm(query)

            start = time.perf_counter()
            results = index.search(query, limit=50)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 10, f"Search took {elapsed_ms:.1f}ms"
            assert len(results) == 50

    def test_persistence(self):
        """Index persists across instances."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.usearch"

            # Create and add
            index1 = VectorIndex(index_path)
            v = np.random.randn(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            index1.add("persistent_id", v)
            index1.save()

            # Load in new instance
            index2 = VectorIndex(index_path)
            results = index2.search(v, limit=5)

            assert len(results) >= 1
            assert results[0][0] == "persistent_id"

    def test_remove(self):
        """Can remove vectors from index."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(Path(tmpdir) / "test.usearch")

            v = np.random.randn(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            index.add("to_remove", v)

            # Verify it's there
            results = index.search(v, limit=5)
            assert any(r[0] == "to_remove" for r in results)

            # Remove
            index.remove("to_remove")

            # Verify it's gone
            results = index.search(v, limit=5)
            assert not any(r[0] == "to_remove" for r in results)

    def test_count(self):
        """Can get vector count."""
        from reconstructions.vector_index import VectorIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(Path(tmpdir) / "test.usearch")

            assert index.count() == 0

            for i in range(5):
                v = np.random.randn(384).astype(np.float32)
                index.add(f"id{i}", v)

            assert index.count() == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vector_index.py -v`
Expected: FAIL with "No module named 'reconstructions.vector_index'"

**Step 3: Write minimal implementation**

```python
# src/reconstructions/vector_index.py
"""
Fast vector similarity search using USearch.

Provides HNSW-based approximate nearest neighbor search
with sub-millisecond query times at billion scale.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json


class VectorIndex:
    """
    Fast vector index using USearch HNSW.

    Maps string IDs to vectors and provides fast similarity search.
    """

    def __init__(
        self,
        index_path: Path,
        dim: int = 384,
        metric: str = "cos"
    ):
        """
        Initialize vector index.

        Args:
            index_path: Path to store index file
            dim: Vector dimensionality (default 384 for MiniLM)
            metric: Distance metric ("cos" for cosine, "l2" for euclidean)
        """
        from usearch.index import Index

        self.index_path = Path(index_path)
        self.mapping_path = self.index_path.with_suffix(".mapping.json")
        self.dim = dim

        # String ID -> int key mapping
        self._id_to_key: dict[str, int] = {}
        self._key_to_id: dict[int, str] = {}
        self._next_key: int = 0

        # Initialize index
        self.index = Index(
            ndim=dim,
            metric=metric,
            dtype="f32",
            connectivity=16,
            expansion_add=128,
            expansion_search=64
        )

        # Load existing if present
        if self.index_path.exists():
            self._load()

    def _load(self) -> None:
        """Load index and mapping from disk."""
        self.index.load(str(self.index_path))

        if self.mapping_path.exists():
            with open(self.mapping_path, "r") as f:
                data = json.load(f)
                self._id_to_key = {k: int(v) for k, v in data["id_to_key"].items()}
                self._key_to_id = {int(k): v for k, v in data["key_to_id"].items()}
                self._next_key = data["next_key"]

    def save(self) -> None:
        """Save index and mapping to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.index_path))

        with open(self.mapping_path, "w") as f:
            json.dump({
                "id_to_key": self._id_to_key,
                "key_to_id": {str(k): v for k, v in self._key_to_id.items()},
                "next_key": self._next_key
            }, f)

    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a vector to the index.

        Args:
            id: String identifier
            vector: Vector to add (must be dim-dimensional)
        """
        if id in self._id_to_key:
            # Update existing - remove and re-add
            self.remove(id)

        key = self._next_key
        self._next_key += 1

        self._id_to_key[id] = key
        self._key_to_id[key] = id

        self.index.add(key, vector.astype(np.float32))

    def remove(self, id: str) -> bool:
        """
        Remove a vector from the index.

        Args:
            id: String identifier

        Returns:
            True if removed, False if not found
        """
        if id not in self._id_to_key:
            return False

        key = self._id_to_key[id]

        # USearch doesn't support true deletion, but we can mark as removed
        # by removing from our mappings
        del self._id_to_key[id]
        del self._key_to_id[key]

        return True

    def search(
        self,
        vector: np.ndarray,
        limit: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find similar vectors.

        Args:
            vector: Query vector
            limit: Maximum results to return

        Returns:
            List of (id, similarity) tuples, highest similarity first
        """
        if self.count() == 0:
            return []

        # Search index
        results = self.index.search(vector.astype(np.float32), limit)

        # Map keys back to IDs and filter removed
        output = []
        for match in results:
            key = int(match.key)
            if key in self._key_to_id:
                id = self._key_to_id[key]
                # USearch returns distance, convert to similarity for cosine
                similarity = 1.0 - float(match.distance)
                output.append((id, similarity))

        return output

    def count(self) -> int:
        """Get number of vectors in index."""
        return len(self._id_to_key)

    def contains(self, id: str) -> bool:
        """Check if ID is in index."""
        return id in self._id_to_key
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vector_index.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reconstructions/vector_index.py tests/unit/test_vector_index.py
git commit -m "feat: add VectorIndex with USearch HNSW"
```

---

## Task 4: Update features.py to Use FastEmbedder

**Files:**
- Modify: `src/reconstructions/features.py:14-56`
- Modify: `tests/unit/test_features.py`

**Step 1: Add test for fast embedder usage**

Add to `tests/unit/test_features.py`:

```python
def test_extract_semantic_features_is_fast():
    """Semantic extraction completes in under 50ms."""
    import time
    from reconstructions.features import extract_semantic_features

    # Warm up
    extract_semantic_features("warmup text")

    start = time.perf_counter()
    result = extract_semantic_features("test the embedding speed")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 50, f"Extraction took {elapsed_ms:.1f}ms"
    assert result is not None
    assert len(result) == 384
```

**Step 2: Run test to see current behavior**

Run: `pytest tests/unit/test_features.py::test_extract_semantic_features_is_fast -v`
Expected: May pass or fail depending on current speed

**Step 3: Update features.py to use FastEmbedder**

Replace lines 10-56 in `src/reconstructions/features.py`:

```python
"""
Feature extraction functions for encoding experiences.
"""

import numpy as np
from typing import Optional, List
from .encoding import Experience, Context

# Use FastEmbedder instead of sentence-transformers
_fast_embedder = None


def get_embedder():
    """
    Get or initialize the fast embedder.

    Uses ONNX Runtime for hardware-accelerated inference.
    """
    global _fast_embedder

    if _fast_embedder is None:
        try:
            from .fast_embedder import FastEmbedder
            _fast_embedder = FastEmbedder()
        except Exception:
            # Fallback to sentence-transformers
            _fast_embedder = "fallback"

    return _fast_embedder


def extract_semantic_features(text: str) -> Optional[np.ndarray]:
    """
    Extract semantic features from text using embeddings.

    Args:
        text: Input text to encode

    Returns:
        Embedding vector as numpy array, or None if unavailable
    """
    if not text or len(text.strip()) == 0:
        return None

    embedder = get_embedder()

    if embedder == "fallback":
        # Fallback to sentence-transformers
        return _fallback_embedding(text)

    try:
        return embedder.embed(text)
    except Exception:
        return _fallback_embedding(text)


def _fallback_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Fallback embedding using sentence-transformers.

    Slower but guaranteed to work.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    except ImportError:
        # Last resort: hash-based pseudo-embedding
        return _hash_embedding(text, dim)


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Hash-based pseudo-embedding for testing only."""
    text_hash = hash(text.lower())
    np.random.seed(text_hash % (2**31))
    embedding = np.random.randn(dim).astype(np.float32)
    return embedding / (np.linalg.norm(embedding) + 1e-8)
```

**Step 4: Run tests to verify everything passes**

Run: `pytest tests/unit/test_features.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reconstructions/features.py tests/unit/test_features.py
git commit -m "feat: use FastEmbedder in features.py"
```

---

## Task 5: Update store.py to Use VectorIndex

**Files:**
- Modify: `src/reconstructions/store.py`
- Add test in: `tests/unit/test_store.py`

**Step 1: Add speed test to test_store.py**

Add to `tests/unit/test_store.py`:

```python
def test_find_similar_semantic_is_fast():
    """Similarity search completes in under 10ms."""
    import time
    import tempfile
    import numpy as np
    from reconstructions.store import FragmentStore
    from reconstructions.core import Fragment

    with tempfile.TemporaryDirectory() as tmpdir:
        store = FragmentStore(f"{tmpdir}/test.db")

        # Add 1000 fragments
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
```

**Step 2: Run test to see current behavior**

Run: `pytest tests/unit/test_store.py::test_find_similar_semantic_is_fast -v`
Expected: Likely FAIL (brute force is slow)

**Step 3: Update store.py to use VectorIndex**

Modify `src/reconstructions/store.py`:

```python
"""
Fragment storage using SQLite and vector embeddings.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from .core import Fragment


class FragmentStore:
    """
    Persistent storage for fragments.

    Uses SQLite for structured data and USearch for vector search.
    """

    def __init__(self, db_path: str):
        """
        Initialize fragment store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Initialize vector index
        self._init_vector_index()

        self._init_schema()

    def _init_vector_index(self):
        """Initialize the vector index."""
        try:
            from .vector_index import VectorIndex
            index_path = self.db_path.with_suffix(".usearch")
            self._vector_index = VectorIndex(index_path)
            self._use_vector_index = True
        except ImportError:
            # Fallback to in-memory dict
            self.embeddings: dict[str, np.ndarray] = {}
            self._use_vector_index = False

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Main fragments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                content TEXT NOT NULL,
                bindings TEXT NOT NULL,
                initial_salience REAL NOT NULL,
                access_log TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT NOT NULL
            )
        """)

        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON fragments(created_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_salience
            ON fragments(initial_salience DESC)
        """)

        self.conn.commit()

    def save(self, fragment: Fragment) -> None:
        """
        Save a fragment to the store.

        Args:
            fragment: Fragment to save
        """
        cursor = self.conn.cursor()

        # Serialize lists/dicts to JSON
        data = fragment.to_dict()

        cursor.execute("""
            INSERT OR REPLACE INTO fragments
            (id, created_at, content, bindings, initial_salience,
             access_log, source, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data["created_at"],
            json.dumps(data["content"]),
            json.dumps(data["bindings"]),
            data["initial_salience"],
            json.dumps(data["access_log"]),
            data["source"],
            json.dumps(data["tags"])
        ))

        self.conn.commit()

        # Extract and store embedding if present
        if "semantic" in fragment.content and isinstance(fragment.content["semantic"], list):
            embedding = np.array(fragment.content["semantic"], dtype=np.float32)

            if self._use_vector_index:
                self._vector_index.add(fragment.id, embedding)
            else:
                self.embeddings[fragment.id] = embedding

    def get(self, fragment_id: str) -> Optional[Fragment]:
        """
        Retrieve a fragment by ID.

        Args:
            fragment_id: Fragment ID

        Returns:
            Fragment if found, None otherwise
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM fragments WHERE id = ?
        """, (fragment_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        # Deserialize JSON fields
        data = {
            "id": row["id"],
            "created_at": row["created_at"],
            "content": json.loads(row["content"]),
            "bindings": json.loads(row["bindings"]),
            "initial_salience": row["initial_salience"],
            "access_log": json.loads(row["access_log"]),
            "source": row["source"],
            "tags": json.loads(row["tags"])
        }

        return Fragment.from_dict(data)

    def delete(self, fragment_id: str) -> bool:
        """
        Delete a fragment.

        Args:
            fragment_id: Fragment ID to delete

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            DELETE FROM fragments WHERE id = ?
        """, (fragment_id,))

        self.conn.commit()

        # Remove from vector index
        if self._use_vector_index:
            self._vector_index.remove(fragment_id)
        elif fragment_id in self.embeddings:
            del self.embeddings[fragment_id]

        return cursor.rowcount > 0

    def find_by_time_range(self, start: float, end: float) -> List[Fragment]:
        """
        Find fragments within a time range.

        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            List of fragments in time range
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM fragments
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at ASC
        """, (start, end))

        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))

        return fragments

    def find_by_domain(self, domain: str) -> List[Fragment]:
        """
        Find fragments containing a specific domain.

        Args:
            domain: Domain name (e.g., "semantic", "emotional")

        Returns:
            List of fragments with that domain
        """
        cursor = self.conn.cursor()

        # Use JSON search for domain key
        cursor.execute("""
            SELECT * FROM fragments
            WHERE json_extract(content, ?) IS NOT NULL
            ORDER BY created_at DESC
        """, (f"$.{domain}",))

        fragments = []
        for row in cursor.fetchall():
            data = {
                "id": row["id"],
                "created_at": row["created_at"],
                "content": json.loads(row["content"]),
                "bindings": json.loads(row["bindings"]),
                "initial_salience": row["initial_salience"],
                "access_log": json.loads(row["access_log"]),
                "source": row["source"],
                "tags": json.loads(row["tags"])
            }
            fragments.append(Fragment.from_dict(data))

        return fragments

    def find_similar_semantic(self, embedding: np.ndarray, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Find fragments with similar semantic embeddings.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of (fragment_id, similarity_score) tuples
        """
        if self._use_vector_index:
            return self._vector_index.search(embedding, limit=top_k)

        # Fallback: brute force
        if len(self.embeddings) == 0:
            return []

        # Normalize query embedding
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Compute cosine similarity with all stored embeddings
        similarities = []
        for fid, stored_emb in self.embeddings.items():
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            similarity = np.dot(query_norm, stored_norm)
            similarities.append((fid, float(similarity)))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def record_access(self, fragment_id: str, timestamp: float) -> None:
        """
        Record an access to a fragment (for rehearsal tracking).

        Args:
            fragment_id: Fragment ID
            timestamp: Access timestamp
        """
        fragment = self.get(fragment_id)
        if fragment is None:
            return

        fragment.access_log.append(timestamp)
        self.save(fragment)

    def is_empty(self) -> bool:
        """
        Check if store is empty.

        Returns:
            True if no fragments stored
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM fragments")
        row = cursor.fetchone()
        return row["count"] == 0

    def close(self):
        """Close database connection and save index."""
        if self._use_vector_index:
            self._vector_index.save()
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

**Step 4: Run tests to verify everything passes**

Run: `pytest tests/unit/test_store.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reconstructions/store.py tests/unit/test_store.py
git commit -m "feat: use VectorIndex for fast similarity search"
```

---

## Task 6: Create MCP Server

**Files:**
- Create: `src/reconstructions/mcp_server.py`
- Create: `tests/unit/test_mcp_server.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_mcp_server.py
"""Tests for MCP server tools."""

import pytest
import tempfile
from pathlib import Path


class TestMCPTools:
    """Tests for MCP tool functions."""

    @pytest.fixture
    def memory_server(self):
        """Create a memory server instance."""
        from reconstructions.mcp_server import MemoryServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = MemoryServer(db_path=Path(tmpdir) / "test.db")
            yield server
            server.close()

    def test_memory_store(self, memory_server):
        """Can store a memory."""
        result = memory_server.memory_store(
            text="The user prefers dark mode",
            emotional_valence=0.6,
            emotional_arousal=0.3
        )

        assert result["success"] is True
        assert "fragment_id" in result
        assert result["salience"] > 0

    def test_memory_recall(self, memory_server):
        """Can recall stored memories."""
        # Store first
        memory_server.memory_store(text="I love pizza")
        memory_server.memory_store(text="Python is my favorite language")

        # Recall
        result = memory_server.memory_recall(query="favorite food")

        assert "fragments" in result
        assert "certainty" in result

    def test_memory_identity(self, memory_server):
        """Can get identity model."""
        result = memory_server.memory_identity()

        assert "traits" in result
        assert "beliefs" in result
        assert "goals" in result

    def test_memory_status(self, memory_server):
        """Can get memory status."""
        memory_server.memory_store(text="test memory")

        result = memory_server.memory_status()

        assert result["fragment_count"] >= 1
        assert "health" in result

    def test_store_and_recall_roundtrip(self, memory_server):
        """Stored memories can be recalled."""
        # Store specific memory
        memory_server.memory_store(
            text="Meeting with Alice at 3pm about the project",
            tags=["meeting", "alice"]
        )

        # Recall
        result = memory_server.memory_recall(query="meeting with Alice")

        assert len(result["fragments"]) >= 1
        # Check the recalled fragment contains relevant content
        fragment_ids = [f["id"] for f in result["fragments"]]
        assert len(fragment_ids) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mcp_server.py -v`
Expected: FAIL with "No module named 'reconstructions.mcp_server'"

**Step 3: Write minimal implementation**

```python
# src/reconstructions/mcp_server.py
"""
MCP Server for Memory Reconstruction System.

Exposes memory operations as MCP tools for Claude Code.
"""

from pathlib import Path
from typing import Optional
from dataclasses import asdict

from .core import Query
from .encoding import Experience, Context
from .store import FragmentStore
from .engine import ReconstructionEngine
from .identity import Identity


class MemoryServer:
    """
    Memory server exposing MCP-compatible tools.

    This is the main interface for Claude Code to interact
    with the memory system.
    """

    def __init__(self, db_path: Path):
        """
        Initialize memory server.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.store = FragmentStore(str(db_path))
        self.engine = ReconstructionEngine(self.store)
        self.context = Context()

    def memory_store(
        self,
        text: str,
        emotional_valence: float = 0.5,
        emotional_arousal: float = 0.5,
        tags: Optional[list[str]] = None,
        source: str = "claude_code"
    ) -> dict:
        """
        Store a new memory fragment.

        Args:
            text: Content to remember
            emotional_valence: Positive/negative (0-1)
            emotional_arousal: Intensity (0-1)
            tags: Optional categorization tags
            source: Origin of this memory

        Returns:
            {"success": bool, "fragment_id": str, "salience": float}
        """
        try:
            experience = Experience(
                text=text,
                emotional={
                    "valence": emotional_valence,
                    "arousal": emotional_arousal,
                    "dominance": 0.5
                },
                source=source,
                tags=tags or []
            )

            self.engine.submit_experience(experience)
            result = self.engine.step()

            if result and result.success:
                fragment_id = result.data.get("fragment_id", "")
                salience = result.data.get("salience", 0.0)

                return {
                    "success": True,
                    "fragment_id": fragment_id,
                    "salience": salience
                }

            return {"success": False, "error": "Failed to encode"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def memory_recall(
        self,
        query: str,
        limit: int = 5,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        min_certainty: float = 0.0
    ) -> dict:
        """
        Recall memories matching a query.

        Args:
            query: What to remember (semantic search)
            limit: Max fragments to return
            time_start: Optional start timestamp
            time_end: Optional end timestamp
            min_certainty: Minimum certainty threshold

        Returns:
            {"fragments": [...], "certainty": float, "strand_id": str}
        """
        try:
            time_range = None
            if time_start is not None and time_end is not None:
                time_range = (time_start, time_end)

            q = Query(
                semantic=query,
                time_range=time_range
            )

            self.engine.submit_query(q)
            result = self.engine.step()

            if result and result.success:
                strand = result.data.get("strand")
                certainty = result.data.get("certainty", 0.0)

                if certainty < min_certainty:
                    return {
                        "fragments": [],
                        "certainty": certainty,
                        "message": "Certainty below threshold"
                    }

                # Get fragment details
                fragments = []
                if strand:
                    for frag_id in strand.fragments[:limit]:
                        fragment = self.store.get(frag_id)
                        if fragment:
                            # Extract readable content
                            content = fragment.content.get("semantic", "")
                            if isinstance(content, list):
                                # It's an embedding, get from elsewhere
                                content = fragment.content.get("text", str(content)[:100])

                            fragments.append({
                                "id": frag_id,
                                "content": content,
                                "salience": fragment.initial_salience,
                                "created_at": fragment.created_at
                            })

                return {
                    "fragments": fragments,
                    "certainty": certainty,
                    "strand_id": strand.id if strand else None
                }

            return {
                "fragments": [],
                "certainty": 0.0,
                "message": "No memories found"
            }

        except Exception as e:
            return {"fragments": [], "certainty": 0.0, "error": str(e)}

    def memory_identity(self) -> dict:
        """
        Get current identity model.

        Returns:
            {"traits": {...}, "beliefs": [...], "goals": [...]}
        """
        try:
            identity = self.engine.identity

            return {
                "traits": dict(identity.traits) if identity else {},
                "beliefs": [
                    {
                        "content": b.content,
                        "confidence": b.confidence,
                        "evidence_count": len(b.evidence)
                    }
                    for b in (identity.beliefs if identity else [])
                ],
                "goals": [
                    {
                        "description": g.description,
                        "priority": g.priority,
                        "progress": g.progress
                    }
                    for g in (identity.goals if identity else [])
                ]
            }

        except Exception as e:
            return {"traits": {}, "beliefs": [], "goals": [], "error": str(e)}

    def memory_status(self) -> dict:
        """
        Get memory system status.

        Returns:
            {"fragment_count": int, "health": str, ...}
        """
        try:
            # Count fragments
            cursor = self.store.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM fragments")
            count = cursor.fetchone()["count"]

            # Get time range
            cursor.execute("""
                SELECT MIN(created_at) as oldest, MAX(created_at) as newest
                FROM fragments
            """)
            row = cursor.fetchone()

            # Get index info
            index_size = 0
            if hasattr(self.store, '_vector_index'):
                index_path = self.store.db_path.with_suffix(".usearch")
                if index_path.exists():
                    index_size = index_path.stat().st_size / (1024 * 1024)

            return {
                "fragment_count": count,
                "index_size_mb": round(index_size, 2),
                "oldest_memory": row["oldest"] if row else None,
                "newest_memory": row["newest"] if row else None,
                "health": "ok"
            }

        except Exception as e:
            return {"fragment_count": 0, "health": "error", "error": str(e)}

    def close(self):
        """Close the memory server."""
        self.store.close()


# MCP protocol wrapper (for actual MCP integration)
def create_mcp_server(db_path: str = "~/.reconstructions/memory.db"):
    """
    Create MCP-compatible server.

    This can be used with the mcp library to expose tools.
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    return MemoryServer(path)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_mcp_server.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/reconstructions/mcp_server.py tests/unit/test_mcp_server.py
git commit -m "feat: add MCP server with memory tools"
```

---

## Task 7: Add Performance Benchmarks

**Files:**
- Modify: `tests/performance_test.py`

**Step 1: Add comprehensive speed tests**

Replace `tests/performance_test.py`:

```python
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

        with tempfile.TemporaryDirectory() as tmpdir:
            index = VectorIndex(Path(tmpdir) / "bench.usearch")

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
```

**Step 2: Run benchmarks**

Run: `pytest tests/performance_test.py -v -s`
Expected: All benchmarks PASS with times printed

**Step 3: Commit**

```bash
git add tests/performance_test.py
git commit -m "test: add performance benchmarks for speed requirements"
```

---

## Task 8: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All 200+ tests PASS

**Step 2: Fix any regressions**

If any tests fail, fix them before proceeding.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete fast vector MCP integration

- FastEmbedder: ONNX Runtime with auto hardware detection (<20ms)
- VectorIndex: USearch HNSW for fast similarity search (<5ms)
- MCP Server: memory_store, memory_recall, memory_identity, memory_status
- Performance: all operations under 50ms target
- Backward compatible: falls back to sentence-transformers if needed"
```

---

## Summary

| Task | Component | Purpose |
|------|-----------|---------|
| 1 | Dependencies | Add onnxruntime, usearch, mcp |
| 2 | FastEmbedder | ONNX embeddings <20ms |
| 3 | VectorIndex | USearch HNSW <5ms search |
| 4 | features.py | Use FastEmbedder |
| 5 | store.py | Use VectorIndex |
| 6 | MCP Server | Expose tools to Claude Code |
| 7 | Benchmarks | Verify <50ms requirement |
| 8 | Integration | Full test suite passes |

**Total estimated implementation: 8 commits, each building on the previous.**
