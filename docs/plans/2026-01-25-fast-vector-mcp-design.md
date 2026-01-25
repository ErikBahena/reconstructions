# Design: Fast Vector Search + MCP Integration

**Date**: 2026-01-25
**Status**: Approved
**Goal**: Enable Claude Code to use reconstructions as a human-like memory system with <50ms operations

## Overview

Transform the reconstructions memory system into an MCP server that Claude Code can call directly. Replace slow components (sentence-transformers, brute-force search) with fast alternatives (ONNX Runtime, USearch) while preserving the human-like memory behavior (reconstruction, decay, salience, identity constraints).

## Constraints

- **Speed**: Both store and recall must complete in <50ms
- **Scale**: Must handle TB-scale data (billions of fragments)
- **Deployment**: Fully local, no external services
- **Hardware**: Must work on any hardware (NVIDIA GPU, Apple Silicon, CPU-only)
- **No artificial delays**: Fast is good; human-like behavior, not human-like latency

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code                               │
│                     (MCP Client / LLM)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Tools
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (Python)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │memory_store │  │memory_recall│  │memory_identity/status   │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Reconstruction Engine                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Fast Embedder    │  │ USearch Index    │  │ SQLite Store  │ │
│  │ (ONNX Runtime)   │  │ (HNSW vectors)   │  │ (metadata)    │ │
│  │ <20ms            │  │ <5ms search      │  │               │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Existing: Salience, Decay, Bindings, Identity, Certainty │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Changes

| Component | Current | New |
|-----------|---------|-----|
| Embeddings | `sentence-transformers` (~100ms) | ONNX Runtime with auto-backend (<20ms) |
| Vector Search | Brute-force O(n) | USearch HNSW (<5ms at any scale) |
| Interface | Ollama LLM | MCP Server (Claude Code is the LLM) |
| Storage | SQLite + in-memory dict | SQLite (metadata) + USearch (vectors) |

## Data Flow

### Store Operation (~25-30ms total)

```
memory_store(text, emotional_valence, emotional_arousal, tags)
    │
    ▼
┌─────────────────────────┐
│ 1. Fast Embed (~15ms)   │  ONNX Runtime, hardware-adaptive
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Calculate Salience   │  Existing formula (emotion, novelty, goals)
│    (~1ms)               │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Create Bindings      │  Temporal + semantic links
│    (~2ms)               │
└───────────┬─────────────┘
            ▼
┌───────────┴───────────┐
│ 4. Parallel Write     │
│  USearch (~3ms)       │
│  SQLite  (~5ms)       │
└───────────────────────┘
```

### Recall Operation (~28-35ms total)

```
memory_recall(query, limit)
    │
    ▼
┌─────────────────────────┐
│ 1. Fast Embed (~15ms)   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. USearch Search       │  Top-50 candidates
│    (~3ms)               │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Spreading Activation │  Expand via bindings
│    (~5ms)               │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Apply Constraints    │  Identity, decay, strength
│    (~3ms)               │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. Assemble Strand      │  Coherent output + certainty
│    (~2ms)               │
└─────────────────────────┘
```

## New Components

### 1. FastEmbedder (`fast_embedder.py`)

Hardware-adaptive ONNX embedding with automatic backend selection.

```python
class FastEmbedder:
    """
    Fast text embedding using ONNX Runtime.
    Auto-selects best backend: CUDA > CoreML > CPU
    """

    def __init__(self, model_path: str = "models/all-MiniLM-L6-v2.onnx"):
        import onnxruntime as ort

        # Auto-select best available backend
        providers = []
        available = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = self._load_tokenizer()

    def embed(self, text: str) -> np.ndarray:
        """Generate 384-dim embedding in <20ms."""
        tokens = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        outputs = self.session.run(None, dict(tokens))
        # Mean pooling
        embedding = outputs[0].mean(axis=1)[0]
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embedding for efficiency."""
        # Implementation for batch processing
```

### 2. VectorIndex (`vector_index.py`)

USearch wrapper with persistence.

```python
class VectorIndex:
    """
    Fast vector similarity search using USearch.
    Handles TB-scale with <5ms queries.
    """

    def __init__(self, index_path: str, dim: int = 384):
        from usearch.index import Index

        self.index = Index(
            ndim=dim,
            metric='cos',  # Cosine similarity
            dtype='f32',
            connectivity=16,  # HNSW parameter
            expansion_add=128,
            expansion_search=64
        )
        self.index_path = Path(index_path)

        if self.index_path.exists():
            self.index.load(str(self.index_path))

    def add(self, id: str, vector: np.ndarray) -> None:
        """Add vector to index."""
        # Convert string ID to int for USearch
        int_id = self._id_to_int(id)
        self.index.add(int_id, vector)

    def search(self, vector: np.ndarray, limit: int = 50) -> list[tuple[str, float]]:
        """Find similar vectors. Returns (id, similarity) pairs."""
        results = self.index.search(vector, limit)
        return [(self._int_to_id(r.key), float(r.distance)) for r in results]

    def save(self) -> None:
        """Persist index to disk."""
        self.index.save(str(self.index_path))

    def _id_to_int(self, id: str) -> int:
        """Map string UUID to int for USearch."""
        # Use hash or maintain separate mapping

    def _int_to_id(self, int_id: int) -> str:
        """Map int back to string UUID."""
```

### 3. MCP Server (`mcp_server.py`)

Tools exposed to Claude Code.

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("reconstructions-memory")

@server.tool()
async def memory_store(
    text: str,
    emotional_valence: float = 0.5,
    emotional_arousal: float = 0.5,
    tags: list[str] | None = None,
    source: str = "claude_code"
) -> dict:
    """
    Store a new memory fragment.

    Args:
        text: Content to remember
        emotional_valence: Positive/negative (0-1, default 0.5 neutral)
        emotional_arousal: Intensity/activation (0-1, default 0.5 moderate)
        tags: Optional tags for categorization
        source: Origin of this memory

    Returns:
        {"success": bool, "fragment_id": str, "salience": float}
    """

@server.tool()
async def memory_recall(
    query: str,
    limit: int = 5,
    time_start: float | None = None,
    time_end: float | None = None,
    min_certainty: float = 0.0
) -> dict:
    """
    Recall memories matching a query.

    Args:
        query: What to remember (semantic search)
        limit: Max fragments to return
        time_start: Optional start timestamp filter
        time_end: Optional end timestamp filter
        min_certainty: Minimum certainty threshold

    Returns:
        {
            "fragments": [{"id", "content", "strength", "age"}],
            "certainty": float,  # 0-1, based on reconstruction variance
            "strand_id": str
        }
    """

@server.tool()
async def memory_identity() -> dict:
    """
    Get current identity model.

    Returns:
        {
            "traits": {"trait_name": strength},
            "beliefs": [{"content", "confidence", "evidence_count"}],
            "goals": [{"description", "priority", "progress"}]
        }
    """

@server.tool()
async def memory_status() -> dict:
    """
    Get memory system status.

    Returns:
        {
            "fragment_count": int,
            "index_size_mb": float,
            "oldest_memory": timestamp,
            "newest_memory": timestamp,
            "health": "ok" | "degraded"
        }
    """
```

## File Structure

```
src/reconstructions/
├── core.py                 # No change
├── store.py                # Refactor: remove in-memory embeddings
├── vector_index.py         # NEW: USearch wrapper
├── fast_embedder.py        # NEW: ONNX Runtime with auto-backend
├── encoding.py             # No change
├── encoder.py              # Update: use fast_embedder
├── features.py             # Update: use fast_embedder
├── salience.py             # No change
├── bindings.py             # No change
├── strength.py             # No change
├── reconstruction.py       # Update: use vector_index
├── certainty.py            # No change
├── constraints.py          # No change
├── identity.py             # No change
├── engine.py               # Minor updates for new components
├── mcp_server.py           # NEW: MCP tool definitions
├── cli.py                  # Keep for direct testing
└── llm_interface.py        # Deprecate (Claude Code replaces this)

models/
└── all-MiniLM-L6-v2.onnx   # Pre-converted ONNX model

data/
├── fragments.db            # SQLite metadata
└── vectors.usearch         # USearch index
```

## Dependencies

```
# requirements.txt additions
onnxruntime >= 1.17.0           # Base ONNX (CPU)
onnxruntime-silicon             # Apple Silicon (optional)
usearch >= 2.0                  # Vector index
mcp >= 0.1.0                    # MCP server framework
transformers                     # For tokenizer only

# Optional for GPU
# onnxruntime-gpu               # NVIDIA CUDA
```

## Implementation Order

1. **fast_embedder.py** — ONNX embedding with hardware detection
2. **vector_index.py** — USearch wrapper with persistence
3. **Update features.py** — Use fast embedder instead of sentence-transformers
4. **Update store.py** — Remove old embedding dict, integrate USearch
5. **Update reconstruction.py** — Use vector index for similarity search
6. **mcp_server.py** — Expose tools to Claude Code
7. **Tests** — Benchmark to verify <50ms for both operations
8. **ONNX model** — Convert and bundle all-MiniLM-L6-v2

## Testing Strategy

### Performance Tests
```python
def test_store_under_50ms():
    start = time.perf_counter()
    engine.store("Test memory content", emotional_arousal=0.7)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 50, f"Store took {elapsed}ms"

def test_recall_under_50ms():
    # Pre-populate with 100k fragments
    start = time.perf_counter()
    result = engine.recall("test query")
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 50, f"Recall took {elapsed}ms"
```

### Scale Tests
- 1k fragments: verify <50ms
- 100k fragments: verify <50ms
- 1M fragments: verify <100ms (stretch goal)

## Migration Path

1. New components are additive — no breaking changes initially
2. Old `features.py` embedding code becomes fallback
3. Feature flag to switch between old/new pipelines
4. Once validated, deprecate old code paths

## Open Questions (Resolved)

- ~~Vector database choice~~ → USearch (10-100x faster than FAISS)
- ~~Embedding speed~~ → ONNX Runtime with hardware auto-detection
- ~~Interface~~ → MCP Server (Claude Code is the LLM)
- ~~Artificial delays~~ → No, fast is good

## Success Criteria

- [ ] `memory_store` completes in <50ms
- [ ] `memory_recall` completes in <50ms
- [ ] Works on Apple Silicon, NVIDIA GPU, and CPU-only
- [ ] Scales to 1M+ fragments without degradation
- [ ] MCP tools callable from Claude Code
- [ ] Existing tests still pass
- [ ] Human-like behavior preserved (reconstruction, decay, certainty)
