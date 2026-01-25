# Reconstructions: Human-Like Memory System

## Quick Summary

This is a **software emulation of human memory principles** — not a database or RAG system. Memory is lossy, reconstructive, emotionally-weighted, and exists to guide behavior rather than record history.

**Status**: All 12 phases complete, 203 tests passing.

## Core Thesis

> Memory consists of time-structured, peak-weighted relational summaries formed across sensory, motor, and emotional domains.

Key principles:
- Memory is **lossy** and **reconstructive**, not a perfect recording
- It's **context-sensitive** — same query can produce different results based on current state
- It's **emotionally weighted** — survival relevance and prediction over accuracy
- It's **action-coupled** — exists to guide behavior, not record history
- It **reconstructs**, not retrieves — fills gaps probabilistically

## Architecture

```
Control Layer (Goals, Attention, Evaluation)
        ↓
Reconstruction Engine (Episodic, Semantic, Identity)
        ↓
Constraint Systems (Rehearsal, Social, Prediction Error, Embodied)
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `core.py` | Fragment, Strand, Query dataclasses |
| `store.py` | SQLite + in-memory vector storage |
| `encoder.py` | Experience → Fragment pipeline |
| `features.py` | Semantic embeddings (sentence-transformers), emotional/temporal features |
| `salience.py` | What makes memories "stick" (emotion 35%, novelty 30%, goals 25%, depth 10%) |
| `bindings.py` | Temporal and semantic links between fragments |
| `strength.py` | Power-law decay + rehearsal strengthening |
| `reconstruction.py` | Spreading activation, candidate selection, gap filling |
| `certainty.py` | Variance tracking — low variance = high subjective certainty |
| `identity.py` | Traits, beliefs, goals that constrain what gets remembered |
| `constraints.py` | Hard/soft constraints on reconstruction |
| `engine.py` | Main loop processing goals (QUERY > ENCODE > REFLECT > MAINTENANCE) |
| `cli.py` | Direct interface: `/store`, `/remember`, `/identity`, `/status` |
| `llm_interface.py` | Ollama integration for natural language |
| `experiments/` | Consciousness probing (self-reference, metacognition, identity continuity) |

## Data Flow

### Encoding (Experience → Fragment)
```
Experience (text, sensory, emotional, motor)
    ↓
extract_all_features() → content dict with embeddings
    ↓
calculate_salience() → importance score (0-1)
    ↓
create_bindings() → temporal + semantic links
    ↓
Fragment saved to store
```

### Retrieval (Query → Strand)
```
Query (semantic text, domains, time range)
    ↓
find_similar_semantic() → candidate fragments
    ↓
spreading_activation() → activation spreads through bindings
    ↓
select_candidates() → top fragments by activation × strength + noise
    ↓
assemble_strand() → coherent output with certainty score
```

## Design Principles

1. **Process-First**: Reconstruction process is the algorithm. LLM is optional interface.
2. **Own the Algorithms**: Salience, decay, binding, reconstruction are explicit formulas, not black-box.
3. **Reconstructive**: Memories assembled from fragments under constraints, not looked up.
4. **Variance as Foundational**: Output stability = subjective certainty (not truth).
5. **Emotion as Operating System**: Control signal modulating encoding, retrieval, priority.
6. **Context-Gated Contradiction**: Contradictions exist but don't co-activate.

## Current Speed Bottlenecks

1. **Embedding generation** — `sentence-transformers` (`all-MiniLM-L6-v2`), ~50-100ms/encoding on CPU
2. **Similarity search** — Brute-force O(n) cosine similarity
3. **LLM intent parsing** — Ollama calls take seconds (if using LLM interface)

## Potential Integration with Claude Code

Three approaches discussed:

| Option | Description |
|--------|-------------|
| **A. Claude as LLM** | Memory provides fragments directly to Claude Code context. No Ollama. |
| **B. MCP Server** | Wrap as MCP tools: `memory_store`, `memory_recall`, etc. |
| **C. Background daemon** | Async encoding/indexing, fast API for queries |

## Comparison with claude-mem

| Aspect | claude-mem | reconstructions |
|--------|-----------|-----------------|
| Model | Log + search (RAG) | Human memory model |
| Storage | Observations | Fragments with salience |
| Retrieval | Keyword/semantic | Spreading activation |
| Forgetting | None | Power-law decay |
| Certainty | N/A | Variance-based |
| Identity | N/A | Traits/beliefs constrain recall |

## File Structure

```
src/reconstructions/
├── core.py              # Data structures
├── store.py             # SQLite + vector storage
├── encoding.py          # Experience/Context definitions
├── encoder.py           # Encoding pipeline
├── features.py          # Feature extraction
├── salience.py          # Salience calculation
├── bindings.py          # Temporal/semantic binding
├── strength.py          # Decay and rehearsal
├── reconstruction.py    # Spreading activation
├── certainty.py         # Variance tracking
├── constraints.py       # Constraint system
├── identity.py          # Identity model
├── engine.py            # Main loop
├── cli.py               # CLI interface
├── llm_interface.py     # Ollama integration
├── main.py              # Entry point
└── experiments/         # Consciousness probing
    ├── probe.py
    └── runner.py

tests/                   # 203 tests
```

## Running

```bash
# CLI mode
python -m reconstructions.cli

# LLM chat (requires Ollama)
python -m reconstructions.llm_interface

# Tests
pytest tests/
```

## Open Questions

1. How to achieve "human-like speed" for Claude Code integration?
2. MCP server vs direct context injection?
3. Can this replace or complement claude-mem's approach?
