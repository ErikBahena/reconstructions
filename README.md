# Reconstructions

A cognitive science-inspired memory system that models how human memory actually works: lossy, reconstructive, and context-sensitive rather than perfect storage.

## Core Idea

Memory is **reconstructed, not retrieved**. Each recall assembles fragments weighted by salience, decay, and binding strength — producing slightly different results each time, just like human memory.

| Property | Description |
|----------|-------------|
| **Lossy** | Deliberately discards most information |
| **Reconstructive** | Rebuilds memories rather than retrieving them |
| **Context-sensitive** | Same cue can produce different memories based on current state |
| **Emotionally weighted** | Prioritizes what matters, not what's accurate |
| **Action-coupled** | Exists to guide behavior, not to record history |

## Quick Start

```bash
# Install
git clone <repo-url> reconstructions
cd reconstructions
pip install -e ".[dev]"

# Run tests
PYTHONPATH=src pytest tests/unit/ -q

# Use as library
python -c "
from reconstructions import FragmentStore, ReconstructionEngine
from reconstructions.encoding import Experience
from reconstructions.core import Query

store = FragmentStore('memory.db')
engine = ReconstructionEngine(store)

# Store a memory
engine.submit_experience(Experience(text='Something happened'))
engine.step()

# Recall
engine.submit_query(Query(semantic='what happened'))
result = engine.step()
"
```

## How It Works

```
Experience --> Encoding Pipeline --> Fragment (SQLite + VectorIndex)
                                         |
                                         v
Query --> Reconstruction Engine --> Strand (assembled memory)
          (spread activation,            |
           select candidates,      Certainty Tracking
           fill gaps,                    |
           assemble)              Identity State
               ^
               |
    Consolidation Scheduler
    (autonomous reconstruction,
     pattern discovery,
     binding strengthening)
```

**Key algorithms:**

- **Salience**: `W_emotional * intensity + W_novelty * novelty + W_goal * relevance + W_depth * depth`
- **Decay**: Power law `strength = salience * (t+1)^(-0.5) + rehearsal_bonus * log(access_count)`
- **Certainty**: `1.0 - variance` where variance = Jaccard distance between repeated reconstructions

Fragments never fully disappear — they decay to a minimum strength (0.01), preserving the semantic network while naturally falling out of reconstructions. This mirrors biological forgetting.

## Autonomous Consolidation

Like human sleep consolidation, the system spontaneously rehearses and organizes memories:

1. **Rehearses** recent salient fragments to strengthen retrieval paths
2. **Discovers patterns** via semantic similarity and creates new bindings
3. **Strengthens bindings** when fragments are frequently co-activated
4. **Adapts frequency** based on activity (10s during high activity, 300s when idle)

Consolidation runs opportunistically during Claude Code hook processing and at session end.

## Self-Improving

The system learns optimal parameters automatically:

- **Weight learning**: Tunes salience weights based on retrieval success/failure, converging after ~100 queries
- **Pattern detection**: Discovers temporal routines, workflow sequences, and project clusters across sessions
- **Identity-aware encoding**: Active goals boost salience of relevant memories (+0.2 * intensity)

All features run in the background with zero configuration.

## Claude Code Integration

Reconstructions integrates with Claude Code via hooks that capture your workflow in real-time.

### Setup

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Make the hook wrapper executable:
   ```bash
   chmod +x run_hook.sh
   ```

3. Add hooks to `~/.claude/settings.json`:
   ```json
   {
     "hooks": {
       "SessionStart": [{"hooks": [{"type": "command", "command": "/path/to/reconstructions/run_hook.sh session_start"}]}],
       "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "/path/to/reconstructions/run_hook.sh user_prompt_submit"}]}],
       "PostToolUse": [{"hooks": [{"type": "command", "command": "/path/to/reconstructions/run_hook.sh post_tool_use"}]}],
       "Stop": [{"hooks": [{"type": "command", "command": "/path/to/reconstructions/run_hook.sh stop"}]}],
       "SessionEnd": [{"hooks": [{"type": "command", "command": "/path/to/reconstructions/run_hook.sh session_end"}]}]
     }
   }
   ```
   Replace `/path/to/reconstructions` with the actual install path.

4. Test:
   ```bash
   echo '{"session_id": "test"}' | ./run_hook.sh session_start
   ```

### Optional: Consolidation Daemon (macOS)

For continuous background consolidation outside Claude Code sessions:

```bash
# Edit com.reconstructions.consolidation.plist with your paths, then:
cp com.reconstructions.consolidation.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.reconstructions.consolidation.plist
```

## MCP Server

Exposes memory tools via Model Context Protocol:

- `memory_store(text, emotional_valence, emotional_arousal)` — Encode new experience
- `memory_recall(query, min_salience)` — Query and reconstruct
- `memory_identity()` — Current identity state
- `memory_status()` — System health metrics

## Project Structure

```
reconstructions/
├── CLAUDE.md                    # Full developer reference
├── README.md                    # This file
├── spec.md                      # Formal process specification
├── pyproject.toml
├── run_hook.sh                  # Hook wrapper script
├── src/reconstructions/
│   ├── core.py                  # Fragment, Strand, Query
│   ├── encoding.py              # Experience input structures
│   ├── encoder.py               # Encoding pipeline
│   ├── features.py              # Feature extraction
│   ├── salience.py              # Salience calculation
│   ├── strength.py              # Decay and rehearsal
│   ├── bindings.py              # Fragment linking
│   ├── store.py                 # SQLite persistence
│   ├── vector_index.py          # USearch HNSW index
│   ├── reconstruction.py        # Retrieval engine
│   ├── consolidation.py         # Autonomous consolidation
│   ├── health.py                # Health monitoring
│   ├── metrics.py               # Quality tracking
│   ├── learning.py              # Weight learning
│   ├── patterns.py              # Cross-session patterns
│   ├── identity.py              # Identity state
│   ├── certainty.py             # Variance tracking
│   ├── engine.py                # Main engine loop
│   ├── mcp_server.py            # MCP server
│   ├── fast_embedder.py         # ONNX embeddings
│   ├── main.py                  # CLI entry point
│   └── claude_code/             # Claude Code hooks
└── tests/
    ├── unit/
    ├── integration/
    └── performance_test.py
```

## Requirements

- Python >= 3.11
- numpy, onnxruntime, usearch, mcp, sentence-transformers

## Theoretical Foundations

This system is grounded in a formal framework where:

- **Memory is not a database** — it's optimized for survival relevance, prediction, and action guidance, not truth or completeness
- **Reconstruction is the genus** — episodic recall, semantic thought, and identity synthesis are all species of generative assembly from fragments
- **The self is a stabilized attractor** — a dynamic basin in reconstruction space, maintained by embodied, social, and memory-based constraints
- **Thought is variance management** — from focused reasoning (low variance) through creativity (high variance) to psychosis (uncontrolled)

See [`spec.md`](spec.md) for the formal axioms, definitions, and derived properties.

See [`CLAUDE.md`](CLAUDE.md) for the full developer reference including configuration, API details, and architecture.

## License

MIT License
