# Reconstructions

A cognitive science-inspired memory system that models how human memory actually works: lossy, reconstructive, and context-sensitive rather than perfect storage.

## Core Concept

Memory is **reconstructed, not retrieved**. Each recall assembles fragments weighted by salience, decay, and binding strength - producing slightly different results each time, just like human memory.

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Fragment** | `core.py` | Atomic unit of memory (multi-modal content, salience, bindings) |
| **Strand** | `core.py` | Reconstruction output (assembled fragments with coherence score) |
| **Query** | `core.py` | Request for memory retrieval (semantic text, filters, temporal range) |
| **Experience** | `encoding.py` | Raw input to be encoded into fragments |
| **FragmentStore** | `store.py` | SQLite persistence layer with batch loading optimization |
| **VectorIndex** | `vector_index.py` | USearch HNSW for fast semantic similarity search |
| **ReconstructionEngine** | `engine.py` | Main goal queue and processing loop |
| **ConsolidationScheduler** | `consolidation.py` | Autonomous reconstruction and pattern discovery |
| **MemoryHealthMonitor** | `health.py` | System health tracking and diagnostics |
| **RetrievalQualityTracker** | `metrics.py` | Query performance metrics and trend analysis |
| **MCP Server** | `mcp_server.py` | Claude Code integration via MCP protocol |

## Architecture Flow

```
Experience → Encoding Pipeline → Fragment (SQLite + VectorIndex)
                                      ↓
Query → Reconstruction Engine → Strand (assembled memory)
        (spread activation,           ↓
         select candidates,     Certainty Tracking
         fill gaps,                   ↓
         assemble)             Identity State
              ↑
              │
   Consolidation Scheduler
   (autonomous reconstruction,
    pattern discovery,
    binding strengthening)
```

## Key Algorithms

- **Salience**: `W_EMOTIONAL×intensity + W_NOVELTY×novelty + W_GOAL×relevance + W_DEPTH×depth`
- **Decay**: Power law `strength = salience × (t+1)^(-decay_rate) + rehearsal_bonus × log(access_count)`
- **Certainty**: `1.0 - variance` where variance = Jaccard distance between repeated reconstructions
- **Consolidation**: Autonomous rehearsal of recent/salient fragments, pattern discovery via semantic similarity, binding strengthening via co-activation

## Autonomous Reconstruction (Consolidation)

The consolidation layer solves a critical problem: **memories can be stored but not found** if retrieval paths are weak. Without consolidation, the system is like a library where books are shelved but never indexed.

### How It Runs in Claude Code

Consolidation runs **opportunistically** during hook processing:

1. **During Active Use** - After each memory encoding (tool use, user prompt, Claude response), the system checks if consolidation is due (default: every 60 seconds)
2. **At Session End** - Final consolidation runs when the session ends, mimicking sleep consolidation

This means consolidation happens automatically as you work, strengthening memory organization in the background without requiring a separate process.

### The Problem

When you encode "ONCAPON RTMP source at rtmp://example.com/stream" and later query "rtmp stream source", the system should find it. But initially:
- The fragment exists in the store
- Semantic similarity might be moderate
- **But binding paths are weak** - the fragment wasn't rehearsed or linked to related concepts
- Result: Query fails to retrieve the memory

### The Solution: Autonomous Reconstruction

Like human memory during sleep/idle time, the system spontaneously:

1. **Rehearses recent salient fragments** - Reconstructs around important memories to strengthen their retrieval paths
2. **Discovers patterns** - Finds semantically similar fragments and creates new bindings between them
3. **Strengthens bindings** - When fragments are frequently co-activated during rehearsal, creates permanent bindings
4. **Builds the semantic network** - Over time, creates a rich web of connections that makes retrieval robust

### How It Works

```python
# Consolidation runs automatically every N seconds (configurable)
engine = ReconstructionEngine(store, enable_consolidation=True)

# Each consolidation cycle:
# 1. Select salient/recent fragments for rehearsal
# 2. Reconstruct around each fragment (activating related fragments)
# 3. Track co-activations to identify frequently-paired fragments
# 4. Create new bindings between co-activated fragments
# 5. Periodically discover new patterns via semantic similarity
```

### Configuration

```python
from reconstructions import ConsolidationConfig

config = ConsolidationConfig()
config.RECENT_WINDOW_HOURS = 24.0  # Rehearse fragments from last 24h
config.MIN_SALIENCE_FOR_REHEARSAL = 0.3  # Only rehearse salient memories
config.REHEARSAL_BATCH_SIZE = 5  # Rehearse 5 fragments per cycle
config.CONSOLIDATION_INTERVAL_SECONDS = 60.0  # Run every minute
config.PATTERN_DISCOVERY_INTERVAL = 10  # Discover patterns every 10 cycles
config.SEMANTIC_SIMILARITY_THRESHOLD = 0.6  # Min similarity to create binding

engine = ReconstructionEngine(
    store,
    enable_consolidation=True,
    consolidation_config=config
)
```

### What Gets Consolidated

The scheduler prioritizes:
- **Recent** fragments (within `RECENT_WINDOW_HOURS`)
- **Salient** fragments (above `MIN_SALIENCE_FOR_REHEARSAL`)
- **Not recently rehearsed** (to avoid over-strengthening)

### Impact on Retrieval

After consolidation:
- Fragments have more binding connections
- Semantic clusters form naturally
- Retrieval paths are stronger and more diverse
- The same query finds more relevant fragments with higher coherence

### Biological Analogy

This mirrors human memory consolidation during:
- **Sleep** - Spontaneous replay strengthens important memories
- **Mind-wandering** - Idle-time processing discovers new connections
- **Meditation** - Self-directed rehearsal of specific memories
- **Daydreaming** - Goal-free exploration of memory space

## Health Monitoring & Performance

**Comprehensive System Visibility**

The health monitoring system provides real-time insights into memory system performance:

```python
from reconstructions import MemoryHealthMonitor

monitor = MemoryHealthMonitor(store, metrics_dir)
report = monitor.diagnose()

# Automatic tracking of:
# - Fragment statistics (count, salience, bindings, access patterns)
# - Consolidation metrics (frequency, rehearsals, pattern discovery)
# - Retrieval performance (latency, coherence, success rate)
# - System health warnings and recommendations
```

**Key Metrics:**
- **Fragment Stats**: Total count, recent growth, avg salience, binding connectivity
- **Consolidation Health**: Last run time, frequency, rehearsals performed
- **Query Performance**: Latency (avg & p95), coherence scores, success rates
- **Warnings**: Detects consolidation delays, poor retrieval, database growth

**Access via `/memory health`:**
```
⏺ Memory System Health Report

  Generated: 2026-02-04 14:12:38
  Database Size: 50.11 MB
  Health Score: 0.9/1.0

  Fragments: 5,379 total | 3,211 last 24h | 0.429 avg salience | 6.82 avg bindings
  Consolidation: Last run 21min ago | 1 run/hour | 5 rehearsals
  Retrieval: 6 queries | 10.0 avg fragments | 0.756 coherence | 100% success
```

**Performance Optimizations**

Batch loading eliminates N+1 query problems during reconstruction:

```python
# Before: 500+ individual database queries
for frag_id in fragment_ids:
    fragment = store.get(frag_id)  # Separate query each time

# After: 3-4 batch queries total
fragments = store.get_many(fragment_ids)  # Single batch load
```

**Impact:**
- Small datasets (20 fragments): **50ms** query latency
- Large datasets (5,000+ fragments): **100-300ms** (was 3+ seconds)
- 10-50x improvement on spreading activation and candidate selection

**Quality Tracking**

The metrics system tracks retrieval quality over time:

```python
from reconstructions import RetrievalQualityTracker

tracker = RetrievalQualityTracker()
metric = tracker.log_query(query, result_strand, latency_ms=125.5)

# Analyze trends
snapshot = tracker.get_snapshot(hours=24)
impact = tracker.consolidation_impact_analysis()  # Before/after comparison
```

## Adaptive Intelligence (Phase 2)

The system adapts to usage patterns and priorities through adaptive consolidation and identity-aware encoding.

### Adaptive Consolidation Scheduling

Consolidation frequency automatically adjusts based on system activity:

```python
from reconstructions.consolidation import AdaptiveConsolidationConfig

config = AdaptiveConsolidationConfig(
    adaptive_scheduling=True,        # Enable adaptive scheduling
    min_interval_seconds=10.0,       # High activity
    max_interval_seconds=300.0,      # Idle
    base_interval_seconds=60.0,      # Normal
    high_encoding_threshold=10,      # fragments/minute
    importance_threshold=0.7         # High salience
)

engine = ReconstructionEngine(store, consolidation_config=config)
```

**Behavior:**
- **High activity** (>10 encodings/min or high salience) → consolidate every 10s
- **Normal activity** → consolidate every 60s (base)
- **Idle** (no activity for 5 min) → consolidate every 300s

This ensures important memories are consolidated quickly while reducing overhead during quiet periods.

### Identity-Aware Encoding

Set active goals to boost salience of relevant memories:

```python
# Set active goal
engine.active_identity.set_active_goal(
    "Learn streaming protocols",
    intensity=0.8  # 0-1 priority
)

# Encode relevant experience - gets automatic boost
exp = Experience(text="RTMP protocol implementation guide")
engine.submit_experience(exp)
result = engine.step()

# Fragment salience boosted by +0.16 (0.2 × 0.8 intensity)
```

**Multiple goals:**
```python
engine.active_identity.set_active_goal("Debug auth", intensity=0.6)
engine.active_identity.set_active_goal("Optimize DB", intensity=0.9)
engine.active_identity.set_active_goal("Learn Rust", intensity=0.7)

# Fragments matching any goal get boosted
# Multiple matches can cumulate (capped at +0.5)
```

**Goal completion:**
```python
engine.active_identity.clear_goal("Learn Rust")  # Mark complete/inactive
```

**How it works:**
- Uses semantic similarity (cosine similarity >0.5) between fragment content and goal description
- Boosts salience by `+0.2 × goal.intensity` for matching fragments
- Total identity boost capped at +0.5
- Makes goal-relevant memories more salient for encoding and retrieval

## Advanced Learning (Phase 3)

The system learns optimal parameters and discovers patterns across sessions. **These features are fully integrated** and work automatically in the background.

### Self-Tuning Salience Weights (Integrated)

The system automatically learns which factors matter most for successful retrieval. **No manual intervention required** - weight learning happens automatically during normal operation:

```python
# Weight learning is enabled by default
engine = ReconstructionEngine(store, enable_weight_learning=True)

# That's it! The engine now:
# 1. Loads previously learned weights from ~/.reconstructions/weights.json (if exists)
# 2. Records feedback after each query (coherence > 0.5 = success)
# 3. Updates weights based on which factors contributed to success/failure
# 4. Saves checkpoints every 10 consolidations
# 5. Uses learned weights for all new salience calculations

# Check current learned weights (optional)
if engine.weight_learner:
    weights = engine.weight_learner.get_current_weights()
    print(weights)
    # {"emotional": 0.32, "novelty": 0.12, "goal": 0.35, "depth": 0.21}
```

**How it works:**
- Every query provides feedback: successful (coherence > 0.5) or failed
- For each retrieved fragment, the system learns which salience factors helped
- Successful retrievals → increase weights for contributing factors
- Failed retrievals → decrease weights slightly
- Weights stay normalized (sum to 1.0) and bounded [0.05, 0.50]
- Converges after ~100 feedback samples
- **Automatically saved** every 10 consolidations to persist across sessions

**Manual weight management (optional):**
```python
from reconstructions.learning import SalienceWeightLearner

# Load specific checkpoint
learner = SalienceWeightLearner.load_checkpoint(Path("weights.json"))

# Create engine with custom learner
engine = ReconstructionEngine(store, enable_weight_learning=False)
engine.weight_learner = learner

# Manually save checkpoint
learner.save_checkpoint(Path("custom_weights.json"))
```

### Cross-Session Pattern Recognition (Integrated)

The system automatically discovers recurring patterns during consolidation. **Runs every 10 consolidations** without manual triggering:

```python
# Pattern detection is enabled by default
engine = ReconstructionEngine(store)

# That's it! The engine now:
# 1. Runs pattern detection every 10 consolidations
# 2. Detects temporal, workflow, and project patterns
# 3. Saves patterns to ~/.reconstructions/patterns.json
# 4. Patterns accumulate and evolve over time

# Access detected patterns (optional)
patterns = engine.get_detected_patterns()
print(f"Found {len(patterns['temporal'])} temporal patterns")
print(f"Found {len(patterns['workflow'])} workflow patterns")
print(f"Found {len(patterns['project'])} project patterns")

# Manually trigger detection (optional)
stats = engine.force_pattern_detection()
print(f"Detected {stats['temporal']} temporal, {stats['workflow']} workflow, {stats['project']} project patterns")
```

**Automatic pattern detection:**
```python
# During consolidation (happens every 60s by default)
# On every 10th consolidation cycle:
# 1. Detect temporal patterns (weekly/daily routines)
# 2. Detect workflow patterns (git workflows, build processes)
# 3. Detect project patterns (semantic clusters)
# 4. Save to ~/.reconstructions/patterns.json
# 5. Include stats in consolidation result
```

**Pattern Types:**
- **TemporalPattern**: Time-based routines (daily/weekly activities)
- **WorkflowPattern**: Common operation sequences (git workflows, build processes)
- **ProjectPattern**: Semantic clusters of related work (detected via embedding similarity)

**Manual pattern analysis (optional):**
```python
from reconstructions.patterns import CrossSessionPatternDetector

detector = CrossSessionPatternDetector(store)

# Detect specific pattern types
temporal = detector.detect_temporal_patterns(min_confidence=0.7)
for pattern in temporal:
    print(f"{pattern.pattern_type}: {pattern.description}")
    # Output: "weekly: Tuesday: streaming, rtmp, video"

workflows = detector.detect_workflow_patterns(min_frequency=3)
for pattern in workflows:
    print(f"Workflow: {' → '.join(pattern.steps)} ({pattern.frequency}x)")
    # Output: "Workflow: git → status → add → commit (15x)"

projects = detector.detect_project_switches(similarity_threshold=0.6)
for pattern in projects:
    print(f"Project: {pattern.project_name} ({len(pattern.fragment_ids)} fragments)")
    # Output: "Project: streaming (23 fragments)"
```

**Use cases:**
- Proactive context loading when pattern detected
- Salience boosting for pattern-relevant fragments
- Workflow automation suggestions
- Understanding your cognitive activity patterns over time

## Running Tests

```bash
pytest tests/                         # All tests
pytest tests/unit/                    # Unit tests only
pytest tests/integration/test_e2e.py  # End-to-end
pytest tests/integration/test_consolidation.py  # Consolidation tests
pytest tests/performance_test.py      # Performance benchmarks
```

## Common Tasks

**Use as Python library:**
```python
from reconstructions import FragmentStore, ReconstructionEngine
from reconstructions.encoding import Experience
from reconstructions.core import Query

store = FragmentStore("memory.db")
engine = ReconstructionEngine(store)

# Store
engine.submit_experience(Experience(text="Something happened"))
engine.step()

# Recall
engine.submit_query(Query(semantic="what happened"))
result = engine.step()
```

**Run MCP server:**
```python
from reconstructions.mcp_server import create_mcp_server
server = create_mcp_server()
```

**CLI:**
```bash
python -m reconstructions.main
python -m reconstructions.main --data-dir ~/.my-memory --db custom.db
```

## Directory Structure

```
reconstructions/
├── core.py           # Fragment, Strand, Query data structures
├── encoding.py       # Experience, Context input structures
├── encoder.py        # Encoding pipeline (features → salience → bindings → store)
├── features.py       # Feature extraction (semantic, emotional, temporal)
├── salience.py       # Salience calculation
├── strength.py       # Decay and rehearsal modeling
├── bindings.py       # Fragment linking (temporal, semantic)
├── store.py          # SQLite FragmentStore
├── vector_index.py   # USearch HNSW vector index
├── reconstruction.py # Core retrieval engine (spread activation, assembly)
├── consolidation.py  # Autonomous reconstruction and pattern discovery
├── certainty.py      # Variance tracking for subjective certainty
├── identity.py       # Trait, Belief, Goal, IdentityEvolver
├── engine.py         # ReconstructionEngine main loop
├── mcp_server.py     # MCP server for Claude Code
├── fast_embedder.py  # ONNX embeddings with hardware auto-selection
├── main.py           # CLI entry point
└── claude_code/      # Claude Code hooks integration
    ├── hooks.py          # Hook handlers
    ├── capture.py        # Tool → Experience conversion
    ├── context_manager.py # Session state management
    └── skills.py         # /memory skill
```

## Dependencies

- **numpy** - Numerical computing
- **onnxruntime** - Fast inference (all-MiniLM-L6-v2 embeddings, 384-dim)
- **usearch** - HNSW vector search
- **mcp** - Model Context Protocol integration
- **sentence-transformers** - Fallback embeddings

## MCP Tools

When running as MCP server, exposes:
- `memory_store(text, emotional_valence, emotional_arousal)` - Encode new experience
- `memory_recall(query, min_salience)` - Query and reconstruct memory
- `memory_identity()` - Get current identity state
- `memory_status()` - System health metrics

## Claude Code Hooks Integration

The system integrates with Claude Code via hooks configured in `~/.claude/settings.json`.

### Configured Hooks

| Hook | What it captures | Source |
|------|------------------|--------|
| **SessionStart** | Initializes memory, loads project context | `session_id`, `project_path` |
| **UserPromptSubmit** | User messages | `prompt` field directly |
| **PostToolUse** | Successful tool calls (Write, Edit, Bash, etc.) | `tool_input`, `tool_output` |
| **PostToolUseFailure** | Failed tool calls with errors | `tool_input`, `error` |
| **Stop** | Claude's responses | Reads from `transcript_path` JSONL |
| **SubagentStart** | Agent spawning events | `agent_type`, `description` |
| **SubagentStop** | Agent completion and output | Reads from `agent_transcript_path` |
| **SessionEnd** | Session summary with stats | `fragments_encoded`, `duration` |

### Hook Architecture

Since each hook runs as a **separate Python process**, context state (recent fragments for temporal bindings) is persisted to `~/.reconstructions/session_state.json` between invocations.

```
Hook fires → Load persisted state → Encode fragment → Save state → Exit
```

### Settings Configuration

```json
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks session_start"}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks user_prompt_submit"}]}],
    "PostToolUse": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks post_tool_use"}]}],
    "PostToolUseFailure": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks post_tool_use_failure"}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks stop"}]}],
    "SubagentStart": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks subagent_start"}]}],
    "SubagentStop": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks subagent_stop"}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "python -m reconstructions.claude_code.hooks session_end"}]}]
  }
}
```

### Data Flow

```
User prompt → UserPromptSubmit hook → Fragment (salience ~0.52)
Tool use    → PostToolUse hook      → Fragment (salience ~0.36-0.48)
Claude reply → Stop hook            → Fragment (reads transcript)
Agent work  → SubagentStop hook     → Fragment (reads agent transcript)
```

### Key Files

```
claude_code/
├── hooks.py           # Hook handlers (on_session_start, on_post_tool_use, etc.)
├── capture.py         # Tool event → Experience conversion
├── context_manager.py # SessionContext singleton + state persistence
└── __main__.py        # CLI entry point for hooks
```

## Design Principles

1. **Lossy by design** - Not all experiences stored equally (salience-weighted)
2. **Reconstructive** - Memory assembled from fragments, not retrieved whole
3. **Cross-domain binding** - Semantic + emotional + sensory integration
4. **Forgetting curve** - Power law decay with rehearsal bonus
5. **Certainty ≠ Truth** - Tracks reconstruction consistency, not accuracy
6. **Identity inertia** - Traits (0.95), beliefs (0.8), goals (0.2) resist change differently
7. **Never fully forget** - Fragments decay to MIN_STRENGTH (0.01), never disappear

## Natural Forgetting: Power Law Decay

The memory system implements biological forgetting through power law decay:

```python
# From strength.py
MIN_STRENGTH: float = 0.01  # Never fully forget
strength = salience × (t+1)^(-0.5) + rehearsal_bonus × log(access_count)
```

This means:
- **Fragments never fully disappear** - they decay to minimum strength (0.01)
- **Low-strength fragments naturally fall out of reconstructions** through weighted scoring
- **But remain accessible via spreading activation** through bindings
- **The semantic network stays intact** - no broken retrieval paths

The system deliberately does NOT implement hard deletion or pruning. Old, low-salience memories fade naturally but remain in the network, preserving the possibility of reactivation through associated memories - just like human memory.
