# Architecture Deep Dive

## Fragment: The Atomic Unit

```python
@dataclass
class Fragment:
    id: str                          # UUID
    created_at: float                 # Unix timestamp
    content: dict[str, Any]          # Multi-modal content
    bindings: dict[str, list[str]]   # Links to other fragments
    initial_salience: float          # 0-1, how important at encoding
    access_log: list[float]          # Timestamps of recalls (rehearsal)
    source: str                      # Where this came from
    tags: list[str]                  # Metadata
```

### Content Domains

```python
content = {
    "semantic": [0.1, -0.3, ...],    # 384-dim embedding vector
    "emotional": {
        "valence": 0.7,              # Positive/negative (0-1)
        "arousal": 0.5,              # Activation level (0-1)
        "dominance": 0.6             # Control/power (0-1)
    },
    "temporal": {
        "absolute": 1706123456.0,    # Unix timestamp
        "sequence_position": 42,      # Order in session
        "context_id": 12345          # Session identifier
    },
    "motor": {...},                  # Action-related (optional)
    "sensory": {...}                 # Perceptual (optional)
}
```

## Salience Formula

```python
salience = (
    0.35 * emotional_intensity +     # Arousal + |valence - 0.5|
    0.30 * novelty +                 # Dissimilarity from recent
    0.25 * goal_relevance +          # Semantic similarity to active goals
    0.10 * processing_depth          # How deeply processed
)
```

## Strength Decay

Power-law forgetting with rehearsal:

```python
def calculate_strength(fragment, current_time):
    base_strength = fragment.initial_salience
    age_hours = (current_time - fragment.created_at) / 3600

    # Power law decay
    decay = 1.0 / (1.0 + age_hours ** 0.5)

    # Rehearsal bonus
    rehearsal_bonus = 0.0
    for access_time in fragment.access_log:
        recency = (current_time - access_time) / 3600
        rehearsal_bonus += 0.1 / (1.0 + recency ** 0.3)

    return min(1.0, base_strength * decay + rehearsal_bonus)
```

## Spreading Activation

```python
def spreading_activation(seed_fragments, store, iterations=3, decay=0.5):
    activation = {fid: 1.0 for fid in seed_fragments}

    for _ in range(iterations):
        new_activation = {}
        for fid, level in activation.items():
            fragment = store.get(fid)
            for binding_type, bound_ids in fragment.bindings.items():
                for bound_id in bound_ids:
                    spread = level * decay
                    new_activation[bound_id] = max(
                        new_activation.get(bound_id, 0),
                        spread
                    )
        activation.update(new_activation)

    return activation
```

## Reconstruction Pipeline

```
1. Query → semantic embedding
2. find_similar_semantic() → seed candidates
3. spreading_activation() → expand via bindings
4. For each candidate: score = activation × strength + noise
5. Select top-k by score
6. assemble_strand() → combine into coherent output
7. Apply constraints (identity, temporal consistency)
8. Return Strand with certainty score
```

## Identity Model

```python
@dataclass
class Identity:
    traits: dict[str, float]         # Stable characteristics (high inertia)
    beliefs: dict[str, Belief]       # Held truths with evidence
    goals: list[Goal]                # Active objectives (low inertia)
    snapshots: list[IdentitySnapshot] # History of self-concept
```

Traits have high inertia (slow to change), beliefs medium, goals low.

Identity constrains reconstruction: memories inconsistent with core identity are suppressed or reframed.

## Variance and Certainty

```python
class VarianceController:
    def track_reconstruction(self, query_hash, strand):
        # Store this reconstruction
        self.history[query_hash].append(strand)

    def get_certainty(self, query_hash):
        reconstructions = self.history[query_hash]
        if len(reconstructions) < 2:
            return 0.5  # Unknown

        # Low variance across reconstructions = high certainty
        variance = self._compute_variance(reconstructions)
        certainty = 1.0 / (1.0 + variance)
        return certainty
```

Key insight: **Certainty is not truth, just stability.**

## Engine Goal Queue

```python
class GoalPriority(Enum):
    QUERY = 0       # Highest: answer queries
    ENCODE = 1      # Store new experiences
    REFLECT = 2     # Background processing
    MAINTENANCE = 3 # Cleanup, consolidation
    IDLE = 4        # Nothing to do
```

## Storage Layer

- **SQLite**: Structured data (fragments, metadata)
- **In-memory dict**: Embeddings for fast similarity search
- **Indexes**: `created_at`, `initial_salience`

Current limitation: brute-force similarity search O(n).

## Speed Optimization Opportunities

1. **Replace sentence-transformers** with faster embeddings (ONNX, quantized)
2. **Vector index** (FAISS, Annoy, HNSWlib) for O(log n) search
3. **Async encoding** — encode in background, don't block
4. **Embedding cache** — skip re-embedding identical text
5. **Skip LLM intent parsing** — Claude Code IS the LLM, interpret directly
