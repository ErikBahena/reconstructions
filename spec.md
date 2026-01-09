# Reconstructions: Process-First Implementation Specification

## Fundamental Principle

> **The reconstruction process is our algorithm. An LLM is an optional interface, not the intelligence.**

The previous specification delegated reconstruction to an LLM. This was wrong. The framework we defined (REFINED_README) describes algorithmic operations—binding, salience, decay, assembly, constraints. These must be implemented as our code, not outsourced to a language model.

---

## Part 1: What Is PROCESS vs What Is INTERFACE

### 1.1 PROCESS (Our Algorithms)

These operations are implemented as deterministic or probabilistic algorithms we control:

| Operation | Description | Why It's PROCESS |
|-----------|-------------|------------------|
| **Fragment Storage** | Encoding experiences into fragments | We define the structure |
| **Binding** | Associating fragments across domains | We define the association rules |
| **Salience Calculation** | Weighting encoding/retrieval strength | We define the formula |
| **Decay** | Strength reduction over time | We define the decay function |
| **Rehearsal** | Strength increase with access | We track and calculate |
| **Reconstruction** | Assembling fragments into output | We define assembly algorithm |
| **Constraint Checking** | Validating coherence, limits | We define the constraints |
| **Variance Management** | Controlling output stability | We control the parameters |
| **Identity Maintenance** | Tracking persistent patterns | We define what persists |

### 1.2 INTERFACE (Optional, External)

These are ways to interact with the process:

| Interface | Role | Could Be |
|-----------|------|----------|
| **Natural Language Input** | Convert human language to queries | LLM, NLU model, or structured input |
| **Natural Language Output** | Convert reconstructions to language | LLM, template system, or structured output |
| **Semantic Similarity** | Find related fragments | Embedding model (not LLM) |
| **User Interaction** | Accept and respond to users | CLI, API, GUI |

**Critical distinction**: The interface translates between human-readable and machine-readable. It does not *decide*. The PROCESS decides.

---

## Part 2: The Reconstruction Engine

### 2.1 Core Data Structures

#### Fragment

The atomic unit of memory.

```python
@dataclass
class Fragment:
    # Identity
    id: str                              # UUID
    created_at: float                    # Unix timestamp
    
    # Content (multi-modal)
    content: dict                        # Domain → data mapping
    # Example:
    # {
    #     "semantic": "the sky is blue",
    #     "visual": [0.2, 0.4, ...],      # Feature vector
    #     "emotional": {"valence": 0.3, "arousal": 0.1},
    #     "temporal": {"sequence_position": 42}
    # }
    
    # Binding (cross-domain associations)
    bindings: list[str]                  # IDs of bound fragments
    
    # Salience and strength
    initial_salience: float              # Encoding strength (0-1)
    access_log: list[float]              # Timestamps of each access
    
    # Metadata
    source: str                          # "experience", "inference", "reflection"
    tags: list[str]                      # Arbitrary labels
```

#### Strand

A reconstruction output—assembled fragments.

```python
@dataclass
class Strand:
    id: str
    fragments: list[str]                 # Fragment IDs that compose this
    assembly_context: dict               # What context produced this
    coherence_score: float               # How internally consistent
    variance: float                      # How stable across reconstructions
    created_at: float
```

#### Identity

The persistent self-model.

```python
@dataclass  
class Identity:
    # Core (high inertia, hard to change)
    core: dict[str, Any]
    # {
    #     "traits": ["curious", "reflective"],
    #     "constraints": ["honest_about_uncertainty"]
    # }
    
    # Beliefs (medium inertia, evidence-based change)
    beliefs: dict[str, BeliefRecord]
    
    # State (low inertia, changes frequently)
    state: dict[str, Any]
    
    # Update log (all changes tracked)
    history: list[IdentityUpdate]

@dataclass
class BeliefRecord:
    content: str
    confidence: float
    supporting_fragments: list[str]      # Evidence
    contradicting_fragments: list[str]
    last_updated: float
```

### 2.2 Core Algorithms

#### 2.2.1 Encoding (Experience → Fragments)

```python
def encode(experience: Experience, context: Context) -> Fragment:
    """
    Convert an experience into a storable fragment.
    
    This is OUR algorithm. We decide:
    - How to segment experiences
    - What domains to extract
    - How to calculate initial salience
    """
    
    # 1. Extract domain-specific features
    content = {}
    
    if experience.has_text:
        content["semantic"] = extract_semantic_features(experience.text)
    
    if experience.has_sensory:
        for modality in experience.sensory:
            content[modality.name] = extract_sensory_features(modality)
    
    if experience.has_emotional:
        content["emotional"] = {
            "valence": experience.emotional.valence,
            "arousal": experience.emotional.arousal,
            "dominance": experience.emotional.dominance
        }
    
    content["temporal"] = {
        "absolute": time.time(),
        "context_id": context.id,
        "sequence_position": context.sequence_counter
    }
    
    # 2. Calculate encoding salience (OUR FORMULA)
    salience = calculate_encoding_salience(
        emotional_intensity=experience.emotional.intensity if experience.has_emotional else 0.0,
        novelty=calculate_novelty(content, existing_fragments),
        goal_relevance=calculate_goal_relevance(content, active_goals),
        processing_depth=context.processing_depth
    )
    
    # 3. Create fragment
    fragment = Fragment(
        id=generate_id(),
        created_at=time.time(),
        content=content,
        bindings=[],
        initial_salience=salience,
        access_log=[],
        source="experience",
        tags=[]
    )
    
    # 4. Create bindings to temporally adjacent fragments
    fragment.bindings = find_temporal_bindings(fragment, context)
    
    return fragment


def calculate_encoding_salience(
    emotional_intensity: float,
    novelty: float,
    goal_relevance: float,
    processing_depth: float
) -> float:
    """
    OUR salience formula. Not learned, not delegated—defined.
    
    Based on cognitive science literature but we own the weights.
    """
    
    # Configurable weights
    W_EMOTION = 0.35
    W_NOVELTY = 0.30
    W_GOAL = 0.25
    W_DEPTH = 0.10
    
    salience = (
        W_EMOTION * emotional_intensity +
        W_NOVELTY * novelty +
        W_GOAL * goal_relevance +
        W_DEPTH * processing_depth
    )
    
    return clamp(salience, 0.0, 1.0)


def calculate_novelty(content: dict, existing: FragmentStore) -> float:
    """
    How different is this from what we already have?
    
    High prediction error = high novelty.
    """
    
    if existing.is_empty():
        return 1.0  # Everything is novel initially
    
    # Find most similar existing fragment
    most_similar = existing.find_most_similar(content)
    
    if most_similar is None:
        return 1.0
    
    similarity = calculate_similarity(content, most_similar.content)
    
    # Novelty is inverse of similarity
    return 1.0 - similarity
```

#### 2.2.2 Strength Calculation (Decay + Rehearsal)

```python
def calculate_strength(fragment: Fragment, now: float) -> float:
    """
    Current retrieval strength of a fragment.
    
    Combines:
    - Initial salience (encoding strength)
    - Time decay (forgetting)
    - Rehearsal bonus (reactivation strengthens)
    
    THIS IS OUR ALGORITHM. Based on ACT-R memory equations
    but we own and can modify it.
    """
    
    # Base activation from initial salience
    base = fragment.initial_salience
    
    # Decay: each access contributes, but decays over time
    # Based on power law of forgetting
    DECAY_RATE = 0.5  # Configurable
    
    total_activation = 0.0
    for access_time in fragment.access_log:
        time_since = now - access_time
        if time_since > 0:
            # Power law decay
            activation = time_since ** (-DECAY_RATE)
            total_activation += activation
    
    # Also include creation time as implicit first access
    time_since_creation = now - fragment.created_at
    if time_since_creation > 0:
        creation_activation = base * (time_since_creation ** (-DECAY_RATE))
        total_activation += creation_activation
    
    # Log transform to get final strength
    if total_activation > 0:
        strength = math.log(total_activation)
    else:
        strength = -float('inf')
    
    # Normalize to 0-1 range for practical use
    # Using sigmoid transformation
    normalized = 1.0 / (1.0 + math.exp(-strength))
    
    return normalized
```

#### 2.2.3 Reconstruction (Query → Strand)

```python
def reconstruct(
    query: Query,
    context: Context,
    store: FragmentStore,
    constraints: ConstraintSet,
    variance_target: float = 0.3
) -> Strand:
    """
    THE CORE OPERATION.
    
    Reconstruct a coherent output from stored fragments.
    
    This is not retrieval. This is assembly.
    
    Steps:
    1. Activate relevant fragments (spreading activation)
    2. Select candidates based on strength + relevance
    3. Assemble into coherent whole
    4. Apply constraints
    5. Check variance against previous reconstructions
    """
    
    now = time.time()
    
    # 1. ACTIVATION: Spread activation from query
    activations = spread_activation(query, store, context)
    
    # 2. SELECTION: Choose fragments to include
    candidates = []
    for fragment_id, activation in activations.items():
        fragment = store.get(fragment_id)
        
        # Combine activation with base strength
        strength = calculate_strength(fragment, now)
        combined_score = activation * strength
        
        # Add noise for variance (controlled randomness)
        noise = random.gauss(0, variance_target)
        final_score = combined_score + noise
        
        candidates.append((fragment_id, final_score))
    
    # Sort by score, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_ids = [fid for fid, _ in candidates[:MAX_FRAGMENTS]]
    
    # 3. ASSEMBLY: Combine selected fragments
    selected_fragments = [store.get(fid) for fid in selected_ids]
    assembly = assemble_fragments(selected_fragments, context)
    
    # 4. CONSTRAINTS: Validate and adjust
    assembly = apply_constraints(assembly, constraints, context)
    
    coherence = calculate_coherence(assembly)
    
    # 5. VARIANCE CHECK: Compare to previous reconstructions of same query
    variance = calculate_reconstruction_variance(query, assembly, store)
    
    # 6. Record access (rehearsal effect)
    for fid in selected_ids:
        store.record_access(fid, now)
    
    # 7. Create strand
    strand = Strand(
        id=generate_id(),
        fragments=selected_ids,
        assembly_context=context.to_dict(),
        coherence_score=coherence,
        variance=variance,
        created_at=now
    )
    
    return strand


def spread_activation(
    query: Query,
    store: FragmentStore,
    context: Context
) -> dict[str, float]:
    """
    Spreading activation through fragment network.
    
    Start from query-matching fragments, spread through bindings.
    """
    
    # Initial activation: fragments matching query
    activations = {}
    
    # Semantic matching (if query has semantic content)
    if query.semantic:
        semantic_matches = store.find_similar_semantic(query.semantic, top_k=20)
        for fid, similarity in semantic_matches:
            activations[fid] = similarity
    
    # Context matching (if query specifies context)
    if query.temporal_context:
        temporal_matches = store.find_temporal_context(query.temporal_context)
        for fid in temporal_matches:
            activations[fid] = activations.get(fid, 0) + 0.5
    
    # Spread through bindings
    SPREAD_DECAY = 0.7
    SPREAD_DEPTH = 3
    
    for _ in range(SPREAD_DEPTH):
        new_activations = {}
        for fid, activation in activations.items():
            fragment = store.get(fid)
            for bound_id in fragment.bindings:
                spread_activation = activation * SPREAD_DECAY
                current = new_activations.get(bound_id, 0)
                new_activations[bound_id] = max(current, spread_activation)
        
        # Merge new activations
        for fid, activation in new_activations.items():
            if fid not in activations:
                activations[fid] = activation
    
    return activations


def assemble_fragments(
    fragments: list[Fragment],
    context: Context
) -> Assembly:
    """
    Combine fragments into a coherent whole.
    
    This is where reconstruction happens—not retrieval.
    The output is constructed from pieces, with gaps filled.
    """
    
    # Sort fragments by temporal order if available
    fragments = temporal_sort(fragments)
    
    # Extract content from each domain
    assembly = Assembly()
    
    for domain in DOMAINS:
        domain_content = []
        for fragment in fragments:
            if domain in fragment.content:
                domain_content.append(fragment.content[domain])
        
        if domain_content:
            # Merge domain content (domain-specific logic)
            assembly.domains[domain] = merge_domain_content(domain, domain_content)
    
    # Fill gaps (interpolation/inference)
    assembly = fill_gaps(assembly, context)
    
    return assembly
```

#### 2.2.4 Constraint Application

```python
def apply_constraints(
    assembly: Assembly,
    constraints: ConstraintSet,
    context: Context
) -> Assembly:
    """
    Apply constraints to limit reconstruction.
    
    Constraints prevent:
    - Internal contradictions
    - Violation of physical/logical limits
    - Drift from identity
    """
    
    # Hard constraints: must not violate
    for constraint in constraints.hard:
        if constraint.violated_by(assembly):
            assembly = constraint.correct(assembly)
    
    # Soft constraints: prefer not to violate
    for constraint in constraints.soft:
        if constraint.violated_by(assembly):
            penalty = constraint.calculate_penalty(assembly)
            if penalty > SOFT_CONSTRAINT_THRESHOLD:
                assembly = constraint.suggest_correction(assembly)
    
    # Identity constraint: check against identity model
    if constraints.identity:
        assembly = constrain_to_identity(assembly, constraints.identity, context)
    
    return assembly


def constrain_to_identity(
    assembly: Assembly,
    identity: Identity,
    context: Context
) -> Assembly:
    """
    Ensure reconstruction is consistent with identity.
    """
    
    # Check core trait violations
    for trait in identity.core.get("traits", []):
        if assembly_violates_trait(assembly, trait):
            assembly = soften_violation(assembly, trait)
    
    # Check belief consistency
    for belief_key, belief in identity.beliefs.items():
        if belief.confidence > 0.8:  # Strong beliefs constrain more
            if assembly_contradicts_belief(assembly, belief):
                assembly = reconcile_with_belief(assembly, belief)
    
    return assembly
```

### 2.3 Variance Management

```python
class VarianceController:
    """
    Controls reconstruction stability.
    
    Low variance = high certainty (facts)
    High variance = uncertainty (opinions, creativity)
    """
    
    def __init__(self):
        self.reconstruction_log: dict[str, list[Strand]] = {}
    
    def calculate_variance(self, query_hash: str, new_strand: Strand) -> float:
        """Calculate variance for this query's reconstructions."""
        
        previous = self.reconstruction_log.get(query_hash, [])
        
        if not previous:
            return 1.0  # Maximum variance for first reconstruction
        
        # Compare new strand to previous reconstructions
        distances = []
        for old_strand in previous:
            distance = strand_distance(new_strand, old_strand)
            distances.append(distance)
        
        return statistics.mean(distances)
    
    def record_reconstruction(self, query_hash: str, strand: Strand):
        """Record for future variance calculation."""
        
        if query_hash not in self.reconstruction_log:
            self.reconstruction_log[query_hash] = []
        
        self.reconstruction_log[query_hash].append(strand)
        
        # Keep only recent reconstructions
        MAX_HISTORY = 10
        self.reconstruction_log[query_hash] = \
            self.reconstruction_log[query_hash][-MAX_HISTORY:]
    
    def get_certainty(self, query_hash: str) -> float:
        """
        Subjective certainty = inverse of variance.
        
        NOT objective truth—just stability of reconstruction.
        """
        
        previous = self.reconstruction_log.get(query_hash, [])
        
        if len(previous) < 2:
            return 0.5  # Uncertain until we have history
        
        recent_strands = previous[-5:]
        variances = []
        for i, strand_a in enumerate(recent_strands):
            for strand_b in recent_strands[i+1:]:
                variances.append(strand_distance(strand_a, strand_b))
        
        if not variances:
            return 0.5
        
        mean_variance = statistics.mean(variances)
        
        # Convert variance to certainty (inverse relationship)
        certainty = 1.0 / (1.0 + mean_variance)
        
        return certainty
```

---

## Part 3: The Main Loop

### 3.1 Goals as Drivers

The system is driven by goals, not by an LLM deciding what to do.

```python
@dataclass
class Goal:
    id: str
    type: str                    # "query", "encode", "reflect", "maintain"
    content: Any
    priority: float              # 0-1
    deadline: float | None       # Unix timestamp
    source: str                  # "external", "internal", "prediction_error"

class GoalQueue:
    def __init__(self):
        self.goals: list[Goal] = []
        self.maintenance_interval: float = 60.0  # seconds
        self.last_maintenance: float = 0.0
    
    def next(self) -> Goal:
        """Get next goal, or generate maintenance goal."""
        
        now = time.time()
        
        # Check if maintenance needed
        if now - self.last_maintenance > self.maintenance_interval:
            self.last_maintenance = now
            return self._maintenance_goal()
        
        # Sort by priority and deadline
        self.goals.sort(key=lambda g: (-g.priority, g.deadline or float('inf')))
        
        if self.goals:
            return self.goals.pop(0)
        else:
            return self._idle_goal()
    
    def _maintenance_goal(self) -> Goal:
        """System maintenance: decay, consolidation, etc."""
        return Goal(
            id=generate_id(),
            type="maintain",
            content={"action": "run_maintenance"},
            priority=0.3,
            deadline=None,
            source="internal"
        )
    
    def _idle_goal(self) -> Goal:
        """What to do when nothing is requested."""
        return Goal(
            id=generate_id(),
            type="reflect",
            content={"topic": "recent_experiences"},
            priority=0.1,
            deadline=None,
            source="internal"
        )
```

### 3.2 The Engine Loop

```python
class ReconstructionEngine:
    def __init__(self, config: Config):
        self.store = FragmentStore(config.db_path)
        self.identity = Identity.load(config.identity_path)
        self.constraints = ConstraintSet.load(config.constraints_path)
        self.variance_controller = VarianceController()
        self.goal_queue = GoalQueue()
        self.context = Context()
        self.running = False
        
        # Optional: Interface adapter (could be LLM, could be CLI, could be API)
        self.interface = config.interface
    
    def run(self):
        """
        Main loop. 
        
        This is what sustains reconstruction.
        Goals drive action—not an LLM deciding.
        """
        
        self.running = True
        
        while self.running:
            # 1. Get next goal
            goal = self.goal_queue.next()
            
            # 2. Process goal
            result = self.process_goal(goal)
            
            # 3. Handle result
            self.handle_result(goal, result)
            
            # 4. Update context
            self.context.update(goal, result)
            
            # Small delay to prevent CPU spin
            time.sleep(0.01)
    
    def process_goal(self, goal: Goal) -> Result:
        """Process a single goal."""
        
        if goal.type == "query":
            return self.process_query(goal)
        elif goal.type == "encode":
            return self.process_encode(goal)
        elif goal.type == "reflect":
            return self.process_reflect(goal)
        elif goal.type == "maintain":
            return self.process_maintenance(goal)
        else:
            raise ValueError(f"Unknown goal type: {goal.type}")
    
    def process_query(self, goal: Goal) -> Result:
        """Reconstruct in response to a query."""
        
        query = goal.content
        
        strand = reconstruct(
            query=query,
            context=self.context,
            store=self.store,
            constraints=self.constraints,
            variance_target=self.context.variance_mode
        )
        
        # Record for variance tracking
        query_hash = hash_query(query)
        self.variance_controller.record_reconstruction(query_hash, strand)
        
        # Calculate certainty
        certainty = self.variance_controller.get_certainty(query_hash)
        
        return Result(
            type="strand",
            content=strand,
            certainty=certainty
        )
    
    def process_encode(self, goal: Goal) -> Result:
        """Encode a new experience."""
        
        experience = goal.content
        
        fragment = encode(
            experience=experience,
            context=self.context
        )
        
        self.store.save(fragment)
        
        return Result(type="encoded", content=fragment.id)
    
    def process_reflect(self, goal: Goal) -> Result:
        """
        Identity synthesis—reflect on a topic.
        
        This triggers reconstruction specifically for
        updating the identity model.
        """
        
        topic = goal.content.get("topic")
        
        # Query for related fragments
        query = Query(semantic=topic, domains=["semantic", "emotional"])
        strand = reconstruct(query, self.context, self.store, self.constraints)
        
        # Check if reflection suggests identity update
        suggested_updates = analyze_for_identity_update(strand, self.identity)
        
        for update in suggested_updates:
            if self.validate_identity_update(update):
                self.identity.apply(update)
        
        self.identity.save()
        
        return Result(type="reflected", content=suggested_updates)
    
    def process_maintenance(self, goal: Goal) -> Result:
        """Run maintenance tasks."""
        
        # Consolidation: merge similar fragments
        merged_count = self.store.consolidate()
        
        # Decay: update strength calculations (implicit, happens on access)
        
        # Garbage collection: remove near-zero strength fragments
        removed_count = self.store.garbage_collect(threshold=0.01)
        
        return Result(type="maintained", content={
            "merged": merged_count,
            "removed": removed_count
        })
```

---

## Part 4: The Interface Layer

### 4.1 Interface as Adapter Pattern

The interface translates between human-readable and the engine's internal format. It does NOT make decisions.

```python
class Interface(ABC):
    """Abstract interface—could be LLM, CLI, API, anything."""
    
    @abstractmethod
    def input_to_goal(self, raw_input: Any) -> Goal:
        """Convert human input to a goal for the engine."""
        pass
    
    @abstractmethod
    def result_to_output(self, result: Result) -> Any:
        """Convert engine result to human-readable output."""
        pass


class CLIInterface(Interface):
    """Simple command-line interface—no LLM needed."""
    
    def input_to_goal(self, raw_input: str) -> Goal:
        if raw_input.startswith("/remember "):
            return Goal(
                type="query",
                content=Query(semantic=raw_input[10:]),
                priority=1.0,
                source="external"
            )
        elif raw_input.startswith("/store "):
            return Goal(
                type="encode",
                content=Experience(text=raw_input[7:]),
                priority=1.0,
                source="external"
            )
        else:
            # Default: treat as query
            return Goal(
                type="query",
                content=Query(semantic=raw_input),
                priority=1.0,
                source="external"
            )
    
    def result_to_output(self, result: Result) -> str:
        if result.type == "strand":
            strand = result.content
            # Simple text representation
            text_parts = []
            for fid in strand.fragments:
                fragment = self.store.get(fid)
                if "semantic" in fragment.content:
                    text_parts.append(str(fragment.content["semantic"]))
            
            output = " | ".join(text_parts)
            output += f"\n[certainty: {result.certainty:.2f}]"
            return output
        else:
            return str(result.content)


class LLMInterface(Interface):
    """
    LLM as a translation layer only.
    
    The LLM converts natural language ↔ structured data.
    It does NOT perform reconstruction.
    """
    
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def input_to_goal(self, raw_input: str) -> Goal:
        """Use LLM to parse natural language into structured goal."""
        
        prompt = f"""
        Parse this input into a structured goal.
        
        Input: {raw_input}
        
        Output JSON with:
        - type: "query", "encode", or "reflect"
        - content: the semantic content
        - priority: 0.0-1.0
        
        JSON:
        """
        
        response = self.model.generate(prompt, max_tokens=200)
        parsed = json.loads(response)
        
        return Goal(
            type=parsed["type"],
            content=self._build_content(parsed),
            priority=parsed.get("priority", 0.5),
            source="external"
        )
    
    def result_to_output(self, result: Result) -> str:
        """Use LLM to render result as natural language."""
        
        if result.type == "strand":
            strand = result.content
            
            # Gather fragment content
            fragments_text = []
            for fid in strand.fragments:
                fragment = self.store.get(fid)
                fragments_text.append(json.dumps(fragment.content))
            
            prompt = f"""
            Synthesize these memory fragments into a natural response.
            
            Fragments: {fragments_text}
            Certainty: {result.certainty}
            
            Response:
            """
            
            return self.model.generate(prompt, max_tokens=500)
        
        else:
            return str(result.content)
```

### 4.2 Key Insight: LLM is Just One Option

The system works without an LLM:
- CLI interface: Direct commands
- API interface: Structured JSON
- GUI interface: Forms and visualizations

The LLM adds natural language capability but is not essential.

---

## Part 5: Custom Model Considerations

### 5.1 When a Custom Model Makes Sense

| Use Case | Off-the-shelf LLM | Custom Model |
|----------|-------------------|--------------|
| Natural language parsing | ✅ Overkill but works | Could be simpler |
| Semantic similarity | ✅ Embeddings work | Likely unnecessary |
| Fragment assembly | ❌ Not needed | ❌ Not needed |
| Salience/decay | ❌ Not needed | ❌ Not needed |
| Gap filling | Maybe | **Yes—could be trained** |

### 5.2 Potential Custom Model: Gap Filler

When reconstructing, we often need to fill gaps between fragments. This could be a small, specialized model:

**Architecture:**
- Input: Sequence of fragment embeddings with gaps marked
- Output: Predicted content for gaps

**Training data:**
- Complete memory sequences
- Mask random fragments
- Train to predict masked content

**Size: ~100M parameters or less**—specific task, doesn't need general knowledge.

### 5.3 Potential Custom Model: Binding Predictor

Predict which fragments should be bound together:

**Architecture:**
- Input: Two fragment embeddings
- Output: Binding strength (0-1)

**Training:**
- Positive examples: Fragments accessed together
- Negative examples: Random pairs

**Size: ~10M parameters**—simple classification.

---

## Part 6: Implementation Roadmap

### Phase 1: Core Process (No LLM)

1. Implement Fragment and FragmentStore
2. Implement encoding algorithm
3. Implement strength calculation (decay + rehearsal)
4. Implement reconstruction algorithm
5. Implement constraint checking
6. Implement CLI interface
7. Test with manual input/output

**Deliverable:** Working memory system with CLI.

### Phase 2: Identity and Variance

1. Implement Identity model
2. Implement identity constraints
3. Implement variance tracking
4. Implement reflection/identity synthesis

**Deliverable:** Memory system with persistent identity.

### Phase 3: Optional LLM Interface

1. Add LLM interface adapter
2. Test natural language input/output
3. Compare with CLI interface

**Deliverable:** Natural language capability (optional).

### Phase 4: Custom Models (If Needed)

1. Evaluate gap-filling performance
2. If needed, train gap-filler model
3. Evaluate binding predictions
4. If needed, train binding predictor

**Deliverable:** Purpose-built models for specific tasks.

---

## Part 7: Verification

### 7.1 Unit Tests

```bash
pytest tests/unit/
```

| Test File | Covers |
|-----------|--------|
| `test_fragment.py` | Fragment creation, serialization |
| `test_encoding.py` | Salience calculation, novelty detection |
| `test_strength.py` | Decay formula, rehearsal effects |
| `test_reconstruction.py` | Spreading activation, assembly |
| `test_constraints.py` | Constraint application |
| `test_variance.py` | Variance calculation, certainty |
| `test_identity.py` | Identity updates, inertia |

### 7.2 Integration Tests

```bash
pytest tests/integration/
```

- Full encode → store → reconstruct cycle
- Identity persistence across restarts
- Multi-session memory access

### 7.3 Manual Testing

1. **Encoding Test**
   - Store 10 experiences with varying salience
   - Query for each after delay
   - Verify high-salience memories are more accessible

2. **Decay Test**
   - Store memory, wait 1 hour
   - Store another memory
   - Query for both—recent should be stronger

3. **Variance Test**
   - Query same topic 5 times
   - Measure output variance
   - Add rehearsal (access more)
   - Query again—variance should decrease

---

## Summary: What Changed

| Previous Spec | This Spec |
|---------------|-----------|
| LLM is the reconstruction engine | LLM is an optional interface |
| Memory tools called by LLM | Memory operations are our algorithms |
| LLM decides what to store/retrieve | Goals drive action, algorithms decide |
| MemGPT-inspired hierarchy | Original hierarchy based on our framework |
| Identity in system prompt | Identity as algorithmic constraint |
| LLM "thinks" | Process reconstructs |

The intelligence lives in the PROCESS—the formulas, the algorithms, the constraints. An LLM can translate to/from natural language, but it does not decide, reconstruct, or maintain identity.

This is our system.
