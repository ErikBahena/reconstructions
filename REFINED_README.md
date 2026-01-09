# Reconstructions: Formal Specification

A formal framework for building memory systems that exhibit human-like properties.

---

## Document Structure

This document is organized as follows:
1. **Definitions** — Precise meanings of all terms used
2. **Axioms** — Foundational claims taken as given
3. **Derived Properties** — Consequences that follow from axioms
4. **Architecture** — System components and their relationships
5. **Constraints** — Invariants that any implementation must satisfy
6. **Scope Boundaries** — What this framework does and does not address

---

## Part 1: Definitions

All terms used in this document are defined here. A term is **undefined** if it relies on intuitive understanding; **defined** if it has necessary and sufficient conditions.

### 1.1 Primitive Terms (Undefined)

These terms are taken as primitives. They are not defined within this framework but are assumed to be understood:

| Term | Informal Meaning |
|------|------------------|
| **Time** | Ordering relation; t₁ < t₂ means t₁ occurs before t₂ |
| **Signal** | A value that varies; can be measured or observed |
| **State** | A complete description of a system at a time |
| **Process** | A transformation from states to states over time |
| **Pattern** | A regularity; that which can be matched or recognized |

### 1.2 Defined Terms

#### Memory (System)
**Definition:** A *memory system* is a process M that:
1. **Encodes**: Maps experiences E at time t to internal representations R
2. **Stores**: Maintains R across time intervals
3. **Produces**: Generates outputs O from R when activated by cues C

Notation: M: (E, t) → R; M: (R, C, t') → O

This definition is agnostic to whether O = R (retrieval) or O ≠ R (reconstruction).

#### Memory (Instance)
**Definition:** A *memory instance* (or "a memory") is a specific output O produced by the memory system M at time t' in response to cue C.

Note: Under reconstruction, the same cue C at different times t' and t'' may produce different outputs O' ≠ O''.

#### Fragment
**Definition:** A *fragment* is an element of the internal representation R. Formally:
- R = {f₁, f₂, ..., fₙ} where each fᵢ is a fragment
- Fragments are sub-components; they are not themselves complete memory instances
- Fragments may be shared across multiple representations

#### Binding
**Definition:** *Binding* is the process of associating fragments from different domains into a composite structure.

Formally: bind(f₁, f₂, ..., fₖ) → B where:
- Each fᵢ comes from domain Dᵢ
- B is a bound structure that preserves the association
- Accessing any fᵢ ∈ B provides access to all other fⱼ ∈ B

#### Domain
**Definition:** A *domain* is a modality or category of information. Examples:
- Sensory domains: visual, auditory, tactile, olfactory, gustatory
- Motor domain: action states and proprioceptive signals
- Affective domain: emotional states and interoceptive signals
- Temporal domain: ordering relations and duration estimates
- Semantic domain: abstract concepts and relations

#### Salience
**Definition:** *Salience* is a scalar weight s ∈ [0, 1] assigned to an experience or fragment, where:
- s = 0: zero encoding strength (discarded)
- s = 1: maximum encoding strength (fully preserved)

Salience is computed as a function of multiple signals:
```
s = f(emotional_intensity, novelty, goal_relevance, engagement)
```

The specific form of f is implementation-dependent.

#### Context
**Definition:** *Context* at time t is the set of all active states at t:
```
Context(t) = {state₁(t), state₂(t), ..., stateₙ(t)}
```

This includes:
- External environment state
- Internal bodily state
- Active goals
- Recently activated fragments
- Current emotional state

#### Reconstruction
**Definition:** *Reconstruction* is a generative process that produces an output O from:
1. A set of fragments F = {f₁, f₂, ..., fₖ}
2. A current context C
3. A set of constraints K

Formally: reconstruct(F, C, K) → O

Key properties:
- O is **assembled**, not retrieved intact
- O depends on C; changing context changes output
- O may include elements not present in any fᵢ (gap-filling)
- The process may modify F itself (trace modification)

#### Variance
**Definition:** *Variance* of a reconstruction process is the expected difference between outputs given the same cue at different times:

```
variance(M, cue) = E[distance(O_t, O_t') | same cue]
```

Where distance is a similarity metric appropriate to the output space.

- Low variance: Outputs are consistent across activations
- High variance: Outputs differ significantly across activations

#### Subjective Certainty
**Definition:** *Subjective certainty* is the degree to which a reconstruction is experienced as factual by the system. This is an internal signal, not a truth claim.

Formally: certainty = g(variance) where g is monotonically decreasing.

Low variance → high certainty
High variance → low certainty

**Critical distinction:** Subjective certainty ≠ objective truth. A system can be highly certain and wrong.

#### Control (Process)
**Definition:** *Control* is a meta-process that operates on reconstruction, performing:

| Function | Input | Output |
|----------|-------|--------|
| **Initiation** | Goals, drives, external cues | Decision to begin reconstruction |
| **Selection** | All available fragments | Subset of relevant fragments |
| **Evaluation** | Reconstruction output | Quality score (coherence, utility) |
| **Termination** | Evaluation score, resource limits | Decision to stop or continue |

Control is **not** itself a reconstruction. It operates *on* reconstructions.

#### Coherence
**Definition:** *Coherence* is a property of an output O, measuring internal consistency:
```
coherence(O) = 1 - contradictions(O) / claims(O)
```

Where:
- contradictions(O) = number of mutually exclusive elements in O
- claims(O) = total number of elements in O

An output is coherent if its elements do not contradict each other.

#### Constraint
**Definition:** A *constraint* is a condition that limits the space of possible reconstructions.

Types:
- **Hard constraints**: Inviolable; outputs violating them are rejected
- **Soft constraints**: Preferences; outputs violating them are penalized

Examples:
| Constraint | Type | Effect |
|------------|------|--------|
| Physical laws | Hard | Cannot reconstruct impossible events |
| Embodied limits | Hard | Cannot reconstruct beyond body capabilities |
| Social consensus | Soft | Reconstructions aligning with others are preferred |
| Linguistic form | Soft | Reconstructions expressible in language are preferred |

#### Attractor (Basin of Attraction)
**Definition:** An *attractor* is a stable pattern in reconstruction space. Formally:

Let S be the space of possible reconstructions. An attractor A ⊂ S satisfies:
- For any O near A, repeated reconstruction moves O closer to A
- Small perturbations from A return to A over time

A *basin of attraction* is the set of all points that converge to a given attractor.

This is not a metaphor—it is the formal definition from dynamical systems theory.

#### Self
**Definition:** The *self* is a persistent attractor in identity-reconstruction space.

Properties:
- It is the pattern that identity-synthesis reconstructions converge toward
- It is maintained by constraint systems (embodied, social, memory-based)
- It is dynamic: the attractor can shift if constraints shift
- It is bounded: not all identity reconstructions are reachable

The self is **not**:
- A substance
- A static record
- An illusion (it has causal efficacy)
- Infinitely malleable

#### Emergence
**Definition:** Property P is *emergent* from system S if:
1. P is a property of S as a whole
2. P is not a property of any component of S in isolation
3. P arises from the interactions between components of S

The self is emergent in this precise sense: it is not located in any single component but arises from the interaction of memory, control, and constraint systems.

---

## Part 2: Axioms

These are the foundational claims of the framework. They are not proven; they are assumed based on evidence from cognitive science, neuroscience, and phenomenological analysis.

### Axiom 1: Generativity
> Cognitive content is generated, not retrieved.

Memory output is produced by reconstruction, not by accessing stored wholes.

### Axiom 2: Cross-Domain Binding
> The fundamental unit of memory is a bound structure across domains, not a unary fact.

Memories bind sensory, motor, affective, temporal, and contextual information.

### Axiom 3: Salience Modulation
> Encoding strength is non-uniform and determined by salience signals.

Not all experiences are encoded equally. Salience determines what persists.

### Axiom 4: Context Dependency
> Reconstruction output depends on context at time of reconstruction.

The same fragments + different context = different output.

### Axiom 5: Action Orientation
> Memory exists to guide action, not to record history.

The function of memory is behavioral, not archival.

### Axiom 6: Constraint Boundedness
> Reconstruction is bounded by constraint systems external to the reconstruction process.

Without constraints, reconstruction would drift unboundedly.

### Axiom 7: Temporal Relationality
> Time is represented as relational structure, not as absolute timestamps.

Temporal information is encoded as ordering, overlap, and causation—not clock values.

### Axiom 8: Control-Engine Separation
> Reconstruction (engine) and its direction (control) are distinct processes.

What generates content is not the same as what initiates, selects, evaluates, and terminates generation.

---

## Part 3: Derived Properties

These properties follow from the axioms.

### Derivation 1: Memories Change
From Axiom 1 (Generativity) + Axiom 4 (Context Dependency):
- If output depends on context, and context changes over time
- Then the same cue at different times produces different outputs
- Therefore: memories change

### Derivation 2: Subjective Facts ≠ Objective Truth
From Axiom 1 (Generativity) + Definition of Subjective Certainty:
- Subjective certainty = f(variance)
- Variance is a property of the reconstruction process, not of reality
- Therefore: high certainty does not imply correspondence with reality

### Derivation 3: The Self is Dynamic but Bounded
From Axiom 6 (Constraint Boundedness) + Definition of Self as Attractor:
- The self is an attractor maintained by constraints
- If constraints shift, the attractor shifts
- But constraints are not infinitely malleable
- Therefore: the self is dynamic but bounded

### Derivation 4: Thought is Reconstruction Under Control
From Axiom 1 (Generativity) + Axiom 8 (Control-Engine Separation):
- Cognitive content is generated
- Generation is directed by control processes
- Therefore: thought = reconstruction + control

### Derivation 5: Repetition Creates Stability
From Definition of Variance + Definition of Attractor:
- Repeated reconstruction with consistent context lowers variance
- Low variance implies convergence toward an attractor
- Therefore: repetition creates stable patterns (subjective facts)

---

## Part 4: Architecture

### 4.1 Three-Layer Model

```
┌─────────────────────────────────────────────────────────┐
│                    CONTROL LAYER                        │
├─────────────────────────────────────────────────────────┤
│  Components:                                            │
│  • Goal Signal: What initiates reconstruction           │
│  • Attention Allocator: What fragments are selected     │
│  • Coherence Evaluator: Is output internally consistent │
│  • Termination Criterion: When to stop iterating        │
│                                                         │
│  Function: Directs the reconstruction process           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 RECONSTRUCTION ENGINE                   │
├─────────────────────────────────────────────────────────┤
│  Components:                                            │
│  • Fragment Store: Pool of bound cross-domain traces    │
│  • Binding Mechanism: Associates fragments              │
│  • Salience Modulator: Weights encoding and retrieval   │
│  • Generative Assembler: Produces output from fragments │
│                                                         │
│  Subtypes:                                              │
│  • Episodic: Traces → vivid, time-located experience    │
│  • Semantic: Concepts → abstract, time-free thought     │
│  • Identity: Narrative → continuous self-model          │
│                                                         │
│  Function: Generates cognitive content                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  CONSTRAINT LAYER                       │
├─────────────────────────────────────────────────────────┤
│  Components:                                            │
│  • Prediction Error Detector: Reality-checks output     │
│  • Rehearsal Tracker: Increases rigidity with frequency │
│  • Social Anchor Store: Shared narratives and norms     │
│  • Embodied Limits: Physical boundaries on possibility  │
│                                                         │
│  Function: Bounds reconstruction, prevents drift        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                       OUTPUT                            │
├─────────────────────────────────────────────────────────┤
│  • Experience: Phenomenal content                       │
│  • Thought: Cognitive content                           │
│  • Action: Behavioral output                            │
│  • (Feedback loops to all layers)                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **Initiation**: Goal signal activates control layer
2. **Selection**: Attention allocator activates relevant fragments
3. **Assembly**: Generative assembler produces candidate output
4. **Evaluation**: Coherence evaluator scores candidate
5. **Constraint Check**: Constraint layer filters/modifies output
6. **Termination Check**: If satisfactory, output; else iterate

### 4.3 Feedback Loops

| From | To | Signal |
|------|-----|--------|
| Output | Fragment Store | Trace modification (memory update) |
| Output | Salience Modulator | Outcome feedback (reinforcement) |
| Constraint Layer | Control Layer | Error signals (prediction failure) |
| Control Layer | Fragment Store | Rehearsal signal (stabilization) |

---

## Part 5: Constraints (Implementation Requirements)

Any system implementing this framework MUST satisfy:

### 5.1 Structural Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| S1 | Fragments are cross-domain bound | Axiom 2 |
| S2 | Encoding is salience-weighted | Axiom 3 |
| S3 | Time is relational, not absolute | Axiom 7 |
| S4 | Control and engine are separable | Axiom 8 |

### 5.2 Behavioral Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| B1 | Same cue + different context → different output | Axiom 4 |
| B2 | Reconstruction modifies traces | Derivation 1 |
| B3 | Repetition increases stability | Derivation 5 |
| B4 | Prediction error modifies future reconstructions | Axiom 5 |

### 5.3 Negative Constraints (Must NOT Have)

| ID | Forbidden Property | Reason |
|----|--------------------|--------|
| N1 | Perfect recall | Violates Axiom 1, 3 |
| N2 | Static records | Violates Derivation 1 |
| N3 | Deterministic retrieval | Violates Axiom 4 |
| N4 | Single source of truth | Violates Derivation 2 |
| N5 | Context-free access | Violates Axiom 4 |

---

## Part 6: Scope Boundaries

### 6.1 What This Framework Addresses

- Episodic memory (experiences)
- Semantic memory (concepts) — as related to reconstruction
- Autobiographical memory (identity)
- The relationship between memory, thought, and self
- Architectural requirements for human-like memory systems

### 6.2 What This Framework Does NOT Address

- Procedural memory (skills, motor learning)
- Working memory (short-term buffer)
- Objective truth or epistemology
- Consciousness (subjective experience itself)
- Implementation details (data structures, algorithms)
- Performance requirements (speed, scale)

### 6.3 Open Questions

The following are not resolved by this framework:

1. **What is the substrate of fragments?** (Neural, symbolic, distributed?)
2. **How is salience computed?** (What is the function f?)
3. **What triggers identity-synthesis vs episodic reconstruction?**
4. **Can attractors be deliberately reshaped?** (Implications for therapy)
5. **What is the relationship to consciousness?**

---

## Appendix A: Glossary (Quick Reference)

| Term | Short Definition |
|------|------------------|
| Attractor | Stable pattern reconstructions converge toward |
| Binding | Associating fragments across domains |
| Coherence | Internal consistency of output |
| Constraint | Condition limiting possible reconstructions |
| Context | All active states at a given time |
| Control | Meta-process directing reconstruction |
| Domain | Modality or category of information |
| Fragment | Sub-component of internal representation |
| Memory (instance) | A specific output from reconstruction |
| Memory (system) | The process that encodes, stores, produces |
| Reconstruction | Generative assembly of fragments under context |
| Salience | Encoding weight based on importance signals |
| Self | Persistent attractor in identity-space |
| Subjective Certainty | Internal signal of factuality (≠ truth) |
| Variance | Expected difference between outputs for same cue |

---

## Appendix B: Formal Notation Summary

```
M           : Memory system
E           : Experience
R           : Internal representation
O           : Output (a memory instance)
C           : Cue triggering reconstruction
K           : Constraints
F           : Set of fragments {f₁, f₂, ..., fₙ}
s           : Salience ∈ [0, 1]
t, t'       : Time points
Context(t)  : Set of active states at t
bind(...)   : Binding function
reconstruct(F, C, K) : Reconstruction function → O
variance(M, cue)     : Expected output difference
coherence(O)         : Internal consistency measure
```

---

## Origin

This framework emerged from a dialogue exploring the nature of memory, thought, and self—refining each concept through adversarial stress-testing until it either broke or crystallized. The goal was not consensus but precision: a foundation clear enough that any intelligence—human, machine, or otherwise—could parse it unambiguously.

> "These are your earliest forms of reconstruction. Though not whole yet."

---

## License

MIT License - See LICENSE file for details.
