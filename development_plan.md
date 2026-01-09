# Reconstructions: Development Plan

A comprehensive progress tracker for building the Reconstructions memory system.

---

## Overview

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Core Data Structures | ✅ Complete |
| Phase 2 | Encoding System | ⬜ Not Started |
| Phase 3 | Strength & Decay | ⬜ Not Started |
| Phase 4 | Reconstruction Engine | ⬜ Not Started |
| Phase 5 | Constraints System | ⬜ Not Started |
| Phase 6 | Identity Model | ⬜ Not Started |
| Phase 7 | Variance & Certainty | ⬜ Not Started |
| Phase 8 | Main Loop & Goals | ⬜ Not Started |
| Phase 9 | CLI Interface | ⬜ Not Started |
| Phase 10 | Integration & Polish | ⬜ Not Started |
| Phase 11 | Optional LLM Interface | ⬜ Not Started |
| Phase 12 | Custom Models (If Needed) | ⬜ Not Started |

---

## Phase 1: Core Data Structures

### 1.1 Fragment Implementation
- [x] Define `Fragment` dataclass with all fields
- [x] Implement `Fragment.to_dict()` serialization
- [x] Implement `Fragment.from_dict()` deserialization
- [x] Write unit tests for Fragment creation
- [x] Write unit tests for Fragment serialization roundtrip

### 1.2 Strand Implementation
- [ ] Define `Strand` dataclass
- [ ] Implement serialization/deserialization
- [ ] Write unit tests

### 1.3 Query Implementation
- [ ] Define `Query` dataclass with semantic, temporal, domain filters
- [ ] Implement query hashing for variance tracking
- [ ] Write unit tests

### 1.4 Fragment Store
- [x] Design SQLite schema for fragments
- [x] Implement `FragmentStore.save(fragment)`
- [x] Implement `FragmentStore.get(id)`
- [x] Implement `FragmentStore.delete(id)`
- [x] Implement `FragmentStore.find_by_time_range(start, end)`
- [x] Implement `FragmentStore.find_by_domain(domain)`
- [x] Set up vector storage (Chroma or custom)
- [x] Implement `FragmentStore.find_similar_semantic(embedding, top_k)`
- [x] Write unit tests for all store operations
- [x] Write integration test: save → get → verify equality

### 1.5 Phase 1 Testing Cycle
- [x] Run all Phase 1 unit tests
- [x] Fix any failures
- [x] Manual test: create fragments via Python REPL
- [x] Performance test: store/retrieve 1000 fragments
- [x] Document any issues or improvements needed

### 1.6 Phase 1 Review
- [ ] Code review for Phase 1
- [ ] Refactor if needed
- [ ] Update spec.md if implementation diverged
- [ ] Mark Phase 1 complete in overview table

---

## Phase 2: Encoding System

### 2.1 Experience Input
- [ ] Define `Experience` dataclass (text, sensory, emotional, etc.)
- [ ] Define `Context` dataclass (current state, goals, sequence)
- [ ] Write unit tests

### 2.2 Feature Extraction
- [ ] Implement `extract_semantic_features(text)` → embedding
- [ ] Choose/integrate embedding model (e.g., sentence-transformers)
- [ ] Implement `extract_emotional_features(input)` → valence/arousal/dominance
- [ ] Implement `extract_temporal_features(context)` → position, timestamp
- [ ] Write unit tests for each extractor

### 2.3 Salience Calculation
- [ ] Implement `calculate_encoding_salience(emotional, novelty, goal, depth)`
- [ ] Implement `calculate_novelty(content, existing_fragments)`
- [ ] Implement `calculate_goal_relevance(content, active_goals)`
- [ ] Make weights configurable
- [ ] Write unit tests with known inputs/outputs
- [ ] Test edge cases (no existing fragments, no goals, etc.)

### 2.4 Binding Creation
- [ ] Implement `find_temporal_bindings(fragment, context)`
- [ ] Implement `create_binding(fragment_a, fragment_b)`
- [ ] Write unit tests

### 2.5 Full Encode Function
- [ ] Implement `encode(experience, context) → Fragment`
- [ ] Integration test: encode experience → verify fragment fields
- [ ] Integration test: encode multiple experiences → verify bindings

### 2.6 Phase 2 Testing Cycle
- [ ] Run all Phase 2 unit tests
- [ ] Run integration tests
- [ ] Manual test: encode various experiences
- [ ] Verify salience values make sense
- [ ] Verify bindings are created correctly
- [ ] Document issues

### 2.7 Phase 2 Review
- [ ] Code review
- [ ] Refactor salience formula if needed
- [ ] Update spec.md if needed
- [ ] Mark Phase 2 complete

---

## Phase 3: Strength & Decay

### 3.1 Access Logging
- [ ] Implement `FragmentStore.record_access(fragment_id, timestamp)`
- [ ] Update Fragment schema to store access log
- [ ] Write unit tests

### 3.2 Decay Function
- [ ] Implement power law decay formula
- [ ] Make `DECAY_RATE` configurable
- [ ] Write unit tests with known decay curves

### 3.3 Rehearsal Bonus
- [ ] Implement rehearsal contribution to strength
- [ ] Write unit tests

### 3.4 Strength Calculation
- [ ] Implement `calculate_strength(fragment, now) → float`
- [ ] Test: new fragment has high strength
- [ ] Test: old fragment with no access has low strength
- [ ] Test: old fragment with many accesses maintains strength
- [ ] Test: decay curve matches expected power law

### 3.5 Phase 3 Testing Cycle
- [ ] Run all unit tests
- [ ] Simulation test: create fragments, simulate time passage, verify decay
- [ ] Simulation test: access fragments, verify strength increase
- [ ] Plot decay curves to visually verify
- [ ] Document issues

### 3.6 Phase 3 Review
- [ ] Code review
- [ ] Tune decay rate if needed
- [ ] Update spec.md if needed
- [ ] Mark Phase 3 complete

---

## Phase 4: Reconstruction Engine

### 4.1 Spreading Activation
- [ ] Implement `spread_activation(query, store, context) → dict[id, activation]`
- [ ] Test: query matching single fragment
- [ ] Test: activation spreads through bindings
- [ ] Test: activation decays with distance
- [ ] Test: multiple query matches combine

### 4.2 Candidate Selection
- [ ] Implement candidate scoring (activation × strength + noise)
- [ ] Implement top-K selection
- [ ] Make `MAX_FRAGMENTS` configurable
- [ ] Make `variance_target` parameter work
- [ ] Write unit tests

### 4.3 Fragment Assembly
- [ ] Implement `temporal_sort(fragments)`
- [ ] Implement `merge_domain_content(domain, contents)`
- [ ] Implement `assemble_fragments(fragments, context) → Assembly`
- [ ] Write unit tests for each

### 4.4 Gap Filling
- [ ] Design gap detection algorithm
- [ ] Implement `fill_gaps(assembly, context) → Assembly`
- [ ] Start with simple interpolation
- [ ] Write unit tests
- [ ] Document where custom model could help

### 4.5 Full Reconstruct Function
- [ ] Implement `reconstruct(query, context, store, constraints, variance) → Strand`
- [ ] Integration test: encode → reconstruct → verify content
- [ ] Integration test: low variance → consistent output
- [ ] Integration test: high variance → variable output

### 4.6 Phase 4 Testing Cycle
- [ ] Run all unit tests
- [ ] Run integration tests
- [ ] Manual test: encode 10 experiences, query, verify reconstruction
- [ ] Test reconstruction with varying variance targets
- [ ] Measure reconstruction time, optimize if needed
- [ ] Document issues

### 4.7 Phase 4 Review
- [ ] Code review
- [ ] Optimize spreading activation if slow
- [ ] Evaluate gap filling quality
- [ ] Update spec.md if needed
- [ ] Mark Phase 4 complete

---

## Phase 5: Constraints System

### 5.1 Constraint Data Structures
- [ ] Define `Constraint` base class
- [ ] Define `HardConstraint` subclass
- [ ] Define `SoftConstraint` subclass
- [ ] Define `ConstraintSet` container
- [ ] Write unit tests

### 5.2 Coherence Checking
- [ ] Implement `calculate_coherence(assembly) → float`
- [ ] Implement contradiction detection
- [ ] Write unit tests

### 5.3 Built-in Constraints
- [ ] Implement `NoContradictionConstraint`
- [ ] Implement `TemporalConsistencyConstraint`
- [ ] Implement `IdentityConstraint` (placeholder, full impl in Phase 6)
- [ ] Write unit tests for each

### 5.4 Constraint Application
- [ ] Implement `apply_constraints(assembly, constraints, context) → Assembly`
- [ ] Test: hard constraint violation → correction applied
- [ ] Test: soft constraint violation → penalty tracked
- [ ] Test: assembly passes constraints → unchanged

### 5.5 Phase 5 Testing Cycle
- [ ] Run all unit tests
- [ ] Integration test: reconstruct with constraints
- [ ] Create test cases with known contradictions
- [ ] Verify constraints catch and correct issues
- [ ] Document constraint effectiveness

### 5.6 Phase 5 Review
- [ ] Code review
- [ ] Add/remove constraints based on testing
- [ ] Update spec.md if needed
- [ ] Mark Phase 5 complete

---

## Phase 6: Identity Model

### 6.1 Identity Data Structures
- [ ] Implement `Identity` dataclass
- [ ] Implement `BeliefRecord` dataclass
- [ ] Implement `IdentityUpdate` dataclass
- [ ] Implement `Identity.save()` and `Identity.load()`
- [ ] Write unit tests

### 6.2 Identity Constraints
- [ ] Implement trait violation detection
- [ ] Implement belief contradiction detection
- [ ] Implement `constrain_to_identity(assembly, identity, context)`
- [ ] Write unit tests

### 6.3 Identity Update System
- [ ] Implement update validation (inertia)
- [ ] Implement core trait protection (requires multiple consistent updates)
- [ ] Implement belief update with evidence linking
- [ ] Implement state updates (low inertia)
- [ ] Write unit tests for each update type

### 6.4 Reflection/Identity Synthesis
- [ ] Implement `analyze_for_identity_update(strand, identity) → list[updates]`
- [ ] Implement basic pattern detection
- [ ] Write unit tests

### 6.5 Phase 6 Testing Cycle
- [ ] Run all unit tests
- [ ] Integration test: reconstruction respects identity
- [ ] Integration test: repeated experiences update beliefs
- [ ] Integration test: core traits resist change
- [ ] Simulate 100 interactions, verify identity stability
- [ ] Document identity drift (if any)

### 6.6 Phase 6 Review
- [ ] Code review
- [ ] Tune inertia parameters
- [ ] Update spec.md if needed
- [ ] Mark Phase 6 complete

---

## Phase 7: Variance & Certainty

### 7.1 Variance Controller
- [ ] Implement `VarianceController` class
- [ ] Implement reconstruction logging
- [ ] Implement `strand_distance(strand_a, strand_b)`
- [ ] Write unit tests

### 7.2 Variance Calculation
- [ ] Implement `calculate_variance(query_hash, new_strand)`
- [ ] Implement `record_reconstruction(query_hash, strand)`
- [ ] Implement history limiting
- [ ] Write unit tests

### 7.3 Certainty Calculation
- [ ] Implement `get_certainty(query_hash) → float`
- [ ] Test: new query → low certainty
- [ ] Test: repeated query with stable output → high certainty
- [ ] Test: repeated query with variable output → low certainty

### 7.4 Integration with Reconstruction
- [ ] Connect variance controller to reconstruct function
- [ ] Record all reconstructions
- [ ] Include certainty in Result
- [ ] Integration tests

### 7.5 Phase 7 Testing Cycle
- [ ] Run all unit tests
- [ ] Integration test: variance decreases with rehearsal
- [ ] Visualization: plot certainty over repeated queries
- [ ] Document variance behavior

### 7.6 Phase 7 Review
- [ ] Code review
- [ ] Verify certainty matches intuition
- [ ] Update spec.md if needed
- [ ] Mark Phase 7 complete

---

## Phase 8: Main Loop & Goals

### 8.1 Goal System
- [ ] Implement `Goal` dataclass
- [ ] Implement `GoalQueue` with priority sorting
- [ ] Implement maintenance goal generation
- [ ] Implement idle goal generation
- [ ] Write unit tests

### 8.2 Result Type
- [ ] Define `Result` dataclass
- [ ] Implement result types: strand, encoded, reflected, maintained
- [ ] Write unit tests

### 8.3 Main Engine
- [ ] Implement `ReconstructionEngine` class
- [ ] Implement `process_goal(goal) → Result`
- [ ] Implement `process_query(goal)`
- [ ] Implement `process_encode(goal)`
- [ ] Implement `process_reflect(goal)`
- [ ] Implement `process_maintenance(goal)`
- [ ] Write unit tests for each processor

### 8.4 Event Loop
- [ ] Implement `run()` main loop
- [ ] Implement graceful shutdown
- [ ] Test: loop processes goals in priority order
- [ ] Test: maintenance runs periodically
- [ ] Test: idle reflection occurs when queue empty

### 8.5 Phase 8 Testing Cycle
- [ ] Run all unit tests
- [ ] Integration test: add goals, verify processing order
- [ ] Run loop for 5 minutes, verify stability
- [ ] Monitor memory usage over time
- [ ] Document issues

### 8.6 Phase 8 Review
- [ ] Code review
- [ ] Tune maintenance interval
- [ ] Update spec.md if needed
- [ ] Mark Phase 8 complete

---

## Phase 9: CLI Interface

### 9.1 CLI Implementation
- [ ] Implement `CLIInterface` class
- [ ] Implement `input_to_goal(raw_input) → Goal`
- [ ] Implement `result_to_output(result) → str`
- [ ] Handle commands: `/remember`, `/store`, `/reflect`, `/identity`, `/exit`
- [ ] Write unit tests

### 9.2 CLI Session
- [ ] Implement REPL loop
- [ ] Implement command history
- [ ] Implement help command
- [ ] Handle errors gracefully

### 9.3 Output Formatting
- [ ] Format strands for readability
- [ ] Show certainty levels
- [ ] Show fragment sources
- [ ] Show binding information (optional flag)

### 9.4 Phase 9 Testing Cycle
- [ ] Manual test all commands
- [ ] Test error cases (invalid commands, empty store)
- [ ] User test: have someone else try the CLI
- [ ] Document UX issues

### 9.5 Phase 9 Review
- [ ] Improve UX based on feedback
- [ ] Update help text
- [ ] Mark Phase 9 complete

---

## Phase 10: Integration & Polish

### 10.1 Full System Integration
- [ ] Create main entry point (`main.py`)
- [ ] Implement configuration file loading
- [ ] Implement data directory setup
- [ ] Test cold start (empty database)
- [ ] Test warm start (existing database)

### 10.2 End-to-End Tests
- [ ] Test: encode 50 experiences → query → verify reconstruction
- [ ] Test: multi-session persistence (restart between sessions)
- [ ] Test: identity evolves appropriately over 100 interactions
- [ ] Test: high-salience memories are more accessible
- [ ] Test: decay works as expected over simulated time

### 10.3 Performance Testing
- [ ] Benchmark encoding speed
- [ ] Benchmark reconstruction speed
- [ ] Benchmark with 1000 fragments
- [ ] Benchmark with 10000 fragments
- [ ] Identify and fix bottlenecks

### 10.4 Documentation
- [ ] Write README.md usage section
- [ ] Document configuration options
- [ ] Document CLI commands
- [ ] Create example usage scripts

### 10.5 Phase 10 Review
- [ ] Full code review
- [ ] Address all TODO comments
- [ ] Clean up dead code
- [ ] Mark Phase 10 complete

---

## Phase 11: Optional LLM Interface

### 11.1 LLM Interface Implementation
- [ ] Implement `LLMInterface` class
- [ ] Implement natural language → Goal parsing
- [ ] Implement Result → natural language rendering
- [ ] Choose and integrate LLM (e.g., Llama 3.2 3B)
- [ ] Write unit tests (mock LLM)

### 11.2 Prompt Engineering
- [ ] Design input parsing prompt
- [ ] Design output rendering prompt
- [ ] Test and iterate on prompts
- [ ] Document final prompts

### 11.3 Integration
- [ ] Make interface configurable (CLI vs LLM)
- [ ] Test switching between interfaces
- [ ] Compare output quality

### 11.4 Phase 11 Testing Cycle
- [ ] Test natural language queries
- [ ] Test conversation flow
- [ ] User test: natural language usability
- [ ] Compare to CLI for accuracy

### 11.5 Phase 11 Review
- [ ] Evaluate LLM necessity
- [ ] Document when to use CLI vs LLM
- [ ] Mark Phase 11 complete

---

## Phase 12: Custom Models (If Needed)

### 12.1 Evaluation
- [ ] Evaluate gap filling quality from Phase 4
- [ ] Evaluate binding prediction accuracy
- [ ] Decide if custom models are needed
- [ ] Document decision rationale

### 12.2 Gap Filler Model (If Needed)
- [ ] Define model architecture (~100M params)
- [ ] Create training data from complete sequences
- [ ] Train model
- [ ] Evaluate improvement
- [ ] Integrate into gap_fill function

### 12.3 Binding Predictor Model (If Needed)
- [ ] Define model architecture (~10M params)
- [ ] Create training data from access patterns
- [ ] Train model
- [ ] Evaluate improvement
- [ ] Integrate into binding prediction

### 12.4 Phase 12 Review
- [ ] Evaluate custom model impact
- [ ] Document performance improvements
- [ ] Update spec.md with final architecture
- [ ] Mark Phase 12 complete

---

## Post-Development

### Deployment
- [ ] Create installation script
- [ ] Test on fresh machine
- [ ] Create Docker container (optional)
- [ ] Document deployment steps

### Future Improvements (Backlog)
- [ ] API interface for programmatic access
- [ ] GUI interface
- [ ] Multi-agent memory sharing
- [ ] Memory import/export
- [ ] Visualization tools

---

## Progress Tracking

| Date | Phase | Task | Notes |
|------|-------|------|-------|
| | | | |

---

## Legend

- ⬜ Not Started
- 🟨 In Progress
- ✅ Complete
- ❌ Blocked
- [ ] Task not done
- [x] Task complete
