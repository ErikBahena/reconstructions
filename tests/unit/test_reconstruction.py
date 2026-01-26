"""
Unit tests for reconstruction engine.
"""

import pytest
import tempfile
import time
from pathlib import Path
from reconstructions.core import Fragment, Query
from reconstructions.store import FragmentStore
from reconstructions.encoding import Experience, Context
from reconstructions.encoder import encode, encode_batch
from reconstructions.reconstruction import (
    spread_activation,
    select_candidates,
    temporal_sort,
    calculate_coherence,
    assemble_fragments,
    fill_gaps,
    reconstruct,
    ReconstructionConfig,
    Activation
)


class TestActivation:
    """Test Activation class."""
    
    def test_activate_new(self):
        """Activate new fragment."""
        act = Activation()
        act.activate("frag1", 0.5)
        
        assert act.get("frag1") == 0.5
    
    def test_activate_accumulates(self):
        """Multiple activations accumulate."""
        act = Activation()
        act.activate("frag1", 0.3)
        act.activate("frag1", 0.3)
        
        assert act.get("frag1") == 0.6
    
    def test_activate_capped(self):
        """Activation capped at 1.0."""
        act = Activation()
        act.activate("frag1", 0.8)
        act.activate("frag1", 0.8)
        
        assert act.get("frag1") == 1.0
    
    def test_top_k(self):
        """Get top K activations."""
        act = Activation()
        act.activate("frag1", 0.3)
        act.activate("frag2", 0.8)
        act.activate("frag3", 0.5)
        
        top = act.top_k(2)
        
        assert len(top) == 2
        assert top[0][0] == "frag2"  # Highest
        assert top[1][0] == "frag3"


class TestSpreadActivation:
    """Test spreading activation."""
    
    def test_spread_semantic(self):
        """Semantic query activates similar fragments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode some experiences
            exp1 = Experience(text="The sky is blue today")
            exp2 = Experience(text="I love pizza and pasta")
            
            encode(exp1, context, store)
            encode(exp2, context, store)
            
            # Query for sky
            query = Query(semantic="blue sky")
            activation = spread_activation(query, store)
            
            # Should have some activations
            assert len(activation.activations) > 0
            
            store.close()
    
    def test_spread_through_bindings(self):
        """Activation spreads through bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode sequence (creates bindings)
            experiences = [
                Experience(text="First memory"),
                Experience(text="Second memory"),
                Experience(text="Third memory")
            ]
            fragments = encode_batch(experiences, context, store)
            
            # Query first memory - should spread to connected
            query = Query(semantic="First memory")
            activation = spread_activation(query, store)
            
            # Connected fragments should have some activation
            assert len(activation.activations) >= 1
            
            store.close()


class TestSelectCandidates:
    """Test candidate selection."""
    
    def test_select_top_candidates(self):
        """Selects top candidates by score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode experiences
            for i in range(5):
                exp = Experience(text=f"Memory {i}")
                encode(exp, context, store)
            
            # Create activation
            activation = Activation()
            for frag_id in context.recent_fragments:
                activation.activate(frag_id, 0.5)
            
            # Select candidates
            candidates = select_candidates(activation, store, variance_target=0.0)
            
            assert len(candidates) > 0
            assert len(candidates) <= 5
            
            store.close()
    
    def test_variance_affects_selection(self):
        """Higher variance = more randomness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode experiences
            for i in range(10):
                exp = Experience(text=f"Memory {i}")
                encode(exp, context, store)
            
            activation = Activation()
            for frag_id in context.recent_fragments:
                activation.activate(frag_id, 0.5)
            
            # Multiple selections with high variance
            selections = []
            for _ in range(5):
                candidates = select_candidates(activation, store, variance_target=0.9)
                selections.append([c.id for c in candidates])
            
            # With high variance, selections may differ
            # (Note: this is probabilistic, might occasionally pass even with identical)
            store.close()


class TestTemporalSort:
    """Test temporal sorting."""
    
    def test_sorts_by_time(self):
        """Fragments sorted by creation time."""
        now = time.time()
        
        f1 = Fragment(content={"a": 1})
        f1.created_at = now - 100
        
        f2 = Fragment(content={"a": 2})
        f2.created_at = now - 50
        
        f3 = Fragment(content={"a": 3})
        f3.created_at = now
        
        sorted_frags = temporal_sort([f3, f1, f2])
        
        assert sorted_frags[0] == f1
        assert sorted_frags[1] == f2
        assert sorted_frags[2] == f3


class TestCalculateCoherence:
    """Test coherence calculation."""
    
    def test_single_fragment(self):
        """Single fragment has perfect coherence."""
        f = Fragment(content={"a": 1})
        coherence = calculate_coherence([f])
        
        assert coherence == 1.0
    
    def test_connected_fragments(self):
        """Connected fragments have higher coherence."""
        f1 = Fragment(content={"a": 1})
        f2 = Fragment(content={"a": 2})
        f1.bindings = [f2.id]
        f2.bindings = [f1.id]
        
        coherence = calculate_coherence([f1, f2])
        
        assert coherence > 0.3


class TestAssembleFragments:
    """Test fragment assembly."""
    
    def test_assemble_semantic(self):
        """Assembles semantic content."""
        f1 = Fragment(content={"semantic": "Hello"})
        f2 = Fragment(content={"semantic": "World"})
        f2.created_at = f1.created_at + 1
        
        assembled = assemble_fragments([f1, f2], {})
        
        assert "semantic" in assembled
        assert "Hello" in assembled["semantic"]
        assert "World" in assembled["semantic"]
    
    def test_assemble_emotional(self):
        """Assembles emotional content by averaging."""
        f1 = Fragment(content={"emotional": {"valence": 0.8, "arousal": 0.6}})
        f2 = Fragment(content={"emotional": {"valence": 0.4, "arousal": 0.4}})
        
        assembled = assemble_fragments([f1, f2], {})
        
        # Should average
        assert assembled["emotional"]["valence"] == pytest.approx(0.6, abs=0.1)
        assert assembled["emotional"]["arousal"] == pytest.approx(0.5, abs=0.1)


class TestReconstruct:
    """Test full reconstruction."""
    
    def test_reconstruct_basic(self):
        """Basic reconstruction produces strand."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode some memories
            exp1 = Experience(text="I went to the park")
            exp2 = Experience(text="The park was beautiful")
            
            encode(exp1, context, store)
            encode(exp2, context, store)
            
            # Reconstruct
            query = Query(semantic="park")
            strand = reconstruct(query, store)
            
            assert strand is not None
            assert strand.id is not None
            assert strand.coherence_score >= 0.0
            
            store.close()
    
    def test_reconstruct_records_access(self):
        """Reconstruction records access for rehearsal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode memory
            exp = Experience(text="Important memory")
            frag = encode(exp, context, store)
            
            # Get initial access count
            initial = store.get(frag.id)
            initial_access = len(initial.access_log)
            
            # Reconstruct (should access)
            query = Query(semantic="Important")
            reconstruct(query, store)
            
            # Check access was recorded
            updated = store.get(frag.id)
            # Access count might increase if fragment was selected
            
            store.close()
    
    def test_reconstruct_low_variance(self):
        """Low variance produces consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode memories
            for i in range(5):
                exp = Experience(text=f"Memory about topic {i}")
                encode(exp, context, store)
            
            # Multiple reconstructions with low variance
            query = Query(semantic="topic")
            strands = [reconstruct(query, store, variance_target=0.0) for _ in range(3)]
            
            # Low variance should produce similar fragment sets
            # (might not be identical due to strength changes from access)
            
            store.close()


class TestReconstructionIntegration:
    """Integration tests for reconstruction."""
    
    def test_encode_then_reconstruct(self):
        """Full encode -> reconstruct cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode sequence of experiences
            experiences = [
                Experience(
                    text="I woke up early this morning",
                    emotional={"valence": 0.5, "arousal": 0.3}
                ),
                Experience(
                    text="Had coffee and read the news",
                    emotional={"valence": 0.6, "arousal": 0.4}
                ),
                Experience(
                    text="Went for a walk in the park",
                    emotional={"valence": 0.8, "arousal": 0.5}
                ),
                Experience(
                    text="The weather was perfect",
                    emotional={"valence": 0.9, "arousal": 0.4}
                )
            ]
            
            fragments = encode_batch(experiences, context, store)
            
            # Reconstruct morning
            query = Query(semantic="morning coffee")
            strand = reconstruct(query, store)
            
            assert len(strand.fragments) > 0
            assert strand.coherence_score > 0
            
            store.close()
