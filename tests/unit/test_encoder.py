"""
Integration tests for the complete encoding pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from reconstructions.encoding import Experience, Context
from reconstructions.store import FragmentStore
from reconstructions.encoder import encode, encode_batch


class TestEncodeBasic:
    """Test basic encoding functionality."""
    
    def test_encode_text_experience(self):
        """Test encoding a simple text experience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            exp = Experience(text="The sky is blue today")
            
            fragment = encode(exp, context, store)
            
            # Verify fragment was created
            assert fragment is not None
            assert fragment.id is not None
            
            # Verify content was extracted
            assert "semantic" in fragment.content
            assert "emotional" in fragment.content
            assert "temporal" in fragment.content
            
            # Verify salience was calculated
            assert 0.0 <= fragment.initial_salience <= 1.0
            
            # Verify fragment was saved
            retrieved = store.get(fragment.id)
            assert retrieved is not None
            
            store.close()
    
    def test_encode_multi_modal_experience(self):
        """Test encoding a multi-modal experience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            exp = Experience(
                text="I see a beautiful sunset",
                sensory={"visual": [0.9, 0.5, 0.3]},
                emotional={"valence": 0.8, "arousal": 0.4},
                tags=["sunset", "beautiful"]
            )
            
            fragment = encode(exp, context, store)
            
            # Verify all modalities preserved
            assert "semantic" in fragment.content
            assert "visual" in fragment.content
            assert "emotional" in fragment.content
            assert fragment.tags == ["sunset", "beautiful"]
            
            # Verify emotional content
            assert fragment.content["emotional"]["valence"] == 0.8
            
            store.close()
    
    def test_encode_updates_context(self):
        """Test that encoding updates the context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            initial_sequence = context.sequence_counter
            initial_recent_count = len(context.recent_fragments)
            
            exp = Experience(text="Test experience")
            fragment = encode(exp, context, store)
            
            # Context should be updated
            assert context.sequence_counter == initial_sequence + 1
            assert len(context.recent_fragments) == initial_recent_count + 1
            assert fragment.id in context.recent_fragments
            
            store.close()


class TestEncodeBindings:
    """Test binding creation during encoding."""
    
    def test_encode_creates_temporal_bindings(self):
        """Test that temporal bindings are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode first experience
            exp1 = Experience(text="First memory")
            frag1 = encode(exp1, context, store)
            
            # Encode second experience
            exp2 = Experience(text="Second memory")
            frag2 = encode(exp2, context, store)
            
            # Second fragment should bind to first
            assert frag1.id in frag2.bindings
            
            # Bidirectional binding should exist
            updated_frag1 = store.get(frag1.id)
            assert frag2.id in updated_frag1.bindings
            
            store.close()
    
    def test_encode_sequence_creates_chain(self):
        """Test that a sequence creates a binding chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode sequence of experiences
            fragments = []
            for i in range(5):
                exp = Experience(text=f"Memory {i}")
                frag = encode(exp, context, store)
                fragments.append(frag)
            
            # Verify chain
            for i in range(1, 5):
                # Each fragment should bind to previous
                prev_id = fragments[i-1].id
                assert prev_id in fragments[i].bindings
            
            store.close()


class TestEncodeSalience:
    """Test salience calculation during encoding."""
    
    def test_high_emotion_high_salience(self):
        """High emotion should produce higher salience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Low emotion experience
            exp_low = Experience(
                text="Neutral observation",
                emotional={"valence": 0.5, "arousal": 0.2}
            )
            frag_low = encode(exp_low, context, store)
            
            # High emotion experience
            exp_high = Experience(
                text="Amazing discovery!",
                emotional={"valence": 0.9, "arousal": 0.9}
            )
            frag_high = encode(exp_high, context, store)
            
            # High emotion should have higher salience
            assert frag_high.initial_salience > frag_low.initial_salience
            
            store.close()
    
    def test_novel_higher_salience(self):
        """Novel experiences should have higher salience than similar ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # First experience (novel)
            exp1 = Experience(text="Unique first experience")
            frag1 = encode(exp1, context, store)
            
            # Very similar experience (not novel)
            exp2 = Experience(text="Unique first experience")
            frag2 = encode(exp2, context, store)
            
            # First should have higher novelty component
            # (though overall salience might vary due to other factors)
            assert frag1.initial_salience >= frag2.initial_salience
            
            store.close()


class TestEncodeBatch:
    """Test batch encoding."""
    
    def test_encode_batch_basic(self):
        """Test encoding a batch of experiences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            experiences = [
                Experience(text="First memory"),
                Experience(text="Second memory"),
                Experience(text="Third memory")
            ]
            
            fragments = encode_batch(experiences, context, store)
            
            assert len(fragments) == 3
            
            # All should be saved
            for frag in fragments:
                retrieved = store.get(frag.id)
                assert retrieved is not None
            
            # Should form a chain
            assert fragments[0].id in fragments[1].bindings
            assert fragments[1].id in fragments[2].bindings
            
            store.close()
    
    def test_encode_batch_updates_context(self):
        """Test that batch encoding updates context correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            initial_sequence = context.sequence_counter
            
            experiences = [
                Experience(text=f"Memory {i}")
                for i in range(5)
            ]
            
            fragments = encode_batch(experiences, context, store)
            
            # Sequence should increment for each
            assert context.sequence_counter == initial_sequence + 5
            
            # Recent fragments should contain all (up to limit)
            for frag in fragments:
                assert frag.id in context.recent_fragments
            
            store.close()


class TestEncodeIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete encoding pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context(
                active_goals=["learn", "test"],
                processing_depth=0.8
            )
            
            exp = Experience(
                text="I am learning about memory systems",
                emotional={"valence": 0.7, "arousal": 0.6},
                sensory={"visual": [0.1, 0.2, 0.3]},
                tags=["learning", "memory"]
            )
            
            fragment = encode(exp, context, store)
            
            # Verify complete fragment
            assert fragment.id is not None
            assert "semantic" in fragment.content
            assert "emotional" in fragment.content
            assert "visual" in fragment.content
            assert "temporal" in fragment.content
            assert fragment.initial_salience > 0.0
            assert fragment.tags == ["learning", "memory"]
            
            # Verify stored correctly
            retrieved = store.get(fragment.id)
            assert retrieved.content == fragment.content
            assert retrieved.initial_salience == fragment.initial_salience
            
            # Verify context updated
            assert fragment.id in context.recent_fragments
            assert context.sequence_counter == 1
            
            store.close()
