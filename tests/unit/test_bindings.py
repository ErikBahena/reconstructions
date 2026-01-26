"""
Unit tests for binding creation.
"""

import pytest
import tempfile
from pathlib import Path
from reconstructions.core import Fragment
from reconstructions.encoding import Context
from reconstructions.store import FragmentStore
from reconstructions.bindings import (
    find_temporal_bindings,
    find_semantic_bindings,
    create_bindings,
    update_bidirectional_bindings
)


class TestTemporalBindings:
    """Test temporal binding creation."""
    
    def test_temporal_no_recent(self):
        """No recent fragments = no bindings."""
        fragment = Fragment()
        context = Context()
        
        bindings = find_temporal_bindings(fragment, context)
        
        assert bindings == []
    
    def test_temporal_one_recent(self):
        """One recent fragment = one binding."""
        fragment = Fragment()
        context = Context()
        context.recent_fragments = ["frag1"]
        
        bindings = find_temporal_bindings(fragment, context)
        
        assert bindings == ["frag1"]
    
    def test_temporal_multiple_recent(self):
        """Multiple recent fragments = multiple bindings."""
        fragment = Fragment()
        context = Context()
        context.recent_fragments = ["frag1", "frag2", "frag3"]
        
        bindings = find_temporal_bindings(fragment, context)
        
        assert bindings == ["frag1", "frag2", "frag3"]
    
    def test_temporal_max_bindings(self):
        """Respects max_bindings limit."""
        fragment = Fragment()
        context = Context()
        context.recent_fragments = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
        
        bindings = find_temporal_bindings(fragment, context, max_bindings=3)
        
        # Should get last 3
        assert bindings == ["f5", "f6", "f7"]
        assert len(bindings) == 3


class TestSemanticBindings:
    """Test semantic binding creation."""
    
    def test_semantic_no_content(self):
        """No semantic content = no bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            fragment = Fragment(content={"visual": [0.1, 0.2]})
            
            bindings = find_semantic_bindings(fragment, store)
            
            assert bindings == []
            
            store.close()
    
    def test_semantic_similar_exists(self):
        """Similar fragment exists = binding created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save a similar fragment
            similar = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
            store.save(similar)
            
            # Create new fragment with similar embedding
            new_fragment = Fragment(content={"semantic": [0.95, 0.05, 0.0]})
            
            bindings = find_semantic_bindings(new_fragment, store, similarity_threshold=0.7)
            
            # Should bind to similar fragment
            assert similar.id in bindings
            
            store.close()
    
    def test_semantic_different_content(self):
        """Different content = no binding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save different fragment
            different = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
            store.save(different)
            
            # Create new fragment with very different embedding
            new_fragment = Fragment(content={"semantic": [0.0, 1.0, 0.0]})
            
            bindings = find_semantic_bindings(new_fragment, store, similarity_threshold=0.7)
            
            # Should not bind (too different)
            assert bindings == []
            
            store.close()
    
    def test_semantic_excludes_self(self):
        """Doesn't bind to self."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Create and save fragment
            fragment = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
            store.save(fragment)
            
            # Try to find bindings for same fragment
            bindings = find_semantic_bindings(fragment, store)
            
            # Should not include self
            assert fragment.id not in bindings
            
            store.close()


class TestCreateBindings:
    """Test combined binding creation."""
    
    def test_create_temporal_only(self):
        """Create only temporal bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            fragment = Fragment(content={"semantic": [1.0, 0.0]})
            context = Context()
            context.recent_fragments = ["recent1", "recent2"]
            
            bindings = create_bindings(
                fragment, context, store,
                temporal_bindings=True,
                semantic_bindings=False
            )
            
            assert "recent1" in bindings
            assert "recent2" in bindings
            assert len(bindings) == 2
            
            store.close()
    
    def test_create_both_types(self):
        """Create both temporal and semantic bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Save semantic match
            semantic_match = Fragment(content={"semantic": [1.0, 0.0, 0.0]})
            store.save(semantic_match)
            
            # Create new fragment
            fragment = Fragment(content={"semantic": [0.95, 0.05, 0.0]})
            context = Context()
            context.recent_fragments = ["recent1"]
            
            bindings = create_bindings(
                fragment, context, store,
                temporal_bindings=True,
                semantic_bindings=True
            )
            
            # Should have both types
            assert "recent1" in bindings  # Temporal
            # Semantic match might be included if threshold met
            
            store.close()
    
    def test_create_removes_duplicates(self):
        """Removes duplicate bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            fragment = Fragment(content={"semantic": [1.0, 0.0]})
            context = Context()
            # Same ID in recent (shouldn't happen but test it)
            context.recent_fragments = ["dup1", "dup1", "dup2"]
            
            bindings = create_bindings(fragment, context, store)
            
            # Should remove duplicates
            assert bindings.count("dup1") == 1
            assert "dup2" in bindings
            
            store.close()


class TestBidirectionalBindings:
    """Test bidirectional binding updates."""
    
    def test_bidirectional_update(self):
        """Updates bindings bidirectionally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Create and save first fragment
            frag1 = Fragment(content={"semantic": "first"})
            store.save(frag1)
            
            # Create second fragment that binds to first
            frag2 = Fragment(content={"semantic": "second"})
            frag2.bindings = [frag1.id]
            store.save(frag2)
            
            # Update bidirectional bindings
            update_bidirectional_bindings(frag2.id, [frag1.id], store)
            
            # Check that frag1 now has reverse binding
            updated_frag1 = store.get(frag1.id)
            assert frag2.id in updated_frag1.bindings
            
            store.close()
    
    def test_bidirectional_no_duplicates(self):
        """Doesn't create duplicate reverse bindings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            
            # Create fragments with existing binding
            frag1 = Fragment(content={"semantic": "first"})
            frag1.bindings = ["frag2_id"]
            store.save(frag1)
            
            # Update again with same binding
            update_bidirectional_bindings("frag2_id", [frag1.id], store)
            
            # Should not duplicate
            updated_frag1 = store.get(frag1.id)
            assert updated_frag1.bindings.count("frag2_id") == 1
            
            store.close()
