"""
End-to-end integration tests.
"""

import pytest
import tempfile
import time
from pathlib import Path
from reconstructions.core import Query
from reconstructions.store import FragmentStore
from reconstructions.encoding import Experience, Context
from reconstructions.encoder import encode, encode_batch
from reconstructions.reconstruction import reconstruct
from reconstructions.engine import ReconstructionEngine, ResultType
from reconstructions.certainty import VarianceController
from reconstructions.main import Config, setup_data_directory


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_encode_and_reconstruct_cycle(self):
        """Full encode → reconstruct cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            vc = VarianceController()
            
            # Encode multiple experiences
            experiences = [
                Experience(text="I went to the park this morning"),
                Experience(text="The weather was beautiful and sunny"),
                Experience(text="I saw children playing on the swings"),
                Experience(text="A dog ran by chasing a ball"),
                Experience(text="I sat on a bench and read a book")
            ]
            
            fragments = encode_batch(experiences, context, store)
            assert len(fragments) == 5
            
            # Query related content
            query = Query(semantic="park morning")
            strand = reconstruct(query, store, variance_controller=vc)
            
            assert strand is not None
            assert len(strand.fragments) > 0
            assert strand.coherence_score >= 0
            
            store.close()
    
    def test_persistence_across_sessions(self):
        """Data persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            
            # Session 1: Store memories
            store1 = FragmentStore(db_path)
            context1 = Context()
            
            exp = Experience(text="Important memory to persist")
            frag = encode(exp, context1, store1)
            frag_id = frag.id
            original_created_at = frag.created_at
            
            store1.close()
            
            # Session 2: Retrieve memories
            store2 = FragmentStore(db_path)
            
            retrieved = store2.get(frag_id)
            assert retrieved is not None
            # Verify core fragment properties were persisted
            assert retrieved.id == frag_id
            assert retrieved.created_at == original_created_at
            assert retrieved.initial_salience > 0
            
            store2.close()
    
    def test_high_salience_more_accessible(self):
        """High salience memories are reconstructed more readily."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            
            # Encode with varying emotional intensity
            for i in range(10):
                # Low salience memories
                exp_low = Experience(
                    text=f"Routine event {i}",
                    emotional={"valence": 0.5, "arousal": 0.1}
                )
                encode(exp_low, context, store)
            
            # High salience memory
            exp_high = Experience(
                text="Exciting important event happened!",
                emotional={"valence": 0.9, "arousal": 0.9}
            )
            high_frag = encode(exp_high, context, store)
            
            # Query - high salience should be included
            query = Query(semantic="event")
            strand = reconstruct(query, store, variance_target=0.0)
            
            # High salience fragment should be in result
            assert high_frag.id in strand.fragments
            
            store.close()
    
    def test_engine_full_workflow(self):
        """Engine handles complete workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            # Submit experiences
            texts = [
                "Learning about machine learning today",
                "Neural networks are fascinating",
                "Deep learning uses multiple layers"
            ]
            
            for text in texts:
                engine.submit_experience(Experience(text=text), priority=1.0)
            
            # Process all encodes
            encode_results = engine.run()
            assert len(encode_results) == 3
            assert all(r.result_type == ResultType.ENCODED for r in encode_results)
            
            # Submit query
            engine.submit_query(Query(semantic="machine learning"))
            query_results = engine.run()
            
            assert len(query_results) == 1
            assert query_results[0].result_type == ResultType.STRAND
            
            store.close()
    
    def test_certainty_builds_over_queries(self):
        """Certainty increases with repeated consistent queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            context = Context()
            vc = VarianceController()
            
            # Encode a specific memory
            exp = Experience(text="Unique specific memory content")
            encode(exp, context, store)
            
            query = Query(semantic="unique specific")
            
            # First query - low certainty
            s1 = reconstruct(query, store, variance_target=0.0, variance_controller=vc)
            
            # Second query - should have higher certainty
            s2 = reconstruct(query, store, variance_target=0.0, variance_controller=vc)
            
            # Certainty should be 1.0 for identical reconstructions
            assert s2.certainty == 1.0
            
            store.close()


class TestConfig:
    """Test configuration system."""
    
    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = Config()
        
        assert config.decay_rate == 0.5
        assert config.max_fragments == 10
        assert config.show_welcome is True
    
    def test_config_save_load(self):
        """Config saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                data_dir=Path(tmpdir),
                decay_rate=0.7,
                max_fragments=20
            )
            
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)
            
            loaded = Config.load(config_path)
            
            assert loaded.decay_rate == 0.7
            assert loaded.max_fragments == 20
    
    def test_setup_data_directory(self):
        """Data directory creation works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "new_subdir" / "data"
            config = Config(data_dir=data_dir)
            
            setup_data_directory(config)
            
            assert data_dir.exists()
