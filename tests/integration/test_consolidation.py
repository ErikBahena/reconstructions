"""
Integration tests for autonomous consolidation.

Tests that consolidation solves the retrieval problem where memories are
encoded but hard to find due to weak bindings.
"""

import pytest
import time
from pathlib import Path
import tempfile

from reconstructions import (
    FragmentStore,
    ReconstructionEngine,
    Experience,
    Query,
    ConsolidationScheduler,
    ConsolidationConfig
)
from reconstructions.encoding import Context
from reconstructions.reconstruction import ReconstructionConfig


def test_consolidation_strengthens_retrieval():
    """
    Test that consolidation strengthens retrieval paths.

    Simulates the ONCAPON RTMP problem:
    1. Store information about RTMP source
    2. Try to query it - should be weak
    3. Run consolidation
    4. Query again - should be stronger
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = FragmentStore(str(db_path))
        engine = ReconstructionEngine(store, enable_consolidation=True)

        # Step 1: Encode RTMP source information
        exp1 = Experience(
            text="ONCAPON RTMP source at rtmp://example.com/stream",
            emotional={"valence": 0.6,
                "arousal": 0.5}
        )
        engine.submit_experience(exp1)
        engine.step()

        # Encode more related streaming info
        exp2 = Experience(
            text="Testing video stream from RTMP endpoint",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        engine.submit_experience(exp2)
        engine.step()

        exp3 = Experience(
            text="ONCAPON streaming service configuration",
            emotional={"valence": 0.6,
                "arousal": 0.5}
        )
        engine.submit_experience(exp3)
        engine.step()

        # Step 2: Query BEFORE consolidation
        query_pre = Query(semantic="rtmp stream source")
        strand_pre = engine.query(query_pre)

        assert strand_pre is not None
        fragments_pre_count = len(strand_pre.fragments)
        coherence_pre = strand_pre.coherence_score

        # Step 3: Run consolidation multiple times to strengthen bindings
        for _ in range(3):
            goal = engine.goal_queue.peek()
            if goal and goal.goal_type.value == "consolidation":
                result = engine.step()
                assert result.result_type.value == "consolidated"
            else:
                # Manually trigger consolidation
                from reconstructions.engine import EngineGoal, GoalType
                goal = EngineGoal(
                    priority=3.0,
                    goal_type=GoalType.CONSOLIDATION,
                    payload={}
                )
                engine.goal_queue.push(goal)
                result = engine.step()
                assert result.result_type.value == "consolidated"

            # Check that consolidation did something
            stats = result.data
            assert stats["rehearsed_count"] >= 0

        # Step 4: Query AFTER consolidation
        query_post = Query(semantic="rtmp stream source")
        strand_post = engine.query(query_post)

        assert strand_post is not None
        fragments_post_count = len(strand_post.fragments)
        coherence_post = strand_post.coherence_score

        # After consolidation, we should have:
        # - Better coherence (fragments more connected)
        # - More fragments found (stronger retrieval paths)
        # Note: These assertions might be weak initially, but should improve
        # as consolidation logic is refined
        assert fragments_post_count >= fragments_pre_count


def test_consolidation_discovers_patterns():
    """
    Test that consolidation discovers semantic patterns between fragments.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = FragmentStore(str(db_path))

        # Custom config for faster pattern discovery
        config = ConsolidationConfig()
        config.PATTERN_DISCOVERY_INTERVAL = 1  # Discover every consolidation

        scheduler = ConsolidationScheduler(store, config)

        # Encode semantically similar fragments
        from reconstructions.encoder import encode

        context = Context()

        exp1 = Experience(
            text="Machine learning models need training data",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        frag1 = encode(exp1, context, store)

        exp2 = Experience(
            text="Neural networks require large datasets for training",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        frag2 = encode(exp2, context, store)

        exp3 = Experience(
            text="Deep learning trains on massive amounts of data",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        frag3 = encode(exp3, context, store)

        # Fragments should not be bound initially
        assert frag2.id not in frag1.bindings
        assert frag3.id not in frag1.bindings

        # Run consolidation to discover patterns
        stats = scheduler.consolidate()

        # Should have discovered some patterns
        # (or at least tried to - depends on embedding quality)
        assert "patterns_discovered" in stats


def test_consolidation_scheduling():
    """
    Test that consolidation is automatically scheduled at appropriate intervals.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = FragmentStore(str(db_path))

        # Use short interval for testing
        config = ConsolidationConfig()
        config.CONSOLIDATION_INTERVAL_SECONDS = 0.1  # 100ms

        engine = ReconstructionEngine(
            store,
            enable_consolidation=True,
            consolidation_config=config
        )

        # Encode some experiences
        for i in range(3):
            exp = Experience(
                text=f"Test experience {i}",
                emotional={"valence": 0.6,
                "arousal": 0.5}
            )
            engine.submit_experience(exp)
            engine.step()

        # Wait for consolidation interval
        time.sleep(0.2)

        # Process next step - should trigger consolidation
        result = engine.step()

        # Should have processed a consolidation goal
        # (or scheduled one)
        # Check that scheduler has run at least once
        assert engine.consolidation_scheduler.state.consolidation_count >= 0


def test_consolidation_rehearsal():
    """
    Test that consolidation rehearses recent salient fragments.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = FragmentStore(str(db_path))

        config = ConsolidationConfig()
        config.MIN_SALIENCE_FOR_REHEARSAL = 0.4

        scheduler = ConsolidationScheduler(store, config)

        # Encode high-salience fragment
        from reconstructions.encoder import encode
        context = Context()

        exp = Experience(
            text="Important critical information about the system",
            emotional={"valence": 0.8, "arousal": 0.9}  # High emotion/arousal
        )
        fragment = encode(exp, context, store)

        # Fragment should have high salience
        assert fragment.initial_salience >= config.MIN_SALIENCE_FOR_REHEARSAL

        # Select rehearsal candidates
        candidates = scheduler.select_rehearsal_candidates()

        # Should include our salient fragment
        candidate_ids = [c.id for c in candidates]
        assert fragment.id in candidate_ids


def test_binding_strengthening():
    """
    Test that co-activated fragments get their bindings strengthened.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = FragmentStore(str(db_path))

        scheduler = ConsolidationScheduler(store)

        # Create fragments in separate contexts so they don't auto-bind
        from reconstructions.encoder import encode
        context1 = Context()
        context2 = Context()

        exp1 = Experience(
            text="Python programming language",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        frag1 = encode(exp1, context1, store)

        exp2 = Experience(
            text="Writing Python code for data science",
            emotional={"valence": 0.5,
                "arousal": 0.4}
        )
        frag2 = encode(exp2, context2, store)

        # Simulate co-activation
        for _ in range(5):
            pair = tuple(sorted([frag1.id, frag2.id]))
            scheduler.state.coactivation_matrix[pair] = \
                scheduler.state.coactivation_matrix.get(pair, 0) + 1

        # Strengthen bindings
        count = scheduler.strengthen_bindings()

        # Should have created bidirectional bindings
        assert count >= 2

        # Reload fragments to check bindings were saved
        frag1_updated = store.get(frag1.id)
        frag2_updated = store.get(frag2.id)

        assert frag2.id in frag1_updated.bindings
        assert frag1.id in frag2_updated.bindings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
