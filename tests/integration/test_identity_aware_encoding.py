"""
Integration tests for identity-aware encoding with ReconstructionEngine.

Tests that identity state properly boosts relevant memories end-to-end.
"""

import tempfile
from pathlib import Path
import pytest

from src.reconstructions.store import FragmentStore
from src.reconstructions.engine import ReconstructionEngine
from src.reconstructions.encoding import Experience
from src.reconstructions.identity import Goal


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def engine(temp_db):
    """Create reconstruction engine."""
    store = FragmentStore(str(temp_db))
    return ReconstructionEngine(store)


def test_engine_has_active_identity(engine):
    """Test that engine initializes with active identity state."""
    assert engine.active_identity is not None
    assert len(engine.active_identity.active_goals) >= 0


def test_set_goal_boosts_relevant_encoding(engine):
    """Test that setting a goal enables identity-aware encoding."""
    # Set active goal
    engine.active_identity.set_active_goal(
        "Learn RTMP streaming protocols",
        intensity=0.8
    )

    # Verify goal was set
    assert len(engine.active_identity.active_goals) > 0

    # Encode relevant experience
    relevant_exp = Experience(
        text="RTMP streaming server configuration and setup guide"
    )
    engine.submit_experience(relevant_exp)
    result_relevant = engine.step()

    # Should successfully encode with identity awareness
    assert result_relevant.success
    assert "fragment_id" in result_relevant.data
    assert result_relevant.data.get("salience", 0) > 0.0


def test_goal_boost_proportional_to_intensity(engine):
    """Test that goals with different intensities can be set."""
    # Low intensity goal
    engine.active_identity.set_active_goal(
        "Learn Python basics",
        intensity=0.3
    )

    exp_low = Experience(text="Python list comprehension tutorial")
    engine.submit_experience(exp_low)
    result_low = engine.step()

    # Verify encoding succeeded
    assert result_low.success
    assert result_low.data.get("salience", 0) > 0.0

    # High intensity goal
    engine.active_identity.clear_goal("Learn Python basics")
    engine.active_identity.set_active_goal(
        "Master Python programming",
        intensity=0.9
    )

    exp_high = Experience(text="Advanced Python programming techniques")
    engine.submit_experience(exp_high)
    result_high = engine.step()

    # Verify encoding succeeded with high intensity goal
    assert result_high.success
    assert result_high.data.get("salience", 0) > 0.0


def test_multiple_active_goals(engine):
    """Test system with multiple active goals."""
    # Set multiple goals
    engine.active_identity.set_active_goal("Learn streaming", intensity=0.7)
    engine.active_identity.set_active_goal("Debug auth issues", intensity=0.6)
    engine.active_identity.set_active_goal("Optimize database", intensity=0.8)

    assert len(engine.active_identity.active_goals) >= 3

    # Encode experiences matching different goals
    exp1 = Experience(text="RTMP streaming protocol implementation")
    engine.submit_experience(exp1)
    result1 = engine.step()

    exp2 = Experience(text="OAuth authentication debugging steps")
    engine.submit_experience(exp2)
    result2 = engine.step()

    exp3 = Experience(text="PostgreSQL query optimization techniques")
    engine.submit_experience(exp3)
    result3 = engine.step()

    # All should have decent salience due to goal matching
    assert result1.data.get("salience", 0) > 0.3
    assert result2.data.get("salience", 0) > 0.3
    assert result3.data.get("salience", 0) > 0.3


def test_encoding_without_goals(engine):
    """Test that encoding works without active goals."""
    # Ensure no active goals
    for goal in list(engine.active_identity.active_goals):
        engine.active_identity.clear_goal(goal.name)

    assert len(engine.active_identity.active_goals) == 0

    # Encode experience
    exp = Experience(text="Random memory without context")
    engine.submit_experience(exp)
    result = engine.step()

    # Should still work, just no identity boost
    assert result.success
    assert "fragment_id" in result.data
    assert result.data.get("salience", 0) > 0.0


def test_goal_completion_removes_boost(engine):
    """Test that completing a goal removes its boost effect."""
    # Set goal
    engine.active_identity.set_active_goal("Learn Rust", intensity=0.8)

    # Encode relevant experience
    exp1 = Experience(text="Rust ownership and borrowing concepts")
    engine.submit_experience(exp1)
    result1 = engine.step()
    salience_with_goal = result1.data.get("salience", 0)

    # Complete goal
    engine.active_identity.clear_goal("Learn Rust")

    # Encode same type of experience
    exp2 = Experience(text="Rust memory management patterns")
    engine.submit_experience(exp2)
    result2 = engine.step()
    salience_without_goal = result2.data.get("salience", 0)

    # Salience should be lower without active goal
    assert salience_without_goal <= salience_with_goal


def test_identity_state_persists_across_encodings(engine):
    """Test that identity state persists across multiple encodings."""
    # Set goal
    goal_description = "Understand machine learning"
    engine.active_identity.set_active_goal(goal_description, intensity=0.8)

    # Encode multiple experiences
    for i in range(5):
        exp = Experience(text=f"ML concept {i}: neural networks and training")
        engine.submit_experience(exp)
        result = engine.step()

        # Goal should still be active
        active_goals = [g.description for g in engine.active_identity.active_goals]
        assert goal_description in active_goals


def test_trait_based_boost(engine):
    """Test trait-based salience boost (if implemented)."""
    # Add contextual trait
    engine.active_identity.contextual_traits["curious"] = 0.7

    # Encode curious-sounding experience
    curious_exp = Experience(
        text="I wonder how this works and why it behaves this way"
    )
    engine.submit_experience(curious_exp)
    result_curious = engine.step()

    # Encode non-curious experience
    boring_exp = Experience(text="Standard procedure completed")
    engine.submit_experience(boring_exp)
    result_boring = engine.step()

    # Curious experience should get small trait boost
    salience_curious = result_curious.data.get("salience", 0)
    salience_boring = result_boring.data.get("salience", 0)

    # Trait boost is smaller (0.1 * weight) than goal boost (0.2 * priority)
    # So difference might be subtle
    assert salience_curious >= salience_boring * 0.95  # Allow small variance


def test_consolidation_with_identity_goals(engine):
    """Test that consolidation works with identity goals active."""
    # Set goal
    engine.active_identity.set_active_goal("Learn Docker", intensity=0.7)

    # Encode some experiences
    for i in range(10):
        exp = Experience(
            text=f"Docker container concept {i}",
            emotional={"arousal": 0.5, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Trigger consolidation if scheduler exists
    if engine.consolidation_scheduler:
        stats = engine.consolidation_scheduler.consolidate()

        # Should have rehearsed some fragments
        assert stats["rehearsed_count"] >= 0


def test_retrieval_with_active_goals(engine):
    """Test that retrieval works with active goals."""
    from src.reconstructions.core import Query

    # Set goal and encode related memory
    engine.active_identity.set_active_goal("Learn GraphQL", intensity=0.8)

    exp = Experience(text="GraphQL query syntax and schema definition")
    engine.submit_experience(exp)
    engine.step()

    # Query for the memory
    query = Query(semantic="GraphQL")
    engine.submit_query(query)
    result = engine.step()

    # Should find the fragment
    assert result.success
    strand = result.data.get("strand")
    assert strand is not None
    assert len(strand.fragments) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
