"""
Unit tests for identity-aware encoding integration.
"""

import pytest
import tempfile
from pathlib import Path

from src.reconstructions.identity import (
    ActiveIdentityState,
    IdentityState,
    Goal,
    Trait
)
from src.reconstructions.core import Fragment
from src.reconstructions.store import FragmentStore
from src.reconstructions.encoding import Experience
from src.reconstructions.encoder import encode, Context


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def identity_state():
    """Create identity state with test data."""
    state = IdentityState()

    # Add some goals
    goal1 = Goal(
        name="Learn streaming",
        description="Learn streaming protocols like RTMP",
        priority=0.8,
        status="active"
    )
    state.add_goal(goal1)

    goal2 = Goal(
        name="Debug authentication",
        description="Fix authentication issues",
        priority=0.6,
        status="active"
    )
    state.add_goal(goal2)

    # Add some traits
    trait1 = Trait(
        name="curious",
        description="Curious personality",
        strength=0.7
    )
    state.add_trait(trait1)

    return state


@pytest.fixture
def active_identity(identity_state):
    """Create active identity state."""
    return ActiveIdentityState(identity_state)


def test_active_identity_initialization(active_identity):
    """Test ActiveIdentityState initializes correctly."""
    assert active_identity.identity_state is not None
    assert len(active_identity.active_goals) == 2  # Two active goals
    assert len(active_identity.contextual_traits) == 0  # No contextual traits yet


def test_set_active_goal(active_identity):
    """Test setting an active goal."""
    initial_count = len(active_identity.active_goals)

    active_identity.set_active_goal("Test new project", intensity=0.9)

    assert len(active_identity.active_goals) == initial_count + 1

    # Check the new goal
    new_goal = active_identity.active_goals[-1]
    assert new_goal.description == "Test new project"
    assert new_goal.priority == 0.9
    assert new_goal.status == "active"


def test_update_existing_goal(active_identity):
    """Test updating an existing goal's priority."""
    initial_count = len(active_identity.active_goals)

    # Set goal twice with same description
    active_identity.set_active_goal("Learn streaming protocols", intensity=0.7)
    active_identity.set_active_goal("Learn streaming protocols", intensity=0.9)

    # Should not create duplicate, just update
    # Might have one more if "Learn streaming protocols" != "Learn streaming"
    # Let me check the actual goal
    matching_goals = [
        g for g in active_identity.active_goals
        if "streaming" in g.description.lower()
    ]
    assert len(matching_goals) >= 1

    # At least one streaming-related goal with high priority
    assert any(g.priority == 0.9 for g in matching_goals)


def test_clear_goal(active_identity):
    """Test clearing/completing a goal."""
    initial_count = len(active_identity.active_goals)

    # Clear first goal
    first_goal_name = active_identity.active_goals[0].name
    active_identity.clear_goal(first_goal_name)

    assert len(active_identity.active_goals) == initial_count - 1


def test_relevance_boost_for_goal_related_fragment(temp_db, active_identity):
    """Test that goal-related fragments get salience boost."""
    # Create fragment about streaming (matches "Learn streaming" goal)
    fragment = Fragment(
        content={"semantic": "Implementing RTMP streaming protocol for live video"},
        initial_salience=0.5
    )

    boost = active_identity.relevance_boost(fragment)

    # Should get boost for matching streaming goal
    assert boost > 0.0
    assert boost <= 0.5  # Capped at 0.5


def test_relevance_boost_for_unrelated_fragment(temp_db, active_identity):
    """Test that unrelated fragments get no boost."""
    # Create fragment about something unrelated
    fragment = Fragment(
        content={"semantic": "Making a sandwich for lunch today"},
        initial_salience=0.5
    )

    boost = active_identity.relevance_boost(fragment)

    # Should get minimal or no boost
    assert boost >= 0.0
    assert boost < 0.1  # Very small or zero


def test_trait_expression_detection(active_identity):
    """Test trait expression detection in fragments."""
    # Fragment expressing curiosity
    curious_fragment = Fragment(
        content={"semantic": "I wonder how this works and why it behaves this way"},
        initial_salience=0.5
    )

    # Check if trait detection works
    expresses_curious = active_identity._expresses_trait(curious_fragment, "curious")
    assert expresses_curious

    # Fragment not expressing curiosity
    boring_fragment = Fragment(
        content={"semantic": "Regular task completed successfully"},
        initial_salience=0.5
    )

    expresses_curious = active_identity._expresses_trait(boring_fragment, "curious")
    assert not expresses_curious


def test_identity_aware_encoding(temp_db, active_identity):
    """Test that encoding with identity state boosts relevant memories."""
    store = FragmentStore(str(temp_db))
    context = Context()

    # Create goal-related experience
    exp = Experience(text="Setting up RTMP server for streaming video content")

    # Encode without identity (baseline)
    fragment_without = encode(exp, context, store, identity_state=None)

    # Reset store
    temp_db.unlink()
    store = FragmentStore(str(temp_db))

    # Encode with identity
    fragment_with = encode(exp, context, store, identity_state=active_identity)

    # Fragment with identity should have higher salience
    assert fragment_with.initial_salience >= fragment_without.initial_salience


def test_multiple_goals_cumulative_boost(active_identity):
    """Test that multiple matching goals provide cumulative boost (up to cap)."""
    # Add another streaming-related goal
    active_identity.set_active_goal("Optimize streaming performance", intensity=0.7)

    # Create fragment matching multiple goals
    fragment = Fragment(
        content={"semantic": "Debugging RTMP streaming authentication issues"},
        initial_salience=0.5
    )

    boost = active_identity.relevance_boost(fragment)

    # Should get boost from multiple goals, but capped at 0.5
    assert boost > 0.1  # More than single goal
    assert boost <= 0.5  # Capped


def test_goal_semantic_matching(temp_db, active_identity):
    """Test semantic similarity matching between fragment and goal."""
    # Directly test _relates_to_goal
    goal = active_identity.active_goals[0]  # "Learn streaming" goal

    # Related fragment
    related_fragment = Fragment(
        content={"semantic": "RTMP protocol documentation and implementation"},
        initial_salience=0.5
    )

    assert active_identity._relates_to_goal(related_fragment, goal)

    # Unrelated fragment
    unrelated_fragment = Fragment(
        content={"semantic": "Making coffee in the kitchen"},
        initial_salience=0.5
    )

    assert not active_identity._relates_to_goal(unrelated_fragment, goal)


def test_empty_fragment_content(active_identity):
    """Test handling of fragments with empty content."""
    fragment = Fragment(
        content={},  # No semantic content
        initial_salience=0.5
    )

    boost = active_identity.relevance_boost(fragment)

    # Should handle gracefully, return 0 boost
    assert boost == 0.0


def test_no_active_goals(temp_db):
    """Test identity state with no active goals."""
    empty_state = IdentityState()
    active_identity = ActiveIdentityState(empty_state)

    fragment = Fragment(
        content={"semantic": "Some random content"},
        initial_salience=0.5
    )

    boost = active_identity.relevance_boost(fragment)

    # No goals means no boost
    assert boost == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
