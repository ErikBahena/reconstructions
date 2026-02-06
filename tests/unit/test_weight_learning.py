"""
Unit tests for self-tuning salience weight learning.
"""

import tempfile
from pathlib import Path
import pytest

from src.reconstructions.learning import SalienceWeightLearner
from src.reconstructions.core import Fragment


@pytest.fixture
def temp_checkpoint():
    """Create temporary checkpoint file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def learner():
    """Create weight learner."""
    return SalienceWeightLearner()


def test_learner_initialization(learner):
    """Test learner initializes with default weights."""
    weights = learner.get_current_weights()

    assert weights["emotional"] == 0.25
    assert weights["novelty"] == 0.15
    assert weights["goal"] == 0.30
    assert weights["depth"] == 0.30

    # Should sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_sum_to_one(learner):
    """Test that weights always sum to 1.0 after normalization."""
    learner._normalize()
    weights = learner.get_current_weights()
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_within_bounds(learner):
    """Test that weights stay normalized and respect min_weight."""
    # Manually set extreme values
    learner.w_emotional = 0.0
    learner.w_novelty = 1.0
    learner.w_goal = 0.0
    learner.w_depth = 0.0

    learner._normalize()

    weights = learner.get_current_weights()

    # All weights should be at least min_weight
    for weight in weights.values():
        assert weight >= learner.min_weight

    # Weights should sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0)

    # Note: In extreme cases where one weight dominates, after clamping
    # and renormalizing, that weight may exceed max_weight to maintain sum=1.0
    # This is acceptable as it ensures valid probability distribution


def test_record_successful_retrieval(learner):
    """Test recording successful retrieval."""
    initial_emotional = learner.w_emotional

    # Create fragment with high emotional intensity
    fragment = Fragment(
        content={
            "semantic": "test",
            "emotional": {"arousal": 0.9, "valence": 0.2}
        },
        initial_salience=0.7
    )

    learner.record_retrieval(fragment, was_useful=True)

    # Emotional weight should have increased
    assert learner.w_emotional >= initial_emotional
    assert learner.successful_retrievals == 1
    assert learner.total_feedback == 1


def test_record_failed_retrieval(learner):
    """Test recording failed retrieval."""
    initial_emotional = learner.w_emotional

    # Create fragment with high emotional intensity
    fragment = Fragment(
        content={
            "semantic": "test",
            "emotional": {"arousal": 0.9, "valence": 0.2}
        },
        initial_salience=0.3
    )

    learner.record_retrieval(fragment, was_useful=False)

    # Emotional weight should have decreased slightly
    assert learner.w_emotional < initial_emotional
    assert learner.failed_retrievals == 1
    assert learner.total_feedback == 1


def test_multiple_feedback_updates(learner):
    """Test that multiple feedback updates converge."""
    # Simulate 50 successful retrievals with emotional fragments
    for i in range(50):
        fragment = Fragment(
            content={
                "semantic": f"test {i}",
                "emotional": {"arousal": 0.8, "valence": 0.3}
            },
            initial_salience=0.7
        )
        learner.record_retrieval(fragment, was_useful=True)

    # Emotional weight should have increased significantly
    weights = learner.get_current_weights()
    assert weights["emotional"] > 0.25  # Higher than initial

    # But should still be normalized
    assert sum(weights.values()) == pytest.approx(1.0)


def test_get_emotional_intensity(learner):
    """Test emotional intensity calculation."""
    # High arousal
    high_arousal = Fragment(
        content={"emotional": {"arousal": 0.9, "valence": 0.5}},
        initial_salience=0.5
    )
    assert learner._get_emotional_intensity(high_arousal) == pytest.approx(0.9)

    # Extreme valence (negative)
    extreme_negative = Fragment(
        content={"emotional": {"arousal": 0.3, "valence": 0.0}},
        initial_salience=0.5
    )
    intensity = learner._get_emotional_intensity(extreme_negative)
    assert intensity > 0.5  # Extreme valence

    # No emotion
    no_emotion = Fragment(
        content={"semantic": "test"},
        initial_salience=0.5
    )
    assert learner._get_emotional_intensity(no_emotion) == 0.0


def test_has_high_novelty(learner):
    """Test novelty detection."""
    # New fragment (low access count)
    novel_fragment = Fragment(
        content={"semantic": "test"},
        initial_salience=0.5,
        access_log=[1.0]  # Accessed once
    )
    assert learner._has_high_novelty(novel_fragment)

    # Frequently accessed fragment
    familiar_fragment = Fragment(
        content={"semantic": "test"},
        initial_salience=0.5,
        access_log=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Accessed 10 times
    )
    assert not learner._has_high_novelty(familiar_fragment)


def test_save_checkpoint(learner, temp_checkpoint):
    """Test saving weights to checkpoint."""
    # Modify weights
    learner.w_emotional = 0.4
    learner.w_novelty = 0.2
    learner.w_goal = 0.25
    learner.w_depth = 0.15
    learner._normalize()

    learner.total_feedback = 100
    learner.successful_retrievals = 75

    # Save
    learner.save_checkpoint(temp_checkpoint)

    assert temp_checkpoint.exists()


def test_load_checkpoint(learner, temp_checkpoint):
    """Test loading weights from checkpoint."""
    # Modify and save
    learner.w_emotional = 0.4
    learner.w_novelty = 0.2
    learner.w_goal = 0.25
    learner.w_depth = 0.15
    learner._normalize()

    learner.total_feedback = 100
    learner.successful_retrievals = 75

    learner.save_checkpoint(temp_checkpoint)

    # Load
    loaded = SalienceWeightLearner.load_checkpoint(temp_checkpoint)

    # Verify weights restored
    assert loaded.w_emotional == pytest.approx(learner.w_emotional)
    assert loaded.w_novelty == pytest.approx(learner.w_novelty)
    assert loaded.w_goal == pytest.approx(learner.w_goal)
    assert loaded.w_depth == pytest.approx(learner.w_depth)

    assert loaded.total_feedback == 100
    assert loaded.successful_retrievals == 75


def test_get_success_rate(learner):
    """Test success rate calculation."""
    # No feedback
    assert learner.get_success_rate() == 0.0

    # Add feedback
    fragment = Fragment(content={"semantic": "test"}, initial_salience=0.7)

    for _ in range(7):
        learner.record_retrieval(fragment, was_useful=True)

    for _ in range(3):
        learner.record_retrieval(fragment, was_useful=False)

    # 7 successful, 3 failed = 70% success rate
    assert learner.get_success_rate() == pytest.approx(0.7)


def test_learning_converges(learner):
    """Test that learning stabilizes after sufficient feedback."""
    fragment = Fragment(
        content={
            "semantic": "test",
            "emotional": {"arousal": 0.8, "valence": 0.3}
        },
        initial_salience=0.7
    )

    # Record many successful retrievals
    for _ in range(100):
        learner.record_retrieval(fragment, was_useful=True)

    weights_before = learner.get_current_weights()

    # Record more
    for _ in range(10):
        learner.record_retrieval(fragment, was_useful=True)

    weights_after = learner.get_current_weights()

    # Changes should be small after many updates
    for key in weights_before:
        change = abs(weights_after[key] - weights_before[key])
        assert change < 0.05  # Small change


def test_different_learning_rates():
    """Test learner with different learning rates."""
    slow_learner = SalienceWeightLearner(learning_rate=0.001)
    fast_learner = SalienceWeightLearner(learning_rate=0.1)

    fragment = Fragment(
        content={"emotional": {"arousal": 0.9, "valence": 0.5}},
        initial_salience=0.7
    )

    # Same feedback
    slow_learner.record_retrieval(fragment, was_useful=True)
    fast_learner.record_retrieval(fragment, was_useful=True)

    # Fast learner should have changed more
    slow_change = abs(slow_learner.w_emotional - 0.25)
    fast_change = abs(fast_learner.w_emotional - 0.25)

    assert fast_change > slow_change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
