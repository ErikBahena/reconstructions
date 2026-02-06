"""
Integration tests for adaptive consolidation scheduler.

Tests end-to-end behavior of adaptive scheduling under different load scenarios.
"""

import time
import tempfile
from pathlib import Path
import pytest

from src.reconstructions.store import FragmentStore
from src.reconstructions.consolidation import (
    ConsolidationScheduler,
    AdaptiveConsolidationConfig
)
from src.reconstructions.engine import ReconstructionEngine
from src.reconstructions.encoding import Experience


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def adaptive_config():
    """Create adaptive config with fast intervals for testing."""
    return AdaptiveConsolidationConfig(
        adaptive_scheduling=True,
        min_interval_seconds=1.0,  # Fast for testing
        max_interval_seconds=10.0,
        base_interval_seconds=3.0,
        high_encoding_threshold=10,
        importance_threshold=0.7,
        RECENT_WINDOW_HOURS=1.0,
        MIN_SALIENCE_FOR_REHEARSAL=0.3,
        REHEARSAL_BATCH_SIZE=3
    )


@pytest.fixture
def engine_with_adaptive(temp_db, adaptive_config):
    """Create engine with adaptive consolidation."""
    store = FragmentStore(str(temp_db))
    engine = ReconstructionEngine(
        store,
        consolidation_config=adaptive_config,
        enable_consolidation=True
    )
    return engine


def test_adaptive_consolidation_with_engine(engine_with_adaptive, adaptive_config):
    """Test that adaptive scheduler is properly integrated with engine."""
    engine = engine_with_adaptive

    # Verify consolidation scheduler exists and is adaptive
    assert engine.consolidation_scheduler is not None
    assert engine.consolidation_scheduler.adaptive_scheduler is not None
    assert engine.consolidation_scheduler.state.activity_monitor is not None
    assert engine.consolidation_scheduler.state.current_interval == adaptive_config.base_interval_seconds


def test_high_encoding_rate_triggers_fast_consolidation(engine_with_adaptive, adaptive_config):
    """Test that high encoding rate reduces consolidation interval."""
    engine = engine_with_adaptive

    # Encode many fragments rapidly (> high_encoding_threshold per minute)
    for i in range(15):
        exp = Experience(
            text=f"High activity fragment {i}",
            emotional={"arousal": 0.5, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Check that adaptive scheduler calculated min interval
    scheduler = engine.consolidation_scheduler
    interval = scheduler.state.current_interval

    # Should be using minimum interval due to high encoding rate
    # Note: interval only gets updated when should_consolidate() is called
    # So let's manually check what the scheduler would calculate
    if scheduler.adaptive_scheduler:
        calculated = scheduler.adaptive_scheduler.calculate_next_interval()
        assert calculated == adaptive_config.min_interval_seconds


def test_important_memory_triggers_fast_consolidation(engine_with_adaptive, adaptive_config):
    """Test that high salience memory triggers faster consolidation."""
    engine = engine_with_adaptive

    # Encode important memory
    exp = Experience(
        text="Critical system failure!",
        emotional={"valence": -0.8, "arousal": 0.9}  # Negative valence, high arousal
    )
    engine.submit_experience(exp)
    result = engine.step()

    # Check salience is high
    assert result.data.get("salience", 0) >= adaptive_config.importance_threshold

    # Check that scheduler would use min interval
    scheduler = engine.consolidation_scheduler
    if scheduler.adaptive_scheduler:
        interval = scheduler.adaptive_scheduler.calculate_next_interval()
        assert interval == adaptive_config.min_interval_seconds


def test_idle_system_uses_slow_consolidation(temp_db, adaptive_config):
    """Test that idle system uses maximum consolidation interval."""
    store = FragmentStore(str(temp_db))
    engine = ReconstructionEngine(
        store,
        consolidation_config=adaptive_config,
        enable_consolidation=True
    )

    # Don't encode anything - system should be idle
    scheduler = engine.consolidation_scheduler

    # Idle system should use max interval
    if scheduler.adaptive_scheduler:
        interval = scheduler.adaptive_scheduler.calculate_next_interval()
        assert interval == adaptive_config.max_interval_seconds


def test_normal_activity_uses_base_interval(engine_with_adaptive, adaptive_config):
    """Test that normal activity uses base consolidation interval."""
    engine = engine_with_adaptive

    # Encode moderate number of fragments with moderate salience
    for i in range(5):
        exp = Experience(
            text=f"Normal fragment {i}",
            emotional={"arousal": 0.4, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Give a moment for activity to settle
    time.sleep(0.1)

    # Check interval calculation
    scheduler = engine.consolidation_scheduler
    if scheduler.adaptive_scheduler:
        interval = scheduler.adaptive_scheduler.calculate_next_interval()
        # Should use base interval (not high rate, not idle, not high salience)
        assert interval == adaptive_config.base_interval_seconds


def test_consolidation_respects_adaptive_interval(engine_with_adaptive, adaptive_config):
    """Test that consolidation only runs when adaptive interval elapses."""
    engine = engine_with_adaptive
    scheduler = engine.consolidation_scheduler

    # Record first consolidation time
    first_consolidation_time = scheduler.state.last_consolidation

    # Set to use min interval (high activity)
    for i in range(15):
        exp = Experience(
            text=f"Fragment {i}",
            emotional={"arousal": 0.5, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Check should_consolidate uses adaptive interval
    # Immediately after, should not consolidate (interval not elapsed)
    assert not scheduler.should_consolidate(first_consolidation_time + 0.5)

    # After min_interval seconds, should consolidate
    assert scheduler.should_consolidate(first_consolidation_time + adaptive_config.min_interval_seconds + 0.1)


def test_activity_monitor_tracks_encodings(engine_with_adaptive):
    """Test that activity monitor properly tracks encoding events."""
    engine = engine_with_adaptive
    monitor = engine.consolidation_scheduler.state.activity_monitor

    initial_count = len(monitor.recent_encodings)

    # Encode fragments
    for i in range(5):
        exp = Experience(
            text=f"Test fragment {i}",
            emotional={"arousal": 0.5, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Monitor should have tracked them
    assert len(monitor.recent_encodings) == initial_count + 5
    assert len(monitor.recent_saliences) == initial_count + 5


def test_encoding_rate_calculation_accuracy(engine_with_adaptive):
    """Test that encoding rate per minute is calculated correctly."""
    engine = engine_with_adaptive
    monitor = engine.consolidation_scheduler.state.activity_monitor

    # Encode 10 fragments
    for i in range(10):
        exp = Experience(text=f"Fragment {i}")
        engine.submit_experience(exp)
        engine.step()

    # Rate should be ~10 per minute (all within last minute)
    rate = monitor.encoding_rate_per_minute()
    assert rate == 10.0


def test_adaptive_scheduler_state_persistence(engine_with_adaptive, adaptive_config):
    """Test that adaptive scheduler state persists across calls."""
    engine = engine_with_adaptive
    scheduler = engine.consolidation_scheduler

    # Record initial state
    initial_interval = scheduler.state.current_interval

    # Trigger high activity
    for i in range(15):
        exp = Experience(
            text=f"Fragment {i}",
            emotional={"arousal": 0.5, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Force interval recalculation by calling should_consolidate
    current_time = time.time()
    scheduler.should_consolidate(current_time)

    # Interval should have changed to min
    assert scheduler.state.current_interval == adaptive_config.min_interval_seconds
    assert scheduler.state.current_interval != initial_interval


def test_consolidation_stats_with_adaptive(engine_with_adaptive):
    """Test that consolidation stats are still tracked with adaptive scheduling."""
    engine = engine_with_adaptive

    # Add some fragments
    for i in range(10):
        exp = Experience(
            text=f"Fragment {i}",
            emotional={"arousal": 0.6, "valence": 0.3}
        )
        engine.submit_experience(exp)
        engine.step()

    # Wait for adaptive interval to elapse
    time.sleep(1.5)

    # Process consolidation
    result = engine.step()

    # Should have consolidation stats if consolidation ran
    if result and result.result_type.value == "consolidated":
        assert "rehearsed_count" in result.data
        assert "bindings_strengthened" in result.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
