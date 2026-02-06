"""
Unit tests for adaptive consolidation scheduler.
"""

import time
from datetime import datetime, timedelta
import pytest

from src.reconstructions.consolidation import (
    ActivityMonitor,
    AdaptiveScheduler,
    AdaptiveConsolidationConfig
)


def test_activity_monitor_initialization():
    """Test ActivityMonitor initializes correctly."""
    monitor = ActivityMonitor()

    assert len(monitor.recent_encodings) == 0
    assert len(monitor.recent_queries) == 0
    assert len(monitor.recent_saliences) == 0


def test_record_encoding():
    """Test recording encoding events."""
    monitor = ActivityMonitor()

    monitor.record_encoding(0.7)
    monitor.record_encoding(0.5)
    monitor.record_encoding(0.8)

    assert len(monitor.recent_encodings) == 3
    assert len(monitor.recent_saliences) == 3
    assert list(monitor.recent_saliences) == [0.7, 0.5, 0.8]


def test_record_query():
    """Test recording query events."""
    monitor = ActivityMonitor()

    monitor.record_query()
    monitor.record_query()

    assert len(monitor.recent_queries) == 2


def test_encoding_rate_per_minute():
    """Test encoding rate calculation."""
    monitor = ActivityMonitor()

    # Record 5 encodings now
    for _ in range(5):
        monitor.record_encoding(0.5)

    rate = monitor.encoding_rate_per_minute()
    assert rate == 5.0

    # Empty monitor should return 0
    empty_monitor = ActivityMonitor()
    assert empty_monitor.encoding_rate_per_minute() == 0.0


def test_has_high_salience_activity():
    """Test high salience detection."""
    monitor = ActivityMonitor()

    # Add low salience fragments
    monitor.record_encoding(0.3)
    monitor.record_encoding(0.4)
    monitor.record_encoding(0.5)

    assert not monitor.has_high_salience_activity(0.7)

    # Add high salience fragment
    monitor.record_encoding(0.8)

    assert monitor.has_high_salience_activity(0.7)


def test_is_idle():
    """Test idle detection."""
    monitor = ActivityMonitor()

    # No activity should be idle
    assert monitor.is_idle()

    # Recent activity should not be idle
    monitor.record_encoding(0.5)
    assert not monitor.is_idle()

    monitor.record_query()
    assert not monitor.is_idle()


def test_adaptive_scheduler_initialization():
    """Test AdaptiveScheduler initializes correctly."""
    config = AdaptiveConsolidationConfig()
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    assert scheduler.config == config
    assert scheduler.monitor == monitor


def test_calculate_interval_disabled():
    """Test interval calculation when adaptive scheduling is disabled."""
    config = AdaptiveConsolidationConfig(adaptive_scheduling=False)
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Should always return base interval
    assert scheduler.calculate_next_interval() == config.base_interval_seconds


def test_calculate_interval_high_encoding_rate():
    """Test interval drops to minimum with high encoding rate."""
    config = AdaptiveConsolidationConfig(
        high_encoding_threshold=10,
        min_interval_seconds=10.0
    )
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Simulate high encoding rate (15 encodings/minute)
    for _ in range(15):
        monitor.record_encoding(0.5)

    interval = scheduler.calculate_next_interval()
    assert interval == config.min_interval_seconds


def test_calculate_interval_important_memory():
    """Test interval drops to minimum with important memory."""
    config = AdaptiveConsolidationConfig(
        importance_threshold=0.7,
        min_interval_seconds=10.0
    )
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Add important memory
    monitor.record_encoding(0.8)

    interval = scheduler.calculate_next_interval()
    assert interval == config.min_interval_seconds


def test_calculate_interval_idle():
    """Test interval increases to maximum when idle."""
    config = AdaptiveConsolidationConfig(
        max_interval_seconds=300.0
    )
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Monitor starts idle (no activity)
    interval = scheduler.calculate_next_interval()
    assert interval == config.max_interval_seconds


def test_calculate_interval_normal():
    """Test interval stays at base for normal activity."""
    config = AdaptiveConsolidationConfig(
        base_interval_seconds=60.0,
        high_encoding_threshold=10,
        importance_threshold=0.7
    )
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Add moderate activity (below high threshold, below importance threshold)
    for _ in range(5):
        monitor.record_encoding(0.5)

    interval = scheduler.calculate_next_interval()
    assert interval == config.base_interval_seconds


def test_deque_maxlen():
    """Test that deques respect maxlen."""
    monitor = ActivityMonitor()

    # Add 150 encodings (maxlen=100)
    for i in range(150):
        monitor.record_encoding(0.5)

    assert len(monitor.recent_encodings) == 100

    # Add 75 saliences (maxlen=50)
    for i in range(75):
        monitor.record_encoding(0.6)

    assert len(monitor.recent_saliences) == 50


def test_multiple_triggers():
    """Test that first matching trigger is used."""
    config = AdaptiveConsolidationConfig(
        high_encoding_threshold=5,
        importance_threshold=0.7,
        min_interval_seconds=10.0
    )
    monitor = ActivityMonitor()
    scheduler = AdaptiveScheduler(config, monitor)

    # Trigger both high encoding rate AND high salience
    for _ in range(10):
        monitor.record_encoding(0.8)

    # Should still return min_interval
    interval = scheduler.calculate_next_interval()
    assert interval == config.min_interval_seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
