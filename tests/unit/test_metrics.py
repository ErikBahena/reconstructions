"""
Unit tests for retrieval quality metrics tracking.
"""

from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import pytest

from src.reconstructions.metrics import (
    QueryMetric,
    RetrievalQualitySnapshot,
    RetrievalQualityTracker
)
from src.reconstructions.core import Query, Strand


def test_query_metric_serialization():
    """Test QueryMetric to/from dict conversion."""
    now = datetime.now()
    metric = QueryMetric(
        timestamp=now,
        query_text="test query",
        fragments_found=5,
        coherence_score=0.75,
        latency_ms=25.5,
        success=True
    )

    # Serialize
    data = metric.to_dict()
    assert data["query_text"] == "test query"
    assert data["fragments_found"] == 5
    assert data["coherence_score"] == 0.75
    assert data["latency_ms"] == 25.5
    assert data["success"] is True

    # Deserialize
    restored = QueryMetric.from_dict(data)
    assert restored.query_text == metric.query_text
    assert restored.fragments_found == metric.fragments_found
    assert restored.coherence_score == metric.coherence_score
    assert restored.latency_ms == metric.latency_ms
    assert restored.success == metric.success


def test_retrieval_quality_snapshot_formatting():
    """Test snapshot string formatting."""
    now = datetime.now()
    snapshot = RetrievalQualitySnapshot(
        period_start=now,
        period_end=now + timedelta(hours=1),
        total_queries=10,
        avg_fragments_found=5.5,
        avg_coherence=0.75,
        avg_latency_ms=50.0,
        success_rate=0.9,
        median_coherence=0.8,
        p95_latency_ms=100.0
    )

    output = str(snapshot)
    assert "Queries: 10" in output
    assert "90.0%" in output  # Success rate
    assert "0.750" in output  # Avg coherence
    assert "50.0ms" in output  # Avg latency


@pytest.fixture
def temp_log_file():
    """Create temporary log file for metrics."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def tracker(temp_log_file):
    """Create metrics tracker."""
    return RetrievalQualityTracker(log_path=temp_log_file)


def test_tracker_initialization(tracker, temp_log_file):
    """Test tracker initializes correctly."""
    assert tracker.log_path == temp_log_file
    assert tracker.success_threshold == 0.5


def test_log_query(tracker):
    """Test logging a query."""
    query = Query(semantic="test query")
    strand = Strand(
        fragments=["frag1", "frag2", "frag3"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    metric = tracker.log_query(query, strand, latency_ms=25.5)

    assert metric.query_text == "test query"
    assert metric.fragments_found == 3
    assert metric.coherence_score == 0.75
    assert metric.latency_ms == 25.5
    assert metric.success is True  # coherence > 0.5

    # Verify it was written to file
    recent = tracker.get_recent_metrics(hours=1)
    assert len(recent) == 1
    assert recent[0].query_text == "test query"


def test_log_query_failure(tracker):
    """Test logging a low-quality query."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=[],
        assembly_context={},
        coherence_score=0.3,  # Low coherence
        variance=0.2,
        certainty=0.8
    )

    metric = tracker.log_query(query, strand, latency_ms=100.0)

    assert metric.success is False  # coherence < 0.5
    assert metric.fragments_found == 0


def test_get_recent_metrics(tracker):
    """Test retrieving recent metrics."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["frag1"],
        assembly_context={},
        coherence_score=0.8,
        variance=0.2,
        certainty=0.8
    )

    # Add metrics
    for i in range(5):
        tracker.log_query(query, strand, latency_ms=50.0)

    # Get all recent
    recent = tracker.get_recent_metrics(hours=24)
    assert len(recent) == 5


def test_calculate_success_rate(tracker):
    """Test success rate calculation."""
    query = Query(semantic="test")

    # Add mix of good and bad queries
    good_strand = Strand(
        fragments=["f1", "f2", "f3"],
        assembly_context={},
        coherence_score=0.8,
        variance=0.2,
        certainty=0.8
    )
    bad_strand = Strand(
        fragments=["f1"],
        assembly_context={},
        coherence_score=0.3,
        variance=0.2,
        certainty=0.8
    )

    # 8 good, 2 bad
    for _ in range(8):
        tracker.log_query(query, good_strand, latency_ms=50.0)
    for _ in range(2):
        tracker.log_query(query, bad_strand, latency_ms=100.0)

    success_rate = tracker.calculate_success_rate(hours=24)
    assert success_rate == 0.8  # 8/10


def test_get_snapshot(tracker):
    """Test snapshot generation."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["f1", "f2"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    # Add some queries
    for _ in range(5):
        tracker.log_query(query, strand, latency_ms=50.0)

    # Get snapshot
    snapshot = tracker.get_snapshot(hours=24)

    assert snapshot is not None
    assert snapshot.total_queries == 5
    assert snapshot.avg_coherence == 0.75
    assert snapshot.avg_latency_ms == 50.0
    assert snapshot.success_rate == 1.0  # All successful


def test_get_trend(tracker):
    """Test trend analysis over time."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["f1", "f2"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    # Add queries
    for _ in range(5):
        tracker.log_query(query, strand, latency_ms=50.0)

    # Get trend (weekly by default)
    trend = tracker.get_trend(hours=1)
    assert len(trend) >= 1
    assert all(isinstance(s, RetrievalQualitySnapshot) for s in trend)


def test_consolidation_impact_analysis(tracker):
    """Test consolidation impact analysis."""
    query = Query(semantic="test")

    # Simulate queries before consolidation (lower quality)
    before_strand = Strand(
        fragments=["f1", "f2"],
        assembly_context={},
        coherence_score=0.6,
        variance=0.3,
        certainty=0.7
    )
    for _ in range(5):
        tracker.log_query(query, before_strand, latency_ms=80.0)

    # Simulate queries after consolidation (higher quality)
    after_strand = Strand(
        fragments=["f1", "f2", "f3", "f4"],
        assembly_context={},
        coherence_score=0.85,
        variance=0.2,
        certainty=0.8
    )
    for _ in range(5):
        tracker.log_query(query, after_strand, latency_ms=60.0)

    # Analyze impact (uses most recent vs older queries)
    impact = tracker.consolidation_impact_analysis()

    # Should show improvement
    assert "coherence_improvement" in impact
    assert "latency_improvement" in impact
    assert impact["coherence_improvement"] > 0  # Should show improvement
    assert impact["latency_improvement"] > 0  # Should show improvement


def test_metrics_persistence(temp_log_file):
    """Test that metrics persist across tracker instances."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["f1"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    # Create tracker and log query
    tracker1 = RetrievalQualityTracker(log_path=temp_log_file)
    tracker1.log_query(query, strand, latency_ms=50.0)

    # Create new tracker instance
    tracker2 = RetrievalQualityTracker(log_path=temp_log_file)

    # Should be able to read the logged metric
    recent = tracker2.get_recent_metrics(hours=24)
    assert len(recent) == 1
    assert recent[0].query_text == "test"
    assert recent[0].latency_ms == 50.0


def test_empty_snapshot(tracker):
    """Test snapshot with no queries."""
    snapshot = tracker.get_snapshot(hours=24)

    # Should return None when no queries
    assert snapshot is None


def test_p95_latency_calculation(tracker):
    """Test p95 latency percentile calculation."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["f1"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    # Add queries with varying latencies
    latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for lat in latencies:
        tracker.log_query(query, strand, latency_ms=float(lat))

    snapshot = tracker.get_snapshot(hours=24)

    # p95 of [10..100] should be close to 95
    assert snapshot.p95_latency_ms >= 90.0
    assert snapshot.p95_latency_ms <= 100.0


def test_clear_old_metrics(tracker):
    """Test clearing old metrics."""
    query = Query(semantic="test")
    strand = Strand(
        fragments=["f1"],
        assembly_context={},
        coherence_score=0.75,
        variance=0.2,
        certainty=0.8
    )

    # Add some queries
    for _ in range(10):
        tracker.log_query(query, strand, latency_ms=50.0)

    # Clear very recent (0 days - should clear all)
    tracker.clear_old_metrics(days=0)

    # Should have no metrics now
    recent = tracker.get_recent_metrics(hours=24)
    assert len(recent) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

