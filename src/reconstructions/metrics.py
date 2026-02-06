"""
Retrieval Quality Metrics Tracking.

Focused module for tracking query performance and consolidation effectiveness.
Can be used independently of the full health monitoring system.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

from .core import Query, Strand


@dataclass
class QueryMetric:
    """Metrics for a single query execution."""
    timestamp: datetime
    query_text: str
    fragments_found: int
    coherence_score: float
    latency_ms: float
    success: bool  # coherence > threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'query_text': self.query_text,
            'fragments_found': self.fragments_found,
            'coherence_score': self.coherence_score,
            'latency_ms': self.latency_ms,
            'success': self.success
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QueryMetric':
        """Load from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            query_text=data['query_text'],
            fragments_found=data['fragments_found'],
            coherence_score=data['coherence_score'],
            latency_ms=data['latency_ms'],
            success=data['success']
        )


@dataclass
class RetrievalQualitySnapshot:
    """Aggregate metrics over a time period."""
    period_start: datetime
    period_end: datetime
    total_queries: int
    avg_fragments_found: float
    avg_coherence: float
    avg_latency_ms: float
    success_rate: float
    median_coherence: float
    p95_latency_ms: float

    def __str__(self) -> str:
        return (
            f"Retrieval Quality ({self.period_start.strftime('%Y-%m-%d %H:%M')} - "
            f"{self.period_end.strftime('%H:%M')})\n"
            f"  Queries: {self.total_queries}\n"
            f"  Success rate: {self.success_rate * 100:.1f}%\n"
            f"  Avg coherence: {self.avg_coherence:.3f} (median: {self.median_coherence:.3f})\n"
            f"  Avg latency: {self.avg_latency_ms:.1f}ms (p95: {self.p95_latency_ms:.1f}ms)\n"
            f"  Avg fragments found: {self.avg_fragments_found:.1f}"
        )


class RetrievalQualityTracker:
    """
    Tracks query quality over time for consolidation impact analysis.

    Provides metrics on:
    - Query success rates
    - Coherence score trends
    - Latency performance
    - Consolidation effectiveness
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        success_threshold: float = 0.5
    ):
        """
        Initialize quality tracker.

        Args:
            log_path: Path to query log file (default: ~/.reconstructions/queries.jsonl)
            success_threshold: Minimum coherence for a successful query
        """
        if log_path is None:
            data_dir = Path.home() / ".reconstructions"
            data_dir.mkdir(parents=True, exist_ok=True)
            log_path = data_dir / "queries.jsonl"

        self.log_path = log_path
        self.success_threshold = success_threshold

    def log_query(
        self,
        query: Query,
        result: Strand,
        latency_ms: float
    ) -> QueryMetric:
        """
        Record a query execution.

        Args:
            query: Query that was executed
            result: Reconstruction result
            latency_ms: Query execution time in milliseconds

        Returns:
            QueryMetric for the recorded query
        """
        metric = QueryMetric(
            timestamp=datetime.now(),
            query_text=query.semantic or "",
            fragments_found=len(result.fragments),
            coherence_score=result.coherence_score,
            latency_ms=latency_ms,
            success=result.coherence_score > self.success_threshold
        )

        # Append to log
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(metric.to_dict()) + '\n')

        return metric

    def get_recent_metrics(self, hours: int = 24) -> List[QueryMetric]:
        """
        Retrieve recent query history.

        Args:
            hours: Number of hours to look back

        Returns:
            List of QueryMetric objects from the time window
        """
        if not self.log_path.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        metrics = []

        with open(self.log_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                metric = QueryMetric.from_dict(data)

                if metric.timestamp >= cutoff:
                    metrics.append(metric)

        return metrics

    def calculate_success_rate(self, hours: int = 24) -> float:
        """
        Calculate percentage of successful queries.

        Args:
            hours: Time window in hours

        Returns:
            Success rate as float between 0 and 1
        """
        metrics = self.get_recent_metrics(hours)

        if not metrics:
            return 0.0

        successful = sum(1 for m in metrics if m.success)
        return successful / len(metrics)

    def get_snapshot(self, hours: int = 24) -> Optional[RetrievalQualitySnapshot]:
        """
        Generate aggregate metrics snapshot.

        Args:
            hours: Time window in hours

        Returns:
            RetrievalQualitySnapshot or None if no data
        """
        metrics = self.get_recent_metrics(hours)

        if not metrics:
            return None

        # Calculate aggregates
        total_queries = len(metrics)
        avg_fragments = sum(m.fragments_found for m in metrics) / total_queries
        avg_coherence = sum(m.coherence_score for m in metrics) / total_queries
        avg_latency = sum(m.latency_ms for m in metrics) / total_queries
        success_rate = sum(1 for m in metrics if m.success) / total_queries

        # Median coherence
        sorted_coherence = sorted(m.coherence_score for m in metrics)
        median_idx = len(sorted_coherence) // 2
        median_coherence = sorted_coherence[median_idx]

        # P95 latency
        sorted_latency = sorted(m.latency_ms for m in metrics)
        p95_idx = int(len(sorted_latency) * 0.95)
        p95_latency = sorted_latency[min(p95_idx, len(sorted_latency) - 1)]

        period_end = datetime.now()
        period_start = period_end - timedelta(hours=hours)

        return RetrievalQualitySnapshot(
            period_start=period_start,
            period_end=period_end,
            total_queries=total_queries,
            avg_fragments_found=avg_fragments,
            avg_coherence=avg_coherence,
            avg_latency_ms=avg_latency,
            success_rate=success_rate,
            median_coherence=median_coherence,
            p95_latency_ms=p95_latency
        )

    def consolidation_impact_analysis(
        self,
        before_hours: int = 24,
        after_hours: int = 24
    ) -> Dict[str, float]:
        """
        Measure retrieval improvement after consolidation.

        Compares metrics before and after a consolidation window.

        Args:
            before_hours: Hours before consolidation started
            after_hours: Hours after consolidation completed

        Returns:
            Dictionary with improvement metrics
        """
        # This is a simplified version - in practice you'd want to mark
        # consolidation runs explicitly to do proper before/after comparison

        # Get all metrics
        all_metrics = self.get_recent_metrics(before_hours + after_hours)

        if len(all_metrics) < 10:
            return {
                "insufficient_data": True,
                "coherence_improvement": 0.0,
                "latency_improvement": 0.0,
                "success_rate_improvement": 0.0
            }

        # Split into before/after (rough approximation)
        split_idx = len(all_metrics) // 2
        before_metrics = all_metrics[:split_idx]
        after_metrics = all_metrics[split_idx:]

        # Calculate improvements
        before_coherence = sum(m.coherence_score for m in before_metrics) / len(before_metrics)
        after_coherence = sum(m.coherence_score for m in after_metrics) / len(after_metrics)

        before_latency = sum(m.latency_ms for m in before_metrics) / len(before_metrics)
        after_latency = sum(m.latency_ms for m in after_metrics) / len(after_metrics)

        before_success = sum(1 for m in before_metrics if m.success) / len(before_metrics)
        after_success = sum(1 for m in after_metrics if m.success) / len(after_metrics)

        return {
            "insufficient_data": False,
            "coherence_improvement": after_coherence - before_coherence,
            "latency_improvement": before_latency - after_latency,  # Positive = faster
            "success_rate_improvement": after_success - before_success,
            "before_coherence": before_coherence,
            "after_coherence": after_coherence,
            "before_latency_ms": before_latency,
            "after_latency_ms": after_latency
        }

    def get_trend(self, hours: int = 168) -> List[RetrievalQualitySnapshot]:
        """
        Get quality trends over time (default: 1 week).

        Args:
            hours: Total time window

        Returns:
            List of hourly snapshots showing trend
        """
        snapshots = []

        # Generate snapshots for each hour window
        for hour_offset in range(0, hours, 24):  # Daily snapshots
            end_time = datetime.now() - timedelta(hours=hour_offset)
            start_time = end_time - timedelta(hours=24)

            # Get metrics for this window
            all_metrics = self.get_recent_metrics(hours=hours)
            window_metrics = [
                m for m in all_metrics
                if start_time <= m.timestamp < end_time
            ]

            if not window_metrics:
                continue

            # Calculate snapshot
            total_queries = len(window_metrics)
            avg_fragments = sum(m.fragments_found for m in window_metrics) / total_queries
            avg_coherence = sum(m.coherence_score for m in window_metrics) / total_queries
            avg_latency = sum(m.latency_ms for m in window_metrics) / total_queries
            success_rate = sum(1 for m in window_metrics if m.success) / total_queries

            sorted_coherence = sorted(m.coherence_score for m in window_metrics)
            median_coherence = sorted_coherence[len(sorted_coherence) // 2]

            sorted_latency = sorted(m.latency_ms for m in window_metrics)
            p95_latency = sorted_latency[int(len(sorted_latency) * 0.95)]

            snapshot = RetrievalQualitySnapshot(
                period_start=start_time,
                period_end=end_time,
                total_queries=total_queries,
                avg_fragments_found=avg_fragments,
                avg_coherence=avg_coherence,
                avg_latency_ms=avg_latency,
                success_rate=success_rate,
                median_coherence=median_coherence,
                p95_latency_ms=p95_latency
            )

            snapshots.append(snapshot)

        return list(reversed(snapshots))  # Oldest first

    def clear_old_metrics(self, days: int = 30):
        """
        Remove metrics older than specified days.

        Args:
            days: Age threshold in days
        """
        if not self.log_path.exists():
            return

        cutoff = datetime.now() - timedelta(days=days)
        temp_path = self.log_path.with_suffix('.tmp')

        # Rewrite log file without old entries
        with open(self.log_path, 'r') as infile, open(temp_path, 'w') as outfile:
            for line in infile:
                if not line.strip():
                    continue

                data = json.loads(line)
                metric = QueryMetric.from_dict(data)

                if metric.timestamp >= cutoff:
                    outfile.write(line)

        # Replace original with filtered version
        temp_path.replace(self.log_path)
