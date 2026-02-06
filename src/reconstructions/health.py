"""
Memory System Health Monitoring.

Tracks system metrics, identifies issues, and generates actionable recommendations.
Provides visibility into consolidation effectiveness, retrieval quality, and
database health.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import os

from .core import Query, Strand
from .store import FragmentStore


@dataclass
class FragmentStats:
    """Statistics about stored fragments."""
    total_count: int
    last_24h_count: int
    avg_salience: float
    avg_bindings: float
    never_accessed_count: int
    low_salience_count: int  # Salience < 0.2
    oldest_fragment_days: float
    newest_fragment_seconds: float


@dataclass
class ConsolidationStats:
    """Statistics about consolidation runs."""
    last_run_ago_seconds: float
    runs_per_hour: float
    target_runs_per_hour: float
    total_rehearsals: int
    bindings_created: int
    patterns_discovered: int
    is_falling_behind: bool


@dataclass
class RetrievalStats:
    """Statistics about query performance."""
    recent_queries_count: int  # Last 24h
    avg_fragments_found: float
    avg_coherence: float
    avg_latency_ms: float
    success_rate: float  # Queries with coherence > 0.5


@dataclass
class MemoryHealthReport:
    """Comprehensive system health snapshot."""
    timestamp: datetime
    database_size_mb: float
    fragment_stats: FragmentStats
    consolidation_stats: ConsolidationStats
    retrieval_stats: RetrievalStats
    warnings: List[str]
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'database_size_mb': self.database_size_mb,
            'fragment_stats': asdict(self.fragment_stats),
            'consolidation_stats': asdict(self.consolidation_stats),
            'retrieval_stats': asdict(self.retrieval_stats),
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class MemoryHealthMonitor:
    """
    Tracks system health and generates diagnostic reports.

    Logs metrics over time to identify trends and issues:
    - Fragment growth rate
    - Consolidation effectiveness
    - Retrieval quality improvements
    - Database health
    """

    def __init__(self, store: FragmentStore, data_dir: Optional[Path] = None):
        """
        Initialize health monitor.

        Args:
            store: FragmentStore to monitor
            data_dir: Directory for metric logs (default: ~/.reconstructions)
        """
        self.store = store

        if data_dir is None:
            data_dir = Path.home() / ".reconstructions"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log_path = data_dir / "metrics.jsonl"
        self.consolidation_log_path = data_dir / "consolidation.jsonl"
        self.queries_log_path = data_dir / "queries.jsonl"

    def diagnose(self) -> MemoryHealthReport:
        """
        Generate comprehensive health report.

        Returns:
            MemoryHealthReport with current system state
        """
        timestamp = datetime.now()

        # Gather statistics
        fragment_stats = self._get_fragment_stats()
        consolidation_stats = self._get_consolidation_stats()
        retrieval_stats = self._get_retrieval_stats()

        # Calculate database size
        db_size_mb = 0.0
        if self.store.db_path.exists():
            db_size_mb = self.store.db_path.stat().st_size / (1024 * 1024)

        # Identify issues and generate recommendations
        warnings = self._identify_issues(fragment_stats, consolidation_stats, retrieval_stats, db_size_mb)
        recommendations = self._suggest_actions(fragment_stats, consolidation_stats, retrieval_stats)

        report = MemoryHealthReport(
            timestamp=timestamp,
            database_size_mb=db_size_mb,
            fragment_stats=fragment_stats,
            consolidation_stats=consolidation_stats,
            retrieval_stats=retrieval_stats,
            warnings=warnings,
            recommendations=recommendations
        )

        # Log report
        self._log_report(report)

        return report

    def log_consolidation_run(
        self,
        rehearsals: int,
        bindings_created: int,
        patterns_discovered: int = 0
    ):
        """
        Track consolidation execution.

        Args:
            rehearsals: Number of fragments rehearsed
            bindings_created: Number of new bindings created
            patterns_discovered: Number of patterns found
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rehearsals': rehearsals,
            'bindings_created': bindings_created,
            'patterns_discovered': patterns_discovered
        }
        self._append_to_log(self.consolidation_log_path, entry)

    def log_query(self, query: Query, result: Strand, latency_ms: float):
        """
        Track query execution.

        Args:
            query: Query that was executed
            result: Reconstruction result
            latency_ms: Query execution time in milliseconds
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query.semantic,
            'fragments_found': len(result.fragments),
            'coherence': result.coherence_score,
            'success': result.coherence_score > 0.5,
            'latency_ms': latency_ms
        }
        self._append_to_log(self.queries_log_path, entry)

    def _get_fragment_stats(self) -> FragmentStats:
        """Calculate fragment statistics from database."""
        cursor = self.store.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM fragments")
        total_count = cursor.fetchone()[0]

        if total_count == 0:
            return FragmentStats(
                total_count=0,
                last_24h_count=0,
                avg_salience=0.0,
                avg_bindings=0.0,
                never_accessed_count=0,
                low_salience_count=0,
                oldest_fragment_days=0.0,
                newest_fragment_seconds=0.0
            )

        # Recent fragments (last 24h)
        cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
        cursor.execute(
            "SELECT COUNT(*) FROM fragments WHERE created_at > ?",
            (cutoff_time,)
        )
        last_24h_count = cursor.fetchone()[0]

        # Average salience
        cursor.execute("SELECT AVG(initial_salience) FROM fragments")
        avg_salience = cursor.fetchone()[0] or 0.0

        # Average bindings per fragment
        cursor.execute("SELECT bindings FROM fragments")
        all_bindings = cursor.fetchall()
        total_bindings = sum(len(json.loads(row[0])) for row in all_bindings)
        avg_bindings = total_bindings / total_count if total_count > 0 else 0.0

        # Never accessed fragments
        cursor.execute("SELECT access_log FROM fragments")
        never_accessed_count = sum(
            1 for row in cursor.fetchall()
            if len(json.loads(row[0])) == 0
        )

        # Low salience fragments
        cursor.execute(
            "SELECT COUNT(*) FROM fragments WHERE initial_salience < 0.2"
        )
        low_salience_count = cursor.fetchone()[0]

        # Age stats
        now = datetime.now().timestamp()
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM fragments")
        min_time, max_time = cursor.fetchone()

        oldest_fragment_days = (now - min_time) / 86400 if min_time else 0.0
        newest_fragment_seconds = now - max_time if max_time else 0.0

        return FragmentStats(
            total_count=total_count,
            last_24h_count=last_24h_count,
            avg_salience=avg_salience,
            avg_bindings=avg_bindings,
            never_accessed_count=never_accessed_count,
            low_salience_count=low_salience_count,
            oldest_fragment_days=oldest_fragment_days,
            newest_fragment_seconds=newest_fragment_seconds
        )

    def _get_consolidation_stats(self) -> ConsolidationStats:
        """Calculate consolidation statistics from logs."""
        # Read recent consolidation runs
        runs = self._read_recent_log(self.consolidation_log_path, hours=24)

        if not runs:
            return ConsolidationStats(
                last_run_ago_seconds=float('inf'),
                runs_per_hour=0.0,
                target_runs_per_hour=60.0,  # Default target
                total_rehearsals=0,
                bindings_created=0,
                patterns_discovered=0,
                is_falling_behind=True
            )

        # Last run time
        last_run_time = datetime.fromisoformat(runs[-1]['timestamp'])
        last_run_ago_seconds = (datetime.now() - last_run_time).total_seconds()

        # Runs per hour
        runs_per_hour = len(runs)  # Already filtered to last 24h
        target_runs_per_hour = 60.0  # Once per minute

        # Totals
        total_rehearsals = sum(r.get('rehearsals', 0) for r in runs)
        bindings_created = sum(r.get('bindings_created', 0) for r in runs)
        patterns_discovered = sum(r.get('patterns_discovered', 0) for r in runs)

        # Check if falling behind
        is_falling_behind = runs_per_hour < (target_runs_per_hour * 0.8)

        return ConsolidationStats(
            last_run_ago_seconds=last_run_ago_seconds,
            runs_per_hour=runs_per_hour,
            target_runs_per_hour=target_runs_per_hour,
            total_rehearsals=total_rehearsals,
            bindings_created=bindings_created,
            patterns_discovered=patterns_discovered,
            is_falling_behind=is_falling_behind
        )

    def _get_retrieval_stats(self) -> RetrievalStats:
        """Calculate retrieval statistics from logs."""
        queries = self._read_recent_log(self.queries_log_path, hours=24)

        if not queries:
            return RetrievalStats(
                recent_queries_count=0,
                avg_fragments_found=0.0,
                avg_coherence=0.0,
                avg_latency_ms=0.0,
                success_rate=0.0
            )

        # Calculate averages
        avg_fragments = sum(q['fragments_found'] for q in queries) / len(queries)
        avg_coherence = sum(q['coherence'] for q in queries) / len(queries)
        avg_latency = sum(q['latency_ms'] for q in queries) / len(queries)

        # Success rate
        successful = sum(1 for q in queries if q['success'])
        success_rate = successful / len(queries)

        return RetrievalStats(
            recent_queries_count=len(queries),
            avg_fragments_found=avg_fragments,
            avg_coherence=avg_coherence,
            avg_latency_ms=avg_latency,
            success_rate=success_rate
        )

    def _identify_issues(
        self,
        fragment_stats: FragmentStats,
        consolidation_stats: ConsolidationStats,
        retrieval_stats: RetrievalStats,
        db_size_mb: float
    ) -> List[str]:
        """Generate warning messages for detected issues."""
        warnings = []

        # Database growing too fast
        if (db_size_mb > 100 and fragment_stats.last_24h_count > 1000) or \
           (db_size_mb > 500):  # Or just very large
            warnings.append(
                f"⚠ Database growing rapidly: {db_size_mb:.1f}MB, "
                f"{fragment_stats.last_24h_count} fragments in last 24h"
            )

        # Many unaccessed fragments
        if fragment_stats.total_count >= 100:
            unaccessed_pct = (fragment_stats.never_accessed_count / fragment_stats.total_count) * 100
            if unaccessed_pct > 30:
                warnings.append(
                    f"⚠ {fragment_stats.never_accessed_count} fragments never accessed "
                    f"({unaccessed_pct:.0f}% of total)"
                )

        # Many low-salience fragments
        if fragment_stats.total_count >= 100:
            low_salience_pct = (fragment_stats.low_salience_count / fragment_stats.total_count) * 100
            if low_salience_pct > 40:
                warnings.append(
                    f"⚠ {fragment_stats.low_salience_count} low-salience fragments "
                    f"({low_salience_pct:.0f}% of total)"
                )

        # Consolidation falling behind
        if consolidation_stats.is_falling_behind:
            warnings.append(
                f"⚠ Consolidation falling behind: {consolidation_stats.runs_per_hour:.0f} runs/hour "
                f"(target: {consolidation_stats.target_runs_per_hour:.0f})"
            )

        # Poor retrieval quality
        if retrieval_stats.recent_queries_count > 10:
            if retrieval_stats.success_rate < 0.5:
                warnings.append(
                    f"⚠ Low retrieval success rate: {retrieval_stats.success_rate * 100:.0f}% "
                    f"(coherence > 0.5)"
                )

        # Few bindings
        if fragment_stats.total_count > 50 and fragment_stats.avg_bindings < 1.0:
            warnings.append(
                f"⚠ Sparse binding network: avg {fragment_stats.avg_bindings:.1f} bindings per fragment"
            )

        return warnings

    def _suggest_actions(
        self,
        fragment_stats: FragmentStats,
        consolidation_stats: ConsolidationStats,
        retrieval_stats: RetrievalStats
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Natural decay (never-accessed fragments are normal and expected)
        # The system uses power law decay - low-strength fragments naturally
        # fall out of reconstructions without needing explicit deletion

        # Consolidation frequency
        if consolidation_stats.is_falling_behind:
            recommendations.append(
                "Increase consolidation frequency or reduce batch size"
            )

        # Binding network
        if fragment_stats.avg_bindings < 1.0 and fragment_stats.total_count > 50:
            recommendations.append(
                "Run more consolidation cycles to strengthen binding network"
            )

        # Retrieval quality
        if retrieval_stats.recent_queries_count > 10:
            if retrieval_stats.avg_coherence < 0.6:
                recommendations.append(
                    "Low coherence scores - run consolidation to improve retrieval"
                )

        return recommendations

    def _append_to_log(self, path: Path, entry: dict):
        """Append JSONL entry to log file."""
        with open(path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _log_report(self, report: MemoryHealthReport):
        """Log full health report."""
        self._append_to_log(self.metrics_log_path, report.to_dict())

    def _read_recent_log(self, path: Path, hours: int = 24) -> List[dict]:
        """Read log entries from last N hours."""
        if not path.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        entries = []

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff:
                    entries.append(entry)

        return entries


def format_health_report(report: MemoryHealthReport) -> str:
    """
    Format health report for display.

    Args:
        report: Health report to format

    Returns:
        Pretty-printed report string
    """
    lines = [
        "Memory System Health Report",
        "=" * 50,
        f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Database: {report.database_size_mb:.2f} MB",
        "",
        "Fragments:",
        f"  Total: {report.fragment_stats.total_count:,}",
        f"  Last 24h: {report.fragment_stats.last_24h_count}",
        f"  Avg salience: {report.fragment_stats.avg_salience:.3f}",
        f"  Avg bindings: {report.fragment_stats.avg_bindings:.2f}",
        f"  Never accessed: {report.fragment_stats.never_accessed_count}",
        f"  Low salience (<0.2): {report.fragment_stats.low_salience_count}",
        "",
        "Consolidation:",
        f"  Last run: {report.consolidation_stats.last_run_ago_seconds:.0f}s ago",
        f"  Frequency: {report.consolidation_stats.runs_per_hour:.0f} runs/hour "
        f"(target: {report.consolidation_stats.target_runs_per_hour:.0f})",
        f"  Total rehearsals: {report.consolidation_stats.total_rehearsals:,}",
        f"  Bindings created: {report.consolidation_stats.bindings_created:,}",
        f"  Patterns discovered: {report.consolidation_stats.patterns_discovered}",
        "",
        "Retrieval:",
        f"  Recent queries: {report.retrieval_stats.recent_queries_count}",
        f"  Avg fragments found: {report.retrieval_stats.avg_fragments_found:.1f}",
        f"  Avg coherence: {report.retrieval_stats.avg_coherence:.3f}",
        f"  Avg latency: {report.retrieval_stats.avg_latency_ms:.1f}ms",
        f"  Success rate: {report.retrieval_stats.success_rate * 100:.0f}%",
        ""
    ]

    if report.warnings:
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  {warning}")
        lines.append("")

    if report.recommendations:
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  • {rec}")
        lines.append("")

    if not report.warnings and not report.recommendations:
        lines.append("✓ System healthy - no issues detected")
        lines.append("")

    return "\n".join(lines)
