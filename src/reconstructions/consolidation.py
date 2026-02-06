"""
Autonomous Reconstruction and Consolidation Engine.

This module implements the "thinking/meditating" layer - spontaneous memory
processing that happens without external queries. It addresses the problem
where memories are encoded but retrieval paths are weak because there's no
consolidation.

Key functions:
1. Spontaneous replay of recent/salient fragments
2. Strengthening of semantic bindings through co-activation
3. Pattern discovery during idle time
4. Rehearsal scheduling based on salience and recency

This is the missing piece that allows memories to be FOUND, not just STORED.
"""

import time
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import numpy as np

from .core import Fragment, Query
from .store import FragmentStore
from .reconstruction import reconstruct, ReconstructionConfig
from .strength import calculate_strength


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation scheduler."""

    # Rehearsal scheduling
    RECENT_WINDOW_HOURS: float = 24.0  # Consider fragments from last 24h
    MIN_SALIENCE_FOR_REHEARSAL: float = 0.3  # Only rehearse salient memories
    REHEARSAL_BATCH_SIZE: int = 5  # Rehearse this many fragments at a time

    # Pattern discovery
    PATTERN_DISCOVERY_INTERVAL: int = 10  # Every N consolidations
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.6  # Min similarity to bind
    MAX_PATTERN_DISCOVERY_FRAGMENTS: int = 50  # Cap fragments to prevent hangs on large DBs

    # Binding strengthening
    BINDING_STRENGTH_BOOST: float = 0.1  # Boost per co-activation
    MAX_BINDING_BOOST: float = 0.5  # Maximum total boost

    # Scheduling
    MIN_IDLE_TIME_SECONDS: float = 5.0  # Require 5s idle before consolidation
    CONSOLIDATION_INTERVAL_SECONDS: float = 60.0  # Run every minute


@dataclass
class AdaptiveConsolidationConfig(ConsolidationConfig):
    """Extended configuration with adaptive scheduling."""

    # Adaptive scheduling
    adaptive_scheduling: bool = True
    min_interval_seconds: float = 10.0  # Fastest rate (high activity)
    max_interval_seconds: float = 300.0  # Slowest rate (idle)
    base_interval_seconds: float = 60.0  # Default interval

    # Triggers for faster consolidation
    high_encoding_threshold: int = 10  # fragments/minute
    importance_threshold: float = 0.7  # High salience threshold


class ActivityMonitor:
    """
    Tracks encoding and retrieval activity for adaptive scheduling.

    Maintains recent activity history to determine optimal consolidation frequency.
    """

    def __init__(self):
        self.recent_encodings: deque = deque(maxlen=100)
        self.recent_queries: deque = deque(maxlen=100)
        self.recent_saliences: deque = deque(maxlen=50)

    def record_encoding(self, salience: float):
        """Record an encoding event with its salience."""
        now = datetime.now()
        self.recent_encodings.append(now)
        self.recent_saliences.append(salience)

    def record_query(self):
        """Record a query event."""
        now = datetime.now()
        self.recent_queries.append(now)

    def encoding_rate_per_minute(self) -> float:
        """
        Calculate recent encoding frequency.

        Returns:
            Number of encodings per minute
        """
        if not self.recent_encodings:
            return 0.0

        now = datetime.now()
        one_minute_ago = now.timestamp() - 60

        recent_count = sum(
            1 for dt in self.recent_encodings
            if dt.timestamp() >= one_minute_ago
        )

        return float(recent_count)

    def has_high_salience_activity(self, threshold: float) -> bool:
        """
        Check if recent fragments are important.

        Args:
            threshold: Salience threshold

        Returns:
            True if any recent fragment exceeds threshold
        """
        if not self.recent_saliences:
            return False

        # Check last 5 fragments
        recent = list(self.recent_saliences)[-5:]
        return any(s >= threshold for s in recent)

    def is_idle(self) -> bool:
        """
        Check if system has been inactive.

        Returns:
            True if no activity in last 5 minutes
        """
        if not self.recent_encodings and not self.recent_queries:
            return True

        now = datetime.now()
        five_minutes_ago = now.timestamp() - 300

        has_recent_encoding = any(
            dt.timestamp() >= five_minutes_ago
            for dt in self.recent_encodings
        )

        has_recent_query = any(
            dt.timestamp() >= five_minutes_ago
            for dt in self.recent_queries
        )

        return not (has_recent_encoding or has_recent_query)


@dataclass
class ConsolidationState:
    """Tracks state of consolidation scheduler."""

    last_consolidation: float = field(default_factory=time.time)
    last_pattern_discovery: float = field(default_factory=time.time)
    consolidation_count: int = 0
    rehearsed_fragments: Set[str] = field(default_factory=set)
    discovered_patterns: List[Tuple[str, str, float]] = field(default_factory=list)

    # Track co-activation for binding strengthening
    coactivation_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # Adaptive scheduling state
    current_interval: float = 60.0  # Current consolidation interval
    activity_monitor: Optional[ActivityMonitor] = None


class AdaptiveScheduler:
    """
    Dynamically adjusts consolidation frequency based on activity.

    High activity (many encodings, important memories) → consolidate more frequently
    Low activity (idle, low salience) → consolidate less frequently
    """

    def __init__(self, config: AdaptiveConsolidationConfig, monitor: ActivityMonitor):
        self.config = config
        self.monitor = monitor

    def calculate_next_interval(self) -> float:
        """
        Calculate optimal consolidation interval based on current activity.

        Returns:
            Interval in seconds until next consolidation
        """
        if not self.config.adaptive_scheduling:
            return self.config.base_interval_seconds

        # High encoding rate → consolidate frequently
        encoding_rate = self.monitor.encoding_rate_per_minute()
        if encoding_rate >= self.config.high_encoding_threshold:
            return self.config.min_interval_seconds

        # Important memories → consolidate quickly
        if self.monitor.has_high_salience_activity(self.config.importance_threshold):
            return self.config.min_interval_seconds

        # Idle system → consolidate less often
        if self.monitor.is_idle():
            return self.config.max_interval_seconds

        # Default interval
        return self.config.base_interval_seconds


class ConsolidationScheduler:
    """
    Autonomous reconstruction scheduler.

    This is the "thinking/meditating" layer that spontaneously reconstructs
    memories to strengthen retrieval paths and discover patterns.

    Without this, memories are just stored but not well-organized for retrieval.
    With this, the system actively builds the semantic network.
    """

    def __init__(
        self,
        store: FragmentStore,
        config: Optional[ConsolidationConfig] = None,
        reconstruction_config: Optional[ReconstructionConfig] = None,
        health_monitor: Optional['MemoryHealthMonitor'] = None
    ):
        self.store = store
        self.config = config or ConsolidationConfig()
        self.reconstruction_config = reconstruction_config or ReconstructionConfig()
        self.state = ConsolidationState()
        self.health_monitor = health_monitor

        # Initialize adaptive scheduling if config supports it
        if isinstance(self.config, AdaptiveConsolidationConfig):
            self.state.activity_monitor = ActivityMonitor()
            self.adaptive_scheduler = AdaptiveScheduler(self.config, self.state.activity_monitor)
            self.state.current_interval = self.config.base_interval_seconds
        else:
            self.adaptive_scheduler = None

    def record_encoding(self, salience: float):
        """
        Record an encoding event for adaptive scheduling.

        Args:
            salience: Salience of encoded fragment
        """
        if self.state.activity_monitor:
            self.state.activity_monitor.record_encoding(salience)

    def record_query(self):
        """Record a query event for adaptive scheduling."""
        if self.state.activity_monitor:
            self.state.activity_monitor.record_query()

    def should_consolidate(self, current_time: Optional[float] = None) -> bool:
        """
        Determine if consolidation should run now.

        Uses adaptive interval if enabled, otherwise fixed interval.

        Args:
            current_time: Current timestamp (defaults to now)

        Returns:
            True if consolidation should run
        """
        if current_time is None:
            current_time = time.time()

        time_since_last = current_time - self.state.last_consolidation

        # Use adaptive interval if available
        if self.adaptive_scheduler:
            self.state.current_interval = self.adaptive_scheduler.calculate_next_interval()
            return time_since_last >= self.state.current_interval
        else:
            return time_since_last >= self.config.CONSOLIDATION_INTERVAL_SECONDS

    def select_rehearsal_candidates(self) -> List[Fragment]:
        """
        Select fragments for spontaneous rehearsal.

        Prioritizes:
        1. Recent fragments (within RECENT_WINDOW_HOURS)
        2. High salience (emotionally/goal-relevant)
        3. Not recently rehearsed

        Returns:
            List of fragments to rehearse
        """
        now = time.time()

        # Get recent salient fragments directly from SQL (avoids loading entire DB)
        recent_fragments = self.store.get_recent_fragments(
            hours=self.config.RECENT_WINDOW_HOURS,
            min_salience=self.config.MIN_SALIENCE_FOR_REHEARSAL
        )

        # Filter out recently rehearsed
        candidates = [
            f for f in recent_fragments
            if f.id not in self.state.rehearsed_fragments
        ]

        # Sort by salience * recency
        def score_fragment(frag: Fragment) -> float:
            recency = (now - frag.created_at) / 3600  # Hours ago
            recency_factor = 1.0 / (1.0 + recency)  # Decay with time
            return frag.initial_salience * 0.7 + recency_factor * 0.3

        candidates.sort(key=score_fragment, reverse=True)

        # Take top N
        selected = candidates[:self.config.REHEARSAL_BATCH_SIZE]

        # Mark as rehearsed
        for frag in selected:
            self.state.rehearsed_fragments.add(frag.id)

        # Clear rehearsed set periodically to allow re-rehearsal
        if len(self.state.rehearsed_fragments) > 20:
            self.state.rehearsed_fragments.clear()

        return selected

    def rehearse_fragment(self, fragment: Fragment) -> None:
        """
        Spontaneously reconstruct around a fragment to strengthen bindings.

        This simulates "thinking about" a memory, which strengthens the
        retrieval paths to it.

        Args:
            fragment: Fragment to rehearse
        """
        # Create query from fragment's semantic content
        semantic_content = fragment.content.get("semantic", "")
        if not semantic_content or not isinstance(semantic_content, str):
            return

        query = Query(semantic=semantic_content)

        # Reconstruct - this activates related fragments
        from .certainty import VarianceController
        variance_controller = VarianceController()

        strand = reconstruct(
            query,
            self.store,
            variance_target=0.2,  # Low variance for consolidation
            config=self.reconstruction_config,
            variance_controller=variance_controller
        )

        # Record co-activations to strengthen bindings
        fragment_ids = strand.fragments
        for i, frag_id_1 in enumerate(fragment_ids):
            for frag_id_2 in fragment_ids[i+1:]:
                pair = tuple(sorted([frag_id_1, frag_id_2]))
                self.state.coactivation_matrix[pair] = \
                    self.state.coactivation_matrix.get(pair, 0) + 1

    def discover_patterns(self) -> List[Tuple[str, str, float]]:
        """
        Discover new semantic patterns during idle time.

        Uses VectorIndex to find similar fragments efficiently instead of
        random sampling with pairwise comparison.

        Returns:
            List of (fragment_id_1, fragment_id_2, similarity) tuples
        """
        discovered = []

        # Get recent salient fragments directly from SQL
        recent = self.store.get_recent_fragments(
            hours=self.config.RECENT_WINDOW_HOURS,
            min_salience=self.config.MIN_SALIENCE_FOR_REHEARSAL
        )

        if len(recent) < 2:
            return discovered

        # Cap to prevent hangs on large databases (1000+ recent fragments)
        if len(recent) > self.config.MAX_PATTERN_DISCOVERY_FRAGMENTS:
            recent = recent[:self.config.MAX_PATTERN_DISCOVERY_FRAGMENTS]

        # For each recent fragment, find its nearest neighbors via VectorIndex
        seen_pairs: Set[Tuple[str, str]] = set()

        for frag in recent:
            # Get the fragment's embedding
            embedding = frag.content.get("semantic")
            if not embedding or not isinstance(embedding, list):
                continue

            query_vec = np.array(embedding, dtype=np.float32)

            # Find top-5 similar fragments
            similar = self.store.find_similar_semantic(query_vec, top_k=6)

            for neighbor_id, similarity in similar:
                # Skip self-match
                if neighbor_id == frag.id:
                    continue

                if similarity < self.config.SEMANTIC_SIMILARITY_THRESHOLD:
                    continue

                # Deduplicate pairs
                pair = tuple(sorted([frag.id, neighbor_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Check if not already bound
                neighbor = self.store.get(neighbor_id)
                if neighbor is None:
                    continue

                if neighbor_id not in frag.bindings and frag.id not in neighbor.bindings:
                    discovered.append((frag.id, neighbor_id, similarity))

        return discovered

    def strengthen_bindings(self) -> int:
        """
        Strengthen bindings based on co-activation patterns.

        When fragments are frequently co-activated (rehearsed together),
        create new bindings between them.

        Returns:
            Number of bindings strengthened
        """
        strengthened = 0

        # Find frequently co-activated pairs
        threshold = 1  # Create binding on first co-activation

        for (frag_id_1, frag_id_2), count in self.state.coactivation_matrix.items():
            if count >= threshold:
                # Get fragments
                frag_1 = self.store.get(frag_id_1)
                frag_2 = self.store.get(frag_id_2)

                if frag_1 is None or frag_2 is None:
                    continue

                # Add bidirectional binding if not present
                if frag_id_2 not in frag_1.bindings:
                    frag_1.bindings.append(frag_id_2)
                    self.store.save(frag_1)
                    strengthened += 1

                if frag_id_1 not in frag_2.bindings:
                    frag_2.bindings.append(frag_id_1)
                    self.store.save(frag_2)
                    strengthened += 1

        return strengthened

    def consolidate(self) -> Dict[str, any]:
        """
        Run one consolidation cycle.

        This is the main autonomous reconstruction process:
        1. Select salient/recent fragments
        2. Rehearse them (spontaneous reconstruction)
        3. Strengthen co-activation bindings
        4. Periodically discover new patterns

        Returns:
            Statistics about consolidation
        """
        now = time.time()
        stats = {
            "rehearsed_count": 0,
            "bindings_strengthened": 0,
            "patterns_discovered": 0,
            "duration_ms": 0
        }

        start_time = time.time()

        # Step 1: Rehearse recent salient fragments
        candidates = self.select_rehearsal_candidates()
        for fragment in candidates:
            self.rehearse_fragment(fragment)
            stats["rehearsed_count"] += 1

        # Step 2: Strengthen bindings based on co-activation
        stats["bindings_strengthened"] = self.strengthen_bindings()

        # Step 3: Periodically discover new patterns
        if (self.state.consolidation_count % self.config.PATTERN_DISCOVERY_INTERVAL == 0):
            patterns = self.discover_patterns()

            # Create bindings for discovered patterns
            for frag_id_1, frag_id_2, similarity in patterns:
                frag_1 = self.store.get(frag_id_1)
                frag_2 = self.store.get(frag_id_2)

                if frag_1 and frag_2:
                    if frag_id_2 not in frag_1.bindings:
                        frag_1.bindings.append(frag_id_2)
                        self.store.save(frag_1)

                    if frag_id_1 not in frag_2.bindings:
                        frag_2.bindings.append(frag_id_1)
                        self.store.save(frag_2)

                    stats["patterns_discovered"] += 1

            self.state.last_pattern_discovery = now

        # Update state
        self.state.last_consolidation = now
        self.state.consolidation_count += 1

        stats["duration_ms"] = int((time.time() - start_time) * 1000)

        # Log consolidation metrics to health monitor
        if self.health_monitor:
            self.health_monitor.log_consolidation_run(
                rehearsals=stats["rehearsed_count"],
                bindings_created=stats["bindings_strengthened"],
                patterns_discovered=stats["patterns_discovered"]
            )

        return stats


def create_consolidation_goals(
    scheduler: ConsolidationScheduler,
    current_time: Optional[float] = None
) -> List[Dict[str, any]]:
    """
    Generate consolidation goals if needed.

    This is called by the ReconstructionEngine to determine if
    autonomous consolidation should run.

    Args:
        scheduler: Consolidation scheduler
        current_time: Current time (defaults to now)

    Returns:
        List of goal payloads for the engine
    """
    if current_time is None:
        current_time = time.time()

    if not scheduler.should_consolidate(current_time):
        return []

    # Return a consolidation goal
    return [{
        "type": "consolidation",
        "priority": 3.0,  # Lower priority than queries/encoding, higher than maintenance
        "scheduler": scheduler
    }]
