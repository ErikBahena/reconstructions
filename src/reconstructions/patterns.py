"""
Cross-Session Pattern Recognition.

Discovers recurring patterns across sessions to enable proactive
memory organization and context loading.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict, Counter

from .core import Fragment
from .store import FragmentStore
from .features import extract_semantic_features


@dataclass
class TemporalPattern:
    """Recurring pattern based on time."""

    pattern_type: str  # "daily", "weekly", "monthly"
    description: str
    confidence: float  # 0-1
    examples: List[str] = field(default_factory=list)  # Fragment IDs
    time_signature: Dict[str, float] = field(default_factory=dict)  # e.g., {"day_of_week": 2, "hour": 14}

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "examples": self.examples,
            "time_signature": self.time_signature
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalPattern':
        """Deserialize from dictionary."""
        return cls(
            pattern_type=data["pattern_type"],
            description=data["description"],
            confidence=data["confidence"],
            examples=data.get("examples", []),
            time_signature=data.get("time_signature", {})
        )


@dataclass
class WorkflowPattern:
    """Common sequence of operations."""

    steps: List[str]  # Sequence of operation keywords
    frequency: int  # How many times observed
    avg_duration_minutes: float
    examples: List[List[str]] = field(default_factory=list)  # Fragment ID sequences

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "steps": self.steps,
            "frequency": self.frequency,
            "avg_duration_minutes": self.avg_duration_minutes,
            "examples": self.examples
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowPattern':
        """Deserialize from dictionary."""
        return cls(
            steps=data["steps"],
            frequency=data["frequency"],
            avg_duration_minutes=data["avg_duration_minutes"],
            examples=data.get("examples", [])
        )


@dataclass
class ProjectPattern:
    """Semantic cluster of related work."""

    project_name: str
    keywords: List[str]
    fragment_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None  # Semantic centroid

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "project_name": self.project_name,
            "keywords": self.keywords,
            "fragment_ids": self.fragment_ids,
            "centroid": self.centroid.tolist() if self.centroid is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProjectPattern':
        """Deserialize from dictionary."""
        centroid = None
        if data.get("centroid"):
            centroid = np.array(data["centroid"])

        return cls(
            project_name=data["project_name"],
            keywords=data["keywords"],
            fragment_ids=data.get("fragment_ids", []),
            centroid=centroid
        )


class CrossSessionPatternDetector:
    """Discovers recurring patterns across sessions."""

    def __init__(self, store: FragmentStore, patterns_path: Optional[Path] = None):
        self.store = store
        self.patterns_path = patterns_path or (Path.home() / ".reconstructions" / "patterns.json")

        # Cached patterns
        self.temporal_patterns: List[TemporalPattern] = []
        self.workflow_patterns: List[WorkflowPattern] = []
        self.project_patterns: List[ProjectPattern] = []

        # Load existing patterns
        self._load_patterns()

    def detect_temporal_patterns(self, min_confidence: float = 0.7) -> List[TemporalPattern]:
        """
        Find time-based patterns (e.g., daily routines).

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected temporal patterns
        """
        patterns = []

        # Get all fragments
        all_fragments = self.store.get_all_fragments()

        if len(all_fragments) < 10:
            return patterns  # Not enough data

        # Group by day of week
        day_clusters = defaultdict(list)
        for fragment in all_fragments:
            dt = datetime.fromtimestamp(fragment.created_at)
            day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
            day_clusters[day_of_week].append(fragment)

        # Find days with significant activity
        total_fragments = len(all_fragments)
        for day, fragments in day_clusters.items():
            if len(fragments) < 5:  # Need at least 5 fragments
                continue

            # Calculate confidence based on frequency
            confidence = len(fragments) / total_fragments

            if confidence < min_confidence / 7:  # Adjust for 7 days
                continue

            # Extract common semantic themes
            themes = self._extract_themes(fragments)
            if themes:
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day]

                pattern = TemporalPattern(
                    pattern_type="weekly",
                    description=f"{day_name}: {', '.join(themes[:3])}",
                    confidence=min(confidence * 7, 1.0),  # Scale up
                    examples=[f.id for f in fragments[:5]],
                    time_signature={"day_of_week": day}
                )
                patterns.append(pattern)

        self.temporal_patterns = patterns
        return patterns

    def detect_workflow_patterns(self, min_frequency: int = 3) -> List[WorkflowPattern]:
        """
        Identify common operation sequences.

        Args:
            min_frequency: Minimum times sequence must occur

        Returns:
            List of detected workflow patterns
        """
        patterns = []

        # Get all fragments sorted by time
        all_fragments = sorted(
            self.store.get_all_fragments(),
            key=lambda f: f.created_at
        )

        if len(all_fragments) < 10:
            return patterns

        # Extract operation keywords from fragments
        operations = []
        for fragment in all_fragments:
            semantic = fragment.content.get("semantic", "")
            if isinstance(semantic, str):
                # Extract key action words
                keywords = self._extract_keywords(semantic)
                if keywords:
                    operations.append((keywords[0], fragment))  # Use first keyword

        # Find common n-grams (sequences of length 2-4)
        for n in range(2, 5):
            ngrams = defaultdict(list)

            for i in range(len(operations) - n + 1):
                # Get sequence of operations
                sequence = tuple(op[0] for op in operations[i:i+n])
                fragment_ids = [op[1].id for op in operations[i:i+n]]

                # Check time window (operations should be within 1 hour)
                time_span = operations[i+n-1][1].created_at - operations[i][1].created_at
                if time_span > 3600:  # 1 hour
                    continue

                ngrams[sequence].append((fragment_ids, time_span))

            # Find frequent patterns
            for sequence, instances in ngrams.items():
                if len(instances) >= min_frequency:
                    avg_duration = sum(ts for _, ts in instances) / len(instances) / 60  # Minutes

                    pattern = WorkflowPattern(
                        steps=list(sequence),
                        frequency=len(instances),
                        avg_duration_minutes=avg_duration,
                        examples=[fids for fids, _ in instances[:5]]
                    )
                    patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        self.workflow_patterns = patterns[:10]  # Keep top 10
        return self.workflow_patterns

    def detect_project_switches(self, similarity_threshold: float = 0.6) -> List[ProjectPattern]:
        """
        Identify semantic clusters representing different projects.

        Args:
            similarity_threshold: Minimum similarity for clustering

        Returns:
            List of detected project patterns
        """
        patterns = []

        # Get all fragments with semantic content
        all_fragments = self.store.get_all_fragments()
        semantic_fragments = []
        embeddings = []

        for fragment in all_fragments:
            semantic = fragment.content.get("semantic", "")
            if semantic and isinstance(semantic, str):
                embedding = extract_semantic_features(semantic)
                if embedding is not None:
                    semantic_fragments.append(fragment)
                    embeddings.append(embedding)

        if len(embeddings) < 5:
            return patterns

        embeddings = np.array(embeddings)

        # Simple clustering: find dense regions
        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Find clusters (fragments with high similarity to many others)
        clustered = set()

        for i in range(len(embeddings)):
            if i in clustered:
                continue

            # Find all fragments similar to this one
            similar = [j for j in range(len(embeddings))
                      if j not in clustered and similarities[i, j] > similarity_threshold]

            if len(similar) >= 3:  # Need at least 3 fragments in cluster
                # Create project pattern
                cluster_fragments = [semantic_fragments[j] for j in similar]
                keywords = self._extract_cluster_keywords(cluster_fragments)

                # Compute centroid
                centroid = embeddings[similar].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)

                pattern = ProjectPattern(
                    project_name=keywords[0] if keywords else f"Project {len(patterns)+1}",
                    keywords=keywords[:5],
                    fragment_ids=[f.id for f in cluster_fragments],
                    centroid=centroid
                )
                patterns.append(pattern)

                # Mark as clustered
                clustered.update(similar)

        self.project_patterns = patterns
        return patterns

    def _extract_themes(self, fragments: List[Fragment]) -> List[str]:
        """Extract common themes from fragments."""
        keywords = []
        for fragment in fragments:
            semantic = fragment.content.get("semantic", "")
            if isinstance(semantic, str):
                keywords.extend(self._extract_keywords(semantic))

        # Count most common
        if not keywords:
            return []

        counter = Counter(keywords)
        return [word for word, count in counter.most_common(5)]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key action/topic words from text."""
        # Simple keyword extraction (could be improved with NLP)
        action_words = ["git", "status", "add", "commit", "push", "pull", "test", "run", "build",
                       "debug", "fix", "implement", "create", "update", "delete", "read", "write",
                       "streaming", "rtmp", "auth", "database", "api", "server", "client"]

        words = text.lower().split()
        return [w for w in words if w in action_words]

    def _extract_cluster_keywords(self, fragments: List[Fragment]) -> List[str]:
        """Extract representative keywords from cluster."""
        all_text = []
        for fragment in fragments:
            semantic = fragment.content.get("semantic", "")
            if isinstance(semantic, str):
                all_text.append(semantic.lower())

        combined = " ".join(all_text)
        words = combined.split()

        # Count word frequency
        counter = Counter(words)

        # Filter common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word, count in counter.most_common(20)
                   if len(word) > 3 and word not in stopwords]

        return keywords[:10]

    def save_patterns(self):
        """Persist detected patterns to disk."""
        data = {
            "temporal_patterns": [p.to_dict() for p in self.temporal_patterns],
            "workflow_patterns": [p.to_dict() for p in self.workflow_patterns],
            "project_patterns": [p.to_dict() for p in self.project_patterns]
        }

        self.patterns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.patterns_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_patterns(self):
        """Load patterns from disk."""
        if not self.patterns_path.exists():
            return

        try:
            with open(self.patterns_path, "r") as f:
                data = json.load(f)

            self.temporal_patterns = [
                TemporalPattern.from_dict(p) for p in data.get("temporal_patterns", [])
            ]
            self.workflow_patterns = [
                WorkflowPattern.from_dict(p) for p in data.get("workflow_patterns", [])
            ]
            self.project_patterns = [
                ProjectPattern.from_dict(p) for p in data.get("project_patterns", [])
            ]
        except Exception:
            # If loading fails, start fresh
            pass

    def get_all_patterns(self) -> Dict[str, List]:
        """Get all detected patterns."""
        return {
            "temporal": self.temporal_patterns,
            "workflow": self.workflow_patterns,
            "project": self.project_patterns
        }
