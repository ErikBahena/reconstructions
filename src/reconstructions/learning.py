"""
Self-Tuning Salience Weight Learning.

Learns optimal salience weights from retrieval feedback to improve
memory quality over time.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SalienceWeightLearner:
    """
    Learns optimal salience weights from retrieval feedback.

    Uses gradient-based learning to adjust weights based on which
    fragments are retrieved in successful queries.
    """

    # Current weights (must sum to 1.0)
    w_emotional: float = 0.25
    w_novelty: float = 0.15
    w_goal: float = 0.30
    w_depth: float = 0.30

    # Learning parameters
    learning_rate: float = 0.01
    min_weight: float = 0.05
    max_weight: float = 0.50

    # Training statistics
    total_feedback: int = 0
    successful_retrievals: int = 0
    failed_retrievals: int = 0

    def record_retrieval(self, fragment: 'Fragment', was_useful: bool):
        """
        Learn from retrieval feedback.

        Args:
            fragment: Fragment that was retrieved
            was_useful: Whether fragment was in a successful query (coherence > 0.5)
        """
        self.total_feedback += 1

        if was_useful:
            self.successful_retrievals += 1
            self._reinforce_contributing_factors(fragment)
        else:
            self.failed_retrievals += 1
            self._penalize_contributing_factors(fragment)

        # Normalize after update
        self._normalize()

    def _reinforce_contributing_factors(self, fragment: 'Fragment'):
        """
        Increase weights for factors that contributed to this fragment's salience.

        Args:
            fragment: Fragment to analyze
        """
        # Extract features from fragment
        emotional_intensity = self._get_emotional_intensity(fragment)
        has_novelty = self._has_high_novelty(fragment)
        has_goal_relevance = self._has_goal_relevance(fragment)
        processing_depth = fragment.content.get("depth", 0.5)

        # Increase weights proportional to feature presence
        if emotional_intensity > 0.5:
            self.w_emotional += self.learning_rate * emotional_intensity

        if has_novelty:
            self.w_novelty += self.learning_rate

        if has_goal_relevance:
            self.w_goal += self.learning_rate

        if processing_depth > 0.5:
            self.w_depth += self.learning_rate * processing_depth

    def _penalize_contributing_factors(self, fragment: 'Fragment'):
        """
        Decrease weights for factors in low-quality retrievals.

        Args:
            fragment: Fragment to analyze
        """
        # Extract features
        emotional_intensity = self._get_emotional_intensity(fragment)
        has_novelty = self._has_high_novelty(fragment)
        has_goal_relevance = self._has_goal_relevance(fragment)
        processing_depth = fragment.content.get("depth", 0.5)

        # Decrease weights slightly
        if emotional_intensity > 0.5:
            self.w_emotional -= self.learning_rate * 0.5 * emotional_intensity

        if has_novelty:
            self.w_novelty -= self.learning_rate * 0.5

        if has_goal_relevance:
            self.w_goal -= self.learning_rate * 0.5

        if processing_depth > 0.5:
            self.w_depth -= self.learning_rate * 0.5 * processing_depth

    def _get_emotional_intensity(self, fragment: 'Fragment') -> float:
        """
        Get emotional intensity of fragment.

        Args:
            fragment: Fragment to analyze

        Returns:
            Emotional intensity (0-1)
        """
        emotional = fragment.content.get("emotional", {})
        if not emotional:
            return 0.0

        arousal = emotional.get("arousal", 0.5)
        valence = emotional.get("valence", 0.5)

        # High arousal OR extreme valence = high emotional intensity
        valence_extremity = abs(valence - 0.5) * 2  # 0-1
        return max(arousal, valence_extremity)

    def _has_high_novelty(self, fragment: 'Fragment') -> bool:
        """
        Check if fragment has high novelty.

        Args:
            fragment: Fragment to check

        Returns:
            True if high novelty
        """
        # Novelty is computed during salience calculation
        # For now, heuristic: low access count = high novelty
        return len(fragment.access_log) < 3

    def _has_goal_relevance(self, fragment: 'Fragment') -> bool:
        """
        Check if fragment has goal relevance.

        Args:
            fragment: Fragment to check

        Returns:
            True if goal-relevant
        """
        # Check if fragment has high goal-relevance features
        # For now, check if it has goal-related tags or context
        tags = fragment.tags
        return any("goal" in tag.lower() for tag in tags)

    def _normalize(self):
        """
        Normalize weights to sum to 1.0 and stay within bounds.
        """
        # First normalize to sum to 1.0
        total = self.w_emotional + self.w_novelty + self.w_goal + self.w_depth
        if total > 0:
            self.w_emotional /= total
            self.w_novelty /= total
            self.w_goal /= total
            self.w_depth /= total

        # Then clamp to bounds
        self.w_emotional = np.clip(self.w_emotional, self.min_weight, self.max_weight)
        self.w_novelty = np.clip(self.w_novelty, self.min_weight, self.max_weight)
        self.w_goal = np.clip(self.w_goal, self.min_weight, self.max_weight)
        self.w_depth = np.clip(self.w_depth, self.min_weight, self.max_weight)

        # Renormalize after clamping (might not sum to 1.0 after clipping)
        total = self.w_emotional + self.w_novelty + self.w_goal + self.w_depth
        if total > 0:
            self.w_emotional /= total
            self.w_novelty /= total
            self.w_goal /= total
            self.w_depth /= total

    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current weight configuration.

        Returns:
            Dictionary of weight names to values
        """
        return {
            "emotional": self.w_emotional,
            "novelty": self.w_novelty,
            "goal": self.w_goal,
            "depth": self.w_depth
        }

    def save_checkpoint(self, path: Path):
        """
        Persist learned weights to disk.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = {
            "weights": self.get_current_weights(),
            "learning_rate": self.learning_rate,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "total_feedback": self.total_feedback,
            "successful_retrievals": self.successful_retrievals,
            "failed_retrievals": self.failed_retrievals
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: Path) -> 'SalienceWeightLearner':
        """
        Restore learned weights from disk.

        Args:
            path: Path to checkpoint file

        Returns:
            SalienceWeightLearner with restored weights
        """
        with open(path, "r") as f:
            checkpoint = json.load(f)

        weights = checkpoint["weights"]
        learner = cls(
            w_emotional=weights["emotional"],
            w_novelty=weights["novelty"],
            w_goal=weights["goal"],
            w_depth=weights["depth"],
            learning_rate=checkpoint["learning_rate"],
            min_weight=checkpoint["min_weight"],
            max_weight=checkpoint["max_weight"],
            total_feedback=checkpoint["total_feedback"],
            successful_retrievals=checkpoint["successful_retrievals"],
            failed_retrievals=checkpoint["failed_retrievals"]
        )

        return learner

    def get_success_rate(self) -> float:
        """
        Calculate success rate of retrievals.

        Returns:
            Success rate (0-1)
        """
        if self.total_feedback == 0:
            return 0.0
        return self.successful_retrievals / self.total_feedback
