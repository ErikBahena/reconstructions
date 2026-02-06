"""
Salience calculation - OUR algorithm for determining encoding strength.

This is the core of the encoding system - these are our formulas,
not delegated to an LLM.
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from .core import Fragment
from .store import FragmentStore

if TYPE_CHECKING:
    from .learning import SalienceWeightLearner


# Configurable weights for salience calculation
class SalienceConfig:
    """Configuration for salience calculation weights."""

    # Component weights (should sum to ~1.0)
    W_EMOTIONAL = 0.35
    W_NOVELTY = 0.30
    W_GOAL = 0.25
    W_DEPTH = 0.10

    # Novelty calculation
    NOVELTY_SIMILARITY_THRESHOLD = 0.70  # Above this = not novel
    NOVELTY_WINDOW_SIZE = 100  # Recent fragments to check

    # Goal relevance
    GOAL_MATCH_BOOST = 0.3  # Boost for matching goals


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def calculate_encoding_salience(
    emotional_intensity: float,
    novelty: float,
    goal_relevance: float,
    processing_depth: float,
    config: Optional[SalienceConfig] = None,
    weight_learner: Optional['SalienceWeightLearner'] = None
) -> float:
    """
    Calculate encoding salience - OUR FORMULA.

    This determines how strongly an experience is encoded.
    Based on cognitive science but we own the weights.

    If weight_learner is provided, uses learned weights instead of config.

    Args:
        emotional_intensity: How emotionally intense (0-1)
        novelty: How novel/unexpected (0-1)
        goal_relevance: How relevant to active goals (0-1)
        processing_depth: How deeply processed (0-1)
        config: Optional custom configuration
        weight_learner: Optional weight learner with adaptive weights

    Returns:
        Salience value (0-1)
    """
    # Use learned weights if available, otherwise use config
    if weight_learner is not None:
        weights = weight_learner.get_current_weights()
        w_emotional = weights["emotional"]
        w_novelty = weights["novelty"]
        w_goal = weights["goal"]
        w_depth = weights["depth"]
    else:
        if config is None:
            config = SalienceConfig()
        w_emotional = config.W_EMOTIONAL
        w_novelty = config.W_NOVELTY
        w_goal = config.W_GOAL
        w_depth = config.W_DEPTH

    salience = (
        w_emotional * emotional_intensity +
        w_novelty * novelty +
        w_goal * goal_relevance +
        w_depth * processing_depth
    )

    return clamp(salience, 0.0, 1.0)


def calculate_emotional_intensity(emotional_features: dict) -> float:
    """
    Calculate emotional intensity from emotional features.
    
    Uses arousal as primary measure, with valence extremes
    also contributing (both very positive and very negative
    emotions are intense).
    
    Args:
        emotional_features: Dict with valence, arousal, dominance
        
    Returns:
        Intensity (0-1)
    """
    arousal = emotional_features.get("arousal", 0.5)
    valence = emotional_features.get("valence", 0.5)
    
    # Arousal is primary component
    intensity = arousal
    
    # Valence extremes add to intensity
    valence_distance_from_neutral = abs(valence - 0.5)
    intensity += valence_distance_from_neutral * 0.3
    
    return clamp(intensity, 0.0, 1.0)


def calculate_novelty(
    content: dict,
    store: FragmentStore,
    config: Optional[SalienceConfig] = None
) -> float:
    """
    Calculate novelty - how different from existing memories.
    
    High prediction error = high novelty.
    This is OUR novelty detection algorithm.
    
    Args:
        content: Fragment content to check
        store: Fragment store to check against
        config: Optional custom configuration
        
    Returns:
        Novelty score (0-1)
    """
    if config is None:
        config = SalienceConfig()
    
    # If store is empty, everything is novel
    if store.is_empty():
        return 1.0
    
    # Check for semantic embedding
    if "semantic" not in content or not isinstance(content["semantic"], list):
        # No semantic content - use moderate novelty
        return 0.5
    
    # Find similar fragments using semantic embedding
    query_embedding = np.array(content["semantic"], dtype=np.float32)
    similar = store.find_similar_semantic(
        query_embedding, 
        top_k=min(config.NOVELTY_WINDOW_SIZE, 10)
    )
    
    if len(similar) == 0:
        return 1.0
    
    # Get highest similarity score
    max_similarity = similar[0][1]  # (id, similarity)
    
    # Convert similarity to novelty
    # If very similar (>threshold), low novelty
    if max_similarity > config.NOVELTY_SIMILARITY_THRESHOLD:
        novelty = 1.0 - max_similarity
    else:
        # Below threshold, consider it novel
        novelty = 0.5 + (1.0 - max_similarity) * 0.5
    
    return clamp(novelty, 0.0, 1.0)


def calculate_goal_relevance(
    content: dict,
    active_goals: List[str],
    config: Optional[SalienceConfig] = None
) -> float:
    """
    Calculate goal relevance - how related to active goals.
    
    Args:
        content: Fragment content
        active_goals: List of active goal strings
        config: Optional custom configuration
        
    Returns:
        Goal relevance (0-1)
    """
    if config is None:
        config = SalienceConfig()
    
    # No active goals = low relevance (creates headroom for goal-relevant fragments)
    if not active_goals or len(active_goals) == 0:
        return 0.3

    # Check semantic content
    if "semantic" not in content:
        return 0.3
    
    # Simple keyword matching for now
    # (Could be enhanced with semantic similarity in future)
    semantic_text = str(content.get("semantic", "")).lower()
    
    matches = 0
    for goal in active_goals:
        goal_lower = goal.lower()
        if goal_lower in semantic_text:
            matches += 1
    
    if matches > 0:
        # Base relevance + boost for matches
        relevance = 0.5 + (matches / len(active_goals)) * config.GOAL_MATCH_BOOST
        return clamp(relevance, 0.0, 1.0)
    
    # No matches = moderate relevance
    return 0.5


def calculate_salience_for_fragment(
    content: dict,
    emotional_features: dict,
    processing_depth: float,
    active_goals: List[str],
    store: FragmentStore,
    config: Optional[SalienceConfig] = None,
    weight_learner: Optional['SalienceWeightLearner'] = None
) -> float:
    """
    Calculate salience for a fragment - complete pipeline.

    This is the main entry point for salience calculation.

    Args:
        content: Fragment content
        emotional_features: Emotional state
        processing_depth: How deeply processed
        active_goals: Active goals
        store: Fragment store
        config: Optional custom configuration
        weight_learner: Optional weight learner with adaptive weights

    Returns:
        Salience score (0-1)
    """
    # Calculate components
    emotional_intensity = calculate_emotional_intensity(emotional_features)
    novelty = calculate_novelty(content, store, config)
    goal_relevance = calculate_goal_relevance(content, active_goals, config)

    # Combine into final salience (use learned weights if available)
    salience = calculate_encoding_salience(
        emotional_intensity=emotional_intensity,
        novelty=novelty,
        goal_relevance=goal_relevance,
        processing_depth=processing_depth,
        config=config,
        weight_learner=weight_learner
    )
    
    return salience
