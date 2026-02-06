"""
Complete encoding pipeline - brings everything together.

This is the main entry point for encoding experiences into fragments.
"""

from typing import Optional, TYPE_CHECKING
from .core import Fragment
from .encoding import Experience, Context
from .store import FragmentStore
from .features import extract_all_features
from .salience import calculate_salience_for_fragment
from .bindings import create_bindings, update_bidirectional_bindings

if TYPE_CHECKING:
    from .learning import SalienceWeightLearner
    from .identity import ActiveIdentityState


def encode(
    experience: Experience,
    context: Context,
    store: FragmentStore,
    create_semantic_bindings: bool = False,
    identity_state: Optional['ActiveIdentityState'] = None,
    weight_learner: Optional['SalienceWeightLearner'] = None
) -> Fragment:
    """
    Encode an experience into a fragment.

    This is THE encoding function - it brings together all components:
    1. Extract features from experience
    2. Calculate salience (using learned weights if available)
    3. Apply identity-aware boost (if identity_state provided)
    4. Create fragment
    5. Create bindings
    6. Update context
    7. Save to store

    Args:
        experience: Experience to encode
        context: Current context
        store: Fragment store
        create_semantic_bindings: Whether to create semantic bindings (expensive)
        identity_state: Optional active identity state for relevance boosting
        weight_learner: Optional weight learner for adaptive salience calculation

    Returns:
        Encoded fragment
    """

    # 1. EXTRACT FEATURES
    content = extract_all_features(experience, context)

    # 2. CALCULATE BASE SALIENCE (using learned weights if available)
    emotional_features = content.get("emotional", {
        "valence": 0.5,
        "arousal": 0.5,
        "dominance": 0.5
    })

    salience = calculate_salience_for_fragment(
        content=content,
        emotional_features=emotional_features,
        processing_depth=context.processing_depth,
        active_goals=context.active_goals,
        store=store,
        weight_learner=weight_learner
    )

    # 3. CREATE TEMPORARY FRAGMENT FOR IDENTITY EVALUATION
    temp_fragment = Fragment(
        content=content,
        initial_salience=salience,
        source=experience.source,
        tags=experience.tags
    )

    # 4. APPLY IDENTITY-AWARE BOOST
    if identity_state is not None:
        identity_boost = identity_state.relevance_boost(temp_fragment)
        salience = min(salience + identity_boost, 1.0)

    # 5. CREATE FINAL FRAGMENT WITH BOOSTED SALIENCE
    fragment = Fragment(
        content=content,
        initial_salience=salience,
        source=experience.source,
        tags=experience.tags
    )
    
    # 6. CREATE BINDINGS
    bindings = create_bindings(
        fragment=fragment,
        context=context,
        store=store,
        temporal_bindings=True,
        semantic_bindings=create_semantic_bindings
    )

    fragment.bindings = bindings

    # 7. SAVE FRAGMENT
    store.save(fragment)
    
    # 6. UPDATE BIDIRECTIONAL BINDINGS
    if bindings:
        update_bidirectional_bindings(fragment.id, bindings, store)
    
    # 7. UPDATE CONTEXT
    context.add_recent_fragment(fragment.id)
    context.increment_sequence()
    
    return fragment


def encode_batch(
    experiences: list[Experience],
    context: Context,
    store: FragmentStore,
    create_semantic_bindings: bool = False
) -> list[Fragment]:
    """
    Encode multiple experiences in sequence.
    
    More efficient than encoding one at a time because it updates
    context state progressively.
    
    Args:
        experiences: List of experiences to encode
        context: Current context (will be updated)
        store: Fragment store
        create_semantic_bindings: Whether to create semantic bindings
        
    Returns:
        List of encoded fragments
    """
    fragments = []
    
    for experience in experiences:
        fragment = encode(
            experience=experience,
            context=context,
            store=store,
            create_semantic_bindings=create_semantic_bindings
        )
        fragments.append(fragment)
    
    return fragments
