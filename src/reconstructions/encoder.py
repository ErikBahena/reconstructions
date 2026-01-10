"""
Complete encoding pipeline - brings everything together.

This is the main entry point for encoding experiences into fragments.
"""

from .core import Fragment
from .encoding import Experience, Context
from .store import FragmentStore
from .features import extract_all_features
from .salience import calculate_salience_for_fragment
from .bindings import create_bindings, update_bidirectional_bindings


def encode(
    experience: Experience,
    context: Context,
    store: FragmentStore,
    create_semantic_bindings: bool = False
) -> Fragment:
    """
    Encode an experience into a fragment.
    
    This is THE encoding function - it brings together all Phase 2 components:
    1. Extract features from experience
    2. Calculate salience
    3. Create fragment
    4. Create bindings
    5. Update context
    6. Save to store
    
    Args:
        experience: Experience to encode
        context: Current context
        store: Fragment store
        create_semantic_bindings: Whether to create semantic bindings (expensive)
        
    Returns:
        Encoded fragment
    """
    
    # 1. EXTRACT FEATURES
    content = extract_all_features(experience, context)
    
    # 2. CALCULATE SALIENCE
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
        store=store
    )
    
    # 3. CREATE FRAGMENT
    fragment = Fragment(
        content=content,
        initial_salience=salience,
        source=experience.source,
        tags=experience.tags
    )
    
    # 4. CREATE BINDINGS
    bindings = create_bindings(
        fragment=fragment,
        context=context,
        store=store,
        temporal_bindings=True,
        semantic_bindings=create_semantic_bindings
    )
    
    fragment.bindings = bindings
    
    # 5. SAVE FRAGMENT
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
