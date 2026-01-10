"""
Binding creation for associating fragments.

Bindings are what give memory its relational structure.
"""

from typing import List
from .core import Fragment
from .encoding import Context
from .store import FragmentStore


def find_temporal_bindings(
    fragment: Fragment,
    context: Context,
    max_bindings: int = 5
) -> List[str]:
    """
    Find temporal bindings for a new fragment.
    
    Binds to recent fragments in the context, creating
    temporal associations for later reconstruction.
    
    Args:
        fragment: New fragment being created
        context: Current context with recent fragments
        max_bindings: Maximum number of temporal bindings
        
    Returns:
        List of fragment IDs to bind to
    """
    # Get recent fragments from context
    recent = context.recent_fragments
    
    if not recent or len(recent) == 0:
        return []
    
    # Bind to most recent fragments (up to max)
    bindings = recent[-max_bindings:]
    
    return bindings


def find_semantic_bindings(
    fragment: Fragment,
    store: FragmentStore,
    similarity_threshold: float = 0.7,
    max_bindings: int = 3
) -> List[str]:
    """
    Find semantic bindings based on content similarity.
    
    Binds to semantically related fragments, even if not
    temporally adjacent.
    
    Args:
        fragment: Fragment to find bindings for
        store: Fragment store
        similarity_threshold: Minimum similarity to bind
        max_bindings: Maximum semantic bindings
        
    Returns:
        List of fragment IDs to bind to
    """
    # Check if fragment has semantic content
    if "semantic" not in fragment.content:
        return []
    
    if not isinstance(fragment.content["semantic"], list):
        return []
    
    # Find similar fragments
    import numpy as np
    embedding = np.array(fragment.content["semantic"], dtype=np.float32)
    
    similar = store.find_similar_semantic(embedding, top_k=max_bindings * 2)
    
    # Filter by threshold and exclude self
    bindings = []
    for frag_id, similarity in similar:
        if frag_id == fragment.id:
            continue  # Don't bind to self
        if similarity >= similarity_threshold:
            bindings.append(frag_id)
        if len(bindings) >= max_bindings:
            break
    
    return bindings


def create_bindings(
    fragment: Fragment,
    context: Context,
    store: FragmentStore,
    temporal_bindings: bool = True,
    semantic_bindings: bool = False
) -> List[str]:
    """
    Create all bindings for a fragment.
    
    Combines temporal and semantic bindings.
    
    Args:
        fragment: Fragment to create bindings for
        context: Current context
        store: Fragment store
        temporal_bindings: Whether to create temporal bindings
        semantic_bindings: Whether to create semantic bindings
        
    Returns:
        List of all fragment IDs to bind to
    """
    all_bindings = []
    
    # Temporal bindings (always created during encoding)
    if temporal_bindings:
        temporal = find_temporal_bindings(fragment, context)
        all_bindings.extend(temporal)
    
    # Semantic bindings (optional, more expensive)
    if semantic_bindings:
        semantic = find_semantic_bindings(fragment, store)
        all_bindings.extend(semantic)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_bindings = []
    for frag_id in all_bindings:
        if frag_id not in seen:
            seen.add(frag_id)
            unique_bindings.append(frag_id)
    
    return unique_bindings


def update_bidirectional_bindings(
    fragment_id: str,
    bound_ids: List[str],
    store: FragmentStore
) -> None:
    """
    Update bindings bidirectionally.
    
    When fragment A binds to fragment B, also add A to B's bindings.
    This makes traversal easier during reconstruction.
    
    Args:
        fragment_id: ID of fragment with new bindings
        bound_ids: IDs of fragments it's bound to
        store: Fragment store
    """
    for bound_id in bound_ids:
        bound_fragment = store.get(bound_id)
        if bound_fragment is None:
            continue
        
        # Add reverse binding if not already present
        if fragment_id not in bound_fragment.bindings:
            bound_fragment.bindings.append(fragment_id)
            store.save(bound_fragment)
