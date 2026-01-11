"""
Reconstruction Engine - bringing memories back.

This module implements the core reconstruction algorithms:
- Spreading activation from query
- Candidate selection based on activation and strength
- Fragment assembly into coherent output
- Gap filling for missing information
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import random

from .core import Fragment, Strand, Query
from .store import FragmentStore
from .strength import calculate_strength
from .features import extract_semantic_features


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction."""
    
    # Spreading activation
    ACTIVATION_DECAY: float = 0.7  # Decay per hop
    MAX_SPREAD_DEPTH: int = 3  # Maximum hops from query
    
    # Candidate selection
    MAX_FRAGMENTS: int = 10  # Max fragments in reconstruction
    MIN_ACTIVATION: float = 0.1  # Minimum activation to include
    
    # Noise for variance
    NOISE_SCALE: float = 0.1  # Scale of random noise
    
    # Gap filling
    GAP_THRESHOLD: float = 0.3  # Temporal gap threshold (hours)


@dataclass
class Activation:
    """Activation state during reconstruction."""
    
    activations: Dict[str, float] = field(default_factory=dict)
    
    def activate(self, fragment_id: str, amount: float) -> None:
        """Add activation to a fragment."""
        current = self.activations.get(fragment_id, 0.0)
        self.activations[fragment_id] = min(current + amount, 1.0)
    
    def get(self, fragment_id: str) -> float:
        """Get activation for a fragment."""
        return self.activations.get(fragment_id, 0.0)
    
    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Get top K fragments by activation."""
        sorted_items = sorted(
            self.activations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_items[:k]


def spread_activation(
    query: Query,
    store: FragmentStore,
    config: Optional[ReconstructionConfig] = None
) -> Activation:
    """
    Spread activation from query through fragment network.
    
    This is the core retrieval mechanism:
    1. Find fragments matching query semantically
    2. Spread activation through bindings
    3. Weight by strength (decayed salience)
    
    Args:
        query: Query to reconstruct
        store: Fragment store
        config: Optional configuration
        
    Returns:
        Activation state
    """
    if config is None:
        config = ReconstructionConfig()
    
    activation = Activation()
    
    # Step 1: Semantic activation (if query has semantic content)
    if query.semantic:
        embedding = extract_semantic_features(query.semantic)
        if embedding is not None:
            similar = store.find_similar_semantic(embedding, top_k=20)
            for frag_id, similarity in similar:
                # Weight by similarity
                activation.activate(frag_id, similarity)
    
    # Step 2: Time range activation (if specified)
    if query.time_range is not None:
        time_start, time_end = query.time_range
        temporal_matches = store.find_by_time_range(time_start, time_end)
        for fragment in temporal_matches:
            activation.activate(fragment.id, 0.5)  # Base temporal activation
    
    # Step 3: Domain activation (if specified)
    if query.domains:
        for domain in query.domains:
            domain_matches = store.find_by_domain(domain)
            for fragment in domain_matches:
                activation.activate(fragment.id, 0.3)  # Base domain activation
    
    # Step 4: Spread through bindings
    for depth in range(config.MAX_SPREAD_DEPTH):
        decay = config.ACTIVATION_DECAY ** (depth + 1)
        
        # Get currently activated fragments
        current_active = list(activation.activations.keys())
        
        for frag_id in current_active:
            current_activation = activation.get(frag_id)
            if current_activation < config.MIN_ACTIVATION:
                continue
            
            # Get fragment and spread to bindings
            fragment = store.get(frag_id)
            if fragment is None:
                continue
            
            for bound_id in fragment.bindings:
                spread_amount = current_activation * decay
                activation.activate(bound_id, spread_amount)
    
    return activation


def select_candidates(
    activation: Activation,
    store: FragmentStore,
    variance_target: float = 0.3,
    config: Optional[ReconstructionConfig] = None
) -> List[Fragment]:
    """
    Select candidate fragments for reconstruction.
    
    Combines activation with strength and adds noise for variance.
    
    Args:
        activation: Activation state
        store: Fragment store
        variance_target: Target variance (0=deterministic, 1=random)
        config: Optional configuration
        
    Returns:
        Selected fragments
    """
    if config is None:
        config = ReconstructionConfig()
    
    candidates = []
    
    for frag_id, act in activation.activations.items():
        if act < config.MIN_ACTIVATION:
            continue
        
        fragment = store.get(frag_id)
        if fragment is None:
            continue
        
        # Calculate combined score
        strength = calculate_strength(fragment)
        
        # Add noise based on variance target
        noise = random.gauss(0, config.NOISE_SCALE * variance_target)
        
        score = (act * 0.6 + strength * 0.4) + noise
        
        candidates.append((fragment, score))
    
    # Sort by score and take top K
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [frag for frag, score in candidates[:config.MAX_FRAGMENTS]]
    
    return selected


def temporal_sort(fragments: List[Fragment]) -> List[Fragment]:
    """
    Sort fragments by temporal order.
    
    Args:
        fragments: Fragments to sort
        
    Returns:
        Temporally sorted fragments
    """
    return sorted(fragments, key=lambda f: f.created_at)


def calculate_coherence(fragments: List[Fragment]) -> float:
    """
    Calculate coherence score for a set of fragments.
    
    Higher coherence = fragments form a consistent narrative.
    
    Args:
        fragments: Fragments to evaluate
        
    Returns:
        Coherence score (0-1)
    """
    if len(fragments) <= 1:
        return 1.0
    
    coherence_factors = []
    
    # Factor 1: Temporal continuity
    sorted_frags = temporal_sort(fragments)
    time_gaps = []
    for i in range(1, len(sorted_frags)):
        gap = sorted_frags[i].created_at - sorted_frags[i-1].created_at
        time_gaps.append(gap)
    
    if time_gaps:
        avg_gap = sum(time_gaps) / len(time_gaps)
        # Smaller gaps = higher continuity
        temporal_coherence = 1.0 / (1.0 + avg_gap / 3600)  # Normalize by hour
        coherence_factors.append(temporal_coherence)
    
    # Factor 2: Binding connections
    fragment_ids = {f.id for f in fragments}
    connection_count = 0
    for fragment in fragments:
        for binding in fragment.bindings:
            if binding in fragment_ids:
                connection_count += 1
    
    max_connections = len(fragments) * (len(fragments) - 1)
    if max_connections > 0:
        binding_coherence = connection_count / max_connections
        coherence_factors.append(binding_coherence)
    
    if coherence_factors:
        return sum(coherence_factors) / len(coherence_factors)
    return 0.5


def assemble_fragments(
    fragments: List[Fragment],
    context: dict
) -> dict:
    """
    Assemble fragments into unified content.
    
    Merges content from multiple fragments by domain.
    
    Args:
        fragments: Fragments to assemble
        context: Assembly context
        
    Returns:
        Assembled content by domain
    """
    if not fragments:
        return {}
    
    assembled = {}
    
    # Group content by domain
    for fragment in temporal_sort(fragments):
        for domain, content in fragment.content.items():
            if domain not in assembled:
                assembled[domain] = []
            assembled[domain].append(content)
    
    # Merge each domain
    merged = {}
    for domain, contents in assembled.items():
        if domain == "semantic":
            # Concatenate text/embeddings
            if all(isinstance(c, str) for c in contents):
                merged[domain] = " ".join(contents)
            else:
                merged[domain] = contents  # Keep as list
        elif domain == "emotional":
            # Average emotional states
            merged[domain] = _average_emotional(contents)
        else:
            # Keep as list
            merged[domain] = contents
    
    return merged


def _average_emotional(contents: list) -> dict:
    """Average emotional state dictionaries."""
    if not contents:
        return {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}
    
    avg = {}
    keys = ["valence", "arousal", "dominance"]
    
    for key in keys:
        values = [c.get(key, 0.5) for c in contents if isinstance(c, dict)]
        if values:
            avg[key] = sum(values) / len(values)
        else:
            avg[key] = 0.5
    
    return avg


def fill_gaps(
    fragments: List[Fragment],
    context: dict,
    config: Optional[ReconstructionConfig] = None
) -> List[Fragment]:
    """
    Fill temporal gaps in fragment sequence.
    
    For now, uses simple interpolation. Could be enhanced
    with custom models in Phase 12.
    
    Args:
        fragments: Fragments with potential gaps
        context: Gap filling context
        config: Optional configuration
        
    Returns:
        Fragments with gaps filled (currently just returns input)
    """
    if config is None:
        config = ReconstructionConfig()
    
    if len(fragments) <= 1:
        return fragments
    
    # For now, just return fragments as-is
    # Gap filling with interpolation/generation can be added later
    return fragments


from .certainty import VarianceController

def reconstruct(
    query: Query,
    store: FragmentStore,
    variance_target: float = 0.3,
    config: Optional[ReconstructionConfig] = None,
    variance_controller: Optional[VarianceController] = None
) -> Strand:
    """
    Reconstruct memory from query.
    
    This is THE reconstruction function - the core of the system.
    
    Args:
        query: Query to reconstruct
        store: Fragment store
        variance_target: Target variance (0=deterministic, 1=random)
        config: Optional configuration
        variance_controller: Optional controller for tracking certainty
        
    Returns:
        Reconstructed strand
    """
    if config is None:
        config = ReconstructionConfig()
    
    # Step 1: Spread activation
    activation = spread_activation(query, store, config)
    
    # Step 2: Select candidates
    candidates = select_candidates(activation, store, variance_target, config)
    
    # Step 3: Fill gaps
    filled = fill_gaps(candidates, {"query": query}, config)
    
    # Step 4: Assemble
    assembly_context = {
        "query": query.semantic if query.semantic else "",
        "variance": variance_target
    }
    assembled = assemble_fragments(filled, assembly_context)
    
    # Step 5: Calculate coherence
    coherence = calculate_coherence(filled)
    
    # Step 6: Create strand
    certainty = 0.0
    
    # Calculate initial strand (needed for variance calculation)
    strand = Strand(
        fragments=[f.id for f in filled],
        assembly_context=assembly_context,
        coherence_score=coherence,
        variance=variance_target,
        certainty=0.0  # Placeholder
    )
    
    # Calculate certainty if controller available
    if variance_controller:
        query_hash = query.to_hash()
        variance_controller.record_reconstruction(query_hash, strand)
        certainty = variance_controller.get_certainty(query_hash)
        strand.certainty = certainty
    
    # Record access for rehearsal
    import time as time_module
    now = time_module.time()
    for fragment in filled:
        store.record_access(fragment.id, now)
    
    return strand
