"""
Reconstruction Engine - bringing memories back.

This module implements the core reconstruction algorithms:
- Spreading activation from query
- Candidate selection based on activation and strength
- Fragment assembly into coherent output
- Gap filling for missing information
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import random

from .core import Fragment, Strand, Query
from .store import FragmentStore
from .strength import calculate_strength
from .features import extract_semantic_features
from .llm_client import LLMConfig, MemoryLLMClient, get_llm_client

logger = logging.getLogger(__name__)


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
    
    # Step 4: Spread through bindings (optimized with batch loading)
    for depth in range(config.MAX_SPREAD_DEPTH):
        decay = config.ACTIVATION_DECAY ** (depth + 1)

        # Get currently activated fragments, capped to top 50 by activation
        # to prevent exponential blowup through binding network
        active_items = [
            (frag_id, act) for frag_id, act in activation.activations.items()
            if act >= config.MIN_ACTIVATION
        ]
        active_items.sort(key=lambda x: x[1], reverse=True)
        current_active = [frag_id for frag_id, _ in active_items[:50]]

        if not current_active:
            break  # No more fragments to spread from

        # Batch load fragments (single query instead of N queries)
        fragments = store.get_many(current_active)

        # Spread activation through bindings
        for frag_id in current_active:
            fragment = fragments.get(frag_id)
            if fragment is None:
                continue

            current_activation = activation.get(frag_id)
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

    Combines activation with strength, temporal clustering, and binding
    connections to select coherent fragment sets.

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

    # Build candidate list with base scores (optimized with batch loading)
    # Filter by activation threshold first
    candidate_ids = [
        frag_id for frag_id, act in activation.activations.items()
        if act >= config.MIN_ACTIVATION
    ]

    if not candidate_ids:
        return []

    # Batch load candidate fragments (single query)
    fragments_dict = store.get_many(candidate_ids)

    # Calculate scores for all candidates
    candidates = []
    for frag_id in candidate_ids:
        fragment = fragments_dict.get(frag_id)
        if fragment is None:
            continue

        act = activation.get(frag_id)
        strength = calculate_strength(fragment)
        base_score = act * 0.6 + strength * 0.4
        candidates.append((fragment, base_score))

    if not candidates:
        return []

    # Greedy selection with coherence bonus
    selected: List[Fragment] = []
    selected_ids: set = set()
    remaining = candidates.copy()

    while len(selected) < config.MAX_FRAGMENTS and remaining:
        best_idx = -1
        best_total_score = -float('inf')

        for i, (fragment, base_score) in enumerate(remaining):
            # Calculate coherence bonus based on already-selected fragments
            coherence_bonus = 0.0

            if selected:
                # Temporal proximity bonus: prefer fragments close in time
                min_time_gap = min(
                    abs(fragment.created_at - s.created_at)
                    for s in selected
                )
                # Bonus decays with time gap (1 hour = half bonus)
                temporal_bonus = 0.3 / (1.0 + min_time_gap / 3600)
                coherence_bonus += temporal_bonus

                # Binding bonus: prefer fragments connected to selected ones
                binding_count = sum(
                    1 for s in selected
                    if s.id in fragment.bindings or fragment.id in s.bindings
                )
                binding_bonus = 0.2 * min(binding_count / 3, 1.0)
                coherence_bonus += binding_bonus

            # Add noise for variance
            noise = random.gauss(0, config.NOISE_SCALE * variance_target)

            total_score = base_score + coherence_bonus + noise

            if total_score > best_total_score:
                best_total_score = total_score
                best_idx = i

        if best_idx >= 0:
            fragment, _ = remaining.pop(best_idx)
            selected.append(fragment)
            selected_ids.add(fragment.id)

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


def llm_rerank(
    candidates: List[Fragment],
    query: Query,
    llm_config: LLMConfig,
) -> List[Fragment]:
    """
    Rerank candidate fragments using an LLM for semantic relevance.

    Sends fragment texts and the query to the LLM, which scores each
    fragment 0-10. Fragments below rerank_min_score are filtered out.

    On any failure, returns candidates unchanged (graceful degradation).

    Args:
        candidates: Fragments to rerank
        query: The reconstruction query
        llm_config: LLM configuration

    Returns:
        Reranked and filtered list of fragments
    """
    if not candidates or not query.semantic:
        return candidates

    client = get_llm_client(llm_config)
    if not client.is_available():
        return candidates

    # Build numbered list of fragment texts
    fragment_lines = []
    for i, frag in enumerate(candidates):
        text = frag.content.get("text", "")
        if not text:
            text = frag.content.get("semantic", "")
            if isinstance(text, list):
                text = "[embedding]"
        # Truncate long texts
        if len(str(text)) > 200:
            text = str(text)[:200] + "..."
        fragment_lines.append(f"{i + 1}. {text}")

    fragments_text = "\n".join(fragment_lines)

    prompt = (
        f"Query: {query.semantic}\n\n"
        f"Memory fragments:\n{fragments_text}\n\n"
        f"Score each fragment's relevance to the query on a 0-10 scale.\n"
        f"Return ONLY a JSON array: [{{\"index\": 1, \"score\": 8}}, ...]\n"
        f"Include ALL fragments. Be strict — irrelevant fragments get 0-2."
    )

    system = (
        "You are a memory relevance scorer. Given a query and memory fragments, "
        "score each fragment's relevance. Return only valid JSON."
    )

    result = client.generate_json(
        prompt=prompt,
        system=system,
        temperature=llm_config.rerank_temperature,
        timeout=llm_config.rerank_timeout,
    )

    if not result.success or not isinstance(result.parsed, list):
        logger.debug("LLM rerank failed: %s", result.error)
        return candidates

    # Build index→score mapping
    scores: Dict[int, int] = {}
    for item in result.parsed:
        if isinstance(item, dict) and "index" in item and "score" in item:
            try:
                idx = int(item["index"]) - 1  # Convert to 0-based
                score = int(item["score"])
                if 0 <= idx < len(candidates):
                    scores[idx] = score
            except (ValueError, TypeError):
                continue

    # Filter by minimum score and sort by score descending
    reranked = []
    for idx, frag in enumerate(candidates):
        score = scores.get(idx, llm_config.rerank_min_score)  # Default: keep if unscored
        if score >= llm_config.rerank_min_score:
            reranked.append((score, idx, frag))

    reranked.sort(key=lambda x: (-x[0], x[1]))

    if not reranked:
        # Don't filter everything out — return original
        return candidates

    return [frag for _, _, frag in reranked]


def synthesize_narrative(
    fragments: List[Fragment],
    query: Query,
    coherence: float,
    llm_config: LLMConfig,
) -> Optional[str]:
    """
    Synthesize a natural language narrative from reconstructed fragments.

    Creates a 2-5 sentence summary that expresses uncertainty proportional
    to the coherence score. Never fabricates details not in the fragments.

    Args:
        fragments: Assembled fragments
        query: The reconstruction query
        coherence: Coherence score (0-1)
        llm_config: LLM configuration

    Returns:
        Narrative string, or None on failure
    """
    if not fragments:
        return None

    client = get_llm_client(llm_config)
    if not client.is_available():
        return None

    # Build fragment texts
    fragment_texts = []
    for frag in fragments:
        text = frag.content.get("text", "")
        if not text:
            text = frag.content.get("semantic", "")
            if isinstance(text, list):
                continue  # Skip embedding-only fragments
        if text:
            # Truncate very long texts
            if len(str(text)) > 300:
                text = str(text)[:300] + "..."
            fragment_texts.append(str(text))

    if not fragment_texts:
        return None

    fragments_block = "\n---\n".join(fragment_texts)

    coherence_guidance = ""
    if coherence < 0.3:
        coherence_guidance = "Express HIGH uncertainty. These memories are fragmented and may not be connected."
    elif coherence < 0.6:
        coherence_guidance = "Express MODERATE uncertainty. Some connections exist but the picture is incomplete."
    else:
        coherence_guidance = "Express reasonable confidence, but note any gaps."

    query_text = query.semantic or "general recall"

    prompt = (
        f"Query: {query_text}\n"
        f"Coherence: {coherence:.2f}\n\n"
        f"Memory fragments:\n{fragments_block}\n\n"
        f"Synthesize these fragments into a 2-5 sentence narrative. "
        f"{coherence_guidance} "
        f"NEVER fabricate details not present in the fragments."
    )

    system = (
        "You are a memory synthesis engine. Combine memory fragments into "
        "a coherent narrative. Be concise and faithful to the source fragments."
    )

    result = client.generate(
        prompt=prompt,
        system=system,
        temperature=llm_config.synthesis_temperature,
        timeout=llm_config.synthesis_timeout,
    )

    if result.success and result.content:
        return result.content

    logger.debug("LLM synthesis failed: %s", result.error)
    return None


from .certainty import VarianceController

def reconstruct(
    query: Query,
    store: FragmentStore,
    variance_target: float = 0.3,
    config: Optional[ReconstructionConfig] = None,
    variance_controller: Optional[VarianceController] = None,
    llm_config: Optional[LLMConfig] = None
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
        llm_config: Optional LLM configuration for reranking and synthesis

    Returns:
        Reconstructed strand
    """
    if config is None:
        config = ReconstructionConfig()

    # Step 1: Spread activation
    activation = spread_activation(query, store, config)

    # Step 2: Select candidates
    candidates = select_candidates(activation, store, variance_target, config)

    # Step 2.5: LLM reranking (optional)
    if llm_config and llm_config.enable_reranking and candidates:
        candidates = llm_rerank(candidates, query, llm_config)

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

    # Step 5.5: LLM synthesis (optional)
    synthesis = None
    if llm_config and llm_config.enable_synthesis and filled:
        synthesis = synthesize_narrative(filled, query, coherence, llm_config)

    # Step 6: Create strand
    certainty = 0.0

    # Calculate initial strand (needed for variance calculation)
    strand = Strand(
        fragments=[f.id for f in filled],
        assembly_context=assembly_context,
        coherence_score=coherence,
        variance=variance_target,
        certainty=0.0,  # Placeholder
        synthesis=synthesis
    )

    # Calculate certainty if controller available
    if variance_controller:
        query_hash = query.to_hash()
        variance_controller.record_reconstruction(query_hash, strand)
        certainty = variance_controller.get_certainty(query_hash)
        strand.certainty = certainty

    # Record access for rehearsal (batch update instead of N individual get+save)
    import time as time_module
    now = time_module.time()
    if filled:
        store.record_access_batch([f.id for f in filled], now)

    return strand
