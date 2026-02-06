"""
Feature extraction functions for encoding experiences.
"""

import logging
import numpy as np
from typing import Optional, List, TYPE_CHECKING
from .encoding import Experience, Context

if TYPE_CHECKING:
    from .llm_client import LLMConfig

from .llm_client import get_llm_client

logger = logging.getLogger(__name__)


# Global embedder (lazy loaded)
_embedder = None
_embedder_type = None  # 'fast', 'sentence_transformers', or 'fallback'


def get_embedder():
    """
    Get or load the embedder.

    Prefers FastEmbedder (ONNX), falls back to sentence-transformers,
    then to hash-based fallback.
    Lazy loaded to avoid startup overhead.

    Returns:
        Tuple of (embedder, type) where type is 'fast', 'sentence_transformers', or 'fallback'
    """
    global _embedder, _embedder_type

    if _embedder is not None:
        return _embedder, _embedder_type

    # Try FastEmbedder first (ONNX-based, fastest)
    try:
        from .fast_embedder import FastEmbedder
        _embedder = FastEmbedder()
        _embedder_type = 'fast'
        return _embedder, _embedder_type
    except Exception:
        pass

    # Fall back to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        _embedder_type = 'sentence_transformers'
        return _embedder, _embedder_type
    except Exception:
        pass

    # No embedder available, will use hash fallback
    _embedder_type = 'fallback'
    return None, _embedder_type


def extract_semantic_features(text: str) -> Optional[np.ndarray]:
    """
    Extract semantic features from text using embeddings.

    Uses FastEmbedder (ONNX) for speed, falls back to sentence-transformers
    or hash-based pseudo-embeddings if unavailable.

    Args:
        text: Input text to encode

    Returns:
        Embedding vector as numpy array, or None if text is empty
    """
    if not text or len(text.strip()) == 0:
        return None

    embedder, embedder_type = get_embedder()

    if embedder_type == 'fast':
        # Use FastEmbedder
        try:
            return embedder.embed(text)
        except Exception:
            # Fall through to fallback
            pass
    elif embedder_type == 'sentence_transformers':
        # Use sentence-transformers
        try:
            embedding = embedder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception:
            # Fall through to fallback
            pass

    # Fallback: create a simple hash-based pseudo-embedding
    # This is NOT semantically meaningful, just for testing
    return _fallback_text_embedding(text)


def _fallback_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Fallback pseudo-embedding when sentence-transformers unavailable.
    
    NOT semantically meaningful - just for testing/development.
    """
    # Use hash to generate deterministic "embedding"
    text_hash = hash(text.lower())
    np.random.seed(text_hash % (2**31))
    embedding = np.random.randn(dim).astype(np.float32)
    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding


def extract_emotional_features(
    emotional_data: Optional[dict]
) -> dict[str, float]:
    """
    Extract emotional features from experience.
    
    Expects valence, arousal, and optionally dominance.
    If not provided, returns neutral values.
    
    Args:
        emotional_data: Dictionary with emotional state
        
    Returns:
        Dictionary with valence, arousal, dominance (all 0.0-1.0)
    """
    if emotional_data is None or len(emotional_data) == 0:
        # Neutral emotional state
        return {
            "valence": 0.5,
            "arousal": 0.5,
            "dominance": 0.5
        }
    
    return {
        "valence": float(emotional_data.get("valence", 0.5)),
        "arousal": float(emotional_data.get("arousal", 0.5)),
        "dominance": float(emotional_data.get("dominance", 0.5))
    }


def extract_temporal_features(context: Context) -> dict[str, float]:
    """
    Extract temporal features from context.
    
    Args:
        context: Current context
        
    Returns:
        Dictionary with temporal features
    """
    import time
    
    return {
        "absolute": time.time(),
        "sequence_position": float(context.sequence_counter),
        "context_id": hash(context.id) % (2**31),  # Numeric ID for storage
    }


def compress_text_with_llm(raw_text: str, llm_config: 'LLMConfig') -> Optional[str]:
    """
    Compress raw experience text into a concise summary using an LLM.

    Produces a 1-2 sentence summary focusing on WHAT was done and WHY,
    stripping file paths, command syntax, and other noise.

    Args:
        raw_text: Raw experience text to compress
        llm_config: LLM configuration

    Returns:
        Compressed summary string, or None on failure
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return None  # Too short to compress

    client = get_llm_client(llm_config)
    if not client.is_available():
        return None

    prompt = (
        f"Text:\n{raw_text[:500]}\n\n"
        f"Summarize this developer activity in 1-2 concise sentences. "
        f"Focus on WHAT was done and WHY, not file paths or command syntax."
    )

    system = (
        "You are a memory compression engine. Produce concise, information-dense "
        "summaries. Strip noise, keep meaning. Reply with ONLY the summary."
    )

    result = client.generate(
        prompt=prompt,
        system=system,
        temperature=llm_config.compress_temperature,
        timeout=llm_config.compress_timeout,
    )

    if result.success and result.content:
        return result.content

    logger.debug("LLM compression failed: %s", result.error)
    return None


def extract_all_features(experience: Experience, context: Context, llm_config: Optional['LLMConfig'] = None) -> dict:
    """
    Extract all features from an experience.

    This is the main entry point for feature extraction.

    Args:
        experience: Experience to extract features from
        context: Current context
        llm_config: Optional LLM config for text compression

    Returns:
        Dictionary mapping domain names to features
    """
    features = {}

    # Semantic features (text)
    if experience.has_text:
        # Store original text for retrieval
        features["text"] = experience.text
        text_to_embed = experience.text

        # LLM compression: create summary and embed that instead
        if llm_config and llm_config.enable_compression:
            summary = compress_text_with_llm(experience.text, llm_config)
            if summary:
                features["summary"] = summary
                text_to_embed = summary

        # Store embedding for similarity search
        embedding = extract_semantic_features(text_to_embed)
        if embedding is not None:
            features["semantic"] = embedding.tolist()
    
    # Sensory features (pass through, already extracted)
    if experience.has_sensory:
        for modality, data in experience.sensory.items():
            features[modality] = data
    
    # Emotional features
    emotional = extract_emotional_features(experience.emotional)
    features["emotional"] = emotional
    
    # Motor features (pass through)
    if experience.has_motor:
        features["motor"] = experience.motor
    
    # Temporal features
    temporal = extract_temporal_features(context)
    features["temporal"] = temporal
    
    return features
