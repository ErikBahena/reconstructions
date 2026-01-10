"""
Feature extraction functions for encoding experiences.
"""

import numpy as np
from typing import Optional, List
from .encoding import Experience, Context


# Global embedding model (lazy loaded)
_embedding_model = None


def get_embedding_model():
    """
    Get or load the embedding model.
    
    Uses sentence-transformers with a small, efficient model.
    Lazy loaded to avoid startup overhead.
    """
    global _embedding_model
    
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using all-MiniLM-L6-v2: small (80MB), fast, good quality
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # Fallback: return None if sentence-transformers not available
            _embedding_model = None
    
    return _embedding_model


def extract_semantic_features(text: str) -> Optional[np.ndarray]:
    """
    Extract semantic features from text using embeddings.
    
    Args:
        text: Input text to encode
        
    Returns:
        Embedding vector as numpy array, or None if model unavailable
    """
    if not text or len(text.strip()) == 0:
        return None
    
    model = get_embedding_model()
    if model is None:
        # Fallback: create a simple hash-based pseudo-embedding
        # This is NOT semantically meaningful, just for testing
        return _fallback_text_embedding(text)
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype(np.float32)


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


def extract_all_features(experience: Experience, context: Context) -> dict:
    """
    Extract all features from an experience.
    
    This is the main entry point for feature extraction.
    
    Args:
        experience: Experience to extract features from
        context: Current context
        
    Returns:
        Dictionary mapping domain names to features
    """
    features = {}
    
    # Semantic features (text)
    if experience.has_text:
        embedding = extract_semantic_features(experience.text)
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
