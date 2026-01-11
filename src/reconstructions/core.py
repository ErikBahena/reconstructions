"""
Core data structures for the Reconstructions memory system.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import time


@dataclass
class Fragment:
    """
    The atomic unit of memory.
    
    A fragment represents a single memory unit with cross-domain bindings,
    salience weighting, and access tracking.
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    
    # Content (multi-modal, domain → data mapping)
    content: dict[str, Any] = field(default_factory=dict)
    # Example structure:
    # {
    #     "semantic": "the sky is blue",
    #     "visual": [0.2, 0.4, ...],      # Feature vector
    #     "emotional": {"valence": 0.3, "arousal": 0.1, "dominance": 0.5},
    #     "temporal": {"sequence_position": 42, "absolute": 1704834567.123}
    # }
    
    # Binding (cross-domain associations)
    bindings: list[str] = field(default_factory=list)  # Fragment IDs
    
    # Salience and strength
    initial_salience: float = 0.5  # Encoding strength (0-1)
    access_log: list[float] = field(default_factory=list)  # Timestamps of each access
    
    # Metadata
    source: str = "experience"  # "experience", "inference", "reflection"
    tags: list[str] = field(default_factory=list)  # Arbitrary labels
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize fragment to dictionary for storage.
        
        Returns:
            Dictionary representation of the fragment
        """
        return {
            "id": self.id,
            "created_at": self.created_at,
            "content": self.content,
            "bindings": self.bindings,
            "initial_salience": self.initial_salience,
            "access_log": self.access_log,
            "source": self.source,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fragment":
        """
        Deserialize fragment from dictionary.
        
        Args:
            data: Dictionary representation of a fragment
            
        Returns:
            Fragment instance
        """
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            content=data["content"],
            bindings=data["bindings"],
            initial_salience=data["initial_salience"],
            access_log=data["access_log"],
            source=data["source"],
            tags=data["tags"]
        )


@dataclass
class Strand:
    """
    A reconstruction output—assembled fragments.
    
    Represents the result of a reconstruction operation, containing
    references to the fragments used and metadata about the reconstruction.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fragments: list[str] = field(default_factory=list)  # Fragment IDs
    assembly_context: dict[str, Any] = field(default_factory=dict)  # Context at reconstruction time
    coherence_score: float = 0.0  # How internally consistent (0-1)
    variance: float = 0.0  # Reconstruction stability measure (target variance)
    certainty: float = 0.0  # Calculated subjective certainty (0-1)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize strand to dictionary."""
        return {
            "id": self.id,
            "fragments": self.fragments,
            "assembly_context": self.assembly_context,
            "coherence_score": self.coherence_score,
            "coherence_score": self.coherence_score,
            "variance": self.variance,
            "certainty": self.certainty,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strand":
        """Deserialize strand from dictionary."""
        return cls(
            id=data["id"],
            fragments=data["fragments"],
            assembly_context=data["assembly_context"],
            coherence_score=data["coherence_score"],
            variance=data["variance"],
            certainty=data.get("certainty", 0.0),
            created_at=data["created_at"]
        )


@dataclass
class Query:
    """
    A query for reconstruction.
    
    Specifies what to reconstruct, with optional filters for domain,
    time range, and minimum salience.
    """
    
    semantic: Optional[str] = None  # Text-based query
    domains: Optional[list[str]] = None  # Filter by domain
    time_range: Optional[tuple[float, float]] = None  # (after, before) timestamps
    min_salience: float = 0.0  # Minimum salience threshold
    
    def to_hash(self) -> str:
        """
        Generate a hash for variance tracking.
        
        Returns:
            Hash string representing this query
        """
        import hashlib
        
        # Create deterministic string representation
        parts = [
            self.semantic or "",
            ",".join(sorted(self.domains)) if self.domains else "",
            f"{self.time_range[0]},{self.time_range[1]}" if self.time_range else "",
            str(self.min_salience)
        ]
        query_str = "|".join(parts)
        
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]
