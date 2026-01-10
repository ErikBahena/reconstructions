"""
Experience and Context data structures for encoding.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List
import time


@dataclass
class Experience:
    """
    An experience to be encoded into memory.
    
    Represents a single event or observation with potentially
    multi-modal content (text, sensory, emotional, motor).
    """
    
    # Text content (semantic)
    text: Optional[str] = None
    
    # Sensory data (multi-modal)
    sensory: dict[str, Any] = field(default_factory=dict)
    # Example:
    # {
    #     "visual": <array or features>,
    #     "auditory": <array or features>,
    #     "tactile": <array or features>,
    # }
    
    # Emotional state during experience
    emotional: Optional[dict[str, float]] = None
    # Example: {"valence": 0.7, "arousal": 0.3, "dominance": 0.5}
    
    # Motor/action context
    motor: Optional[dict[str, Any]] = None
    # Example: {"action": "walking", "speed": 1.2}
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    source: str = "external"  # "external", "internal", "imagined"
    tags: List[str] = field(default_factory=list)
    
    @property
    def has_text(self) -> bool:
        """Check if experience has text content."""
        return self.text is not None and len(self.text) > 0
    
    @property
    def has_sensory(self) -> bool:
        """Check if experience has sensory data."""
        return len(self.sensory) > 0
    
    @property
    def has_emotional(self) -> bool:
        """Check if experience has emotional data."""
        return self.emotional is not None and len(self.emotional) > 0
    
    @property
    def has_motor(self) -> bool:
        """Check if experience has motor data."""
        return self.motor is not None and len(self.motor) > 0


@dataclass
class Context:
    """
    Contextual state during encoding or reconstruction.
    
    Tracks the current situation, active goals, and sequential
    position for proper temporal binding.
    """
    
    # Unique context identifier
    id: str = field(default_factory=lambda: f"ctx_{int(time.time() * 1000)}")
    
    # Sequential position (for temporal ordering)
    sequence_counter: int = 0
    
    # Active goals
    active_goals: List[str] = field(default_factory=list)
    
    # Current state
    state: dict[str, Any] = field(default_factory=dict)
    # Example:
    # {
    #     "mode": "exploration",
    #     "focus": "learning",
    #     "environment": "home"
    # }
    
    # Processing depth (for salience calculation)
    processing_depth: float = 0.5  # 0.0 = shallow, 1.0 = deep
    
    # Variance mode (for reconstruction)
    variance_mode: float = 0.3  # Controls reconstruction stability
    
    # Recent fragments (for binding)
    recent_fragments: List[str] = field(default_factory=list)  # Fragment IDs
    
    # Timestamp
    created_at: float = field(default_factory=time.time)
    
    def increment_sequence(self) -> None:
        """Increment the sequence counter."""
        self.sequence_counter += 1
    
    def add_recent_fragment(self, fragment_id: str, max_recent: int = 10) -> None:
        """
        Add a fragment to recent history.
        
        Args:
            fragment_id: Fragment ID to add
            max_recent: Maximum number of recent fragments to keep
        """
        self.recent_fragments.append(fragment_id)
        if len(self.recent_fragments) > max_recent:
            self.recent_fragments.pop(0)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "id": self.id,
            "sequence_counter": self.sequence_counter,
            "active_goals": self.active_goals,
            "state": self.state,
            "processing_depth": self.processing_depth,
            "variance_mode": self.variance_mode,
            "recent_fragments": self.recent_fragments,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Context":
        """Deserialize context from dictionary."""
        return cls(
            id=data["id"],
            sequence_counter=data["sequence_counter"],
            active_goals=data["active_goals"],
            state=data["state"],
            processing_depth=data["processing_depth"],
            variance_mode=data["variance_mode"],
            recent_fragments=data["recent_fragments"],
            created_at=data["created_at"]
        )
