"""
Identity Model - the persistent self.

This module defines the core attributes of identity (Traits, Beliefs, Goals)
and the system for tracking identity state evolution.
"""

from dataclasses import dataclass, field
import time
import uuid
from typing import List, Optional, Any, Dict, Set


@dataclass
class IdentityAttribute:
    """Base class for identity attributes."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    strength: float = 0.5  # 0-1 confidence/importance
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "strength": self.strength,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }


@dataclass
class Trait(IdentityAttribute):
    """
    A stable personality characteristic.
    
    Examples: "Curious", "Cautious", "Analytical"
    """
    pass


@dataclass
class Belief(IdentityAttribute):
    """
    A held truth about the world or self.
    
    Examples: "The world is generally safe", "I am good at math"
    """
    evidence_fragments: List[str] = field(default_factory=list)  # IDs of supporting fragments falseable?
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["evidence_fragments"] = self.evidence_fragments
        return d


@dataclass
class Goal(IdentityAttribute):
    """
    Active objective driving behavior.
    
    Examples: "Learn Python", "Find specific document"
    """
    status: str = "active"  # active, completed, abandoned
    priority: float = 0.5   # 0-1
    deadline: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["status"] = self.status
        d["priority"] = self.priority
        d["deadline"] = self.deadline
        return d


@dataclass
class IdentityState:
    """
    Snapshot of identity at a point in time.
    
    Contains the active set of traits, beliefs, and goals.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    traits: Dict[str, Trait] = field(default_factory=dict)  # id -> Trait
    beliefs: Dict[str, Belief] = field(default_factory=dict)  # id -> Belief
    goals: Dict[str, Goal] = field(default_factory=dict)  # id -> Goal
    
    # Emotional baseline for this state
    baseline_mood: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.5,
        "arousal": 0.5
    })
    
    def add_trait(self, trait: Trait) -> None:
        """Add or update a trait."""
        self.traits[trait.id] = trait
        
    def add_belief(self, belief: Belief) -> None:
        """Add or update a belief."""
        self.beliefs[belief.id] = belief
        
    def add_goal(self, goal: Goal) -> None:
        """Add or update a goal."""
        self.goals[goal.id] = goal
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete state."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
            "beliefs": {k: v.to_dict() for k, v in self.beliefs.items()},
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "baseline_mood": self.baseline_mood
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityState":
        """Deserialize from dictionary."""
        state = cls(
            id=data["id"],
            timestamp=data["timestamp"],
            baseline_mood=data.get("baseline_mood", {"valence": 0.5, "arousal": 0.5})
        )
        
        # Reconstruct objects
        for t_data in data["traits"].values():
            trait = Trait(
                id=t_data["id"],
                name=t_data["name"],
                description=t_data["description"],
                strength=t_data["strength"],
                created_at=t_data["created_at"],
                last_updated=t_data["last_updated"]
            )
            state.traits[trait.id] = trait
            
        for b_data in data["beliefs"].values():
            belief = Belief(
                id=b_data["id"],
                name=b_data["name"],
                description=b_data["description"],
                strength=b_data["strength"],
                created_at=b_data["created_at"],
                last_updated=b_data["last_updated"],
                evidence_fragments=b_data.get("evidence_fragments", [])
            )
            state.beliefs[belief.id] = belief
            
        for g_data in data["goals"].values():
            goal = Goal(
                id=g_data["id"],
                name=g_data["name"],
                description=g_data["description"],
                strength=g_data["strength"],
                created_at=g_data["created_at"],
                last_updated=g_data["last_updated"],
                status=g_data.get("status", "active"),
                priority=g_data.get("priority", 0.5),
                deadline=g_data.get("deadline")
            )
            state.goals[goal.id] = goal
            
        return state


class IdentityStore:
    """
    Storage for identity states.
    
    Manages persistence of the self-model.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # For Phase 6, simpler in-memory implementation is fine for testing
        # Will expand to full SQLite storage if needed, or share FragmentStore DB
        self._current_state: IdentityState = IdentityState()
        self._history: List[IdentityState] = []
        
    def get_current_state(self) -> IdentityState:
        """Get the current active identity state."""
        return self._current_state
    
    def update_state(self, new_state: IdentityState) -> None:
        """Update to a new state, archiving the old one."""
        self._history.append(self._current_state)
        self._current_state = new_state
        
    def save(self) -> None:
        """Persist to disk (placeholder)."""
        pass
    
    def load(self) -> None:
        """Load from disk (placeholder)."""
        pass


class IdentityEvolver:
    """
    Handles the evolution of identity over time.
    
    Applies inertia rules:
    - Traits: High inertia, slow change
    - Beliefs: Medium inertia, require evidence
    - Goals: Low inertia, task-dependent
    """
    
    TRAIT_INERTIA = 0.95   # 0-1, higher = harder to change
    BELIEF_INERTIA = 0.8
    GOAL_INERTIA = 0.2
    
    @staticmethod
    def update_strength(current: float, target: float, inertia: float) -> float:
        """
        Calculate new strength based on target and inertia.
        
        New = Current * Inertia + Target * (1 - Inertia)
        """
        return current * inertia + target * (1.0 - inertia)
    
    def evolve(self, state: IdentityState, evidence: Dict[str, Any]) -> IdentityState:
        """
        Evolve identity state based on new evidence.
        
        Args:
            state: Current identity state
            evidence: Dictionary of updates (e.g., {"traits": {"Curious": 0.8}})
            
        Returns:
            New identity state
        """
        import copy
        new_state = copy.deepcopy(state)
        new_state.timestamp = time.time()
        new_state.id = str(uuid.uuid4())
        
        # Update Traits
        if "traits" in evidence:
            for name, target_strength in evidence["traits"].items():
                # Find by name
                trait_id = None
                for tid, t in new_state.traits.items():
                    if t.name == name:
                        trait_id = tid
                        break
                
                if trait_id:
                    # Update existing
                    current = new_state.traits[trait_id].strength
                    new_strength = self.update_strength(current, target_strength, self.TRAIT_INERTIA)
                    new_state.traits[trait_id].strength = new_strength
                    new_state.traits[trait_id].last_updated = new_state.timestamp
                else:
                    # Create new (if strong enough evidence)
                    if target_strength > 0.3:
                        new_trait = Trait(name=name, strength=target_strength)
                        new_state.add_trait(new_trait)
        
        # Update Beliefs
        if "beliefs" in evidence:
            for name, update_data in evidence["beliefs"].items():
                target_strength = update_data.get("strength", 0.5)
                supporting_fragments = update_data.get("fragments", [])
                
                belief_id = None
                for bid, b in new_state.beliefs.items():
                    if b.name == name:
                        belief_id = bid
                        break
                
                if belief_id:
                    current = new_state.beliefs[belief_id].strength
                    new_strength = self.update_strength(current, target_strength, self.BELIEF_INERTIA)
                    new_state.beliefs[belief_id].strength = new_strength
                    new_state.beliefs[belief_id].evidence_fragments.extend(supporting_fragments)
                    new_state.beliefs[belief_id].last_updated = new_state.timestamp
                else:
                    if target_strength > 0.3:
                        new_belief = Belief(
                            name=name,
                            strength=target_strength,
                            evidence_fragments=supporting_fragments
                        )
                        new_state.add_belief(new_belief)
        
        # Update Goals
        if "goals" in evidence:
            for name, update_data in evidence["goals"].items():
                status = update_data.get("status")
                priority = update_data.get("priority")
                
                goal_id = None
                for gid, g in new_state.goals.items():
                    if g.name == name:
                        goal_id = gid
                        break
                
                if goal_id:
                    if status:
                        new_state.goals[goal_id].status = status
                    if priority is not None:
                        current = new_state.goals[goal_id].priority
                        new_priority = self.update_strength(current, priority, self.GOAL_INERTIA)
                        new_state.goals[goal_id].priority = new_priority
                    new_state.goals[goal_id].last_updated = new_state.timestamp
                else:
                    new_goal = Goal(
                        name=name,
                        status=status or "active",
                        priority=priority if priority is not None else 0.5
                    )
                    new_state.add_goal(new_goal)
                    
        return new_state
