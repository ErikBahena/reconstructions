"""
Unit tests for identity model.
"""

import pytest
import time
from src.reconstructions.identity import (
    Trait, 
    Belief, 
    Goal, 
    IdentityState, 
    IdentityStore
)


class TestIdentityAttributes:
    """Test individual attribute classes."""
    
    def test_trait_creation(self):
        """Create a trait."""
        t = Trait(name="Curious", description="Likes to learn")
        
        assert t.name == "Curious"
        assert t.id is not None
        assert t.strength == 0.5  # Default
    
    def test_belief_with_evidence(self):
        """Create a belief with evidence."""
        b = Belief(
            name="Self-efficacy",
            evidence_fragments=["frag1", "frag2"]
        )
        
        assert len(b.evidence_fragments) == 2
        d = b.to_dict()
        assert "evidence_fragments" in d
    
    def test_goal_properties(self):
        """Create a goal with properties."""
        g = Goal(
            name="Finish Project",
            priority=0.9,
            status="active"
        )
        
        assert g.priority == 0.9
        assert g.status == "active"
        d = g.to_dict()
        assert d["priority"] == 0.9


class TestIdentityState:
    """Test identity state container."""
    
    def test_state_management(self):
        """Add attributes to state."""
        state = IdentityState()
        
        t = Trait(name="T1")
        b = Belief(name="B1")
        g = Goal(name="G1")
        
        state.add_trait(t)
        state.add_belief(b)
        state.add_goal(g)
        
        assert len(state.traits) == 1
        assert len(state.beliefs) == 1
        assert len(state.goals) == 1
        assert state.traits[t.id].name == "T1"
    
    def test_serialization_cycle(self):
        """Test to_dict and from_dict cycle."""
        state = IdentityState()
        state.add_trait(Trait(name="T1"))
        state.add_goal(Goal(name="G1", priority=0.8))
        
        data = state.to_dict()
        restored = IdentityState.from_dict(data)
        
        assert len(restored.traits) == 1
        assert len(restored.goals) == 1
        
        goal = list(restored.goals.values())[0]
        assert goal.name == "G1"
        assert goal.priority == 0.8
        
        # IDs preserved
        assert goal.id == list(state.goals.values())[0].id


class TestIdentityStore:
    """Test identity storage."""
    
    def test_store_updates(self):
        """Update state in store."""
        store = IdentityStore()
        initial = store.get_current_state()
        
        # Create new state
        new_state = IdentityState()
        new_state.add_trait(Trait(name="New"))
        
        store.update_state(new_state)
        
        current = store.get_current_state()
        assert len(current.traits) == 1
        assert len(store._history) == 1
        assert store._history[0] == initial
