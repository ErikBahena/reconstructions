"""
Unit tests for identity evolution.
"""

import pytest
from reconstructions.identity import (
    IdentityEvolver, 
    IdentityState, 
    Trait, 
    Belief, 
    Goal
)


class TestIdentityEvolver:
    """Test identity evolution logic."""
    
    def test_update_strength_inertia(self):
        """Test basic inertia calculation."""
        evolver = IdentityEvolver()
        
        # High inertia (0.9) - requires strong push to move
        current = 0.5
        target = 1.0
        inertia = 0.9
        
        new_val = evolver.update_strength(current, target, inertia)
        
        # 0.5 * 0.9 + 1.0 * 0.1 = 0.45 + 0.1 = 0.55
        assert new_val == pytest.approx(0.55)
        
        # Low inertia (0.1) - moves quickly
        inertia = 0.1
        new_val = evolver.update_strength(current, target, inertia)
        
        # 0.5 * 0.1 + 1.0 * 0.9 = 0.05 + 0.9 = 0.95
        assert new_val == pytest.approx(0.95)

    def test_evolve_traits(self):
        """Test trait evolution (high inertia)."""
        state = IdentityState()
        t = Trait(name="Curious", strength=0.5)
        state.add_trait(t)
        
        evolver = IdentityEvolver()
        
        # Evidence that I am VERY curious
        evidence = {
            "traits": {
                "Curious": 1.0
            }
        }
        
        new_state = evolver.evolve(state, evidence)
        new_trait = list(new_state.traits.values())[0]
        
        # Should increase but slowly due to high inertia
        assert new_trait.strength > 0.5
        assert new_trait.strength < 0.6  # Expect small jump (approx 0.525 with 0.95 inertia)
        
        # ID should remain (logical identity, distinct object)
        # Note: implementation creates deepcopy, so IDs in dict keys update
        assert new_trait.name == "Curious"
        
    def test_evolve_goals(self):
        """Test goal evolution (low inertia)."""
        state = IdentityState()
        g = Goal(name="Finish Task", priority=0.5)
        state.add_goal(g)
        
        evolver = IdentityEvolver()
        
        # Task becomes critical
        evidence = {
            "goals": {
                "Finish Task": {"priority": 1.0}
            }
        }
        
        new_state = evolver.evolve(state, evidence)
        new_goal = list(new_state.goals.values())[0]
        
        # Should increase rapidly
        assert new_goal.priority > 0.8  # Expect approx 0.9 with 0.2 inertia
        
    def test_create_new_attributes(self):
        """Test creating new attributes from evidence."""
        state = IdentityState()
        evolver = IdentityEvolver()
        
        evidence = {
            "traits": {"New Trait": 0.8},
            "beliefs": {"New Belief": {"strength": 0.7}}
        }
        
        new_state = evolver.evolve(state, evidence)
        
        assert len(new_state.traits) == 1
        assert len(new_state.beliefs) == 1
        
        t = list(new_state.traits.values())[0]
        assert t.name == "New Trait"
        # Newly created attributes usually start fresh or at target?
        # Implementation sets them to target strength directly
        assert t.strength == 0.8
