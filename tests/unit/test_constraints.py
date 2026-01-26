"""
Unit tests for constraints system.
"""

import pytest
import time
from reconstructions.core import Fragment
from reconstructions.constraints import (
    Constraint,
    ConstraintType,
    ConstraintResult,
    ConstraintViolation,
    NoContradictionConstraint,
    TemporalConsistencyConstraint,
    CoherenceConstraint,
    ConstraintSet,
    get_default_constraints,
    apply_constraints
)


class TestConstraintViolation:
    """Test ConstraintViolation dataclass."""
    
    def test_create_violation(self):
        """Create basic violation."""
        v = ConstraintViolation(
            constraint_name="Test",
            severity=0.5,
            description="Test violation"
        )
        
        assert v.constraint_name == "Test"
        assert v.severity == 0.5
        assert not v.correction_applied


class TestNoContradictionConstraint:
    """Test contradiction detection."""
    
    def test_no_contradictions(self):
        """No contradictions when fragments are different times."""
        now = time.time()
        
        f1 = Fragment(content={"semantic": "sky is blue"})
        f1.created_at = now - 100
        
        f2 = Fragment(content={"semantic": "grass is green"})
        f2.created_at = now
        
        constraint = NoContradictionConstraint()
        result = constraint.check([f1, f2], {})
        
        assert result == ConstraintResult.SATISFIED
    
    def test_contradiction_same_time(self):
        """Contradiction when conflicting content at same time."""
        now = time.time()
        
        f1 = Fragment(content={"semantic": "sky is blue"})
        f1.created_at = now
        
        f2 = Fragment(content={"semantic": "sky is red"})
        f2.created_at = now + 1  # Same 10-second bucket
        
        constraint = NoContradictionConstraint()
        result = constraint.check([f1, f2], {})
        
        assert result == ConstraintResult.VIOLATED
    
    def test_single_fragment(self):
        """Single fragment always satisfies."""
        f = Fragment(content={"semantic": "test"})
        
        constraint = NoContradictionConstraint()
        result = constraint.check([f], {})
        
        assert result == ConstraintResult.SATISFIED
    
    def test_get_violation(self):
        """Get violation details."""
        now = time.time()
        
        f1 = Fragment(content={"semantic": "A"})
        f1.created_at = now
        
        f2 = Fragment(content={"semantic": "B"})
        f2.created_at = now + 1
        
        constraint = NoContradictionConstraint()
        violation = constraint.get_violation([f1, f2], {})
        
        assert violation is not None
        assert violation.constraint_name == "NoContradiction"


class TestTemporalConsistencyConstraint:
    """Test temporal consistency."""
    
    def test_consistent_order(self):
        """Consistent temporal order passes."""
        now = time.time()
        
        f1 = Fragment(content={"a": 1})
        f1.created_at = now - 100
        
        f2 = Fragment(content={"a": 2})
        f2.created_at = now - 50
        f2.bindings = [f1.id]
        
        f3 = Fragment(content={"a": 3})
        f3.created_at = now
        f3.bindings = [f2.id]
        
        constraint = TemporalConsistencyConstraint()
        result = constraint.check([f1, f2, f3], {})
        
        assert result == ConstraintResult.SATISFIED
    
    def test_single_fragment(self):
        """Single fragment always consistent."""
        f = Fragment(content={"a": 1})
        
        constraint = TemporalConsistencyConstraint()
        result = constraint.check([f], {})
        
        assert result == ConstraintResult.SATISFIED


class TestCoherenceConstraint:
    """Test coherence constraint."""
    
    def test_high_coherence(self):
        """High coherence passes."""
        f1 = Fragment(content={"a": 1})
        f2 = Fragment(content={"a": 2})
        
        # Bidirectional bindings
        f1.bindings = [f2.id]
        f2.bindings = [f1.id]
        
        constraint = CoherenceConstraint(min_coherence=0.3)
        result = constraint.check([f1, f2], {})
        
        assert result == ConstraintResult.SATISFIED
    
    def test_low_coherence(self):
        """Low coherence fails."""
        f1 = Fragment(content={"a": 1})
        f2 = Fragment(content={"a": 2})
        f3 = Fragment(content={"a": 3})
        # No bindings between them
        
        constraint = CoherenceConstraint(min_coherence=0.5)
        result = constraint.check([f1, f2, f3], {})
        
        assert result == ConstraintResult.VIOLATED
    
    def test_custom_threshold(self):
        """Custom coherence threshold works."""
        constraint = CoherenceConstraint(min_coherence=0.8)
        
        f1 = Fragment(content={"a": 1})
        f2 = Fragment(content={"a": 2})
        f1.bindings = [f2.id]
        f2.bindings = [f1.id]
        
        result = constraint.check([f1, f2], {})
        
        assert result == ConstraintResult.SATISFIED


class TestConstraintSet:
    """Test constraint set."""
    
    def test_add_constraint(self):
        """Add constraint to set."""
        cs = ConstraintSet()
        cs.add(NoContradictionConstraint())
        
        assert len(cs.constraints) == 1
    
    def test_check_all_no_violations(self):
        """Check all with no violations."""
        cs = ConstraintSet()
        cs.add(NoContradictionConstraint())
        
        f = Fragment(content={"a": 1})
        violations = cs.check_all([f], {})
        
        assert len(violations) == 0
    
    def test_check_all_with_violations(self):
        """Check all with violations."""
        now = time.time()
        
        cs = ConstraintSet()
        cs.add(NoContradictionConstraint())
        
        f1 = Fragment(content={"semantic": "X"})
        f1.created_at = now
        f2 = Fragment(content={"semantic": "Y"})
        f2.created_at = now + 1
        
        violations = cs.check_all([f1, f2], {})
        
        assert len(violations) == 1
    
    def test_has_hard_violation(self):
        """Detect hard violations."""
        now = time.time()
        
        cs = ConstraintSet()
        cs.add(NoContradictionConstraint())  # HARD
        
        f1 = Fragment(content={"semantic": "A"})
        f1.created_at = now
        f2 = Fragment(content={"semantic": "B"})
        f2.created_at = now + 1
        
        assert cs.has_hard_violation([f1, f2], {})


class TestDefaultConstraints:
    """Test default constraint set."""
    
    def test_get_defaults(self):
        """Get default constraints."""
        cs = get_default_constraints()
        
        assert len(cs.constraints) == 3
        
        names = [c.name for c in cs.constraints]
        assert "NoContradiction" in names
        assert "TemporalConsistency" in names
        assert "Coherence" in names


class TestApplyConstraints:
    """Test constraint application."""
    
    def test_apply_no_violations(self):
        """Apply constraints with no violations."""
        cs = get_default_constraints()
        
        f1 = Fragment(content={"a": 1})
        f2 = Fragment(content={"a": 2})
        f1.created_at = time.time() - 100
        f2.created_at = time.time()
        f1.bindings = [f2.id]
        f2.bindings = [f1.id]
        
        corrected, violations = apply_constraints([f1, f2], cs, {})
        
        # Should have no hard violations
        hard_violations = [v for v in violations if v.severity > 0.7]
        assert len(hard_violations) == 0
    
    def test_apply_returns_fragments(self):
        """Apply returns fragments."""
        cs = ConstraintSet()
        cs.add(NoContradictionConstraint())
        
        f = Fragment(content={"a": 1})
        corrected, violations = apply_constraints([f], cs, {})
        
        assert len(corrected) == 1
        assert corrected[0].id == f.id
