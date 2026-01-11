"""
Constraints System - ensuring coherent reconstructions.

Constraints maintain consistency and prevent contradictions
in reconstructed memories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum

from .core import Fragment, Strand


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preferred but not required


class ConstraintResult(Enum):
    """Result of constraint check."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    constraint_name: str
    severity: float  # 0-1, higher = worse
    description: str
    affected_fragments: List[str] = field(default_factory=list)
    correction_applied: bool = False


class Constraint(ABC):
    """Base class for all constraints."""
    
    name: str = "BaseConstraint"
    constraint_type: ConstraintType = ConstraintType.SOFT
    
    @abstractmethod
    def check(self, fragments: List[Fragment], context: Dict[str, Any]) -> ConstraintResult:
        """
        Check if constraint is satisfied.
        
        Args:
            fragments: Fragments to check
            context: Reconstruction context
            
        Returns:
            Constraint result
        """
        pass
    
    @abstractmethod
    def get_violation(self, fragments: List[Fragment], context: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """
        Get details of any violation.
        
        Args:
            fragments: Fragments to check
            context: Reconstruction context
            
        Returns:
            Violation details or None if satisfied
        """
        pass
    
    def correct(self, fragments: List[Fragment], context: Dict[str, Any]) -> List[Fragment]:
        """
        Attempt to correct violation (optional).
        
        Default: return fragments unchanged.
        
        Args:
            fragments: Fragments with violation
            context: Reconstruction context
            
        Returns:
            Corrected fragments
        """
        return fragments


class NoContradictionConstraint(Constraint):
    """
    Ensure no contradictory content in reconstruction.
    
    Detects when fragments have conflicting information.
    """
    
    name = "NoContradiction"
    constraint_type = ConstraintType.HARD
    
    def check(self, fragments: List[Fragment], context: Dict[str, Any]) -> ConstraintResult:
        if len(fragments) <= 1:
            return ConstraintResult.SATISFIED
        
        # Check for temporal contradictions (same time, different content)
        times_seen = {}
        for frag in fragments:
            t = frag.created_at
            # Bucket by 10 second windows
            bucket = int(t // 10)
            if bucket in times_seen:
                # Multiple fragments in same time bucket
                other = times_seen[bucket]
                if self._content_conflicts(frag.content, other.content):
                    return ConstraintResult.VIOLATED
            times_seen[bucket] = frag
        
        return ConstraintResult.SATISFIED
    
    def _content_conflicts(self, content_a: Dict, content_b: Dict) -> bool:
        """Check if two content dicts conflict."""
        # Simple check: conflicting semantic content
        if "semantic" in content_a and "semantic" in content_b:
            # Could use more sophisticated comparison
            # For now, different strings at same time = conflict
            if isinstance(content_a["semantic"], str) and isinstance(content_b["semantic"], str):
                if content_a["semantic"] != content_b["semantic"]:
                    return True
        return False
    
    def get_violation(self, fragments: List[Fragment], context: Dict[str, Any]) -> Optional[ConstraintViolation]:
        if self.check(fragments, context) == ConstraintResult.SATISFIED:
            return None
        
        return ConstraintViolation(
            constraint_name=self.name,
            severity=0.8,
            description="Contradictory content detected in same time period",
            affected_fragments=[f.id for f in fragments]
        )


class TemporalConsistencyConstraint(Constraint):
    """
    Ensure temporal ordering is consistent.
    
    Events should follow logical temporal sequence.
    """
    
    name = "TemporalConsistency"
    constraint_type = ConstraintType.HARD
    
    def check(self, fragments: List[Fragment], context: Dict[str, Any]) -> ConstraintResult:
        if len(fragments) <= 1:
            return ConstraintResult.SATISFIED
        
        # Check that fragments are in reasonable temporal order
        sorted_frags = sorted(fragments, key=lambda f: f.created_at)
        
        # Check for impossible time jumps (future to past references)
        for i in range(1, len(sorted_frags)):
            prev = sorted_frags[i-1]
            curr = sorted_frags[i]
            
            # Check if current references something after it temporally
            if self._references_future(curr, sorted_frags[i+1:] if i+1 < len(sorted_frags) else []):
                return ConstraintResult.VIOLATED
        
        return ConstraintResult.SATISFIED
    
    def _references_future(self, fragment: Fragment, future_fragments: List[Fragment]) -> bool:
        """Check if fragment references fragments from the future."""
        future_ids = {f.id for f in future_fragments}
        for binding in fragment.bindings:
            if binding in future_ids:
                return True
        return False
    
    def get_violation(self, fragments: List[Fragment], context: Dict[str, Any]) -> Optional[ConstraintViolation]:
        if self.check(fragments, context) == ConstraintResult.SATISFIED:
            return None
        
        return ConstraintViolation(
            constraint_name=self.name,
            severity=0.7,
            description="Temporal inconsistency detected",
            affected_fragments=[f.id for f in fragments]
        )


class CoherenceConstraint(Constraint):
    """
    Ensure minimum coherence level.
    
    Fragments should form a coherent narrative.
    """
    
    name = "Coherence"
    constraint_type = ConstraintType.SOFT
    min_coherence: float = 0.3
    
    def __init__(self, min_coherence: float = 0.3):
        self.min_coherence = min_coherence
    
    def check(self, fragments: List[Fragment], context: Dict[str, Any]) -> ConstraintResult:
        if len(fragments) <= 1:
            return ConstraintResult.SATISFIED
        
        coherence = self._calculate_coherence(fragments)
        
        if coherence >= self.min_coherence:
            return ConstraintResult.SATISFIED
        return ConstraintResult.VIOLATED
    
    def _calculate_coherence(self, fragments: List[Fragment]) -> float:
        """Calculate coherence score."""
        if len(fragments) <= 1:
            return 1.0
        
        # Binding connectivity
        fragment_ids = {f.id for f in fragments}
        connections = 0
        for frag in fragments:
            for binding in frag.bindings:
                if binding in fragment_ids:
                    connections += 1
        
        max_connections = len(fragments) * (len(fragments) - 1)
        if max_connections > 0:
            return connections / max_connections
        return 0.5
    
    def get_violation(self, fragments: List[Fragment], context: Dict[str, Any]) -> Optional[ConstraintViolation]:
        if self.check(fragments, context) == ConstraintResult.SATISFIED:
            return None
        
        coherence = self._calculate_coherence(fragments)
        return ConstraintViolation(
            constraint_name=self.name,
            severity=0.5,
            description=f"Coherence {coherence:.2f} below threshold {self.min_coherence}",
            affected_fragments=[f.id for f in fragments]
        )


@dataclass
class ConstraintSet:
    """Collection of constraints to apply."""
    
    constraints: List[Constraint] = field(default_factory=list)
    
    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)
    
    def check_all(self, fragments: List[Fragment], context: Dict[str, Any]) -> List[ConstraintViolation]:
        """
        Check all constraints.
        
        Returns:
            List of violations (empty if all satisfied)
        """
        violations = []
        for constraint in self.constraints:
            violation = constraint.get_violation(fragments, context)
            if violation:
                violations.append(violation)
        return violations
    
    def has_hard_violation(self, fragments: List[Fragment], context: Dict[str, Any]) -> bool:
        """Check if any hard constraints are violated."""
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.HARD:
                if constraint.check(fragments, context) == ConstraintResult.VIOLATED:
                    return True
        return False


def get_default_constraints() -> ConstraintSet:
    """Get the default constraint set."""
    cs = ConstraintSet()
    cs.add(NoContradictionConstraint())
    cs.add(TemporalConsistencyConstraint())
    cs.add(CoherenceConstraint(min_coherence=0.2))
    return cs


def apply_constraints(
    fragments: List[Fragment],
    constraint_set: ConstraintSet,
    context: Dict[str, Any]
) -> tuple[List[Fragment], List[ConstraintViolation]]:
    """
    Apply constraints to fragments.
    
    Attempts to correct violations where possible.
    
    Args:
        fragments: Fragments to constrain
        constraint_set: Constraints to apply
        context: Reconstruction context
        
    Returns:
        Tuple of (possibly corrected fragments, remaining violations)
    """
    current_fragments = fragments
    all_violations = []
    
    for constraint in constraint_set.constraints:
        violation = constraint.get_violation(current_fragments, context)
        if violation:
            # Try to correct
            corrected = constraint.correct(current_fragments, context)
            
            # Check if correction worked
            new_violation = constraint.get_violation(corrected, context)
            if new_violation:
                # Correction failed
                all_violations.append(new_violation)
            else:
                # Correction succeeded
                violation.correction_applied = True
                all_violations.append(violation)
                current_fragments = corrected
    
    return current_fragments, all_violations
