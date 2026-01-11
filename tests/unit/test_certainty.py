"""
Unit tests for variance and certainty system.
"""

import pytest
from src.reconstructions.core import Strand
from src.reconstructions.certainty import (
    VarianceController,
    calculate_strand_distance
)


class TestStrandDistance:
    """Test strand distance calculation."""
    
    def test_identical_strands(self):
        """Identical strands have 0 distance."""
        s1 = Strand(fragments=["a", "b", "c"])
        s2 = Strand(fragments=["a", "b", "c"])
        
        dist = calculate_strand_distance(s1, s2)
        assert dist == 0.0
    
    def test_disjoint_strands(self):
        """Disjoint strands have 1.0 distance."""
        s1 = Strand(fragments=["a", "b"])
        s2 = Strand(fragments=["c", "d"])
        
        dist = calculate_strand_distance(s1, s2)
        assert dist == 1.0
    
    def test_partial_overlap(self):
        """Partially overlapping strands have intermediate distance."""
        # 2 overlapping, 2 unique = 2 intersection, 4 union, 0.5 similarity, 0.5 distance
        s1 = Strand(fragments=["a", "b", "c"])
        s2 = Strand(fragments=["b", "c", "d"])
        
        dist = calculate_strand_distance(s1, s2)
        assert dist == 0.5


class TestVarianceController:
    """Test variance controller."""
    
    def test_initial_certainty_low(self):
        """New query has low certainty (high variance)."""
        vc = VarianceController()
        certainty = vc.get_certainty("hash1")
        
        # With no history, variance is 1.0 (default assumption of uncertainty)
        # So certainty should be 0.0
        assert certainty == 0.0
    
    def test_single_record_low_certainty(self):
        """Single record still has low certainty (insufficient history)."""
        vc = VarianceController()
        s1 = Strand(fragments=["a"])
        vc.record_reconstruction("hash1", s1)
        
        certainty = vc.get_certainty("hash1")
        assert certainty == 0.0
    
    def test_consistent_history_high_certainty(self):
        """Consistent history yields high certainty."""
        vc = VarianceController()
        s1 = Strand(fragments=["a", "b"])
        s2 = Strand(fragments=["a", "b"])
        
        vc.record_reconstruction("hash1", s1)
        vc.record_reconstruction("hash1", s2)
        
        certainty = vc.get_certainty("hash1")
        assert certainty == 1.0  # Perfect consistency
    
    def test_variable_history_low_certainty(self):
        """Variable history yields low certainty."""
        vc = VarianceController()
        s1 = Strand(fragments=["a"])
        s2 = Strand(fragments=["b"])  # Totally different
        
        vc.record_reconstruction("hash1", s1)
        vc.record_reconstruction("hash1", s2)
        
        certainty = vc.get_certainty("hash1")
        # Distance = 1.0, Variance = 1.0, Certainty = 0.0
        assert certainty == 0.0
        
    def test_history_limit(self):
        """Respects history limit."""
        vc = VarianceController(history_size=2)
        s1 = Strand(fragments=["a"])
        s2 = Strand(fragments=["b"])
        s3 = Strand(fragments=["c"])
        
        vc.record_reconstruction("hash1", s1)
        vc.record_reconstruction("hash1", s2) 
        # History now [s1, s2], calc variance on these (dist=1.0) -> cert=0.0
        
        vc.record_reconstruction("hash1", s3)
        # History now [s2, s3] (s1 dropped), calc variance on these (dist=1.0) -> cert=0.0
        
        # Now make it consistent with latest
        s4 = Strand(fragments=["c"])
        vc.record_reconstruction("hash1", s4)
        # History now [s3, s4] (identical), calc variance -> dist=0 -> cert=1.0
        
        certainty = vc.get_certainty("hash1")
        assert certainty == 1.0
    
    def test_calculate_variance_with_new(self):
        """Calculate variance including a hypothetical new strand."""
        vc = VarianceController()
        s1 = Strand(fragments=["a"])
        vc.record_reconstruction("hash1", s1)
        
        s2 = Strand(fragments=["a"]) # Consistent
        
        # Check variance if we WERE to add s2
        var = vc.calculate_variance("hash1", new_strand=s2)
        # [s1, s2] -> dist 0 -> var 0
        assert var == 0.0
        
        # Verify s2 wasn't actually added
        assert len(vc._history["hash1"]) == 1
