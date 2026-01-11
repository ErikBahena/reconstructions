"""
Unit tests for strength and decay calculations.
"""

import pytest
import time
from src.reconstructions.core import Fragment
from src.reconstructions.strength import (
    calculate_decay,
    calculate_rehearsal_bonus,
    calculate_strength,
    get_strength_at_time,
    simulate_decay_curve,
    DecayConfig
)


class TestDecay:
    """Test decay function."""
    
    def test_decay_immediate(self):
        """No time elapsed = minimal decay."""
        strength = calculate_decay(1.0, 0.0)
        
        # Should be close to initial
        assert strength > 0.9
    
    def test_decay_over_time(self):
        """Strength decays over time."""
        strength_1h = calculate_decay(1.0, 3600)  # 1 hour
        strength_24h = calculate_decay(1.0, 86400)  # 24 hours
        strength_1w = calculate_decay(1.0, 604800)  # 1 week
        
        # Later = weaker
        assert strength_1h > strength_24h
        assert strength_24h > strength_1w
    
    def test_decay_respects_initial(self):
        """Higher initial salience = higher decayed strength."""
        high = calculate_decay(1.0, 3600)
        low = calculate_decay(0.5, 3600)
        
        assert high > low
    
    def test_decay_never_zero(self):
        """Strength never reaches zero (min floor)."""
        strength = calculate_decay(1.0, 31536000)  # 1 year
        
        assert strength > 0
        assert strength >= DecayConfig.MIN_STRENGTH
    
    def test_decay_custom_config(self):
        """Custom decay rate works."""
        config = DecayConfig()
        config.DECAY_RATE = 1.0  # Faster decay
        
        fast_decay = calculate_decay(1.0, 3600, config)
        normal_decay = calculate_decay(1.0, 3600)
        
        assert fast_decay < normal_decay


class TestRehearsalBonus:
    """Test rehearsal bonus calculation."""
    
    def test_no_access(self):
        """No accesses = no bonus."""
        bonus = calculate_rehearsal_bonus(0)
        assert bonus == 0.0
    
    def test_single_access(self):
        """Single access gives small bonus."""
        bonus = calculate_rehearsal_bonus(1)
        
        assert bonus > 0
        assert bonus < DecayConfig.MAX_REHEARSAL_BONUS
    
    def test_multiple_accesses(self):
        """More accesses = bigger bonus."""
        bonus_1 = calculate_rehearsal_bonus(1)
        bonus_5 = calculate_rehearsal_bonus(5)
        bonus_10 = calculate_rehearsal_bonus(10)
        
        assert bonus_5 > bonus_1
        assert bonus_10 > bonus_5
    
    def test_diminishing_returns(self):
        """Bonus has diminishing returns."""
        bonus_10 = calculate_rehearsal_bonus(10)
        bonus_100 = calculate_rehearsal_bonus(100)
        
        # 100 is 10x more accesses but not 10x more bonus
        assert bonus_100 < bonus_10 * 5
    
    def test_max_bonus(self):
        """Bonus is capped at maximum."""
        bonus = calculate_rehearsal_bonus(10000)
        
        assert bonus <= DecayConfig.MAX_REHEARSAL_BONUS


class TestCalculateStrength:
    """Test complete strength calculation."""
    
    def test_new_fragment_high_strength(self):
        """New fragment with high salience has high strength."""
        fragment = Fragment(
            content={"test": "data"},
            initial_salience=0.9
        )
        
        strength = calculate_strength(fragment)
        
        assert strength > 0.8
    
    def test_old_fragment_low_strength(self):
        """Old fragment with no access has low strength."""
        fragment = Fragment(
            content={"test": "data"},
            initial_salience=0.9
        )
        # Simulate created 1 week ago
        fragment.created_at = time.time() - 604800
        
        strength = calculate_strength(fragment)
        
        assert strength < 0.5
    
    def test_accessed_fragment_maintains_strength(self):
        """Fragment with accesses maintains more strength."""
        # Fragment without accesses
        frag_no_access = Fragment(
            content={"test": "data"},
            initial_salience=0.9
        )
        frag_no_access.created_at = time.time() - 86400  # 1 day ago
        
        # Fragment with accesses
        frag_accessed = Fragment(
            content={"test": "data"},
            initial_salience=0.9
        )
        frag_accessed.created_at = time.time() - 86400  # 1 day ago
        frag_accessed.access_log = [time.time() - i * 3600 for i in range(5)]
        
        strength_no = calculate_strength(frag_no_access)
        strength_yes = calculate_strength(frag_accessed)
        
        assert strength_yes > strength_no
    
    def test_strength_never_exceeds_one(self):
        """Strength is capped at 1.0."""
        fragment = Fragment(
            content={"test": "data"},
            initial_salience=1.0
        )
        fragment.access_log = [time.time()] * 100
        
        strength = calculate_strength(fragment)
        
        assert strength <= 1.0


class TestStrengthAtTime:
    """Test strength at specific time."""
    
    def test_strength_at_creation(self):
        """Strength at creation time is near initial."""
        fragment = Fragment(
            content={"test": "data"},
            initial_salience=0.8
        )
        
        strength = get_strength_at_time(fragment, fragment.created_at + 1)
        
        assert strength > 0.75
    
    def test_strength_decreases_over_time(self):
        """Strength decreases over simulated time."""
        fragment = Fragment(
            content={"test": "data"},
            initial_salience=0.8
        )
        
        t0 = fragment.created_at
        s1 = get_strength_at_time(fragment, t0 + 3600)   # 1 hour
        s2 = get_strength_at_time(fragment, t0 + 86400)  # 1 day
        s3 = get_strength_at_time(fragment, t0 + 604800) # 1 week
        
        assert s1 > s2 > s3


class TestSimulateDecayCurve:
    """Test decay curve simulation."""
    
    def test_curve_generation(self):
        """Generates correct number of points."""
        curve = simulate_decay_curve(1.0, 24, steps=24)
        
        assert len(curve) == 25  # 0 to 24 inclusive
    
    def test_curve_monotonic_decrease(self):
        """Curve monotonically decreases."""
        curve = simulate_decay_curve(1.0, 168, steps=100)  # 1 week
        
        for i in range(1, len(curve)):
            assert curve[i][1] <= curve[i-1][1]
    
    def test_curve_starts_high(self):
        """Curve starts near initial salience."""
        curve = simulate_decay_curve(0.9, 24, steps=10)
        
        assert curve[0][1] > 0.8


class TestPowerLawShape:
    """Test that decay follows power law shape."""
    
    def test_rapid_initial_then_slow(self):
        """Power law: rapid initial decay, then slower."""
        # Compare decay rates at different times
        s0 = calculate_decay(1.0, 0)
        s1h = calculate_decay(1.0, 3600)
        s2h = calculate_decay(1.0, 7200)
        s24h = calculate_decay(1.0, 86400)
        s48h = calculate_decay(1.0, 172800)
        
        # First hour loss vs second hour loss
        loss_1st_hour = s0 - s1h
        loss_2nd_hour = s1h - s2h
        
        # First day loss vs second day loss
        loss_1st_day = s1h - s24h
        loss_2nd_day = s24h - s48h
        
        # Power law: early losses > later losses
        assert loss_1st_hour >= loss_2nd_hour
        assert loss_1st_day >= loss_2nd_day
