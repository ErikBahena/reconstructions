"""
Strength and decay calculations for memory fragments.

This module implements the power law of forgetting and rehearsal effects.
"""

import math
import time
from typing import Optional
from .core import Fragment


class DecayConfig:
    """Configuration for decay calculations."""
    
    # Power law decay exponent (Wixted & Ebbesen, 1991)
    # Higher values = faster forgetting
    DECAY_RATE: float = 0.5
    
    # Minimum strength floor (never fully forget)
    MIN_STRENGTH: float = 0.01
    
    # Time scaling (seconds per "time unit")
    # Default: 1 hour = 3600 seconds
    TIME_SCALE: float = 3600.0
    
    # Rehearsal bonus per access
    REHEARSAL_BONUS: float = 0.1
    
    # Maximum rehearsal contribution
    MAX_REHEARSAL_BONUS: float = 0.5


def calculate_decay(
    initial_salience: float,
    time_elapsed: float,
    config: Optional[DecayConfig] = None
) -> float:
    """
    Calculate memory decay using power law of forgetting.
    
    Formula: strength = initial_salience * (t + 1)^(-decay_rate)
    
    This models the Ebbinghaus forgetting curve with power law decay.
    
    Args:
        initial_salience: Initial encoding strength (0-1)
        time_elapsed: Time since encoding in seconds
        config: Optional decay configuration
        
    Returns:
        Decayed strength (0-1)
    """
    if config is None:
        config = DecayConfig()
    
    # Convert time to time units
    t = time_elapsed / config.TIME_SCALE
    
    # Avoid division by zero: min time is 1 time unit
    t = max(t, 0.001)
    
    # Power law decay
    decay_factor = math.pow(t + 1, -config.DECAY_RATE)
    
    # Apply to initial salience
    strength = initial_salience * decay_factor
    
    return max(strength, config.MIN_STRENGTH)


def calculate_rehearsal_bonus(
    access_count: int,
    config: Optional[DecayConfig] = None
) -> float:
    """
    Calculate rehearsal bonus from access history.
    
    Each access strengthens the memory, but with diminishing returns.
    
    Args:
        access_count: Number of times fragment was accessed
        config: Optional decay configuration
        
    Returns:
        Rehearsal bonus (0 to MAX_REHEARSAL_BONUS)
    """
    if config is None:
        config = DecayConfig()
    
    if access_count <= 0:
        return 0.0
    
    # Diminishing returns: log-based bonus
    bonus = config.REHEARSAL_BONUS * math.log1p(access_count)
    
    return min(bonus, config.MAX_REHEARSAL_BONUS)


def calculate_strength(
    fragment: Fragment,
    now: Optional[float] = None,
    config: Optional[DecayConfig] = None
) -> float:
    """
    Calculate current memory strength.
    
    Combines:
    - Initial salience
    - Time decay (power law)
    - Rehearsal bonus (from access count)
    
    Args:
        fragment: Fragment to calculate strength for
        now: Current time (defaults to time.time())
        config: Optional decay configuration
        
    Returns:
        Current strength (0-1)
    """
    if config is None:
        config = DecayConfig()
    
    if now is None:
        now = time.time()
    
    # Time since encoding
    time_elapsed = max(0, now - fragment.created_at)
    
    # Calculate decay
    decayed_strength = calculate_decay(
        fragment.initial_salience,
        time_elapsed,
        config
    )
    
    # Calculate rehearsal bonus
    access_count = len(fragment.access_log)
    rehearsal = calculate_rehearsal_bonus(access_count, config)
    
    # Combine: base decay + rehearsal bonus
    # Cap at 1.0
    strength = min(decayed_strength + rehearsal, 1.0)
    
    return strength


def get_strength_at_time(
    fragment: Fragment,
    target_time: float,
    config: Optional[DecayConfig] = None
) -> float:
    """
    Calculate strength at a specific point in time.
    
    Useful for simulating/testing decay curves.
    
    Args:
        fragment: Fragment to calculate strength for
        target_time: Time to calculate strength at
        config: Optional decay configuration
        
    Returns:
        Strength at target time
    """
    return calculate_strength(fragment, now=target_time, config=config)


def simulate_decay_curve(
    initial_salience: float,
    duration_hours: float,
    steps: int = 100,
    config: Optional[DecayConfig] = None
) -> list[tuple[float, float]]:
    """
    Simulate a decay curve for visualization/testing.
    
    Args:
        initial_salience: Initial encoding strength
        duration_hours: Duration to simulate
        steps: Number of time points
        config: Optional decay configuration
        
    Returns:
        List of (hours, strength) tuples
    """
    if config is None:
        config = DecayConfig()
    
    curve = []
    duration_seconds = duration_hours * 3600
    
    for i in range(steps + 1):
        t = (i / steps) * duration_seconds
        hours = t / 3600
        strength = calculate_decay(initial_salience, t, config)
        curve.append((hours, strength))
    
    return curve
