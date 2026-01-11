"""
Variance and Certainty System.

Tracks the stability of reconstructions over time to determine certainty.
Certainty is high when reconstructions for the same query are consistent (low variance).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import statistics

from .core import Strand


def calculate_strand_distance(strand_a: Strand, strand_b: Strand) -> float:
    """
    Calculate distance between two strands (0.0 to 1.0).
    
    Uses Jaccard distance of fragment IDs.
    0.0 = Identical fragment sets
    1.0 = Disjoint fragment sets
    """
    set_a = set(strand_a.fragments)
    set_b = set(strand_b.fragments)
    
    if not set_a and not set_b:
        return 0.0
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


class VarianceController:
    """
    Tracks reconstruction variance and calculates certainty.
    
    Maintains a history of recent reconstructions for each query hash.
    """
    
    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        # Map query_hash -> List[Strand]
        self._history: Dict[str, List[Strand]] = {}
    
    def record_reconstruction(self, query_hash: str, strand: Strand) -> None:
        """
        Record a reconstruction for a query.
        
        Args:
            query_hash: Hash of the query
            strand: The resulting strand
        """
        if query_hash not in self._history:
            self._history[query_hash] = []
        
        history = self._history[query_hash]
        history.append(strand)
        
        # Keep only recent history
        if len(history) > self.history_size:
            history.pop(0)

    def calculate_variance(self, query_hash: str, new_strand: Optional[Strand] = None) -> float:
        """
        Calculate variance for a query.
        
        Args:
            query_hash: Hash of the query
            new_strand: Optional new strand to include in calculation (temporary)
            
        Returns:
            Variance score (0.0 to 1.0)
        """
        history = self._history.get(query_hash, [])
        if new_strand:
            history = history + [new_strand]
            
        if len(history) < 2:
            return 1.0  # High variance (uncertainty) if insufficient history
        
        # Calculate pairwise distances between all strands in history
        distances = []
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                dist = calculate_strand_distance(history[i], history[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
            
        # Variance is the average distance
        # If all consistent (dist=0), variance=0
        # If highly variable (dist=1), variance=1
        avg_distance = sum(distances) / len(distances)
        return avg_distance

    def get_certainty(self, query_hash: str) -> float:
        """
        Calculate subjective certainty for a query.
        
        Certainty = 1.0 - Variance
        
        Returns:
            Certainty score (0.0 to 1.0)
        """
        variance = self.calculate_variance(query_hash)
        
        # Invert variance for certainty
        # Variance 0 (consistent) -> Certainty 1
        # Variance 1 (inconsistent) -> Certainty 0
        return 1.0 - variance
