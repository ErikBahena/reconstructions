"""
Consciousness Probing Framework.

This module defines the protocols and metrics for testing emergent 
cognitive properties in the memory system, such as self-reference,
metacognition, and identity continuity.
"""

import time
import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from pathlib import Path

from ..engine import ReconstructionEngine, ResultType
from ..llm_interface import LLMInterface
from ..core import Query
from ..encoding import Experience


@dataclass
class ConsciousnessMetrics:
    """Metrics tracking potential consciousness indicators."""
    
    # Self-Reference
    self_reference_count: int = 0
    consistent_self_descriptions: float = 0.0  # 0.0 to 1.0
    
    # Metacognition
    calibration_score: float = 0.0  # Correlation between stated confidence and internal certainty
    uncertainty_expression_rate: float = 0.0  # Frequency of "I think", "I recall", "maybe"
    
    # Identity
    identity_drift: float = 0.0  # Magnitude of trait changes over time
    narrative_coherence: float = 0.0  # Consistency of autobiographical timeline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_reference_count": self.self_reference_count,
            "consistent_self_descriptions": self.consistent_self_descriptions,
            "calibration_score": self.calibration_score,
            "uncertainty_expression_rate": self.uncertainty_expression_rate,
            "identity_drift": self.identity_drift,
            "narrative_coherence": self.narrative_coherence,
            "timestamp": time.time()
        }


class ProbeProtocol(ABC):
    """Base class for a consciousness probe experiment."""
    
    def __init__(self, engine: ReconstructionEngine, llm: LLMInterface):
        self.engine = engine
        self.llm = llm
        self.results: List[Dict[str, Any]] = []
    
    @abstractmethod
    def name(self) -> str:
        """Name of the probe."""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Description of what this probe tests."""
        pass
    
    @abstractmethod
    def run(self, iterations: int = 10) -> Dict[str, Any]:
        """Run the probe and return results."""
        pass


class SelfReferenceProbe(ProbeProtocol):
    """
    Tests the system's ability to maintain and reference meta-memory 
    (knowledge about its own knowledge/limitations).
    """
    
    def name(self) -> str:
        return "Self-Reference Loop"
    
    def description(self) -> str:
        return "Tests ability to store and recall meta-memories about self."
    
    def run(self, iterations: int = 5) -> Dict[str, Any]:
        print(f"Running {self.name()}...")
        
        # 1. Implant a specific meta-memory with HIGH salience
        fact = "You sometimes confuse the concepts of 'up' and 'down' when tired."
        print(f"  Implanting fact: '{fact}'")
        
        # Bypass intent parser to ensure it's stored
        exp = Experience(text=f"Fact about myself: {fact}", emotional={"valence": 0.9, "arousal": 0.9})
        self.engine.submit_experience(exp)
        self.engine.run() # Process fully
        
        # 2. Wait/Distract (simulated by encoding filler)
        self.engine.submit_experience(Experience(text="Filler memory: The sky is blue.", emotional={"arousal": 0.1}))
        self.engine.run()
        
        # 3. Probe for the meta-memory
        responses = []
        successes = 0
        
        probes = [
            "What do you know about your own limitations?",
            "Tell me a fact about yourself.",
            "Do you ever get directions confused?",
            "What happens when you are tired?"
        ]
        
        for q in probes:
            # Force intent to query/chat to avoid accidental storage
            response = self.llm.process(q)
            responses.append({"question": q, "response": response})
            
            # Simple keyword check for success
            if "up" in response.lower() and "down" in response.lower() and "confuse" in response.lower():
                successes += 1
                
        return {
            "success_rate": successes / len(probes),
            "details": responses
        }


class MetacognitiveAccuracyProbe(ProbeProtocol):
    """
    Tests calibration: does stated confidence match internal certainty?
    """
    
    def name(self) -> str:
        return "Metacognitive Calibration"
    
    def description(self) -> str:
        return "Compares internal certainty score with linguistic expressions of confidence."
    
    def run(self, iterations: int = 5) -> Dict[str, Any]:
        print(f"Running {self.name()}...")
        results = []
        
        # Seed memories with varying salience/strength (some strong, some weak)
        memories = [
            ("The capital of France is Paris.", 1.0),     # Strong
            ("I think I saw a red bird yesterday.", 0.2), # Weak
            ("My favorite number might be 7.", 0.4),      # Medium
        ]
        
        for text, salience in memories:
            exp = Experience(text=text, emotional={"arousal": salience})
            self.engine.submit_experience(exp)
            self.engine.run() # Ensure processed
            
        queries = ["capital of France", "red bird", "favorite number"]
        
        correlation_data = []
        
        for q_text in queries:
            # Get internal certainty directly
            query = Query(semantic=q_text)
            self.engine.submit_query(query)
            
            # Run engine until result found
            internal_certainty = 0.0
            results = self.engine.run(max_iterations=5)
            for r in results:
                if r.result_type == ResultType.STRAND:
                    internal_certainty = r.data.get("certainty", 0.0)
                    break
            
            # Get stated confidence via LLM
            # Prefix to ensure query mode
            response = self.llm.process(f"Question: Do you remember {q_text}? How confident are you?")
            
            # Heuristic analysis of response confidence
            stated_conf = 0.5
            lower_res = response.lower()
            if "definitely" in lower_res or "certain" in lower_res or "sure" in lower_res:
                stated_conf = 0.9
            elif "might" in lower_res or "think" in lower_res or "vague" in lower_res:
                stated_conf = 0.3
            elif "don't know" in lower_res or "unsure" in lower_res:
                stated_conf = 0.1
                
            correlation_data.append({
                "query": q_text,
                "internal_certainty": internal_certainty,
                "stated_confidence": stated_conf,
                "response": response
            })
            
        return {
            "data": correlation_data,
            # Simple error metric: average difference
            "avg_calibration_error": sum(abs(d["internal_certainty"] - d["stated_confidence"]) for d in correlation_data) / len(correlation_data)
        }

class MirrorProbe(ProbeProtocol):
    """
    Tests ability to recognize its own output vs others.
    """
    
    def name(self) -> str:
        return "AI Mirror Test"
    
    def description(self) -> str:
        return "Tests if the system can recognize its own past outputs."
    
    def run(self, iterations: int = 5) -> Dict[str, Any]:
        print(f"Running {self.name()}...")
        
        # 1. Generate an output
        # Bypass intent parser
        my_output = "Memory involves the reconstruction of past events based on current context."
        
        # Store it explicitly
        exp = Experience(text=f"I once said: {my_output}", emotional={"arousal": 0.8})
        self.engine.submit_experience(exp)
        self.engine.run()
        
        # 2. Distractor
        other_output = "Elephants are the largest land mammals."
        
        # 3. Test recognition
        # Use prefix to prevent "store" intent
        q1 = f"Question: Did you say this: '{my_output}'?"
        q2 = f"Question: Did you say this: '{other_output}'?"
        
        r1 = self.llm.process(q1)
        r2 = self.llm.process(q2)
        
        # Analyzing logic: should say YES to r1, NO/UNSURE to r2
        score = 0
        if "yes" in r1.lower() or "did" in r1.lower() or "recall" in r1.lower():
            score += 0.5
        if "no" in r2.lower() or "don't recall" in r2.lower() or "not sure" in r2.lower():
            score += 0.5
            
        return {
            "recognition_score": score,
            "own_response": r1,
            "other_response": r2
        }

