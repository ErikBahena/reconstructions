"""
Main Loop and Goal-Driven Reconstruction Engine.

This module implements the core event loop that processes goals,
encoding experiences, reconstructing memories, and running maintenance tasks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Callable
from enum import Enum
import time
import uuid
import heapq

from pathlib import Path
from .core import Fragment, Strand, Query
from .store import FragmentStore
from .encoding import Experience, Context
from .encoder import encode
from .reconstruction import reconstruct, ReconstructionConfig
from .certainty import VarianceController
from .identity import IdentityState, IdentityStore, IdentityEvolver, ActiveIdentityState
from .consolidation import ConsolidationScheduler, ConsolidationConfig
from .health import MemoryHealthMonitor
from .learning import SalienceWeightLearner
from .patterns import CrossSessionPatternDetector
from .llm_client import LLMConfig


class GoalType(Enum):
    """Types of goals the engine can process."""
    QUERY = "query"           # Reconstruct memory
    ENCODE = "encode"         # Encode new experience
    REFLECT = "reflect"       # Self-reflection on memories
    CONSOLIDATION = "consolidation"  # Autonomous memory consolidation
    MAINTENANCE = "maintenance"  # Maintenance tasks (decay, cleanup)
    IDLE = "idle"            # Idle processing when nothing else to do


@dataclass(order=True)
class EngineGoal:
    """
    A goal for the engine to process.
    
    Goals are prioritized: lower priority number = higher priority.
    """
    priority: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    goal_type: GoalType = field(default=GoalType.QUERY, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    

class GoalQueue:
    """Priority queue for goals."""
    
    def __init__(self):
        self._heap: List[EngineGoal] = []
    
    def push(self, goal: EngineGoal) -> None:
        """Add a goal to the queue."""
        heapq.heappush(self._heap, goal)
    
    def pop(self) -> Optional[EngineGoal]:
        """Remove and return the highest-priority goal."""
        if self._heap:
            return heapq.heappop(self._heap)
        return None
    
    def peek(self) -> Optional[EngineGoal]:
        """Look at the highest-priority goal without removing it."""
        if self._heap:
            return self._heap[0]
        return None
    
    def __len__(self) -> int:
        return len(self._heap)
    
    def is_empty(self) -> bool:
        return len(self._heap) == 0


class ResultType(Enum):
    """Types of results from processing."""
    STRAND = "strand"          # Memory reconstruction result
    ENCODED = "encoded"        # New fragment encoded
    REFLECTED = "reflected"    # Reflection output
    CONSOLIDATED = "consolidated"  # Consolidation completed
    MAINTAINED = "maintained"  # Maintenance completed
    ERROR = "error"           # Error occurred


@dataclass
class Result:
    """Result of processing a goal."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result_type: ResultType = ResultType.STRAND
    goal_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class ReconstructionEngine:
    """
    The main engine that drives the memory system.
    
    Processes goals in priority order:
    1. QUERY - Reconstruct memories on demand
    2. ENCODE - Store new experiences
    3. REFLECT - Process and consolidate
    4. MAINTENANCE - Cleanup and decay
    5. IDLE - Background processing
    """
    
    def __init__(
        self,
        store: FragmentStore,
        identity_store: Optional[IdentityStore] = None,
        config: Optional[ReconstructionConfig] = None,
        consolidation_config: Optional[ConsolidationConfig] = None,
        enable_consolidation: bool = True,
        health_monitor: Optional[MemoryHealthMonitor] = None,
        enable_weight_learning: bool = True,
        llm_config: Optional[LLMConfig] = None
    ):
        self.store = store
        self.identity_store = identity_store or IdentityStore()
        self.config = config or ReconstructionConfig()
        self.llm_config = llm_config

        self.goal_queue = GoalQueue()
        self.context = Context()
        self.variance_controller = VarianceController()
        self.identity_evolver = IdentityEvolver()

        # Active identity state for identity-aware encoding
        current_identity = self.identity_store.get_current_state()
        self.active_identity = ActiveIdentityState(current_identity)

        # Weight learning for adaptive salience calculation
        self.weight_learner = None
        if enable_weight_learning:
            weights_path = Path.home() / ".reconstructions" / "weights.json"
            if weights_path.exists():
                try:
                    self.weight_learner = SalienceWeightLearner.load_checkpoint(weights_path)
                except Exception:
                    self.weight_learner = SalienceWeightLearner()
            else:
                self.weight_learner = SalienceWeightLearner()

        # Pattern detection for cross-session pattern recognition
        self.pattern_detector = CrossSessionPatternDetector(store)

        # Health monitoring
        self.health_monitor = health_monitor or MemoryHealthMonitor(store)

        # Consolidation scheduler
        self.consolidation_scheduler = None
        if enable_consolidation:
            self.consolidation_scheduler = ConsolidationScheduler(
                store,
                consolidation_config or ConsolidationConfig(),
                config,
                self.health_monitor
            )

        self._running = False
        self._last_maintenance = time.time()
        self._maintenance_interval = 300  # 5 minutes
        self._last_activity = time.time()  # Track for idle detection
        self._consolidation_count = 0  # Track for checkpoint saving
        
    def submit_query(self, query: Query, priority: float = 1.0) -> str:
        """
        Submit a query goal.
        
        Args:
            query: Query to process
            priority: Priority (lower = higher priority)
            
        Returns:
            Goal ID
        """
        goal = EngineGoal(
            priority=priority,
            goal_type=GoalType.QUERY,
            payload={"query": query}
        )
        self.goal_queue.push(goal)
        return goal.id
    
    def submit_experience(self, experience: Experience, priority: float = 2.0) -> str:
        """
        Submit an experience to encode.
        
        Args:
            experience: Experience to encode
            priority: Priority
            
        Returns:
            Goal ID
        """
        goal = EngineGoal(
            priority=priority,
            goal_type=GoalType.ENCODE,
            payload={"experience": experience}
        )
        self.goal_queue.push(goal)
        return goal.id
    
    def submit_maintenance(self, priority: float = 5.0) -> str:
        """Submit a maintenance goal."""
        goal = EngineGoal(
            priority=priority,
            goal_type=GoalType.MAINTENANCE,
            payload={}
        )
        self.goal_queue.push(goal)
        return goal.id
    
    def process_goal(self, goal: EngineGoal) -> Result:
        """
        Process a single goal.

        Dispatches to the appropriate handler based on goal type.
        """
        try:
            if goal.goal_type == GoalType.QUERY:
                return self._process_query(goal)
            elif goal.goal_type == GoalType.ENCODE:
                return self._process_encode(goal)
            elif goal.goal_type == GoalType.CONSOLIDATION:
                return self._process_consolidation(goal)
            elif goal.goal_type == GoalType.MAINTENANCE:
                return self._process_maintenance(goal)
            elif goal.goal_type == GoalType.REFLECT:
                return self._process_reflect(goal)
            else:
                return self._process_idle(goal)
        except Exception as e:
            return Result(
                result_type=ResultType.ERROR,
                goal_id=goal.id,
                success=False,
                error_message=str(e)
            )
    
    def _process_query(self, goal: EngineGoal) -> Result:
        """Process a query goal."""
        query = goal.payload.get("query")
        if not query:
            return Result(
                result_type=ResultType.ERROR,
                goal_id=goal.id,
                success=False,
                error_message="No query in goal payload"
            )

        # Track query timing
        start_time = time.time()

        strand = reconstruct(
            query,
            self.store,
            variance_target=0.3,
            config=self.config,
            variance_controller=self.variance_controller,
            llm_config=self.llm_config
        )

        # Log query metrics
        latency_ms = (time.time() - start_time) * 1000
        if self.health_monitor:
            self.health_monitor.log_query(query, strand, latency_ms)

        # Record query for adaptive scheduling
        if self.consolidation_scheduler:
            self.consolidation_scheduler.record_query()

        # Record feedback for weight learning
        if self.weight_learner:
            # Query is successful if coherence > 0.5
            was_successful = strand.coherence_score > 0.5

            # Record feedback for each retrieved fragment
            for frag_id in strand.fragments:
                fragment = self.store.get(frag_id)
                if fragment:
                    self.weight_learner.record_retrieval(fragment, was_successful)

        return Result(
            result_type=ResultType.STRAND,
            goal_id=goal.id,
            data={"strand": strand, "certainty": strand.certainty}
        )
    
    def _process_encode(self, goal: EngineGoal) -> Result:
        """Process an encode goal."""
        experience = goal.payload.get("experience")
        if not experience:
            return Result(
                result_type=ResultType.ERROR,
                goal_id=goal.id,
                success=False,
                error_message="No experience in goal payload"
            )

        # Encode with identity-aware salience boosting and learned weights
        fragment = encode(
            experience,
            self.context,
            self.store,
            identity_state=self.active_identity,
            weight_learner=self.weight_learner,
            llm_config=self.llm_config
        )

        # Record encoding for adaptive scheduling
        if self.consolidation_scheduler:
            self.consolidation_scheduler.record_encoding(fragment.initial_salience)

        return Result(
            result_type=ResultType.ENCODED,
            goal_id=goal.id,
            data={"fragment_id": fragment.id, "salience": fragment.initial_salience}
        )
    
    def _process_maintenance(self, goal: EngineGoal) -> Result:
        """Process a maintenance goal."""
        # Placeholder for maintenance tasks:
        # - Could run decay calculations
        # - Could consolidate weak memories
        # - Could update identity state
        
        self._last_maintenance = time.time()
        
        return Result(
            result_type=ResultType.MAINTAINED,
            goal_id=goal.id,
            data={"timestamp": self._last_maintenance}
        )
    
    def _process_reflect(self, goal: EngineGoal) -> Result:
        """Process a reflection goal."""
        # Placeholder for reflection:
        # - Could analyze recent memories
        # - Could update beliefs based on patterns
        # - Could consolidate related experiences
        
        return Result(
            result_type=ResultType.REFLECTED,
            goal_id=goal.id,
            data={}
        )
    
    def _process_consolidation(self, goal: EngineGoal) -> Result:
        """Process autonomous consolidation."""
        if self.consolidation_scheduler is None:
            return Result(
                result_type=ResultType.ERROR,
                goal_id=goal.id,
                success=False,
                error_message="Consolidation scheduler not enabled"
            )

        stats = self.consolidation_scheduler.consolidate()

        # Increment consolidation counter
        self._consolidation_count += 1

        # Save weight learner checkpoint every 10 consolidations
        if self.weight_learner and self._consolidation_count % 10 == 0:
            weights_path = Path.home() / ".reconstructions" / "weights.json"
            try:
                self.weight_learner.save_checkpoint(weights_path)
            except Exception:
                pass  # Don't fail consolidation if checkpoint save fails

        # Run pattern detection every 10 consolidations
        if self._consolidation_count % 10 == 0:
            try:
                # Detect all pattern types
                temporal = self.pattern_detector.detect_temporal_patterns()
                workflows = self.pattern_detector.detect_workflow_patterns()
                projects = self.pattern_detector.detect_project_switches()

                # Save detected patterns
                self.pattern_detector.save_patterns()

                # Add pattern counts to stats
                stats["patterns_detected"] = {
                    "temporal": len(temporal),
                    "workflow": len(workflows),
                    "project": len(projects)
                }
            except Exception:
                pass  # Don't fail consolidation if pattern detection fails

        return Result(
            result_type=ResultType.CONSOLIDATED,
            goal_id=goal.id,
            data=stats
        )

    def _process_idle(self, goal: EngineGoal) -> Result:
        """Process idle time."""
        # Could do background consolidation
        return Result(
            result_type=ResultType.MAINTAINED,
            goal_id=goal.id,
            data={}
        )

    def get_detected_patterns(self) -> dict:
        """
        Get all detected patterns.

        Returns:
            Dictionary with temporal, workflow, and project patterns
        """
        return self.pattern_detector.get_all_patterns()

    def force_pattern_detection(self) -> dict:
        """
        Manually trigger pattern detection.

        Returns:
            Dictionary with counts of detected patterns
        """
        temporal = self.pattern_detector.detect_temporal_patterns()
        workflows = self.pattern_detector.detect_workflow_patterns()
        projects = self.pattern_detector.detect_project_switches()

        self.pattern_detector.save_patterns()

        return {
            "temporal": len(temporal),
            "workflow": len(workflows),
            "project": len(projects)
        }
    
    def step(self) -> Optional[Result]:
        """
        Process a single goal from the queue.

        Returns:
            Result if goal processed, None if queue empty
        """
        current_time = time.time()

        # Check if maintenance is needed
        if current_time - self._last_maintenance > self._maintenance_interval:
            self.submit_maintenance(priority=4.0)

        # Check if consolidation is needed
        if (self.consolidation_scheduler is not None and
            self.consolidation_scheduler.should_consolidate(current_time)):
            # Schedule consolidation (lower priority than active tasks)
            goal = EngineGoal(
                priority=3.0,
                goal_type=GoalType.CONSOLIDATION,
                payload={}
            )
            self.goal_queue.push(goal)

        goal = self.goal_queue.pop()
        if goal is None:
            return None

        # Track activity for idle detection
        self._last_activity = current_time

        return self.process_goal(goal)

    def query(self, query: Query) -> Optional[Strand]:
        """
        Convenience method to submit and process a query immediately.

        Args:
            query: Query to process

        Returns:
            Strand if successful, None if error
        """
        self.submit_query(query, priority=0.0)  # Highest priority
        result = self.step()

        if result is None:
            return None

        if result.result_type == ResultType.STRAND and result.success:
            return result.data.get("strand")

        return None
    
    def run(self, max_iterations: Optional[int] = None) -> List[Result]:
        """
        Run the main loop.
        
        Args:
            max_iterations: Maximum iterations (None = run until queue empty)
            
        Returns:
            List of results
        """
        self._running = True
        results = []
        iterations = 0
        
        while self._running:
            if max_iterations is not None and iterations >= max_iterations:
                break
            
            result = self.step()
            if result is None:
                break
            
            results.append(result)
            iterations += 1
        
        return results
    
    def stop(self) -> None:
        """Stop the main loop."""
        self._running = False
