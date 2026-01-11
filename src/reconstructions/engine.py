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

from .core import Fragment, Strand, Query
from .store import FragmentStore
from .encoding import Experience, Context
from .encoder import encode
from .reconstruction import reconstruct, ReconstructionConfig
from .certainty import VarianceController
from .identity import IdentityState, IdentityStore, IdentityEvolver


class GoalType(Enum):
    """Types of goals the engine can process."""
    QUERY = "query"           # Reconstruct memory
    ENCODE = "encode"         # Encode new experience
    REFLECT = "reflect"       # Self-reflection on memories
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
        config: Optional[ReconstructionConfig] = None
    ):
        self.store = store
        self.identity_store = identity_store or IdentityStore()
        self.config = config or ReconstructionConfig()
        
        self.goal_queue = GoalQueue()
        self.context = Context()
        self.variance_controller = VarianceController()
        self.identity_evolver = IdentityEvolver()
        
        self._running = False
        self._last_maintenance = time.time()
        self._maintenance_interval = 300  # 5 minutes
        
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
        
        strand = reconstruct(
            query,
            self.store,
            variance_target=0.3,
            config=self.config,
            variance_controller=self.variance_controller
        )
        
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
        
        fragment = encode(experience, self.context, self.store)
        
        return Result(
            result_type=ResultType.ENCODED,
            goal_id=goal.id,
            data={"fragment_id": fragment.id}
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
    
    def _process_idle(self, goal: EngineGoal) -> Result:
        """Process idle time."""
        # Could do background consolidation
        return Result(
            result_type=ResultType.MAINTAINED,
            goal_id=goal.id,
            data={}
        )
    
    def step(self) -> Optional[Result]:
        """
        Process a single goal from the queue.
        
        Returns:
            Result if goal processed, None if queue empty
        """
        # Check if maintenance is needed
        if time.time() - self._last_maintenance > self._maintenance_interval:
            self.submit_maintenance(priority=4.0)
        
        goal = self.goal_queue.pop()
        if goal is None:
            return None
        
        return self.process_goal(goal)
    
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
