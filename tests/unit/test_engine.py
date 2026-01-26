"""
Unit tests for main loop and goal-driven engine.
"""

import pytest
import tempfile
from pathlib import Path
from reconstructions.core import Query
from reconstructions.store import FragmentStore
from reconstructions.encoding import Experience
from reconstructions.engine import (
    GoalType,
    EngineGoal,
    GoalQueue,
    ResultType,
    Result,
    ReconstructionEngine
)


class TestGoalQueue:
    """Test goal queue priority."""
    
    def test_push_pop_single(self):
        """Push and pop a single goal."""
        q = GoalQueue()
        g = EngineGoal(priority=1.0, goal_type=GoalType.QUERY)
        q.push(g)
        
        popped = q.pop()
        assert popped is not None
        assert popped.id == g.id
    
    def test_priority_order(self):
        """Goals pop in priority order (lowest first)."""
        q = GoalQueue()
        
        g1 = EngineGoal(priority=3.0, goal_type=GoalType.QUERY)
        g2 = EngineGoal(priority=1.0, goal_type=GoalType.ENCODE)
        g3 = EngineGoal(priority=2.0, goal_type=GoalType.MAINTENANCE)
        
        q.push(g1)
        q.push(g2)
        q.push(g3)
        
        assert q.pop().priority == 1.0
        assert q.pop().priority == 2.0
        assert q.pop().priority == 3.0
    
    def test_pop_empty(self):
        """Pop empty queue returns None."""
        q = GoalQueue()
        assert q.pop() is None
    
    def test_is_empty(self):
        """is_empty works correctly."""
        q = GoalQueue()
        assert q.is_empty()
        
        q.push(EngineGoal(priority=1.0))
        assert not q.is_empty()


class TestResult:
    """Test Result dataclass."""
    
    def test_create_result(self):
        """Create a result."""
        r = Result(
            result_type=ResultType.STRAND,
            goal_id="test-goal",
            data={"strand": "data"}
        )
        
        assert r.result_type == ResultType.STRAND
        assert r.success is True
        assert r.id is not None


class TestReconstructionEngine:
    """Test the reconstruction engine."""
    
    def test_submit_query(self):
        """Submit a query goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            query = Query(semantic="test")
            goal_id = engine.submit_query(query)
            
            assert goal_id is not None
            assert len(engine.goal_queue) == 1
            
            store.close()
    
    def test_submit_experience(self):
        """Submit an experience goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            exp = Experience(text="Test memory")
            goal_id = engine.submit_experience(exp)
            
            assert goal_id is not None
            assert len(engine.goal_queue) == 1
            
            store.close()
    
    def test_process_encode(self):
        """Process encoding goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            exp = Experience(text="Test memory")
            engine.submit_experience(exp)
            
            result = engine.step()
            
            assert result is not None
            assert result.result_type == ResultType.ENCODED
            assert "fragment_id" in result.data
            
            store.close()
    
    def test_process_query(self):
        """Process query goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            # First encode something
            exp = Experience(text="Test memory about cats")
            engine.submit_experience(exp, priority=1.0)
            engine.step()
            
            # Then query
            query = Query(semantic="cats")
            engine.submit_query(query, priority=1.0)
            
            result = engine.step()
            
            assert result is not None
            assert result.result_type == ResultType.STRAND
            assert "strand" in result.data
            
            store.close()
    
    def test_run_multiple_goals(self):
        """Run processes multiple goals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            # Submit multiple goals
            engine.submit_experience(Experience(text="Memory 1"))
            engine.submit_experience(Experience(text="Memory 2"))
            engine.submit_experience(Experience(text="Memory 3"))
            
            results = engine.run()
            
            assert len(results) == 3
            assert all(r.result_type == ResultType.ENCODED for r in results)
            
            store.close()
    
    def test_priority_ordering(self):
        """Goals processed in priority order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            # Submit with different priorities
            engine.submit_experience(Experience(text="Low priority"), priority=5.0)
            engine.submit_experience(Experience(text="High priority"), priority=1.0)
            engine.submit_experience(Experience(text="Medium priority"), priority=3.0)
            
            # Check order via goal IDs
            g1 = engine.goal_queue.pop()
            g2 = engine.goal_queue.pop()
            g3 = engine.goal_queue.pop()
            
            assert g1.priority == 1.0
            assert g2.priority == 3.0
            assert g3.priority == 5.0
            
            store.close()
    
    def test_stop(self):
        """Stop signal works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            
            engine._running = True
            engine.stop()
            
            assert engine._running is False
            
            store.close()
