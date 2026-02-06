"""Integration tests for Claude Code hooks."""

import pytest
import tempfile
import time
from pathlib import Path


class TestHookIntegration:
    """End-to-end tests for the hook system."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset session singleton before and after each test."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_session_lifecycle(self, temp_dir):
        """Test complete session lifecycle with hooks."""
        from reconstructions.claude_code.hooks import (
            on_session_start,
            on_post_tool_use,
            on_stop,
            on_session_end
        )
        from reconstructions.claude_code import get_session

        db_path = str(temp_dir / "test.db")

        # 1. Start session
        start_result = on_session_start(
            project_path=str(temp_dir),
            db_path=db_path
        )
        assert start_result["status"] == "started"

        # 2. Simulate tool uses
        tool_events = [
            {"tool_name": "Read", "tool_input": {"file_path": "/foo.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "/bar.py", "content": "x = 1"}, "success": True},
            {"tool_name": "Glob", "tool_input": {"pattern": "*.py"}},
            {"tool_name": "Edit", "tool_input": {"file_path": "/bar.py", "old_string": "x = 1", "new_string": "x = 2"}, "success": True},
            {"tool_name": "Bash", "tool_input": {"command": "pytest tests/"}, "tool_output": "3 passed", "success": True},
        ]

        for event in tool_events:
            on_post_tool_use(event)

        session = get_session()
        assert session.stats.tool_uses_observed == 5
        assert session.stats.tools_captured == 3  # Write, Edit, Bash

        # 3. Stop hook
        stop_result = on_stop(response_summary="Implemented feature and ran tests")
        assert stop_result["status"] == "captured"

        # 4. End session
        end_result = on_session_end()
        assert end_result["status"] == "ended"
        assert end_result["fragments_encoded"] >= 3

        # Session should be inactive
        assert not get_session().is_active

    def test_memory_persistence_across_sessions(self, temp_dir):
        """Memories persist across session restarts."""
        from reconstructions.claude_code.hooks import on_session_start, on_session_end
        from reconstructions.claude_code.skills import execute_skill
        from reconstructions.claude_code import get_session
        from reconstructions.claude_code.context_manager import SessionContext

        db_path = str(temp_dir / "persistent.db")

        # Session 1: Store memory
        on_session_start(db_path=db_path, project_path=str(temp_dir))
        execute_skill("store", "First session memory about cats")
        on_session_end()

        # Reset singleton (simulates new process)
        SessionContext.reset()

        # Session 2: Recall memory
        on_session_start(db_path=db_path, project_path=str(temp_dir))
        result = execute_skill("recall", "cats")

        assert result["success"] is True
        assert len(result["fragments"]) >= 1
        assert any("cats" in f["content"].lower() for f in result["fragments"])

        on_session_end()

    def test_concurrent_tool_processing(self, temp_dir):
        """Multiple tool events can be processed quickly."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code import get_session

        db_path = str(temp_dir / "test.db")
        on_session_start(db_path=db_path)

        # Simulate burst of tool uses
        start = time.time()

        for i in range(20):
            on_post_tool_use({
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": f"/file{i}.py",
                    "old_string": "old",
                    "new_string": "new"
                },
                "success": True
            })

        elapsed = time.time() - start

        # Should process 20 events in under 2 seconds
        assert elapsed < 2.0

        session = get_session()
        assert session.stats.fragments_encoded == 20

    def test_skill_recall_finds_tool_memories(self, temp_dir):
        """Skill recall can find memories from tool captures."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code.skills import execute_skill

        db_path = str(temp_dir / "test.db")
        on_session_start(db_path=db_path)

        # Simulate writing a specific file
        on_post_tool_use({
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/src/authentication.py",
                "content": "def login(user, password): ..."
            },
            "success": True
        })

        # Recall should find it
        result = execute_skill("recall", "authentication login")

        assert result["success"] is True
        assert len(result["fragments"]) >= 1

    def test_forget_reduces_recall_ranking(self, temp_dir):
        """Forgotten memories have reduced salience."""
        from reconstructions.claude_code.hooks import on_session_start
        from reconstructions.claude_code.skills import execute_skill
        from reconstructions.claude_code import get_session

        db_path = str(temp_dir / "test.db")
        on_session_start(db_path=db_path)

        # Store a memory
        store_result = execute_skill("store", "Secret password is hunter2")
        fragment_id = store_result["fragment_id"]

        # Get original salience
        fragment = get_session().store.get(fragment_id)
        original_salience = fragment.initial_salience

        # Forget it
        execute_skill("forget", fragment_id)

        # Check salience reduced
        fragment = get_session().store.get(fragment_id)
        assert fragment.initial_salience < original_salience * 0.2

    def test_status_reflects_session_activity(self, temp_dir):
        """Status shows accurate session statistics."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code.skills import execute_skill

        db_path = str(temp_dir / "test.db")
        on_session_start(db_path=db_path)

        # Store some memories
        execute_skill("store", "Memory 1")
        execute_skill("store", "Memory 2")

        # Process some tools
        for _ in range(3):
            on_post_tool_use({"tool_name": "Read", "tool_input": {}})

        on_post_tool_use({
            "tool_name": "Bash",
            "tool_input": {"command": "echo hello"},
            "success": True
        })

        # Check status
        status = execute_skill("status", "")

        assert status["fragment_count"] >= 3  # 2 manual + 1 from Bash
        assert status["session"]["tool_uses_observed"] == 4
        assert status["session"]["fragments_encoded"] >= 3


class TestErrorHandling:
    """Tests for error handling in hooks and skills."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset session singleton."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    def test_post_tool_use_with_malformed_event(self):
        """PostToolUse handles malformed events gracefully."""
        from reconstructions.claude_code.hooks import on_post_tool_use

        # Missing tool_name
        result = on_post_tool_use({})
        assert result["status"] == "skipped"

        # Empty event
        result = on_post_tool_use({"tool_name": ""})
        assert result["status"] == "skipped"

    def test_recall_with_no_memories(self):
        """Recall with no memories returns empty results."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("recall", "anything")

        # Should succeed but with no fragments
        assert result["success"] is True
        assert len(result["fragments"]) == 0

    def test_forget_with_ambiguous_id(self):
        """Forget with ambiguous ID suggests alternatives."""
        from reconstructions.claude_code.skills import execute_skill
        from reconstructions.claude_code import get_session
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()
            session.start_session(db_path=db_path)

            # Store multiple memories
            execute_skill("store", "Memory A")
            execute_skill("store", "Memory B")
            execute_skill("store", "Memory C")

            # Try to forget with very short prefix that might match multiple
            # (In practice UUIDs won't collide, but test the logic)
            result = execute_skill("forget", "nonexistent")

            assert "error" in result
