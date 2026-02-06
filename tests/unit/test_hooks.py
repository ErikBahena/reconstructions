"""Tests for Claude Code hooks."""

import pytest
import tempfile
from pathlib import Path


class TestHooks:
    """Tests for hook functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset session singleton before and after each test."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield str(Path(tmpdir) / "test.db")

    def test_session_start_hook(self, temp_db):
        """on_session_start initializes session."""
        from reconstructions.claude_code.hooks import on_session_start
        from reconstructions.claude_code import get_session

        result = on_session_start(
            project_path="/test/project",
            db_path=temp_db
        )

        assert result["status"] == "started"
        assert result["session_id"] is not None
        assert get_session().is_active

    def test_session_start_already_active(self, temp_db):
        """on_session_start with active session returns already_active."""
        from reconstructions.claude_code.hooks import on_session_start

        # Start first
        on_session_start(db_path=temp_db)

        # Start again
        result = on_session_start(db_path=temp_db)

        assert result["status"] == "already_active"

    def test_post_tool_use_captures_write(self, temp_db):
        """on_post_tool_use captures Write events."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code import get_session

        on_session_start(db_path=temp_db)

        event = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/foo/bar.py",
                "content": "print('hello')"
            },
            "success": True
        }

        result = on_post_tool_use(event)

        assert result["status"] == "captured"
        assert result["fragment_id"] is not None
        assert get_session().stats.fragments_encoded == 1

    def test_post_tool_use_skips_read(self, temp_db):
        """on_post_tool_use skips Read events."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code import get_session

        on_session_start(db_path=temp_db)

        event = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/foo/bar.py"}
        }

        result = on_post_tool_use(event)

        assert result["status"] == "skipped"
        assert result["reason"] == "noise_tool"
        assert get_session().stats.fragments_encoded == 0

    def test_post_tool_use_lazy_starts_session(self):
        """on_post_tool_use starts session if not active."""
        from reconstructions.claude_code.hooks import on_post_tool_use
        from reconstructions.claude_code import get_session

        event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/foo.py", "content": "test"},
            "success": True
        }

        result = on_post_tool_use(event)

        assert result["status"] == "captured"
        assert get_session().is_active

    def test_stop_hook_encodes_summary(self, temp_db, tmp_path):
        """on_stop encodes assistant response from transcript."""
        from reconstructions.claude_code.hooks import on_session_start, on_stop
        from reconstructions.claude_code import get_session
        import json

        on_session_start(db_path=temp_db)

        # Create a mock transcript file with an assistant message
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            json.dumps({"role": "user", "content": "Hello"}) + "\n" +
            json.dumps({"role": "assistant", "content": "I implemented feature X for you."}) + "\n"
        )

        result = on_stop(event={"transcript_path": str(transcript_path)})

        assert result["status"] == "captured"
        assert get_session().stats.fragments_encoded == 1

    def test_stop_hook_without_summary(self, temp_db):
        """on_stop without summary returns ok."""
        from reconstructions.claude_code.hooks import on_session_start, on_stop

        on_session_start(db_path=temp_db)

        result = on_stop()

        assert result["status"] == "ok"

    def test_stop_hook_without_session(self):
        """on_stop without session returns no_session."""
        from reconstructions.claude_code.hooks import on_stop

        result = on_stop()

        assert result["status"] == "no_session"

    def test_session_end_hook(self, temp_db):
        """on_session_end finalizes session."""
        from reconstructions.claude_code.hooks import (
            on_session_start,
            on_post_tool_use,
            on_session_end
        )
        from reconstructions.claude_code import get_session

        on_session_start(db_path=temp_db)

        # Capture some events
        on_post_tool_use({
            "tool_name": "Edit",
            "tool_input": {"file_path": "/foo.py", "old_string": "a", "new_string": "b"},
            "success": True
        })

        result = on_session_end()

        assert result["status"] == "ended"
        assert result["fragments_encoded"] >= 1
        assert not get_session().is_active

    def test_session_end_without_session(self):
        """on_session_end without session returns no_session."""
        from reconstructions.claude_code.hooks import on_session_end

        result = on_session_end()

        assert result["status"] == "no_session"

    def test_tool_uses_observed_tracked(self, temp_db):
        """Tool uses are tracked even when skipped."""
        from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
        from reconstructions.claude_code import get_session

        on_session_start(db_path=temp_db)

        # Send some noise events
        for _ in range(5):
            on_post_tool_use({"tool_name": "Read", "tool_input": {}})

        # Send one captured event
        on_post_tool_use({
            "tool_name": "Write",
            "tool_input": {"file_path": "/x.py", "content": "y"},
            "success": True
        })

        stats = get_session().stats
        assert stats.tool_uses_observed == 6
        assert stats.tools_captured == 1
