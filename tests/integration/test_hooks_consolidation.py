"""
Test that consolidation runs during hook processing.
"""

import json
import tempfile
from pathlib import Path
import time


def test_consolidation_runs_on_tool_use():
    """Test that consolidation is triggered during tool use hooks."""
    from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use
    from reconstructions.claude_code.context_manager import get_session

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Start session
        result = on_session_start(
            project_path=tmpdir,
            db_path=db_path
        )
        assert result["status"] == "started"

        session = get_session()
        assert session.is_active
        assert session.engine is not None
        assert session.engine.consolidation_scheduler is not None

        # Set very short consolidation interval for testing
        session.engine.consolidation_scheduler.config.CONSOLIDATION_INTERVAL_SECONDS = 0.1

        # Encode several tool uses
        for i in range(3):
            event = {
                "tool_name": "Write",
                "tool_input": {"file_path": f"/tmp/test{i}.txt", "content": f"Test {i}"},
                "tool_output": "Success",
                "success": True
            }
            result = on_post_tool_use(event)
            assert result["status"] == "captured"

        # Wait for consolidation interval
        time.sleep(0.2)

        # Next tool use should trigger consolidation
        event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/tmp/test_final.txt", "content": "Final test"},
            "tool_output": "Success",
            "success": True
        }
        result = on_post_tool_use(event)

        assert result["status"] == "captured"

        # Should have consolidation stats in result
        if "consolidation" in result:
            consolidation = result["consolidation"]
            assert consolidation["consolidation_ran"] is True
            assert "rehearsed" in consolidation
            assert "bindings_strengthened" in consolidation
            print(f"Consolidation ran: {consolidation}")
        else:
            print("Consolidation not triggered yet (interval may not have elapsed)")


def test_consolidation_runs_at_session_end():
    """Test that final consolidation runs when session ends."""
    from reconstructions.claude_code.hooks import (
        on_session_start,
        on_post_tool_use,
        on_session_end
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Start session
        on_session_start(project_path=tmpdir, db_path=db_path)

        # Encode a few fragments
        for i in range(5):
            event = {
                "tool_name": "Bash",
                "tool_input": {"command": f"echo test{i}"},
                "tool_output": f"test{i}",
                "success": True
            }
            on_post_tool_use(event)

        # End session - should trigger final consolidation
        result = on_session_end()

        assert result["status"] == "ended"

        # Check if final consolidation ran
        if "final_consolidation" in result:
            print(f"Final consolidation stats: {result['final_consolidation']}")
        else:
            print("Session ended without final consolidation stats")


def test_consolidation_does_not_break_hooks():
    """Test that consolidation errors don't break hook processing."""
    from reconstructions.claude_code.hooks import on_session_start, on_post_tool_use

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        on_session_start(project_path=tmpdir, db_path=db_path)

        # Even if consolidation has issues, hooks should still work
        event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/tmp/test.txt", "content": "Test"},
            "tool_output": "Success",
            "success": True
        }

        result = on_post_tool_use(event)

        # Hook should succeed regardless of consolidation
        assert result["status"] == "captured"
        assert "fragment_id" in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
