"""Tests for Claude Code skills."""

import pytest
import tempfile
from pathlib import Path


class TestSkills:
    """Tests for skill operations."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset session singleton before and after each test."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    @pytest.fixture
    def session(self):
        """Create an active session."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()
            session.start_session(db_path=db_path, project_path=tmpdir)
            yield session

    def test_store_operation(self, session):
        """store operation creates fragment."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("store", "Remember this important note")

        assert result["success"] is True
        assert "fragment_id" in result
        assert result["salience"] > 0

    def test_store_empty_text_fails(self, session):
        """store with empty text returns error."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("store", "")

        assert "error" in result

    def test_recall_operation(self, session):
        """recall operation searches memories."""
        from reconstructions.claude_code.skills import execute_skill

        # Store first
        execute_skill("store", "I love pizza for dinner")

        # Recall
        result = execute_skill("recall", "favorite food")

        assert result["success"] is True
        assert "fragments" in result
        assert "certainty" in result

    def test_recall_empty_query_fails(self, session):
        """recall with empty query returns error."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("recall", "")

        assert "error" in result

    def test_status_operation(self, session):
        """status operation returns system info."""
        from reconstructions.claude_code.skills import execute_skill

        # Store a memory first
        execute_skill("store", "test memory")

        result = execute_skill("status", "")

        assert "fragment_count" in result
        assert result["fragment_count"] >= 1
        assert "session" in result

    def test_identity_operation(self, session):
        """identity operation returns identity state."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("identity", "")

        assert "traits" in result
        assert "beliefs" in result
        assert "goals" in result

    def test_recent_operation(self, session):
        """recent operation lists recent fragments."""
        from reconstructions.claude_code.skills import execute_skill

        # Store some memories
        execute_skill("store", "Memory one")
        execute_skill("store", "Memory two")
        execute_skill("store", "Memory three")

        result = execute_skill("recent", "2")

        assert "fragments" in result
        assert len(result["fragments"]) == 2

    def test_recent_default_count(self, session):
        """recent uses default count of 10."""
        from reconstructions.claude_code.skills import execute_skill

        # Store 3 memories
        for i in range(3):
            execute_skill("store", f"Memory {i}")

        result = execute_skill("recent", "")

        assert len(result["fragments"]) == 3  # Only 3 stored

    def test_forget_operation(self, session):
        """forget operation reduces salience."""
        from reconstructions.claude_code.skills import execute_skill

        # Store a memory
        store_result = execute_skill("store", "Something I want to forget")
        fragment_id = store_result["fragment_id"]

        # Forget it
        forget_result = execute_skill("forget", fragment_id)

        assert forget_result["success"] is True
        assert forget_result["new_salience"] < forget_result["original_salience"]

    def test_forget_partial_id(self, session):
        """forget works with partial ID."""
        from reconstructions.claude_code.skills import execute_skill

        # Store a memory
        store_result = execute_skill("store", "Something to forget")
        fragment_id = store_result["fragment_id"]

        # Forget with partial ID (first 8 chars)
        forget_result = execute_skill("forget", fragment_id[:8])

        assert forget_result["success"] is True

    def test_forget_nonexistent_id(self, session):
        """forget with nonexistent ID returns error."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("forget", "nonexistent-id-12345")

        assert "error" in result

    def test_unknown_operation(self, session):
        """Unknown operation returns error with available list."""
        from reconstructions.claude_code.skills import execute_skill

        result = execute_skill("unknown_op", "")

        assert "error" in result
        assert "available" in result
        assert "store" in result["available"]

    def test_format_output_store(self, session):
        """format_output handles store result."""
        from reconstructions.claude_code.skills import execute_skill, format_output

        result = execute_skill("store", "test")
        output = format_output(result)

        assert "Stored memory" in output
        assert "Salience" in output

    def test_format_output_error(self):
        """format_output handles error result."""
        from reconstructions.claude_code.skills import format_output

        output = format_output({"error": "Something went wrong"})

        assert "Error:" in output
        assert "Something went wrong" in output

    def test_format_output_recall_empty(self, session):
        """format_output handles empty recall."""
        from reconstructions.claude_code.skills import format_output

        output = format_output({"success": True, "fragments": [], "certainty": 0.0})

        assert "No memories found" in output

    def test_format_output_status(self, session):
        """format_output handles status result."""
        from reconstructions.claude_code.skills import execute_skill, format_output

        result = execute_skill("status", "")
        output = format_output(result)

        assert "Total memories" in output


class TestSkillsWithoutSession:
    """Tests for skill behavior without active session."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset session singleton."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    def test_skill_auto_starts_session(self):
        """Skills auto-start session if needed."""
        from reconstructions.claude_code.skills import execute_skill
        from reconstructions.claude_code import get_session

        result = execute_skill("status", "")

        assert get_session().is_active
        assert "fragment_count" in result
