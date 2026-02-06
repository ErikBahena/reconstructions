"""Tests for the session context manager."""

import pytest
import tempfile
from pathlib import Path


class TestSessionContext:
    """Tests for SessionContext singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from reconstructions.claude_code.context_manager import SessionContext
        SessionContext.reset()
        yield
        SessionContext.reset()

    def test_singleton_pattern(self):
        """get_session returns same instance."""
        from reconstructions.claude_code import get_session

        session1 = get_session()
        session2 = get_session()

        assert session1 is session2

    def test_start_session(self):
        """Can start a session."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()

            result = session.start_session(db_path=db_path, project_path=tmpdir)

            assert result is True
            assert session.is_active
            assert session.store is not None
            assert session.engine is not None
            assert session.context is not None
            assert session.stats is not None

    def test_start_session_twice_returns_false(self):
        """Starting session twice returns False."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()

            first = session.start_session(db_path=db_path)
            second = session.start_session(db_path=db_path)

            assert first is True
            assert second is False

    def test_end_session(self):
        """Can end a session."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()
            session.start_session(db_path=db_path)

            stats = session.end_session()

            assert stats is not None
            assert session.is_active is False
            assert session.store is None

    def test_end_session_without_start(self):
        """Ending without start returns None."""
        from reconstructions.claude_code import get_session

        session = get_session()
        stats = session.end_session()

        assert stats is None

    def test_stats_tracking(self):
        """Session stats are tracked correctly."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()
            session.start_session(db_path=db_path)

            session.increment_fragments_encoded()
            session.increment_fragments_encoded()
            session.increment_queries_processed()
            session.increment_tool_uses_observed()
            session.increment_tool_uses_observed()
            session.increment_tool_uses_observed()
            session.increment_tools_captured()

            assert session.stats.fragments_encoded == 2
            assert session.stats.queries_processed == 1
            assert session.stats.tool_uses_observed == 3
            assert session.stats.tools_captured == 1

    def test_stats_to_dict(self):
        """SessionStats.to_dict works correctly."""
        from reconstructions.claude_code.context_manager import SessionStats

        stats = SessionStats()
        stats.fragments_encoded = 5
        stats.queries_processed = 3

        d = stats.to_dict()

        assert d["fragments_encoded"] == 5
        assert d["queries_processed"] == 3
        assert "duration_seconds" in d
        assert "started_at" in d

    def test_project_path_stored(self):
        """Project path is stored in context."""
        from reconstructions.claude_code import get_session

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            session = get_session()
            session.start_session(db_path=db_path, project_path=tmpdir)

            assert session.project_path == Path(tmpdir)
            assert session.context.state["project"] == tmpdir
