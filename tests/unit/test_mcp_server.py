"""Tests for MCP server tools."""

import pytest
import tempfile
from pathlib import Path


class TestMCPTools:
    """Tests for MCP tool functions."""

    @pytest.fixture
    def memory_server(self):
        """Create a memory server instance."""
        from reconstructions.mcp_server import MemoryServer

        with tempfile.TemporaryDirectory() as tmpdir:
            server = MemoryServer(db_path=Path(tmpdir) / "test.db")
            yield server
            server.close()

    def test_memory_store(self, memory_server):
        """Can store a memory."""
        result = memory_server.memory_store(
            text="The user prefers dark mode",
            emotional_valence=0.6,
            emotional_arousal=0.3
        )

        assert result["success"] is True
        assert "fragment_id" in result
        assert result["salience"] > 0

    def test_memory_recall(self, memory_server):
        """Can recall stored memories."""
        # Store first
        memory_server.memory_store(text="I love pizza")
        memory_server.memory_store(text="Python is my favorite language")

        # Recall
        result = memory_server.memory_recall(query="favorite food")

        assert "fragments" in result
        assert "certainty" in result

    def test_memory_identity(self, memory_server):
        """Can get identity model."""
        result = memory_server.memory_identity()

        assert "traits" in result
        assert "beliefs" in result
        assert "goals" in result

    def test_memory_status(self, memory_server):
        """Can get memory status."""
        memory_server.memory_store(text="test memory")

        result = memory_server.memory_status()

        assert result["fragment_count"] >= 1
        assert "health" in result

    def test_store_and_recall_roundtrip(self, memory_server):
        """Stored memories can be recalled."""
        # Store specific memory
        memory_server.memory_store(
            text="Meeting with Alice at 3pm about the project",
            tags=["meeting", "alice"]
        )

        # Recall
        result = memory_server.memory_recall(query="meeting with Alice")

        assert len(result["fragments"]) >= 1
        # Check the recalled fragment contains relevant content
        fragment_ids = [f["id"] for f in result["fragments"]]
        assert len(fragment_ids) > 0
