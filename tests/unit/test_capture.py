"""Tests for tool capture functionality."""

import pytest


class TestShouldCaptureTool:
    """Tests for should_capture_tool function."""

    def test_skips_read_tools(self):
        """Read-only tools are skipped."""
        from reconstructions.claude_code.capture import should_capture_tool

        for tool in ["Read", "Glob", "Grep", "LS", "Cat", "Head", "Tail", "Find"]:
            event = {"tool_name": tool}
            assert should_capture_tool(event) is False

    def test_captures_write_tools(self):
        """Write operations are captured."""
        from reconstructions.claude_code.capture import should_capture_tool

        for tool in ["Write", "Edit", "Bash", "NotebookEdit"]:
            event = {"tool_name": tool}
            assert should_capture_tool(event) is True

    def test_captures_web_tools(self):
        """Web operations are captured."""
        from reconstructions.claude_code.capture import should_capture_tool

        for tool in ["WebFetch", "WebSearch"]:
            event = {"tool_name": tool}
            assert should_capture_tool(event) is True

    def test_skips_unknown_tools(self):
        """Unknown tools are skipped (conservative)."""
        from reconstructions.claude_code.capture import should_capture_tool

        event = {"tool_name": "UnknownTool"}
        assert should_capture_tool(event) is False

    def test_empty_tool_name(self):
        """Empty tool name is skipped."""
        from reconstructions.claude_code.capture import should_capture_tool

        event = {"tool_name": ""}
        assert should_capture_tool(event) is False


class TestCaptureToolExperience:
    """Tests for capture_tool_experience function."""

    def test_returns_none_for_noise_tool(self):
        """Noise tools return None."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {"tool_name": "Read", "tool_input": {"file_path": "/foo"}}
        result = capture_tool_experience(event)

        assert result is None

    def test_captures_write_event(self):
        """Write events are captured."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/foo/bar.py",
                "content": "print('hello world')"
            },
            "tool_output": "File written",
            "success": True
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert "bar.py" in experience.text
        assert "print" in experience.text
        assert experience.source == "claude_code"
        assert "tool:write" in experience.tags
        assert "outcome:success" in experience.tags
        assert "lang:py" in experience.tags

    def test_captures_edit_event(self):
        """Edit events are captured."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/foo/bar.js",
                "old_string": "const x = 1",
                "new_string": "const x = 2"
            },
            "success": True
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert "bar.js" in experience.text
        assert "const x = 1" in experience.text
        assert "const x = 2" in experience.text
        assert "lang:js" in experience.tags

    def test_captures_bash_event(self):
        """Bash events are captured."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "npm install react"},
            "tool_output": "installed 50 packages",
            "success": True
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert "npm install react" in experience.text
        assert "installed" in experience.text

    def test_captures_failed_command(self):
        """Failed commands get failure tag."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "npm test"},
            "tool_output": "1 test failed",
            "success": False
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert "outcome:failure" in experience.tags
        # Failure reduces valence
        assert experience.emotional["valence"] < 0.5

    def test_truncates_long_content(self):
        """Long content is truncated."""
        from reconstructions.claude_code.capture import capture_tool_experience

        long_content = "x" * 1000
        event = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/foo/bar.py",
                "content": long_content
            },
            "success": True
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert len(experience.text) < 500  # Should be truncated

    def test_project_tag_added(self):
        """Project tag is added when path provided."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/foo/bar.py", "content": "test"},
            "success": True
        }

        experience = capture_tool_experience(event, project_path="/path/to/myproject")

        assert "project:myproject" in experience.tags

    def test_web_search_captured(self):
        """Web search events are captured."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "WebSearch",
            "tool_input": {"query": "python asyncio tutorial"},
            "success": True
        }

        experience = capture_tool_experience(event)

        assert experience is not None
        assert "python asyncio tutorial" in experience.text


class TestEmotionalState:
    """Tests for emotional state computation."""

    def test_success_increases_valence(self):
        """Success increases emotional valence."""
        from reconstructions.claude_code.capture import capture_tool_experience

        success_event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/foo.py", "content": "test"},
            "success": True
        }
        fail_event = {
            "tool_name": "Write",
            "tool_input": {"file_path": "/foo.py", "content": "test"},
            "success": False
        }

        success_exp = capture_tool_experience(success_event)
        fail_exp = capture_tool_experience(fail_event)

        assert success_exp.emotional["valence"] > fail_exp.emotional["valence"]

    def test_emotional_values_clamped(self):
        """Emotional values stay in 0-1 range."""
        from reconstructions.claude_code.capture import capture_tool_experience

        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "test"},
            "success": False
        }

        experience = capture_tool_experience(event)

        for key in ["valence", "arousal", "dominance"]:
            assert 0.0 <= experience.emotional[key] <= 1.0
