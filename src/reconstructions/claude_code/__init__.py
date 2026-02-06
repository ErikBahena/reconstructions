"""
Claude Code integration for the Reconstructions memory system.

Provides hooks and skills for native Claude Code integration.
"""

from .context_manager import get_session, SessionContext, SessionStats
from .capture import should_capture_tool, capture_tool_experience
from .hooks import (
    on_session_start,
    on_user_prompt_submit,
    on_post_tool_use,
    on_stop,
    on_session_end
)
from .skills import execute_skill, format_output

__all__ = [
    # Context management
    "get_session",
    "SessionContext",
    "SessionStats",
    # Capture
    "should_capture_tool",
    "capture_tool_experience",
    # Hooks
    "on_session_start",
    "on_user_prompt_submit",
    "on_post_tool_use",
    "on_stop",
    "on_session_end",
    # Skills
    "execute_skill",
    "format_output",
]
