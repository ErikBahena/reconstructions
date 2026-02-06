"""
Tool event capture for Claude Code integration.

Converts tool use events into Experience objects for encoding,
with intelligent filtering to avoid noise.
"""

from typing import Any, Optional

from ..encoding import Experience


# Tools that are typically noise (read-only, high-frequency)
NOISE_TOOLS = frozenset({
    "Read",
    "Glob",
    "Grep",
    "LS",
    "Cat",
    "Head",
    "Tail",
    "Find",
    "TaskList",
    "TaskGet",
    "mcp__ide__getDiagnostics",
})

# Tools that represent significant actions worth capturing
SIGNIFICANT_TOOLS = frozenset({
    "Write",
    "Edit",
    "Bash",
    "WebFetch",
    "WebSearch",
    "NotebookEdit",
    "Task",
    "TaskCreate",
    "TaskUpdate",
})

# Emotional mappings for different tool outcomes
# Wider range to create meaningful salience differentiation
TOOL_EMOTION_MAP = {
    "Write": {"valence": 0.7, "arousal": 0.7, "dominance": 0.7},
    "Edit": {"valence": 0.6, "arousal": 0.5, "dominance": 0.6},
    "Bash": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
    "WebFetch": {"valence": 0.5, "arousal": 0.4, "dominance": 0.4},
    "WebSearch": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
    "NotebookEdit": {"valence": 0.6, "arousal": 0.5, "dominance": 0.6},
}

# Success/failure emotional modifiers
SUCCESS_MODIFIER = {"valence": 0.1, "arousal": -0.1, "dominance": 0.1}
FAILURE_MODIFIER = {"valence": -0.3, "arousal": 0.3, "dominance": -0.2}

# Processing depth by source type (used by hooks.py)
PROCESSING_DEPTH = {
    "user_input": 0.8,         # Deliberate, high-level intent
    "tool_failure": 0.7,       # Errors are informative
    "Write": 0.6,              # Significant creation
    "Edit": 0.6,               # Significant modification
    "NotebookEdit": 0.6,       # Significant modification
    "Bash": 0.4,               # Routine execution
    "WebFetch": 0.5,           # Information gathering
    "WebSearch": 0.5,          # Information gathering
    "Task": 0.5,               # Delegation
    "TaskCreate": 0.4,         # Task management
    "TaskUpdate": 0.4,         # Task management
    "assistant_response": 0.5, # Claude's output
    "subagent_output": 0.5,    # Agent results
    "session_end": 0.3,        # Metadata
}


def should_capture_tool(event: dict[str, Any]) -> bool:
    """
    Determine if a tool use event should be captured.

    Filters out noise tools and high-frequency read operations.

    Args:
        event: Tool use event from Claude Code hook.
               Expected keys: tool_name, tool_input, tool_output, success

    Returns:
        True if the tool should be captured, False otherwise
    """
    tool_name = event.get("tool_name", "")

    # Skip noise tools
    if tool_name in NOISE_TOOLS:
        return False

    # Capture significant tools
    if tool_name in SIGNIFICANT_TOOLS:
        return True

    # Default: skip unknown tools (conservative approach)
    return False


def capture_tool_experience(
    event: dict[str, Any],
    project_path: Optional[str] = None
) -> Optional[Experience]:
    """
    Convert a tool use event into an Experience.

    Args:
        event: Tool use event from Claude Code hook.
               Expected keys: tool_name, tool_input, tool_output, success
        project_path: Optional project path for context

    Returns:
        Experience object if capture is appropriate, None otherwise
    """
    if not should_capture_tool(event):
        return None

    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    tool_output = event.get("tool_output", "")
    success = event.get("success", True)

    # Build text description
    text = _build_experience_text(tool_name, tool_input, tool_output, success)
    if not text:
        return None

    # Determine emotional state
    emotional = _compute_emotional_state(tool_name, success)

    # Build tags
    tags = _build_tags(tool_name, tool_input, success, project_path)

    return Experience(
        text=text,
        emotional=emotional,
        source="claude_code",
        tags=tags
    )


def _build_experience_text(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    success: bool
) -> Optional[str]:
    """Build descriptive text for an experience."""

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "unknown")
        content_preview = _truncate(tool_input.get("content", ""), 200)
        status = "created" if success else "failed to create"
        return f"File {status}: {file_path}\nContent preview: {content_preview}"

    elif tool_name == "Edit":
        file_path = tool_input.get("file_path", "unknown")
        old_str = _truncate(tool_input.get("old_string", ""), 100)
        new_str = _truncate(tool_input.get("new_string", ""), 100)
        status = "edited" if success else "failed to edit"
        return f"File {status}: {file_path}\nChanged: '{old_str}' -> '{new_str}'"

    elif tool_name == "Bash":
        command = _truncate(tool_input.get("command", ""), 200)
        output_preview = _truncate(str(tool_output), 300)
        status = "succeeded" if success else "failed"
        return f"Command {status}: {command}\nOutput: {output_preview}"

    elif tool_name == "WebFetch":
        url = tool_input.get("url", "unknown")
        prompt = tool_input.get("prompt", "")
        return f"Fetched web content from: {url}\nQuery: {prompt}"

    elif tool_name == "WebSearch":
        query = tool_input.get("query", "")
        return f"Web search: {query}"

    elif tool_name == "NotebookEdit":
        notebook_path = tool_input.get("notebook_path", "unknown")
        edit_mode = tool_input.get("edit_mode", "replace")
        return f"Notebook {edit_mode}: {notebook_path}"

    elif tool_name == "Task":
        description = tool_input.get("description", "")
        subagent = tool_input.get("subagent_type", "general")
        return f"Spawned {subagent} agent: {description}"

    elif tool_name == "TaskCreate":
        subject = tool_input.get("subject", "")
        return f"Created task: {subject}"

    elif tool_name == "TaskUpdate":
        task_id = tool_input.get("taskId", "")
        status = tool_input.get("status", "")
        return f"Updated task {task_id}: status={status}" if status else f"Updated task {task_id}"

    return None


def _compute_emotional_state(tool_name: str, success: bool) -> dict[str, float]:
    """Compute emotional state for a tool use."""
    # Start with base emotion for tool type
    base = TOOL_EMOTION_MAP.get(tool_name, {"valence": 0.5, "arousal": 0.5, "dominance": 0.5})

    # Apply success/failure modifier
    modifier = SUCCESS_MODIFIER if success else FAILURE_MODIFIER

    return {
        "valence": _clamp(base["valence"] + modifier["valence"]),
        "arousal": _clamp(base["arousal"] + modifier["arousal"]),
        "dominance": _clamp(base["dominance"] + modifier["dominance"])
    }


def _build_tags(
    tool_name: str,
    tool_input: dict[str, Any],
    success: bool,
    project_path: Optional[str]
) -> list[str]:
    """Build tags for an experience."""
    tags = [f"tool:{tool_name.lower()}"]

    if not success:
        tags.append("outcome:failure")
    else:
        tags.append("outcome:success")

    # Add file-related tags
    file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if file_path:
        # Extract extension
        if "." in str(file_path):
            ext = str(file_path).rsplit(".", 1)[-1].lower()
            if ext in ("py", "js", "ts", "tsx", "jsx", "rs", "go", "java", "rb", "md"):
                tags.append(f"lang:{ext}")

    # Add project tag if available
    if project_path:
        project_name = str(project_path).rsplit("/", 1)[-1]
        tags.append(f"project:{project_name}")

    return tags


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, value))
