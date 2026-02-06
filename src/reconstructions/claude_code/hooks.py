"""
Claude Code hooks for the memory system.

Implements hook handlers for SessionStart, PostToolUse, Stop, and SessionEnd.
Can be invoked via CLI: python -m reconstructions.claude_code.hooks <hook_name>
"""

import sys
import json
import os
from typing import Any, Optional

from .context_manager import get_session, SessionContext
from .capture import should_capture_tool, capture_tool_experience, PROCESSING_DEPTH
from ..encoding import Experience
from ..core import Query
from ..encoder import encode


def _maybe_consolidate() -> Optional[dict[str, Any]]:
    """
    Check if consolidation should run and execute it if needed.

    This implements opportunistic consolidation - during normal hook processing,
    we check if the consolidation interval has elapsed and run consolidation
    in the same process.

    Returns:
        Consolidation stats if ran, None otherwise
    """
    session = get_session()

    if not session.is_active or not session.engine:
        return None

    # Check if consolidation scheduler exists and is ready
    scheduler = session.engine.consolidation_scheduler
    if not scheduler or not scheduler.should_consolidate():
        return None

    try:
        # Run consolidation
        stats = scheduler.consolidate()
        return {
            "consolidation_ran": True,
            "rehearsed": stats.get("rehearsed_count", 0),
            "bindings_strengthened": stats.get("bindings_strengthened", 0),
            "patterns_discovered": stats.get("patterns_discovered", 0),
            "duration_ms": stats.get("duration_ms", 0)
        }
    except Exception as e:
        # Don't let consolidation errors break the hook
        return {
            "consolidation_ran": False,
            "error": str(e)
        }


def on_session_start(
    project_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Handle session start hook.

    Initializes the memory system for a new Claude Code session.

    Args:
        project_path: Path to current project
        db_path: Optional database path override

    Returns:
        Status dict with session info
    """
    session = get_session()

    if session.is_active:
        return {
            "status": "already_active",
            "session_id": session.context.id if session.context else None
        }

    success = session.start_session(db_path=db_path, project_path=project_path)

    if not success:
        return {"status": "failed", "error": "Could not start session"}

    # Optionally recall project context
    result = {
        "status": "started",
        "session_id": session.context.id if session.context else None,
        "project": str(session.project_path) if session.project_path else None,
        "db_path": str(session._db_path) if session._db_path else None
    }

    # Check if we have existing memories for this project
    if session.store and session.engine and session.project_path:
        project_name = session.project_path.name

        # Query for project-related memories
        query = Query(semantic=f"project {project_name}")
        session.engine.submit_query(query)
        query_result = session.engine.step()

        if query_result and query_result.success:
            strand = query_result.data.get("strand")
            if strand and strand.fragments:
                result["recalled_fragments"] = len(strand.fragments)

    return result


def on_post_tool_use(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle post-tool-use hook.

    Evaluates tool use events and encodes significant ones as memories.

    Args:
        event: Tool use event with keys:
            - tool_name: Name of the tool
            - tool_input: Tool input parameters
            - tool_output: Tool output/result
            - success: Whether the tool succeeded

    Returns:
        Status dict indicating if the event was captured
    """
    session = get_session()

    if not session.is_active:
        # Lazily start session if not active
        session.start_session()

    session.increment_tool_uses_observed()

    # Check if this tool should be captured
    if not should_capture_tool(event):
        return {"status": "skipped", "reason": "noise_tool"}

    # Convert to experience
    experience = capture_tool_experience(
        event,
        project_path=str(session.project_path) if session.project_path else None
    )

    if experience is None:
        return {"status": "skipped", "reason": "no_experience"}

    # Encode using the session's context (persisted between hooks)
    if session.store and session.context:
        try:
            # Set source-specific processing depth
            tool_name = event.get("tool_name", "")
            session.context.processing_depth = PROCESSING_DEPTH.get(tool_name, 0.5)

            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False  # Too expensive per-call
            )

            session.increment_fragments_encoded()
            session.increment_tools_captured()

            # Persist state for next hook call (saves recent_fragments)
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {
                "status": "captured",
                "fragment_id": fragment.id,
                "tool": event.get("tool_name")
            }

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    return {"status": "failed", "reason": "no_store_or_context"}


def on_stop(event: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Handle stop hook (when Claude finishes responding).

    Reads Claude's response from the transcript file since the Stop event
    doesn't include response content directly.

    Args:
        event: Stop event containing:
            - transcript_path: Path to conversation JSONL file
            - stop_hook_active: bool
            - session_id: str

    Returns:
        Status dict
    """
    session = get_session()

    if not session.is_active:
        return {"status": "no_session"}

    if event is None:
        event = {}

    # Get Claude's response from the transcript file
    response_text = None
    transcript_path = event.get("transcript_path")

    if transcript_path:
        try:
            response_text = _extract_last_assistant_response(transcript_path)
        except Exception:
            pass

    if response_text and session.store and session.context:
        # Truncate long responses
        truncated = response_text[:1000] if len(response_text) > 1000 else response_text

        session.context.processing_depth = PROCESSING_DEPTH.get("assistant_response", 0.5)

        # Encode Claude's response as a memory
        experience = Experience(
            text=f"Claude responded: {truncated}",
            emotional={"valence": 0.6, "arousal": 0.3, "dominance": 0.6},
            source="assistant_response",
            tags=["output:assistant", "conversation"]
        )

        try:
            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session.increment_fragments_encoded()
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {
                "status": "captured",
                "fragment_id": fragment.id
            }

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception:
            pass

    return {"status": "ok"}


def on_user_prompt_submit(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle user prompt submit hook.

    Captures user messages as memories for later recall.

    Args:
        event: Event with keys:
            - prompt: The user's message text
            - session_id: Session identifier

    Returns:
        Status dict indicating if the prompt was captured
    """
    session = get_session()

    if not session.is_active:
        session.start_session()

    prompt = event.get("prompt", "").strip()

    # Skip empty or very short prompts
    if not prompt or len(prompt) < 10:
        return {"status": "skipped", "reason": "too_short"}

    # Skip command-like prompts (starting with /)
    if prompt.startswith("/"):
        return {"status": "skipped", "reason": "command"}

    # Encode the user prompt as a memory
    if session.store and session.context:
        session.context.processing_depth = PROCESSING_DEPTH.get("user_input", 0.8)

        experience = Experience(
            text=f"User said: {prompt}",
            emotional={"valence": 0.7, "arousal": 0.6, "dominance": 0.6},
            source="user_input",
            tags=["input:user", "conversation"]
        )

        try:
            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session.increment_fragments_encoded()
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {
                "status": "captured",
                "fragment_id": fragment.id
            }

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    return {"status": "failed", "reason": "no_store_or_context"}


def on_post_tool_use_failure(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle post-tool-use-failure hook.

    Captures failed tool executions - errors are important context!

    Args:
        event: Tool use event with failure details

    Returns:
        Status dict
    """
    session = get_session()

    if not session.is_active:
        session.start_session()

    session.increment_tool_uses_observed()

    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    error = event.get("error", event.get("tool_output", "Unknown error"))

    # Build error experience text
    if tool_name == "Bash":
        command = tool_input.get("command", "")[:200]
        text = f"Command failed: {command}\nError: {str(error)[:300]}"
    elif tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "unknown")
        text = f"File operation failed: {tool_name} {file_path}\nError: {str(error)[:300]}"
    else:
        text = f"Tool failed: {tool_name}\nError: {str(error)[:300]}"

    # Failures have higher arousal (frustration/attention)
    experience = Experience(
        text=text,
        emotional={"valence": 0.2, "arousal": 0.8, "dominance": 0.3},
        source="claude_code",
        tags=[f"tool:{tool_name.lower()}", "outcome:failure", "error"]
    )

    if session.store and session.context:
        session.context.processing_depth = PROCESSING_DEPTH.get("tool_failure", 0.7)
        try:
            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session.increment_fragments_encoded()
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {
                "status": "captured",
                "fragment_id": fragment.id,
                "tool": tool_name
            }

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    return {"status": "failed", "reason": "no_store_or_context"}


def on_subagent_start(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle subagent-start hook.

    Captures when subagents are spawned for complex tasks.

    Args:
        event: Subagent start event

    Returns:
        Status dict
    """
    session = get_session()

    if not session.is_active:
        session.start_session()

    agent_type = event.get("agent_type", event.get("subagent_type", "unknown"))
    description = event.get("description", event.get("prompt", ""))[:200]

    text = f"Spawned {agent_type} agent: {description}"

    experience = Experience(
        text=text,
        emotional={"valence": 0.5, "arousal": 0.5, "dominance": 0.6},
        source="claude_code",
        tags=[f"agent:{agent_type.lower()}", "agent:start"]
    )

    if session.store and session.context:
        session.context.processing_depth = PROCESSING_DEPTH.get("Task", 0.5)
        try:
            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session.increment_fragments_encoded()
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {"status": "captured", "fragment_id": fragment.id}

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception:
            pass

    return {"status": "ok"}


def on_subagent_stop(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle subagent-stop hook.

    Reads the agent's transcript to capture its full output.

    Args:
        event: Subagent stop event containing:
            - agent_type: Type of agent (Explore, Plan, Bash, etc.)
            - agent_id: Unique agent identifier
            - agent_transcript_path: Path to agent's conversation JSONL

    Returns:
        Status dict
    """
    session = get_session()

    if not session.is_active:
        session.start_session()

    agent_type = event.get("agent_type", event.get("subagent_type", "unknown"))
    agent_id = event.get("agent_id", "")

    # Try to get agent's output from its transcript
    agent_output = None
    agent_transcript_path = event.get("agent_transcript_path")

    if agent_transcript_path:
        try:
            agent_output = _extract_last_assistant_response(agent_transcript_path)
        except Exception:
            pass

    # Truncate long outputs
    if agent_output:
        truncated = agent_output[:800] if len(agent_output) > 800 else agent_output
        text = f"Agent {agent_type} completed:\n{truncated}"
    else:
        text = f"Agent {agent_type} completed (no output captured)"

    experience = Experience(
        text=text,
        emotional={"valence": 0.6, "arousal": 0.3, "dominance": 0.5},
        source="subagent_output",
        tags=[f"agent:{agent_type.lower()}", "agent:stop", "output:agent"]
    )

    if session.store and session.context:
        session.context.processing_depth = PROCESSING_DEPTH.get("subagent_output", 0.5)
        try:
            fragment = encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session.increment_fragments_encoded()
            session._save_persisted_state()

            # Opportunistic consolidation check
            consolidation_stats = _maybe_consolidate()

            result = {"status": "captured", "fragment_id": fragment.id}

            if consolidation_stats:
                result["consolidation"] = consolidation_stats

            return result
        except Exception:
            pass

    return {"status": "ok"}


def on_session_end() -> dict[str, Any]:
    """
    Handle session end hook.

    Encodes a session summary and cleans up resources.

    Returns:
        Final session statistics
    """
    session = get_session()

    if not session.is_active:
        return {"status": "no_session"}

    # Encode session summary
    if session.store and session.context and session.stats:
        session.context.processing_depth = PROCESSING_DEPTH.get("session_end", 0.3)
        stats = session.stats.to_dict()
        summary = (
            f"Session ended. Duration: {stats['duration_seconds']:.0f}s. "
            f"Encoded {stats['fragments_encoded']} memories from "
            f"{stats['tools_captured']}/{stats['tool_uses_observed']} tool uses."
        )

        experience = Experience(
            text=summary,
            emotional={"valence": 0.6, "arousal": 0.2, "dominance": 0.5},
            source="claude_code",
            tags=["session:end"]
        )

        try:
            encode(
                experience=experience,
                context=session.context,
                store=session.store,
                create_semantic_bindings=False
            )
            session._save_persisted_state()
        except Exception:
            pass

    # Run final consolidation before ending session (like sleep consolidation)
    consolidation_stats = _maybe_consolidate()

    # Force consolidation even if interval hasn't elapsed (session ending)
    if not consolidation_stats and session.engine and session.engine.consolidation_scheduler:
        try:
            consolidation_stats = session.engine.consolidation_scheduler.consolidate()
        except Exception:
            consolidation_stats = None

    # End session and get final stats
    final_stats = session.end_session()

    if final_stats:
        result = {
            "status": "ended",
            **final_stats.to_dict()
        }
        if consolidation_stats:
            result["final_consolidation"] = consolidation_stats
        return result

    return {"status": "ended"}


def _extract_last_assistant_response(transcript_path: str) -> Optional[str]:
    """
    Extract the last assistant response from a transcript JSONL file.

    The transcript is a JSONL file where each line is a conversation event.
    We look for the last assistant message.

    Args:
        transcript_path: Path to the transcript JSONL file

    Returns:
        The assistant's response text, or None if not found
    """
    try:
        with open(transcript_path, "r") as f:
            lines = f.readlines()

        # Process lines in reverse to find the last assistant message
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)

                # Handle different transcript formats
                # Format 1: Direct message with role
                if entry.get("role") == "assistant":
                    content = entry.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Content blocks - extract text
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                texts.append(block)
                        if texts:
                            return "\n".join(texts)

                # Format 2: Message wrapper
                if entry.get("type") == "message" and entry.get("message", {}).get("role") == "assistant":
                    content = entry["message"].get("content", "")
                    if isinstance(content, str):
                        return content

                # Format 3: Assistant turn
                if entry.get("type") == "assistant" or entry.get("sender") == "assistant":
                    return entry.get("text") or entry.get("content") or entry.get("message")

            except json.JSONDecodeError:
                continue

    except (FileNotFoundError, IOError):
        pass

    return None


def _read_stdin_event() -> dict[str, Any]:
    """Read a JSON event from stdin."""
    try:
        data = sys.stdin.read()
        if data.strip():
            return json.loads(data)
    except json.JSONDecodeError:
        pass
    return {}


def main():
    """CLI entry point for hooks."""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No hook specified"}))
        sys.exit(1)

    hook_name = sys.argv[1]

    # Get project path from environment
    project_path = os.environ.get("PWD", os.getcwd())

    if hook_name == "session_start":
        result = on_session_start(project_path=project_path)

    elif hook_name == "post_tool_use":
        event = _read_stdin_event()
        result = on_post_tool_use(event)

    elif hook_name == "stop":
        event = _read_stdin_event()
        result = on_stop(event=event)

    elif hook_name == "session_end":
        result = on_session_end()

    elif hook_name == "user_prompt_submit":
        event = _read_stdin_event()
        result = on_user_prompt_submit(event)

    elif hook_name == "post_tool_use_failure":
        event = _read_stdin_event()
        result = on_post_tool_use_failure(event)

    elif hook_name == "subagent_start":
        event = _read_stdin_event()
        result = on_subagent_start(event)

    elif hook_name == "subagent_stop":
        event = _read_stdin_event()
        result = on_subagent_stop(event)

    else:
        result = {"error": f"Unknown hook: {hook_name}"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
