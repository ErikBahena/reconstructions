"""
Plugin manifest and loader for Claude Code integration.

Provides metadata about the reconstructions plugin for Claude Code discovery.
"""

from pathlib import Path
from typing import Any

# Plugin metadata
PLUGIN_NAME = "reconstructions"
PLUGIN_VERSION = "0.1.0"
PLUGIN_DESCRIPTION = "Reconstructive memory system for Claude Code"


def get_plugin_manifest() -> dict[str, Any]:
    """
    Get the plugin manifest for Claude Code.

    Returns:
        Plugin manifest dict with hooks and skills info
    """
    return {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "description": PLUGIN_DESCRIPTION,
        "hooks": {
            "SessionStart": {
                "handler": "reconstructions.claude_code.hooks:on_session_start",
                "description": "Initialize memory system for session"
            },
            "PostToolUse": {
                "handler": "reconstructions.claude_code.hooks:on_post_tool_use",
                "description": "Capture significant tool uses as memories"
            },
            "Stop": {
                "handler": "reconstructions.claude_code.hooks:on_stop",
                "description": "Encode task completions"
            },
            "SessionEnd": {
                "handler": "reconstructions.claude_code.hooks:on_session_end",
                "description": "Finalize and cleanup memory session"
            }
        },
        "skills": {
            "memory": {
                "handler": "reconstructions.claude_code.skills:execute_skill",
                "description": "Store and recall from reconstructive memory",
                "operations": ["store", "recall", "status", "identity", "recent", "forget"]
            }
        }
    }


def get_hook_command(hook_name: str, python_path: str = "python") -> str:
    """
    Get the shell command for a hook.

    Args:
        hook_name: Name of the hook (SessionStart, PostToolUse, etc.)
        python_path: Path to Python interpreter (use absolute path for reliability)

    Returns:
        Shell command to execute the hook
    """
    hook_map = {
        "SessionStart": "session_start",
        "PostToolUse": "post_tool_use",
        "Stop": "stop",
        "SessionEnd": "session_end"
    }

    internal_name = hook_map.get(hook_name, hook_name.lower())
    return f"{python_path} -m reconstructions.claude_code.hooks {internal_name}"


def get_skill_command(skill_name: str, operation: str, args: str = "") -> str:
    """
    Get the shell command for a skill operation.

    Args:
        skill_name: Name of the skill (memory)
        operation: Operation to perform
        args: Optional arguments

    Returns:
        Shell command to execute the skill
    """
    if skill_name != "memory":
        raise ValueError(f"Unknown skill: {skill_name}")

    # Use proper dispatcher to avoid RuntimeWarning
    cmd = f"python -m reconstructions.claude_code skills {operation}"
    if args:
        cmd += f" {args}"
    return cmd


def generate_settings_json(output_path: Path, python_path: str = "python") -> None:
    """
    Generate .claude/settings.json for hook registration.

    Args:
        output_path: Path to write settings.json
        python_path: Path to Python interpreter (use absolute path for reliability)
    """
    import json

    settings = {
        "hooks": {
            "SessionStart": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": get_hook_command("SessionStart", python_path)
                        }
                    ]
                }
            ],
            "PostToolUse": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": get_hook_command("PostToolUse", python_path)
                        }
                    ]
                }
            ],
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": get_hook_command("Stop", python_path)
                        }
                    ]
                }
            ],
            "SessionEnd": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": get_hook_command("SessionEnd", python_path)
                        }
                    ]
                }
            ]
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(settings, f, indent=2)


def generate_skill_md(output_path: Path) -> None:
    """
    Generate .claude/skills/memory/SKILL.md for skill registration.

    Args:
        output_path: Path to write SKILL.md
    """
    skill_content = """---
name: memory
description: Store and recall from reconstructive memory system
argument-hint: "store|recall|status|identity|recent|forget [args]"
---

# Memory Skill

Interact with the reconstructive memory system.

## Operations

### store <text>
Store a new memory manually.

Example: `/memory store Important meeting notes about project X`

### recall <query>
Recall memories matching a semantic query.

Example: `/memory recall project X meetings`

### status
Show memory system status including fragment count and session stats.

Example: `/memory status`

### identity
View current identity state (traits, beliefs, goals).

Example: `/memory identity`

### recent [n]
List the n most recent memories (default 10).

Example: `/memory recent 5`

### forget <id>
Accelerate decay for a memory (soft forget).

Example: `/memory forget abc123`
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(skill_content)
