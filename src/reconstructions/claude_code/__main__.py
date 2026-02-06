"""
CLI entry point for Claude Code hooks.

Usage:
    python -m reconstructions.claude_code hooks <hook_name>
    python -m reconstructions.claude_code skills <operation> [args]
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m reconstructions.claude_code <command> [args]")
        print("Commands: hooks, skills")
        sys.exit(1)

    command = sys.argv[1]

    if command == "hooks":
        from .hooks import main as hooks_main
        sys.argv = sys.argv[1:]  # Remove 'hooks' from argv
        hooks_main()

    elif command == "skills":
        from .skills import main as skills_main
        sys.argv = sys.argv[1:]  # Remove 'skills' from argv
        skills_main()

    else:
        print(f"Unknown command: {command}")
        print("Commands: hooks, skills")
        sys.exit(1)


if __name__ == "__main__":
    main()
