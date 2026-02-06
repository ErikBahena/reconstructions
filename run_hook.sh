#!/bin/bash
# Wrapper script for Claude Code hooks that ensures correct Python path

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add the src directory to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the hook with all arguments passed through
python3 -m reconstructions.claude_code.hooks "$@"
