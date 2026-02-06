"""
Session context manager for Claude Code integration.

Manages the lifecycle of the memory system within a Claude Code session,
providing a singleton interface for hooks and skills to interact with.

Since hooks are invoked as separate processes, this module persists
critical context state (recent_fragments) to disk between calls.
"""

import os
import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from ..store import FragmentStore
from ..engine import ReconstructionEngine
from ..certainty import VarianceController
from ..encoding import Context
from ..health import MemoryHealthMonitor


# State file for persisting context between process invocations
STATE_FILENAME = "session_state.json"
MAX_RECENT_FRAGMENTS = 20  # How many recent fragments to track for bindings


@dataclass
class SessionStats:
    """Statistics for the current session."""

    started_at: float = field(default_factory=time.time)
    fragments_encoded: int = 0
    queries_processed: int = 0
    tool_uses_observed: int = 0
    tools_captured: int = 0

    def to_dict(self) -> dict:
        """Serialize stats to dictionary."""
        return {
            "started_at": self.started_at,
            "duration_seconds": time.time() - self.started_at,
            "fragments_encoded": self.fragments_encoded,
            "queries_processed": self.queries_processed,
            "tool_uses_observed": self.tool_uses_observed,
            "tools_captured": self.tools_captured
        }


class SessionContext:
    """
    Singleton managing session state for Claude Code integration.

    Holds the ReconstructionEngine, FragmentStore, and tracks session
    statistics. Thread-safe for concurrent hook access.
    """

    _instance: Optional["SessionContext"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._active = False
        self._store: Optional[FragmentStore] = None
        self._engine: Optional[ReconstructionEngine] = None
        self._context: Optional[Context] = None
        self._variance_controller: Optional[VarianceController] = None
        self._health_monitor: Optional[MemoryHealthMonitor] = None
        self._stats: Optional[SessionStats] = None
        self._db_path: Optional[Path] = None
        self._project_path: Optional[Path] = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None and cls._instance._active:
                cls._instance.end_session()
            cls._instance = None

    @property
    def is_active(self) -> bool:
        """Check if a session is currently active."""
        return self._active

    @property
    def store(self) -> Optional[FragmentStore]:
        """Get the fragment store."""
        return self._store

    @property
    def engine(self) -> Optional[ReconstructionEngine]:
        """Get the reconstruction engine."""
        return self._engine

    @property
    def context(self) -> Optional[Context]:
        """Get the current encoding context."""
        return self._context

    @property
    def variance_controller(self) -> Optional[VarianceController]:
        """Get the variance controller."""
        return self._variance_controller

    @property
    def health_monitor(self) -> Optional[MemoryHealthMonitor]:
        """Get the health monitor."""
        return self._health_monitor

    @property
    def stats(self) -> Optional[SessionStats]:
        """Get session statistics."""
        return self._stats

    @property
    def project_path(self) -> Optional[Path]:
        """Get the current project path."""
        return self._project_path

    def _state_file_path(self) -> Path:
        """Get path to the state persistence file."""
        if self._db_path:
            return self._db_path.parent / STATE_FILENAME
        return Path.home() / ".reconstructions" / STATE_FILENAME

    def _load_persisted_state(self) -> None:
        """Load persisted context state from disk."""
        state_path = self._state_file_path()
        if not state_path.exists():
            return

        try:
            with open(state_path) as f:
                state = json.load(f)

            # Restore recent fragments to context for temporal bindings
            if self._context and "recent_fragments" in state:
                self._context.recent_fragments = state["recent_fragments"][-MAX_RECENT_FRAGMENTS:]

            # Restore sequence counter
            if self._context and "sequence_counter" in state:
                self._context.sequence_counter = state["sequence_counter"]

        except (json.JSONDecodeError, OSError):
            pass  # Start fresh if state is corrupted

    def _save_persisted_state(self) -> None:
        """Save context state to disk for next hook invocation."""
        if not self._context:
            return

        state_path = self._state_file_path()

        try:
            state = {
                "recent_fragments": self._context.recent_fragments[-MAX_RECENT_FRAGMENTS:],
                "sequence_counter": self._context.sequence_counter,
                "updated_at": time.time()
            }

            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w") as f:
                json.dump(state, f)

        except OSError:
            pass  # Non-fatal, just lose bindings

    def start_session(
        self,
        db_path: Optional[str] = None,
        project_path: Optional[str] = None
    ) -> bool:
        """
        Start a new session.

        Args:
            db_path: Path to database file. Defaults to ~/.reconstructions/memory.db
            project_path: Path to current project. Defaults to current working directory.

        Returns:
            True if session started successfully, False if already active
        """
        if self._active:
            return False

        # Determine paths
        if db_path is None:
            db_path = os.environ.get(
                "RECONSTRUCTIONS_DB",
                str(Path.home() / ".reconstructions" / "memory.db")
            )

        if project_path is None:
            project_path = os.environ.get("PWD", os.getcwd())

        self._db_path = Path(db_path)
        self._project_path = Path(project_path)

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._store = FragmentStore(str(self._db_path))
        self._context = Context()
        self._variance_controller = VarianceController()

        # Initialize health monitor first (needed by engine)
        self._health_monitor = MemoryHealthMonitor(self._store, self._db_path.parent)

        # Create engine with health monitoring enabled
        self._engine = ReconstructionEngine(
            self._store,
            enable_consolidation=True,  # Enable autonomous consolidation
            health_monitor=self._health_monitor  # Pass health monitor for metrics tracking
        )

        self._stats = SessionStats()

        # Restore persisted state (recent_fragments for bindings)
        self._load_persisted_state()

        # Set project context
        self._context.state["project"] = str(self._project_path)
        self._context.state["session_id"] = self._context.id

        self._active = True
        return True

    def end_session(self) -> Optional[SessionStats]:
        """
        End the current session.

        Returns:
            Final session statistics, or None if no session was active
        """
        if not self._active:
            return None

        final_stats = self._stats

        # Close store (saves vector index)
        if self._store is not None:
            self._store.close()

        # Reset state
        self._store = None
        self._engine = None
        self._context = None
        self._variance_controller = None
        self._health_monitor = None
        self._stats = None
        self._active = False

        return final_stats

    def increment_fragments_encoded(self) -> None:
        """Increment the fragments encoded counter."""
        if self._stats:
            self._stats.fragments_encoded += 1

    def increment_queries_processed(self) -> None:
        """Increment the queries processed counter."""
        if self._stats:
            self._stats.queries_processed += 1

    def increment_tool_uses_observed(self) -> None:
        """Increment the tool uses observed counter."""
        if self._stats:
            self._stats.tool_uses_observed += 1

    def increment_tools_captured(self) -> None:
        """Increment the tools captured counter."""
        if self._stats:
            self._stats.tools_captured += 1


def get_session() -> SessionContext:
    """Get the global session context singleton."""
    return SessionContext()
