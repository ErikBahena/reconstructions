"""
MCP Server for Memory Reconstruction System.

Exposes memory operations as MCP-compatible tools for Claude Code.
"""

from pathlib import Path
from typing import Optional

from .core import Query
from .encoding import Experience, Context
from .store import FragmentStore
from .engine import ReconstructionEngine


class MemoryServer:
    """
    Memory server exposing MCP-compatible tools.

    This is the main interface for Claude Code to interact
    with the memory system.
    """

    def __init__(self, db_path: Path):
        """
        Initialize memory server.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.store = FragmentStore(str(db_path))
        self.engine = ReconstructionEngine(self.store)
        self.context = Context()

    def memory_store(
        self,
        text: str,
        emotional_valence: float = 0.5,
        emotional_arousal: float = 0.5,
        tags: Optional[list[str]] = None,
        source: str = "claude_code"
    ) -> dict:
        """
        Store a new memory fragment.

        Args:
            text: Content to remember
            emotional_valence: Positive/negative (0-1)
            emotional_arousal: Intensity (0-1)
            tags: Optional categorization tags
            source: Origin of this memory

        Returns:
            {"success": bool, "fragment_id": str, "salience": float}
        """
        try:
            experience = Experience(
                text=text,
                emotional={
                    "valence": emotional_valence,
                    "arousal": emotional_arousal,
                    "dominance": 0.5
                },
                source=source,
                tags=tags or []
            )

            self.engine.submit_experience(experience)
            result = self.engine.step()

            if result and result.success:
                fragment_id = result.data.get("fragment_id", "")

                # Get salience from the stored fragment
                fragment = self.store.get(fragment_id)
                salience = fragment.initial_salience if fragment else 0.5

                return {
                    "success": True,
                    "fragment_id": fragment_id,
                    "salience": salience
                }

            return {"success": False, "error": "Failed to encode"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def memory_recall(
        self,
        query: str,
        limit: int = 5,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        min_certainty: float = 0.0
    ) -> dict:
        """
        Recall memories matching a query.

        Args:
            query: What to remember (semantic search)
            limit: Max fragments to return
            time_start: Optional start timestamp
            time_end: Optional end timestamp
            min_certainty: Minimum certainty threshold

        Returns:
            {"fragments": [...], "certainty": float, "strand_id": str}
        """
        try:
            time_range = None
            if time_start is not None and time_end is not None:
                time_range = (time_start, time_end)

            q = Query(
                semantic=query,
                time_range=time_range
            )

            self.engine.submit_query(q)
            result = self.engine.step()

            if result and result.success:
                strand = result.data.get("strand")
                certainty = result.data.get("certainty", 0.0)

                if certainty < min_certainty:
                    return {
                        "fragments": [],
                        "certainty": certainty,
                        "message": "Certainty below threshold"
                    }

                # Get fragment details
                fragments = []
                if strand:
                    for frag_id in strand.fragments[:limit]:
                        fragment = self.store.get(frag_id)
                        if fragment:
                            # Extract readable content
                            content = fragment.content.get("text", "")
                            if not content:
                                # Try semantic - if it's a string use it, otherwise describe
                                semantic = fragment.content.get("semantic")
                                if isinstance(semantic, str):
                                    content = semantic
                                elif isinstance(semantic, list):
                                    content = f"[embedding:{len(semantic)}d]"

                            fragments.append({
                                "id": frag_id,
                                "content": content,
                                "salience": fragment.initial_salience,
                                "created_at": fragment.created_at
                            })

                return {
                    "fragments": fragments,
                    "certainty": certainty,
                    "strand_id": strand.id if strand else None
                }

            return {
                "fragments": [],
                "certainty": 0.0,
                "message": "No memories found"
            }

        except Exception as e:
            return {"fragments": [], "certainty": 0.0, "error": str(e)}

    def memory_identity(self) -> dict:
        """
        Get current identity model.

        Returns:
            {"traits": {...}, "beliefs": [...], "goals": [...]}
        """
        try:
            identity_state = self.engine.identity_store.get_current_state()

            return {
                "traits": {
                    name: {
                        "strength": trait.strength,
                        "description": trait.description
                    }
                    for name, trait in identity_state.traits.items()
                } if identity_state.traits else {},
                "beliefs": [
                    {
                        "content": belief.name,
                        "confidence": belief.strength,
                        "evidence_count": len(belief.evidence_fragments)
                    }
                    for belief in identity_state.beliefs.values()
                ] if identity_state.beliefs else [],
                "goals": [
                    {
                        "description": goal.name,
                        "priority": goal.priority,
                        "progress": 0.0  # Not tracked currently
                    }
                    for goal in identity_state.goals.values()
                ] if identity_state.goals else []
            }

        except Exception as e:
            return {"traits": {}, "beliefs": [], "goals": [], "error": str(e)}

    def memory_status(self) -> dict:
        """
        Get memory system status.

        Returns:
            {"fragment_count": int, "health": str, ...}
        """
        try:
            # Count fragments
            cursor = self.store.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM fragments")
            count = cursor.fetchone()["count"]

            # Get time range
            cursor.execute("""
                SELECT MIN(created_at) as oldest, MAX(created_at) as newest
                FROM fragments
            """)
            row = cursor.fetchone()

            # Get index info
            index_size = 0
            if hasattr(self.store, '_vector_index') and self.store._vector_index is not None:
                index_path = self.store.db_path.with_suffix(".usearch")
                if index_path.exists():
                    index_size = index_path.stat().st_size / (1024 * 1024)

            return {
                "fragment_count": count,
                "index_size_mb": round(index_size, 2),
                "oldest_memory": row["oldest"] if row else None,
                "newest_memory": row["newest"] if row else None,
                "health": "ok"
            }

        except Exception as e:
            return {"fragment_count": 0, "health": "error", "error": str(e)}

    def close(self):
        """Close the memory server."""
        self.store.close()


# MCP protocol wrapper (for actual MCP integration)
def create_mcp_server(db_path: str = "~/.reconstructions/memory.db"):
    """
    Create MCP-compatible server.

    This can be used with the mcp library to expose tools.
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    return MemoryServer(path)
