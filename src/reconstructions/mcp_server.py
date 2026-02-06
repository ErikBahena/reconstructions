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
from .health import MemoryHealthMonitor, format_health_report
from .consolidation import ConsolidationConfig


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

        # Initialize health monitoring
        self.health_monitor = MemoryHealthMonitor(self.store, db_path.parent)

        # Initialize engine with health monitoring
        consolidation_config = ConsolidationConfig()
        self.engine = ReconstructionEngine(
            self.store,
            enable_consolidation=True,
            consolidation_config=consolidation_config,
            health_monitor=self.health_monitor
        )

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

    def memory_status(self, format_text: bool = False) -> dict:
        """
        Get comprehensive memory system health status.

        Args:
            format_text: If True, return formatted text report

        Returns:
            Health report dict or formatted string
        """
        try:
            report = self.health_monitor.diagnose()

            if format_text:
                return {
                    "status": "ok",
                    "report": format_health_report(report)
                }

            # Return structured data
            return {
                "status": "ok",
                "timestamp": report.timestamp.isoformat(),
                "database_size_mb": report.database_size_mb,
                "fragments": {
                    "total": report.fragment_stats.total_count,
                    "last_24h": report.fragment_stats.last_24h_count,
                    "avg_salience": report.fragment_stats.avg_salience,
                    "avg_bindings": report.fragment_stats.avg_bindings,
                    "never_accessed": report.fragment_stats.never_accessed_count,
                    "low_salience": report.fragment_stats.low_salience_count
                },
                "consolidation": {
                    "last_run_seconds_ago": report.consolidation_stats.last_run_ago_seconds,
                    "runs_per_hour": report.consolidation_stats.runs_per_hour,
                    "total_rehearsals": report.consolidation_stats.total_rehearsals,
                    "bindings_created": report.consolidation_stats.bindings_created
                },
                "retrieval": {
                    "recent_queries": report.retrieval_stats.recent_queries_count,
                    "avg_coherence": report.retrieval_stats.avg_coherence,
                    "avg_latency_ms": report.retrieval_stats.avg_latency_ms,
                    "success_rate": report.retrieval_stats.success_rate
                },
                "warnings": report.warnings,
                "recommendations": report.recommendations
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def memory_consolidate(self) -> dict:
        """
        Manually trigger memory consolidation.

        Returns:
            Consolidation statistics
        """
        try:
            if not self.engine.consolidation_scheduler:
                return {
                    "success": False,
                    "error": "Consolidation not enabled"
                }

            stats = self.engine.consolidation_scheduler.consolidate()

            return {
                "success": True,
                "rehearsed": stats.get("rehearsed_count", 0),
                "bindings_strengthened": stats.get("bindings_strengthened", 0),
                "patterns_discovered": stats.get("patterns_discovered", 0),
                "duration_ms": stats.get("duration_ms", 0)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

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
