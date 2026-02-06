"""
Claude Code skills for the memory system.

Implements the /memory skill with operations: store, recall, status, identity, recent, forget.
"""

import time
from typing import Any, Optional

from .context_manager import get_session
from ..core import Query
from ..encoding import Experience
from ..health import format_health_report


def execute_skill(operation: str, args: str = "") -> dict[str, Any]:
    """
    Execute a memory skill operation.

    Args:
        operation: The operation to perform
        args: Arguments for the operation

    Returns:
        Result dict with operation output
    """
    session = get_session()

    # Ensure session is active
    if not session.is_active:
        session.start_session()

    operations = {
        "store": _store,
        "recall": _recall,
        "status": _status,
        "health": _health,
        "consolidate": _consolidate,
        "identity": _identity,
        "recent": _recent,
        "forget": _forget
    }

    if operation in operations:
        return operations[operation](args)
    else:
        return {
            "error": f"Unknown operation: {operation}",
            "available": list(operations.keys())
        }


def _store(text: str) -> dict[str, Any]:
    """
    Store a new memory manually.

    Args:
        text: Text content to store

    Returns:
        Result with fragment ID and salience
    """
    if not text.strip():
        return {"error": "No text provided for storage"}

    session = get_session()
    if not session.engine or not session.store:
        return {"error": "Session not initialized"}

    experience = Experience(
        text=text.strip(),
        emotional={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
        source="manual",
        tags=["manual:store"]
    )

    session.engine.submit_experience(experience)
    result = session.engine.step()

    if result and result.success:
        fragment_id = result.data.get("fragment_id", "")
        fragment = session.store.get(fragment_id)
        salience = fragment.initial_salience if fragment else 0.5

        session.increment_fragments_encoded()

        return {
            "success": True,
            "fragment_id": fragment_id,
            "salience": round(salience, 3)
        }

    return {"success": False, "error": "Failed to encode memory"}


def _recall(query_text: str) -> dict[str, Any]:
    """
    Recall memories matching a query.

    Args:
        query_text: Semantic query text

    Returns:
        Result with matching fragments and certainty
    """
    if not query_text.strip():
        return {"error": "No query provided"}

    session = get_session()
    if not session.engine or not session.store:
        return {"error": "Session not initialized"}

    query = Query(semantic=query_text.strip())
    session.engine.submit_query(query)
    result = session.engine.step()

    session.increment_queries_processed()

    if result and result.success:
        strand = result.data.get("strand")
        certainty = result.data.get("certainty", 0.0)

        fragments = []
        if strand:
            for frag_id in strand.fragments[:10]:  # Limit to 10
                fragment = session.store.get(frag_id)
                if fragment:
                    # Prefer summary over raw text
                    content = fragment.content.get("summary", "") or fragment.content.get("text", "")
                    if not content:
                        semantic = fragment.content.get("semantic")
                        if isinstance(semantic, str):
                            content = semantic
                        elif isinstance(semantic, list):
                            content = f"[embedding:{len(semantic)}d]"

                    fragments.append({
                        "id": frag_id,
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "salience": round(fragment.initial_salience, 3),
                        "age_hours": round((time.time() - fragment.created_at) / 3600, 1)
                    })

        result_dict = {
            "success": True,
            "fragments": fragments,
            "certainty": round(certainty, 3),
            "strand_id": strand.id if strand else None
        }
        if strand and strand.synthesis:
            result_dict["synthesis"] = strand.synthesis
        return result_dict

    return {"success": False, "fragments": [], "certainty": 0.0}


def _status(args: str = "") -> dict[str, Any]:
    """
    Get basic memory system status.

    Returns:
        Status dict with fragment count, session stats, etc.
    """
    session = get_session()
    if not session.store:
        return {"error": "Session not initialized"}

    # Get fragment count
    cursor = session.store.conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM fragments")
    count = cursor.fetchone()["count"]

    # Get time range
    cursor.execute("""
        SELECT MIN(created_at) as oldest, MAX(created_at) as newest
        FROM fragments
    """)
    row = cursor.fetchone()

    # Get session stats
    session_stats = session.stats.to_dict() if session.stats else {}

    result = {
        "fragment_count": count,
        "oldest_memory": _format_timestamp(row["oldest"]) if row and row["oldest"] else None,
        "newest_memory": _format_timestamp(row["newest"]) if row and row["newest"] else None,
        "session": {
            "duration_seconds": round(session_stats.get("duration_seconds", 0)),
            "fragments_encoded": session_stats.get("fragments_encoded", 0),
            "queries_processed": session_stats.get("queries_processed", 0),
            "tool_uses_observed": session_stats.get("tool_uses_observed", 0)
        }
    }

    # Add index info if available
    if hasattr(session.store, '_vector_index') and session.store._vector_index is not None:
        result["index_size"] = session.store._vector_index.count()

    return result


def _health(args: str = "") -> dict[str, Any]:
    """
    Get comprehensive health report with warnings and recommendations.

    Returns:
        Detailed health report
    """
    session = get_session()
    if not session.health_monitor:
        return {"error": "Health monitoring not available"}

    try:
        report = session.health_monitor.diagnose()

        # Return formatted text report
        return {
            "report_text": format_health_report(report),
            "warnings_count": len(report.warnings),
            "recommendations_count": len(report.recommendations),
            "health_score": _calculate_health_score(report)
        }
    except Exception as e:
        return {"error": f"Health check failed: {str(e)}"}


def _consolidate(args: str = "") -> dict[str, Any]:
    """
    Manually trigger memory consolidation.

    Returns:
        Consolidation statistics
    """
    session = get_session()
    if not session.engine:
        return {"error": "Session not initialized"}

    if not session.engine.consolidation_scheduler:
        return {"error": "Consolidation not enabled"}

    try:
        stats = session.engine.consolidation_scheduler.consolidate()

        return {
            "success": True,
            "rehearsed": stats.get("rehearsed_count", 0),
            "bindings_strengthened": stats.get("bindings_strengthened", 0),
            "patterns_discovered": stats.get("patterns_discovered", 0),
            "duration_ms": stats.get("duration_ms", 0),
            "message": f"Consolidated {stats.get('rehearsed_count', 0)} fragments, "
                      f"created {stats.get('bindings_strengthened', 0)} bindings"
        }
    except Exception as e:
        return {"error": f"Consolidation failed: {str(e)}"}


def _identity(args: str = "") -> dict[str, Any]:
    """
    Get current identity model.

    Returns:
        Identity state with traits, beliefs, and goals
    """
    session = get_session()
    if not session.engine:
        return {"error": "Session not initialized"}

    try:
        identity_state = session.engine.identity_store.get_current_state()

        return {
            "traits": {
                name: {
                    "strength": round(trait.strength, 3),
                    "description": trait.description
                }
                for name, trait in (identity_state.traits or {}).items()
            },
            "beliefs": [
                {
                    "content": belief.name,
                    "confidence": round(belief.strength, 3),
                    "evidence_count": len(belief.evidence_fragments)
                }
                for belief in (identity_state.beliefs or {}).values()
            ],
            "goals": [
                {
                    "description": goal.name,
                    "priority": round(goal.priority, 3)
                }
                for goal in (identity_state.goals or {}).values()
            ]
        }
    except Exception as e:
        return {"traits": {}, "beliefs": [], "goals": [], "error": str(e)}


def _recent(args: str) -> dict[str, Any]:
    """
    List recent memories.

    Args:
        args: Optional count (default 10)

    Returns:
        List of recent fragments
    """
    session = get_session()
    if not session.store:
        return {"error": "Session not initialized"}

    # Parse count from args
    try:
        count = int(args.strip()) if args.strip() else 10
        count = min(max(1, count), 50)  # Clamp to 1-50
    except ValueError:
        count = 10

    cursor = session.store.conn.cursor()
    cursor.execute("""
        SELECT id, created_at, content, initial_salience, tags
        FROM fragments
        ORDER BY created_at DESC
        LIMIT ?
    """, (count,))

    import json as _json

    fragments = []
    for row in cursor.fetchall():
        content = _json.loads(row["content"])
        text = content.get("text", "")
        if not text:
            semantic = content.get("semantic")
            if isinstance(semantic, str):
                text = semantic
            elif isinstance(semantic, list):
                text = f"[embedding:{len(semantic)}d]"

        fragments.append({
            "id": row["id"][:8],  # Short ID
            "content": text[:100] + "..." if len(text) > 100 else text,
            "salience": round(row["initial_salience"], 3),
            "age_hours": round((time.time() - row["created_at"]) / 3600, 1),
            "tags": _json.loads(row["tags"])[:3]  # First 3 tags
        })

    return {"count": len(fragments), "fragments": fragments}


def _forget(fragment_id: str) -> dict[str, Any]:
    """
    Accelerate decay for a memory (soft forget).

    This doesn't delete the memory but reduces its strength significantly.

    Args:
        fragment_id: Fragment ID to forget (can be partial)

    Returns:
        Status of the forget operation
    """
    if not fragment_id.strip():
        return {"error": "No fragment ID provided"}

    session = get_session()
    if not session.store:
        return {"error": "Session not initialized"}

    fragment_id = fragment_id.strip()

    # Try to find the fragment (support partial ID matching)
    cursor = session.store.conn.cursor()
    cursor.execute("""
        SELECT id FROM fragments WHERE id LIKE ?
    """, (fragment_id + "%",))

    rows = cursor.fetchall()

    if len(rows) == 0:
        return {"error": f"No fragment found matching: {fragment_id}"}
    elif len(rows) > 1:
        matches = [r["id"][:8] for r in rows[:5]]
        return {"error": f"Multiple fragments match. Be more specific: {matches}"}

    full_id = rows[0]["id"]
    fragment = session.store.get(full_id)

    if fragment is None:
        return {"error": f"Could not load fragment: {full_id}"}

    # Reduce salience significantly (accelerate decay)
    original_salience = fragment.initial_salience
    fragment.initial_salience = max(0.01, fragment.initial_salience * 0.1)

    # Save the modified fragment
    session.store.save(fragment)

    return {
        "success": True,
        "fragment_id": full_id[:8],
        "original_salience": round(original_salience, 3),
        "new_salience": round(fragment.initial_salience, 3),
        "message": "Memory weakened (accelerated decay)"
    }


def _format_timestamp(ts: Optional[float]) -> Optional[str]:
    """Format a timestamp as a human-readable string."""
    if ts is None:
        return None

    from datetime import datetime
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M")


def _calculate_health_score(report) -> float:
    """Calculate overall health score from report (0-1)."""
    score = 1.0

    # Deduct for warnings
    score -= len(report.warnings) * 0.1

    # Bonus for good metrics
    if report.retrieval_stats.success_rate > 0.7:
        score += 0.1

    if report.consolidation_stats.runs_per_hour >= report.consolidation_stats.target_runs_per_hour * 0.8:
        score += 0.1

    return max(0.0, min(1.0, score))


def format_output(result: dict[str, Any]) -> str:
    """
    Format skill output for display.

    Args:
        result: Result dict from skill operation

    Returns:
        Formatted string output
    """
    if "error" in result:
        return f"Error: {result['error']}"

    output_lines = []

    # Health report
    if "report_text" in result:
        output_lines.append(result["report_text"])
        score = result.get("health_score", 0.0)
        health_emoji = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
        output_lines.append(f"\n{health_emoji} Health Score: {score * 100:.0f}/100")
        return "\n".join(output_lines)

    # Pruning result
    if "candidates" in result:
        if result.get("dry_run"):
            output_lines.append("DRY RUN - No fragments actually deleted")
        output_lines.append(f"Candidates identified: {result['candidates']}")
        output_lines.append(f"Would/Did prune: {result['total_pruned']} fragments")
        if result.get("message"):
            output_lines.append(f"\n{result['message']}")
        return "\n".join(output_lines)

    # Consolidation result
    if "rehearsed" in result:
        output_lines.append(result.get("message", "Consolidation complete"))
        output_lines.append(f"Duration: {result.get('duration_ms', 0)}ms")
        return "\n".join(output_lines)

    if "success" in result:
        if result.get("success"):
            if "fragment_id" in result and "fragments" not in result:
                # Store result
                output_lines.append(f"Stored memory: {result['fragment_id'][:8]}")
                output_lines.append(f"Salience: {result['salience']}")
            elif "message" in result:
                output_lines.append(result["message"])
                if "fragment_id" in result:
                    output_lines.append(f"Fragment: {result.get('fragment_id', 'unknown')}")
        else:
            output_lines.append("Operation failed")

    if "fragments" in result:
        # Recall or recent result
        fragments = result["fragments"]
        if not fragments:
            output_lines.append("No memories found")
        else:
            if "synthesis" in result:
                output_lines.append(f"Synthesis: {result['synthesis']}")
                output_lines.append("")
            if "certainty" in result:
                output_lines.append(f"Certainty: {result['certainty']}")
            output_lines.append(f"Found {len(fragments)} memories:")
            for f in fragments:
                output_lines.append(f"  [{f.get('id', '?')[:8]}] {f.get('content', '')}")

    if "fragment_count" in result:
        # Status result
        output_lines.append(f"Total memories: {result['fragment_count']}")
        if result.get("oldest_memory"):
            output_lines.append(f"Oldest: {result['oldest_memory']}")
            output_lines.append(f"Newest: {result['newest_memory']}")
        if "session" in result:
            s = result["session"]
            output_lines.append(f"Session: {s['fragments_encoded']} encoded, {s['queries_processed']} queries")

    if "traits" in result:
        # Identity result
        if result["traits"]:
            output_lines.append("Traits:")
            for name, data in result["traits"].items():
                output_lines.append(f"  {name}: {data['strength']} - {data.get('description', '')}")
        if result["beliefs"]:
            output_lines.append("Beliefs:")
            for b in result["beliefs"]:
                output_lines.append(f"  {b['content']} (confidence: {b['confidence']})")
        if result["goals"]:
            output_lines.append("Goals:")
            for g in result["goals"]:
                output_lines.append(f"  {g['description']} (priority: {g['priority']})")
        if not result["traits"] and not result["beliefs"] and not result["goals"]:
            output_lines.append("No identity state yet")

    return "\n".join(output_lines) if output_lines else str(result)


def main():
    """CLI entry point for skills."""
    import sys
    import json
    import warnings

    # Suppress the sys.modules warning when running as -m
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*sys.modules.*")

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: skills.py <operation> [args]"}), file=sys.stderr)
        sys.exit(1)

    operation = sys.argv[1]
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    result = execute_skill(operation, args)

    # Format output based on operation
    if operation in ["recall", "recent"]:
        _print_recall_result(result)
    elif operation == "health":
        _print_health_result(result)
    elif operation == "consolidate":
        _print_consolidation_result(result)
    else:
        # Default: clean JSON
        print(json.dumps(result, indent=2))


def _print_recall_result(result):
    """Print recall results in a clean, readable format."""
    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    fragments = result.get("fragments", [])
    certainty = result.get("certainty", 0.0)

    if not fragments:
        print("No memories found")
        return

    # Show synthesis if available
    synthesis = result.get("synthesis")
    if synthesis:
        print(f"\nSynthesis: {synthesis}\n")

    print(f"\nFound {len(fragments)} memories (certainty: {certainty:.2f}):\n")
    for i, frag in enumerate(fragments, 1):
        content = frag.get("content", "")
        # Truncate long content
        if len(content) > 100:
            content = content[:97] + "..."
        salience = frag.get("salience", 0)
        age_hours = frag.get("age_hours", 0)

        print(f"{i}. {content}")
        print(f"   Salience: {salience:.3f} | Age: {age_hours:.1f}h\n")


def _print_health_result(result):
    """Print health report in a clean format."""
    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    if "report_text" in result:
        print(result["report_text"])
    else:
        import json
        print(json.dumps(result, indent=2))


def _print_consolidation_result(result):
    """Print consolidation results cleanly."""
    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print(f"\n✓ Consolidation complete")
    print(f"  Rehearsed: {result.get('rehearsed', 0)} fragments")
    print(f"  Bindings: {result.get('bindings_strengthened', 0)} strengthened")
    print(f"  Patterns: {result.get('patterns_discovered', 0)} discovered")
    print(f"  Duration: {result.get('duration_ms', 0):.0f}ms\n")


if __name__ == "__main__":
    main()
