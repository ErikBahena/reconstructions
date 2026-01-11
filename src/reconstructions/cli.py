"""
CLI Interface for the Memory System.

Provides an interactive REPL for encoding, querying, and exploring memories.
"""

import os
import sys
import tempfile
from typing import Optional, List
from pathlib import Path

from .core import Query
from .store import FragmentStore
from .encoding import Experience
from .engine import ReconstructionEngine, Result, ResultType


class CLI:
    """
    Interactive command-line interface for the memory system.
    
    Commands:
        /store <text>   - Store a new memory
        /remember <query> - Query memories
        /status         - Show system status
        /identity       - Show identity state
        /help           - Show help
        /exit           - Exit the CLI
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the CLI.
        
        Args:
            db_path: Path to database (default: temp directory)
        """
        if db_path is None:
            db_path = str(Path(tempfile.gettempdir()) / "reconstructions.db")
        
        self.db_path = db_path
        self.store = FragmentStore(db_path)
        self.engine = ReconstructionEngine(self.store)
        self._running = False
        
    def run(self) -> None:
        """Run the interactive REPL."""
        self._running = True
        self._print_welcome()
        
        while self._running:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                
                output = self.process_input(user_input)
                if output:
                    print(output)
                    
            except KeyboardInterrupt:
                print("\nUse /exit to quit.")
            except EOFError:
                self._running = False
                
        self._cleanup()
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and return output.
        
        Args:
            user_input: Raw input from user
            
        Returns:
            Output string to display
        """
        # Check for commands
        if user_input.startswith("/"):
            return self._handle_command(user_input)
        
        # Default: treat as a query
        return self._handle_remember(user_input)
    
    def _handle_command(self, cmd: str) -> str:
        """Handle a slash command."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "/store":
            return self._handle_store(args)
        elif command == "/remember":
            return self._handle_remember(args)
        elif command == "/status":
            return self._handle_status()
        elif command == "/identity":
            return self._handle_identity()
        elif command == "/help":
            return self._handle_help()
        elif command == "/exit" or command == "/quit":
            self._running = False
            return "Goodbye!"
        else:
            return f"Unknown command: {command}. Type /help for available commands."
    
    def _handle_store(self, text: str) -> str:
        """Store a new memory."""
        if not text:
            return "Usage: /store <text to remember>"
        
        exp = Experience(text=text)
        self.engine.submit_experience(exp)
        result = self.engine.step()
        
        if result and result.success:
            fragment_id = result.data.get("fragment_id", "unknown")
            return f"✓ Stored memory (id: {fragment_id[:8]}...)"
        else:
            return "✗ Failed to store memory"
    
    def _handle_remember(self, query_text: str) -> str:
        """Query memories."""
        if not query_text:
            return "Usage: /remember <what to recall> or just type your query"
        
        query = Query(semantic=query_text)
        self.engine.submit_query(query)
        result = self.engine.step()
        
        if result and result.success:
            return self._format_strand_result(result)
        else:
            return "No memories found."
    
    def _format_strand_result(self, result: Result) -> str:
        """Format a strand result for display."""
        strand = result.data.get("strand")
        if not strand:
            return "No memories found."
        
        certainty = result.data.get("certainty", 0.0)
        fragment_count = len(strand.fragments)
        
        lines = [
            f"📝 Reconstructed Memory",
            f"   Fragments: {fragment_count}",
            f"   Coherence: {strand.coherence_score:.2f}",
            f"   Certainty: {certainty:.2f}",
            ""
        ]
        
        # Show fragment contents
        if fragment_count > 0:
            lines.append("   Fragments:")
            for frag_id in strand.fragments[:5]:  # Limit to first 5
                fragment = self.store.get(frag_id)
                if fragment:
                    content = fragment.content.get("semantic", "")
                    if isinstance(content, str) and len(content) > 50:
                        content = content[:50] + "..."
                    lines.append(f"   • {content}")
            
            if fragment_count > 5:
                lines.append(f"   ... and {fragment_count - 5} more")
        
        return "\n".join(lines)
    
    def _handle_status(self) -> str:
        """Show system status."""
        # Count fragments via embeddings dict
        fragment_count = len(self.store.embeddings)
        queue_size = len(self.engine.goal_queue)
        
        return f"""System Status:
   Database: {self.db_path}
   Fragments: {fragment_count}
   Pending Goals: {queue_size}"""
    
    def _handle_identity(self) -> str:
        """Show identity state."""
        state = self.engine.identity_store.get_current_state()
        
        lines = ["Identity State:"]
        
        if state.traits:
            lines.append(f"   Traits: {len(state.traits)}")
            for t in list(state.traits.values())[:3]:
                lines.append(f"   • {t.name} ({t.strength:.2f})")
        else:
            lines.append("   Traits: None defined")
            
        if state.beliefs:
            lines.append(f"   Beliefs: {len(state.beliefs)}")
        else:
            lines.append("   Beliefs: None defined")
            
        if state.goals:
            lines.append(f"   Goals: {len(state.goals)}")
        else:
            lines.append("   Goals: None defined")
        
        return "\n".join(lines)
    
    def _handle_help(self) -> str:
        """Show help text."""
        return """Memory System CLI

Commands:
  /store <text>     Store a new memory
  /remember <query> Query memories (or just type your query)
  /status           Show system status
  /identity         Show identity state
  /help             Show this help
  /exit             Exit the CLI

Examples:
  > /store I went to the park today and saw beautiful flowers
  > /remember park flowers
  > What did I do yesterday?"""

    def _print_welcome(self) -> None:
        """Print welcome message."""
        print("""
╔══════════════════════════════════════════╗
║     Memory Reconstruction System         ║
║     Type /help for commands              ║
╚══════════════════════════════════════════╝
""")
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        self.store.close()
        print("\nSession ended.")


def main():
    """Entry point for the CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Reconstruction System CLI")
    parser.add_argument("--db", type=str, help="Path to database file")
    args = parser.parse_args()
    
    cli = CLI(db_path=args.db)
    cli.run()


if __name__ == "__main__":
    main()
