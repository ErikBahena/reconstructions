"""
Unit tests for CLI interface.
"""

import pytest
import tempfile
from pathlib import Path
from src.reconstructions.cli import CLI


class TestCLI:
    """Test CLI functionality."""
    
    def test_init_creates_store(self):
        """CLI initializes with a store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            assert cli.store is not None
            assert cli.engine is not None
            
            cli._cleanup()
    
    def test_process_store_command(self):
        """Process /store command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/store I went to the park today")
            
            assert "✓" in output or "Stored" in output
            
            cli._cleanup()
    
    def test_process_remember_command(self):
        """Process /remember command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            # First store something
            cli.process_input("/store I love pizza and pasta")
            
            # Then query
            output = cli.process_input("/remember pizza")
            
            assert "Memory" in output or "Reconstructed" in output
            
            cli._cleanup()
    
    def test_process_status_command(self):
        """Process /status command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/status")
            
            assert "Status" in output
            assert "Fragments" in output
            
            cli._cleanup()
    
    def test_process_identity_command(self):
        """Process /identity command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/identity")
            
            assert "Identity" in output
            
            cli._cleanup()
    
    def test_process_help_command(self):
        """Process /help command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/help")
            
            assert "/store" in output
            assert "/remember" in output
            assert "/exit" in output
            
            cli._cleanup()
    
    def test_process_exit_command(self):
        """Process /exit command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/exit")
            
            assert cli._running is False
            assert "Goodbye" in output
            
            cli._cleanup()
    
    def test_unknown_command(self):
        """Unknown command shows error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            output = cli.process_input("/unknown")
            
            assert "Unknown command" in output
            
            cli._cleanup()
    
    def test_plain_text_as_query(self):
        """Plain text is treated as query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cli = CLI(db_path=db_path)
            
            # Store something first
            cli.process_input("/store Testing memory")
            
            # Plain text query
            output = cli.process_input("testing")
            
            # Should get a result (not an error)
            assert "Memory" in output or "found" in output.lower() or "Reconstructed" in output
            
            cli._cleanup()
