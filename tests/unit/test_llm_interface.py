"""
Unit tests for LLM interface.

These tests mock the Ollama API to test the interface logic.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from reconstructions.llm_interface import (
    OllamaConfig,
    OllamaClient,
    LLMInterface
)
from reconstructions.store import FragmentStore
from reconstructions.engine import ReconstructionEngine
from reconstructions.encoding import Experience


class TestOllamaConfig:
    """Test Ollama configuration."""
    
    def test_default_config(self):
        """Default configuration is sensible."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.model == "gemma3:4b"
        assert config.timeout == 60
    
    def test_generate_url(self):
        """Generate URL is correct."""
        config = OllamaConfig()
        assert config.generate_url == "http://localhost:11434/api/generate"


class TestOllamaClient:
    """Test Ollama client (mocked)."""
    
    @patch('reconstructions.llm_interface.requests.post')
    def test_generate_success(self, mock_post):
        """Generate returns response on success."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        result = client.generate("Hello")
        
        assert result == "Hello!"
        mock_post.assert_called_once()
    
    @patch('reconstructions.llm_interface.requests.post')
    def test_generate_error(self, mock_post):
        """Generate handles errors gracefully."""
        import requests as req
        mock_post.side_effect = req.exceptions.RequestException("Connection failed")
        
        client = OllamaClient()
        result = client.generate("Hello")
        
        assert "Error" in result
    
    @patch('reconstructions.llm_interface.requests.get')
    def test_is_available(self, mock_get):
        """is_available checks connection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        client = OllamaClient()
        assert client.is_available() is True


class TestLLMInterface:
    """Test LLM interface logic."""
    
    def test_parse_intent_store(self):
        """Parse store intent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            # Mock the generate method
            with patch.object(llm.client, 'generate') as mock_gen:
                mock_gen.return_value = '{"intent": "store", "keywords": [], "store_content": "remember this"}'
                
                result = llm._parse_intent("Remember that I like pizza")
                
                assert result["intent"] == "store"
            
            store.close()
    
    def test_parse_intent_query(self):
        """Parse query intent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            with patch.object(llm.client, 'generate') as mock_gen:
                mock_gen.return_value = '{"intent": "query", "keywords": ["pizza"], "store_content": ""}'
                
                result = llm._parse_intent("What food do I like?")
                
                assert result["intent"] == "query"
                assert "pizza" in result["keywords"]
            
            store.close()
    
    def test_parse_intent_fallback(self):
        """Falls back to chat on invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            with patch.object(llm.client, 'generate') as mock_gen:
                mock_gen.return_value = 'This is not JSON'
                
                result = llm._parse_intent("Hello")
                
                assert result["intent"] == "chat"
            
            store.close()
    
    def test_store_memory(self):
        """Store memory through interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            result = llm._store_memory("I love pizza")
            
            assert "remember" in result.lower()
            # Verify something was stored
            assert not store.is_empty()
            
            store.close()
    
    def test_format_memories_empty(self):
        """Format empty memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            result = llm._format_memories_for_context(None)
            
            assert "No relevant memories" in result
            
            store.close()
    
    def test_handle_meta_query(self):
        """Handle meta queries about memory system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FragmentStore(str(Path(tmpdir) / "test.db"))
            engine = ReconstructionEngine(store)
            llm = LLMInterface(engine)
            
            result = llm._handle_meta_query("How does your memory work?")
            
            assert "reconstructive" in result.lower()
            assert "fragments" in result.lower()
            
            store.close()
