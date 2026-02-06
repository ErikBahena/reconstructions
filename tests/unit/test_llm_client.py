"""Tests for the LLM client module."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from reconstructions.llm_client import (
    LLMConfig,
    LLMResult,
    MemoryLLMClient,
    get_llm_client,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        config = LLMConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.model == "gemma3:4b"
        assert config.enable_reranking is True
        assert config.enable_compression is True
        assert config.enable_synthesis is True
        assert config.rerank_timeout == 10
        assert config.rerank_min_score == 3
        assert config.availability_cache_ttl == 60.0

    def test_custom_values(self):
        config = LLMConfig(model="llama3:8b", enable_reranking=False, rerank_min_score=5)
        assert config.model == "llama3:8b"
        assert config.enable_reranking is False
        assert config.rerank_min_score == 5


class TestLLMResult:
    """Tests for LLMResult dataclass."""

    def test_success_result(self):
        result = LLMResult(success=True, content="hello", latency_ms=50.0)
        assert result.success is True
        assert result.content == "hello"
        assert result.error is None

    def test_failure_result(self):
        result = LLMResult(success=False, error="connection refused")
        assert result.success is False
        assert result.error == "connection refused"
        assert result.content == ""

    def test_json_result(self):
        result = LLMResult(success=True, content='[{"a":1}]', parsed=[{"a": 1}])
        assert result.parsed == [{"a": 1}]


class TestMemoryLLMClient:
    """Tests for MemoryLLMClient."""

    def test_is_available_success(self):
        client = MemoryLLMClient(LLMConfig())
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            assert client.is_available() is True

    def test_is_available_failure(self):
        client = MemoryLLMClient(LLMConfig())

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("refused")
            assert client.is_available() is False

    def test_is_available_caching(self):
        client = MemoryLLMClient(LLMConfig(availability_cache_ttl=60.0))
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            # First call hits the server
            assert client.is_available() is True
            assert mock_requests.get.call_count == 1
            # Second call uses cache
            assert client.is_available() is True
            assert mock_requests.get.call_count == 1

    def test_is_available_cache_expiry(self):
        client = MemoryLLMClient(LLMConfig(availability_cache_ttl=0.0))
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            client.is_available()
            client._available_checked_at = 0  # Force expiry
            client.is_available()
            assert mock_requests.get.call_count == 2

    def test_generate_success(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Generated text"}

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = client.generate("test prompt")

        assert result.success is True
        assert result.content == "Generated text"
        assert result.latency_ms > 0

    def test_generate_server_unavailable(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = False
        client._available_checked_at = time.time()

        result = client.generate("test prompt")
        assert result.success is False
        assert "not available" in result.error

    def test_generate_http_error(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = client.generate("test prompt")

        assert result.success is False
        assert "500" in result.error

    def test_generate_connection_error(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.side_effect = ConnectionError("refused")
            result = client.generate("test prompt")

        assert result.success is False

    def test_generate_json_success(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": '[{"index": 1, "score": 8}]'}

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = client.generate_json("return json")

        assert result.success is True
        assert result.parsed == [{"index": 1, "score": 8}]

    def test_generate_json_with_code_fences(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": '```json\n[{"index": 1, "score": 5}]\n```'
        }

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = client.generate_json("return json")

        assert result.success is True
        assert result.parsed == [{"index": 1, "score": 5}]

    def test_generate_json_parse_failure(self):
        client = MemoryLLMClient(LLMConfig())
        client._available = True
        client._available_checked_at = time.time()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "not valid json at all"}

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = client.generate_json("return json")

        assert result.success is False
        assert "JSON parse error" in result.error


class TestGetLLMClient:
    """Tests for the module-level singleton."""

    def test_returns_client(self):
        # Reset the singleton
        import reconstructions.llm_client as mod
        mod._client = None
        client = get_llm_client(LLMConfig())
        assert isinstance(client, MemoryLLMClient)

    def test_singleton_behavior(self):
        import reconstructions.llm_client as mod
        mod._client = None
        c1 = get_llm_client(LLMConfig())
        c2 = get_llm_client()
        assert c1 is c2

    def teardown_method(self):
        # Reset singleton after each test
        import reconstructions.llm_client as mod
        mod._client = None
