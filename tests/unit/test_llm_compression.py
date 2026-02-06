"""Tests for LLM text compression in features.py."""

import time
from unittest.mock import patch, MagicMock

import pytest

from reconstructions.encoding import Experience, Context
from reconstructions.llm_client import LLMConfig, LLMResult, MemoryLLMClient
from reconstructions.features import compress_text_with_llm, extract_all_features


class TestCompressTextWithLLM:
    """Tests for the compress_text_with_llm function."""

    def setup_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def teardown_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def test_empty_text_returns_none(self):
        config = LLMConfig()
        assert compress_text_with_llm("", config) is None

    def test_short_text_returns_none(self):
        config = LLMConfig()
        assert compress_text_with_llm("short", config) is None

    def test_unavailable_llm_returns_none(self):
        config = LLMConfig()
        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("refused")
            result = compress_text_with_llm(
                "This is a long enough text to trigger compression for sure", config
            )
        assert result is None

    def test_successful_compression(self):
        config = LLMConfig()
        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(
            success=True, content="Edited Python test file for unit testing."
        )

        with patch("reconstructions.features.get_llm_client", return_value=mock_client):
            result = compress_text_with_llm(
                "Ran pytest tests/unit/test_encoder.py with --verbose flag and all 12 tests passed in 2.3s",
                config,
            )

        assert result == "Edited Python test file for unit testing."

    def test_llm_failure_returns_none(self):
        config = LLMConfig()
        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(
            success=False, error="timeout"
        )

        with patch("reconstructions.features.get_llm_client", return_value=mock_client):
            result = compress_text_with_llm(
                "This is a long enough text to trigger compression for sure", config
            )

        assert result is None

    def test_llm_returns_empty_string(self):
        config = LLMConfig()
        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=True, content="")

        with patch("reconstructions.features.get_llm_client", return_value=mock_client):
            result = compress_text_with_llm(
                "This is a long enough text to trigger compression for sure", config
            )

        assert result is None


class TestExtractAllFeaturesWithCompression:
    """Tests for extract_all_features with LLM compression."""

    def setup_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def teardown_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def test_without_llm_config_no_summary(self):
        """Without llm_config, no summary field should be created."""
        exp = Experience(text="Test experience text content here")
        ctx = Context()
        features = extract_all_features(exp, ctx)
        assert "text" in features
        assert "summary" not in features

    def test_with_compression_disabled(self):
        """With compression disabled, no summary field should be created."""
        config = LLMConfig(enable_compression=False)
        exp = Experience(text="Test experience text content here")
        ctx = Context()
        features = extract_all_features(exp, ctx, llm_config=config)
        assert "text" in features
        assert "summary" not in features

    def test_with_compression_enabled_stores_summary(self):
        """With compression enabled, summary should be stored alongside text."""
        config = LLMConfig(enable_compression=True)
        exp = Experience(text="Ran full test suite with 343 tests passing in 5.2 seconds")
        ctx = Context()

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(
            success=True, content="All 343 tests passed successfully."
        )

        with patch("reconstructions.features.get_llm_client", return_value=mock_client):
            features = extract_all_features(exp, ctx, llm_config=config)

        assert features["text"] == "Ran full test suite with 343 tests passing in 5.2 seconds"
        assert features["summary"] == "All 343 tests passed successfully."

    def test_compression_failure_still_has_text(self):
        """If compression fails, text is still stored and embedded normally."""
        config = LLMConfig(enable_compression=True)
        exp = Experience(text="Some developer activity text for testing")
        ctx = Context()

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=False, error="timeout")

        with patch("reconstructions.features.get_llm_client", return_value=mock_client):
            features = extract_all_features(exp, ctx, llm_config=config)

        assert features["text"] == "Some developer activity text for testing"
        assert "summary" not in features
        assert "semantic" in features  # Embedding still created from raw text
