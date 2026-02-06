"""Tests for LLM synthesis in reconstruction."""

import time
from unittest.mock import patch, MagicMock

import pytest

from reconstructions.core import Fragment, Strand, Query
from reconstructions.llm_client import LLMConfig, LLMResult, MemoryLLMClient
from reconstructions.reconstruction import synthesize_narrative


def _make_fragment(text: str) -> Fragment:
    """Helper to create a fragment with text content."""
    return Fragment(
        content={"text": text},
        initial_salience=0.5,
        created_at=time.time(),
    )


class TestSynthesizeNarrative:
    """Tests for the synthesize_narrative function."""

    def setup_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def teardown_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def test_empty_fragments_returns_none(self):
        config = LLMConfig()
        result = synthesize_narrative([], Query(semantic="test"), 0.5, config)
        assert result is None

    def test_unavailable_llm_returns_none(self):
        config = LLMConfig()
        frags = [_make_fragment("hello")]

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("refused")
            result = synthesize_narrative(frags, Query(semantic="test"), 0.5, config)

        assert result is None

    def test_successful_synthesis(self):
        config = LLMConfig()
        frags = [
            _make_fragment("Set up RTMP streaming source"),
            _make_fragment("Configured GStreamer pipeline for video encoding"),
        ]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(
            success=True,
            content="The developer set up an RTMP streaming source and configured a GStreamer pipeline for video encoding.",
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = synthesize_narrative(frags, Query(semantic="streaming setup"), 0.8, config)

        assert result is not None
        assert "RTMP" in result
        assert "GStreamer" in result

    def test_synthesis_includes_coherence_guidance(self):
        """Verify the prompt includes coherence-appropriate guidance."""
        config = LLMConfig()
        frags = [_make_fragment("fragment text")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=True, content="narrative")

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            # Low coherence
            synthesize_narrative(frags, Query(semantic="test"), 0.2, config)

            # Check prompt contains uncertainty guidance
            call_args = mock_client.generate.call_args
            prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
            assert "HIGH uncertainty" in prompt

    def test_high_coherence_guidance(self):
        config = LLMConfig()
        frags = [_make_fragment("fragment text")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=True, content="narrative")

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            synthesize_narrative(frags, Query(semantic="test"), 0.9, config)

            call_args = mock_client.generate.call_args
            prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
            assert "reasonable confidence" in prompt

    def test_llm_failure_returns_none(self):
        config = LLMConfig()
        frags = [_make_fragment("hello")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=False, error="timeout")

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = synthesize_narrative(frags, Query(semantic="test"), 0.5, config)

        assert result is None

    def test_skips_embedding_only_fragments(self):
        """Fragments with only embeddings (no text) should be skipped."""
        config = LLMConfig()
        frags = [
            Fragment(content={"semantic": [0.1] * 384}, initial_salience=0.5),
            _make_fragment("actual text fragment"),
        ]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = LLMResult(success=True, content="narrative")

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = synthesize_narrative(frags, Query(semantic="test"), 0.5, config)

        assert result == "narrative"
        # The prompt should only contain the text fragment
        call_args = mock_client.generate.call_args
        prompt = call_args.kwargs.get("prompt", call_args[0][0] if call_args[0] else "")
        assert "actual text fragment" in prompt

    def test_all_embedding_fragments_returns_none(self):
        """If all fragments are embedding-only, return None."""
        config = LLMConfig()
        frags = [
            Fragment(content={"semantic": [0.1] * 384}, initial_salience=0.5),
        ]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = synthesize_narrative(frags, Query(semantic="test"), 0.5, config)

        assert result is None


class TestStrandSynthesisField:
    """Tests for the synthesis field on Strand."""

    def test_strand_default_synthesis_is_none(self):
        strand = Strand()
        assert strand.synthesis is None

    def test_strand_with_synthesis(self):
        strand = Strand(synthesis="A narrative summary.")
        assert strand.synthesis == "A narrative summary."

    def test_strand_to_dict_includes_synthesis(self):
        strand = Strand(synthesis="Summary text.")
        d = strand.to_dict()
        assert d["synthesis"] == "Summary text."

    def test_strand_to_dict_synthesis_none(self):
        strand = Strand()
        d = strand.to_dict()
        assert d["synthesis"] is None

    def test_strand_from_dict_with_synthesis(self):
        d = {
            "id": "test-id",
            "fragments": [],
            "assembly_context": {},
            "coherence_score": 0.5,
            "variance": 0.3,
            "certainty": 0.8,
            "synthesis": "Reconstructed narrative.",
            "created_at": 1234567890.0,
        }
        strand = Strand.from_dict(d)
        assert strand.synthesis == "Reconstructed narrative."

    def test_strand_from_dict_without_synthesis(self):
        """Backward compatibility: old strands without synthesis field."""
        d = {
            "id": "test-id",
            "fragments": [],
            "assembly_context": {},
            "coherence_score": 0.5,
            "variance": 0.3,
            "certainty": 0.8,
            "created_at": 1234567890.0,
        }
        strand = Strand.from_dict(d)
        assert strand.synthesis is None
