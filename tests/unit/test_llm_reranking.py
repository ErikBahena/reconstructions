"""Tests for LLM reranking in reconstruction."""

import time
from unittest.mock import patch, MagicMock

import pytest

from reconstructions.core import Fragment, Query
from reconstructions.llm_client import LLMConfig, LLMResult, MemoryLLMClient
from reconstructions.reconstruction import llm_rerank


def _make_fragment(text: str, frag_id: str = None) -> Fragment:
    """Helper to create a fragment with text content."""
    return Fragment(
        id=frag_id or f"frag-{text[:8]}",
        content={"text": text, "semantic": [0.1] * 384},
        initial_salience=0.5,
        created_at=time.time(),
    )


class TestLLMRerank:
    """Tests for the llm_rerank function."""

    def setup_method(self):
        """Reset singleton before each test."""
        import reconstructions.llm_client as mod
        mod._client = None

    def teardown_method(self):
        import reconstructions.llm_client as mod
        mod._client = None

    def test_empty_candidates_returns_empty(self):
        config = LLMConfig()
        result = llm_rerank([], Query(semantic="test"), config)
        assert result == []

    def test_no_semantic_query_returns_unchanged(self):
        config = LLMConfig()
        frags = [_make_fragment("hello")]
        result = llm_rerank(frags, Query(), config)
        assert result == frags

    def test_unavailable_llm_returns_unchanged(self):
        config = LLMConfig()
        frags = [_make_fragment("hello"), _make_fragment("world")]

        with patch("reconstructions.llm_client.requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("refused")
            result = llm_rerank(frags, Query(semantic="test"), config)

        assert result == frags

    def test_successful_rerank_filters_low_scores(self):
        config = LLMConfig(rerank_min_score=3)
        frags = [
            _make_fragment("relevant RTMP streaming"),
            _make_fragment("grocery list for shopping"),
            _make_fragment("another relevant stream"),
        ]

        # Mock the LLM to return scores
        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate_json.return_value = LLMResult(
            success=True,
            content="",
            parsed=[
                {"index": 1, "score": 9},
                {"index": 2, "score": 1},  # Below threshold
                {"index": 3, "score": 7},
            ],
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = llm_rerank(frags, Query(semantic="RTMP streaming"), config)

        # Grocery fragment (score=1) should be filtered out
        assert len(result) == 2
        assert result[0].content["text"] == "relevant RTMP streaming"
        assert result[1].content["text"] == "another relevant stream"

    def test_rerank_sorts_by_score_descending(self):
        config = LLMConfig(rerank_min_score=0)
        frags = [
            _make_fragment("low relevance"),
            _make_fragment("high relevance"),
            _make_fragment("medium relevance"),
        ]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate_json.return_value = LLMResult(
            success=True,
            parsed=[
                {"index": 1, "score": 3},
                {"index": 2, "score": 9},
                {"index": 3, "score": 6},
            ],
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = llm_rerank(frags, Query(semantic="test"), config)

        assert result[0].content["text"] == "high relevance"
        assert result[1].content["text"] == "medium relevance"
        assert result[2].content["text"] == "low relevance"

    def test_json_parse_failure_returns_unchanged(self):
        config = LLMConfig()
        frags = [_make_fragment("hello"), _make_fragment("world")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate_json.return_value = LLMResult(
            success=False, error="JSON parse error"
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = llm_rerank(frags, Query(semantic="test"), config)

        assert result == frags

    def test_all_filtered_returns_original(self):
        """If all fragments score below threshold, return originals rather than empty."""
        config = LLMConfig(rerank_min_score=5)
        frags = [_make_fragment("a"), _make_fragment("b")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate_json.return_value = LLMResult(
            success=True,
            parsed=[
                {"index": 1, "score": 1},
                {"index": 2, "score": 2},
            ],
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = llm_rerank(frags, Query(semantic="test"), config)

        # Should return originals, not empty list
        assert result == frags

    def test_malformed_json_items_handled(self):
        """Malformed items in the JSON array are skipped gracefully."""
        config = LLMConfig(rerank_min_score=0)
        frags = [_make_fragment("a"), _make_fragment("b")]

        mock_client = MagicMock(spec=MemoryLLMClient)
        mock_client.is_available.return_value = True
        mock_client.generate_json.return_value = LLMResult(
            success=True,
            parsed=[
                {"index": 1, "score": 8},
                {"bad": "item"},            # Missing index/score
                {"index": "x", "score": 5}, # Non-numeric index
                {"index": 2, "score": 6},
            ],
        )

        with patch("reconstructions.reconstruction.get_llm_client", return_value=mock_client):
            result = llm_rerank(frags, Query(semantic="test"), config)

        assert len(result) == 2
