"""
Unit tests for query decomposition module.

Tests the pure logic portions: hash computation, JSON parsing,
fallback behavior. Mocks the Groq LLM and DB calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.common.query_decomposition import (
    _compute_decomposition_hash,
    _call_groq,
    DECOMPOSITION_VERSION,
)


# ── _compute_decomposition_hash ──────────────────────────────────────────────

class TestComputeDecompositionHash:
    def test_deterministic(self):
        h1 = _compute_decomposition_hash("NLP for law")
        h2 = _compute_decomposition_hash("NLP for law")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = _compute_decomposition_hash("NLP for Law")
        h2 = _compute_decomposition_hash("nlp for law")
        assert h1 == h2

    def test_different_queries_different_hashes(self):
        h1 = _compute_decomposition_hash("NLP for law")
        h2 = _compute_decomposition_hash("reinforcement learning")
        assert h1 != h2

    def test_includes_version(self):
        """Changing DECOMPOSITION_VERSION should change the hash."""
        h = _compute_decomposition_hash("test query")
        # Hash should be a valid SHA256 hex string
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── _call_groq ────────────────────────────────────────────────────────────────

class TestCallGroq:
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("app.common.query_decomposition.OpenAI")
    def test_successful_call(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "topic": "NLP",
            "domain": "law",
            "aspect": None,
            "suggested_specificity": "specific",
            "reasoning": "Test",
        })
        mock_client.chat.completions.create.return_value = mock_resp

        result = _call_groq("Test prompt")
        assert result is not None
        assert result["topic"] == "NLP"
        assert result["domain"] == "law"

    @patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False)
    def test_no_api_key(self):
        # Remove GROQ_API_KEY
        import os
        old = os.environ.pop("GROQ_API_KEY", None)
        try:
            result = _call_groq("Test prompt")
            assert result is None
        finally:
            if old:
                os.environ["GROQ_API_KEY"] = old

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("app.common.query_decomposition.OpenAI")
    def test_invalid_json_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "not valid json"
        mock_client.chat.completions.create.return_value = mock_resp

        result = _call_groq("Test prompt")
        assert result is None


# ── decompose_query (integration with mocked deps) ───────────────────────────

class TestDecomposeQuery:
    @patch("app.common.query_decomposition._call_groq")
    @patch("app.common.query_decomposition._cache_decomposition")
    @patch("app.common.query_decomposition._get_cached_decomposition")
    def test_cache_hit(self, mock_get_cache, mock_set_cache, mock_groq):
        from app.common.query_decomposition import decompose_query

        cached = {"topic": "NLP", "domain": "law", "aspect": None}
        mock_get_cache.return_value = cached
        mock_conn = MagicMock()

        result = decompose_query(mock_conn, "NLP for law")
        assert result["topic"] == "NLP"
        mock_groq.assert_not_called()

    @patch("app.common.query_decomposition._call_groq")
    @patch("app.common.query_decomposition._cache_decomposition")
    @patch("app.common.query_decomposition._get_cached_decomposition")
    def test_cache_miss_calls_groq(self, mock_get_cache, mock_set_cache, mock_groq):
        from app.common.query_decomposition import decompose_query

        mock_get_cache.return_value = None
        mock_groq.return_value = {
            "topic": "NLP",
            "domain": "law",
            "aspect": None,
            "suggested_specificity": "specific",
            "reasoning": "Topic is NLP, domain is law.",
        }
        mock_conn = MagicMock()

        result = decompose_query(mock_conn, "NLP for law")
        assert result["topic"] == "NLP"
        assert result["domain"] == "law"
        mock_groq.assert_called_once()
        mock_set_cache.assert_called_once()

    @patch("app.common.query_decomposition._call_groq")
    @patch("app.common.query_decomposition._cache_decomposition")
    @patch("app.common.query_decomposition._get_cached_decomposition")
    def test_groq_failure_fallback(self, mock_get_cache, mock_set_cache, mock_groq):
        from app.common.query_decomposition import decompose_query

        mock_get_cache.return_value = None
        mock_groq.return_value = None
        mock_conn = MagicMock()

        result = decompose_query(mock_conn, "NLP for law")
        # Fallback: entire query becomes the topic
        assert result["topic"] == "NLP for law"
        assert result["domain"] is None
        assert result["suggested_specificity"] == "broad"


# ── decompose_paper ───────────────────────────────────────────────────────────

class TestDecomposePaper:
    @patch("app.common.query_decomposition._call_groq")
    @patch("app.common.query_decomposition._cache_decomposition")
    @patch("app.common.query_decomposition._get_cached_decomposition")
    def test_decomposes_paper_metadata(self, mock_get_cache, mock_set_cache, mock_groq):
        from app.common.query_decomposition import decompose_paper

        mock_get_cache.return_value = None
        mock_groq.return_value = {
            "topic": "attention mechanisms",
            "domain": "machine translation",
            "aspect": "self-attention",
            "keywords": ["attention", "transformer"],
            "suggested_specificity": "specific",
        }
        mock_conn = MagicMock()

        result = decompose_paper(
            mock_conn,
            title="Attention Is All You Need",
            abstract="We propose a new architecture...",
            authors=["Vaswani", "Shazeer"],
            year=2017,
            venue="NeurIPS",
        )
        assert result["topic"] == "attention mechanisms"
        assert result["keywords"] == ["attention", "transformer"]

    @patch("app.common.query_decomposition._call_groq")
    @patch("app.common.query_decomposition._cache_decomposition")
    @patch("app.common.query_decomposition._get_cached_decomposition")
    def test_paper_fallback(self, mock_get_cache, mock_set_cache, mock_groq):
        from app.common.query_decomposition import decompose_paper

        mock_get_cache.return_value = None
        mock_groq.return_value = None
        mock_conn = MagicMock()

        result = decompose_paper(mock_conn, title="Some Paper")
        # Fallback uses title as topic
        assert result["topic"] == "Some Paper"
        assert result["keywords"] == []
