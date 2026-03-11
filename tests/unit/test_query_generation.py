"""
Unit tests for multi-query generation module.

Tests query generation per scope and map_focus mode.
Pure functions, no DB or network.
"""
from __future__ import annotations

import pytest

from app.common.query_generation import (
    generate_retrieval_queries,
    generate_citation_map_queries,
)
from tests.fixtures.decomposition_responses import (
    make_intersection_query,
    make_broad_query,
    make_specific_query,
)


# ── generate_retrieval_queries ────────────────────────────────────────────────

class TestGenerateRetrievalQueries:
    def test_intersection_scope(self):
        structured = make_intersection_query()
        queries = generate_retrieval_queries(structured, "intersection")
        assert len(queries) >= 3
        # Should contain combined queries
        assert any("natural language processing" in q.lower() and "law" in q.lower() for q in queries)

    def test_intersection_with_aspect(self):
        structured = make_specific_query()
        queries = generate_retrieval_queries(structured, "intersection")
        # With aspect, should have 4 queries
        assert len(queries) >= 4
        assert any("named entity recognition" in q.lower() for q in queries)

    def test_broad_scope(self):
        structured = make_intersection_query()
        queries = generate_retrieval_queries(structured, "broad")
        assert len(queries) >= 2
        # Should include topic-only query
        assert any(q.lower() == "natural language processing" for q in queries)

    def test_topic_focused_scope(self):
        structured = make_broad_query()
        queries = generate_retrieval_queries(structured, "topic_focused")
        assert len(queries) >= 2
        assert any("transformer architectures" in q.lower() for q in queries)

    def test_topic_focused_with_domain(self):
        structured = make_intersection_query()
        queries = generate_retrieval_queries(structured, "topic_focused")
        assert len(queries) == 3  # topic, recent advances, topic+domain

    def test_domain_focused_scope(self):
        structured = make_intersection_query()
        queries = generate_retrieval_queries(structured, "domain_focused")
        assert len(queries) >= 3
        assert any("law" in q.lower() for q in queries)

    def test_no_domain_fallback(self):
        structured = make_broad_query()  # domain=None
        queries = generate_retrieval_queries(structured, "intersection")
        # Falls through to else branch since domain=None
        assert len(queries) >= 1
        assert queries[0].lower() == "transformer architectures"

    def test_unknown_scope_fallback(self):
        structured = make_intersection_query()
        queries = generate_retrieval_queries(structured, "unknown_scope")
        assert len(queries) >= 1


# ── generate_citation_map_queries ─────────────────────────────────────────────

class TestGenerateCitationMapQueries:
    def test_core_cluster(self):
        structured = make_intersection_query()
        queries = generate_citation_map_queries(structured, "core_cluster")
        assert len(queries) >= 2
        assert any("law" in q.lower() for q in queries)

    def test_landscape(self):
        structured = make_intersection_query()
        queries = generate_citation_map_queries(structured, "landscape")
        assert len(queries) >= 3
        # Landscape should include topic-only and domain-only queries
        assert any(q.lower() == "natural language processing" for q in queries)
        assert any(q.lower() == "law" for q in queries)

    def test_evolution(self):
        structured = make_intersection_query()
        queries = generate_citation_map_queries(structured, "evolution")
        assert len(queries) >= 3
        assert any("survey" in q.lower() or "history" in q.lower() for q in queries)

    def test_no_domain_fallback(self):
        structured = make_broad_query()  # domain=None
        queries = generate_citation_map_queries(structured, "core_cluster")
        assert len(queries) >= 1
        assert queries[0].lower() == "transformer architectures"
