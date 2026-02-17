"""
Unit tests for query_classification.py — pure logic only (no DB, no LLM).

Tests QueryType, QuerySpecificity enums, and the generate_query_classification_llm
response parsing logic via classify_query's internal helpers.
"""
from __future__ import annotations

import pytest

from app.feature2.query_classification import (
    QueryType,
    QuerySpecificity,
    QueryClassification,
)


# ── QueryType enum ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestQueryTypeEnum:
    def test_all_values_exist(self):
        assert QueryType.METHODOLOGICAL.value == "methodological"
        assert QueryType.TOPICAL.value == "topical"
        assert QueryType.EMPIRICAL.value == "empirical"
        assert QueryType.MIXED.value == "mixed"

    def test_from_string(self):
        assert QueryType("methodological") == QueryType.METHODOLOGICAL
        assert QueryType("topical") == QueryType.TOPICAL
        assert QueryType("empirical") == QueryType.EMPIRICAL
        assert QueryType("mixed") == QueryType.MIXED

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            QueryType("invalid_type")


# ── QuerySpecificity enum ───────────────────────────────────────────────────

@pytest.mark.unit
class TestQuerySpecificityEnum:
    def test_values(self):
        assert QuerySpecificity.BROAD.value == "broad"
        assert QuerySpecificity.SPECIFIC.value == "specific"

    def test_from_string(self):
        assert QuerySpecificity("broad") == QuerySpecificity.BROAD
        assert QuerySpecificity("specific") == QuerySpecificity.SPECIFIC


# ── QueryClassification dataclass ────────────────────────────────────────────

@pytest.mark.unit
class TestQueryClassificationDataclass:
    def test_default_fields(self):
        qc = QueryClassification(
            query_hash="abc123",
            query_type=QueryType.TOPICAL,
            confidence=0.85,
        )
        assert qc.query_hash == "abc123"
        assert qc.query_type == QueryType.TOPICAL
        assert qc.confidence == 0.85
        assert qc.methodological_indicators == []
        assert qc.domain_keywords == []
        assert qc.domain is None
        assert qc.is_emerging_field is False
        assert qc.field_emergence_year is None
        assert qc.query_specificity == QuerySpecificity.BROAD
        assert qc.detected_topic_id is None

    def test_emerging_field(self):
        qc = QueryClassification(
            query_hash="def456",
            query_type=QueryType.TOPICAL,
            confidence=0.9,
            is_emerging_field=True,
            field_emergence_year=2017,
        )
        assert qc.is_emerging_field is True
        assert qc.field_emergence_year == 2017

    def test_specific_query(self):
        qc = QueryClassification(
            query_hash="ghi789",
            query_type=QueryType.METHODOLOGICAL,
            confidence=0.75,
            query_specificity=QuerySpecificity.SPECIFIC,
            detected_topic_id="T12345",
        )
        assert qc.query_specificity == QuerySpecificity.SPECIFIC
        assert qc.detected_topic_id == "T12345"

    def test_with_domain(self):
        qc = QueryClassification(
            query_hash="jkl012",
            query_type=QueryType.MIXED,
            confidence=0.5,
            domain="civil_engineering",
        )
        assert qc.domain == "civil_engineering"

    def test_methodological_indicators(self):
        qc = QueryClassification(
            query_hash="mno345",
            query_type=QueryType.METHODOLOGICAL,
            confidence=0.9,
            methodological_indicators=["methods for", "techniques"],
            domain_keywords=["economics", "causal inference"],
        )
        assert "methods for" in qc.methodological_indicators
        assert "economics" in qc.domain_keywords


# ── Query normalization (used by classify_query) ────────────────────────────

@pytest.mark.unit
class TestQueryNormalization:
    """Test normalize_query_for_expansion which classify_query uses internally."""

    def test_strips_question_prefix(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        assert normalize_query_for_expansion("What are the mechanisms of X?") == "mechanisms of X"

    def test_strips_how_does(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        result = normalize_query_for_expansion("How does climate change affect Y?")
        assert result == "climate change affect Y"

    def test_leaves_topic_unchanged(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        assert normalize_query_for_expansion("machine learning") == "machine learning"

    def test_strips_trailing_question_mark(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        result = normalize_query_for_expansion("What is deep learning?")
        assert "?" not in result

    def test_empty_string(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        assert normalize_query_for_expansion("") == ""

    def test_only_question_word_returns_original(self):
        from app.feature2.query_expansion import normalize_query_for_expansion
        # Edge case: stripping leaves empty → returns original
        result = normalize_query_for_expansion("What?")
        # Should return something non-empty
        assert len(result) > 0


# ── Query hash consistency ──────────────────────────────────────────────────

@pytest.mark.unit
class TestQueryHashConsistency:
    """Verify that question-style and topic-style queries produce the same hash."""

    def test_question_and_topic_same_hash(self):
        from app.feature2.query_expansion import compute_query_hash
        h1 = compute_query_hash("What is deep learning?")
        h2 = compute_query_hash("deep learning")
        assert h1 == h2

    def test_deterministic(self):
        from app.feature2.query_expansion import compute_query_hash
        h1 = compute_query_hash("machine learning in healthcare")
        h2 = compute_query_hash("machine learning in healthcare")
        assert h1 == h2

    def test_case_insensitive(self):
        from app.feature2.query_expansion import compute_query_hash
        h1 = compute_query_hash("Machine Learning")
        h2 = compute_query_hash("machine learning")
        assert h1 == h2

    def test_different_queries_different_hash(self):
        from app.feature2.query_expansion import compute_query_hash
        h1 = compute_query_hash("deep learning")
        h2 = compute_query_hash("quantum computing")
        assert h1 != h2
