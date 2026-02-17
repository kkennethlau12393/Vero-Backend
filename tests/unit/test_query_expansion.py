"""
Unit tests for query_expansion.py — pure functions only (no LLM, no DB).

Tests normalization, hashing, tokenization, concept parsing, and
legacy term building.
"""
from __future__ import annotations

import pytest

from app.feature2.query_expansion import (
    normalize_query_for_expansion,
    compute_query_hash,
    _tokenize_query,
    _parse_concepts,
    _build_legacy_expansion_terms,
    ExpandedConcept,
    QueryExpansion,
    QUESTION_PREFIXES,
)


# ── normalize_query_for_expansion ────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizeQuery:
    def test_what_are(self):
        assert normalize_query_for_expansion("What are the effects of X?") == "effects of X"

    def test_how_does(self):
        assert normalize_query_for_expansion("How does gravity work?") == "gravity work"

    def test_where_is(self):
        result = normalize_query_for_expansion("Where is the evidence for Y?")
        assert "evidence" in result

    def test_plain_topic_unchanged(self):
        assert normalize_query_for_expansion("climate change adaptation") == "climate change adaptation"

    def test_trailing_question_mark_removed(self):
        result = normalize_query_for_expansion("deep learning?")
        assert result == "deep learning"

    def test_empty_string(self):
        assert normalize_query_for_expansion("") == ""

    def test_whitespace_only(self):
        result = normalize_query_for_expansion("   ")
        assert result.strip() == ""

    def test_preserves_important_content(self):
        result = normalize_query_for_expansion("What is the impact of CRISPR on gene therapy?")
        assert "CRISPR" in result
        assert "gene therapy" in result

    def test_does_not_strip_mid_sentence(self):
        """Only strips from the beginning."""
        result = normalize_query_for_expansion("the impact of what we know")
        # "the" at start should not be stripped (it's not a question prefix)
        assert "impact" in result

    def test_case_insensitive_stripping(self):
        result = normalize_query_for_expansion("WHAT IS deep learning?")
        assert "deep learning" in result


# ── compute_query_hash ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestComputeQueryHash:
    def test_is_sha256(self):
        h = compute_query_hash("test query")
        assert len(h) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        h1 = compute_query_hash("transformer architecture")
        h2 = compute_query_hash("transformer architecture")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_query_hash("Deep Learning")
        h2 = compute_query_hash("deep learning")
        assert h1 == h2

    def test_question_normalized(self):
        h1 = compute_query_hash("What is deep learning?")
        h2 = compute_query_hash("deep learning")
        assert h1 == h2

    def test_different_queries_different_hash(self):
        h1 = compute_query_hash("neural networks")
        h2 = compute_query_hash("quantum computing")
        assert h1 != h2


# ── _tokenize_query ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTokenizeQuery:
    def test_basic_tokenization(self):
        tokens = _tokenize_query("deep learning for NLP")
        assert "deep" in tokens
        assert "learning" in tokens
        assert "nlp" in tokens

    def test_punctuation_removed(self):
        tokens = _tokenize_query("self-supervised learning: a survey")
        # Punctuation should be replaced with spaces
        assert "survey" in tokens

    def test_lowercase(self):
        tokens = _tokenize_query("BERT Fine-Tuning")
        assert all(t == t.lower() for t in tokens)

    def test_empty(self):
        assert _tokenize_query("") == []

    def test_whitespace_only(self):
        assert _tokenize_query("   ") == []


# ── _parse_concepts ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestParseConcepts:
    def test_basic_parsing(self):
        data = {
            "concepts": [
                {
                    "term": "deep learning",
                    "importance": 0.9,
                    "synonyms": ["DL", "neural networks"],
                    "related": ["machine learning"],
                    "foundational_works": ["AlexNet"],
                },
            ]
        }
        concepts = _parse_concepts(data)
        assert len(concepts) == 1
        assert concepts[0].term == "deep learning"
        assert concepts[0].importance == 0.9
        assert "DL" in concepts[0].synonyms
        assert "machine learning" in concepts[0].related_terms
        assert "AlexNet" in concepts[0].foundational_works

    def test_empty_concepts(self):
        assert _parse_concepts({}) == []
        assert _parse_concepts({"concepts": []}) == []

    def test_missing_term_skipped(self):
        data = {"concepts": [{"importance": 0.5}]}
        assert _parse_concepts(data) == []

    def test_non_dict_entry_skipped(self):
        data = {"concepts": ["invalid", {"term": "valid", "importance": 0.5}]}
        concepts = _parse_concepts(data)
        assert len(concepts) == 1
        assert concepts[0].term == "valid"

    def test_default_importance(self):
        data = {"concepts": [{"term": "test"}]}
        concepts = _parse_concepts(data)
        assert concepts[0].importance == 0.5

    def test_whitespace_stripped(self):
        data = {"concepts": [{"term": "  deep learning  ", "synonyms": [" DL "]}]}
        concepts = _parse_concepts(data)
        assert concepts[0].term == "deep learning"
        assert concepts[0].synonyms[0] == "DL"


# ── _build_legacy_expansion_terms ────────────────────────────────────────────

@pytest.mark.unit
class TestBuildLegacyExpansionTerms:
    def test_basic(self):
        concepts = [
            ExpandedConcept(
                term="diffusion",
                importance=0.9,
                synonyms=["denoising"],
                related_terms=["generative"],
                foundational_works=["DDPM"],
            ),
        ]
        terms = _build_legacy_expansion_terms(concepts)
        # Foundational works first, then synonyms, then related
        assert "DDPM" in terms
        assert "denoising" in terms
        assert "generative" in terms

    def test_deduplication(self):
        concepts = [
            ExpandedConcept(term="a", importance=0.9, synonyms=["shared"]),
            ExpandedConcept(term="b", importance=0.8, synonyms=["shared"]),
        ]
        terms = _build_legacy_expansion_terms(concepts)
        # "shared" should appear only once
        assert terms.count("shared") == 1

    def test_case_insensitive_dedup(self):
        concepts = [
            ExpandedConcept(term="a", importance=0.9, synonyms=["BERT"]),
            ExpandedConcept(term="b", importance=0.8, foundational_works=["bert"]),
        ]
        terms = _build_legacy_expansion_terms(concepts)
        lower_terms = [t.lower() for t in terms]
        assert lower_terms.count("bert") == 1

    def test_empty(self):
        assert _build_legacy_expansion_terms([]) == []

    def test_foundational_works_first(self):
        concepts = [
            ExpandedConcept(
                term="transformers",
                importance=0.9,
                synonyms=["attention models"],
                related_terms=["sequence models"],
                foundational_works=["Attention Is All You Need"],
            ),
        ]
        terms = _build_legacy_expansion_terms(concepts)
        # Foundational works should come before synonyms
        fw_idx = terms.index("Attention Is All You Need")
        syn_idx = terms.index("attention models")
        assert fw_idx < syn_idx


# ── QUESTION_PREFIXES regex ──────────────────────────────────────────────────

@pytest.mark.unit
class TestQuestionPrefixesRegex:
    @pytest.mark.parametrize("query,expected_stripped", [
        ("What is deep learning", "deep learning"),
        ("How does BERT work", "BERT work"),
        ("Why are transformers effective", "transformers effective"),
        ("Which methods are best", "methods are best"),
        ("Can we use CRISPR safely", "we use CRISPR safely"),
        ("Does temperature affect results", "temperature affect results"),
    ])
    def test_various_question_words(self, query, expected_stripped):
        result = QUESTION_PREFIXES.sub("", query.strip()).rstrip("?").strip()
        assert result == expected_stripped

    def test_no_match_for_non_question(self):
        text = "climate change effects on agriculture"
        result = QUESTION_PREFIXES.sub("", text)
        assert result == text
