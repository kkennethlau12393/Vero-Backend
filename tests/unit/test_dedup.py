"""
Unit tests for title deduplication and similarity functions.

Tests real functions from app.feature2.tfidf_similarity.
"""
from __future__ import annotations

import pytest

from app.feature2.tfidf_similarity import (
    title_word_set,
    jaccard_similarity,
    dedupe_by_title_similarity,
    tokenize_text,
    compute_tf,
    compute_idf,
    cosine_similarity_tfidf,
)


# ── title_word_set ───────────────────────────────────────────────────────────

class TestTitleWordSet:
    def test_basic(self):
        words = title_word_set("Deep Learning for Natural Language Processing")
        assert "deep" in words
        assert "learning" in words
        assert "natural" in words
        # "for" is a stopword
        assert "for" not in words

    def test_empty_string(self):
        assert title_word_set("") == set()

    def test_none(self):
        assert title_word_set(None) == set()

    def test_short_words_excluded(self):
        words = title_word_set("A Go To NLP")
        # "go", "to" are ≤2 chars or stopwords
        assert "go" not in words
        assert "nlp" in words

    def test_punctuation_removed(self):
        words = title_word_set("Self-Supervised Learning: A Survey")
        # Hyphen is removed, so "Self-Supervised" → "selfsupervised"
        assert "selfsupervised" in words
        assert "survey" in words
        assert "learning" in words


# ── jaccard_similarity ───────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"deep", "learning"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        assert sim == pytest.approx(0.5)

    def test_empty_set(self):
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity(set(), set()) == 0.0


# ── dedupe_by_title_similarity ───────────────────────────────────────────────

class TestDedupeByTitleSimilarity:
    def test_exact_duplicates_removed(self):
        scored = [("W1", 1.0), ("W2", 0.9)]
        titles = {
            "W1": "Attention Is All You Need",
            "W2": "Attention Is All You Need",
        }
        result = dedupe_by_title_similarity(scored, titles, threshold=0.5)
        assert len(result) == 1
        assert result[0][0] == "W1"  # higher score kept

    def test_different_titles_kept(self):
        scored = [("W1", 1.0), ("W2", 0.9)]
        titles = {
            "W1": "Attention Is All You Need",
            "W2": "ImageNet Classification with Deep Convolutional Neural Networks",
        }
        result = dedupe_by_title_similarity(scored, titles, threshold=0.5)
        assert len(result) == 2

    def test_empty_title_kept(self):
        scored = [("W1", 1.0), ("W2", 0.9)]
        titles = {"W1": "", "W2": "Some Paper Title"}
        result = dedupe_by_title_similarity(scored, titles, threshold=0.5)
        assert len(result) == 2

    def test_near_duplicates(self):
        scored = [("W1", 1.0), ("W2", 0.8)]
        titles = {
            "W1": "Deep Residual Learning for Image Recognition",
            "W2": "Deep Residual Learning for Image Classification",
        }
        result = dedupe_by_title_similarity(scored, titles, threshold=0.5)
        # High overlap → should be deduped
        assert len(result) == 1

    def test_no_candidates(self):
        assert dedupe_by_title_similarity([], {}, 0.5) == []


# ── tokenize_text ────────────────────────────────────────────────────────────

class TestTokenizeText:
    def test_basic(self):
        tokens = tokenize_text("Deep learning models for NLP tasks")
        assert "deep" in tokens
        assert "learning" in tokens
        # "for" is a stop word
        assert "for" not in tokens

    def test_empty(self):
        assert tokenize_text("") == []
        assert tokenize_text(None) == []

    def test_short_words_removed(self):
        tokens = tokenize_text("An AI is great")
        assert "an" not in tokens  # stopword
        assert "ai" not in tokens  # len <= 2


# ── compute_tf ───────────────────────────────────────────────────────────────

class TestComputeTF:
    def test_uniform(self):
        tf = compute_tf(["a", "b", "c"])
        assert tf["a"] == pytest.approx(1 / 3)

    def test_repeated_term(self):
        tf = compute_tf(["deep", "deep", "learning"])
        assert tf["deep"] == pytest.approx(2 / 3)

    def test_empty(self):
        assert compute_tf([]) == {}


# ── compute_idf ──────────────────────────────────────────────────────────────

class TestComputeIDF:
    def test_single_doc(self):
        idf = compute_idf([["deep", "learning"]])
        # idf = log(2/2) + 1 = 1.0
        assert idf["deep"] == pytest.approx(1.0)

    def test_rare_term_higher_idf(self):
        docs = [["deep", "learning"], ["deep", "networks"], ["attention", "networks"]]
        idf = compute_idf(docs)
        # "attention" appears in 1 doc, "deep" in 2 → attention has higher idf
        assert idf["attention"] > idf["deep"]


# ── cosine_similarity_tfidf ─────────────────────────────────────────────────

class TestCosineSimilarityTFIDF:
    def test_identical_vectors(self):
        vec = {"deep": 0.5, "learning": 0.3}
        assert cosine_similarity_tfidf(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity_tfidf({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_empty_vector(self):
        assert cosine_similarity_tfidf({}, {"a": 1.0}) == 0.0
