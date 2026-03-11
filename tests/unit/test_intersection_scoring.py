"""
Unit tests for intersection scoring module.

Tests text-based term matching and scope-aware scoring without
any DB or network calls.
"""
from __future__ import annotations

import pytest

from app.common.intersection_scoring import (
    normalize_text,
    term_match_score,
    score_candidate,
    filter_candidates,
)
from tests.fixtures.decomposition_responses import (
    make_candidate_paper,
    make_intersection_query,
    make_broad_query,
    make_specific_query,
)


# ── normalize_text ───────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("  multiple   spaces  ") == "multiple spaces"

    def test_empty(self):
        assert normalize_text("") == ""


# ── term_match_score ─────────────────────────────────────────────────────────

class TestTermMatchScore:
    def test_exact_phrase_match(self):
        assert term_match_score("machine learning", "advances in machine learning") == 1.0

    def test_partial_word_match(self):
        score = term_match_score("natural language processing", "natural language models")
        # 2 out of 3 words match
        assert pytest.approx(score, abs=0.01) == 2.0 / 3.0

    def test_no_match(self):
        assert term_match_score("quantum computing", "natural language processing") == 0.0

    def test_empty_terms(self):
        assert term_match_score("", "some text") == 0.0

    def test_empty_text(self):
        assert term_match_score("some terms", "") == 0.0

    def test_case_insensitive(self):
        assert term_match_score("NLP", "nlp methods for text analysis") == 1.0

    def test_full_overlap(self):
        assert term_match_score("deep learning", "deep learning for images") == 1.0


# ── term_match_score with aliases ────────────────────────────────────────────

class TestTermMatchScoreAliases:
    def test_abbreviation_in_text(self):
        """Full form as terms, abbreviation in text → match via alias."""
        score = term_match_score(
            "natural language processing",
            "NLP methods for text analysis",
            aliases=["NLP", "computational linguistics"],
        )
        assert score == 1.0

    def test_abbreviation_as_terms(self):
        """Abbreviation as terms, full form in text → match via alias."""
        score = term_match_score(
            "NLP",
            "Natural language processing for legal documents",
            aliases=["natural language processing"],
        )
        assert score == 1.0

    def test_no_alias_match(self):
        """Neither terms nor aliases match the text."""
        score = term_match_score(
            "natural language processing",
            "quantum physics experiments",
            aliases=["NLP", "computational linguistics"],
        )
        assert score == 0.0

    def test_empty_aliases(self):
        """Empty alias list → behaves like no aliases."""
        score = term_match_score("NLP", "natural language processing", aliases=[])
        assert score == 0.0

    def test_none_aliases(self):
        """None aliases → backward compatible."""
        score = term_match_score("deep learning", "deep learning models", aliases=None)
        assert score == 1.0

    def test_partial_alias_match(self):
        """Alias has partial word overlap → returns fractional score."""
        score = term_match_score(
            "reinforcement learning",
            "learning algorithms for control",
            aliases=["RL", "reward-based learning"],
        )
        # "RL" doesn't match, "reward-based learning" has 1/2 word overlap ("learning")
        # Original "reinforcement learning" has 1/2 overlap ("learning")
        assert 0.0 < score < 1.0

    def test_best_alias_wins(self):
        """Returns the best score across all forms."""
        score = term_match_score(
            "graph neural networks",
            "GNN for molecular modeling",
            aliases=["GNN", "graph networks", "message passing neural networks"],
        )
        # "GNN" is exact match in text
        assert score == 1.0


# ── score_candidate with aliases ─────────────────────────────────────────────

class TestScoreCandidateWithAliases:
    def test_abbreviation_topic_matches_via_alias(self):
        """Paper says 'NLP' but structured has 'natural language processing' + aliases."""
        paper = make_candidate_paper(
            title="NLP for Legal Document Classification",
            abstract="We use NLP methods to classify legal texts.",
        )
        structured = make_intersection_query()  # topic="natural language processing", aliases=["NLP", ...]
        score = score_candidate(paper, structured, "intersection")
        # "NLP" matches via topic_aliases, "legal" matches via domain_aliases
        assert score > 0.5

    def test_domain_alias_matches(self):
        """Paper says 'legal' but domain is 'law' — should match via alias."""
        paper = make_candidate_paper(
            title="Natural Language Processing for Legal Analysis",
            abstract="Computational linguistics applied to legal reasoning.",
        )
        structured = make_intersection_query()  # domain="law", domain_aliases=["legal", ...]
        score = score_candidate(paper, structured, "intersection")
        assert score > 0.5

    def test_no_aliases_backward_compat(self):
        """Structured dict without alias keys → works like before."""
        paper = make_candidate_paper(
            title="Natural Language Processing for Legal Documents",
            abstract="NLP for law.",
        )
        structured = {"topic": "natural language processing", "domain": "law", "aspect": None}
        score = score_candidate(paper, structured, "intersection")
        assert score > 0.5


# ── score_candidate (intersection scope) ─────────────────────────────────────

class TestScoreCandidateIntersection:
    def test_both_topic_and_domain_match(self):
        paper = make_candidate_paper(
            title="Natural Language Processing for Legal Document Analysis",
            abstract="We apply NLP techniques to law and legal reasoning.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "intersection")
        # Both topic ("natural language processing") and domain ("law") match
        assert score > 0.5

    def test_topic_only_match_low_score(self):
        paper = make_candidate_paper(
            title="Natural Language Processing for Image Captioning",
            abstract="We apply NLP to describe images.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "intersection")
        # Topic matches but domain ("law") does not → multiplicative → near 0
        assert score < 0.1

    def test_domain_only_match_low_score(self):
        paper = make_candidate_paper(
            title="Statistical Methods in Law",
            abstract="We study statistical approaches in legal analysis.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "intersection")
        # Domain matches but topic ("natural language processing") does not → near 0
        assert score < 0.1

    def test_with_aspect_boost(self):
        structured = make_specific_query()
        paper = make_candidate_paper(
            title="BERT Fine-tuning for Medical Named Entity Recognition",
            abstract="We fine-tune BERT models for medical NER tasks.",
        )
        score = score_candidate(paper, structured, "intersection")
        assert score > 0.5


# ── score_candidate (topic_focused scope) ─────────────────────────────────────

class TestScoreCandidateTopicFocused:
    def test_topic_match_high(self):
        paper = make_candidate_paper(
            title="Transformer Architectures: A Survey",
            abstract="We review transformer architectures and their variants.",
        )
        structured = make_broad_query()
        score = score_candidate(paper, structured, "topic_focused")
        # Topic matches strongly, no domain → domain_score=1.0 → high total
        assert score > 0.7

    def test_no_topic_match(self):
        paper = make_candidate_paper(
            title="Reinforcement Learning for Robotics",
            abstract="RL approaches for robot control.",
        )
        structured = make_broad_query()
        score = score_candidate(paper, structured, "topic_focused")
        assert score < 0.3


# ── score_candidate (domain_focused scope) ────────────────────────────────────

class TestScoreCandidateDomainFocused:
    def test_domain_match_high(self):
        paper = make_candidate_paper(
            title="Computational Methods in Law",
            abstract="AI applications in legal text analysis and law.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "domain_focused")
        # Domain weight=0.8, topic weight=0.2
        assert score > 0.5

    def test_topic_only_low(self):
        paper = make_candidate_paper(
            title="Natural Language Processing Survey",
            abstract="Overview of NLP methods.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "domain_focused")
        # Topic matches but weight is only 0.2
        assert score < 0.5


# ── score_candidate (broad scope) ─────────────────────────────────────────────

class TestScoreCandidateBroad:
    def test_average_of_topic_and_domain(self):
        paper = make_candidate_paper(
            title="Natural Language Processing for Legal Analysis",
            abstract="NLP applied to law.",
        )
        structured = make_intersection_query()
        score = score_candidate(paper, structured, "broad")
        # Average of topic and domain scores
        assert score > 0.3

    def test_no_domain_full_score(self):
        paper = make_candidate_paper(
            title="Transformer Architectures Review",
            abstract="We study transformer architectures.",
        )
        structured = make_broad_query()  # domain=None → domain_score=1.0
        score = score_candidate(paper, structured, "broad")
        assert score > 0.5


# ── filter_candidates ─────────────────────────────────────────────────────────

class TestFilterCandidates:
    def test_filters_below_min_score(self):
        papers = [
            make_candidate_paper(
                title="Natural Language Processing for Legal Analysis",
                abstract="We apply natural language processing to law.",
                work_id="W1",
            ),
            make_candidate_paper(title="Unrelated Physics", abstract="Quantum physics.", work_id="W2"),
        ]
        structured = make_intersection_query()
        result = filter_candidates(papers, structured, "intersection", min_score=0.2)
        # Only the NLP+law paper should pass
        assert len(result) == 1
        assert result[0]["work_id"] == "W1"

    def test_adds_relevance_score(self):
        papers = [
            make_candidate_paper(title="Natural Language Processing for Legal Documents", abstract="NLP and law.", work_id="W1"),
        ]
        structured = make_intersection_query()
        result = filter_candidates(papers, structured, "intersection", min_score=0.0)
        assert "_relevance_score" in result[0]

    def test_sorted_by_score_descending(self):
        papers = [
            make_candidate_paper(title="Law review", abstract="Legal analysis.", work_id="W1"),
            make_candidate_paper(title="NLP for Legal Analysis and Law", abstract="Natural language processing for law.", work_id="W2"),
        ]
        structured = make_intersection_query()
        result = filter_candidates(papers, structured, "intersection", min_score=0.0)
        if len(result) >= 2:
            assert result[0]["_relevance_score"] >= result[1]["_relevance_score"]

    def test_empty_input(self):
        result = filter_candidates([], make_intersection_query(), "intersection")
        assert result == []
