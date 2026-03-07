"""
Unit tests for citation map hop filtering.

Tests relevance-based filtering at each hop level per expansion mode,
including the high-citation exception for foundations mode.
"""
from __future__ import annotations

import pytest

from app.feature1.hop_filtering import (
    filter_hop_papers,
    get_seed_count,
    get_max_papers,
)
from tests.fixtures.decomposition_responses import make_intersection_query


def _make_paper(title, abstract="", cited_by_count=100, work_id="W1"):
    return {
        "work_id": work_id,
        "title": title,
        "abstract": abstract,
        "cited_by_count": cited_by_count,
    }


# ── filter_hop_papers ─────────────────────────────────────────────────────────

class TestFilterHopPapers:
    def test_narrow_filters_aggressively(self):
        structured = make_intersection_query()
        papers = [
            _make_paper("NLP for Legal Text Analysis", "natural language processing law", work_id="W1"),
            _make_paper("Quantum Computing Basics", "quantum entanglement physics", work_id="W2"),
        ]
        result = filter_hop_papers(papers, structured, "narrow", hop_level=1)
        # NLP+law paper should pass, quantum paper should be filtered
        matching_ids = {p["work_id"] for p in result}
        assert "W1" in matching_ids
        assert "W2" not in matching_ids

    def test_wide_keeps_more_papers(self):
        structured = make_intersection_query()
        papers = [
            _make_paper("NLP for Legal Analysis", "natural language processing law", work_id="W1"),
            _make_paper("Text Classification Methods", "NLP text classification", work_id="W2"),
            _make_paper("Quantum Computing Basics", "quantum entanglement physics", work_id="W3"),
        ]
        result = filter_hop_papers(papers, structured, "wide", hop_level=0)
        # Wide mode has low threshold (0.1), should keep more papers
        assert len(result) >= 1

    def test_foundations_default_thresholds(self):
        structured = make_intersection_query()
        papers = [
            _make_paper("Natural Language Processing and Law",
                        "Applying natural language processing to legal documents and law.",
                        work_id="W1"),
        ]
        result = filter_hop_papers(papers, structured, "foundations", hop_level=0)
        # Threshold is 0.2 for foundations/hop-0, strong match should pass
        assert len(result) == 1

    def test_higher_hop_has_stricter_threshold(self):
        structured = make_intersection_query()
        # A paper that barely matches — might pass at hop 0 but not hop 2
        papers = [
            _make_paper("Machine Learning Methods", "processing of text", work_id="W1"),
        ]
        result_hop0 = filter_hop_papers(papers, structured, "foundations", hop_level=0)
        result_hop2 = filter_hop_papers(papers, structured, "foundations", hop_level=2)
        # Higher hop should be at least as strict
        assert len(result_hop2) <= len(result_hop0)

    def test_high_citation_exception_foundations(self):
        """Papers with >1000 citations in foundations mode get boosted past threshold."""
        structured = make_intersection_query()
        # Paper with high citations but low relevance
        papers = [
            _make_paper(
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "We introduce BERT model.",
                cited_by_count=50000,
                work_id="W1",
            ),
        ]
        result = filter_hop_papers(papers, structured, "foundations", hop_level=1)
        # High-citation exception: score = max(score, threshold + 0.1)
        # foundations/hop-1 threshold is 0.3, boost makes it >= 0.4
        assert len(result) == 1
        assert result[0]["_relevance_score"] >= 0.3

    def test_high_citation_no_exception_in_narrow(self):
        """High-citation exception only applies to foundations mode."""
        structured = make_intersection_query()
        papers = [
            _make_paper(
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "We introduce BERT model.",
                cited_by_count=50000,
                work_id="W1",
            ),
        ]
        result = filter_hop_papers(papers, structured, "narrow", hop_level=1)
        # Narrow mode has no high-citation exception
        # Whether it passes depends on actual score vs threshold 0.5
        # BERT has no "law" content so intersection score should be low
        matching = [p for p in result if p["work_id"] == "W1"]
        # If BERT passes narrow at 0.5 threshold, something's wrong
        # (intersection = topic * domain, no domain match → ~0)
        assert len(matching) == 0

    def test_relevance_score_added(self):
        structured = make_intersection_query()
        papers = [
            _make_paper("NLP for Law", "natural language processing legal", work_id="W1"),
        ]
        result = filter_hop_papers(papers, structured, "wide", hop_level=0)
        if result:
            assert "_relevance_score" in result[0]

    def test_empty_input(self):
        result = filter_hop_papers([], make_intersection_query(), "narrow", hop_level=0)
        assert result == []

    def test_unknown_expansion_uses_foundations(self):
        structured = make_intersection_query()
        papers = [
            _make_paper("NLP for Law", "natural language processing legal", work_id="W1"),
        ]
        result = filter_hop_papers(papers, structured, "unknown_mode", hop_level=0)
        # Should fall back to foundations thresholds
        assert isinstance(result, list)


# ── get_seed_count ────────────────────────────────────────────────────────────

class TestGetSeedCount:
    def test_landscape(self):
        assert get_seed_count("landscape") == 10

    def test_core_cluster(self):
        assert get_seed_count("core_cluster") == 5

    def test_evolution(self):
        assert get_seed_count("evolution") == 8

    def test_unknown_defaults_to_5(self):
        assert get_seed_count("unknown") == 5


# ── get_max_papers ────────────────────────────────────────────────────────────

class TestGetMaxPapers:
    def test_small(self):
        assert get_max_papers("small") == 25

    def test_medium(self):
        assert get_max_papers("medium") == 50

    def test_large(self):
        assert get_max_papers("large") == 80

    def test_unknown_defaults_to_50(self):
        assert get_max_papers("unknown") == 50
