"""
Unit tests for focus filtering and depth limiting.

Tests reordering/truncation logic per focus mode. Pure functions,
no DB or network.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.common.focus_filtering import apply_focus_filter, apply_depth_limit, _get_field


def _make_papers(specs, llm_relevance=0.8):
    """Create paper dicts from (title, year, cited_by_count) tuples.

    llm_relevance: default LLM relevance score for all papers (can be
    overridden per-paper by passing 4-tuples instead of 3-tuples).
    """
    papers = []
    for i, spec in enumerate(specs):
        if len(spec) == 4:
            title, year, cites, rel = spec
        else:
            title, year, cites = spec
            rel = llm_relevance
        papers.append({
            "title": title, "year": year, "cited_by_count": cites,
            "work_id": f"W{i}",
            "score_breakdown": {"raw": {"llm_relevance": rel}},
        })
    return papers


# ── apply_focus_filter ────────────────────────────────────────────────────────

class TestApplyFocusFilter:
    def test_foundational_sorts_by_citations(self):
        papers = _make_papers([
            ("Paper A", 2020, 50),
            ("Paper B", 2018, 500),
            ("Paper C", 2015, 200),
        ])
        result = apply_focus_filter(papers, "foundational")
        assert result[0]["title"] == "Paper B"
        assert result[1]["title"] == "Paper C"
        assert result[2]["title"] == "Paper A"

    def test_recent_filters_old_papers(self):
        current_year = datetime.now().year
        # Build 12 recent + 3 old papers to exceed MIN_RESULTS
        specs = [(f"Recent {i}", current_year - i % 5, 50) for i in range(12)]
        specs += [("Old Paper", 2010, 500), ("Very Old", 2000, 1000), ("Ancient", 1990, 2000)]
        papers = _make_papers(specs)
        result = apply_focus_filter(papers, "recent")
        # Old papers (2010, 2000, 1990) should be filtered out
        years = [_get_field(p, "year") for p in result]
        assert all(y >= current_year - 5 for y in years)

    def test_recent_fallback_when_few_papers(self):
        current_year = datetime.now().year
        papers = _make_papers([
            ("Old Paper", 2010, 500),
            ("Recent Paper", current_year - 1, 50),
            ("Very Old", 2000, 1000),
            ("Also Recent", current_year, 10),
        ])
        result = apply_focus_filter(papers, "recent")
        # Fewer than MIN_RESULTS, so fallback returns all papers
        assert len(result) == 4

    def test_surveys_boosts_survey_titles(self):
        papers = _make_papers([
            ("Deep Learning Methods", 2020, 100),
            ("A Survey of NLP Techniques", 2021, 80),
            ("Systematic Review of Transformers", 2022, 60),
            ("BERT for Text Classification", 2020, 120),
        ])
        result = apply_focus_filter(papers, "surveys")
        # Survey/review papers should come first
        assert "survey" in result[0]["title"].lower() or "review" in result[0]["title"].lower()
        assert "survey" in result[1]["title"].lower() or "review" in result[1]["title"].lower()

    def test_all_time_no_change(self):
        papers = _make_papers([
            ("Paper A", 2020, 100),
            ("Paper B", 2018, 200),
        ])
        result = apply_focus_filter(papers, "all_time")
        assert result[0]["title"] == "Paper A"
        assert result[1]["title"] == "Paper B"

    def test_unknown_focus_no_change(self):
        papers = _make_papers([("Paper A", 2020, 100)])
        result = apply_focus_filter(papers, "something_else")
        assert len(result) == 1

    def test_empty_list(self):
        result = apply_focus_filter([], "foundational")
        assert result == []

    def test_foundational_with_preview_subdict(self):
        """Test that focus filtering works with ranked result items (preview subdict)."""
        papers = [
            {"work_id": "W1", "score": 0.9, "preview": {"title": "Paper A", "year": 2020, "cited_by_count": 50}},
            {"work_id": "W2", "score": 0.8, "preview": {"title": "Paper B", "year": 2018, "cited_by_count": 500}},
        ]
        result = apply_focus_filter(papers, "foundational")
        assert result[0]["work_id"] == "W2"  # Higher citations first


# ── apply_depth_limit ─────────────────────────────────────────────────────────

class TestApplyDepthLimit:
    def test_high_level_limits_to_15(self):
        papers = [{"title": f"Paper {i}"} for i in range(50)]
        result = apply_depth_limit(papers, "high_level")
        assert len(result) == 15

    def test_comprehensive_limits_to_50(self):
        papers = [{"title": f"Paper {i}"} for i in range(100)]
        result = apply_depth_limit(papers, "comprehensive")
        assert len(result) == 50

    def test_fewer_than_limit(self):
        papers = [{"title": f"Paper {i}"} for i in range(5)]
        result = apply_depth_limit(papers, "high_level")
        assert len(result) == 5

    def test_unknown_depth_defaults_to_50(self):
        papers = [{"title": f"Paper {i}"} for i in range(100)]
        result = apply_depth_limit(papers, "something")
        assert len(result) == 50

    def test_empty_list(self):
        result = apply_depth_limit([], "high_level")
        assert result == []
