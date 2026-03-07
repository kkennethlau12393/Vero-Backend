"""
Unit tests for focus filtering and depth limiting.

Tests reordering/truncation logic per focus mode. Pure functions,
no DB or network.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.common.focus_filtering import apply_focus_filter, apply_depth_limit


def _make_papers(specs):
    """Create paper dicts from (title, year, cited_by_count) tuples."""
    return [
        {"title": title, "year": year, "cited_by_count": cites, "work_id": f"W{i}"}
        for i, (title, year, cites) in enumerate(specs)
    ]


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

    def test_recent_puts_recent_first(self):
        current_year = datetime.now().year
        papers = _make_papers([
            ("Old Paper", 2010, 500),
            ("Recent Paper", current_year - 1, 50),
            ("Very Old", 2000, 1000),
            ("Also Recent", current_year, 10),
        ])
        result = apply_focus_filter(papers, "recent")
        # Recent papers (within 3 years) come first
        assert result[0]["title"] == "Recent Paper"
        assert result[1]["title"] == "Also Recent"
        # Older papers follow
        assert result[2]["title"] == "Old Paper"

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
    def test_high_level_limits_to_20(self):
        papers = [{"title": f"Paper {i}"} for i in range(50)]
        result = apply_depth_limit(papers, "high_level")
        assert len(result) == 20

    def test_comprehensive_limits_to_60(self):
        papers = [{"title": f"Paper {i}"} for i in range(100)]
        result = apply_depth_limit(papers, "comprehensive")
        assert len(result) == 60

    def test_fewer_than_limit(self):
        papers = [{"title": f"Paper {i}"} for i in range(5)]
        result = apply_depth_limit(papers, "high_level")
        assert len(result) == 5

    def test_unknown_depth_defaults_to_60(self):
        papers = [{"title": f"Paper {i}"} for i in range(100)]
        result = apply_depth_limit(papers, "something")
        assert len(result) == 60

    def test_empty_list(self):
        result = apply_depth_limit([], "high_level")
        assert result == []
