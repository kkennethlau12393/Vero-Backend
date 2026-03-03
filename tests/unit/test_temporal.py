"""
Unit tests for temporal_map_service.py — pure logic only.

Tests era labelling, milestone identification, and era grouping
from pre-computed era data (no DB, no LLM).
"""
from __future__ import annotations

import pytest

from app.feature2.temporal_map_service import (
    get_era_label,
    get_era_bounds,
    identify_milestones,
    group_papers_by_era,
)

# ── get_era_label ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetEraLabel:
    def test_decade_mapping(self):
        assert get_era_label(2017) == "2010s"
        assert get_era_label(2020) == "2020s"
        assert get_era_label(1999) == "1990s"
        assert get_era_label(2000) == "2000s"
        assert get_era_label(1985) == "1980s"

    def test_none_year(self):
        assert get_era_label(None) == "Unknown"

    def test_very_old_year(self):
        assert get_era_label(1950) == "1950s"

    def test_boundary_years(self):
        assert get_era_label(2010) == "2010s"
        assert get_era_label(2019) == "2010s"
        assert get_era_label(2009) == "2000s"


# ── get_era_bounds ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestGetEraBounds:
    def test_normal_era(self):
        assert get_era_bounds("2010s") == (2010, 2019)
        assert get_era_bounds("1990s") == (1990, 1999)

    def test_unknown_era(self):
        assert get_era_bounds("Unknown") == (0, 0)

    def test_invalid_era(self):
        assert get_era_bounds("abc") == (0, 0)


# ── identify_milestones ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestIdentifyMilestones:
    def test_top_cited_papers_are_milestones(self):
        papers = [
            {"work_id": "W1", "cited_by_count": 10000, "year": 2015},
            {"work_id": "W2", "cited_by_count": 500, "year": 2015},
            {"work_id": "W3", "cited_by_count": 200, "year": 2015},
            {"work_id": "W4", "cited_by_count": 50, "year": 2015},
        ]
        era_papers = {"2010s": papers}
        milestones = identify_milestones(papers, era_papers, percentile=0.9)
        assert "W1" in milestones

    def test_low_citation_not_milestone(self):
        papers = [
            {"work_id": "W1", "cited_by_count": 50, "year": 2015},
            {"work_id": "W2", "cited_by_count": 40, "year": 2016},
        ]
        era_papers = {"2010s": papers}
        milestones = identify_milestones(papers, era_papers, percentile=0.9)
        # Neither paper has > 100 citations, so no milestones
        assert len(milestones) == 0

    def test_empty_era(self):
        milestones = identify_milestones([], {}, percentile=0.9)
        assert len(milestones) == 0

    def test_single_high_cited_paper(self):
        papers = [{"work_id": "W1", "cited_by_count": 5000, "year": 2010}]
        era_papers = {"2010s": papers}
        milestones = identify_milestones(papers, era_papers, percentile=0.9)
        assert "W1" in milestones

    def test_cross_era_milestones(self):
        papers_2000s = [
            {"work_id": "W1", "cited_by_count": 8000, "year": 2005},
            {"work_id": "W2", "cited_by_count": 200, "year": 2003},
        ]
        papers_2010s = [
            {"work_id": "W3", "cited_by_count": 3000, "year": 2015},
            {"work_id": "W4", "cited_by_count": 100, "year": 2018},
        ]
        all_papers = papers_2000s + papers_2010s
        era_papers = {"2000s": papers_2000s, "2010s": papers_2010s}
        milestones = identify_milestones(all_papers, era_papers, percentile=0.5)
        assert "W1" in milestones
        assert "W3" in milestones


# ── group_papers_by_era ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestGroupPapersByEra:
    def test_basic_grouping(self):
        papers = [
            {"work_id": "W1", "year": 2005, "cited_by_count": 100, "title": "A"},
            {"work_id": "W2", "year": 2015, "cited_by_count": 200, "title": "B"},
            {"work_id": "W3", "year": 2005, "cited_by_count": 300, "title": "C"},
        ]
        result = group_papers_by_era(papers)
        assert "2000s" in result
        assert "2010s" in result
        assert len(result["2000s"]["papers"]) == 2
        assert len(result["2010s"]["papers"]) == 1

    def test_none_year_goes_to_unknown(self):
        papers = [{"work_id": "W1", "year": None, "cited_by_count": 50, "title": "X"}]
        result = group_papers_by_era(papers)
        assert "Unknown" in result

    def test_papers_sorted_by_citations_within_era(self):
        papers = [
            {"work_id": "W1", "year": 2010, "cited_by_count": 100, "title": "A"},
            {"work_id": "W2", "year": 2015, "cited_by_count": 500, "title": "B"},
            {"work_id": "W3", "year": 2012, "cited_by_count": 300, "title": "C"},
        ]
        result = group_papers_by_era(papers)
        era = result["2010s"]["papers"]
        # Should be sorted descending by citation count
        assert era[0]["work_id"] == "W2"
        assert era[1]["work_id"] == "W3"
        assert era[2]["work_id"] == "W1"

    def test_milestone_flag(self):
        papers = [
            {"work_id": "W1", "year": 2015, "cited_by_count": 5000, "title": "Seminal"},
        ]
        milestones = {"W1"}
        result = group_papers_by_era(papers, milestones)
        assert result["2010s"]["papers"][0]["is_milestone"] is True
        assert result["2010s"]["milestone_count"] == 1

    def test_empty_papers(self):
        result = group_papers_by_era([])
        assert len(result) == 0

    def test_era_metadata(self):
        papers = [{"work_id": "W1", "year": 2015, "cited_by_count": 100, "title": "T"}]
        result = group_papers_by_era(papers)
        era = result["2010s"]
        assert era["label"] == "2010s"
        assert era["start_year"] == 2010
        assert era["end_year"] == 2019
        assert era["is_breakthrough_era"] is False


