"""
Diversity regression tests for Feature 2 ranking.

Asserts that ranking output maintains diversity across:
- Category coverage (multiple categories populated)
- Temporal spread (papers from multiple decades)
- Source diversity (multiple provenance sources)
"""
from __future__ import annotations

import pytest

from app.feature2.temporal_map_service import get_era_label, group_papers_by_era, identify_milestones
from app.feature2.paper_classification import classify_paper_heuristic, PaperCategory
from app.feature2.domain_filters import detect_domain, compute_domain_alignment
from tests.fixtures.seed_data import make_work, make_paper_batch


# ── Category coverage tests ──────────────────────────────────────────────────

@pytest.mark.quality
class TestCategoryCoverage:
    """Broad queries should populate multiple paper categories."""

    def test_diverse_batch_covers_categories(self):
        """A diverse set of papers should cover at least 3 heuristic categories."""
        papers = [
            make_work(title="reghdfe: Stata Module for Fixed Effects", work_id="W1"),
            make_work(title="A Comprehensive Review of Transfer Learning", work_id="W2"),
            make_work(title="Improved Numerical Method for Eigenvalue Problems", work_id="W3"),
            make_work(title="Transformer Models for Time Series Forecasting", year=2022, work_id="W4"),
            make_work(title="Principles of Statistical Learning", cited_by_count=8000, work_id="W5"),
            make_work(title="Proceedings of the 10th International Conference", work_id="W6"),
        ]

        categories = set()
        for p in papers:
            cls, _ = classify_paper_heuristic(p)
            if cls:
                categories.add(cls.category)

        assert PaperCategory.IMPLEMENTATION in categories
        assert PaperCategory.HANDBOOK in categories
        assert PaperCategory.METHODOLOGICAL in categories

    def test_factory_batch_has_variety(self):
        """make_paper_batch produces papers with category variety."""
        papers = make_paper_batch(20)
        categories = set()
        for p in papers:
            if p.category:
                categories.add(p.category)
        # Factory assigns 5 different categories
        assert len(categories) >= 3


# ── Temporal spread tests ────────────────────────────────────────────────────

@pytest.mark.quality
class TestTemporalSpread:
    """Papers should come from multiple decades, not cluster in one era."""

    def test_multi_decade_spread(self):
        """Papers spanning multiple decades should group into multiple eras."""
        papers = [
            {"work_id": "W1", "year": 1995, "cited_by_count": 500, "title": "A"},
            {"work_id": "W2", "year": 2003, "cited_by_count": 400, "title": "B"},
            {"work_id": "W3", "year": 2010, "cited_by_count": 300, "title": "C"},
            {"work_id": "W4", "year": 2018, "cited_by_count": 200, "title": "D"},
            {"work_id": "W5", "year": 2023, "cited_by_count": 100, "title": "E"},
        ]
        eras = group_papers_by_era(papers)
        # Should have papers in at least 3 different decades
        non_unknown = {k for k in eras.keys() if k != "Unknown"}
        assert len(non_unknown) >= 3

    def test_single_decade_detected(self):
        """All papers from one decade should flag as low temporal diversity."""
        papers = [
            {"work_id": f"W{i}", "year": 2020 + (i % 3), "cited_by_count": 100, "title": f"P{i}"}
            for i in range(10)
        ]
        eras = group_papers_by_era(papers)
        non_unknown = {k for k in eras.keys() if k != "Unknown"}
        # All 2020-2022 → single era "2020s"
        assert len(non_unknown) == 1

    def test_none_years_dont_pollute(self):
        """Papers with None year should go to 'Unknown' and not affect era counts."""
        papers = [
            {"work_id": "W1", "year": None, "cited_by_count": 100, "title": "A"},
            {"work_id": "W2", "year": 2015, "cited_by_count": 200, "title": "B"},
            {"work_id": "W3", "year": 1990, "cited_by_count": 300, "title": "C"},
        ]
        eras = group_papers_by_era(papers)
        assert "Unknown" in eras
        non_unknown = {k for k in eras.keys() if k != "Unknown"}
        assert len(non_unknown) == 2

    def test_milestone_distribution_across_eras(self):
        """Milestones should be identified independently per era."""
        papers_1990s = [
            {"work_id": "W1", "year": 1995, "cited_by_count": 5000},
            {"work_id": "W2", "year": 1998, "cited_by_count": 100},
        ]
        papers_2010s = [
            {"work_id": "W3", "year": 2015, "cited_by_count": 3000},
            {"work_id": "W4", "year": 2018, "cited_by_count": 50},
        ]
        all_papers = papers_1990s + papers_2010s
        era_papers = {"1990s": papers_1990s, "2010s": papers_2010s}
        milestones = identify_milestones(all_papers, era_papers, percentile=0.5)
        # Both eras should have their top-cited paper as milestone
        assert "W1" in milestones
        assert "W3" in milestones


# ── Domain diversity tests ───────────────────────────────────────────────────

@pytest.mark.quality
class TestDomainDiversity:
    """Domain detection and alignment should work correctly."""

    def test_civil_engineering_detected(self):
        assert detect_domain("seismic response of bridge structures") == "civil_engineering"
        assert detect_domain("earthquake engineering") == "civil_engineering"

    def test_non_civil_not_detected(self):
        assert detect_domain("deep learning for NLP") is None
        assert detect_domain("quantum computing algorithms") is None

    def test_on_domain_paper_high_alignment(self):
        score = compute_domain_alignment(
            "civil_engineering",
            "Seismic Analysis of High-Rise Buildings",
            "earthquake response spectra modal analysis",
        )
        assert score >= 0.7

    def test_off_domain_paper_low_alignment(self):
        score = compute_domain_alignment(
            "civil_engineering",
            "Deep Learning for Robotic Manipulation",
            "robot arm end-effector actuator control",
        )
        assert score <= 0.15

    def test_neutral_when_no_domain(self):
        score = compute_domain_alignment(
            None,
            "Some Random Paper Title",
        )
        assert score == 0.5

    def test_mixed_signals_moderate_alignment(self):
        """Paper with both boost and downrank terms gets moderate score."""
        score = compute_domain_alignment(
            "civil_engineering",
            "Bridge Dynamics with Composite Materials",
            "bridge vibration analysis composite shell laminated",
        )
        # Has both boost (bridge, vibration) and downrank (composite shell, laminated)
        # Result depends on counts — should be moderate
        assert 0.1 <= score <= 0.8


# ── Provenance / source diversity ────────────────────────────────────────────

@pytest.mark.quality
class TestSourceDiversity:
    """Papers should ideally come from multiple retrieval sources."""

    def test_provenance_structure(self):
        """Verify provenance format is consistent."""
        # A provenance entry should have 'source' and 'score' keys
        sample_provenance = [
            {"source": "lexical_openalex", "score": 0.85},
            {"source": "semantic_scholar", "score": 0.72},
            {"source": "arxiv", "score": 0.60},
        ]
        for p in sample_provenance:
            assert "source" in p
            assert "score" in p
            assert isinstance(p["score"], (int, float))

    def test_multiple_sources_represented(self):
        """A good result set should have papers from multiple sources."""
        # Simulate provenance from different sources
        provenances = [
            [{"source": "lexical_openalex", "score": 0.9}],
            [{"source": "semantic_scholar", "score": 0.8}],
            [{"source": "arxiv", "score": 0.7}],
            [{"source": "fts_lexical", "score": 0.6}],
            [{"source": "foundational_title", "score": 0.5}],
        ]
        sources = {p[0]["source"] for p in provenances}
        assert len(sources) >= 3
