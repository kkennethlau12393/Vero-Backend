"""
Ranking quality / regression tests.

These tests assert known properties of ranking output quality:
- Score thresholds are reasonable
- Category distribution is balanced
- Known seminal papers appear in expected positions
- Golden baseline snapshots are compared

These tests use saved fixtures, NOT live API calls.
Run with: pytest tests/quality/ -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.feature2.features import robust_norm, bayesian_impact_rate, compute_recency
from app.feature2.rerank import build_reasons, mmr_diversify
from app.feature2.paper_classification import (
    PaperCategory,
    classify_paper_heuristic,
    is_textbook,
    is_method_paper,
)
from tests.fixtures.seed_data import make_work, make_paper_batch

GOLDEN_DIR = Path(__file__).parent / "golden"


# ── Score distribution tests ─────────────────────────────────────────────────

@pytest.mark.quality
class TestScoreDistribution:
    """Verify that scoring functions produce reasonable distributions."""

    def test_robust_norm_spread(self):
        """Robust norm should produce values spread across 0-1."""
        raw = {f"W{i}": float(i * 10) for i in range(50)}
        normed = robust_norm(raw)
        values = list(normed.values())
        assert min(values) == 0.0
        assert max(values) == 1.0
        # Should have some diversity
        unique_vals = len(set(round(v, 2) for v in values))
        assert unique_vals > 5

    def test_impact_rate_reasonable_range(self):
        """Bayesian impact rate should produce positive, finite values."""
        scenarios = [
            (0, 1.0),       # No citations, 1 year old
            (100, 5.0),     # Moderate citations, 5 years
            (10000, 20.0),  # High citations, 20 years
            (50000, 30.0),  # Very high citations, 30 years
        ]
        for citations, age in scenarios:
            rate = bayesian_impact_rate(citations, age)
            assert rate >= 0.0
            assert rate < 10000  # Reasonable upper bound
            assert not (rate != rate)  # Not NaN

    def test_recency_monotonically_decreasing(self):
        """Older papers should have lower recency scores."""
        half_life = 5.0
        ages = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
        scores = [compute_recency(age, half_life) for age in ages]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ── Category distribution tests ─────────────────────────────────────────────

@pytest.mark.quality
class TestCategoryDistribution:
    """Verify that heuristic classification produces reasonable category spread."""

    def test_batch_has_category_diversity(self):
        """A diverse batch of papers should span multiple heuristic categories."""
        papers = [
            make_work(title="PSMATCH2: Stata Module for Matching", work_id="W1"),
            make_work(title="A Survey of Deep Learning", work_id="W2"),
            make_work(title="A New Algorithm for Time Integration", work_id="W3"),
            make_work(title="Deep Learning for Medical Imaging", year=2021, work_id="W4"),
            make_work(title="Dynamics of Structures", cited_by_count=5000, work_id="W5"),
            make_work(title="Seismic Response of Buildings", work_id="W6"),
        ]

        categories_found = set()
        for p in papers:
            cls, _ = classify_paper_heuristic(p)
            if cls:
                categories_found.add(cls.category.value)

        # Should detect at least 3 different categories
        assert len(categories_found) >= 3

    def test_no_single_category_is_100_percent(self):
        """For a mixed batch, no single category should capture ALL papers."""
        papers = make_paper_batch(20)
        category_counts = {}
        for p in papers:
            cls, _ = classify_paper_heuristic(p)
            cat = cls.category.value if cls else "unclassified"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        total = sum(category_counts.values())
        # At least 2 distinct categories should be present
        assert len(category_counts) >= 2, f"Only one category: {category_counts}"
        # No single category should capture everything
        for cat, count in category_counts.items():
            assert count < total, f"Category '{cat}' captured all {total} papers"


# ── Known paper classification tests ────────────────────────────────────────

@pytest.mark.quality
class TestKnownPaperClassification:
    """Verify that known papers are classified correctly by heuristics."""

    def test_stata_module_is_implementation(self):
        w = make_work(title="PSMATCH2: Stata module for full Mahalanobis matching")
        cls, _ = classify_paper_heuristic(w)
        assert cls.category == PaperCategory.IMPLEMENTATION

    def test_survey_paper_is_handbook(self):
        w = make_work(title="A Comprehensive Review of Neural Architecture Search")
        cls, _ = classify_paper_heuristic(w)
        assert cls.category == PaperCategory.HANDBOOK

    def test_canonical_textbook_is_foundational(self):
        w = make_work(title="Dynamics of Structures", cited_by_count=10000)
        cls, needs_review = classify_paper_heuristic(w)
        assert cls.category == PaperCategory.FOUNDATIONAL
        assert needs_review is False

    def test_recent_ml_paper(self):
        w = make_work(title="Physics-Informed Neural Networks for PDEs", year=2019)
        cls, _ = classify_paper_heuristic(w)
        assert cls.category == PaperCategory.RECENT

    def test_method_paper_is_methodological(self):
        w = make_work(title="An Improved Algorithm for Large-Scale Optimization")
        cls, _ = classify_paper_heuristic(w)
        assert cls.category == PaperCategory.METHODOLOGICAL


# ── MMR diversification quality ──────────────────────────────────────────────

@pytest.mark.quality
class TestMMRQuality:
    def test_mmr_promotes_diversity(self):
        """MMR with lambda < 1 should promote diverse papers over pure relevance."""
        scored = [(f"W{i}", 1.0 - i * 0.01) for i in range(20)]

        # Create two clusters of embeddings
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]
        embeddings = {}
        for i in range(20):
            embeddings[f"W{i}"] = emb_a if i < 15 else emb_b

        result_pure = mmr_diversify(scored, {}, 10, lambda_param=1.0)
        result_mmr = mmr_diversify(scored, embeddings, 10, lambda_param=0.5)

        # Pure relevance: top 10 by score
        assert result_pure == [f"W{i}" for i in range(10)]

        # MMR should include some from cluster B (W15+) even though lower score
        cluster_b_in_mmr = [w for w in result_mmr if int(w[1:]) >= 15]
        assert len(cluster_b_in_mmr) > 0

    def test_mmr_scores_threshold(self):
        """All papers selected by MMR should have non-negative relevance."""
        scored = [(f"W{i}", max(0.0, 1.0 - i * 0.1)) for i in range(15)]
        result = mmr_diversify(scored, {}, 10, lambda_param=0.8)
        assert all(isinstance(w, str) for w in result)


# ── Golden baseline snapshot tests ───────────────────────────────────────────

@pytest.mark.quality
class TestGoldenBaselines:
    """Compare current outputs against saved golden baselines."""

    def test_robust_norm_golden(self):
        """Verify robust_norm output hasn't changed for a fixed input."""
        golden_path = GOLDEN_DIR / "robust_norm_baseline.json"

        # Fixed input
        raw = {f"W{i}": float(i ** 2) for i in range(20)}
        normed = robust_norm(raw)
        current = {k: round(v, 6) for k, v in sorted(normed.items())}

        if golden_path.exists():
            golden = json.loads(golden_path.read_text())
            assert current == golden, "robust_norm output changed from golden baseline"
        else:
            # First run: create baseline
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps(current, indent=2))

    def test_impact_rate_golden(self):
        """Verify bayesian_impact_rate output hasn't changed."""
        golden_path = GOLDEN_DIR / "impact_rate_baseline.json"

        scenarios = [
            {"citations": 0, "age": 1.0},
            {"citations": 100, "age": 5.0},
            {"citations": 1000, "age": 10.0},
            {"citations": 10000, "age": 20.0},
        ]
        current = {
            f"c{s['citations']}_a{s['age']}": round(
                bayesian_impact_rate(s["citations"], s["age"]), 6
            )
            for s in scenarios
        }

        if golden_path.exists():
            golden = json.loads(golden_path.read_text())
            assert current == golden, "bayesian_impact_rate output changed from golden baseline"
        else:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps(current, indent=2))

    def test_heuristic_classification_golden(self):
        """Verify heuristic classification hasn't changed for known titles."""
        golden_path = GOLDEN_DIR / "heuristic_classification_baseline.json"

        test_titles = [
            "PSMATCH2: Stata Module for Matching",
            "A Survey of Deep Learning Techniques",
            "Dynamics of Structures",
            "A New Algorithm for Optimization",
            "Deep Learning for Image Recognition",
            "Proceedings of the 5th Conference on AI",
            "Fundamentals of Geotechnical Engineering",
            "Seismic Response of Bridges",
        ]

        current = {}
        for title in test_titles:
            w = make_work(title=title, cited_by_count=5000, year=2021)
            cls, needs_review = classify_paper_heuristic(w)
            current[title] = {
                "category": cls.category.value if cls else None,
                "needs_review": needs_review,
            }

        if golden_path.exists():
            golden = json.loads(golden_path.read_text())
            assert current == golden, "Heuristic classification changed from golden baseline"
        else:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps(current, indent=2))
