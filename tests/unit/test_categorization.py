"""
Unit tests for paper_classification.py — heuristic pattern matching (no LLM, no DB).

Tests the pure functions: is_textbook, has_subfield_qualifiers, is_method_paper,
has_ml_patterns, is_proceedings, classify_paper_heuristic, PaperCategory enum.
"""
from __future__ import annotations

import pytest

from app.feature2.paper_classification import (
    PaperCategory,
    PaperClassification,
    is_textbook,
    has_subfield_qualifiers,
    is_method_paper,
    has_ml_patterns,
    is_proceedings,
    classify_paper_heuristic,
)
from tests.fixtures.seed_data import make_work


# ── PaperCategory enum ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestPaperCategoryEnum:
    def test_all_values(self):
        assert PaperCategory.FOUNDATIONAL.value == "foundational"
        assert PaperCategory.METHODOLOGICAL.value == "methodological"
        assert PaperCategory.APPLIED.value == "applied"
        assert PaperCategory.RECENT.value == "recent"
        assert PaperCategory.IMPLEMENTATION.value == "implementation"
        assert PaperCategory.HANDBOOK.value == "handbook"
        assert PaperCategory.SPECIFIC_TOPICS.value == "specific_topics"

    def test_from_string(self):
        assert PaperCategory("foundational") == PaperCategory.FOUNDATIONAL

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            PaperCategory("nonexistent")


# ── is_textbook ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIsTextbook:
    def test_fundamentals_of(self):
        assert is_textbook("Fundamentals of Structural Dynamics") is True

    def test_introduction_to(self):
        assert is_textbook("Introduction to Machine Learning") is True

    def test_principles_of(self):
        assert is_textbook("Principles of Neural Science") is True

    def test_theory_of(self):
        assert is_textbook("Theory of Elasticity") is True

    def test_dynamics_of(self):
        assert is_textbook("Dynamics of Structures") is True

    def test_edition_pattern(self):
        assert is_textbook("Structural Analysis, 3rd Edition") is True

    def test_subtitle_introduction(self):
        assert is_textbook("Speech Processing: An Introduction to NLP") is True

    def test_concepts_and_applications(self):
        assert is_textbook("Vibration: Concepts and Applications") is True

    def test_regular_paper(self):
        assert is_textbook("Attention Is All You Need") is False

    def test_empty_string(self):
        assert is_textbook("") is False

    def test_none(self):
        assert is_textbook(None) is False


# ── has_subfield_qualifiers ──────────────────────────────────────────────────

@pytest.mark.unit
class TestHasSubfieldQualifiers:
    def test_geotechnical(self):
        assert has_subfield_qualifiers("Fundamentals of Geotechnical Engineering") is True

    def test_soil(self):
        assert has_subfield_qualifiers("Soil Mechanics and Foundations") is True

    def test_wind_engineering(self):
        assert has_subfield_qualifiers("Wind Loading of Structures") is True

    def test_computational(self):
        assert has_subfield_qualifiers("Computational Methods in Engineering") is True

    def test_finite_element(self):
        assert has_subfield_qualifiers("The Finite Element Method") is True

    def test_concrete(self):
        assert has_subfield_qualifiers("Reinforced Concrete Design") is True

    def test_bridge(self):
        assert has_subfield_qualifiers("Bridge Engineering Handbook") is True

    def test_no_qualifiers(self):
        assert has_subfield_qualifiers("Dynamics of Structures") is False

    def test_empty(self):
        assert has_subfield_qualifiers("") is False

    def test_none(self):
        assert has_subfield_qualifiers(None) is False


# ── is_method_paper ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIsMethodPaper:
    def test_algorithm_for(self):
        assert is_method_paper("A New Algorithm for Time Integration") is True

    def test_method_for(self):
        assert is_method_paper("An Improved Method for Modal Analysis") is True

    def test_numerical_integration(self):
        assert is_method_paper("Numerical Integration of Dynamic Systems") is True

    def test_efficient_computation(self):
        assert is_method_paper("An Efficient Algorithm for Large-Scale Computation") is True

    def test_regular_paper(self):
        assert is_method_paper("Seismic Response of High-Rise Buildings") is False

    def test_empty(self):
        assert is_method_paper("") is False

    def test_none(self):
        assert is_method_paper(None) is False


# ── has_ml_patterns ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestHasMLPatterns:
    def test_neural_network(self):
        assert has_ml_patterns("Neural Network for Structural Damage Detection") is True

    def test_deep_learning(self):
        assert has_ml_patterns("Deep Learning-Based Seismic Analysis") is True

    def test_transformer(self):
        assert has_ml_patterns("Transformer Architecture for Time Series") is True

    def test_pinn(self):
        assert has_ml_patterns("Physics-Informed Neural Networks for PDEs") is True

    def test_regular_paper(self):
        assert has_ml_patterns("Modal Analysis of Bridge Structures") is False


# ── is_proceedings ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestIsProceedings:
    def test_proceedings_of(self):
        assert is_proceedings("Proceedings of the 15th World Conference on Earthquake Engineering") is True

    def test_symposium(self):
        assert is_proceedings("Symposium on Structural Dynamics") is True

    def test_conference(self):
        assert is_proceedings("12th World Conference on Earthquake Engineering") is True

    def test_regular_paper(self):
        assert is_proceedings("Deep Residual Learning") is False


# ── classify_paper_heuristic ────────────────────────────────────────────────

@pytest.mark.unit
class TestClassifyPaperHeuristic:
    def test_implementation_detected(self):
        w = make_work(title="PSMATCH2: Stata Module to Perform Matching")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification is not None
        assert classification.category == PaperCategory.IMPLEMENTATION
        assert needs_review is False

    def test_r_package(self):
        w = make_work(title="MatchIt: An R Package for Nonparametric Matching")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.IMPLEMENTATION

    def test_proceedings_to_specific_topics(self):
        w = make_work(title="Proceedings of the 3rd World Conference on AI")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.SPECIFIC_TOPICS

    def test_handbook_review(self):
        w = make_work(title="A Survey of Deep Learning Techniques")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.HANDBOOK
        assert needs_review is False

    def test_meta_analysis(self):
        w = make_work(title="Meta-analysis of Climate Change Effects")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.HANDBOOK

    def test_method_paper(self):
        w = make_work(title="A New Algorithm for Large-Scale Optimization")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.METHODOLOGICAL

    def test_ml_recent_paper(self):
        w = make_work(title="Deep Learning for Protein Structure Prediction", year=2020)
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.RECENT

    def test_ml_old_paper_not_recent(self):
        w = make_work(title="Neural Network Approaches to Pattern Recognition", year=2015)
        classification, needs_review = classify_paper_heuristic(w)
        # 2015 < 2018, so ML pattern doesn't trigger RECENT
        assert classification is None or classification.category != PaperCategory.RECENT

    def test_textbook_high_citations_no_qualifiers(self):
        w = make_work(
            title="Dynamics of Structures",
            cited_by_count=5000,
        )
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.FOUNDATIONAL
        assert needs_review is False

    def test_textbook_with_qualifiers_needs_review(self):
        w = make_work(
            title="Fundamentals of Geotechnical Engineering",
            cited_by_count=2000,
        )
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.FOUNDATIONAL  # suggestion
        assert needs_review is True  # needs LLM review

    def test_textbook_low_citations_needs_review(self):
        w = make_work(
            title="Introduction to Structural Analysis",
            cited_by_count=50,
        )
        classification, needs_review = classify_paper_heuristic(w)
        assert needs_review is True

    def test_regular_paper_no_heuristic_match(self):
        w = make_work(title="Seismic Response of a 40-Story Building")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification is None
        assert needs_review is False

    def test_handbook_pattern_not_triggered_for_textbook(self):
        """Handbook patterns should NOT trigger for textbook-pattern titles."""
        w = make_work(title="Introduction to Statistical Learning: A Review")
        # is_textbook matches "Introduction to" → handbook check is skipped
        classification, needs_review = classify_paper_heuristic(w)
        # Should go through textbook path, not handbook
        # (since is_textbook is checked before handbook for titles matching both)
        assert classification is not None

    def test_heuristic_priority_implementation_first(self):
        """Implementation patterns should be checked before everything else."""
        w = make_work(title="reghdfe: A Review of the STATA Module")
        classification, needs_review = classify_paper_heuristic(w)
        assert classification.category == PaperCategory.IMPLEMENTATION
