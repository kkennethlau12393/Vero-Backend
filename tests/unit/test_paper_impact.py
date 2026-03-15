"""
Unit tests for app/feature3/paper_impact_analytics.py pure functions.

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.paper_impact_analytics import (
    _is_review_guideline_paper,
    _is_software_tool_paper,
    calculate_impact_score,
)


@pytest.mark.unit
class TestCalculateImpactScore:
    def test_no_references(self):
        score = calculate_impact_score(1000, [])
        assert score == 1.0

    def test_high_impact(self):
        refs = [{"cited_by_count": 100}, {"cited_by_count": 200}]
        score = calculate_impact_score(10000, refs)
        assert score > 0.7

    def test_low_impact(self):
        refs = [{"cited_by_count": 5000}, {"cited_by_count": 10000}]
        score = calculate_impact_score(100, refs)
        assert score < 0.3

    def test_equal_citations(self):
        refs = [{"cited_by_count": 500}]
        score = calculate_impact_score(500, refs)
        assert 0.0 < score < 1.0

    def test_zero_ref_citations(self):
        refs = [{"cited_by_count": 0}, {"cited_by_count": 0}]
        score = calculate_impact_score(1000, refs)
        assert score > 0.0

    def test_normalized_to_0_1(self):
        refs = [{"cited_by_count": 100}]
        score = calculate_impact_score(50000, refs)
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestIsSoftwareToolPaper:
    def test_library_in_title(self):
        assert _is_software_tool_paper("scikit-learn: Machine Learning library in Python", None) is True

    def test_package_in_title(self):
        assert _is_software_tool_paper("lme4: a statistical package for R", None) is True

    def test_toolkit_in_title(self):
        assert _is_software_tool_paper("NLTK: Natural Language Toolkit", None) is True

    def test_web_server_in_title(self):
        assert _is_software_tool_paper("BLAST web server for sequence alignment", None) is True

    def test_pip_install_in_abstract(self):
        assert _is_software_tool_paper("MyTool", "Install using pip install mytool for quick setup") is True

    def test_github_available_in_abstract(self):
        assert _is_software_tool_paper("MyTool", "Source code is available on GitHub for download") is True

    def test_normal_paper(self):
        assert _is_software_tool_paper(
            "Attention Is All You Need",
            "We propose a new architecture based solely on attention mechanisms"
        ) is False

    def test_none_inputs(self):
        assert _is_software_tool_paper(None, None) is False


@pytest.mark.unit
class TestIsReviewGuidelinePaper:
    def test_review_in_title(self):
        assert _is_review_guideline_paper("A comprehensive review of deep learning", None) is True

    def test_survey_in_title(self):
        assert _is_review_guideline_paper("A survey of transformer architectures", None) is True

    def test_meta_analysis_in_title(self):
        assert _is_review_guideline_paper("Meta-analysis of treatment outcomes", None) is True

    def test_guideline_in_title(self):
        assert _is_review_guideline_paper("PRISMA guidelines for systematic reviews", None) is True

    def test_systematic_review_in_abstract(self):
        assert _is_review_guideline_paper(
            "Treatment Outcomes",
            "We systematically reviewed all published studies on this topic"
        ) is True

    def test_hallmarks_pattern(self):
        assert _is_review_guideline_paper("Hallmarks of Cancer", None) is True

    def test_normal_paper(self):
        assert _is_review_guideline_paper(
            "Deep Residual Learning for Image Recognition",
            "We present a residual learning framework for training deeper networks"
        ) is False

    def test_none_inputs(self):
        assert _is_review_guideline_paper(None, None) is False
