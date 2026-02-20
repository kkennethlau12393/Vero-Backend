"""
Unit tests for pure logic functions in app/feature3/node_details_service.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.node_details_service import (
    _check_grounding_consistency,
    _default_response,
    _extract_cited_work_ids,
    _generate_relevance,
    _generate_summary_from_abstract,
    _extract_keywords_from_abstract,
    _has_banned_verbs,
    _scrub_banned_verbs,
    _truncate_text,
)


@pytest.mark.unit
class TestTruncateText:
    def test_short_text_unchanged(self):
        assert _truncate_text("hello world") == "hello world"

    def test_empty_string(self):
        assert _truncate_text("") == ""

    def test_long_text_truncated(self):
        text = "a " * 300
        result = _truncate_text(text, max_chars=50)
        assert len(result) <= 55  # 50 + "..."
        assert result.endswith("...")

    def test_exactly_max_chars(self):
        text = "x" * 500
        result = _truncate_text(text, max_chars=500)
        assert result == text


@pytest.mark.unit
class TestScrubBannedVerbs:
    def test_explores_replaced(self):
        result = _scrub_banned_verbs("This paper explores the effect")
        assert "explores" not in result.lower()
        assert "demonstrates" in result.lower()

    def test_discusses_replaced(self):
        result = _scrub_banned_verbs("The study discusses findings")
        assert "discusses" not in result.lower()

    def test_examines_replaced(self):
        result = _scrub_banned_verbs("This work examines the data")
        assert "examines" not in result.lower()

    def test_investigates_replaced(self):
        result = _scrub_banned_verbs("The paper investigates mechanisms")
        assert "investigates" not in result.lower()

    def test_assesses_replaced(self):
        result = _scrub_banned_verbs("This study assesses the impact")
        assert "assesses" not in result.lower()

    def test_evaluates_replaced(self):
        result = _scrub_banned_verbs("This work evaluates the model")
        assert "evaluates" not in result.lower()

    def test_past_tense(self):
        result = _scrub_banned_verbs("The study explored new methods")
        assert "explored" not in result.lower()

    def test_gerund_form(self):
        result = _scrub_banned_verbs("By examining the data carefully")
        assert "examining" not in result.lower()

    def test_no_banned_verbs_unchanged(self):
        text = "This paper introduces a novel transformer architecture."
        assert _scrub_banned_verbs(text) == text

    def test_empty_string(self):
        assert _scrub_banned_verbs("") == ""

    def test_none_input(self):
        assert _scrub_banned_verbs(None) is None


@pytest.mark.unit
class TestHasBannedVerbs:
    def test_detects_explores(self):
        assert _has_banned_verbs("This paper explores") is True

    def test_detects_investigates(self):
        assert _has_banned_verbs("This study investigates") is True

    def test_clean_text(self):
        assert _has_banned_verbs("This paper introduces a new method") is False

    def test_empty_string(self):
        assert _has_banned_verbs("") is False

    def test_none_returns_false(self):
        assert _has_banned_verbs(None) is False


@pytest.mark.unit
class TestExtractCitedWorkIds:
    def test_extracts_work_ids(self):
        text = "Builds on W1234567890 and extends W9876543210"
        result = _extract_cited_work_ids(text)
        assert "W1234567890" in result
        assert "W9876543210" in result

    def test_multiple_ids(self):
        text = "References [W1111111111] and [W2222222222] and W3333333333"
        result = _extract_cited_work_ids(text)
        assert len(result) == 3

    def test_no_ids(self):
        result = _extract_cited_work_ids("No work IDs here")
        assert result == set()


@pytest.mark.unit
class TestCheckGroundingConsistency:
    def test_perfect_consistency(self):
        assessment = {
            "novelty_explanation": "Based on [W1111111111] this paper advances the field.",
            "whats_new": "Builds on W1111111111",
            "compared_to_prior_work": None,
            "grounding_papers": [
                {"work_id": "W1111111111"},
            ],
        }
        result = _check_grounding_consistency(assessment, [], [])
        assert result["consistency_score"] == 1.0
        assert result["orphaned_citations"] == []

    def test_orphaned_citations(self):
        assessment = {
            "novelty_explanation": "Based on W1111111111 and W2222222222",
            "whats_new": None,
            "compared_to_prior_work": None,
            "grounding_papers": [
                {"work_id": "W1111111111"},
            ],
        }
        result = _check_grounding_consistency(assessment, [], [])
        assert "W2222222222" in result["orphaned_citations"]

    def test_no_citations_in_text(self):
        assessment = {
            "novelty_explanation": "This paper is novel because it uses new methods.",
            "whats_new": None,
            "compared_to_prior_work": None,
            "grounding_papers": [{"work_id": "W1111111111"}],
        }
        result = _check_grounding_consistency(assessment, [], [])
        assert result["consistency_score"] == 1.0  # No cited_ids -> score is 1.0


@pytest.mark.unit
class TestGenerateRelevance:
    def test_field_landmark_highly_cited(self):
        paper = {"title": "Deep Learning Foundations", "year": 2012, "cited_by_count": 50000}
        result = _generate_relevance(paper, "field_landmark")
        assert "foundational" in result.lower()

    def test_field_landmark_old_paper(self):
        paper = {"title": "Early Neural Networks", "year": 1990, "cited_by_count": 500}
        result = _generate_relevance(paper, "field_landmark")
        assert "early" in result.lower() or "methods" in result.lower()

    def test_cited_reference(self):
        paper = {"title": "Some Method Paper", "year": 2018, "cited_by_count": 100}
        result = _generate_relevance(paper, "cited_reference")
        assert "cited" in result.lower() or "contribution" in result.lower()


@pytest.mark.unit
class TestGenerateSummaryFromAbstract:
    def test_extracts_first_sentences(self):
        abstract = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = _generate_summary_from_abstract(abstract)
        assert "First sentence" in result

    def test_max_3_sentences(self):
        abstract = "A. B. C. D. E. F."
        result = _generate_summary_from_abstract(abstract)
        # Should have at most 3 sentences
        assert result.count(".") <= 4

    def test_no_abstract_uses_title(self):
        result = _generate_summary_from_abstract(None, title="My Paper Title")
        assert "My Paper Title" in result

    def test_no_abstract_no_title(self):
        result = _generate_summary_from_abstract(None, title=None)
        assert "unavailable" in result.lower()


@pytest.mark.unit
class TestExtractKeywordsFromAbstract:
    def test_extracts_quoted_terms(self):
        abstract = 'We introduce "attention mechanism" for sequence modeling.'
        result = _extract_keywords_from_abstract(abstract)
        assert any("attention" in kw for kw in result)

    def test_matches_academic_patterns(self):
        abstract = "We use deep learning and neural networks for natural language processing."
        result = _extract_keywords_from_abstract(abstract)
        assert len(result) > 0

    def test_max_10_keywords(self):
        abstract = "deep learning neural network machine learning natural language processing computer vision reinforcement learning transformer attention convolutional recurrent optimization classification regression clustering embedding"
        result = _extract_keywords_from_abstract(abstract)
        assert len(result) <= 10

    def test_empty_abstract(self):
        assert _extract_keywords_from_abstract(None) == []
        assert _extract_keywords_from_abstract("") == []


@pytest.mark.unit
class TestDefaultResponse:
    def test_includes_ref_and_landmark(self):
        refs = [{"work_id": "W111", "title": "Ref 1", "year": 2015, "cited_by_count": 100}]
        landmarks = [{"work_id": "W222", "title": "Landmark 1", "year": 2010, "cited_by_count": 5000}]
        result = _default_response("My Paper", refs, landmarks)
        assert result["novelty_assessment"]["novelty_level"] == "medium"
        assert result["novelty_assessment"]["confidence"] == "low"
        gp_ids = {gp["work_id"] for gp in result["novelty_assessment"]["grounding_papers"]}
        assert "W111" in gp_ids
        assert "W222" in gp_ids

    def test_empty_refs_and_landmarks(self):
        result = _default_response("My Paper", [], [])
        assert result["novelty_assessment"]["novelty_level"] == "medium"
        assert result["novelty_assessment"]["grounding_papers"] == []
