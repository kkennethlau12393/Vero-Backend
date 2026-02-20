"""
Unit tests for pure logic functions in app/feature3/grounding_supplement.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.grounding_supplement import (
    _extract_search_terms,
    _map_field_to_s2_fields,
    assess_grounding_needs,
)


@pytest.mark.unit
class TestAssessGroundingNeeds:
    def test_sufficient_grounding(self):
        refs = [{"work_id": f"W{i}"} for i in range(3)]
        landmarks = [{"work_id": f"L{i}"} for i in range(3)]
        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)
        assert refs_needed == 0
        assert landmarks_needed == 0
        assert reason == "sufficient"

    def test_needs_refs(self):
        refs = []
        landmarks = [{"work_id": f"L{i}"} for i in range(3)]
        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)
        assert refs_needed >= 2

    def test_needs_landmarks(self):
        refs = [{"work_id": f"W{i}"} for i in range(3)]
        landmarks = []
        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)
        assert landmarks_needed >= 3

    def test_needs_both(self):
        refs_needed, landmarks_needed, reason = assess_grounding_needs([], [])
        assert refs_needed > 0
        assert landmarks_needed > 0

    def test_total_below_minimum(self):
        refs = [{"work_id": "W1"}, {"work_id": "W2"}]
        landmarks = []
        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)
        # With 2 refs and 0 landmarks, need at least 3 landmarks
        assert landmarks_needed >= 3


@pytest.mark.unit
class TestExtractSearchTerms:
    def test_removes_stopwords(self):
        terms = _extract_search_terms("This paper presents a novel approach", None)
        assert "this" not in terms.lower()
        assert "paper" not in terms.lower()

    def test_uses_title_and_abstract(self):
        terms = _extract_search_terms(
            "Transformer Architecture",
            "Self-attention mechanisms enable parallel processing of sequences."
        )
        assert len(terms.split()) > 0

    def test_max_8_terms(self):
        terms = _extract_search_terms(
            "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10",
            "Abstract with many additional unique important technical scientific terms here."
        )
        assert len(terms.split()) <= 8

    def test_deduplicates(self):
        terms = _extract_search_terms("Transformer Transformer Architecture", None)
        words = terms.split()
        assert len(words) == len(set(words))


@pytest.mark.unit
class TestMapFieldToS2Fields:
    def test_psychology_mapping(self):
        result = _map_field_to_s2_fields("Psychology")
        assert result is not None
        assert "Psychology" in result

    def test_computer_science_mapping(self):
        result = _map_field_to_s2_fields("Computer Science")
        assert result is not None
        assert "Computer Science" in result

    def test_environmental_keywords(self):
        result = _map_field_to_s2_fields(
            "applied",
            title="Climate Change Assessment Report",
        )
        assert result is not None
        assert "Environmental Science" in result

    def test_generic_falls_back_to_title(self):
        result = _map_field_to_s2_fields(
            "applied",
            title="Depression Treatment with Cognitive Behavioral Therapy",
        )
        assert result is not None
        assert "Psychology" in result or "Medicine" in result

    def test_unmappable(self):
        result = _map_field_to_s2_fields(None, title=None, abstract=None)
        assert result is None

    def test_biology_keywords(self):
        result = _map_field_to_s2_fields(
            "general",
            title="CRISPR gene editing in human cells",
        )
        assert result is not None
        assert "Biology" in result
