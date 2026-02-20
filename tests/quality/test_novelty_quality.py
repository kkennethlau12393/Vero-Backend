"""
Quality/regression tests for Feature 3 (Novelty Assessment) output properties.

Uses constructed data to verify structural invariants, grounding paper constraints,
novelty level rules, and banned verb enforcement. No live API calls.

Mark: @pytest.mark.quality
"""
import json

import pytest

from app.feature3.node_details_service import (
    _has_banned_verbs,
    _scrub_banned_verbs,
    _validate_llm_response,
)
from app.feature3.schemas import (
    GroundingPaper,
    NodeDetailsResponse,
    NoveltyAssessment,
)
from tests.fixtures.novelty_responses import (
    NOVELTY_LLM_RESPONSE_HIGH,
    NOVELTY_LLM_RESPONSE_MEDIUM_REVIEW,
    NOVELTY_LLM_RESPONSE_PIONEERING,
    make_landmarks,
    make_references,
)


@pytest.mark.quality
class TestNoveltySchemaCompleteness:
    """Verify all required fields are present and valid in novelty assessments."""

    def test_novelty_level_valid_enum(self):
        valid = {"low", "medium", "high", "pioneering"}
        for level in valid:
            assessment = NoveltyAssessment(
                novelty_level=level,
                confidence="high",
                novelty_explanation="Test explanation text here.",
            )
            assert assessment.novelty_level == level

    def test_confidence_valid_enum(self):
        valid = {"low", "medium", "high"}
        for conf in valid:
            assessment = NoveltyAssessment(
                novelty_level="medium",
                confidence=conf,
                novelty_explanation="Test explanation text here.",
            )
            assert assessment.confidence == conf

    def test_grounding_paper_required_fields(self):
        gp = GroundingPaper(
            work_id="W1234567890",
            title="Test Paper",
            year=2020,
            cited_by_count=500,
            relationship="cited_reference",
            relevance="Introduced key technique used in this work.",
        )
        assert gp.work_id.startswith("W")
        assert gp.relationship in ("cited_reference", "field_landmark")
        assert len(gp.relevance) > 20

    def test_relationship_valid_enum(self):
        for rel in ("cited_reference", "field_landmark"):
            gp = GroundingPaper(
                work_id="W1", title="T", relationship=rel,
                relevance="Relevant because it established the baseline method."
            )
            assert gp.relationship == rel


@pytest.mark.quality
class TestGroundingPaperConstraints:
    """Verify grounding paper count and composition rules."""

    def test_validate_ensures_min_grounding_count(self):
        """_validate_llm_response should top up grounding to at least 5."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        gps = validated["novelty_assessment"]["grounding_papers"]
        assert len(gps) >= 5

    def test_validate_caps_at_max_grounding(self):
        """_validate_llm_response should cap grounding at 7."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        refs = make_references(10)
        landmarks = make_landmarks(10)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        gps = validated["novelty_assessment"]["grounding_papers"]
        assert len(gps) <= 7

    def test_validate_includes_landmarks(self):
        """Should include field_landmark papers in grounding."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        gps = validated["novelty_assessment"]["grounding_papers"]
        landmark_ids = {lm["work_id"] for lm in landmarks}
        landmark_count = sum(1 for gp in gps if gp["work_id"] in landmark_ids)
        assert landmark_count >= 3

    def test_no_self_citation(self):
        """Target paper should never appear in its own grounding papers."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        target_id = "W1234567890"
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(
            llm_result, refs, landmarks, target_work_id=target_id
        )
        gps = validated["novelty_assessment"]["grounding_papers"]
        gp_ids = {gp["work_id"] for gp in gps}
        assert target_id not in gp_ids


@pytest.mark.quality
class TestNoveltyLevelConstraints:
    """Verify novelty level classification rules."""

    def test_pioneering_allows_null_whats_new(self):
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_PIONEERING)
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        assert validated["novelty_assessment"]["novelty_level"] == "pioneering"
        # whats_new can be None for pioneering
        assert validated["novelty_assessment"]["whats_new"] is None

    def test_non_pioneering_requires_whats_new(self):
        """For non-pioneering papers, whats_new should not be None after validation."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        # High-level papers must have whats_new
        assert validated["novelty_assessment"]["whats_new"] is not None

    def test_invalid_level_defaults_to_medium(self):
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        llm_result["novelty_assessment"]["novelty_level"] = "invalid_level"
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        assert validated["novelty_assessment"]["novelty_level"] == "medium"

    def test_invalid_confidence_defaults_to_low(self):
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        llm_result["novelty_assessment"]["confidence"] = "invalid"
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        assert validated["novelty_assessment"]["confidence"] == "low"


@pytest.mark.quality
class TestBannedVerbEnforcement:
    """Verify post-LLM banned verb scrubbing works correctly."""

    def test_scrub_is_idempotent(self):
        text = "This paper demonstrates a novel approach."
        assert _scrub_banned_verbs(_scrub_banned_verbs(text)) == _scrub_banned_verbs(text)

    def test_validated_response_has_no_banned_verbs(self):
        """After _validate_llm_response, no banned verbs should remain."""
        # Inject banned verbs into LLM response
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        llm_result["summary"] = "This paper explores the effects of attention mechanisms."
        llm_result["novelty_assessment"]["whats_new"] = "This work examines new approaches."
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)

        assert not _has_banned_verbs(validated["summary"])
        assert not _has_banned_verbs(validated["novelty_assessment"]["whats_new"])

    def test_relevance_scrubbed(self):
        """Grounding paper relevance strings should also be scrubbed."""
        llm_result = json.loads(NOVELTY_LLM_RESPONSE_HIGH)
        # Inject banned verb into a grounding paper relevance
        llm_result["novelty_assessment"]["grounding_papers"][0]["relevance"] = \
            "This paper explores related techniques in the same domain."
        refs = make_references(5)
        landmarks = make_landmarks(5)
        validated = _validate_llm_response(llm_result, refs, landmarks)
        for gp in validated["novelty_assessment"]["grounding_papers"]:
            assert not _has_banned_verbs(gp.get("relevance", ""))
