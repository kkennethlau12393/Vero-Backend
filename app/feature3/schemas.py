"""
Pydantic models for Feature 3 (Node Details Pop-up).

This module defines the request/response schemas for the node details
endpoint, including the novelty assessment structure with grounding papers.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class GroundingPaper(BaseModel):
    """A paper that grounds the novelty assessment."""

    work_id: str
    title: str
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    relationship: Literal["cited_reference", "field_landmark"]
    relevance: str  # How this paper relates to the novelty claim


class NoveltyAssessment(BaseModel):
    """LLM-generated novelty assessment grounded in actual papers."""

    # Optional for pioneering works (nothing to compare to)
    whats_new: Optional[str] = None
    compared_to_prior_work: Optional[str] = None
    novelty_level: Literal["low", "medium", "high", "pioneering"]
    confidence: Literal["low", "medium", "high"]
    novelty_explanation: str  # For pioneering: explains WHY it's pioneering
    grounding_papers: list[GroundingPaper] = []  # Papers that support the assessment
    context_depth: Literal["abstract_only", "full_text"] = "abstract_only"


class ConnectedWork(BaseModel):
    """A paper connected to the target node via citation in the map."""

    work_id: str
    title: Optional[str] = None
    relationship: Literal["cites", "cited_by"]


# ============================================================================
# Timeline Schemas (Per-Node Timeline Feature)
# ============================================================================

class TimelinePaper(BaseModel):
    """A paper in the timeline."""

    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    relationship: Literal["reference", "landmark", "citing"]


class TimelineSection(BaseModel):
    """A group of papers from the same era."""

    era: str  # "1990s", "2000s", etc.
    papers: list[TimelinePaper] = []


class PaperImpactAnalysis(BaseModel):
    """Analysis of whether a paper represents a paradigm shift."""

    is_paradigm_shift: bool  # Did this paper change the field?
    impact_score: float  # 0-1 based on citation velocity vs predecessors
    before_approach: Optional[str] = None  # Methodology in references
    after_approach: Optional[str] = None  # Methodology in citing papers
    shift_description: Optional[str] = None  # Description of what changed


class NodeTimeline(BaseModel):
    """Complete timeline for a node."""

    target_work_id: str
    target_year: Optional[int] = None
    backward: list[TimelineSection] = []  # References + Landmarks (before target)
    forward: list[TimelineSection] = []  # Citing papers (after target)
    impact_analysis: Optional[PaperImpactAnalysis] = None


class NodeDetailsResponse(BaseModel):
    """Full response for node details pop-up."""

    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = []
    venue: Optional[str] = None
    cited_by_count: int = 0
    abstract: Optional[str] = None
    summary: str
    keywords: list[str]
    novelty_assessment: Optional[NoveltyAssessment] = None
    connected_works: list[ConnectedWork] = []
    assessment_unavailable_reason: Optional[str] = None
    timeline: Optional[NodeTimeline] = None  # Per-node timeline (when include_timeline=true)
    # Topic info for "Research this topic" feature
    primary_topic_id: Optional[str] = None  # OpenAlex topic ID (e.g., "T12345")
    topic_display_name: Optional[str] = None  # Human-readable name (e.g., "Computer Vision")
    # Access info
    access_status: Optional[Literal["open", "closed", "unknown"]] = None
    pdf_url: Optional[str] = None
    doi_url: Optional[str] = None
    oa_status: Optional[str] = None  # gold, green, hybrid, bronze, closed
