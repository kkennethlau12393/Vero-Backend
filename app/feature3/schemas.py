"""
Pydantic models for Feature 3 (Node Details Pop-up).

This module defines the request/response schemas for the node details
endpoint, including the novelty assessment structure with grounding papers.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel


class GroundingPaper(BaseModel):
    """A paper that grounds the novelty assessment."""

    work_id: str
    title: str
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    relationship: Literal["cited_reference", "field_landmark"]
    relevance: str  # How this paper relates to the novelty claim
    authors: list[str] = []


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
    abstract: Optional[str] = None
    authors: list[str] = []


class TimelineSection(BaseModel):
    """A group of papers from the same era."""

    era: str  # "1990s", "2000s", etc.
    papers: list[TimelinePaper] = []


class StructuredCitation(BaseModel):
    """A citation reference within structured narrative text."""

    ref: int
    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    authors: list[str] = []


class StructuredText(BaseModel):
    """Narrative text with numbered citation references and metadata."""

    text: str
    citations: list[StructuredCitation] = []


class TechnicalTerm(BaseModel):
    """A technical term extracted from the narrative with a tooltip explanation."""

    term: str
    explanation: str


class EraSubsection(BaseModel):
    """A thematic subsection within an era commentary."""

    heading: str
    body: Union[str, StructuredText]


class EraCommentary(BaseModel):
    """Narrative commentary for a single era in the timeline."""

    era: str  # "1990s", "2000s", etc.
    headline: Union[str, StructuredText]  # One-line era title
    narrative: Union[str, StructuredText]  # Flat fallback built from subsections
    subsections: list[EraSubsection] = []  # 2-4 thematic subsections
    key_work_ids: list[str] = []


class ResearchLineageNarrative(BaseModel):
    """The vertical evolution story of a research lineage through one paper's lens."""

    historical_context: Union[str, StructuredText]  # 3-5 sentences
    contribution_statement: Union[str, StructuredText]  # 2-3 sentences
    downstream_impact: Union[str, StructuredText]  # 3-5 sentences
    era_commentaries: list[EraCommentary] = []
    cross_domain_influence: Optional[Union[str, StructuredText]] = None
    paper_type: Optional[str] = None  # software|review|foundational|empirical|measurement
    is_paradigm_shift: bool = False
    impact_score: float = 0.0  # 0-1 citation velocity vs predecessors
    technical_terms: list[TechnicalTerm] = []


class NodeTimeline(BaseModel):
    """Complete timeline for a node."""

    target_work_id: str
    target_year: Optional[int] = None
    backward: list[TimelineSection] = []  # References + Landmarks (before target)
    forward: list[TimelineSection] = []  # Citing papers (after target)
    narrative: Optional[ResearchLineageNarrative] = None  # Rich evolution narrative
    # Backward-compat flat fields from old impact_analysis
    before_approach: Optional[str] = None
    after_approach: Optional[str] = None
    shift_description: Optional[str] = None


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
