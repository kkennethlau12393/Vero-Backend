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

    whats_new: str
    compared_to_prior_work: str
    novelty_level: Literal["low", "medium", "high"]
    confidence: Literal["low", "medium", "high"]
    novelty_explanation: str
    grounding_papers: list[GroundingPaper] = []  # Papers that support the assessment


class ConnectedWork(BaseModel):
    """A paper connected to the target node via citation in the map."""

    work_id: str
    title: Optional[str] = None
    relationship: Literal["cites", "cited_by"]


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
