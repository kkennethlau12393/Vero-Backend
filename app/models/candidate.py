"""
Candidate Models - Discovery Results

Pool of papers fetched from OpenAlex during discovery phase.
"""

from typing import Optional, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CandidateItem(BaseModel):
    """Individual paper in a candidate set with discovery metadata."""
    work_id: str = Field(..., description="OpenAlex work identifier")
    reason: str = Field(..., description="Why included (e.g., 'cited_by_seed')")
    reason_detail: Optional[str] = Field(None, description="Additional context")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class CandidateSet(BaseModel):
    """
    Pool of candidate papers from discovery phase.
    
    Immutable snapshot of papers fetched from OpenAlex based on seed and parameters.
    Foundation for graph generation and ranking features.
    
    Seed types:
    - "paper": From specific paper (DOI, OpenAlex ID, PDF)
    - "topic_text": From free text query → topics
    - "node_subtopic": From clicking topic in existing map
    """
    candidate_set_id: Optional[str] = Field(None, description="Unique identifier")
    seed_type: Literal["paper", "topic_text", "node_subtopic"] = Field(...)
    seed_ref: str = Field(..., description="work_id, topic_id, or query hash")
    field_context_id: Optional[str] = Field(None, description="OpenAlex field ID")
    params_json: dict[str, Any] = Field(default_factory=dict, description="Discovery params")
    items: list[CandidateItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_count: Optional[int] = Field(None)
