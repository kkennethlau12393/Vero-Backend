"""
Ranking Models - Scored Recommendations

Scored and ordered paper recommendations.
"""

from typing import Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Transparent score calculation breakdown."""
    total_score: float = Field(..., ge=0.0, le=1.0)
    topic_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    citation_count: Optional[float] = Field(None, ge=0.0, le=1.0)
    recency: Optional[float] = Field(None, ge=0.0, le=1.0)
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    graph_centrality: Optional[float] = Field(None, ge=0.0, le=1.0)
    custom_factors: dict[str, float] = Field(default_factory=dict)


class RankedItem(BaseModel):
    """Single paper with ranking score and explanation."""
    work_id: str = Field(..., description="OpenAlex work ID")
    score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: Optional[ScoreBreakdown] = Field(None)
    reasons: list[str] = Field(default_factory=list)
    rank_position: Optional[int] = Field(None, ge=1)


class RankingContext(BaseModel):
    """
    Ranking context definition.
    
    Types:
    - topic: "Papers about X"
    - subtopic: "Papers in category"
    - seed_paper: "Similar papers"
    """
    context_type: Literal["topic", "subtopic", "seed_paper"] = Field(...)
    topic_id: Optional[str] = Field(None)
    subtopic_id: Optional[str] = Field(None)
    seed_work_id: Optional[str] = Field(None)
    filters: dict[str, Any] = Field(default_factory=dict)


class RankedList(BaseModel):
    """
    Scored and ordered paper recommendations.
    
    Output of ranking engine with transparent scoring and explanations.
    """
    rank_id: Optional[str] = Field(None, description="Unique identifier")
    context: RankingContext = Field(...)
    candidate_set_id: str = Field(..., description="Source CandidateSet")
    ranked_items: list[RankedItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    algorithm: Optional[str] = Field(None)
    total_candidates: Optional[int] = Field(None)
