"""
Graph Models - Citation Graph Data

Temporary raw citation graph before finalization.
"""

from typing import Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from app.models.work import WorkRefThin
from app.models.topic import TopicQueryRef


class CitationEdge(BaseModel):
    """Directed citation: from_work cites to_work."""
    from_work_id: str = Field(..., description="Source work (cites)")
    to_work_id: str = Field(..., description="Target work (cited)")
    citation_context: Optional[str] = Field(None, description="Citation context")


class GraphStats(BaseModel):
    """Quality and coverage metrics for graph draft."""
    node_count: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)
    nodes_with_abstract: int = Field(0, ge=0)
    nodes_with_topics: int = Field(0, ge=0)
    nodes_resolved: int = Field(0, ge=0)
    abstract_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)
    topic_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_citations_per_node: Optional[float] = Field(None, ge=0.0)
    connected_components: Optional[int] = Field(None, ge=1)


class GraphDraft(BaseModel):
    """
    Temporary raw citation graph before finalization.
    
    Ephemeral structure that:
    - Contains basic nodes/edges from OpenAlex
    - Can be enriched progressively (abstracts, topics)
    - Provides stats for user review
    - Expires after 24 hours (default)
    - Converts to permanent Map upon confirmation
    """
    graph_draft_id: Optional[str] = Field(None, description="Unique identifier")
    seed: Union[WorkRefThin, TopicQueryRef] = Field(..., description="Graph seed")
    candidate_set_id: str = Field(..., description="Source CandidateSet")
    nodes: list[WorkRefThin] = Field(default_factory=list)
    edges: list[CitationEdge] = Field(default_factory=list)
    stats: Optional[GraphStats] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=24)
    )
