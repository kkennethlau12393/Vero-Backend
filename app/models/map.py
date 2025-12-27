"""
Map Models - Finalized Visualization Graphs

Final stored citation graph with visualization data.
"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.work import WorkRef
from app.models.graph import CitationEdge
from app.models.topic import TopicRef


class LayoutCoordinates(BaseModel):
    """2D position for visualization."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class MapNode(BaseModel):
    """Full node in finalized map with visualization metadata."""
    work: WorkRef = Field(..., description="Full work metadata")
    assigned_subtopic: Optional[TopicRef] = Field(None, description="Category")
    position: Optional[LayoutCoordinates] = Field(None, description="2D position")
    is_seed: bool = Field(False, description="Is seed paper")
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class SubtopicDefinition(BaseModel):
    """
    Node grouping strategy.
    
    Rules:
    - Default: Subfield level
    - < 4 subfields: Topic level
    - One subfield > 80% nodes: Topic level
    - User can toggle if auto-selected
    """
    level: Literal["domain", "field", "subfield", "topic"] = Field(...)
    auto_selected: bool = Field(...)
    reason: Optional[str] = Field(None)
    user_override: bool = Field(False)


class FieldContext(BaseModel):
    """Research field context."""
    field_id: str = Field(..., description="OpenAlex field ID (F...)")
    field_name: str = Field(..., description="Field name")
    subfield_count: Optional[int] = Field(None)


class Map(BaseModel):
    """
    Final stored citation graph with visualization data.
    
    Permanent, shareable graph with:
    - Full WorkRef nodes (complete metadata)
    - Topic assignments and categorization
    - Layout coordinates for consistent rendering
    - Versioned (modifications create new version)
    """
    map_id: Optional[str] = Field(None, description="Unique identifier")
    graph_draft_id: str = Field(..., description="Source GraphDraft")
    nodes: list[MapNode] = Field(default_factory=list)
    edges: list[CitationEdge] = Field(default_factory=list)
    field_context: FieldContext = Field(...)
    subtopic_definition: SubtopicDefinition = Field(...)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(1, ge=1)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    is_public: bool = Field(False)
