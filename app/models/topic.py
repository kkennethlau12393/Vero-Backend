"""
Topic Models - Research Topics and Subject Areas

OpenAlex topic classification system with hierarchical structure.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TopicHierarchy(BaseModel):
    """OpenAlex topic hierarchy: Domain → Field → Subfield → Topic."""
    domain_id: Optional[str] = Field(None, description="OpenAlex domain ID")
    domain_name: Optional[str] = Field(None, description="Domain display name")
    field_id: Optional[str] = Field(None, description="OpenAlex field ID")
    field_name: Optional[str] = Field(None, description="Field display name")
    subfield_id: Optional[str] = Field(None, description="OpenAlex subfield ID")
    subfield_name: Optional[str] = Field(None, description="Subfield display name")


class TopicRef(BaseModel):
    """
    Reference to a research topic/subject area.
    
    Score meaning depends on context:
    - In Work.topics: How strongly work belongs to topic
    - In query analysis: How strongly query relates to topic
    """
    topic_id: Optional[str] = Field(None, description="OpenAlex topic ID (T...)")
    display_name: str = Field(..., description="Human-readable topic name")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    hierarchy: Optional[TopicHierarchy] = Field(None, description="Topic hierarchy")
    
    # Legacy fields (deprecated)
    subfield: Optional[str] = Field(None, description="[DEPRECATED] Use hierarchy")
    field: Optional[str] = Field(None, description="[DEPRECATED] Use hierarchy")
    domain: Optional[str] = Field(None, description="[DEPRECATED] Use hierarchy")


class TopicQueryRef(BaseModel):
    """Collection of topics extracted from user query text analysis."""
    query_text: str = Field(..., description="Original user query")
    topics: list['TopicRef'] = Field(default_factory=list, description="Extracted topics")
    analyzed_at: Optional[str] = Field(None, description="ISO 8601 timestamp")
