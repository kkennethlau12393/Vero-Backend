"""
Work Models - Academic Papers/Works

Canonical representation of research papers regardless of input source
(DOI, OpenAlex ID, PDF, or free text).
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class IngestState(str, Enum):
    """Data completeness status for a work."""
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"


class Author(BaseModel):
    """Author information for a work."""
    name: str = Field(..., description="Author's full name")
    author_id: Optional[str] = Field(None, description="OpenAlex author ID (A...)")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    position: Optional[int] = Field(None, description="Author position in author list")


class TopicRefSimple(BaseModel):
    """Lightweight topic reference used within WorkRef."""
    topic_id: Optional[str] = Field(None, description="OpenAlex topic ID (T...)")
    display_name: str = Field(..., description="Human-readable topic name")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score (0-1)")


class WorkRef(BaseModel):
    """
    Internal representation of an academic work/paper.
    
    Canonical structure regardless of input source (DOI, OpenAlex ID, PDF, text).
    Supports progressive enrichment from minimal to full metadata.
    """
    # Identifiers
    work_id: Optional[str] = Field(None, description="Primary identifier, preferably OpenAlex W... ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    
    # Core metadata
    title: Optional[str] = Field(None, description="Work title")
    abstract: Optional[str] = Field(None, description="Work abstract")
    year: Optional[int] = Field(None, ge=1000, le=2100, description="Publication year")
    venue: Optional[str] = Field(None, description="Publication venue")
    
    # Authors
    authors: List[Author] = Field(default_factory=list, description="List of authors")
    
    # Citation data
    cited_by_count: Optional[int] = Field(None, ge=0, description="Citation count")
    references: List[str] = Field(default_factory=list, description="Cited work IDs")
    
    # Topics
    primary_topic: Optional[TopicRefSimple] = Field(None, description="Main topic")
    topics: List[TopicRefSimple] = Field(default_factory=list, description="All topics")
    
    # Open access
    oa_pdf_url: Optional[HttpUrl] = Field(None, description="Open access PDF URL")
    
    # Quality tracking
    ingest_state: IngestState = Field(..., description="Data completeness status")
    
    class Config:
        use_enum_values = True


class WorkRefThin(BaseModel):
    """Lightweight WorkRef with minimal fields for performance."""
    work_id: Optional[str] = Field(None, description="Work identifier")
    title: Optional[str] = Field(None, description="Work title")
    year: Optional[int] = Field(None, description="Publication year")
    cited_by_count: Optional[int] = Field(None, description="Citation count")
    ingest_state: IngestState = Field(..., description="Data completeness")
    
    class Config:
        use_enum_values = True
