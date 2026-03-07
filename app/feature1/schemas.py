"""
Schemas for Feature 1: Citation Map Retrieval.

Two input modes:
1. Seed Paper Mode: Given a work_id, find top-k connections
2. NL Query Mode: Find the most influential seed paper from query, then expand
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from uuid import UUID


class CitationMapRequest(BaseModel):
    """Request to build a citation map.

    Supports multiple seed identification modes (only ONE should be provided):
    1. seed_work_id: Direct OpenAlex work ID (for internal use)
    2. seed_doi: DOI from PDF metadata (primary - most PDFs have DOI)
    3. seed_title: Paper title (fallback - all PDFs have title)
    4. query_text: Natural language query (exploratory - finds most influential seed)
    """
    # Seed identification modes (provide ONE)
    seed_work_id: Optional[str] = None  # Direct OpenAlex ID (internal use)
    seed_doi: Optional[str] = None      # DOI (from PDF metadata - primary)
    seed_title: Optional[str] = None    # Paper title (fallback - all PDFs have this)
    query_text: Optional[str] = None    # NL query (exploratory search)

    # Expansion config
    citing_limit: int = 15      # Papers that cite the seed (incoming edges)
    references_limit: int = 15  # Papers the seed cites (outgoing edges)
    min_citations: int = 0      # Minimum citation count threshold for inclusion

    # Multi-hop expansion
    total_nodes: Optional[int] = None  # Target nodes; defaults to citing_limit + references_limit + 1

    # Output control
    create_graph_draft: bool = True

    # Structured query fields (auto-decomposed if not provided)
    topic: Optional[str] = None
    domain: Optional[str] = None
    aspect: Optional[str] = None

    # Intent options (all have defaults for backward compatibility)
    map_focus: Optional[str] = None     # "landscape", "core_cluster", "evolution"
    expansion: Optional[str] = None     # "narrow", "foundations", "wide"
    map_size: Optional[str] = None      # "small", "medium", "large"


class CitationNode(BaseModel):
    """A node in the citation graph."""
    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: int = 0
    abstract: Optional[str] = None
    authors: list[str] = []
    venue: Optional[str] = None
    doi: Optional[str] = None
    is_open_access: Optional[bool] = None
    subtopic: Optional[str] = None
    is_seed: bool = False
    hop: int = 0  # Distance from seed (0=seed, 1=direct connection, 2=2-hop, etc.)
    relationship: Literal["seed", "cites_seed", "cited_by_seed", "network"]  # network = multi-hop


class CitationEdge(BaseModel):
    """An edge in the citation graph.

    Edge direction: from_work_id cites to_work_id
    """
    from_work_id: str
    to_work_id: str


class SeedSelectionInfo(BaseModel):
    """Information about how the seed paper was selected."""
    seed_work_id: str
    seed_title: Optional[str] = None
    selection_strategy: str  # "direct" or "highest_cited_from_query"
    selection_reason: Optional[str] = None
    candidates_considered: int = 0


class CitationMapStats(BaseModel):
    """Statistics about the citation map."""
    total_nodes: int = 0
    citing_found: int = 0      # Papers that cite the seed (1-hop)
    references_found: int = 0  # Papers the seed cites (1-hop)
    network_nodes: int = 0     # 2+ hop papers (multi-hop mode only)
    edges_count: int = 0
    max_hop: int = 0           # Maximum hop distance from seed


class CitationMapResponse(BaseModel):
    """Response with citation graph data."""
    citation_map_id: Optional[UUID] = None  # Set after persistence
    seed_info: SeedSelectionInfo
    nodes: list[CitationNode]
    edges: list[CitationEdge]

    # If create_graph_draft=True
    graph_draft_id: Optional[UUID] = None

    stats: CitationMapStats


class CitationMapListItem(BaseModel):
    """Lightweight summary for listing saved citation maps."""
    citation_map_id: UUID
    seed_work_id: str
    seed_title: Optional[str] = None
    query_text: Optional[str] = None
    node_count: int = 0
    created_at: datetime
