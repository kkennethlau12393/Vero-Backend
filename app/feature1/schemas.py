"""
Schemas for Feature 1: Citation Map Retrieval.

Two input modes:
1. Seed Paper Mode: Given a work_id, find top-k connections
2. NL Query Mode: Find the most influential seed paper from query, then expand
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from uuid import UUID


class CitationMapRequest(BaseModel):
    """Request to build a citation map."""
    # Mode 1: Direct seed paper
    seed_work_id: Optional[str] = None

    # Mode 2: Natural language query
    query_text: Optional[str] = None

    # Expansion config
    citing_limit: int = 15      # Papers that cite the seed (incoming edges)
    references_limit: int = 15  # Papers the seed cites (outgoing edges)
    min_citations: int = 0      # Minimum citation count threshold for inclusion

    # Multi-hop expansion (recommended for richer networks)
    use_multi_hop: bool = True  # Use multi-hop citation network exploration
    total_nodes: Optional[int] = None  # Target nodes for multi-hop; defaults to citing_limit + references_limit + 1

    # Output control
    create_graph_draft: bool = True


class CitationNode(BaseModel):
    """A node in the citation graph."""
    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: int = 0
    abstract: Optional[str] = None
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
    seed_info: SeedSelectionInfo
    nodes: list[CitationNode]
    edges: list[CitationEdge]

    # If create_graph_draft=True
    graph_draft_id: Optional[UUID] = None

    stats: CitationMapStats
