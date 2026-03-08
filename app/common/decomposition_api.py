"""
API router for query decomposition endpoint.

Exposes a standalone endpoint for frontend to preview query decomposition
before submitting a full ranking or citation map request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.auth.tenant import get_tenant_id
from app.common.query_decomposition import decompose_query
from app.db import make_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/query", tags=["query"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazily construct the database engine once per process."""
    return make_engine()


# ── Request / Response Models ────────────────────────────────────────────────

class DecomposeRequest(BaseModel):
    """Request to decompose a query into structured components."""
    query_text: str
    entry_type: Optional[Literal["ranked", "citation"]] = None


class RankOptionSet(BaseModel):
    """Which rank options to show the user (None = hide that option)."""
    scope: Optional[list[str]] = None
    focus: list[str] = ["foundational", "recent", "surveys", "all_time"]
    depth: list[str] = ["high_level", "comprehensive"]


class RankDefaults(BaseModel):
    """Smart defaults for rank options based on query analysis."""
    scope: Optional[str] = None
    focus: str = "all_time"
    depth: str = "comprehensive"


class CitationOptionSet(BaseModel):
    """Which citation map options to show the user."""
    scope: Optional[list[str]] = None  # None = hide (no domain)
    drift: list[str] = ["strict", "moderate", "open"]
    temporal: list[str] = ["seminal", "recent", "all"]
    map_size: list[str] = ["small", "medium", "large"]


class CitationDefaults(BaseModel):
    """Smart defaults for citation map options based on query analysis."""
    scope: Optional[str] = None
    drift: str = "moderate"
    temporal: str = "all"
    map_size: str = "medium"


class DecomposeResponse(BaseModel):
    """Response with decomposed query components and option guidance."""
    topic: str
    topic_aliases: list[str] = []
    domain: Optional[str] = None
    domain_aliases: list[str] = []
    aspect: Optional[str] = None
    aspect_aliases: list[str] = []
    suggested_specificity: str = "broad"
    reasoning: Optional[str] = None
    # Option guidance (populated based on entry_type)
    available_options: Optional[RankOptionSet] = None
    defaults: Optional[RankDefaults] = None
    citation_options: Optional[CitationOptionSet] = None
    citation_defaults: Optional[CitationDefaults] = None


@router.post("/decompose", response_model=DecomposeResponse)
def decompose_query_endpoint(
    req: DecomposeRequest,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Auto-decompose a query into structured components.

    Called by frontend to show the refine panel before submitting
    a full ranking or citation map request. Requires authentication.
    """
    with engine.begin() as conn:
        result = decompose_query(conn, req.query_text)

    has_domain = bool(result.get("domain"))
    has_aspect = bool(result.get("aspect"))
    specificity = result.get("suggested_specificity", "broad")

    # ── Rank options ─────────────────────────────────────────────────────
    rank_options = None
    rank_defaults = None
    if req.entry_type is None or req.entry_type == "ranked":
        rank_options = RankOptionSet()
        rank_defaults = RankDefaults()

        if has_domain:
            rank_options.scope = ["broad", "intersection", "topic_focused", "domain_focused"]
            if specificity in ("specific", "balanced"):
                rank_defaults.scope = "intersection"
            else:
                rank_defaults.scope = "broad"
        else:
            rank_options.scope = None
            rank_defaults.scope = "broad"

        if not has_aspect:
            rank_defaults.depth = "high_level"

    # ── Citation map options ─────────────────────────────────────────────
    cite_options = None
    cite_defaults = None
    if req.entry_type is None or req.entry_type == "citation":
        cite_options = CitationOptionSet()
        cite_defaults = CitationDefaults(drift="moderate", temporal="all", map_size="medium")

        if has_domain:
            cite_options.scope = ["broad", "intersection", "topic_focused", "domain_focused"]
            if specificity in ("specific", "balanced"):
                cite_defaults.scope = "intersection"
            else:
                cite_defaults.scope = "broad"
        else:
            cite_options.scope = None
            cite_defaults.scope = "broad"

    return DecomposeResponse(
        topic=result.get("topic", req.query_text),
        topic_aliases=result.get("topic_aliases", []),
        domain=result.get("domain"),
        domain_aliases=result.get("domain_aliases", []),
        aspect=result.get("aspect"),
        aspect_aliases=result.get("aspect_aliases", []),
        suggested_specificity=result.get("suggested_specificity", "broad"),
        reasoning=result.get("reasoning"),
        available_options=rank_options,
        defaults=rank_defaults,
        citation_options=cite_options,
        citation_defaults=cite_defaults,
    )
