"""
API router for Feature 1: Citation Map Retrieval.
"""
from __future__ import annotations

from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.engine import Engine

from app.auth.tenant import get_tenant_id
from app.db import make_engine
from app.feature1.citation_map_service import build_citation_map
from app.feature1.schemas import CitationMapRequest, CitationMapResponse

router = APIRouter(prefix="/v1", tags=["citation-map"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


@router.post("/citation-map", response_model=CitationMapResponse)
def build_citation_map_endpoint(
    req: CitationMapRequest,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Build a citation graph around a seed paper.

    Two modes:
    1. Provide seed_work_id directly
    2. Provide query_text to find the best seed paper automatically

    Returns nodes and edges of the citation graph, plus optionally
    a graph_draft_id that can be passed to /v1/maps/build.
    """
    # Validate input
    if not req.seed_work_id and not req.query_text:
        raise HTTPException(
            status_code=400,
            detail="Either seed_work_id or query_text must be provided",
        )

    try:
        return build_citation_map(
            engine,
            tenant_id=tenant_id,
            request=req,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
