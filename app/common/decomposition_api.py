"""
API router for query decomposition endpoint.

Exposes a standalone endpoint for frontend to preview query decomposition
before submitting a full ranking or citation map request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional
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


class DecomposeRequest(BaseModel):
    """Request to decompose a query into structured components."""
    query_text: str


class DecomposeResponse(BaseModel):
    """Response with decomposed query components."""
    topic: str
    topic_aliases: list[str] = []
    domain: Optional[str] = None
    domain_aliases: list[str] = []
    aspect: Optional[str] = None
    aspect_aliases: list[str] = []
    suggested_specificity: str = "broad"
    reasoning: Optional[str] = None


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

    return DecomposeResponse(
        topic=result.get("topic", req.query_text),
        topic_aliases=result.get("topic_aliases", []),
        domain=result.get("domain"),
        domain_aliases=result.get("domain_aliases", []),
        aspect=result.get("aspect"),
        aspect_aliases=result.get("aspect_aliases", []),
        suggested_specificity=result.get("suggested_specificity", "broad"),
        reasoning=result.get("reasoning"),
    )
