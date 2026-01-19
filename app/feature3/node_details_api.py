"""
API router for Feature 3 (Node Details Pop-up).

This module exposes an endpoint to retrieve detailed information about
a specific node in a citation map, including LLM-generated summary,
keywords, and novelty assessment.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.engine import Engine

from app.auth.tenant import get_tenant_id
from app.db import make_engine
from app.feature3.node_details_service import get_node_details
from app.feature3.schemas import NodeDetailsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/feature3", tags=["feature3-node-details"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazily construct the database engine once per process."""
    return make_engine()


@router.get(
    "/maps/{map_id}/nodes/{work_id}/details",
    response_model=NodeDetailsResponse,
)
def get_node_details_endpoint(
    map_id: UUID,
    work_id: str,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Get detailed pop-up information for a node in the citation map.

    Returns metadata (title, year, authors, etc.), an LLM-generated summary,
    extracted keywords, novelty assessment, and connected works.
    """
    try:
        return get_node_details(
            engine,
            tenant_id=tenant_id,
            map_id=map_id,
            work_id=work_id,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg in ("map_not_found", "node_not_found", "work_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in get_node_details endpoint")
        raise HTTPException(status_code=500, detail=str(e))
