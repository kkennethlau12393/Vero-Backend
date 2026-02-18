"""
FastAPI router for Feature 4 (Methodology Comparison).

POST /v1/maps/{map_id}/compare-methodologies
"""

from __future__ import annotations

import logging
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.engine import Engine

from app.auth.tenant import get_workspace_id
from app.db import make_engine
from app.feature4.compare_service import compare_methodologies
from app.feature4.schemas import MethodologyCompareRequest, MethodologyComparisonResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["methodology"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazily construct the database engine once per process."""
    return make_engine()


@router.post(
    "/maps/{map_id}/compare-methodologies",
    response_model=MethodologyComparisonResponse,
)
def compare_methodologies_endpoint(
    map_id: str,
    req: MethodologyCompareRequest,
    engine: Engine = Depends(get_engine),
    workspace_id: UUID = Depends(get_workspace_id),
):
    """Compare methodologies of 2-4 papers from a citation map."""
    try:
        result = compare_methodologies(
            engine=engine,
            tenant_id=workspace_id,
            map_id=map_id,
            work_ids=req.work_ids,
        )
        return result

    except ValueError as e:
        msg = str(e)
        if "not found in map" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in compare_methodologies endpoint")
        raise HTTPException(status_code=500, detail=str(e))
