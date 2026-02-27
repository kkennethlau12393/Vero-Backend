"""
API router for Feature 3 (Node Details Pop-up).

This module exposes endpoints to retrieve detailed information about
a specific node/work, including LLM-generated summary, keywords,
novelty assessment, and timeline narrative.

Supports two access paths:
- /maps/{map_id}/nodes/{work_id}/details  — from a citation map (includes connected works)
- /rank-jobs/{rank_job_id}/works/{work_id}/details  — from a rank job (no edges)
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
from app.feature5.activity_logger import log_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["node-details"])


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
    include_novelty: bool = False,
    include_timeline: bool = False,
    force_regenerate: bool = False,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Get detailed pop-up information for a node in the citation map.

    By default returns lightweight metadata only (title, year, authors, venue,
    abstract, connected works, access info). No LLM call.

    Args:
        include_novelty: If True, runs the LLM-based novelty assessment
            (summary, keywords, grounding papers). Triggered by user action.
        include_timeline: If True, includes a temporal timeline showing
            references, landmarks, and citing papers grouped by era.
        force_regenerate: If True, clears persisted assessment and regenerates
            via LLM. Use when the assessment version or prompts have changed.
    """
    try:
        result = get_node_details(
            engine,
            tenant_id=tenant_id,
            map_id=map_id,
            work_id=work_id,
            include_novelty=include_novelty,
            include_timeline=include_timeline,
            force_regenerate=force_regenerate,
        )
        # Log activity
        with engine.connect() as conn:
            log_activity(
                conn, tenant_id, "node_details_viewed",
                map_id=map_id,
                work_id=work_id,
                node_count=1,
                metadata={"include_novelty": include_novelty, "include_timeline": include_timeline},
            )
            if include_novelty:
                log_activity(
                    conn, tenant_id, "novelty_assessed",
                    map_id=map_id,
                    work_id=work_id,
                    node_count=1,
                )
            if include_timeline:
                log_activity(
                    conn, tenant_id, "timeline_per_node",
                    map_id=map_id,
                    work_id=work_id,
                    node_count=1,
                )
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg in ("map_not_found", "node_not_found", "work_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception(f"Unexpected error for work_id={work_id}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/rank-jobs/{rank_job_id}/works/{work_id}/details",
    response_model=NodeDetailsResponse,
)
def get_work_details_from_rank_job(
    rank_job_id: UUID,
    work_id: str,
    include_novelty: bool = False,
    include_timeline: bool = False,
    force_regenerate: bool = False,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Get detailed information for a work in a rank job's results.

    Same as the map-based endpoint but accessed via rank_job_id instead of
    map_id. No connected_works (edges) since rank jobs don't have a graph.
    """
    try:
        result = get_node_details(
            engine,
            tenant_id=tenant_id,
            rank_job_id=rank_job_id,
            work_id=work_id,
            include_novelty=include_novelty,
            include_timeline=include_timeline,
            force_regenerate=force_regenerate,
        )
        # Log activity
        with engine.connect() as conn:
            log_activity(
                conn, tenant_id, "node_details_viewed",
                rank_job_id=rank_job_id,
                work_id=work_id,
                node_count=1,
                metadata={"include_novelty": include_novelty, "include_timeline": include_timeline},
            )
        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg in ("rank_job_not_found", "work_not_in_results", "work_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in get_work_details_from_rank_job endpoint")
        raise HTTPException(status_code=500, detail=str(e))
