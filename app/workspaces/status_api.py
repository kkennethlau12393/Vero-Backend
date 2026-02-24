"""
Workspace status API — single source of truth for what a workspace contains
and what jobs are currently running.

Used by the frontend dashboard to poll for job completion and determine
which tabs/artifacts are available in a workspace.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.tenant import get_tenant_id
from app.db import make_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


class JobStatus(BaseModel):
    id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class WorkspaceStatusResponse(BaseModel):
    workspace_id: str
    has_ranked_list: bool
    has_citation_map: bool
    rank_job: Optional[JobStatus] = None
    citation_map_id: Optional[str] = None
    last_updated_at: Optional[str] = None


class BatchStatusRequest(BaseModel):
    workspace_ids: list[str]


def _get_workspace_status(conn, tenant_id: UUID) -> WorkspaceStatusResponse:
    """Derive workspace capabilities and job states from existing tables."""

    # Latest rank job for this workspace.
    # A workspace has a ranked list only if a completed job has persisted rows.
    rank_row = conn.execute(
        text("""
            SELECT rank_job_id, status, started_at, completed_at, error_json
            FROM rank_jobs
            WHERE tenant_id = :t
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"t": tenant_id},
    ).mappings().first()

    rank_job = None
    has_ranked_list = False
    last_updated = None

    if rank_row:
        rank_results_row = conn.execute(
            text("""
                SELECT 1
                FROM rank_results
                WHERE rank_job_id = :rank_job_id
                LIMIT 1
            """),
            {"rank_job_id": rank_row["rank_job_id"]},
        ).first()
        has_ranked_list = rank_row["status"] == "completed" and bool(rank_results_row)
        error_msg = None
        if rank_row["error_json"]:
            err = rank_row["error_json"]
            error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)

        rank_job = JobStatus(
            id=str(rank_row["rank_job_id"]),
            status=rank_row["status"],
            started_at=rank_row["started_at"].isoformat() if rank_row["started_at"] else None,
            completed_at=rank_row["completed_at"].isoformat() if rank_row["completed_at"] else None,
            error=error_msg,
        )
        last_updated = (rank_row["completed_at"] or rank_row["started_at"])

    # Latest citation map for this workspace
    map_row = conn.execute(
        text("""
            SELECT map_id, created_at
            FROM maps
            WHERE tenant_id = :t
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"t": tenant_id},
    ).mappings().first()

    has_citation_map = map_row is not None
    citation_map_id = str(map_row["map_id"]) if map_row else None

    if map_row and map_row["created_at"]:
        map_ts = map_row["created_at"]
        if last_updated is None or map_ts > last_updated:
            last_updated = map_ts

    return WorkspaceStatusResponse(
        workspace_id=str(tenant_id),
        has_ranked_list=has_ranked_list,
        has_citation_map=has_citation_map,
        rank_job=rank_job,
        citation_map_id=citation_map_id,
        last_updated_at=last_updated.isoformat() if last_updated else None,
    )


@router.get("/status", response_model=WorkspaceStatusResponse)
def get_workspace_status(
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Get the current status of a workspace.

    Returns what artifacts exist (ranked list, citation map) and the
    state of any in-progress jobs. The workspace is identified by the
    X-Workspace-Id header (same as all other endpoints).
    """
    try:
        with engine.connect() as conn:
            return _get_workspace_status(conn, tenant_id)
    except Exception as e:
        logger.exception("Error fetching workspace status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/status/batch", response_model=list[WorkspaceStatusResponse])
def get_workspace_status_batch(
    req: BatchStatusRequest,
    engine: Engine = Depends(get_engine),
):
    """
    Get status for multiple workspaces in a single call.

    Used by the dashboard to efficiently check all workspace cards.
    Does NOT require X-Workspace-Id header — workspace IDs are in the body.
    """
    if len(req.workspace_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 workspace IDs per batch")

    results = []
    try:
        with engine.connect() as conn:
            for ws_id in req.workspace_ids:
                try:
                    tid = UUID(ws_id)
                except ValueError:
                    continue
                results.append(_get_workspace_status(conn, tid))
        return results
    except Exception as e:
        logger.exception("Error fetching batch workspace status")
        raise HTTPException(status_code=500, detail=str(e))
