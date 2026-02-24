"""
API endpoints for Feature 5: Research Gap Analysis.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.tenant import get_tenant_id
from app.db import make_engine
from app.feature5.activity_logger import get_activity_summary
from app.feature5.coverage_tracker import get_gap_analysis_status, track_feature_usage
from app.feature5.gap_service import get_cached_gap_analysis, run_gap_analysis
from app.feature5.schemas import ActivitySummary, GapAnalysisResponse, GapAnalysisStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["gap-analysis"])


def get_engine() -> Engine:
    """Get database engine."""
    return make_engine()


def verify_map_ownership(engine: Engine, map_id: UUID, tenant_id: UUID) -> None:
    """Verify that the map belongs to the requesting tenant."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT map_id FROM maps WHERE map_id = :map_id AND tenant_id = :tenant_id"),
            {"map_id": map_id, "tenant_id": tenant_id},
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Map not found")


def verify_rank_job_ownership(engine: Engine, rank_job_id: UUID, tenant_id: UUID) -> None:
    """Verify that the rank job belongs to the requesting tenant."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT rank_job_id FROM rank_jobs WHERE rank_job_id = :rjid AND tenant_id = :tid"),
            {"rjid": rank_job_id, "tid": tenant_id},
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Rank job not found")


# ============================================================================
# Map-based gap analysis endpoints (existing)
# ============================================================================

@router.get("/maps/{map_id}/gap-analysis/status", response_model=GapAnalysisStatus)
def get_gap_analysis_status_endpoint(
    map_id: UUID,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> GapAnalysisStatus:
    """
    Check if gap analysis is unlocked for a map.

    Returns the current coverage percentage, breakdown, and unlock status.
    The gap analysis button should be enabled when unlocked=true.
    """
    try:
        verify_map_ownership(engine, map_id, tenant_id)
        status = get_gap_analysis_status(engine, map_id=map_id)
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting gap analysis status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maps/{map_id}/gap-analysis", response_model=GapAnalysisResponse)
def run_gap_analysis_endpoint(
    map_id: UUID,
    force_refresh: bool = False,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> GapAnalysisResponse:
    """
    Run gap analysis on a map.

    This is a comprehensive analysis that:
    1. Detects gaps from internal data (citation graph, ranked list, timeline, etc.)
    2. Synthesizes gaps using LLM
    3. Validates gaps against external sources using GPT-5.2 + web_search

    The analysis requires at least 50% coverage (unlocked status) to run.
    """
    try:
        verify_map_ownership(engine, map_id, tenant_id)

        # Check if unlocked
        status = get_gap_analysis_status(engine, map_id=map_id)
        if not status.unlocked:
            raise HTTPException(
                status_code=403,
                detail=f"Gap analysis not unlocked. {status.message}",
            )

        # Check for cached results (with tenant filtering)
        if not force_refresh:
            cached = get_cached_gap_analysis(engine, map_id, tenant_id=tenant_id)
            if cached:
                logger.info(f"Returning cached gap analysis for map {map_id}")
                return cached

        # Run full analysis
        logger.info(f"Running gap analysis for map {map_id}")
        result = run_gap_analysis(engine, map_id, tenant_id)
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error running gap analysis")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/maps/{map_id}/gap-analysis/results", response_model=Optional[GapAnalysisResponse])
def get_gap_analysis_results_endpoint(
    map_id: UUID,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> Optional[GapAnalysisResponse]:
    """
    Get cached gap analysis results for a map.

    Returns None if no analysis has been run yet.
    """
    try:
        verify_map_ownership(engine, map_id, tenant_id)
        return get_cached_gap_analysis(engine, map_id, tenant_id=tenant_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting gap analysis results")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maps/{map_id}/track-feature")
def track_feature_usage_endpoint(
    map_id: UUID,
    feature_type: str,
    work_id: Optional[str] = None,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> dict:
    """
    Track feature usage for coverage calculation.

    This should be called when a user uses a feature on the map
    to update the coverage percentage for gap analysis unlock.
    """
    valid_types = [
        "timeline_map_wide",
        "timeline_per_node",
        "ranked_list_per_node",
        "citation_graph_per_node",
    ]

    if feature_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature_type. Must be one of: {valid_types}",
        )

    if feature_type != "timeline_map_wide" and not work_id:
        raise HTTPException(
            status_code=400,
            detail="work_id is required for per-node features",
        )

    try:
        verify_map_ownership(engine, map_id, tenant_id)

        with engine.connect() as conn:
            track_feature_usage(
                conn=conn,
                map_id=map_id,
                feature_type=feature_type,
                work_id=work_id,
            )

        # Return updated status
        status = get_gap_analysis_status(engine, map_id=map_id)
        return {
            "tracked": True,
            "coverage_pct": status.coverage_pct,
            "unlocked": status.unlocked,
            "message": status.message,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error tracking feature usage")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Rank-based gap analysis endpoints (new)
# ============================================================================

@router.get("/rank/{rank_job_id}/gap-analysis/status", response_model=GapAnalysisStatus)
def get_rank_gap_analysis_status_endpoint(
    rank_job_id: UUID,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> GapAnalysisStatus:
    """
    Check if gap analysis is unlocked for a rank job.

    Returns the current coverage percentage, breakdown, and unlock status.
    Coverage is calculated from research_activity_log entries scoped to this rank_job_id.
    """
    try:
        verify_rank_job_ownership(engine, rank_job_id, tenant_id)
        status = get_gap_analysis_status(engine, rank_job_id=rank_job_id)
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting rank gap analysis status")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Activity summary endpoint
# ============================================================================

@router.get("/research-activity/summary", response_model=ActivitySummary)
def get_research_activity_summary(
    map_id: Optional[UUID] = Query(None),
    rank_job_id: Optional[UUID] = Query(None),
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
) -> ActivitySummary:
    """
    Get a summary of all research activities for a map or rank job.

    Returns counts and details per activity type, including:
    - How many methodology comparisons (with 2/3/4 node breakdown)
    - How many novelty assessments (with work_ids)
    - How many timelines (per-node and map-wide)
    - How many node details viewed
    - Coverage percentage and unlock status

    At least one of map_id or rank_job_id must be provided.
    """
    if not map_id and not rank_job_id:
        raise HTTPException(
            status_code=400,
            detail="At least one of map_id or rank_job_id must be provided",
        )

    try:
        # Verify ownership
        if map_id:
            verify_map_ownership(engine, map_id, tenant_id)
        if rank_job_id:
            verify_rank_job_ownership(engine, rank_job_id, tenant_id)

        with engine.connect() as conn:
            summary = get_activity_summary(conn, map_id=map_id, rank_job_id=rank_job_id)

        # Add coverage percentage
        status = get_gap_analysis_status(engine, map_id=map_id, rank_job_id=rank_job_id)

        return ActivitySummary(
            entry_point=summary.get("entry_point"),
            activities=summary.get("activities", {}),
            coverage_pct=status.coverage_pct,
            first_activity_at=summary.get("first_activity_at"),
            last_activity_at=summary.get("last_activity_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting activity summary")
        raise HTTPException(status_code=500, detail=str(e))
