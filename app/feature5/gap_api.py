"""
API endpoints for Feature 5: Research Gap Analysis.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.tenant import get_workspace_id
from app.db import make_engine
from app.feature5.coverage_tracker import get_gap_analysis_status, track_feature_usage
from app.feature5.gap_service import get_cached_gap_analysis, run_gap_analysis
from app.feature5.schemas import GapAnalysisResponse, GapAnalysisStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["gap-analysis"])


def get_engine() -> Engine:
    """Get database engine."""
    return make_engine()


def verify_map_ownership(engine: Engine, map_id: UUID, workspace_id: UUID) -> None:
    """Verify that the map belongs to the requesting workspace."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT map_id FROM maps WHERE map_id = :map_id AND workspace_id = :workspace_id"),
            {"map_id": map_id, "workspace_id": workspace_id},
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Map not found")


@router.get("/maps/{map_id}/gap-analysis/status", response_model=GapAnalysisStatus)
def get_gap_analysis_status_endpoint(
    map_id: UUID,
    engine: Engine = Depends(get_engine),
    workspace_id: UUID = Depends(get_workspace_id),
) -> GapAnalysisStatus:
    """
    Check if gap analysis is unlocked for a map.

    Returns the current coverage percentage, breakdown, and unlock status.
    The gap analysis button should be enabled when unlocked=true.
    """
    try:
        verify_map_ownership(engine, map_id, workspace_id)
        status = get_gap_analysis_status(engine, map_id)
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
    workspace_id: UUID = Depends(get_workspace_id),
) -> GapAnalysisResponse:
    """
    Run gap analysis on a map.

    This is a comprehensive analysis that:
    1. Detects gaps from internal data (citation graph, ranked list, timeline, etc.)
    2. Synthesizes gaps using LLM
    3. Validates gaps against external sources using GPT-5.2 + web_search

    The analysis requires at least 50% coverage (unlocked status) to run.

    Args:
        map_id: Map ID to analyze
        force_refresh: If true, re-run analysis even if cached results exist

    Returns:
        GapAnalysisResponse with detected and validated research gaps
    """
    try:
        verify_map_ownership(engine, map_id, workspace_id)

        # Check if unlocked
        status = get_gap_analysis_status(engine, map_id)
        if not status.unlocked:
            raise HTTPException(
                status_code=403,
                detail=f"Gap analysis not unlocked. {status.message}",
            )

        # Check for cached results (with workspace filtering)
        if not force_refresh:
            cached = get_cached_gap_analysis(engine, map_id, workspace_id=workspace_id)
            if cached:
                logger.info(f"Returning cached gap analysis for map {map_id}")
                return cached

        # Run full analysis
        logger.info(f"Running gap analysis for map {map_id}")
        result = run_gap_analysis(engine, map_id, workspace_id)
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
    workspace_id: UUID = Depends(get_workspace_id),
) -> Optional[GapAnalysisResponse]:
    """
    Get cached gap analysis results for a map.

    Returns None if no analysis has been run yet.
    """
    try:
        verify_map_ownership(engine, map_id, workspace_id)
        return get_cached_gap_analysis(engine, map_id, workspace_id=workspace_id)
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
    workspace_id: UUID = Depends(get_workspace_id),
) -> dict:
    """
    Track feature usage for coverage calculation.

    This should be called when a user uses a feature on the map
    to update the coverage percentage for gap analysis unlock.

    Args:
        map_id: Map ID
        feature_type: One of:
            - timeline_map_wide: Map-wide timeline analysis
            - timeline_per_node: Per-node timeline
            - ranked_list_per_node: Per-node ranked list
            - citation_graph_per_node: Per-node citation graph
        work_id: Work ID for per-node features (required for per-node types)

    Returns:
        Updated coverage status
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
        verify_map_ownership(engine, map_id, workspace_id)

        with engine.connect() as conn:
            track_feature_usage(
                conn=conn,
                map_id=map_id,
                feature_type=feature_type,
                work_id=work_id,
            )

        # Return updated status
        status = get_gap_analysis_status(engine, map_id)
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
