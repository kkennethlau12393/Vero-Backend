"""
Coverage tracker for Feature 5: Research Gap Analysis.

Calculates the coverage percentage for a map or rank job based on which
features have been used, determining if gap analysis should be unlocked.

Supports both entry points:
  - Citation map (map_id): queries gap_feature_usage + cache tables (legacy)
  - Rank (rank_job_id): queries research_activity_log
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature5.schemas import CoverageBreakdown, GapAnalysisStatus

logger = logging.getLogger(__name__)

# Coverage weights
WEIGHT_MAP_WIDE_TIMELINE = 0.25  # 25%
WEIGHT_METHODOLOGY_COMPARISON = 0.15  # 15% each, capped
METHODOLOGY_COMPARISON_CAP = 0.30  # Max 30% from methodology
WEIGHT_PER_NODE_MAX = 0.30  # Up to 30% from per-node exploration

# Unlock threshold
UNLOCK_THRESHOLD = 0.50  # 50%


# ============================================================================
# Map-based coverage (legacy — reads from cache tables + gap_feature_usage)
# ============================================================================

def get_map_node_count(conn: Connection, map_id: UUID) -> int:
    """Get total number of nodes in a map."""
    result = conn.execute(
        text("SELECT COUNT(*) FROM map_nodes WHERE map_id = :map_id"),
        {"map_id": map_id},
    ).scalar()
    return result or 0


def has_map_wide_timeline(conn: Connection, map_id: UUID) -> bool:
    """Check if map-wide timeline analysis has been run."""
    result = conn.execute(
        text("""
            SELECT 1 FROM gap_feature_usage
            WHERE map_id = :map_id
            AND feature_type = 'timeline_map_wide'
            LIMIT 1
        """),
        {"map_id": map_id},
    ).first()
    return result is not None


def get_methodology_comparison_count(conn: Connection, map_id: UUID) -> int:
    """Count methodology comparisons done for papers in this map."""
    result = conn.execute(
        text("""
            SELECT COUNT(DISTINCT comparison_hash)
            FROM methodology_comparison_cache mcc
            WHERE EXISTS (
                SELECT 1 FROM map_nodes mn
                WHERE mn.map_id = :map_id
                AND mn.work_id = ANY(mcc.work_ids)
            )
        """),
        {"map_id": map_id},
    ).scalar()
    return result or 0


def get_explored_node_count(conn: Connection, map_id: UUID) -> int:
    """Count nodes that have been explored with any feature."""
    result = conn.execute(
        text("""
            SELECT COUNT(DISTINCT mn.work_id)
            FROM map_nodes mn
            WHERE mn.map_id = :map_id
            AND (
                EXISTS (
                    SELECT 1 FROM node_details_cache ndc
                    WHERE ndc.work_id = mn.work_id
                )
                OR
                EXISTS (
                    SELECT 1 FROM methodology_comparison_cache mcc
                    WHERE mn.work_id = ANY(mcc.work_ids)
                )
                OR
                EXISTS (
                    SELECT 1 FROM gap_feature_usage gfu
                    WHERE gfu.map_id = :map_id
                    AND gfu.work_id = mn.work_id
                    AND gfu.feature_type IN ('timeline_per_node', 'ranked_list_per_node', 'citation_graph_per_node')
                )
            )
        """),
        {"map_id": map_id},
    ).scalar()
    return result or 0


def _calculate_map_coverage(conn: Connection, map_id: UUID) -> CoverageBreakdown:
    """Calculate coverage from map-based tables (legacy path)."""
    total_nodes = get_map_node_count(conn, map_id)
    if total_nodes == 0:
        return CoverageBreakdown(total_nodes=0, entry_point="citation_map")

    timeline_pct = WEIGHT_MAP_WIDE_TIMELINE if has_map_wide_timeline(conn, map_id) else 0.0
    methodology_count = get_methodology_comparison_count(conn, map_id)
    methodology_pct = min(
        methodology_count * WEIGHT_METHODOLOGY_COMPARISON,
        METHODOLOGY_COMPARISON_CAP,
    )

    nodes_explored = get_explored_node_count(conn, map_id)
    exploration_ratio = nodes_explored / total_nodes if total_nodes > 0 else 0.0
    per_node_pct = exploration_ratio * WEIGHT_PER_NODE_MAX

    # Also check activity log for this map (supplements legacy tables)
    novelty_count = conn.execute(
        text("""
            SELECT COUNT(*) FROM research_activity_log
            WHERE map_id = :mid AND activity_type = 'novelty_assessed'
        """),
        {"mid": map_id},
    ).scalar() or 0

    return CoverageBreakdown(
        map_wide_timeline=timeline_pct,
        methodology_comparisons=methodology_pct,
        per_node_exploration=per_node_pct,
        methodology_comparison_count=methodology_count,
        novelty_count=novelty_count,
        nodes_explored=nodes_explored,
        total_nodes=total_nodes,
        entry_point="citation_map",
    )


# ============================================================================
# Rank-based coverage (new — reads from research_activity_log)
# ============================================================================

def _get_rank_job_result_count(conn: Connection, rank_job_id: UUID) -> int:
    """Get total number of ranked results for a rank job."""
    result = conn.execute(
        text("""
            SELECT COUNT(*) FROM rank_results
            WHERE rank_job_id = :rjid
        """),
        {"rjid": rank_job_id},
    ).scalar()
    return result or 0


def _calculate_rank_coverage(conn: Connection, rank_job_id: UUID) -> CoverageBreakdown:
    """Calculate coverage from activity log for rank_job_id context."""
    total_nodes = _get_rank_job_result_count(conn, rank_job_id)
    if total_nodes == 0:
        return CoverageBreakdown(total_nodes=0, entry_point="rank")

    # Timeline map-wide: binary 25%
    has_timeline = conn.execute(
        text("""
            SELECT 1 FROM research_activity_log
            WHERE rank_job_id = :rjid AND activity_type = 'timeline_map_wide'
            LIMIT 1
        """),
        {"rjid": rank_job_id},
    ).first()
    timeline_pct = WEIGHT_MAP_WIDE_TIMELINE if has_timeline else 0.0

    # Methodology comparisons: 15% each, capped at 30%
    methodology_count = conn.execute(
        text("""
            SELECT COUNT(*) FROM research_activity_log
            WHERE rank_job_id = :rjid AND activity_type = 'methodology_compared'
        """),
        {"rjid": rank_job_id},
    ).scalar() or 0
    methodology_pct = min(
        methodology_count * WEIGHT_METHODOLOGY_COMPARISON,
        METHODOLOGY_COMPARISON_CAP,
    )

    # Per-node exploration: unique work_ids across node-level activities
    explored_row = conn.execute(
        text("""
            SELECT COUNT(DISTINCT wid) FROM (
                SELECT unnest(work_ids) AS wid
                FROM research_activity_log
                WHERE rank_job_id = :rjid
                AND activity_type IN ('novelty_assessed', 'timeline_per_node', 'node_details_viewed', 'methodology_compared')
            ) sub
        """),
        {"rjid": rank_job_id},
    ).scalar()
    nodes_explored = explored_row or 0
    exploration_ratio = nodes_explored / total_nodes if total_nodes > 0 else 0.0
    per_node_pct = exploration_ratio * WEIGHT_PER_NODE_MAX

    # Novelty count
    novelty_count = conn.execute(
        text("""
            SELECT COUNT(*) FROM research_activity_log
            WHERE rank_job_id = :rjid AND activity_type = 'novelty_assessed'
        """),
        {"rjid": rank_job_id},
    ).scalar() or 0

    return CoverageBreakdown(
        map_wide_timeline=timeline_pct,
        methodology_comparisons=methodology_pct,
        per_node_exploration=per_node_pct,
        methodology_comparison_count=methodology_count,
        novelty_count=novelty_count,
        nodes_explored=nodes_explored,
        total_nodes=total_nodes,
        entry_point="rank",
    )


# ============================================================================
# Public API
# ============================================================================

def calculate_coverage(
    conn: Connection,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> CoverageBreakdown:
    """
    Calculate coverage breakdown for a map or rank job.

    At least one of map_id or rank_job_id must be provided.
    If both are given, map_id takes precedence (legacy behavior).
    """
    if map_id:
        return _calculate_map_coverage(conn, map_id)
    elif rank_job_id:
        return _calculate_rank_coverage(conn, rank_job_id)
    else:
        return CoverageBreakdown(total_nodes=0)


def get_gap_analysis_status(
    engine: Engine,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> GapAnalysisStatus:
    """
    Get gap analysis unlock status for a map or rank job.

    Returns status including whether unlocked, coverage %, and message.
    """
    with engine.connect() as conn:
        breakdown = calculate_coverage(conn, map_id=map_id, rank_job_id=rank_job_id)

        total_pct = (
            breakdown.map_wide_timeline +
            breakdown.methodology_comparisons +
            breakdown.per_node_exploration
        )

        unlocked = total_pct >= UNLOCK_THRESHOLD

        if unlocked:
            message = f"Gap analysis available ({total_pct:.0%} explored)"
        else:
            remaining = UNLOCK_THRESHOLD - total_pct
            message = f"Explored {total_pct:.0%} of research area. Explore more to unlock."

        return GapAnalysisStatus(
            unlocked=unlocked,
            coverage_pct=total_pct,
            breakdown=breakdown,
            message=message,
        )


def track_feature_usage(
    conn: Connection,
    map_id: UUID,
    feature_type: str,
    work_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Track that a feature was used on a map or node.

    Legacy function — writes to gap_feature_usage table.
    New code should use activity_logger.log_activity() instead.
    """
    conn.execute(
        text("""
            INSERT INTO gap_feature_usage (map_id, feature_type, work_id, metadata)
            VALUES (:map_id, :feature_type, :work_id, :metadata)
            ON CONFLICT (map_id, feature_type, COALESCE(work_id, ''))
            DO UPDATE SET
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """),
        {
            "map_id": map_id,
            "feature_type": feature_type,
            "work_id": work_id,
            "metadata": metadata,
        },
    )
    conn.commit()


def get_available_data_sources(
    conn: Connection,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> List[str]:
    """
    Get list of data sources available for gap detection.

    Returns list of data source names that have data.
    """
    sources = []

    if map_id:
        # Citation graph entry point
        edge_count = conn.execute(
            text("SELECT COUNT(*) FROM map_edges WHERE map_id = :map_id"),
            {"map_id": map_id},
        ).scalar() or 0

        if edge_count > 0:
            sources.append("citation_graph")
        else:
            sources.append("ranked_list")

        if has_map_wide_timeline(conn, map_id):
            sources.append("timeline")

        if get_methodology_comparison_count(conn, map_id) > 0:
            sources.append("methodology")

        novelty_count = conn.execute(
            text("""
                SELECT COUNT(*) FROM node_details_cache ndc
                WHERE ndc.novelty_assessment IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM map_nodes mn
                    WHERE mn.map_id = :map_id
                    AND mn.work_id = ndc.work_id
                )
            """),
            {"map_id": map_id},
        ).scalar() or 0

        if novelty_count > 0:
            sources.append("novelty")

    elif rank_job_id:
        # Rank entry point — always has ranked_list
        sources.append("ranked_list")

        # Check activity log for other sources
        activity_types = conn.execute(
            text("""
                SELECT DISTINCT activity_type FROM research_activity_log
                WHERE rank_job_id = :rjid
            """),
            {"rjid": rank_job_id},
        ).scalars().all()

        type_set = set(activity_types)
        if "timeline_map_wide" in type_set:
            sources.append("timeline")
        if "methodology_compared" in type_set:
            sources.append("methodology")
        if "novelty_assessed" in type_set:
            sources.append("novelty")

    return sources
