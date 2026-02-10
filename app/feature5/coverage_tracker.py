"""
Coverage tracker for Feature 5: Research Gap Analysis.

Calculates the coverage percentage for a map based on which features
have been used, determining if gap analysis should be unlocked.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature5.schemas import CoverageBreakdown, GapAnalysisStatus

logger = logging.getLogger(__name__)

# Coverage weights (from plan)
WEIGHT_MAP_WIDE_TIMELINE = 0.25  # 25%
WEIGHT_METHODOLOGY_COMPARISON = 0.15  # 15% each, capped
METHODOLOGY_COMPARISON_CAP = 0.30  # Max 30% from methodology
WEIGHT_PER_NODE_MAX = 0.30  # Up to 30% from per-node exploration

# Unlock threshold
UNLOCK_THRESHOLD = 0.50  # 50%


def get_map_node_count(conn: Connection, map_id: UUID) -> int:
    """Get total number of nodes in a map."""
    result = conn.execute(
        text("SELECT COUNT(*) FROM map_nodes WHERE map_id = :map_id"),
        {"map_id": map_id},
    ).scalar()
    return result or 0


def has_map_wide_timeline(conn: Connection, map_id: UUID) -> bool:
    """
    Check if map-wide timeline analysis has been run.

    Timeline is considered "run" if temporal analytics were generated
    for the map's associated rank job.
    """
    # Check if feature usage tracking exists
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
    """
    Count methodology comparisons done for papers in this map.

    Looks at methodology_comparison_cache for comparisons involving
    papers from this map.
    """
    # Get work_ids in this map
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
    """
    Count nodes that have been explored with any feature.

    A node is "explored" if it has:
    - Node details (summary/novelty) cached
    - Per-node timeline generated
    - Been part of a methodology comparison
    - Has a per-node ranked list or citation graph
    """
    result = conn.execute(
        text("""
            SELECT COUNT(DISTINCT mn.work_id)
            FROM map_nodes mn
            WHERE mn.map_id = :map_id
            AND (
                -- Has node details cached
                EXISTS (
                    SELECT 1 FROM node_details_cache ndc
                    WHERE ndc.work_id = mn.work_id
                )
                OR
                -- Part of methodology comparison
                EXISTS (
                    SELECT 1 FROM methodology_comparison_cache mcc
                    WHERE mn.work_id = ANY(mcc.work_ids)
                )
                OR
                -- Has per-node feature usage tracked
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


def calculate_coverage(
    conn: Connection,
    map_id: UUID,
) -> CoverageBreakdown:
    """
    Calculate coverage breakdown for a map.

    Returns breakdown of coverage percentage by source.
    """
    total_nodes = get_map_node_count(conn, map_id)
    if total_nodes == 0:
        return CoverageBreakdown(total_nodes=0)

    # Map-wide timeline
    timeline_pct = WEIGHT_MAP_WIDE_TIMELINE if has_map_wide_timeline(conn, map_id) else 0.0

    # Methodology comparisons
    methodology_count = get_methodology_comparison_count(conn, map_id)
    methodology_pct = min(
        methodology_count * WEIGHT_METHODOLOGY_COMPARISON,
        METHODOLOGY_COMPARISON_CAP,
    )

    # Per-node exploration
    nodes_explored = get_explored_node_count(conn, map_id)
    exploration_ratio = nodes_explored / total_nodes if total_nodes > 0 else 0.0
    per_node_pct = exploration_ratio * WEIGHT_PER_NODE_MAX

    return CoverageBreakdown(
        map_wide_timeline=timeline_pct,
        methodology_comparisons=methodology_pct,
        per_node_exploration=per_node_pct,
        methodology_comparison_count=methodology_count,
        nodes_explored=nodes_explored,
        total_nodes=total_nodes,
    )


def get_gap_analysis_status(
    engine: Engine,
    map_id: UUID,
) -> GapAnalysisStatus:
    """
    Get gap analysis unlock status for a map.

    Returns status including whether unlocked, coverage %, and message.
    """
    with engine.connect() as conn:
        breakdown = calculate_coverage(conn, map_id)

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

    Args:
        conn: Database connection
        map_id: Map ID
        feature_type: One of: timeline_map_wide, timeline_per_node,
                      ranked_list_per_node, citation_graph_per_node
        work_id: Optional work ID for per-node features
        metadata: Optional additional metadata
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
    map_id: UUID,
) -> List[str]:
    """
    Get list of data sources available for gap detection.

    Returns list of data source names that have data for this map.
    """
    sources = []

    # Always have citation graph or ranked list from entry point
    # Check if this map has citation graph structure
    edge_count = conn.execute(
        text("SELECT COUNT(*) FROM map_edges WHERE map_id = :map_id"),
        {"map_id": map_id},
    ).scalar() or 0

    if edge_count > 0:
        sources.append("citation_graph")
    else:
        sources.append("ranked_list")

    # Check for timeline
    if has_map_wide_timeline(conn, map_id):
        sources.append("timeline")

    # Check for methodology comparisons
    if get_methodology_comparison_count(conn, map_id) > 0:
        sources.append("methodology")

    # Check for novelty assessments
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

    return sources
