"""
Coverage tracker for Feature 5: Research Gap Analysis.

Calculates the coverage percentage for a map or rank job based on which
features have been used, determining if gap analysis should be unlocked.

Coverage formula:
  - Map-wide timeline: 15% (binary)
  - Node-specific tasks (novelty + per-node timeline): 4% each, cap 28%
  - Methodology comparisons: 2-node=4%, 3-node=6%, 4-node=8%, cap 30%
    Penalty: -1% per node already compared in a prior comparison

Supports both entry points:
  - Citation map (map_id): queries gap_feature_usage + activity log
  - Rank (rank_job_id): queries research_activity_log
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature5.schemas import CoverageBreakdown, GapAnalysisStatus

logger = logging.getLogger(__name__)

# Coverage weights
WEIGHT_MAP_WIDE_TIMELINE = 0.15  # 15%

WEIGHT_NODE_SPECIFIC_ACTION = 0.04  # 4% per unique (work_id, activity_type)
NODE_SPECIFIC_CAP = 0.28  # Max 28%

METHODOLOGY_BASE = {2: 0.04, 3: 0.06, 4: 0.08}  # Base % by node count
METHODOLOGY_STALE_PENALTY = 0.01  # -1% per already-compared node
METHODOLOGY_CAP = 0.30  # Max 30%

# Unlock threshold
UNLOCK_THRESHOLD = 0.50  # 50%


# ============================================================================
# Shared coverage helpers
# ============================================================================

def _calculate_methodology_contribution(
    comparisons: List[Tuple[List[str], int]],
) -> Tuple[float, float, int]:
    """
    Calculate methodology contribution with stale-node penalty.

    Args:
        comparisons: list of (work_ids, node_count) ordered by created_at.

    Returns:
        (methodology_pct, total_penalty, comparison_count)
    """
    seen_nodes: set[str] = set()
    total = 0.0
    total_penalty = 0.0
    count = 0

    for work_ids, node_count in comparisons:
        base = METHODOLOGY_BASE.get(node_count, METHODOLOGY_BASE.get(4, 0.08))
        if node_count < 2:
            continue

        stale_count = sum(1 for wid in work_ids if wid in seen_nodes)
        penalty = stale_count * METHODOLOGY_STALE_PENALTY
        total_penalty += penalty

        contribution = max(base - penalty, 0.0)
        total += contribution
        count += 1

        seen_nodes.update(work_ids)

    return min(total, METHODOLOGY_CAP), total_penalty, count


def _calculate_node_specific_contribution(
    activities: List[Tuple[str, str]],
) -> Tuple[float, int]:
    """
    Calculate node-specific contribution (novelty + timeline per node).

    4% per unique (work_id, activity_type) pair, cap 28%.

    Args:
        activities: list of (activity_type, work_id) for novelty_assessed
                    and timeline_per_node.

    Returns:
        (node_specific_pct, unique_pair_count)
    """
    unique_pairs: set[Tuple[str, str]] = set()
    for activity_type, work_id in activities:
        if activity_type in ("novelty_assessed", "timeline_per_node") and work_id:
            unique_pairs.add((work_id, activity_type))

    count = len(unique_pairs)
    return min(count * WEIGHT_NODE_SPECIFIC_ACTION, NODE_SPECIFIC_CAP), count


# ============================================================================
# Activity log queries (shared by both paths)
# ============================================================================

def _get_methodology_comparisons_from_log(
    conn: Connection,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> List[Tuple[List[str], int]]:
    """Fetch ordered methodology comparisons from activity log.

    Methodology work_ids are stored in metadata->'work_ids' (JSON array)
    since the table has a single work_id column.
    """
    if map_id:
        where = "map_id = :ctx_id"
    elif rank_job_id:
        where = "rank_job_id = :ctx_id"
    else:
        return []

    rows = conn.execute(
        text(f"""
            SELECT metadata, node_count
            FROM research_activity_log
            WHERE {where} AND activity_type = 'methodology_compared'
            ORDER BY created_at ASC
        """),
        {"ctx_id": map_id or rank_job_id},
    ).mappings().all()

    result = []
    for row in rows:
        meta = row["metadata"] or {}
        work_ids = meta.get("work_ids", []) if isinstance(meta, dict) else []
        result.append((list(work_ids), row["node_count"] or 0))
    return result


def _get_node_specific_activities_from_log(
    conn: Connection,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> List[Tuple[str, str]]:
    """Fetch novelty + timeline_per_node activities from activity log.

    Returns list of (activity_type, work_id) tuples.
    """
    if map_id:
        where = "map_id = :ctx_id"
    elif rank_job_id:
        where = "rank_job_id = :ctx_id"
    else:
        return []

    rows = conn.execute(
        text(f"""
            SELECT activity_type, work_id
            FROM research_activity_log
            WHERE {where}
            AND activity_type IN ('novelty_assessed', 'timeline_per_node')
        """),
        {"ctx_id": map_id or rank_job_id},
    ).mappings().all()

    return [(row["activity_type"], row["work_id"]) for row in rows if row["work_id"]]


# ============================================================================
# Map-based coverage
# ============================================================================

def get_map_node_count(conn: Connection, map_id: UUID) -> int:
    """Get total number of nodes in a map."""
    result = conn.execute(
        text("SELECT COUNT(*) FROM map_nodes WHERE map_id = :map_id"),
        {"map_id": map_id},
    ).scalar()
    return result or 0


def has_map_wide_timeline(conn: Connection, map_id: UUID) -> bool:
    """Check if map-wide timeline analysis has been run (legacy or activity log)."""
    # Check legacy table
    result = conn.execute(
        text("""
            SELECT 1 FROM gap_feature_usage
            WHERE map_id = :map_id
            AND feature_type = 'timeline_map_wide'
            LIMIT 1
        """),
        {"map_id": map_id},
    ).first()
    if result is not None:
        return True

    # Also check activity log
    try:
        result = conn.execute(
            text("""
                SELECT 1 FROM research_activity_log
                WHERE map_id = :map_id AND activity_type = 'timeline_map_wide'
                LIMIT 1
            """),
            {"map_id": map_id},
        ).first()
        return result is not None
    except Exception:
        return False


def get_methodology_comparison_count(conn: Connection, map_id: UUID) -> int:
    """Count methodology comparisons done for papers in this map (legacy cache)."""
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


def _calculate_map_coverage(conn: Connection, map_id: UUID) -> CoverageBreakdown:
    """Calculate coverage for a citation map."""
    total_nodes = get_map_node_count(conn, map_id)
    if total_nodes == 0:
        return CoverageBreakdown(total_nodes=0, entry_point="citation_map")

    # Timeline: 15% binary
    timeline_pct = WEIGHT_MAP_WIDE_TIMELINE if has_map_wide_timeline(conn, map_id) else 0.0

    # Methodology: prefer activity log (has ordering for penalty), fall back to cache
    try:
        comparisons = _get_methodology_comparisons_from_log(conn, map_id=map_id)
    except Exception:
        comparisons = []

    if comparisons:
        methodology_pct, stale_penalty, methodology_count = (
            _calculate_methodology_contribution(comparisons)
        )
    else:
        # Legacy fallback: simple count from cache table, average 4% each (2-node default)
        methodology_count = get_methodology_comparison_count(conn, map_id)
        methodology_pct = min(methodology_count * 0.04, METHODOLOGY_CAP)
        stale_penalty = 0.0

    # Node-specific: prefer activity log, fall back to explored node count
    try:
        node_activities = _get_node_specific_activities_from_log(conn, map_id=map_id)
    except Exception:
        node_activities = []

    if node_activities:
        node_specific_pct, node_specific_count = (
            _calculate_node_specific_contribution(node_activities)
        )
    else:
        # Legacy fallback: count explored nodes, each = 4%
        explored = conn.execute(
            text("""
                SELECT COUNT(DISTINCT mn.work_id)
                FROM map_nodes mn
                WHERE mn.map_id = :map_id
                AND (
                    EXISTS (
                        SELECT 1 FROM node_details_cache ndc
                        WHERE ndc.work_id = mn.work_id
                        AND ndc.novelty_assessment IS NOT NULL
                    )
                )
            """),
            {"map_id": map_id},
        ).scalar() or 0
        node_specific_count = explored
        node_specific_pct = min(explored * WEIGHT_NODE_SPECIFIC_ACTION, NODE_SPECIFIC_CAP)

    # Novelty count
    try:
        novelty_count = conn.execute(
            text("""
                SELECT COUNT(*) FROM research_activity_log
                WHERE map_id = :mid AND activity_type = 'novelty_assessed'
            """),
            {"mid": map_id},
        ).scalar() or 0
    except Exception:
        novelty_count = 0

    # Nodes explored (unique work_ids across all node-level activities)
    nodes_explored = len({wid for _, wid in node_activities if wid}) if node_activities else 0

    return CoverageBreakdown(
        map_wide_timeline=timeline_pct,
        methodology_comparisons=methodology_pct,
        node_specific_exploration=node_specific_pct,
        methodology_comparison_count=methodology_count,
        methodology_stale_penalty=stale_penalty,
        node_specific_count=node_specific_count,
        novelty_count=novelty_count,
        nodes_explored=nodes_explored,
        total_nodes=total_nodes,
        entry_point="citation_map",
    )


# ============================================================================
# Rank-based coverage (reads from research_activity_log)
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

    # Timeline map-wide: binary 15%
    has_timeline = conn.execute(
        text("""
            SELECT 1 FROM research_activity_log
            WHERE rank_job_id = :rjid AND activity_type = 'timeline_map_wide'
            LIMIT 1
        """),
        {"rjid": rank_job_id},
    ).first()
    timeline_pct = WEIGHT_MAP_WIDE_TIMELINE if has_timeline else 0.0

    # Methodology comparisons with stale penalty
    comparisons = _get_methodology_comparisons_from_log(conn, rank_job_id=rank_job_id)
    methodology_pct, stale_penalty, methodology_count = (
        _calculate_methodology_contribution(comparisons)
    )

    # Node-specific: novelty + timeline_per_node
    node_activities = _get_node_specific_activities_from_log(conn, rank_job_id=rank_job_id)
    node_specific_pct, node_specific_count = (
        _calculate_node_specific_contribution(node_activities)
    )

    # Novelty count
    novelty_count = conn.execute(
        text("""
            SELECT COUNT(*) FROM research_activity_log
            WHERE rank_job_id = :rjid AND activity_type = 'novelty_assessed'
        """),
        {"rjid": rank_job_id},
    ).scalar() or 0

    # Nodes explored (unique work_ids across node-level activities)
    nodes_explored = len({wid for _, wid in node_activities if wid})

    return CoverageBreakdown(
        map_wide_timeline=timeline_pct,
        methodology_comparisons=methodology_pct,
        node_specific_exploration=node_specific_pct,
        methodology_comparison_count=methodology_count,
        methodology_stale_penalty=stale_penalty,
        node_specific_count=node_specific_count,
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
            breakdown.node_specific_exploration
        )

        unlocked = total_pct >= UNLOCK_THRESHOLD

        if unlocked:
            message = f"Gap analysis available ({total_pct:.0%} explored)"
        else:
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
