"""
Activity logger for Feature 5: Research Gap Analysis.

Append-only logging of user research activities. Used for:
- Coverage calculation (gating gap analysis unlock)
- Activity history / summary for the frontend
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

# Valid activity types
ACTIVITY_TYPES = {
    "citation_map_created",
    "rank_job_created",
    "novelty_assessed",
    "timeline_per_node",
    "timeline_map_wide",
    "methodology_compared",
    "node_details_viewed",
    "subtopics_generated",
    "drill_down",
}


def log_activity(
    conn: Connection,
    tenant_id: UUID,
    activity_type: str,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
    work_ids: Optional[List[str]] = None,
    node_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append an activity to the research activity log.

    At least one of map_id or rank_job_id must be provided.
    Failures are logged but never raise — activity logging must not
    break the primary API response.
    """
    if activity_type not in ACTIVITY_TYPES:
        logger.warning(f"Unknown activity_type: {activity_type}")
        return

    # Creation events (entry points) may not have map_id or rank_job_id yet
    creation_types = {"citation_map_created", "rank_job_created"}
    if map_id is None and rank_job_id is None and activity_type not in creation_types:
        logger.warning(f"log_activity called without map_id or rank_job_id: {activity_type}")
        return

    try:
        conn.execute(
            text("""
                INSERT INTO research_activity_log
                    (tenant_id, map_id, rank_job_id, activity_type, work_ids, node_count, metadata)
                VALUES
                    (:tid, :mid, :rjid, :atype, :wids, :nc, CAST(:meta AS jsonb))
            """),
            {
                "tid": tenant_id,
                "mid": map_id,
                "rjid": rank_job_id,
                "atype": activity_type,
                "wids": work_ids or [],
                "nc": node_count,
                "meta": json.dumps(metadata) if metadata else "{}",
            },
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log activity {activity_type}: {e}")


def get_activity_summary(
    conn: Connection,
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """
    Get a summary of research activities for a map or rank job.

    Returns counts and details per activity type.
    """
    if map_id is None and rank_job_id is None:
        return {"activities": {}}

    # Build WHERE clause based on context
    if map_id and rank_job_id:
        where = "map_id = :mid OR rank_job_id = :rjid"
        params: Dict[str, Any] = {"mid": map_id, "rjid": rank_job_id}
    elif map_id:
        where = "map_id = :mid"
        params = {"mid": map_id}
    else:
        where = "rank_job_id = :rjid"
        params = {"rjid": rank_job_id}

    rows = conn.execute(
        text(f"""
            SELECT activity_type, work_ids, node_count, created_at
            FROM research_activity_log
            WHERE {where}
            ORDER BY created_at ASC
        """),
        params,
    ).mappings().all()

    if not rows:
        return {
            "entry_point": None,
            "activities": {},
            "coverage_pct": 0.0,
            "first_activity_at": None,
            "last_activity_at": None,
        }

    # Aggregate by activity type
    activities: Dict[str, Any] = {}
    first_at = None
    last_at = None
    entry_point = None

    for row in rows:
        atype = row["activity_type"]
        wids = row["work_ids"] or []
        nc = row["node_count"] or 0
        ts = row["created_at"]

        if first_at is None:
            first_at = ts
        last_at = ts

        # Detect entry point from first creation activity
        if entry_point is None:
            if atype == "citation_map_created":
                entry_point = "citation_map"
            elif atype == "rank_job_created":
                entry_point = "rank"

        if atype not in activities:
            activities[atype] = {"count": 0}

        activities[atype]["count"] += 1

        # Type-specific aggregation
        if atype == "methodology_compared":
            by_nc = activities[atype].setdefault("by_node_count", {})
            nc_key = str(nc)
            by_nc[nc_key] = by_nc.get(nc_key, 0) + 1
            compared = activities[atype].setdefault("work_ids_compared", [])
            for wid in wids:
                if wid not in compared:
                    compared.append(wid)
        elif atype in ("novelty_assessed", "timeline_per_node", "node_details_viewed"):
            wid_list = activities[atype].setdefault("work_ids", [])
            for wid in wids:
                if wid not in wid_list:
                    wid_list.append(wid)

    return {
        "entry_point": entry_point,
        "activities": activities,
        "first_activity_at": first_at.isoformat() if first_at else None,
        "last_activity_at": last_at.isoformat() if last_at else None,
    }
