"""
Temporal map service for Feature 2.

This module builds temporal maps from ranked query results, grouping
papers by era (decade) and identifying milestone papers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)


def get_era_label(year: Optional[int]) -> str:
    """Convert a year to an era label (decade)."""
    if year is None:
        return "Unknown"
    decade = (year // 10) * 10
    return f"{decade}s"


def get_era_bounds(era_label: str) -> tuple[int, int]:
    """Get start and end years for an era label."""
    if era_label == "Unknown":
        return (0, 0)
    try:
        decade = int(era_label.replace("s", ""))
        return (decade, decade + 9)
    except ValueError:
        return (0, 0)


def get_ranked_papers_for_temporal_map(
    conn: Connection,
    rank_job_id: UUID,
    subtopic_id: Optional[str] = None,
    topic_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get ranked papers for building a temporal map.

    Args:
        conn: Database connection
        rank_job_id: ID of the rank job
        subtopic_id: Optional subtopic filter (uses _topic_id mapping)
        topic_id: Optional direct topic ID filter

    Returns list of dicts with: work_id, rank_index, title, year, cited_by_count, primary_topic_id
    """
    query = """
        SELECT
            rr.work_id,
            rr.rank_index,
            w.title,
            w.year,
            w.cited_by_count,
            w.primary_topic_id
        FROM rank_results rr
        JOIN works w ON w.work_id = rr.work_id
        WHERE rr.rank_job_id = :rank_job_id
    """

    params: Dict[str, Any] = {"rank_job_id": rank_job_id}

    # Filter by topic if specified
    if topic_id:
        query += " AND w.primary_topic_id = :topic_id"
        params["topic_id"] = topic_id

    query += " ORDER BY rr.rank_index"

    rows = conn.execute(text(query), params).mappings().all()

    return [
        {
            "work_id": row["work_id"],
            "rank_index": row["rank_index"],
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": row["cited_by_count"] or 0,
            "primary_topic_id": row["primary_topic_id"],
        }
        for row in rows
    ]


def identify_milestones(
    papers: List[Dict[str, Any]],
    era_papers: Dict[str, List[Dict[str, Any]]],
    percentile: float = 0.9,
) -> set:
    """
    Identify milestone papers based on citation count.

    A paper is a milestone if its citation count is in the top percentile
    for its era.

    Returns set of work_ids that are milestones.
    """
    milestones = set()

    for era, era_paper_list in era_papers.items():
        if not era_paper_list:
            continue

        # Calculate citation threshold for this era
        citations = sorted([p["cited_by_count"] for p in era_paper_list])
        threshold_idx = int(len(citations) * percentile)
        threshold = citations[threshold_idx] if threshold_idx < len(citations) else citations[-1]

        # Mark papers above threshold as milestones
        for paper in era_paper_list:
            if paper["cited_by_count"] >= threshold and paper["cited_by_count"] > 100:
                milestones.add(paper["work_id"])

    return milestones


def group_papers_by_era(
    papers: List[Dict[str, Any]],
    milestones: Optional[set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Group papers by era (decade) and build era summaries.

    Returns dict: era_label -> era dict with papers and metadata
    """
    era_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for paper in papers:
        era = get_era_label(paper.get("year"))
        paper_entry = {
            "work_id": paper["work_id"],
            "title": paper.get("title"),
            "year": paper.get("year"),
            "cited_by_count": paper.get("cited_by_count", 0),
            "is_milestone": paper["work_id"] in milestones if milestones else False,
            "rank_in_results": paper.get("rank_index"),
        }
        era_map[era].append(paper_entry)

    # Build era summaries
    result = {}
    for era, era_papers in era_map.items():
        start_year, end_year = get_era_bounds(era)
        milestone_count = sum(1 for p in era_papers if p.get("is_milestone", False))

        # Sort by citation count within era
        sorted_papers = sorted(era_papers, key=lambda p: -(p.get("cited_by_count") or 0))

        result[era] = {
            "label": era,
            "start_year": start_year,
            "end_year": end_year,
            "papers": sorted_papers,
            "milestone_count": milestone_count,
            "is_breakthrough_era": False,  # Will be set by analytics
        }

    return result


def build_temporal_map(
    engine: Engine,
    *,
    rank_job_id: UUID,
    subtopic_id: Optional[str] = None,
    topic_id: Optional[str] = None,
    scope_label: Optional[str] = None,
    include_analytics: bool = False,
) -> Dict[str, Any]:
    """
    Build a temporal map from ranked query results.

    Args:
        engine: Database engine
        rank_job_id: ID of the rank job
        subtopic_id: Optional subtopic filter
        topic_id: Optional direct topic ID filter
        scope_label: Label for the scope (e.g., "Machine Learning")
        include_analytics: Whether to include breakthrough/evolution analytics

    Returns:
        TemporalMapResponse dict
    """
    with engine.connect() as conn:
        # Get papers
        papers = get_ranked_papers_for_temporal_map(
            conn,
            rank_job_id,
            subtopic_id=subtopic_id,
            topic_id=topic_id,
        )

        if not papers:
            logger.warning(f"No papers found for temporal map: {rank_job_id}")
            return {
                "rank_job_id": str(rank_job_id),
                "scope": "subtopic" if subtopic_id or topic_id else "broad",
                "scope_label": scope_label or "Unknown",
                "topic_id": topic_id,
                "subtopic_id": subtopic_id,
                "eras": [],
                "analytics": None,
            }

        # Group by era first to calculate milestone thresholds
        era_papers_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for paper in papers:
            era = get_era_label(paper.get("year"))
            era_papers_raw[era].append(paper)

        # Identify milestones
        milestones = identify_milestones(papers, dict(era_papers_raw))

        # Build final era groupings
        era_data = group_papers_by_era(papers, milestones)

        # Sort eras chronologically
        sorted_eras = []
        for era in sorted(era_data.keys(), key=lambda e: get_era_bounds(e)[0]):
            sorted_eras.append(era_data[era])

        # Build analytics if requested
        analytics = None
        if include_analytics:
            from app.feature2.temporal_analytics import analyze_temporal_map

            analytics = analyze_temporal_map(
                conn,
                papers=papers,
                era_data=era_data,
                topic_id=topic_id,
            )

        return {
            "rank_job_id": str(rank_job_id),
            "scope": "subtopic" if subtopic_id or topic_id else "broad",
            "scope_label": scope_label or "Research Results",
            "topic_id": topic_id,
            "subtopic_id": subtopic_id,
            "eras": sorted_eras,
            "analytics": analytics,
        }
