"""
Node timeline builder for Feature 3.

This module builds a temporal timeline around a target paper, showing:
- Backward: References + Landmarks (papers before the target)
- Forward: Citing papers (papers after the target)

Papers are grouped by era (decades) for visualization.

Note: Paper fetching and caching is handled by paper_cache.py for sharing
between timeline and novelty assessment features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Literal

from sqlalchemy.engine import Connection

from app.feature3.paper_cache import get_citing_papers

logger = logging.getLogger(__name__)

MAX_CITING_PAPERS = 20  # Limit for timeline (more than grounding supplement)


# ============================================================================
# Era Grouping Utilities
# ============================================================================

def get_era_label(year: Optional[int]) -> str:
    """Convert a year to an era label (decade)."""
    if year is None:
        return "Unknown"
    decade = (year // 10) * 10
    return f"{decade}s"


def group_papers_by_era(
    papers: List[Dict[str, Any]],
    relationship: Literal["reference", "landmark", "citing"],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group papers by era (decade).

    Returns dict: era_label -> list of papers with relationship added.
    """
    era_map: Dict[str, List[Dict[str, Any]]] = {}

    for paper in papers:
        year = paper.get("year")
        era = get_era_label(year)

        paper_entry = {
            "work_id": paper.get("work_id"),
            "title": paper.get("title"),
            "year": year,
            "cited_by_count": paper.get("cited_by_count") or 0,
            "relationship": relationship,
        }

        if era not in era_map:
            era_map[era] = []
        era_map[era].append(paper_entry)

    return era_map


def merge_era_maps(*era_maps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Merge multiple era maps into one."""
    merged: Dict[str, List[Dict[str, Any]]] = {}

    for era_map in era_maps:
        for era, papers in era_map.items():
            if era not in merged:
                merged[era] = []
            merged[era].extend(papers)

    return merged


def era_map_to_sections(
    era_map: Dict[str, List[Dict[str, Any]]],
    sort_ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert era map to sorted list of TimelineSection dicts.

    Args:
        era_map: Dict of era -> papers
        sort_ascending: If True, sort eras chronologically (oldest first)
    """
    sections = []

    for era, papers in era_map.items():
        # Sort papers within era by citation count (most cited first)
        sorted_papers = sorted(papers, key=lambda p: -(p.get("cited_by_count") or 0))
        sections.append({
            "era": era,
            "papers": sorted_papers,
        })

    # Sort sections by era (extract decade number for sorting)
    def era_sort_key(section: Dict) -> int:
        era = section["era"]
        if era == "Unknown":
            return 9999 if sort_ascending else -9999
        try:
            return int(era.replace("s", ""))
        except ValueError:
            return 9999 if sort_ascending else -9999

    sections.sort(key=era_sort_key, reverse=not sort_ascending)
    return sections




# ============================================================================
# Timeline Builder
# ============================================================================

def build_node_timeline(
    conn: Connection,
    work_id: str,
    target_year: Optional[int],
    references: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    include_impact_analysis: bool = True,
    target_title: Optional[str] = None,
    target_abstract: Optional[str] = None,
    target_cited_by_count: int = 0,
) -> Dict[str, Any]:
    """
    Build a complete timeline for a node.

    Args:
        conn: Database connection
        work_id: Target paper's OpenAlex ID
        target_year: Target paper's publication year
        references: Papers the target cites (from referenced_works_json)
        landmarks: Landmark papers for context (from landmark_works_json)
        include_impact_analysis: If True, run LLM impact analysis
        target_title: Target paper's title (for impact analysis)
        target_abstract: Target paper's abstract (for impact analysis)
        target_cited_by_count: Target paper's citation count (for impact analysis)

    Returns:
        NodeTimeline dict with backward, forward, impact_analysis, and target info
    """
    # Group backward papers by era
    ref_eras = group_papers_by_era(references, "reference")
    landmark_eras = group_papers_by_era(landmarks, "landmark")
    backward_eras = merge_era_maps(ref_eras, landmark_eras)

    # Get citing papers (forward) - uses shared cache with lazy expansion
    citing_papers = get_citing_papers(conn, work_id, limit=MAX_CITING_PAPERS)
    forward_eras = group_papers_by_era(citing_papers, "citing")

    # Convert to sections
    backward_sections = era_map_to_sections(backward_eras, sort_ascending=True)
    forward_sections = era_map_to_sections(forward_eras, sort_ascending=True)

    # Run impact analysis if requested
    impact_analysis = None
    if include_impact_analysis and citing_papers:
        from app.feature3.paper_impact_analytics import analyze_paper_impact

        logger.info(f"Running impact analysis for {work_id}")
        impact_analysis = analyze_paper_impact(
            title=target_title or "",
            abstract=target_abstract,
            year=target_year,
            cited_by_count=target_cited_by_count,
            references=references,
            citing_papers=citing_papers,
        )
        logger.info(
            f"Impact analysis complete: paradigm_shift={impact_analysis.get('is_paradigm_shift')}, "
            f"score={impact_analysis.get('impact_score')}"
        )

    return {
        "target_work_id": work_id,
        "target_year": target_year,
        "backward": backward_sections,
        "forward": forward_sections,
        "impact_analysis": impact_analysis,
    }
