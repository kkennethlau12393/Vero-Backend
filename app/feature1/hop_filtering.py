"""
Citation map hop filtering for structured queries.

Filters papers at each hop of citation network expansion based on
relevance to the structured query, preventing topic drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.common.intersection_scoring import score_candidate


# Relevance thresholds per expansion mode and hop level
_THRESHOLDS = {
    "narrow":      {0: 0.4, 1: 0.5, 2: 0.6},
    "foundations":  {0: 0.2, 1: 0.3, 2: 0.4},
    "wide":        {0: 0.1, 1: 0.15, 2: 0.2},
}


def filter_hop_papers(
    papers: List[Dict[str, Any]],
    structured: Dict[str, Optional[str]],
    expansion: str,
    hop_level: int,
) -> List[Dict[str, Any]]:
    """Filter papers at each hop based on expansion preference and relevance.

    Parameters
    ----------
    papers : list[dict]
        Papers to filter. Must have 'title' and optionally 'abstract', 'cited_by_count'.
    structured : dict
        Decomposed query with keys: topic, domain, aspect.
    expansion : str
        One of: narrow, foundations, wide.
    hop_level : int
        Current hop distance from seed (0, 1, or 2).

    Returns
    -------
    list[dict]
        Filtered papers with '_relevance_score' added.
    """
    threshold = _THRESHOLDS.get(expansion, _THRESHOLDS["foundations"]).get(hop_level, 0.4)

    filtered = []
    for p in papers:
        score = score_candidate(p, structured, "intersection")

        # Exception: very high citation papers pass at reduced threshold
        # These are likely foundational papers worth showing on the map
        if expansion == "foundations" and p.get("cited_by_count", 0) > 1000:
            score = max(score, threshold + 0.1)

        if score >= threshold:
            p["_relevance_score"] = score
            filtered.append(p)

    return filtered


def get_seed_count(map_focus: str) -> int:
    """Determine number of seed papers based on map focus."""
    return {
        "landscape": 10,
        "core_cluster": 5,
        "evolution": 8,
    }.get(map_focus, 5)


def get_max_papers(map_size: str) -> int:
    """Determine max papers based on map size preference."""
    return {
        "small": 20,
        "medium": 40,
        "large": 60,
    }.get(map_size, 40)
