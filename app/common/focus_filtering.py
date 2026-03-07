"""
Focus filtering for structured query results.

Applies time-based and type-based filters/reordering based on user's
focus and depth preferences.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _get_field(paper: Dict[str, Any], field: str, default: Any = None) -> Any:
    """Get a field from a paper dict, checking preview subdict as fallback.

    Ranked result items store metadata under 'preview'; raw paper dicts
    store it at the top level. This helper checks both.
    """
    val = paper.get(field)
    if val is not None:
        return val
    preview = paper.get("preview")
    if isinstance(preview, dict):
        return preview.get(field, default)
    return default


def apply_focus_filter(papers: List[Dict[str, Any]], focus: str) -> List[Dict[str, Any]]:
    """Filter/reorder papers based on focus preference.

    Parameters
    ----------
    papers : list[dict]
        Papers to filter. Each should have 'title', 'year', 'cited_by_count'
        either at top level or inside a 'preview' subdict.
    focus : str
        One of: foundational, recent, surveys, all_time.

    Returns
    -------
    list[dict]
        Reordered (not necessarily reduced) paper list.
    """
    if focus == "foundational":
        # Prefer high-citation, older papers
        return sorted(papers, key=lambda p: _get_field(p, "cited_by_count", 0), reverse=True)

    elif focus == "recent":
        # Filter to last 3 years, boost by recency
        current_year = datetime.now().year
        recent = [p for p in papers if _get_field(p, "year", 0) >= current_year - 3]
        older = [p for p in papers if _get_field(p, "year", 0) < current_year - 3]
        # Return recent first, then older as fallback
        return recent + older

    elif focus == "surveys":
        # Boost papers with survey/review in title
        survey_terms = {"survey", "review", "systematic review", "meta-analysis", "overview", "tutorial"}
        surveys = []
        others = []
        for p in papers:
            title_lower = _get_field(p, "title", "").lower()
            if any(term in title_lower for term in survey_terms):
                surveys.append(p)
            else:
                others.append(p)
        return surveys + others

    else:  # all_time
        return papers


def apply_depth_limit(papers: List[Dict[str, Any]], depth: str) -> List[Dict[str, Any]]:
    """Limit results based on depth preference.

    Parameters
    ----------
    papers : list[dict]
        Papers to limit.
    depth : str
        One of: high_level, comprehensive.

    Returns
    -------
    list[dict]
        Truncated paper list.
    """
    if depth == "high_level":
        return papers[:20]
    else:  # comprehensive
        return papers[:60]
