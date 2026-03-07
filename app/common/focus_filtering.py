"""
Focus filtering for structured query results.

Applies time-based and type-based filters based on user's focus and depth
preferences. Filters aggressively but falls back gracefully to ensure
results are never empty.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _get_field(paper: Dict[str, Any], field: str, default: Any = None) -> Any:
    """Get field from paper dict, checking preview subdict as fallback."""
    val = paper.get(field)
    if val is not None:
        return val
    preview = paper.get("preview")
    if isinstance(preview, dict):
        return preview.get(field, default)
    return default


MIN_RESULTS = 10  # Never filter below this count


def _filter_with_fallback(
    papers: List[Dict[str, Any]],
    strict_fn,
    relaxed_fn,
    sort_fn=None,
) -> List[Dict[str, Any]]:
    """Apply strict filter, fall back to relaxed, then to full list.

    1. Try strict filter — if >= MIN_RESULTS, use it
    2. Try relaxed filter — if >= MIN_RESULTS, use it
    3. Fall back to full list (just reorder via sort_fn)
    """
    strict = strict_fn(papers)
    if len(strict) >= MIN_RESULTS:
        return sort_fn(strict) if sort_fn else strict

    relaxed = relaxed_fn(papers)
    if len(relaxed) >= MIN_RESULTS:
        return sort_fn(relaxed) if sort_fn else relaxed

    # Fallback: return all, optionally sorted
    return sort_fn(papers) if sort_fn else papers


def apply_focus_filter(papers: List[Dict[str, Any]], focus: str) -> List[Dict[str, Any]]:
    """Filter papers based on focus preference.

    - foundational: high-citation older papers (filter, not just sort)
    - recent: last 5 years only (relaxed: last 10 years)
    - surveys: only review/survey papers (relaxed: surveys first + others)
    - all_time: no filtering
    """
    current_year = datetime.now().year

    if focus == "foundational":
        return _filter_with_fallback(
            papers,
            strict_fn=lambda ps: [
                p for p in ps
                if _get_field(p, "cited_by_count", 0) >= 100
                and _get_field(p, "year", current_year) <= current_year - 3
            ],
            relaxed_fn=lambda ps: [
                p for p in ps
                if _get_field(p, "cited_by_count", 0) >= 50
            ],
            sort_fn=lambda ps: sorted(
                ps, key=lambda p: _get_field(p, "cited_by_count", 0), reverse=True
            ),
        )

    elif focus == "recent":
        return _filter_with_fallback(
            papers,
            strict_fn=lambda ps: sorted(
                [p for p in ps if _get_field(p, "year", 0) >= current_year - 5],
                key=lambda p: _get_field(p, "year", 0),
                reverse=True,
            ),
            relaxed_fn=lambda ps: sorted(
                [p for p in ps if _get_field(p, "year", 0) >= current_year - 10],
                key=lambda p: _get_field(p, "year", 0),
                reverse=True,
            ),
        )

    elif focus == "surveys":
        survey_terms = {
            "survey", "review", "systematic review", "meta-analysis",
            "overview", "tutorial", "comprehensive review", "state of the art",
            "literature review", "critical review",
        }

        def is_survey(p):
            title = _get_field(p, "title", "").lower()
            return any(term in title for term in survey_terms)

        surveys = [p for p in papers if is_survey(p)]
        others = [p for p in papers if not is_survey(p)]

        # Strict: only surveys if enough
        if len(surveys) >= 5:
            return surveys
        # Fallback: surveys first, then others
        return surveys + others

    else:  # all_time
        return papers


def apply_depth_limit(papers: List[Dict[str, Any]], depth: str) -> List[Dict[str, Any]]:
    """Limit result count based on depth preference.

    - high_level: top 15 papers (key papers only)
    - comprehensive: top 50 papers (thorough coverage)
    """
    if depth == "high_level":
        return papers[:15]
    else:  # comprehensive
        return papers[:50]
