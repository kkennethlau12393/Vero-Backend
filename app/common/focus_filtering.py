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


def _get_llm_relevance(paper: Dict[str, Any]) -> float:
    """Extract LLM relevance score from paper's score breakdown."""
    breakdown = paper.get("score_breakdown", paper.get("breakdown", {}))
    raw = breakdown.get("raw", {})
    return raw.get("llm_relevance", 0)


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
    """Filter papers based on focus preference. Returns flat ranked list.

    - foundational: high-citation papers (>=500 strict, >=100 relaxed), sorted by citations
    - recent: last 3 years (relaxed: 5 years), sorted by year desc then citations
    - surveys: papers classified as review/survey by LLM or title, cap 25
    - all_time: no filtering
    """
    current_year = datetime.now().year

    if focus == "foundational":
        result = _filter_with_fallback(
            papers,
            strict_fn=lambda ps: [
                p for p in ps
                if _get_field(p, "cited_by_count", 0) >= 500
                and _get_llm_relevance(p) >= 0.55
            ],
            relaxed_fn=lambda ps: [
                p for p in ps
                if _get_field(p, "cited_by_count", 0) >= 100
                and _get_llm_relevance(p) >= 0.45
            ],
            sort_fn=lambda ps: sorted(
                ps, key=lambda p: _get_field(p, "cited_by_count", 0), reverse=True
            ),
        )
        return result[:25]

    elif focus == "recent":
        result = _filter_with_fallback(
            papers,
            strict_fn=lambda ps: [
                p for p in ps if _get_field(p, "year", 0) >= current_year - 3
            ],
            relaxed_fn=lambda ps: [
                p for p in ps if _get_field(p, "year", 0) >= current_year - 5
            ],
            sort_fn=lambda ps: sorted(
                ps,
                key=lambda p: (_get_field(p, "year", 0), _get_field(p, "cited_by_count", 0)),
                reverse=True,
            ),
        )
        return result[:25]

    elif focus == "surveys":
        survey_terms = {
            "survey", "review", "systematic review", "meta-analysis",
            "overview", "tutorial", "comprehensive review", "state of the art",
            "literature review", "critical review",
        }

        def is_survey(p):
            title = _get_field(p, "title", "").lower()
            paper_type = p.get("breakdown", {}).get("paper_type", "")
            if paper_type in ("review", "survey"):
                return True
            return any(term in title for term in survey_terms)

        surveys = [p for p in papers if is_survey(p)]
        others = [p for p in papers if not is_survey(p)]

        if len(surveys) >= 5:
            return surveys[:25]
        return (surveys + others)[:25]

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
