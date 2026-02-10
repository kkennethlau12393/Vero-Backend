"""
Paper categorization using LLM paper types + citation-based heuristics.

The LLM classifies paper TYPE (foundational/methodology/review/application/etc).
Python determines OUTPUT CATEGORY using type + citations + year.

4 OUTPUT CATEGORIES:
- foundational: High-citation seminal papers that introduced key concepts
- methodology: Methods, algorithms, tools, frameworks
- reviews: Surveys, reviews, meta-analyses, tutorials
- applications: Clinical trials, case studies, real-world implementations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Minimum LLM relevance score - papers below this are filtered out
MIN_LLM_RELEVANCE_THRESHOLD = 0.50

# Category limits - 4 categories
DEFAULT_CATEGORY_LIMITS = {
    "foundational": 8,
    "methodology": 12,
    "reviews": 8,
    "applications": 10,
}

@dataclass
class DynamicThresholds:
    """Thresholds computed from the field's distribution."""
    legendary: int           # Top ~15% - field-defining papers
    foundational_min: int    # Top ~25% - high-impact papers
    methodology_min: int     # Top ~60% - decent citations
    review_min: int          # Top ~70% - validated reviews
    application_min: int     # Top ~85% - some validation
    min_foundational_age: int  # Median age - papers older than this can be foundational


def compute_dynamic_thresholds(
    all_citations: List[int],
    all_ages: Optional[List[int]] = None,
) -> DynamicThresholds:
    """Compute thresholds based on the field's distribution.

    Uses percentiles so niche fields have lower thresholds than high-citation fields.
    Age threshold is based on median age of papers in the pool.
    No hardcoded minimums - purely relative to the candidate pool.
    """
    if len(all_citations) < 5:
        # Too few papers - use zeros (everything passes)
        return DynamicThresholds(
            legendary=0,
            foundational_min=0,
            methodology_min=0,
            review_min=0,
            application_min=0,
            min_foundational_age=0,
        )

    sorted_cites = sorted(all_citations, reverse=True)
    n = len(sorted_cites)

    # Get citation values at percentile positions
    def get_percentile(pct: float) -> int:
        idx = min(int(n * pct), n - 1)
        return sorted_cites[idx]

    # All thresholds are purely percentile-based
    legendary = get_percentile(0.15)           # Top 15%
    foundational_min = get_percentile(0.25)    # Top 25%
    methodology_min = get_percentile(0.60)     # Top 60%
    review_min = get_percentile(0.70)          # Top 70%
    application_min = get_percentile(0.85)     # Top 85%

    # Compute age threshold: 25th percentile (older than 75% of papers)
    # Using 25th instead of median prevents mature fields from having too-high thresholds
    if all_ages and len(all_ages) >= 5:
        sorted_ages = sorted(all_ages)
        age_25th = sorted_ages[len(sorted_ages) // 4]
        min_foundational_age = max(2, age_25th)
    else:
        min_foundational_age = 2

    thresholds = DynamicThresholds(
        legendary=legendary,
        foundational_min=foundational_min,
        methodology_min=methodology_min,
        review_min=review_min,
        application_min=application_min,
        min_foundational_age=min_foundational_age,
    )

    logger.info(
        f"Dynamic thresholds: legendary={legendary}, foundational={foundational_min}, "
        f"methodology={methodology_min}, review={review_min}, application={application_min}"
    )

    return thresholds


def get_min_citations_for_age(age_years: int) -> int:
    """Get minimum citation threshold based on paper age.

    Older papers have had more time to accumulate citations, so we expect more.
    This filters out old obscure papers while allowing newer impactful ones.
    """
    if age_years >= 15:
        return 100  # 15+ years old: should have 100+ cites
    elif age_years >= 10:
        return 50   # 10-14 years: should have 50+ cites
    elif age_years >= 5:
        return 25   # 5-9 years: should have 25+ cites
    elif age_years >= 2:
        return 10   # 2-4 years: should have 10+ cites
    else:
        return 0    # 0-1 years: new papers, no floor


def determine_output_category(
    paper_type: str,
    citations: int,
    year: int,
    current_year: int,
    llm_relevance: float,
    thresholds: Optional[DynamicThresholds] = None,
) -> Optional[str]:
    """Determine output category using paper type + citations + year.

    Returns None if paper should be filtered out entirely.

    Logic:
    1. foundational: Top 15% citations + older than median age
    2. reviews: LLM-typed reviews
    3. applications: LLM-typed applications
    4. methodology: Everything else with decent citations
    """
    age_years = current_year - year if year else 0

    # Use dynamic thresholds (should always be provided, but default to zeros if not)
    if thresholds is None:
        thresholds = DynamicThresholds(
            legendary=0,
            foundational_min=0,
            methodology_min=0,
            review_min=0,
            application_min=0,
            min_foundational_age=2,
        )

    # === CATEGORY ROUTING ===

    is_old_enough = age_years >= thresholds.min_foundational_age
    is_highly_cited = citations >= thresholds.legendary  # Top 15%

    # Priority 1: Foundational - top 15% citations + old enough
    # Use lower relevance threshold (0.25 = LOW tier) to allow older terminology
    # but still filter out completely unrelated papers
    FOUNDATIONAL_MIN_RELEVANCE = 0.25
    if is_highly_cited and is_old_enough and llm_relevance >= FOUNDATIONAL_MIN_RELEVANCE:
        return "foundational"

    # For non-foundational papers, apply standard LLM relevance filter
    if llm_relevance < MIN_LLM_RELEVANCE_THRESHOLD:
        return None

    # Priority 2: Reviews - LLM-typed reviews with enough citations
    if paper_type == "review":
        if citations >= thresholds.review_min:
            return "reviews"
        else:
            return None

    # Priority 3: Applications - LLM-typed applications with validation
    if paper_type == "application":
        is_current_year = (year == current_year)
        if citations >= thresholds.application_min or (is_current_year and citations >= 0):
            return "applications"
        else:
            return None

    # Priority 4: Methodology - methods, tools, algorithms
    if paper_type in ("methodology", "foundational", "theoretical", "other"):
        # Use dynamic threshold, with recency exceptions
        if citations >= thresholds.methodology_min:
            return "methodology"
        elif age_years <= 1 and citations >= 5:
            return "methodology"
        elif age_years <= 3 and citations >= 10:
            return "methodology"
        else:
            return None

    # Fallback: methodology
    return "methodology"


def partition_results_by_category(
    ranked_items: List[Dict],
    llm_scores: Dict[str, Dict[str, Any]],
    category_limits: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Dict]]:
    """Partition ranked results into 4 categories using LLM types + citations.

    Args:
        ranked_items: List of ranked result dicts with 'work_id' key
        llm_scores: Dict mapping work_id to {"score": float, "paper_type": str}
        category_limits: Optional dict overriding default limits

    Returns:
        Dict with 4 category keys, each containing a list of items up to the limit.
    """
    limits = {**DEFAULT_CATEGORY_LIMITS, **(category_limits or {})}
    current_year = date.today().year

    # Compute dynamic thresholds based on citation and age distributions
    all_citations = []
    all_ages = []
    for item in ranked_items:
        preview = item.get("preview", {})
        citations = preview.get("cited_by_count", 0) or 0
        year = preview.get("year")
        all_citations.append(citations)
        if year and year <= current_year:
            all_ages.append(current_year - year)

    thresholds = compute_dynamic_thresholds(all_citations, all_ages)

    # Collect ALL papers per category (no limits yet)
    buckets: Dict[str, List[Dict]] = {
        "foundational": [],
        "methodology": [],
        "reviews": [],
        "applications": [],
    }

    filtered_stats = {"low_relevance": 0, "low_citations": 0, "bad_data": 0}

    for item in ranked_items:
        work_id = item.get("work_id")
        preview = item.get("preview", {})
        title = preview.get("title", "")
        year = preview.get("year")
        citations = preview.get("cited_by_count", 0) or 0

        # === DATA QUALITY FILTERS ===
        if not title or year is None:
            filtered_stats["bad_data"] += 1
            continue
        if year > current_year:
            filtered_stats["bad_data"] += 1
            continue
        if year >= current_year and citations == 0:
            filtered_stats["bad_data"] += 1
            continue

        # Get LLM data
        llm_data = llm_scores.get(work_id, {})
        relevance = llm_data.get("score", 0.0)
        paper_type = llm_data.get("paper_type", "other")

        # === DETERMINE CATEGORY ===
        category = determine_output_category(
            paper_type=paper_type,
            citations=citations,
            year=year,
            current_year=current_year,
            llm_relevance=relevance,
            thresholds=thresholds,
        )

        if category is None:
            if relevance < MIN_LLM_RELEVANCE_THRESHOLD:
                filtered_stats["low_relevance"] += 1
            else:
                filtered_stats["low_citations"] += 1
            continue

        buckets[category].append(item)

    # Sort each bucket and truncate to limits
    # - Foundational/Methodology/Reviews: by citations (highest first)
    # - Applications: blended 65% citations + 35% recency (recent applications matter)
    for category in buckets:
        if category == "applications" and buckets[category]:
            # Compute blended score for applications
            apps = buckets[category]
            citations_list = [x.get("preview", {}).get("cited_by_count", 0) or 0 for x in apps]
            years_list = [x.get("preview", {}).get("year", 2000) or 2000 for x in apps]

            # Min-max normalize
            cite_min, cite_max = min(citations_list), max(citations_list)
            year_min, year_max = min(years_list), max(years_list)
            cite_range = cite_max - cite_min if cite_max > cite_min else 1
            year_range = year_max - year_min if year_max > year_min else 1

            def app_score(x):
                cites = x.get("preview", {}).get("cited_by_count", 0) or 0
                year = x.get("preview", {}).get("year", 2000) or 2000
                cite_norm = (cites - cite_min) / cite_range
                year_norm = (year - year_min) / year_range  # Higher year = more recent = higher score
                return 0.65 * cite_norm + 0.35 * year_norm

            apps.sort(key=app_score, reverse=True)
        else:
            # Other categories: sort by citations only
            buckets[category].sort(
                key=lambda x: x.get("preview", {}).get("cited_by_count", 0) or 0,
                reverse=True,
            )
        buckets[category] = buckets[category][:limits.get(category, 10)]

    # Log filtering stats
    total_filtered = sum(filtered_stats.values())
    if total_filtered > 0:
        logger.info(
            f"Filtered {total_filtered} papers: "
            f"{filtered_stats['low_relevance']} low relevance, "
            f"{filtered_stats['low_citations']} low citations, "
            f"{filtered_stats['bad_data']} bad data"
        )

    counts = {k: len(v) for k, v in buckets.items()}
    logger.info(f"Partitioned by category: {counts}")

    return buckets
