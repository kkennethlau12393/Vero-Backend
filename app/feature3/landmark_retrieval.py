"""
Landmark retrieval for Feature 3.

This module finds influential papers in the same topic/field as the target paper,
spread across time to provide historical context for novelty assessment.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

# OpenAlex API settings
OPENALEX_TIMEOUT = 10

# Bounds for landmark count
MIN_LANDMARKS = 3
MAX_LANDMARKS = 6


def _verify_topic_from_openalex(work_id: str) -> Optional[str]:
    """Fetch and return the actual topic_id from OpenAlex for a work.

    Returns the topic_id (e.g., "T12535") or None if lookup fails.
    """
    if not work_id or not work_id.startswith("W"):
        return None

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        resp = requests.get(url, timeout=OPENALEX_TIMEOUT)
        if resp.status_code != 200:
            return None

        data = resp.json()
        primary_topic = data.get("primary_topic", {})
        topic_url = primary_topic.get("id")
        if topic_url and "/" in topic_url:
            return topic_url.rsplit("/", 1)[-1]
        return None
    except Exception as e:
        logger.debug(f"OpenAlex topic lookup failed for {work_id}: {e}")
        return None


def _update_work_topic(conn: Connection, work_id: str, correct_topic_id: str) -> None:
    """Update a work's topic_id in the database."""
    try:
        conn.execute(
            text("""
                UPDATE works SET primary_topic_id = :topic_id
                WHERE work_id = :work_id
            """),
            {"work_id": work_id, "topic_id": correct_topic_id},
        )
        conn.commit()
        logger.info(f"Updated topic for {work_id} to {correct_topic_id}")
    except Exception as e:
        logger.warning(f"Failed to update topic for {work_id}: {e}")


def calc_landmark_count(span_years: int) -> int:
    """
    Dynamically calculate landmark count based on field span.

    Uses logarithmic scaling - grows quickly for young fields,
    slowly for mature fields. Natural ceiling effect.

    Formula: 3 + log2(span / 5), clamped to [3, 6]

    Results:
    - span=2 years  → 3 landmarks (emerging)
    - span=5 years  → 3 landmarks
    - span=10 years → 4 landmarks
    - span=20 years → 5 landmarks
    - span=40+ years → 6 landmarks (mature)
    """
    if span_years <= 0:
        return MIN_LANDMARKS

    raw = MIN_LANDMARKS + math.log2(max(1, span_years / 5))
    return min(MAX_LANDMARKS, max(MIN_LANDMARKS, round(raw)))


def get_topic_landmarks(
    conn: Connection,
    topic_id: Optional[str],
    before_year: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Find influential papers in the same topic, spread across time.

    Uses a fully dynamic algorithm:
    1. Calculate the topic's time span
    2. Determine landmark count using logarithmic scaling
    3. Divide time into periods based on count
    4. Pick highest-cited paper from each period

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    if not topic_id:
        logger.info("No topic_id provided, skipping landmark retrieval")
        return []

    if not before_year:
        before_year = 2025  # Default to current year

    # Step 1: Find the topic's time range
    time_range = _get_topic_time_range(conn, topic_id, before_year)
    if not time_range:
        logger.info(f"No papers found in topic {topic_id} before {before_year}")
        return []

    earliest_year, latest_year = time_range
    span = latest_year - earliest_year

    if span <= 0:
        # All papers in same year - just return top by citations
        return _get_top_papers_in_topic(conn, topic_id, before_year, limit=MIN_LANDMARKS)

    # Step 2: Calculate landmark count
    landmark_count = calc_landmark_count(span)
    logger.info(
        f"Topic {topic_id}: span={span} years ({earliest_year}-{latest_year}), "
        f"landmark_count={landmark_count}"
    )

    # Step 3: Divide into periods
    period_length = span / landmark_count
    periods = []
    for i in range(landmark_count):
        period_start = earliest_year + i * period_length
        period_end = earliest_year + (i + 1) * period_length
        periods.append((period_start, period_end))

    # Step 4: Pick highest-cited paper from each period
    landmarks = []
    for period_start, period_end in periods:
        paper = _get_top_paper_in_period(
            conn, topic_id, before_year, period_start, period_end
        )
        if paper:
            landmarks.append(paper)

    # Step 5: Verify landmarks have correct topics (DB data may be stale)
    # This prevents cross-domain contamination from incorrectly-classified papers
    verified_landmarks = []
    for lm in landmarks:
        lm_work_id = lm.get("work_id")
        lm_db_topic = lm.get("primary_topic_id")

        # Verify against OpenAlex for high-cited papers (worth the API call)
        if lm.get("cited_by_count", 0) > 1000:
            actual_topic = _verify_topic_from_openalex(lm_work_id)
            if actual_topic and actual_topic != lm_db_topic:
                # Stale data - update DB and skip this landmark
                logger.warning(
                    f"Stale topic for landmark {lm_work_id}: "
                    f"DB={lm_db_topic}, OpenAlex={actual_topic}. Updating and skipping."
                )
                _update_work_topic(conn, lm_work_id, actual_topic)
                continue
            elif actual_topic and actual_topic == topic_id:
                # Verified correct - keep it
                verified_landmarks.append(lm)
            elif not actual_topic:
                # Couldn't verify - keep it (OpenAlex might be down)
                verified_landmarks.append(lm)
        else:
            # Lower-cited papers - trust DB data
            verified_landmarks.append(lm)

    if len(verified_landmarks) < len(landmarks):
        logger.info(
            f"Topic verification: {len(landmarks)} landmarks -> {len(verified_landmarks)} "
            f"(removed {len(landmarks) - len(verified_landmarks)} with stale topics)"
        )

    logger.info(f"Found {len(verified_landmarks)} verified landmarks for topic {topic_id}")
    return verified_landmarks


def _get_topic_time_range(
    conn: Connection,
    topic_id: str,
    before_year: int,
) -> Optional[tuple[int, int]]:
    """Get the earliest and latest year for papers in this topic."""
    row = conn.execute(
        text("""
            SELECT MIN(year) as earliest, MAX(year) as latest
            FROM works
            WHERE primary_topic_id = :topic_id
            AND year IS NOT NULL
            AND year < :before_year
        """),
        {"topic_id": topic_id, "before_year": before_year},
    ).mappings().first()

    if not row or row["earliest"] is None:
        return None

    return (int(row["earliest"]), int(row["latest"]))


def _get_top_paper_in_period(
    conn: Connection,
    topic_id: str,
    before_year: int,
    period_start: float,
    period_end: float,
) -> Optional[Dict[str, Any]]:
    """Get the highest-cited paper in the given time period."""
    row = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id
            FROM works
            WHERE primary_topic_id = :topic_id
            AND year IS NOT NULL
            AND year >= :period_start
            AND year < :period_end
            AND year < :before_year
            ORDER BY cited_by_count DESC NULLS LAST
            LIMIT 1
        """),
        {
            "topic_id": topic_id,
            "period_start": int(period_start),
            "period_end": int(period_end) + 1,  # Include the end year
            "before_year": before_year,
        },
    ).mappings().first()

    if not row:
        return None

    return {
        "work_id": row["work_id"],
        "title": row["title"],
        "year": row["year"],
        "cited_by_count": int(row["cited_by_count"] or 0),
        "abstract": row["abstract"],
        "category": row["category"],
        "primary_topic_id": row["primary_topic_id"],
    }


def _get_top_papers_in_topic(
    conn: Connection,
    topic_id: str,
    before_year: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fallback: get top papers by citation count (no period filtering)."""
    rows = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id
            FROM works
            WHERE primary_topic_id = :topic_id
            AND year IS NOT NULL
            AND year < :before_year
            ORDER BY cited_by_count DESC NULLS LAST
            LIMIT :limit
        """),
        {"topic_id": topic_id, "before_year": before_year, "limit": limit},
    ).mappings().all()

    return [
        {
            "work_id": row["work_id"],
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": int(row["cited_by_count"] or 0),
            "abstract": row["abstract"],
            "category": row["category"],
            "primary_topic_id": row["primary_topic_id"],
        }
        for row in rows
    ]
