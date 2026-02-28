"""
Landmark retrieval for Feature 3.

This module finds influential papers in the same topic/field as the target paper,
spread across time to provide historical context for novelty assessment.

Key design: Query OpenAlex and Semantic Scholar for top-cited papers in the topic,
not just what's already in our DB. Insert discovered papers into the works table
so they're available for grounding.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.paper_identity import decode_openalex_abstract
from app.shared.s2_keys import get_s2_headers

logger = logging.getLogger(__name__)

# API settings
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
OPENALEX_TIMEOUT = 15
S2_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
S2_DELAY = 1.0  # 1 request/second rate limit

# Bounds for landmark count
MIN_LANDMARKS = 3
MAX_LANDMARKS = 6


def _verify_topic_from_openalex(work_id: str) -> Optional[str]:
    """Fetch and return the actual topic_id from OpenAlex for a work."""
    if not work_id or not work_id.startswith("W"):
        return None

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
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
    except Exception as e:
        logger.warning(f"Failed to update topic for {work_id}: {e}")


def calc_landmark_count(span_years: int) -> int:
    """
    Dynamically calculate landmark count based on field span.

    Formula: 3 + log2(span / 5), clamped to [3, 6]
    """
    if span_years <= 0:
        return MIN_LANDMARKS

    raw = MIN_LANDMARKS + math.log2(max(1, span_years / 5))
    return min(MAX_LANDMARKS, max(MIN_LANDMARKS, round(raw)))


# ---------------------------------------------------------------------------
# External source retrieval
# ---------------------------------------------------------------------------

def _search_openalex_topic_landmarks(
    topic_id: str,
    before_year: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Query OpenAlex for top-cited papers in a topic.
    Returns papers sorted by citation count.
    """
    for attempt in range(MAX_RETRIES):
        try:
            url = "https://api.openalex.org/works"
            params = {
                "filter": f"primary_topic.id:T{topic_id.lstrip('T')},publication_year:<{before_year}",
                "sort": "cited_by_count:desc",
                "per-page": limit,
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)

            if resp.status_code == 429:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"OpenAlex 429 for landmarks, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            results = resp.json().get("results", [])

            papers = []
            for w in results:
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue

                wid = wid_full.rsplit("/", 1)[-1]
                primary_topic = w.get("primary_topic", {})
                pt_id = None
                if primary_topic and primary_topic.get("id"):
                    pt_url = primary_topic["id"]
                    if "/" in pt_url:
                        pt_id = pt_url.rsplit("/", 1)[-1]

                oa_authors = [
                    au.get("author", {}).get("display_name") or au.get("display_name")
                    for au in w.get("authorships", [])
                    if au.get("author", {}).get("display_name") or au.get("display_name")
                ]

                papers.append({
                    "work_id": wid,
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract": decode_openalex_abstract(w.get("abstract_inverted_index")),
                    "category": None,
                    "primary_topic_id": pt_id,
                    "source": "openalex",
                    "authors": oa_authors,
                })

            logger.info(f"OpenAlex topic landmarks: {len(papers)} papers for topic {topic_id}")
            return papers

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"OpenAlex topic landmark search failed: {e}")
    return []


def _search_s2_landmarks(
    query: str,
    before_year: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Query Semantic Scholar for highly-cited papers matching a query.
    Uses bulk endpoint with year filter.
    """
    for attempt in range(MAX_RETRIES):
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
            params = {
                "query": query,
                "fields": "paperId,title,year,citationCount,externalIds,abstract,authors",
                "limit": min(100, limit * 5),  # Fetch more to filter
                "year": f"-{before_year - 1}",
            }
            resp = requests.get(url, params=params, headers=get_s2_headers(), timeout=S2_TIMEOUT)

            if resp.status_code == 429:
                backoff = 2.0 * (2 ** attempt)
                logger.info(f"S2 429 for landmarks, waiting {backoff}s")
                time.sleep(backoff)
                continue

            if resp.status_code == 400:
                # Query too broad, try regular endpoint
                logger.info("S2 bulk 400, skipping")
                return []

            resp.raise_for_status()
            data = resp.json()
            s2_papers = data.get("data", [])

            # Sort by citation count and take top N
            s2_papers.sort(key=lambda p: -(p.get("citationCount") or 0))
            s2_papers = s2_papers[:limit]

            papers = []
            for p in s2_papers:
                ext_ids = p.get("externalIds") or {}
                oa_id = ext_ids.get("CorpusId")  # Not OpenAlex ID
                doi = ext_ids.get("DOI")

                # Try to get OpenAlex work_id from externalIds
                work_id = None
                if "DBLP" in ext_ids or doi:
                    # We'll resolve via DOI later if needed
                    work_id = f"S{p.get('paperId', '')}"
                else:
                    work_id = f"S{p.get('paperId', '')}"

                s2_authors = [a.get("name") for a in (p.get("authors") or []) if a.get("name")]

                papers.append({
                    "work_id": work_id,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "cited_by_count": p.get("citationCount") or 0,
                    "abstract": p.get("abstract"),
                    "category": None,
                    "primary_topic_id": None,
                    "source": "s2",
                    "doi": doi,
                    "s2_paper_id": p.get("paperId"),
                    "authors": s2_authors,
                })

            logger.info(f"S2 landmarks: {len(papers)} papers for query '{query[:40]}'")
            time.sleep(S2_DELAY)  # Rate limit
            return papers

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                is_429 = "429" in str(e)
                if is_429:
                    backoff = 2.0 * (2 ** attempt)
                time.sleep(backoff)
            else:
                logger.warning(f"S2 landmark search failed: {e}")
    return []


def _resolve_s2_to_openalex(
    s2_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolve S2 papers to OpenAlex work_ids via DOI lookup.
    Papers without DOIs or that can't be resolved are dropped.
    """
    papers_with_dois = [p for p in s2_papers if p.get("doi")]
    if not papers_with_dois:
        return []

    resolved = []
    # Batch lookup by DOIs
    dois = [p["doi"] for p in papers_with_dois]
    for i in range(0, len(dois), 50):
        batch = dois[i:i + 50]
        try:
            doi_filter = "|".join(batch)
            url = "https://api.openalex.org/works"
            params = {
                "filter": f"doi:{doi_filter}",
                "per-page": len(batch),
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(2)
                resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            if resp.status_code != 200:
                continue

            data = resp.json()
            for w in data.get("results", []):
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue
                wid = wid_full.rsplit("/", 1)[-1]

                primary_topic = w.get("primary_topic", {})
                pt_id = None
                if primary_topic and primary_topic.get("id"):
                    pt_url = primary_topic["id"]
                    if "/" in pt_url:
                        pt_id = pt_url.rsplit("/", 1)[-1]

                resolved.append({
                    "work_id": wid,
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract": decode_openalex_abstract(w.get("abstract_inverted_index")),
                    "category": None,
                    "primary_topic_id": pt_id,
                    "source": "s2_resolved",
                })
        except Exception as e:
            logger.warning(f"S2->OpenAlex resolution failed: {e}")

    logger.info(f"Resolved {len(resolved)}/{len(papers_with_dois)} S2 papers to OpenAlex")
    return resolved


def _insert_papers_to_db(conn: Connection, papers: List[Dict[str, Any]]) -> None:
    """Insert discovered papers into the works table."""
    for p in papers:
        wid = p.get("work_id")
        if not wid or not wid.startswith("W"):
            continue
        try:
            authors = p.get("authors", [])
            conn.execute(
                text("""
                    INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id, authors_json)
                    VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id, :authors_json)
                    ON CONFLICT (work_id) DO UPDATE SET
                        cited_by_count = GREATEST(works.cited_by_count, EXCLUDED.cited_by_count),
                        abstract = COALESCE(NULLIF(works.abstract, ''), EXCLUDED.abstract),
                        primary_topic_id = COALESCE(works.primary_topic_id, EXCLUDED.primary_topic_id),
                        authors_json = COALESCE(works.authors_json, EXCLUDED.authors_json)
                """),
                {
                    "work_id": wid,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "cited_by_count": p.get("cited_by_count") or 0,
                    "abstract": p.get("abstract"),
                    "primary_topic_id": p.get("primary_topic_id"),
                    "authors_json": json.dumps(authors) if authors else None,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to insert paper {wid}: {e}")

    try:
        conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_topic_landmarks(
    conn: Connection,
    topic_id: Optional[str],
    before_year: Optional[int],
    target_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find influential papers in the same topic, spread across time.

    Strategy:
    1. Query OpenAlex for top-cited papers in the topic (external source)
    2. Query S2 for highly-cited papers by keyword (supplementary source)
    3. Merge, deduplicate, insert into DB
    4. Select landmarks spread across time periods
    5. Verify topic correctness for high-cited papers

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    if not topic_id:
        logger.info("No topic_id provided, skipping landmark retrieval")
        return []

    if not before_year:
        before_year = 2025

    # Step 1: Fetch from external sources in parallel
    all_papers = []

    # Build a search query from the target title for S2
    s2_query = (target_title or "")[:100]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        # OpenAlex: topic-based search (most reliable)
        futures[executor.submit(
            _search_openalex_topic_landmarks, topic_id, before_year, 20
        )] = "openalex"

        # S2: keyword-based search (supplementary coverage)
        if s2_query:
            futures[executor.submit(
                _search_s2_landmarks, s2_query, before_year, 15
            )] = "s2"

        for future in as_completed(futures):
            source = futures[future]
            try:
                papers = future.result()
                all_papers.extend(papers)
            except Exception as e:
                logger.warning(f"Landmark fetch from {source} failed: {e}")

    # Step 2: Resolve S2 papers to OpenAlex work_ids
    s2_only = [p for p in all_papers if p.get("source") == "s2"]
    oa_papers = [p for p in all_papers if p.get("source") == "openalex"]

    resolved_s2 = _resolve_s2_to_openalex(s2_only) if s2_only else []

    # Step 3: Merge and deduplicate by work_id
    seen_wids = set()
    merged = []
    for p in oa_papers + resolved_s2:
        wid = p.get("work_id")
        if wid and wid not in seen_wids and wid.startswith("W"):
            seen_wids.add(wid)
            merged.append(p)

    # Step 4: Insert all discovered papers into DB
    if merged:
        _insert_papers_to_db(conn, merged)
        logger.info(f"Landmark retrieval: {len(merged)} unique papers from external sources")

    # Also include papers already in DB for this topic (they may have been
    # ingested from prior rank jobs/maps and not returned by the API search)
    db_papers = _get_all_topic_papers_from_db(conn, topic_id, before_year, limit=30)
    for p in db_papers:
        wid = p.get("work_id")
        if wid and wid not in seen_wids:
            seen_wids.add(wid)
            merged.append(p)

    if not merged:
        logger.info(f"No landmark candidates found for topic {topic_id}")
        return []

    # Step 5: Sort by citations and select spread across time
    merged.sort(key=lambda x: -(x.get("cited_by_count") or 0))

    # Calculate time range from the merged set
    years = [p["year"] for p in merged if p.get("year")]
    if not years:
        return merged[:MIN_LANDMARKS]

    earliest_year = min(years)
    latest_year = max(y for y in years if y < before_year) if any(y < before_year for y in years) else max(years)
    span = latest_year - earliest_year

    if span <= 0:
        return merged[:MIN_LANDMARKS]

    landmark_count = calc_landmark_count(span)
    logger.info(
        f"Topic {topic_id}: span={span} years ({earliest_year}-{latest_year}), "
        f"landmark_count={landmark_count}, candidates={len(merged)}"
    )

    # Select one top-cited paper per time period
    period_length = span / landmark_count
    landmarks = []
    for i in range(landmark_count):
        period_start = earliest_year + i * period_length
        period_end = earliest_year + (i + 1) * period_length

        # Find top-cited paper in this period
        best = None
        for p in merged:
            y = p.get("year")
            if y is not None and period_start <= y < period_end + 1 and y < before_year:
                if best is None or (p.get("cited_by_count") or 0) > (best.get("cited_by_count") or 0):
                    best = p
        if best:
            landmarks.append(best)

    # Step 6: Verify topic correctness for high-cited landmarks
    verified_landmarks = []
    for lm in landmarks:
        lm_work_id = lm.get("work_id")
        lm_db_topic = lm.get("primary_topic_id")

        if lm.get("cited_by_count", 0) > 1000:
            actual_topic = _verify_topic_from_openalex(lm_work_id)
            if actual_topic and actual_topic != lm_db_topic:
                logger.warning(
                    f"Stale topic for landmark {lm_work_id}: "
                    f"DB={lm_db_topic}, OpenAlex={actual_topic}. Updating and skipping."
                )
                _update_work_topic(conn, lm_work_id, actual_topic)
                continue
            elif actual_topic and actual_topic == topic_id:
                verified_landmarks.append(lm)
            elif not actual_topic:
                verified_landmarks.append(lm)
        else:
            verified_landmarks.append(lm)

    if len(verified_landmarks) < len(landmarks):
        logger.info(
            f"Topic verification: {len(landmarks)} landmarks -> {len(verified_landmarks)} "
            f"(removed {len(landmarks) - len(verified_landmarks)} with stale topics)"
        )

    logger.info(f"Found {len(verified_landmarks)} verified landmarks for topic {topic_id}")
    return verified_landmarks


def _get_all_topic_papers_from_db(
    conn: Connection,
    topic_id: str,
    before_year: int,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Get top papers in topic from DB (existing ingested papers)."""
    rows = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id, authors_json
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
            "source": "db",
            "authors": row["authors_json"] or [],
        }
        for row in rows
    ]
