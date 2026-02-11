"""
Shared paper caching for Feature 3.

This module provides a unified caching layer for citing papers, landmarks,
and references. Both node_timeline and grounding_supplement (novelty) use
this module to ensure cache sharing and lazy expansion.

Key principle: The cache grows to accommodate the maximum needs of any feature.
If novelty caches 3 papers and timeline needs 20, timeline expands the cache.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.paper_identity import decode_openalex_abstract

logger = logging.getLogger(__name__)

OPENALEX_TIMEOUT = 15

# Allowed column names for cache read/write (prevents SQL injection)
_ALLOWED_COLUMNS = frozenset({"citing_works_json", "landmark_works_json", "referenced_works_json"})


# ============================================================================
# Generic Cache Read/Write
# ============================================================================

def read_cache(
    conn: Connection,
    work_id: str,
    column: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Read cached papers from a JSON column in the works table.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        column: Column name (citing_works_json, landmark_works_json, referenced_works_json)

    Returns:
        List of paper dicts if cached, None if not cached
    """
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"Invalid cache column: {column}")

    try:
        query = f"SELECT {column} FROM works WHERE work_id = :work_id"
        row = conn.execute(text(query), {"work_id": work_id}).mappings().first()

        if not row:
            return None

        cached = row[column]
        if cached is None:
            return None

        if isinstance(cached, list):
            return cached

        # Handle case where it's stored as JSON string
        if isinstance(cached, str):
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                return None

        return None

    except Exception as e:
        logger.warning(f"Failed to read cache {column} for {work_id}: {e}")
        return None


def write_cache(
    conn: Connection,
    work_id: str,
    column: str,
    papers: List[Dict[str, Any]],
) -> None:
    """
    Write papers to a JSON column in the works table.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        column: Column name
        papers: List of paper dicts to cache
    """
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"Invalid cache column: {column}")

    try:
        query = f"UPDATE works SET {column} = :papers WHERE work_id = :work_id"
        conn.execute(
            text(query),
            {"work_id": work_id, "papers": json.dumps(papers)},
        )
        conn.commit()
        logger.info(f"Cached {len(papers)} papers in {column} for {work_id}")
    except Exception as e:
        logger.warning(f"Failed to write cache {column} for {work_id}: {e}")


# ============================================================================
# OpenAlex Fetch Functions
# ============================================================================

def _extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    """Extract abstract from OpenAlex work data."""
    return decode_openalex_abstract(work.get("abstract_inverted_index"))


def fetch_citing_papers_openalex(
    work_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch papers that cite a given work from OpenAlex.

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    Sorted by citation count (most cited first).
    """
    if not work_id:
        return []

    try:
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"cites:{work_id}",
            "sort": "cited_by_count:desc",
            "per-page": limit,
        }

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        works = data.get("results", [])

        result = []
        for w in works:
            wid_full = w.get("id")
            if not wid_full or "/" not in wid_full:
                continue

            wid = wid_full.rsplit("/", 1)[-1]

            result.append({
                "work_id": wid,
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count") or 0,
                "abstract": _extract_abstract(w),
            })

        logger.info(f"Fetched {len(result)} citing papers for {work_id} from OpenAlex")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch citing papers from OpenAlex for {work_id}: {e}")
        return []


def fetch_references_openalex(
    work_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch papers that a given work cites (its references) from OpenAlex.

    Returns list of dicts with: work_id, title, year, cited_by_count
    """
    if not work_id:
        return []

    try:
        # First get the work to find its referenced_works
        url = f"https://api.openalex.org/works/{work_id}"
        resp = requests.get(url, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        work_data = resp.json()
        referenced_works = work_data.get("referenced_works", [])

        if not referenced_works:
            return []

        # Extract work IDs
        ref_ids = []
        for ref_url in referenced_works[:limit]:
            if "/" in ref_url:
                ref_ids.append(ref_url.rsplit("/", 1)[-1])

        if not ref_ids:
            return []

        # Fetch details for referenced works
        # Use filter to get multiple works at once
        filter_str = "|".join(ref_ids[:50])  # OpenAlex limit
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"openalex_id:{filter_str}",
            "per-page": 50,
        }

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        works = data.get("results", [])

        result = []
        for w in works:
            wid_full = w.get("id")
            if not wid_full or "/" not in wid_full:
                continue

            wid = wid_full.rsplit("/", 1)[-1]

            result.append({
                "work_id": wid,
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count") or 0,
            })

        logger.info(f"Fetched {len(result)} references for {work_id} from OpenAlex")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch references from OpenAlex for {work_id}: {e}")
        return []


def fetch_highly_cited_papers_openalex(
    search_terms: str,
    before_year: int,
    limit: int = 10,
    primary_topic_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch highly-cited papers from OpenAlex by search terms.

    Used for supplementing landmarks when LLM suggestions can't be resolved.
    """
    if not search_terms or len(search_terms) < 5:
        return []

    try:
        url = "https://api.openalex.org/works"
        filters = [f"publication_year:<{before_year}", "cited_by_count:>300"]

        if primary_topic_id:
            filters.append(f"primary_topic.id:{primary_topic_id}")

        params = {
            "search": search_terms[:100],
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per-page": limit * 2,
        }

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        results = resp.json().get("results", [])

        papers = []
        for r in results:
            if len(papers) >= limit:
                break

            paper_title = r.get("title", "")
            if not paper_title:
                continue

            work_id = r.get("id", "")
            if work_id and "/" in work_id:
                work_id = work_id.rsplit("/", 1)[-1]

            papers.append({
                "work_id": work_id,
                "title": paper_title,
                "year": r.get("publication_year"),
                "cited_by_count": r.get("cited_by_count", 0),
                "abstract": _extract_abstract(r),
            })

        logger.info(f"OpenAlex highly-cited search found {len(papers)} papers")
        return papers

    except Exception as e:
        logger.warning(f"OpenAlex highly-cited search error: {e}")
        return []


# ============================================================================
# Main Cached Access Functions (with lazy expansion)
# ============================================================================

def get_citing_papers(
    conn: Connection,
    work_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get papers that cite a given work, with caching and lazy expansion.

    If cache has fewer than `limit` papers, fetches more and expands cache.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        limit: Number of citing papers needed

    Returns:
        List of citing paper dicts
    """
    cached = read_cache(conn, work_id, "citing_works_json")

    # Cache hit with enough papers
    if cached is not None and len(cached) >= limit:
        logger.info(f"Citing papers cache hit for {work_id}: {len(cached)} >= {limit}")
        return cached[:limit]

    # Need to fetch (cache miss or need more)
    logger.info(f"Citing papers cache {'has ' + str(len(cached)) if cached else 'miss'} for {work_id}, fetching {limit}")
    papers = fetch_citing_papers_openalex(work_id, limit)

    # Update cache if we got more than what's cached
    if papers and (cached is None or len(papers) > len(cached)):
        write_cache(conn, work_id, "citing_works_json", papers)

    return papers[:limit]


def get_references(
    conn: Connection,
    work_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get papers that a work cites (references), with caching and lazy expansion.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        limit: Number of references needed

    Returns:
        List of reference paper dicts
    """
    cached = read_cache(conn, work_id, "referenced_works_json")

    # Cache hit with enough papers
    if cached is not None and len(cached) >= limit:
        logger.info(f"References cache hit for {work_id}: {len(cached)} >= {limit}")
        return cached[:limit]

    # Need to fetch (cache miss or need more)
    logger.info(f"References cache {'has ' + str(len(cached)) if cached else 'miss'} for {work_id}, fetching {limit}")
    papers = fetch_references_openalex(work_id, limit)

    # Update cache if we got more than what's cached
    if papers and (cached is None or len(papers) > len(cached)):
        write_cache(conn, work_id, "referenced_works_json", papers)

    return papers[:limit]


def get_landmarks(
    conn: Connection,
    work_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get landmark papers for a work, with caching.

    Note: Landmarks are typically generated by LLM suggestion + OpenAlex lookup,
    not directly fetchable. This function reads the cache; use supplement_landmarks()
    to populate it.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        limit: Number of landmarks needed

    Returns:
        List of landmark paper dicts (may be empty if not yet generated)
    """
    cached = read_cache(conn, work_id, "landmark_works_json")

    if cached is not None:
        logger.info(f"Landmarks cache hit for {work_id}: {len(cached)} papers")
        return cached[:limit]

    logger.info(f"Landmarks cache miss for {work_id}")
    return []


def cache_landmarks(
    conn: Connection,
    work_id: str,
    landmarks: List[Dict[str, Any]],
    merge_with_existing: bool = True,
) -> None:
    """
    Cache landmark papers, optionally merging with existing cache.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        landmarks: List of landmark paper dicts
        merge_with_existing: If True, adds to existing cache; if False, replaces
    """
    if merge_with_existing:
        existing = read_cache(conn, work_id, "landmark_works_json") or []

        # Merge by work_id, avoiding duplicates
        existing_ids = {p.get("work_id") for p in existing}
        merged = existing.copy()

        for lm in landmarks:
            if lm.get("work_id") not in existing_ids:
                merged.append(lm)
                existing_ids.add(lm.get("work_id"))

        write_cache(conn, work_id, "landmark_works_json", merged)
    else:
        write_cache(conn, work_id, "landmark_works_json", landmarks)


def cache_references(
    conn: Connection,
    work_id: str,
    references: List[Dict[str, Any]],
    merge_with_existing: bool = True,
) -> None:
    """
    Cache reference papers, optionally merging with existing cache.

    Args:
        conn: Database connection
        work_id: OpenAlex work ID
        references: List of reference paper dicts
        merge_with_existing: If True, adds to existing cache; if False, replaces
    """
    if merge_with_existing:
        existing = read_cache(conn, work_id, "referenced_works_json") or []

        # Merge by work_id, avoiding duplicates
        existing_ids = {p.get("work_id") for p in existing}
        merged = existing.copy()

        for ref in references:
            if ref.get("work_id") not in existing_ids:
                merged.append(ref)
                existing_ids.add(ref.get("work_id"))

        write_cache(conn, work_id, "referenced_works_json", merged)
    else:
        write_cache(conn, work_id, "referenced_works_json", references)
