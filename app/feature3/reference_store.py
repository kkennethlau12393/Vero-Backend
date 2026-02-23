"""
Reference store for Feature 3.

This module fetches and caches the papers that a work cites (its references)
from OpenAlex, storing them in the works.referenced_works_json column.

Key design: We fetch ALL referenced work_ids, insert them all into the DB,
then sort by citation count and return the top N. This ensures we get the
most influential references, not arbitrary ones.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.paper_identity import decode_openalex_abstract

logger = logging.getLogger(__name__)

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
OPENALEX_BATCH_SIZE = 50
OPENALEX_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# How many top-cited references to return for grounding
MAX_REFERENCES_TO_RETURN = 10


def get_referenced_works(
    conn: Connection,
    work_id: str,
) -> List[Dict[str, Any]]:
    """
    Get the papers that this work cites (its references).

    1. Check if works.referenced_works_json is already populated
    2. If not, fetch from OpenAlex API
    3. Store in referenced_works_json column
    4. Ensure ALL referenced works exist in works table
    5. Sort by citation count and return top N

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    # Step 1: Check cache
    cached = _get_cached_references(conn, work_id)
    if cached is not None:
        logger.info(f"Reference cache hit for {work_id}: {len(cached)} refs")
    else:
        # Step 2: Fetch from OpenAlex
        cached = _fetch_references_from_openalex(work_id)
        if not cached:
            logger.info(f"No references found for {work_id}")
            _cache_references(conn, work_id, [])
            return []
        # Step 3: Cache all reference IDs
        _cache_references(conn, work_id, cached)

    # Step 4: Ensure ALL referenced works exist in our DB
    # This is the key fix: fetch all, not just first N
    _ensure_works_exist(conn, cached)

    # Step 5: Load ALL references, sort by citations, return top N
    return _load_reference_details(conn, cached, limit=MAX_REFERENCES_TO_RETURN)


def _get_cached_references(conn: Connection, work_id: str) -> Optional[List[str]]:
    """Check if referenced_works_json is populated for this work."""
    row = conn.execute(
        text("""
            SELECT referenced_works_json
            FROM works
            WHERE work_id = :work_id
        """),
        {"work_id": work_id},
    ).mappings().first()

    if not row:
        return None

    refs = row["referenced_works_json"]
    if refs is None:
        return None

    if isinstance(refs, list):
        return refs
    return None


def _cache_references(conn: Connection, work_id: str, ref_ids: List[str]) -> None:
    """Store referenced work IDs in the works table."""
    try:
        conn.execute(
            text("""
                UPDATE works
                SET referenced_works_json = :refs
                WHERE work_id = :work_id
            """),
            {"work_id": work_id, "refs": json.dumps(ref_ids)},
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache references for {work_id}: {e}")


def _fetch_references_from_openalex(work_id: str) -> List[str]:
    """Fetch referenced_works from OpenAlex API with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            url = f"https://api.openalex.org/works/{work_id}"
            params = {}
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY
            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)

            if resp.status_code == 429:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"OpenAlex 429 for refs, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            data = resp.json()
            referenced_works = data.get("referenced_works", [])

            ref_ids = []
            for ref_url in referenced_works:
                if isinstance(ref_url, str) and "/" in ref_url:
                    ref_id = ref_url.rsplit("/", 1)[-1]
                    ref_ids.append(ref_id)

            logger.info(f"Fetched {len(ref_ids)} references for {work_id} from OpenAlex")
            return ref_ids

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch references from OpenAlex for {work_id}: {e}")
    return []


def _ensure_works_exist(conn: Connection, work_ids: List[str]) -> None:
    """
    Ensure the referenced works exist in our works table.
    Fetch missing ones from OpenAlex in batches.
    """
    if not work_ids:
        return

    # Check which works are missing
    existing = conn.execute(
        text("SELECT work_id FROM works WHERE work_id = ANY(:ids)"),
        {"ids": work_ids},
    ).scalars().all()
    existing_set = set(existing)

    missing = [wid for wid in work_ids if wid not in existing_set]
    if not missing:
        return

    logger.info(f"Fetching {len(missing)} missing referenced works from OpenAlex")

    # Fetch in batches
    for i in range(0, len(missing), OPENALEX_BATCH_SIZE):
        batch = missing[i : i + OPENALEX_BATCH_SIZE]
        _fetch_and_insert_works(conn, batch)


def _fetch_and_insert_works(conn: Connection, work_ids: List[str]) -> None:
    """Fetch works from OpenAlex and insert into DB with retry logic."""
    if not work_ids:
        return

    for attempt in range(MAX_RETRIES):
        try:
            ids_param = "|".join(f"https://openalex.org/{wid}" for wid in work_ids)
            url = "https://api.openalex.org/works"
            params = {"filter": f"openalex:{ids_param}", "per-page": len(work_ids)}
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)

            if resp.status_code == 429:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"OpenAlex 429 for batch insert, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            data = resp.json()
            works = data.get("results", [])

            for w in works:
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue

                wid = wid_full.rsplit("/", 1)[-1]
                title = w.get("title")
                year = w.get("publication_year")
                cited_by_count = w.get("cited_by_count") or 0

                abstract = decode_openalex_abstract(w.get("abstract_inverted_index"))

                primary_topic = w.get("primary_topic", {})
                topic_id = None
                if primary_topic and primary_topic.get("id"):
                    topic_url = primary_topic["id"]
                    if "/" in topic_url:
                        topic_id = topic_url.rsplit("/", 1)[-1]

                conn.execute(
                    text("""
                        INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id)
                        VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id)
                        ON CONFLICT (work_id) DO NOTHING
                    """),
                    {
                        "work_id": wid,
                        "title": title,
                        "year": year,
                        "cited_by_count": cited_by_count,
                        "abstract": abstract,
                        "primary_topic_id": topic_id,
                    },
                )

            conn.commit()
            return  # Success

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch/insert works: {e}")


def _load_reference_details(
    conn: Connection,
    ref_ids: List[str],
    limit: int = MAX_REFERENCES_TO_RETURN,
) -> List[Dict[str, Any]]:
    """Load ALL references from DB, sort by citation count, return top N."""
    if not ref_ids:
        return []

    rows = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id
            FROM works
            WHERE work_id = ANY(:ids)
            ORDER BY cited_by_count DESC NULLS LAST
        """),
        {"ids": ref_ids},
    ).mappings().all()

    results = [
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

    # Return top N by citation count (already sorted by SQL)
    return results[:limit]
