"""
Reference store for Feature 3.

This module fetches and caches the papers that a work cites (its references)
from OpenAlex, storing them in the works.referenced_works_json column.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

OPENALEX_BATCH_SIZE = 50
MAX_REFERENCES_TO_FETCH = 7


def get_referenced_works(
    conn: Connection,
    work_id: str,
) -> List[Dict[str, Any]]:
    """
    Get the papers that this work cites (its references).

    1. Check if works.referenced_works_json is already populated
    2. If not, fetch from OpenAlex API
    3. Store in referenced_works_json column
    4. Ensure referenced works exist in works table
    5. Return top references with title/abstract

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    # Step 1: Check cache
    cached = _get_cached_references(conn, work_id)
    if cached is not None:
        logger.info(f"Reference cache hit for {work_id}: {len(cached)} refs")
        # Ensure referenced works exist in DB (may be missing if cache was populated
        # but works weren't inserted, or if works table was modified)
        top_refs = cached[:MAX_REFERENCES_TO_FETCH]
        _ensure_works_exist(conn, top_refs)
        return _load_reference_details(conn, top_refs)

    # Step 2: Fetch from OpenAlex
    ref_ids = _fetch_references_from_openalex(work_id)
    if not ref_ids:
        logger.info(f"No references found for {work_id}")
        # Cache empty result
        _cache_references(conn, work_id, [])
        return []

    # Step 3: Cache the reference IDs
    _cache_references(conn, work_id, ref_ids)

    # Step 4: Ensure referenced works exist in our DB
    _ensure_works_exist(conn, ref_ids[:MAX_REFERENCES_TO_FETCH])

    # Step 5: Load and return details
    return _load_reference_details(conn, ref_ids[:MAX_REFERENCES_TO_FETCH])


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

    # refs is a list of work_ids
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
    """Fetch referenced_works from OpenAlex API."""
    try:
        # OpenAlex work endpoint
        url = f"https://api.openalex.org/works/{work_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        referenced_works = data.get("referenced_works", [])

        # Extract work IDs from full URLs
        ref_ids = []
        for ref_url in referenced_works:
            if isinstance(ref_url, str) and "/" in ref_url:
                ref_id = ref_url.rsplit("/", 1)[-1]
                ref_ids.append(ref_id)

        logger.info(f"Fetched {len(ref_ids)} references for {work_id} from OpenAlex")
        return ref_ids

    except Exception as e:
        logger.warning(f"Failed to fetch references from OpenAlex for {work_id}: {e}")
        return []


def _ensure_works_exist(conn: Connection, work_ids: List[str]) -> None:
    """
    Ensure the referenced works exist in our works table.
    Fetch missing ones from OpenAlex.
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
    """Fetch works from OpenAlex and insert into DB."""
    if not work_ids:
        return

    try:
        ids_param = "|".join(f"https://openalex.org/{wid}" for wid in work_ids)
        url = "https://api.openalex.org/works"
        params = {"filter": f"openalex:{ids_param}", "per-page": len(work_ids)}

        resp = requests.get(url, params=params, timeout=15)
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

            # Decode abstract from inverted index
            abstract = _decode_abstract(w.get("abstract_inverted_index"))

            # Extract primary_topic_id for cross-domain filtering
            primary_topic = w.get("primary_topic", {})
            topic_id = None
            if primary_topic and primary_topic.get("id"):
                topic_url = primary_topic["id"]
                if "/" in topic_url:
                    topic_id = topic_url.rsplit("/", 1)[-1]

            # Insert (ignore conflicts)
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

    except Exception as e:
        logger.warning(f"Failed to fetch/insert works: {e}")


def _decode_abstract(inverted_index: Any) -> Optional[str]:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None

    position_map: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                position_map[int(pos)] = word
            except (ValueError, TypeError):
                continue

    if not position_map:
        return None

    ordered = [position_map[i] for i in sorted(position_map.keys())]
    return " ".join(ordered).strip() or None


def _load_reference_details(
    conn: Connection,
    ref_ids: List[str],
) -> List[Dict[str, Any]]:
    """Load title, year, cited_by_count, abstract for referenced works."""
    if not ref_ids:
        return []

    rows = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id
            FROM works
            WHERE work_id = ANY(:ids)
        """),
        {"ids": ref_ids},
    ).mappings().all()

    # Preserve order and sort by citation count
    results = []
    for row in rows:
        results.append({
            "work_id": row["work_id"],
            "title": row["title"],
            "year": row["year"],
            "cited_by_count": int(row["cited_by_count"] or 0),
            "abstract": row["abstract"],
            "category": row["category"],
            "primary_topic_id": row["primary_topic_id"],
        })

    # Sort by citation count descending (most influential first)
    results.sort(key=lambda x: -x["cited_by_count"])

    return results
