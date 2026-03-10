"""
Reference store for Feature 3.

This module fetches and caches the papers that a work cites (its references)
from OpenAlex and Semantic Scholar, storing them in the works.referenced_works_json column.

Key design: We fetch ALL referenced work_ids from both sources, insert them
all into the DB, then sort by citation count and return the top N. This ensures
we get the most influential references, not arbitrary ones.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.abstract_enrichment import S2_RATE_LIMITER
from app.feature3.paper_identity import decode_openalex_abstract
from app.shared.s2_keys import get_s2_headers

logger = logging.getLogger(__name__)

from app.shared.oa_keys import get_oa_api_key
OPENALEX_BATCH_SIZE = 50
OPENALEX_TIMEOUT = 15
S2_TIMEOUT = 15
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

    Routes to the correct source based on work_id prefix:
    - W* (OpenAlex) → fetch from OpenAlex API, cross-reference with S2
    - S2: (Semantic Scholar) → fetch from S2 by paperId directly
    - AX: (ArXiv) → fetch from S2 by ArXiv ID

    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    # Step 1: Check cache
    cached = _get_cached_references(conn, work_id)
    if cached is not None:
        logger.info(f"Reference cache hit for {work_id}: {len(cached)} refs")
    else:
        # Route to correct source based on work_id prefix
        if work_id.startswith("S2:"):
            # S2 paper: fetch references directly by paperId
            s2_paper_id = work_id[3:]
            s2_refs = _fetch_references_from_s2_by_id(s2_paper_id)
            if s2_refs:
                cached = [r["work_id"] for r in s2_refs]
                _insert_s2_only_papers(conn, s2_refs)
                logger.info(f"References for {work_id}: {len(cached)} from S2 (by paperId)")
            else:
                cached = []

        elif work_id.startswith("AX:"):
            # ArXiv paper: fetch references from S2 by ArXiv ID
            arxiv_id = work_id[3:]
            s2_refs = _fetch_references_from_s2_by_id(f"ArXiv:{arxiv_id}")
            if s2_refs:
                cached = [r["work_id"] for r in s2_refs]
                _insert_s2_only_papers(conn, s2_refs)
                logger.info(f"References for {work_id}: {len(cached)} from S2 (by ArXiv ID)")
            else:
                cached = []

        else:
            # OpenAlex paper (W* prefix): existing path
            # Step 2a: Fetch from OpenAlex (also get DOI for S2 cross-referencing)
            oa_ref_ids, target_doi = _fetch_references_from_openalex(work_id)

            # Step 2b: If no DOI from OpenAlex response, try DB
            if not target_doi:
                row = conn.execute(
                    text("SELECT doi FROM works WHERE work_id = :wid"),
                    {"wid": work_id},
                ).mappings().first()
                if row and row["doi"]:
                    raw_doi = row["doi"]
                    target_doi = raw_doi.replace("https://doi.org/", "") if raw_doi.startswith("https://") else raw_doi

            # Step 2c: Cross-reference with S2
            s2_refs = _fetch_references_from_s2(target_doi) if target_doi else []

            # Step 2d: Merge reference lists
            if s2_refs:
                cached = _merge_s2_references(oa_ref_ids, s2_refs)
                _insert_s2_only_papers(conn, s2_refs)
                logger.info(
                    f"References for {work_id}: {len(oa_ref_ids)} from OpenAlex, "
                    f"{len(s2_refs)} from S2, {len(cached)} merged"
                )
            else:
                cached = oa_ref_ids

        if not cached:
            logger.info(f"No references found for {work_id}")
            _cache_references(conn, work_id, [])
            return []

        # Step 3: Cache merged reference IDs
        _cache_references(conn, work_id, cached)

    # Step 4: Ensure W* referenced works exist in DB (S* already inserted)
    w_ids = [wid for wid in cached if wid.startswith("W")]
    _ensure_works_exist(conn, w_ids)

    # Step 5: Load ALL references (W* and S*), sort by citations, return top N
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


def _fetch_references_from_openalex(work_id: str) -> Tuple[List[str], Optional[str]]:
    """Fetch referenced_works and DOI from OpenAlex API with retry logic.

    Returns:
        (ref_ids, doi) - ref_ids is list of W* work IDs, doi for S2 cross-referencing
    """
    for attempt in range(MAX_RETRIES):
        try:
            url = f"https://api.openalex.org/works/{work_id}"
            params = {}
            oa_key = get_oa_api_key()
            if oa_key:
                params["api_key"] = oa_key
            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)

            if resp.status_code == 429:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"OpenAlex 429 for refs, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            data = resp.json()

            # Extract DOI for S2 cross-referencing
            doi_raw = data.get("doi")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            referenced_works = data.get("referenced_works", [])

            ref_ids = []
            for ref_url in referenced_works:
                if isinstance(ref_url, str) and "/" in ref_url:
                    ref_id = ref_url.rsplit("/", 1)[-1]
                    ref_ids.append(ref_id)

            logger.info(f"Fetched {len(ref_ids)} references for {work_id} from OpenAlex (DOI: {doi})")
            return ref_ids, doi

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch references from OpenAlex for {work_id}: {e}")
    return [], None


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
            oa_key = get_oa_api_key()
            if oa_key:
                params["api_key"] = oa_key

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

                authors = [
                    au.get("author", {}).get("display_name") or au.get("display_name")
                    for au in w.get("authorships", [])
                    if au.get("author", {}).get("display_name") or au.get("display_name")
                ]

                primary_topic = w.get("primary_topic", {})
                topic_id = None
                if primary_topic and primary_topic.get("id"):
                    topic_url = primary_topic["id"]
                    if "/" in topic_url:
                        topic_id = topic_url.rsplit("/", 1)[-1]

                conn.execute(
                    text("""
                        INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id, authors_json)
                        VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id, :authors_json)
                        ON CONFLICT (work_id) DO UPDATE SET
                            authors_json = COALESCE(works.authors_json, EXCLUDED.authors_json)
                    """),
                    {
                        "work_id": wid,
                        "title": title,
                        "year": year,
                        "cited_by_count": cited_by_count,
                        "abstract": abstract,
                        "primary_topic_id": topic_id,
                        "authors_json": json.dumps(authors) if authors else None,
                    },
                )

            conn.commit()
            return  # Success

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch/insert works: {e}")


def _fetch_references_from_s2_by_id(paper_id: str) -> List[Dict[str, Any]]:
    """
    Fetch references from Semantic Scholar by paper ID (S2 hash or ArXiv:id).

    Used for S2: and AX: work_ids where we can't go through OpenAlex.
    Returns list of dicts with: work_id, title, year, cited_by_count, doi, authors.
    """
    if not paper_id:
        return []

    for attempt in range(MAX_RETRIES):
        try:
            S2_RATE_LIMITER.wait()
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
            params = {
                "fields": "paperId,title,year,citationCount,externalIds,authors",
                "limit": 1000,
            }
            resp = requests.get(url, params=params, headers=get_s2_headers(), timeout=S2_TIMEOUT)

            if resp.status_code == 404:
                logger.info(f"S2 references not found for paper {paper_id}")
                return []
            if resp.status_code == 429:
                backoff = 2.0 * (2 ** attempt)
                logger.info(f"S2 429 for references, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            data = resp.json()

            # S2 returns null data when publisher elides references
            if data.get("data") is None:
                logger.info(f"S2 references elided by publisher for {paper_id}")
                return []

            results = []
            for item in data["data"]:
                ref = item.get("citedPaper")
                if not ref or not isinstance(ref, dict) or not ref.get("title"):
                    continue

                ext_ids = ref.get("externalIds") or {}
                oa_id = ext_ids.get("OpenAlex")
                ref_doi = ext_ids.get("DOI")
                s2_id = ref.get("paperId")

                if oa_id and oa_id.startswith("W"):
                    ref_work_id = oa_id
                elif s2_id:
                    ref_work_id = f"S{s2_id}"
                else:
                    continue

                s2_authors = [a.get("name") for a in (ref.get("authors") or []) if a.get("name")]

                results.append({
                    "work_id": ref_work_id,
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                    "cited_by_count": ref.get("citationCount") or 0,
                    "doi": ref_doi,
                    "s2_paper_id": s2_id,
                    "authors": s2_authors,
                })

            logger.info(f"S2 references by ID: {len(results)} papers for {paper_id}")
            return results

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch S2 references for {paper_id}: {e}")
    return []


def _fetch_references_from_s2(doi: str) -> List[Dict[str, Any]]:
    """
    Fetch references from Semantic Scholar by DOI.

    Returns list of dicts with: work_id (W* if OpenAlex ID available, else S*),
    title, year, cited_by_count, doi, authors.
    """
    if not doi:
        return []

    for attempt in range(MAX_RETRIES):
        try:
            S2_RATE_LIMITER.wait()
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/references"
            params = {
                "fields": "paperId,title,year,citationCount,externalIds,authors",
                "limit": 1000,
            }
            resp = requests.get(url, params=params, headers=get_s2_headers(), timeout=S2_TIMEOUT)

            if resp.status_code == 404:
                logger.debug(f"S2 references not found for DOI:{doi}")
                return []
            if resp.status_code == 429:
                backoff = 2.0 * (2 ** attempt)
                logger.info(f"S2 429 for references, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("data") or []:
                ref = item.get("citedPaper")
                if not ref or not isinstance(ref, dict) or not ref.get("title"):
                    continue

                ext_ids = ref.get("externalIds") or {}
                oa_id = ext_ids.get("OpenAlex")
                ref_doi = ext_ids.get("DOI")
                s2_id = ref.get("paperId")

                # Use OpenAlex W* ID if available, else S* ID
                if oa_id and oa_id.startswith("W"):
                    work_id = oa_id
                elif s2_id:
                    work_id = f"S{s2_id}"
                else:
                    continue

                s2_authors = [a.get("name") for a in (ref.get("authors") or []) if a.get("name")]

                results.append({
                    "work_id": work_id,
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                    "cited_by_count": ref.get("citationCount") or 0,
                    "doi": ref_doi,
                    "s2_paper_id": s2_id,
                    "authors": s2_authors,
                })

            logger.info(f"S2 references: {len(results)} papers for DOI:{doi}")
            return results

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"Failed to fetch S2 references for DOI:{doi}: {e}")
    return []


def _merge_s2_references(
    oa_ref_ids: List[str],
    s2_refs: List[Dict[str, Any]],
) -> List[str]:
    """
    Merge S2 references into OpenAlex reference ID list.

    S2 papers with OpenAlex externalId -> W* ID (already in list or added).
    S2 papers without OpenAlex ID -> S* ID (new coverage).
    Returns merged list of work_id strings (W* and S*).
    """
    existing = set(oa_ref_ids)
    merged = list(oa_ref_ids)

    s2_only_count = 0
    for ref in s2_refs:
        wid = ref.get("work_id")
        if wid and wid not in existing:
            merged.append(wid)
            existing.add(wid)
            if wid.startswith("S"):
                s2_only_count += 1

    if s2_only_count > 0:
        logger.info(f"S2 added {s2_only_count} references (S* IDs) not in OpenAlex")

    return merged


def _insert_s2_only_papers(
    conn: Connection,
    s2_refs: List[Dict[str, Any]],
) -> None:
    """Insert S2-only papers (S* IDs) into the works table."""
    s2_only = [r for r in s2_refs if r.get("work_id", "").startswith("S")]
    if not s2_only:
        return

    for ref in s2_only:
        wid = ref["work_id"]
        authors = ref.get("authors", [])
        try:
            conn.execute(
                text("""
                    INSERT INTO works (work_id, title, year, cited_by_count, doi, authors_json)
                    VALUES (:work_id, :title, :year, :cited_by_count, :doi, :authors_json)
                    ON CONFLICT (work_id) DO NOTHING
                """),
                {
                    "work_id": wid,
                    "title": ref.get("title"),
                    "year": ref.get("year"),
                    "cited_by_count": ref.get("cited_by_count") or 0,
                    "doi": ref.get("doi"),
                    "authors_json": json.dumps(authors) if authors else None,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to insert S2 paper {wid}: {e}")

    try:
        conn.commit()
    except Exception:
        pass

    logger.info(f"Inserted {len(s2_only)} S2-only reference papers")


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
            SELECT work_id, title, year, cited_by_count, abstract, category, primary_topic_id, authors_json
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
            "authors": row["authors_json"] or [],
        }
        for row in rows
    ]

    # Return top N by citation count (already sorted by SQL)
    return results[:limit]
