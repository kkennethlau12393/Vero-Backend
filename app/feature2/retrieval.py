"""
Feature 2 retrieval utilities for production‑grade direct ranking.

This module implements Stage‑1 candidate generation for direct ranking using
multiple CO-EQUAL sources. No source is primary or secondary - each has unique
coverage and all supplement each other.

Retrieval Sources (all co-equal, run in parallel):
1. OpenAlex: ~200 papers/query × 10-15 expanded queries = ~500-1000 unique
   - Lexical search with LLM-powered query expansion
   - Good coverage of published academic papers

2. OpenAlex Recent (2018+): 250 papers from original query
   - Ensures recent developments are not drowned out by classics

3. Semantic Scholar: 100+ papers from original query
   - Different coverage than OpenAlex (some papers only in S2)
   - Good citation data and paper recommendations

4. ArXiv: 100+ papers from original query
   - Preprints and ML papers with delayed indexing elsewhere
   - Often has papers before they appear in other sources

5. CrossRef/PubMed/DBLP: Additional coverage for specific domains

Each source uses its own ID system (W... for OpenAlex, S2:... for Semantic Scholar,
AX:... for ArXiv). Title-based deduplication unifies papers across sources.

Total pool after dedup: typically 800-1800 papers → capped at DEFAULT_LIMIT_POOL (3000)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Connection

# Ensure .env is loaded (in case this module is imported before main.py's load_dotenv)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5  # seconds

# Current year for validation (papers shouldn't have future years)
import datetime
CURRENT_YEAR = datetime.datetime.now().year

# Semantic Scholar configuration
# Co-equal source with different coverage than OpenAlex
SEMANTIC_SCHOLAR_ENABLED = True
SEMANTIC_SCHOLAR_LIMIT = 2000  # Papers from Semantic Scholar (bulk endpoint, 1000 per page)
SEMANTIC_SCHOLAR_DELAY = 1.0  # Rate limit: 1 request/second
S2_HIGHLY_CITED_LIMIT = 100  # Highly-cited papers from S2 (sorted by citations, not recency)

# ArXiv configuration
# Co-equal source for preprints and ML papers
ARXIV_ENABLED = True
ARXIV_LIMIT = 500  # Papers from ArXiv

# CrossRef configuration (general scholarly works via DOI)
CROSSREF_ENABLED = True
CROSSREF_LIMIT = 100  # Papers from CrossRef

# PubMed configuration (biomedical literature)
PUBMED_ENABLED = True
PUBMED_LIMIT = 100  # Papers from PubMed

# DBLP configuration (computer science)
DBLP_ENABLED = False 
DBLP_LIMIT = 100  # Papers from DBLP

# OpenCitations configuration (free, open citation index)
OPENCITATIONS_ENABLED = True  # Use OpenCitations for citation counts (fast, free)

# OpenAlex API key (improves rate limits significantly)
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")

# OpenAlex recent papers configuration
# Note: OpenAlex limits per-page to 200 when using filters
OPENALEX_RECENT_LIMIT = 200  # Recent papers (2018+) per query

# OpenAlex highly-cited papers retrieval configuration
# This retrieves the most-cited papers for a query to ensure foundational works are included
# Increased from 50 to 100 to better capture seminal works in broad topic areas
OPENALEX_HIGHLY_CITED_LIMIT = 100  # Top cited papers to retrieve per query

from .work_topic_store import WorkStore
from .query_expansion import expand_query, QueryExpansion, ExpandedConcept, normalize_query_for_expansion
# Constants defining sensible defaults for candidate generation
DEFAULT_K_LEXICAL = 200
DEFAULT_K_TOPIC = 200
DEFAULT_LIMIT_POOL = 3000
PIPELINE_VERSION = "openalex_v2"


# ============================================================================
# Deduplication Utility Functions
# ============================================================================

def normalize_doi_for_dedup(doi: str | None) -> str | None:
    """Normalize DOI for deduplication (lowercase, strip URL prefixes).

    Examples:
        "10.1234/example" → "10.1234/example"
        "https://doi.org/10.1234/example" → "10.1234/example"
        "DOI:10.1234/example" → "10.1234/example"
    """
    if not doi:
        return None
    from app.feature3.paper_identity import normalize_doi
    return normalize_doi(doi)


def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    """Normalize ArXiv ID for deduplication.

    Strips prefixes like "arXiv:" and version suffixes like "v1", "v2".

    Examples:
        "1706.03762" → "1706.03762"
        "arXiv:1706.03762" → "1706.03762"
        "1706.03762v3" → "1706.03762"
        "arxiv:2301.07041v2" → "2301.07041"
    """
    if not arxiv_id:
        return None
    # Strip prefixes like "arXiv:" or "arxiv:"
    clean = arxiv_id.lower().replace("arxiv:", "").strip()
    # Strip version suffixes (e.g., "1706.03762v3" → "1706.03762")
    if "v" in clean:
        parts = clean.split("v")
        if len(parts) == 2 and parts[1].isdigit():
            clean = parts[0]
    return clean


def build_external_id_maps(
    conn: Connection,
    dois: list[str],
    arxiv_ids: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Pre-fetch existing papers by DOI and ArXiv ID from the works table.

    This enables cross-source deduplication by identifying papers that were
    previously ingested from different sources but represent the same work.

    Args:
        conn: Database connection
        dois: List of DOIs to look up (normalized format)
        arxiv_ids: List of ArXiv IDs to look up (normalized format)

    Returns:
        Tuple of (doi_to_work_id, arxiv_id_to_work_id) dictionaries
    """
    doi_to_work_id = {}
    arxiv_id_to_work_id = {}

    if dois:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (doi) doi, work_id
            FROM works
            WHERE doi = ANY(:dois)
            ORDER BY doi, work_id
        """), {"dois": list(set(dois))}).mappings().all()
        doi_to_work_id = {row["doi"]: row["work_id"] for row in rows}

    if arxiv_ids:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (arxiv_id) arxiv_id, work_id
            FROM works
            WHERE arxiv_id = ANY(:arxiv_ids)
            ORDER BY arxiv_id, work_id
        """), {"arxiv_ids": list(set(arxiv_ids))}).mappings().all()
        arxiv_id_to_work_id = {row["arxiv_id"]: row["work_id"] for row in rows}

    return doi_to_work_id, arxiv_id_to_work_id


def _search_semantic_scholar(query: str, k: int = SEMANTIC_SCHOLAR_LIMIT) -> List[Tuple[str, Dict[str, Any]]]:
    """Search Semantic Scholar for papers.

    Tries the bulk endpoint first (supports up to 10M papers via pagination).
    Falls back to regular search endpoint if bulk fails (e.g., query too broad).

    Returns list of (semantic_scholar_id, metadata_dict) tuples.
    The metadata includes title, year, citationCount for later matching.
    """
    if not query or not SEMANTIC_SCHOLAR_ENABLED:
        return []

    import os

    # Use API key if available (reduces rate limiting)
    headers = {}
    s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    # Try bulk endpoint first
    result = _search_s2_bulk(query, k, headers)
    if result is not None:
        return result

    # Fall back to regular paginated search (limited to 1000 total)
    logger.info(f"S2 bulk failed for '{query[:30]}', falling back to regular search")
    return _search_s2_regular(query, min(k, 1000), headers)


def _search_s2_bulk(query: str, k: int, headers: dict) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Search using S2 bulk endpoint with continuation tokens."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids = set()
    continuation_token = None
    page_size = min(1000, k)

    while len(out) < k:
        params = {
            "query": query,
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": page_size,
        }
        if continuation_token:
            params["token"] = continuation_token

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)

                # 400 error = query too broad, fall back to regular search
                if resp.status_code == 400:
                    return None

                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                continuation_token = data.get("token")

                if not results:
                    return out

                for paper in results:
                    paper_id = paper.get("paperId")
                    if not paper_id or paper_id in seen_ids:
                        continue

                    seen_ids.add(paper_id)
                    external_ids = paper.get("externalIds") or {}

                    out.append((paper_id, {
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                        "openalex_id": external_ids.get("OpenAlex"),
                        "arxiv_id": external_ids.get("ArXiv"),
                        "doi": external_ids.get("DOI"),
                    }))

                    if len(out) >= k:
                        return out

                if not continuation_token:
                    return out

                time.sleep(SEMANTIC_SCHOLAR_DELAY)
                break

            except requests.exceptions.RequestException as e:
                is_rate_limit = "429" in str(e)
                if attempt < MAX_RETRIES - 1:
                    backoff = (2 ** (attempt + 1)) if is_rate_limit else RETRY_BACKOFF_BASE * (2 ** attempt)
                    if is_rate_limit:
                        logger.info(f"S2 bulk rate limited, waiting {backoff}s")
                    time.sleep(backoff)
                    continue
                logger.warning(f"S2 bulk search failed: {e}")
                return out
            except Exception as e:
                logger.warning(f"S2 bulk search error: {e}")
                return out
        else:
            break

    return out


def _search_s2_regular(query: str, k: int, headers: dict) -> List[Tuple[str, Dict[str, Any]]]:
    """Search using regular S2 endpoint with offset pagination (max 1000 total)."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids = set()
    offset = 0
    page_size = 100  # Regular endpoint max is 100 per page

    while len(out) < k and offset < 1000:
        params = {
            "query": query,
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": page_size,
            "offset": offset,
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                total_available = data.get("total", 0)

                if not results:
                    return out

                for paper in results:
                    paper_id = paper.get("paperId")
                    if not paper_id or paper_id in seen_ids:
                        continue

                    seen_ids.add(paper_id)
                    external_ids = paper.get("externalIds") or {}

                    out.append((paper_id, {
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "citations": paper.get("citationCount", 0),
                        "openalex_id": external_ids.get("OpenAlex"),
                        "arxiv_id": external_ids.get("ArXiv"),
                        "doi": external_ids.get("DOI"),
                    }))

                    if len(out) >= k:
                        return out

                offset += page_size
                if offset >= min(total_available, 1000):
                    return out

                time.sleep(SEMANTIC_SCHOLAR_DELAY)
                break

            except requests.exceptions.RequestException as e:
                is_rate_limit = "429" in str(e)
                if attempt < MAX_RETRIES - 1:
                    backoff = (2 ** (attempt + 1)) if is_rate_limit else RETRY_BACKOFF_BASE * (2 ** attempt)
                    if is_rate_limit:
                        logger.info(f"S2 regular rate limited, waiting {backoff}s")
                    time.sleep(backoff)
                    continue
                logger.warning(f"S2 regular search failed: {e}")
                return out
            except Exception as e:
                logger.warning(f"S2 regular search error: {e}")
                return out
        else:
            break

    return out


def _search_s2_highly_cited(query: str, k: int = 50) -> List[Tuple[str, Dict[str, Any]]]:
    """Search S2 for highly-cited papers, sorted by citation count.

    This complements the regular S2 search by specifically targeting
    foundational papers that may be older but highly influential.

    Returns list of (semantic_scholar_id, metadata_dict) tuples.
    """
    if not query or not SEMANTIC_SCHOLAR_ENABLED:
        return []

    import os

    headers = {}
    s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    out: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids = set()

    # Search with citation sort - S2 allows sort=citationCount:desc
    params = {
        "query": query,
        "fields": "paperId,title,year,citationCount,externalIds",
        "limit": min(k, 100),  # Max 100 per page
        "sort": "citationCount:desc",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])

            for paper in results:
                paper_id = paper.get("paperId")
                if not paper_id or paper_id in seen_ids:
                    continue

                seen_ids.add(paper_id)
                external_ids = paper.get("externalIds") or {}

                out.append((paper_id, {
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "openalex_id": external_ids.get("OpenAlex"),
                    "arxiv_id": external_ids.get("ArXiv"),
                    "doi": external_ids.get("DOI"),
                }))

            return out

        except requests.exceptions.RequestException as e:
            is_rate_limit = "429" in str(e)
            if attempt < MAX_RETRIES - 1:
                backoff = (2 ** (attempt + 1)) if is_rate_limit else RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(backoff)
                continue
            logger.warning(f"S2 highly-cited search failed: {e}")
            return out
        except Exception as e:
            logger.warning(f"S2 highly-cited search error: {e}")
            return out

    return out


def _search_crossref(query: str, k: int = CROSSREF_LIMIT) -> List[Tuple[str, Dict[str, Any]]]:
    """Search CrossRef for papers via DOI metadata.

    CrossRef provides excellent coverage of published papers with DOIs.
    No API key required, but we use polite pool with mailto for faster responses.

    Returns list of (doi, metadata_dict) tuples.
    """
    if not query or not CROSSREF_ENABLED:
        return []

    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": k,
        "select": "DOI,title,published,is-referenced-by-count,author,container-title",
    }
    headers = {
        "User-Agent": "Alexandria/1.0 (mailto:support@alexandria.app)",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])

            out: List[Tuple[str, Dict[str, Any]]] = []
            for item in items:
                doi = item.get("DOI")
                if not doi:
                    continue

                # Extract title (CrossRef returns list)
                titles = item.get("title", [])
                title = titles[0] if titles else ""

                # Extract year from published date-parts
                year = None
                published = item.get("published", {})
                date_parts = published.get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]

                # Citation count
                citations = item.get("is-referenced-by-count", 0)

                out.append((doi, {
                    "title": title,
                    "year": year,
                    "citations": citations,
                    "doi": doi,
                    "venue": (item.get("container-title") or [""])[0],
                }))
            return out

        except requests.exceptions.RequestException as e:
            is_rate_limit = "429" in str(e)
            if attempt < MAX_RETRIES - 1:
                # Longer backoff for rate limits (2s, 4s, 8s)
                backoff = (2 ** (attempt + 1)) if is_rate_limit else RETRY_BACKOFF_BASE * (2 ** attempt)
                if is_rate_limit:
                    logger.info(f"CrossRef rate limited, waiting {backoff}s before retry {attempt + 2}")
                time.sleep(backoff)
                continue
            logger.warning(f"CrossRef search failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"CrossRef search error: {e}")
            return []

    return []


def _search_pubmed(query: str, k: int = PUBMED_LIMIT) -> List[Tuple[str, Dict[str, Any]]]:
    """Search PubMed for biomedical literature.

    Two-step process: search for IDs, then fetch details.
    No API key required for basic use (3 req/s limit).

    Returns list of (pmid, metadata_dict) tuples.
    """
    if not query or not PUBMED_ENABLED:
        return []

    import xml.etree.ElementTree as ET

    # Step 1: Search for PubMed IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": k,
        "retmode": "json",
    }

    try:
        resp = requests.get(search_url, params=search_params, timeout=15)
        resp.raise_for_status()
        search_data = resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Step 2: Fetch details for found papers
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids[:k]),
            "retmode": "xml",
        }

        time.sleep(0.35)  # Rate limit: ~3 req/s without API key
        resp = requests.get(fetch_url, params=fetch_params, timeout=15)
        resp.raise_for_status()

        # Parse XML response
        root = ET.fromstring(resp.content)
        out: List[Tuple[str, Dict[str, Any]]] = []

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None:
                continue
            pmid = pmid_elem.text

            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Extract year
            year = None
            pub_date = article.find(".//PubDate/Year")
            if pub_date is not None and pub_date.text:
                try:
                    year = int(pub_date.text)
                except ValueError:
                    pass

            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Extract DOI if available
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            out.append((pmid, {
                "title": title,
                "year": year,
                "citations": 0,  # PubMed doesn't provide citation counts directly
                "doi": doi,
                "venue": journal,
                "pmid": pmid,
            }))

        return out

    except requests.exceptions.RequestException as e:
        logger.warning(f"PubMed search failed: {e}")
        return []
    except ET.ParseError as e:
        logger.warning(f"PubMed XML parse error: {e}")
        return []
    except Exception as e:
        logger.warning(f"PubMed search error: {e}")
        return []


def _search_dblp(query: str, k: int = DBLP_LIMIT) -> List[Tuple[str, Dict[str, Any]]]:
    """Search DBLP for computer science literature.

    DBLP has excellent coverage of CS conferences and journals.
    No API key required, very generous rate limits.

    Returns list of (dblp_key, metadata_dict) tuples.
    """
    if not query or not DBLP_ENABLED:
        return []

    url = "https://dblp.org/search/publ/api"
    params = {
        "q": query,
        "h": k,
        "format": "json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("result", {})
            hits = result.get("hits", {}).get("hit", [])

            out: List[Tuple[str, Dict[str, Any]]] = []
            for hit in hits:
                info = hit.get("info", {})
                dblp_key = info.get("key")
                if not dblp_key:
                    continue

                title = info.get("title", "")
                # Remove trailing period that DBLP sometimes adds
                if title.endswith("."):
                    title = title[:-1]

                # Extract year
                year = None
                year_str = info.get("year")
                if year_str:
                    try:
                        year = int(year_str)
                    except ValueError:
                        pass

                # Extract venue
                venue = info.get("venue", "")

                # Extract DOI if available
                doi = info.get("doi")

                out.append((dblp_key, {
                    "title": title,
                    "year": year,
                    "citations": 0,  # DBLP doesn't provide citation counts
                    "doi": doi,
                    "venue": venue,
                    "dblp_key": dblp_key,
                }))

            return out

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"DBLP search failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"DBLP search error: {e}")
            return []

    return []


def _validate_and_correct_metadata(
    conn: Connection,
    work_ids: List[str],
) -> int:
    """Validate paper metadata and attempt to correct suspicious entries.

    This is a GENERAL fix for data quality issues across ALL sources.

    Detects papers with suspicious metadata:
    1. Future years (year > current year) - clearly wrong
    2. Very recent years with high citations (suspicious - may be wrong paper)

    For suspicious papers, attempts to find correct metadata via Semantic Scholar
    title search and updates the DB if better data is found.

    Returns count of papers corrected.
    """
    if not work_ids:
        return 0

    import os
    from sqlalchemy import text

    # Load current metadata
    works = WorkStore.load_many(conn, work_ids)
    if not works:
        return 0

    # Find suspicious papers
    suspicious: List[Tuple[str, str, int, int]] = []  # (work_id, title, year, citations)

    for wid, work in works.items():
        year = work.year
        title = work.title or ""
        citations = work.cited_by_count or 0

        # Skip if no year or title
        if not year or not title:
            continue

        # Flag 1: Future year (clearly wrong)
        if year > CURRENT_YEAR:
            suspicious.append((wid, title, year, citations))
            logger.debug(f"Suspicious paper (future year {year}): {title[:50]}...")
            continue

        # Flag 2: Very recent year but seems like it should be older
        # Heuristic: Papers from last 2 years with 5000+ citations are suspicious
        # (Most papers don't accumulate that many citations in 2 years)
        if year >= CURRENT_YEAR - 1 and citations >= 5000:
            suspicious.append((wid, title, year, citations))
            logger.debug(f"Suspicious paper (recent {year} with {citations} cites): {title[:50]}...")
            continue

    if not suspicious:
        return 0

    logger.info(f"Found {len(suspicious)} papers with suspicious metadata, attempting correction")

    corrected = 0
    already_corrected: set = set()

    # FIRST: Check if we have a better version of the same paper in our DB
    # (e.g., ArXiv version with correct metadata)
    for wid, title, bad_year, bad_citations in suspicious:
        title_lower = title.lower().strip()
        # Search for other papers with very similar titles
        similar = conn.execute(
            text("""
                SELECT work_id, title, year, cited_by_count
                FROM works
                WHERE work_id != :wid
                AND lower(title) = :title_lower
                AND year IS NOT NULL
                ORDER BY cited_by_count DESC
                LIMIT 1
            """),
            {"wid": wid, "title_lower": title_lower},
        ).mappings().first()

        if similar:
            s_year = similar["year"]
            s_cites = similar["cited_by_count"] or 0

            # If the similar paper has earlier year or more citations, use its data
            if (s_year and s_year < bad_year) or (s_cites > bad_citations * 1.5):
                logger.info(
                    f"Found better duplicate in DB: {title[:40]}... "
                    f"Using {similar['work_id']} data (year={s_year}, cites={s_cites}) "
                    f"instead of {wid} (year={bad_year}, cites={bad_citations})"
                )
                conn.execute(
                    text("""
                        UPDATE works
                        SET year = :better_year,
                            cited_by_count = GREATEST(cited_by_count, :better_cites)
                        WHERE work_id = :wid
                    """),
                    {"wid": wid, "better_year": s_year, "better_cites": s_cites},
                )
                corrected += 1
                already_corrected.add(wid)

    # S2 correction disabled due to aggressive rate limiting
    # Skip remaining suspicious papers - rely on DB corrections only
    if False:  # Disabled
        s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"x-api-key": s2_api_key} if s2_api_key else {}

    for wid, title, bad_year, bad_citations in suspicious:
        # Skip if already corrected from DB
        if wid in already_corrected:
            continue

        # S2 validation disabled - skip all S2 API calls
        continue

        # Search S2 by title (DISABLED)
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            # Use exact title match for precision
            params = {
                "query": title[:100],  # Truncate long titles
                "fields": "paperId,title,year,citationCount",
                "limit": 5,
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)

            if resp.status_code == 429:
                logger.debug("S2 rate limited during validation, skipping remaining")
                break

            if resp.status_code != 200:
                continue

            data = resp.json()
            papers = data.get("data", [])

            # Find best match by title similarity
            best_match = None
            best_score = 0
            title_lower = title.lower().strip()

            for paper in papers:
                s2_title = (paper.get("title") or "").lower().strip()
                s2_year = paper.get("year")
                s2_cites = paper.get("citationCount", 0)

                # Calculate title similarity (simple containment check)
                if s2_title == title_lower:
                    score = 1.0
                elif title_lower in s2_title or s2_title in title_lower:
                    score = 0.8
                else:
                    # Check word overlap
                    title_words = set(title_lower.split())
                    s2_words = set(s2_title.split())
                    overlap = len(title_words & s2_words) / max(len(title_words), 1)
                    score = overlap * 0.7

                # Boost score if S2 has earlier year and more citations
                if s2_year and s2_year < bad_year and s2_cites > bad_citations:
                    score += 0.2

                if score > best_score and score >= 0.7:
                    best_score = score
                    best_match = paper

            if best_match:
                s2_year = best_match.get("year")
                s2_cites = best_match.get("citationCount", 0)

                # Only update if S2 data is clearly better
                # Better = earlier year OR significantly more citations
                should_update = False

                if s2_year and s2_year < bad_year:
                    should_update = True
                    logger.info(f"Correcting year: {title[:40]}... {bad_year} → {s2_year}")

                if s2_cites > bad_citations * 2:  # S2 has 2x more citations
                    should_update = True
                    logger.info(f"Correcting citations: {title[:40]}... {bad_citations} → {s2_cites}")

                if should_update:
                    # Update DB with better metadata
                    conn.execute(
                        text("""
                            UPDATE works
                            SET year = CASE
                                    WHEN :s2_year < year OR year > :current_year THEN :s2_year
                                    ELSE year
                                END,
                                cited_by_count = GREATEST(cited_by_count, :s2_cites)
                            WHERE work_id = :work_id
                        """),
                        {
                            "work_id": wid,
                            "s2_year": s2_year,
                            "s2_cites": s2_cites,
                            "current_year": CURRENT_YEAR,
                        },
                    )
                    corrected += 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.debug(f"Failed to validate {wid}: {e}")
            continue

    if corrected > 0:
        logger.info(f"Corrected metadata for {corrected} papers")

    return corrected


def _search_openalex_highly_cited(query: str, k: int = OPENALEX_HIGHLY_CITED_LIMIT) -> List[Tuple[str, float]]:
    """Search OpenAlex for highly-cited papers sorted by citation count.

    This ensures foundational/seminal papers are retrieved regardless of
    recency bias in default search ranking. Uses sort=cited_by_count:desc
    to get the most influential papers for a query.

    Returns list of (work_id, citation_count) tuples.
    """
    if not query:
        return []

    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "sort": "cited_by_count:desc",
        "per-page": min(k, 200),
        "select": "id,cited_by_count",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            out: List[Tuple[str, float]] = []
            for r in results:
                oa_id = r.get("id", "")
                if oa_id.startswith("https://openalex.org/"):
                    work_id = oa_id.replace("https://openalex.org/", "")
                    citations = r.get("cited_by_count", 0)
                    out.append((work_id, float(citations)))

            logger.info(f"OpenAlex highly-cited search: {len(out)} papers for '{query[:30]}...'")
            return out

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"OpenAlex highly-cited search HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenAlex highly-cited search error: {e}")
            return []

    return []


def _search_openalex_by_title(title: str, k: int = 10) -> List[Tuple[str, float]]:
    """Search OpenAlex for papers by title using title.search filter.

    This is specifically designed for finding foundational papers by their
    exact or near-exact title (e.g., "Attention Is All You Need").
    Uses filter=title.search which is MUCH better at title matching than
    the generic search= parameter.

    Returns list of (work_id, citation_count) tuples.
    """
    if not title or len(title) < 4:
        return []

    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{title}",
        "sort": "cited_by_count:desc",
        "per_page": min(k, 50),
        "select": "id,cited_by_count,title",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            out: List[Tuple[str, float]] = []
            for r in results:
                oa_id = r.get("id", "")
                if oa_id.startswith("https://openalex.org/"):
                    work_id = oa_id.replace("https://openalex.org/", "")
                    citations = r.get("cited_by_count", 0)
                    out.append((work_id, float(citations)))

            if out:
                logger.info(f"OpenAlex title search: {len(out)} papers for '{title[:40]}...'")
            return out

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"OpenAlex title search HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenAlex title search error: {e}")
            return []

    return []


def _search_arxiv(query: str, k: int = ARXIV_LIMIT) -> List[Tuple[str, Dict[str, Any]]]:
    """Search ArXiv for papers, returning paper metadata.

    ArXiv is a co-equal source specializing in preprints and ML papers.

    Returns list of (arxiv_id, metadata_dict) tuples.
    """
    if not query or not ARXIV_ENABLED:
        return []

    import xml.etree.ElementTree as ET

    base_url = "https://export.arxiv.org/api/query"
    # Replace spaces with AND for better matching
    # Note: We must build the URL manually because requests URL-encodes '+' to '%2B',
    # which breaks the ArXiv API's AND query syntax
    from urllib.parse import quote
    words = query.split()[:5]
    search_terms = [f"all:{quote(word)}" for word in words]
    search_query = "+AND+".join(search_terms)
    url = f"{base_url}?search_query={search_query}&max_results={k}&sortBy=relevance"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            out: List[Tuple[str, Dict[str, Any]]] = []
            for entry in root.findall("atom:entry", ns):
                id_elem = entry.find("atom:id", ns)
                if id_elem is None or id_elem.text is None:
                    continue

                arxiv_url = id_elem.text
                arxiv_id = arxiv_url.split("/abs/")[-1].split("v")[0] if "/abs/" in arxiv_url else None
                if not arxiv_id:
                    continue

                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

                published_elem = entry.find("atom:published", ns)
                year = None
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, TypeError):
                        pass

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text)

                out.append((arxiv_id, {
                    "title": title,
                    "year": year,
                    "citations": 0,
                    "authors": authors,
                    "arxiv_id": arxiv_id,
                }))

            # S2 enrichment disabled due to aggressive rate limiting
            # ArXiv papers will use internal reference counting instead
            return out

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"ArXiv search failed: {e}")
            return []
        except ET.ParseError as e:
            logger.warning(f"ArXiv XML parse error: {e}")
            return []
        except Exception as e:
            logger.warning(f"ArXiv search error: {e}")
            return []

    return []


def _enrich_arxiv_with_crossref(
    papers: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Enrich ArXiv papers with citation counts from CrossRef (by title search).

    CrossRef is fast (no rate limits with polite pool) and provides:
    - Citation counts (is-referenced-by-count) for published versions
    - DOIs for cross-source linking

    This is faster than S2 or OpenAlex sequential lookups by using parallel requests.
    """
    if not papers:
        return papers

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import urllib.parse

    enrichment_map: Dict[str, Dict[str, Any]] = {}

    def fetch_crossref_by_title(arxiv_id: str, title: str) -> Tuple[str, Optional[Dict]]:
        """Search CrossRef by title to find published version."""
        if not title or len(title) < 10:
            return arxiv_id, None

        try:
            # URL-encode the title for query
            url = "https://api.crossref.org/works"
            params = {
                "query.title": title,
                "rows": 3,
                "select": "DOI,title,is-referenced-by-count",
            }
            headers = {
                "User-Agent": "Alexandria/1.0 (mailto:support@alexandria.app)",
            }
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("message", {}).get("items", [])

                # Find best title match
                title_lower = title.lower().strip()
                for item in items:
                    item_titles = item.get("title", [])
                    if item_titles:
                        item_title = item_titles[0].lower().strip()
                        # Check for high similarity (simple check: starts with same words)
                        if item_title == title_lower or title_lower[:50] in item_title or item_title[:50] in title_lower:
                            return arxiv_id, {
                                "citations": item.get("is-referenced-by-count", 0),
                                "doi": item.get("DOI"),
                            }
            return arxiv_id, None
        except Exception:
            return arxiv_id, None

    # Run parallel requests (CrossRef allows high concurrency with polite pool)
    papers_to_enrich = papers[:50]  # Limit to 50 papers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fetch_crossref_by_title, arxiv_id, meta.get("title", ""))
            for arxiv_id, meta in papers_to_enrich
        ]

        for future in as_completed(futures):
            try:
                arxiv_id, result = future.result()
                if result and result.get("citations", 0) > 0:
                    enrichment_map[arxiv_id] = result
            except Exception:
                pass

    # Apply enrichment to papers
    enriched = []
    for arxiv_id, meta in papers:
        if arxiv_id in enrichment_map:
            enrich_data = enrichment_map[arxiv_id]
            meta["citations"] = enrich_data["citations"]
            meta["doi"] = enrich_data.get("doi")
        enriched.append((arxiv_id, meta))

    enriched_count = len(enrichment_map)
    if enriched_count > 0:
        logger.info(f"Enriched {enriched_count}/{len(papers)} ArXiv papers with CrossRef citation data")

    return enriched


def _enrich_arxiv_with_opencitations(
    papers: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Enrich ArXiv papers with citation counts from OpenCitations (free, fast).

    ArXiv papers have DOIs in format: 10.48550/arXiv.XXXX.XXXXX
    OpenCitations provides citation counts via: /citation-count/doi:{doi}

    This is much faster than Scraping Dog or Semantic Scholar.
    """
    if not papers or not OPENCITATIONS_ENABLED:
        return papers

    from concurrent.futures import ThreadPoolExecutor, as_completed

    enrichment_map: Dict[str, Dict[str, Any]] = {}

    def fetch_opencitations_count(arxiv_id: str) -> Tuple[str, Optional[Dict]]:
        """Get citation count from OpenCitations using ArXiv DOI."""
        # Construct ArXiv DOI: 10.48550/arXiv.XXXX.XXXXX
        doi = f"10.48550/arXiv.{arxiv_id}"

        try:
            url = f"https://api.opencitations.net/index/v2/citation-count/doi:{doi}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Response is a list with one item: [{"count": "123", "doi": "..."}]
                if data and len(data) > 0:
                    count_str = data[0].get("count", "0")
                    citations = int(count_str) if count_str else 0
                    if citations > 0:
                        return arxiv_id, {"citations": citations}
            return arxiv_id, None
        except Exception as e:
            logger.debug(f"OpenCitations enrichment failed for {arxiv_id}: {e}")
            return arxiv_id, None

    # Enrich all papers (OpenCitations is fast and free)
    papers_to_enrich = papers[:50]

    # Run parallel requests (OpenCitations allows high concurrency)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(fetch_opencitations_count, arxiv_id)
            for arxiv_id, _ in papers_to_enrich
        ]

        for future in as_completed(futures):
            try:
                arxiv_id, result = future.result()
                if result and result.get("citations", 0) > 0:
                    enrichment_map[arxiv_id] = result
            except Exception:
                pass

    # Apply enrichment to papers
    enriched = []
    for arxiv_id, meta in papers:
        if arxiv_id in enrichment_map:
            meta["citations"] = enrichment_map[arxiv_id]["citations"]
            logger.info(f"OpenCitations enriched ArXiv {arxiv_id}: {meta['citations']} cites")
        enriched.append((arxiv_id, meta))

    enriched_count = len(enrichment_map)
    if enriched_count > 0:
        logger.info(f"Enriched {enriched_count}/{len(papers)} ArXiv papers with OpenCitations data")

    return enriched


def _enrich_arxiv_with_citations(
    papers: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Enrich ArXiv papers with citation counts and external IDs from Semantic Scholar.

    S2 has comprehensive coverage of ArXiv papers and provides:
    - Accurate citation counts (critical for ranking)
    - External IDs (OpenAlex, DOI) for cross-source linking

    This enables proper deduplication and ensures seminal papers like DDPM
    get correct citation data regardless of which source retrieved them.
    """
    if not papers:
        return papers

    # S2 enrichment completely disabled due to aggressive rate limiting
    # Using internal reference counting instead for foundational paper detection
    return papers

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}

    # Store enrichment data: arxiv_id -> {citations, openalex_id, doi, s2_id}
    enrichment_map: Dict[str, Dict[str, Any]] = {}

    # Query S2 for citation counts AND external IDs (limit to 20 for foundational papers)
    rate_limit_retries = 0
    for arxiv_id, _ in papers[:20]:
        try:
            # Request external IDs to enable cross-source linking
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount,externalIds,paperId"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                citations = data.get("citationCount", 0)
                external_ids = data.get("externalIds") or {}
                s2_id = data.get("paperId")

                enrichment_map[arxiv_id] = {
                    "citations": citations,
                    "openalex_id": external_ids.get("OpenAlex"),
                    "doi": external_ids.get("DOI"),
                    "s2_id": s2_id,
                }
                if citations:
                    logger.info(f"Enriched ArXiv {arxiv_id}: {citations} cites, OA={external_ids.get('OpenAlex')}")
                rate_limit_retries = 0  # Reset on success
            elif resp.status_code == 429:
                rate_limit_retries += 1
                if rate_limit_retries >= 3:
                    logger.warning("S2 rate limited during enrichment, stopping after 3 retries")
                    break
                # Exponential backoff: 2s, 4s, 8s
                backoff = 2 ** rate_limit_retries
                logger.info(f"S2 rate limited, waiting {backoff}s before retry")
                time.sleep(backoff)
                continue  # Retry the same paper
            # Delay between requests to stay under rate limit
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"S2 enrichment failed for {arxiv_id}: {e}")

    # Apply enrichment to papers
    enriched = []
    for arxiv_id, meta in papers:
        if arxiv_id in enrichment_map:
            enrich_data = enrichment_map[arxiv_id]
            meta["citations"] = enrich_data["citations"]
            meta["openalex_id"] = enrich_data.get("openalex_id")
            meta["doi"] = enrich_data.get("doi")
            meta["s2_id"] = enrich_data.get("s2_id")
        enriched.append((arxiv_id, meta))

    enriched_count = len(enrichment_map)
    if enriched_count > 0:
        logger.info(f"Enriched {enriched_count}/{len(papers)} ArXiv papers with S2 data")

    return enriched


def _ingest_arxiv_papers(
    conn: Connection,
    papers: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Ingest papers from ArXiv into the works table.

    ArXiv is a co-equal source - papers use AX:{arxiv_id} as their ID.
    DOI/ArXiv ID-based deduplication: if paper with same ID exists, use existing work_id.
    Title-based fallback: if no ID match but same title exists, use existing work_id.

    Returns:
        Tuple of:
        - List of work_ids that were successfully ingested
        - Dict mapping lowercase title -> work_id for cross-source deduplication
        - Dict mapping normalized arxiv_id -> work_id for cross-source deduplication
    """
    if not papers:
        return [], {}

    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy import bindparam

    # Track titles and arxiv_ids to work_ids for cross-source deduplication
    title_to_work_id: Dict[str, str] = {}
    arxiv_id_to_work_id: Dict[str, str] = {}

    # Pre-fetch existing papers by ArXiv ID for deduplication (primary)
    arxiv_ids_to_check = [normalize_arxiv_id(arxiv_id) for arxiv_id, _ in papers]
    arxiv_ids_to_check = [aid for aid in arxiv_ids_to_check if aid]  # Remove None

    # Pre-fetch existing papers by title for fallback deduplication
    titles_to_check = [meta.get("title") for _, meta in papers if meta.get("title")]
    existing_by_title: Dict[str, Tuple[str, int]] = {}  # title -> (work_id, citations)

    # Build ArXiv ID dedup map
    if arxiv_ids_to_check:
        try:
            rows = conn.execute(
                text("""
                    SELECT DISTINCT ON (arxiv_id)
                        arxiv_id, work_id
                    FROM works
                    WHERE arxiv_id = ANY(:arxiv_ids)
                    ORDER BY arxiv_id, work_id
                """),
                {"arxiv_ids": arxiv_ids_to_check},
            ).fetchall()
            for row in rows:
                arxiv_id_to_work_id[row[0]] = row[1]
        except Exception as e:
            logger.warning(f"Failed to check existing papers by arxiv_id: {e}")

    # Build title dedup map (fallback)
    if titles_to_check:
        try:
            rows = conn.execute(
                text("""
                    SELECT DISTINCT ON (lower(title))
                        work_id, title, cited_by_count
                    FROM works
                    WHERE lower(title) = ANY(:titles)
                    ORDER BY lower(title), cited_by_count DESC NULLS LAST
                """),
                {"titles": [t.lower() for t in titles_to_check if t]},
            ).fetchall()
            for row in rows:
                existing_by_title[row[1].lower()] = (row[0], row[2] or 0)
        except Exception as e:
            logger.warning(f"Failed to check existing papers by title: {e}")

    work_ids = []
    for arxiv_id, meta in papers:
        title = meta.get("title", "")
        normalized_arxiv = normalize_arxiv_id(arxiv_id)

        # Multi-stage deduplication:
        # Stage 1: ArXiv ID match (primary)
        if normalized_arxiv and normalized_arxiv in arxiv_id_to_work_id:
            work_id = arxiv_id_to_work_id[normalized_arxiv]
            logger.debug(f"ArXiv {arxiv_id} deduped to existing {work_id} (ArXiv ID match)")
        # Stage 2: Title match (fallback)
        elif title and title.lower() in existing_by_title:
            existing_id, _ = existing_by_title[title.lower()]
            work_id = existing_id
            logger.debug(f"ArXiv {arxiv_id} deduped to existing {work_id} (title match)")
        # Stage 3: Create new work_id
        else:
            work_id = f"AX:{arxiv_id}"

        work_ids.append(work_id)

        # Update dedup maps for subsequent sources
        if normalized_arxiv:
            arxiv_id_to_work_id[normalized_arxiv] = work_id
        if title:
            title_to_work_id[title.lower()] = work_id

        try:
            # Upsert with preference for better data
            conn.execute(
                text("""
                    INSERT INTO works (
                        work_id, title, year, cited_by_count,
                        authors_json, venue, primary_topic_id,
                        primary_topic_score, topics_json, is_retracted,
                        abstract, arxiv_id
                    ) VALUES (
                        :work_id, :title, :year, :cited_by_count,
                        :authors_json, :venue, :primary_topic_id,
                        :primary_topic_score, :topics_json, :is_retracted,
                        :abstract, :arxiv_id
                    )
                    ON CONFLICT (work_id) DO UPDATE
                    SET
                        -- Keep higher citation count (S2 enrichment is usually accurate)
                        cited_by_count = GREATEST(
                            COALESCE(works.cited_by_count, 0),
                            COALESCE(EXCLUDED.cited_by_count, 0)
                        ),
                        -- Fill in title if missing
                        title = COALESCE(works.title, EXCLUDED.title),
                        -- Prefer earlier year (ArXiv often has correct original pub date)
                        year = CASE
                            WHEN works.year IS NULL THEN EXCLUDED.year
                            WHEN EXCLUDED.year IS NULL THEN works.year
                            WHEN EXCLUDED.year < works.year THEN EXCLUDED.year
                            ELSE works.year
                        END,
                        -- Fill in abstract if missing
                        abstract = COALESCE(works.abstract, EXCLUDED.abstract),
                        -- Fill in arxiv_id if missing
                        arxiv_id = COALESCE(works.arxiv_id, EXCLUDED.arxiv_id)
                """).bindparams(
                    bindparam("authors_json", type_=JSONB),
                    bindparam("topics_json", type_=JSONB),
                ),
                {
                    "work_id": work_id,
                    "title": meta.get("title"),
                    "year": meta.get("year"),
                    "cited_by_count": meta.get("citations", 0),  # Use enriched citation count
                    "authors_json": meta.get("authors", []),
                    "venue": "arXiv",
                    "primary_topic_id": None,
                    "primary_topic_score": None,
                    "topics_json": [],
                    "is_retracted": False,
                    "abstract": meta.get("abstract"),
                    "arxiv_id": normalized_arxiv,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to ingest ArXiv paper {work_id}: {e}")

    return work_ids, title_to_work_id, arxiv_id_to_work_id


def _ingest_semantic_scholar_papers(
    conn: Connection,
    papers: List[Tuple[str, Dict[str, Any]]],
    arxiv_titles: Optional[Dict[str, str]] = None,
    arxiv_ids: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Ingest papers from Semantic Scholar into the works table with cross-source linking.

    Uses multi-stage deduplication to avoid duplicates:
    1. DOI match (primary) → use existing work_id
    2. ArXiv ID match → use existing work_id
    3. Title match → use existing work_id (includes freshly-inserted papers from prior sources)
    4. No match → create new S2:{id}

    This ensures papers retrieved from multiple sources are properly unified.

    Parameters:
        conn: Database connection
        papers: List of (s2_id, metadata) tuples from Semantic Scholar
        arxiv_titles: Optional dict mapping lowercase title -> work_id from ArXiv ingestion
        arxiv_ids: Optional dict mapping normalized arxiv_id -> work_id from ArXiv ingestion

    Returns:
        Tuple of:
        - List of work_ids that were successfully ingested
        - Dict mapping lowercase title -> work_id for cross-source deduplication
        - Dict mapping normalized arxiv_id -> work_id for cross-source deduplication
    """
    if not papers:
        return []

    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy import bindparam

    work_ids = []
    rows_to_insert = []

    # Initialize dedup maps (combine fresh ArXiv data with DB lookups)
    doi_to_work_id: Dict[str, str] = {}
    arxiv_id_to_work_id: Dict[str, str] = arxiv_ids.copy() if arxiv_ids else {}
    title_to_work_id: Dict[str, str] = arxiv_titles.copy() if arxiv_titles else {}

    # Extract DOIs and ArXiv IDs from S2 papers for pre-fetching
    dois_to_check = []
    arxiv_ids_to_check = []
    for s2_id, meta in papers:
        doi = meta.get("doi")
        arxiv_id = meta.get("arxiv_id")
        if doi:
            normalized_doi = normalize_doi_for_dedup(doi)
            if normalized_doi:
                dois_to_check.append(normalized_doi)
        if arxiv_id:
            normalized_arxiv = normalize_arxiv_id(arxiv_id)
            if normalized_arxiv:
                arxiv_ids_to_check.append(normalized_arxiv)

    # Pre-fetch existing papers by DOI and ArXiv ID from DB
    db_doi_map, db_arxiv_map = build_external_id_maps(conn, dois_to_check, arxiv_ids_to_check)
    # Merge DB results into our maps (don't overwrite fresh ArXiv data)
    for doi, wid in db_doi_map.items():
        if doi not in doi_to_work_id:
            doi_to_work_id[doi] = wid
    for arxiv_id, wid in db_arxiv_map.items():
        if arxiv_id not in arxiv_id_to_work_id:
            arxiv_id_to_work_id[arxiv_id] = wid

    # Pre-fetch existing papers by title for fallback deduplication
    titles_to_check = [meta.get("title") for _, meta in papers if meta.get("title")]
    existing_by_title: Dict[str, Tuple[str, int]] = {}  # title -> (work_id, citations)

    if titles_to_check:
        try:
            rows = conn.execute(
                text("""
                    SELECT DISTINCT ON (lower(title))
                        work_id, title, cited_by_count
                    FROM works
                    WHERE lower(title) = ANY(:titles)
                    ORDER BY lower(title), cited_by_count DESC NULLS LAST
                """),
                {"titles": [t.lower() for t in titles_to_check if t]},
            ).fetchall()
            for row in rows:
                existing_by_title[row[1].lower()] = (row[0], row[2] or 0)
        except Exception as e:
            logger.warning(f"Failed to check existing papers by title: {e}")

    # Merge existing title lookups into title_to_work_id
    for title_lower, (wid, _) in existing_by_title.items():
        if title_lower not in title_to_work_id:
            title_to_work_id[title_lower] = wid

    for s2_id, meta in papers:
        title = meta.get("title", "")
        doi = meta.get("doi")
        arxiv_id = meta.get("arxiv_id")

        # Normalize external IDs
        normalized_doi = normalize_doi_for_dedup(doi) if doi else None
        normalized_arxiv = normalize_arxiv_id(arxiv_id) if arxiv_id else None

        # Multi-stage deduplication:
        # Stage 1: DOI match (highest priority)
        if normalized_doi and normalized_doi in doi_to_work_id:
            work_id = doi_to_work_id[normalized_doi]
            logger.debug(f"S2 {s2_id[:12]} deduped to existing {work_id} (DOI match)")
        # Stage 2: ArXiv ID match
        elif normalized_arxiv and normalized_arxiv in arxiv_id_to_work_id:
            work_id = arxiv_id_to_work_id[normalized_arxiv]
            logger.debug(f"S2 {s2_id[:12]} deduped to existing {work_id} (ArXiv ID match)")
        # Stage 3: Title match (fallback)
        elif title and title.lower() in title_to_work_id:
            work_id = title_to_work_id[title.lower()]
            logger.debug(f"S2 {s2_id[:12]} deduped to existing {work_id} (title match)")
        # Stage 4: Create new work_id
        else:
            work_id = f"S2:{s2_id[:20]}"

        work_ids.append(work_id)

        # Update dedup maps for subsequent sources
        if normalized_doi:
            doi_to_work_id[normalized_doi] = work_id
        if normalized_arxiv:
            arxiv_id_to_work_id[normalized_arxiv] = work_id
        if title:
            title_to_work_id[title.lower()] = work_id

        # Prepare row for insertion
        rows_to_insert.append({
            "work_id": work_id,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "cited_by_count": meta.get("citations", 0),
            "authors_json": [],  # S2 search doesn't return full author info
            "venue": None,
            "primary_topic_id": None,
            "primary_topic_score": None,
            "topics_json": [],
            "is_retracted": False,
            "abstract": None,
            "doi": normalized_doi,
            "arxiv_id": normalized_arxiv,
            "source": "semantic_scholar",
        })

    # Batch upsert - preserve better metadata from S2:
    # - Keep HIGHER citation count (S2 often more accurate than OpenAlex)
    # - Keep EARLIER year (prevents future-dated papers from wrong OpenAlex data)
    # - Fill in title/year if missing
    if rows_to_insert:
        for row in rows_to_insert:
            try:
                conn.execute(
                    text("""
                        INSERT INTO works (
                            work_id, title, year, cited_by_count,
                            authors_json, venue, primary_topic_id,
                            primary_topic_score, topics_json, is_retracted,
                            abstract, doi, arxiv_id
                        ) VALUES (
                            :work_id, :title, :year, :cited_by_count,
                            :authors_json, :venue, :primary_topic_id,
                            :primary_topic_score, :topics_json, :is_retracted,
                            :abstract, :doi, :arxiv_id
                        )
                        ON CONFLICT (work_id) DO UPDATE
                        SET
                            -- Keep higher citation count (S2 often more accurate)
                            cited_by_count = GREATEST(
                                COALESCE(works.cited_by_count, 0),
                                COALESCE(EXCLUDED.cited_by_count, 0)
                            ),
                            -- Keep earlier year (prevents future-dated wrong papers)
                            year = CASE
                                WHEN works.year IS NULL THEN EXCLUDED.year
                                WHEN EXCLUDED.year IS NULL THEN works.year
                                WHEN EXCLUDED.year < works.year THEN EXCLUDED.year
                                ELSE works.year
                            END,
                            -- Fill in title if missing
                            title = COALESCE(works.title, EXCLUDED.title),
                            -- Fill in external IDs if missing
                            doi = COALESCE(works.doi, EXCLUDED.doi),
                            arxiv_id = COALESCE(works.arxiv_id, EXCLUDED.arxiv_id)
                    """).bindparams(
                        bindparam("authors_json", type_=JSONB),
                        bindparam("topics_json", type_=JSONB),
                    ),
                    row,
                )
            except Exception as e:
                logger.warning(f"Failed to ingest S2 paper {row['work_id']}: {e}")

    return work_ids, title_to_work_id, arxiv_id_to_work_id


def _search_openalex(query: str, k: int, year_filter: Optional[str] = None) -> List[Tuple[str, float]]:
    """Perform a lexical search against OpenAlex and return up to `k` work IDs.

    Each result is returned as a tuple `(work_id, score)` where `work_id` is
    the OpenAlex work ID (e.g. "W1234567890") and `score` is a heuristic
    representing the search ranking.  The OpenAlex search endpoint does
    not return a numeric score, so we derive a simple score of
    `1/(rank_index+1)` to keep higher‑ranked results ahead of lower ones.

    If the request fails (e.g. due to network issues or missing API key),
    this function returns an empty list. Retries up to MAX_RETRIES times
    with exponential backoff on transient failures.

    Parameters
    ----------
    query : str
        Search query string.
    k : int
        Maximum number of results to return.
    year_filter : str, optional
        OpenAlex year filter (e.g., "2018-2025" for recent papers).
    """
    if not query:
        return []

    params = {
        "search": query,
        "per-page": k,
    }
    if year_filter:
        # OpenAlex filter syntax: publication_year:>2017 for "2018 and later"
        # The format "2018-2026" needs to be converted to ">2017"
        if "-" in year_filter:
            start_year = int(year_filter.split("-")[0])
            params["filter"] = f"publication_year:>{start_year - 1}"
            logger.debug(f"OpenAlex year filter: {year_filter} -> {params['filter']}")
        else:
            params["filter"] = f"publication_year:{year_filter}"
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    url = "https://api.openalex.org/works"

    results = []
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            logger.debug(f"OpenAlex API returned {len(results)} results for query: {query[:30]}")
            break
        except requests.exceptions.RequestException as e:
            logger.debug(f"OpenAlex request error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            return []
        except Exception as e:
            logger.debug(f"OpenAlex unexpected error: {e}")
            return []

    out: List[Tuple[str, float]] = []
    for idx, result in enumerate(results):
        work_id_full = result.get("id")
        if not work_id_full or "/" not in work_id_full:
            continue
        work_id = work_id_full.rsplit("/", 1)[-1]
        score = 1.0 / float(idx + 1)
        out.append((work_id, score))
    return out

def _retrieve_topic_pool(conn: Connection, topic_id: str, k: int) -> List[Tuple[str, float]]:
    """Retrieve works by primary topic as a backstop.

    Selects works whose primary topic matches the given `topic_id`, ordered
    by their primary topic score, citation count, year and ID.  Returns
    `(work_id, score)` tuples where `score` is the primary topic score.
    """
    if not topic_id:
        return []
    rows = conn.execute(
        text(
            """
            SELECT work_id, COALESCE(primary_topic_score, 0) AS score
            FROM works
            WHERE primary_topic_id = :topic
            ORDER BY COALESCE(primary_topic_score, 0) DESC,
                     COALESCE(cited_by_count, 0) DESC,
                     COALESCE(year, 0) DESC,
                     work_id ASC
            LIMIT :k
            """
        ),
        {"topic": topic_id, "k": k},
    ).mappings().all()
    return [(r["work_id"], float(r["score"])) for r in rows]

def generate_candidates_direct(
    conn: Connection,
    *,
    tenant_id: Any,
    query_text: str,
    context_json: Optional[Dict[str, Any]] = None,
    filters_json: Optional[Dict[str, Any]] = None,
    rank_params_json: Optional[Dict[str, Any]] = None,
    limit_pool: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], QueryExpansion]:
    """Generate a candidate pool for direct ranking using OpenAlex lexical search.

    This function orchestrates:
    1. Query expansion via LLM
    2. Lexical retrieval via OpenAlex for each expanded query
    3. Merge + dedupe results and apply year filters
    4. Topic backstop (optional)

    Parameters
    ----------
    conn : Connection
        Active SQLAlchemy connection.
    tenant_id : Any
        Tenant ID used for candidate set insertion (not used yet, but kept for
        future tenant‑scoped logic).
    query_text : str
        The user query used for retrieval.
    context_json : dict, optional
        Additional context, including `target_topic_id` if available.
    filters_json : dict, optional
        Hard filters such as `year_min` and `year_max`.
    rank_params_json : dict, optional
        Parameters controlling pool size and per‑source limits.
    limit_pool : int, optional
        Maximum number of candidates to return (defaults to DEFAULT_LIMIT_POOL).

    Returns
    -------
    Tuple[List[Dict[str, Any]], QueryExpansion]
        A tuple of:
        - List of dicts each with `work_id`, `provenance`, and `partition` keys
        - QueryExpansion object with original_terms, expansion_terms, query_hash
    """
    context_json = context_json or {}
    filters_json = filters_json or {}
    rank_params_json = rank_params_json or {}

    # Normalize query to strip question prefixes for consistent retrieval
    query_text = normalize_query_for_expansion(query_text)

    # Determine retrieval limits
    k_lex = int(rank_params_json.get("k_lexical", DEFAULT_K_LEXICAL))
    k_topic = int(rank_params_json.get("k_topic", DEFAULT_K_TOPIC))
    max_pool = int(limit_pool or rank_params_json.get("limit_pool", DEFAULT_LIMIT_POOL))

    if k_lex < 0:
        k_lex = DEFAULT_K_LEXICAL
    if k_topic < 0:
        k_topic = DEFAULT_K_TOPIC

    # 1) Query expansion via LLM
    query_expansion = expand_query(conn, query_text)
    logger.info(f"Query expansion: {len(query_expansion.concepts)} concepts, {len(query_expansion.expansion_terms)} expansion terms")

    # 2) Build importance-weighted search queries from concepts
    # Structure: list of (query_string, importance_weight, source_concept)
    weighted_queries: List[Tuple[str, float, Optional[str]]] = []

    # Always include original query with full weight
    weighted_queries.append((query_text, 1.0, None))

    # If we have structured concepts, use importance-weighted expansion
    if query_expansion.concepts:
        for concept in query_expansion.concepts:
            importance = concept.importance
            # Synonyms get full importance weight (safe, direct alternatives)
            for syn in concept.synonyms:
                if syn.lower() != query_text.lower():
                    weighted_queries.append((syn, importance, concept.term))
            # Related terms get reduced weight (can drift)
            for rel in concept.related_terms:
                if rel.lower() != query_text.lower():
                    weighted_queries.append((rel, importance * 0.5, concept.term))
    else:
        # Fallback to legacy flat expansion
        for term in query_expansion.expansion_terms:
            if term.lower() != query_text.lower():
                weighted_queries.append((term, 0.7, None))

    # Sort by importance (highest first) to prioritize important concept searches
    weighted_queries.sort(key=lambda x: -x[1])

    # LATENCY OPTIMIZATION: Limit to top N queries to reduce API calls
    # Original query is always first (weight 1.0), keep up to 5 more
    MAX_OPENALEX_QUERIES = 6
    if len(weighted_queries) > MAX_OPENALEX_QUERIES:
        logger.info(f"Limiting OpenAlex queries from {len(weighted_queries)} to {MAX_OPENALEX_QUERIES}")
        weighted_queries = weighted_queries[:MAX_OPENALEX_QUERIES]

    # Allocate retrieval budget based on importance
    total_weight = sum(w for _, w, _ in weighted_queries)
    candidate_map: Dict[str, List[Dict[str, Any]]] = {}

    # PARALLELIZATION: Run all OpenAlex searches concurrently
    # This dramatically reduces retrieval time from O(n_queries * latency) to O(latency)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Thread-safe lock for candidate_map updates
    candidate_map_lock = threading.Lock()

    # Store OpenAlex results temporarily - we'll dedupe against ArXiv/S2 later
    # This is needed because ArXiv/S2 may have better metadata for the same paper
    openalex_results: List[Tuple[str, str, Dict[str, Any]]] = []  # (work_id, title, prov_entry)
    openalex_results_lock = threading.Lock()

    def search_and_store(q: str, weight: float, source_concept: Optional[str], is_recent: bool = False):
        """Execute a single OpenAlex search and store results thread-safely."""
        if is_recent:
            year_filter = "2018-2026"
            query_k = OPENALEX_RECENT_LIMIT  # 250 recent papers per query
            source = "lexical_openalex_recent"
        elif source_concept == "foundational_work":
            # Foundational works get special source tag for priority boosting in ranking
            year_filter = None
            query_k = 20  # Small batch for exact title search
            source = "foundational_title"
        else:
            year_filter = None
            query_k = max(10, int(k_lex * (weight / total_weight)))
            source = "lexical_openalex"

        lex_results = _search_openalex(q, query_k, year_filter=year_filter)
        suffix = " (recent 2018+)" if is_recent else ""
        logger.info(f"OpenAlex{suffix}: {len(lex_results)} results for '{q[:50]}' (weight={weight:.2f})")

        # Store results temporarily - we'll add to candidate_map after ArXiv/S2 dedup
        with openalex_results_lock:
            for idx, (wid, score) in enumerate(lex_results):
                prov_entry = {
                    "source": source,
                    "rank": idx + 1,
                    "score": score * weight,
                    "query": q,
                    "concept": source_concept,
                }
                openalex_results.append((wid, None, prov_entry))  # title fetched later

    # Build list of all search tasks
    search_tasks = []
    for q, weight, source_concept in weighted_queries:
        search_tasks.append((q, weight, source_concept, False))

    # 2b) Recent papers retrieval - separate search for 2018+ papers to ensure temporal diversity
    # Use original query only for recent search (250 papers)
    search_tasks.append((query_text, 1.0, None, True))

    # Semantic Scholar and ArXiv results (populated by parallel search)
    s2_papers: List[Tuple[str, Dict[str, Any]]] = []
    arxiv_papers: List[Tuple[str, Dict[str, Any]]] = []
    s2_lock = threading.Lock()
    arxiv_lock = threading.Lock()

    # CrossRef, PubMed, DBLP results (populated by parallel search)
    crossref_papers: List[Tuple[str, Dict[str, Any]]] = []
    pubmed_papers: List[Tuple[str, Dict[str, Any]]] = []
    dblp_papers: List[Tuple[str, Dict[str, Any]]] = []
    crossref_lock = threading.Lock()
    pubmed_lock = threading.Lock()
    dblp_lock = threading.Lock()

    def search_semantic_scholar_and_store(q: str):
        """Execute Semantic Scholar search and store results thread-safely."""
        try:
            results = _search_semantic_scholar(q, SEMANTIC_SCHOLAR_LIMIT)
            if results:
                logger.info(f"Semantic Scholar: {len(results)} results for '{q[:50]}'")
                with s2_lock:
                    s2_papers.extend(results)
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")

    def search_s2_highly_cited_and_store(q: str):
        """Execute S2 highly-cited search (sorted by citations) and store results."""
        try:
            results = _search_s2_highly_cited(q, S2_HIGHLY_CITED_LIMIT)
            if results:
                logger.info(f"S2 highly-cited: {len(results)} results for '{q[:50]}'")
                with s2_lock:
                    s2_papers.extend(results)
        except Exception as e:
            logger.warning(f"S2 highly-cited search failed: {e}")

    def search_arxiv_and_store(q: str):
        """Execute ArXiv search and store results thread-safely."""
        try:
            results = _search_arxiv(q, ARXIV_LIMIT)
            if results:
                logger.info(f"ArXiv: {len(results)} results for '{q[:50]}'")
                with arxiv_lock:
                    arxiv_papers.extend(results)
        except Exception as e:
            logger.warning(f"ArXiv search failed: {e}")

    def search_crossref_and_store(q: str):
        """Execute CrossRef search and store results thread-safely."""
        try:
            results = _search_crossref(q, CROSSREF_LIMIT)
            if results:
                logger.info(f"CrossRef: {len(results)} results for '{q[:50]}'")
                with crossref_lock:
                    crossref_papers.extend(results)
        except Exception as e:
            logger.warning(f"CrossRef search failed: {e}")

    def search_pubmed_and_store(q: str):
        """Execute PubMed search and store results thread-safely."""
        try:
            results = _search_pubmed(q, PUBMED_LIMIT)
            if results:
                logger.info(f"PubMed: {len(results)} results for '{q[:50]}'")
                with pubmed_lock:
                    pubmed_papers.extend(results)
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")

    def search_dblp_and_store(q: str):
        """Execute DBLP search and store results thread-safely."""
        try:
            results = _search_dblp(q, DBLP_LIMIT)
            if results:
                logger.info(f"DBLP: {len(results)} results for '{q[:50]}'")
                with dblp_lock:
                    dblp_papers.extend(results)
        except Exception as e:
            logger.warning(f"DBLP search failed: {e}")

    # Execute all searches in parallel (limit to 10 concurrent requests)
    MAX_CONCURRENT_SEARCHES = 10
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SEARCHES) as executor:
        futures = [
            executor.submit(search_and_store, q, weight, source_concept, is_recent)
            for q, weight, source_concept, is_recent in search_tasks
        ]

        # Build list of queries to search on ArXiv and Semantic Scholar
        # Include original query + foundational works + expanded terms to catch seminal papers
        # that may use specific terminology (e.g., "DDPM" for diffusion models)
        arxiv_s2_queries = [query_text]  # Always include original query

        # Add foundational works and expanded terms
        # For drill-down (skip_external=True): 2 expanded = 3 total ArXiv calls
        # For normal rank: 4 expanded = 5 total ArXiv calls
        skip_external = rank_params_json.get("skip_external_sources", False)
        MAX_EXPANDED_QUERIES = 2 if skip_external else 4
        if MAX_EXPANDED_QUERIES > 0 and query_expansion.concepts:
            expanded_terms_with_importance = []
            for concept in query_expansion.concepts:
                # Add foundational works with highest importance (seminal paper names/titles)
                # These are critical for catching original papers in a field
                for fw in getattr(concept, 'foundational_works', []):
                    if fw.lower() != query_text.lower() and len(fw) > 2:
                        expanded_terms_with_importance.append((fw, concept.importance * 1.5))
                # Add synonyms with full importance (safe, direct alternatives)
                for syn in concept.synonyms:
                    if syn.lower() != query_text.lower() and len(syn) > 3:
                        expanded_terms_with_importance.append((syn, concept.importance))
                # Add high-importance related terms
                for rel in concept.related_terms:
                    if rel.lower() != query_text.lower() and len(rel) > 3:
                        expanded_terms_with_importance.append((rel, concept.importance * 0.7))

            # Sort by importance and take top N unique terms
            expanded_terms_with_importance.sort(key=lambda x: -x[1])
            seen_terms = {query_text.lower()}
            for term, _ in expanded_terms_with_importance:
                if term.lower() not in seen_terms and len(arxiv_s2_queries) <= MAX_EXPANDED_QUERIES:
                    arxiv_s2_queries.append(term)
                    seen_terms.add(term.lower())
        elif query_expansion.expansion_terms:
            # Fallback to legacy expansion - take top N terms
            for term in query_expansion.expansion_terms[:MAX_EXPANDED_QUERIES]:
                if term.lower() != query_text.lower() and len(term) > 3:
                    arxiv_s2_queries.append(term)

        logger.info(f"ArXiv/S2 search queries: {arxiv_s2_queries}")

        # Run Semantic Scholar and ArXiv searches for original + expanded queries (in parallel)
        # Each source has unique coverage - papers may only exist in one source
        # Skip for drill-down (skip_external=True) to reduce latency
        if not skip_external:
            if SEMANTIC_SCHOLAR_ENABLED:
                # Combine all expansion terms into ONE S2 bulk query (avoids rate limiting)
                s2_bulk_query = " ".join(arxiv_s2_queries)
                logger.info(f"S2 combined bulk query: {s2_bulk_query[:80]}...")
                futures.append(executor.submit(search_semantic_scholar_and_store, s2_bulk_query))

                # Build combined S2 highly-cited query from original + synonyms + foundational titles
                # Synonyms are key for finding foundational papers (e.g., "transformers" → "Attention Is All You Need")
                top_synonyms = []
                foundational_titles = []
                for concept in query_expansion.concepts:
                    for syn in concept.synonyms[:2]:
                        if syn and len(syn) > 2 and syn.lower() != query_text.lower():
                            top_synonyms.append(syn)
                    for fw in getattr(concept, 'foundational_works', []):
                        if fw and len(fw) > 5 and fw.lower() != query_text.lower():
                            foundational_titles.append(fw)

                # Combine: original + top synonyms + foundational titles
                highly_cited_terms = [query_text] + top_synonyms[:3] + foundational_titles[:2]
                s2_highly_cited_query = " ".join(highly_cited_terms)
                logger.info(f"S2 combined highly-cited query: {s2_highly_cited_query[:80]}...")
                futures.append(executor.submit(search_s2_highly_cited_and_store, s2_highly_cited_query))

                # Still search foundational works in OpenAlex/ArXiv (no rate limit issues)
                for fw_title in foundational_titles[:5]:
                    futures.append(executor.submit(search_and_store, fw_title, 1.0, "foundational_work", False))
                    if ARXIV_ENABLED:
                        futures.append(executor.submit(search_arxiv_and_store, fw_title))
            if ARXIV_ENABLED:
                # ArXiv can handle multiple queries (no rate limit issues like S2)
                for q in arxiv_s2_queries:
                    futures.append(executor.submit(search_arxiv_and_store, q))

        # Run CrossRef, PubMed, DBLP searches (original query only, parallel)
        # These provide additional coverage across different domains
        # Skip for lightweight operations (drill-down) to reduce latency
        skip_external = rank_params_json.get("skip_external_sources", False)
        if not skip_external:
            if CROSSREF_ENABLED:
                futures.append(executor.submit(search_crossref_and_store, query_text))
            if PUBMED_ENABLED:
                futures.append(executor.submit(search_pubmed_and_store, query_text))
            if DBLP_ENABLED:
                futures.append(executor.submit(search_dblp_and_store, query_text))
        else:
            logger.info("Skipping external sources (CrossRef, PubMed, DBLP) for lightweight operation")

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Search failed: {e}")

    # 2c) Ingest ArXiv papers FIRST (they get enriched with S2 data and canonical IDs)
    # This must happen before S2 ingestion so that S2 papers can link to existing ArXiv entries
    arxiv_title_map: Dict[str, str] = {}  # title -> work_id for cross-source deduplication
    arxiv_id_map: Dict[str, str] = {}  # arxiv_id -> work_id for cross-source deduplication
    doi_map: Dict[str, str] = {}  # doi -> work_id for cross-source deduplication
    if arxiv_papers:
        # Deduplicate by ArXiv ID
        seen_arxiv_ids = set()
        unique_arxiv_papers = []
        for arxiv_id, meta in arxiv_papers:
            if arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                unique_arxiv_papers.append((arxiv_id, meta))

        # Ingest into DB and get work_ids + dedup maps (DOI, ArXiv ID, title)
        arxiv_work_ids, arxiv_title_map, arxiv_id_map = _ingest_arxiv_papers(conn, unique_arxiv_papers)
        logger.info(f"Ingested {len(arxiv_work_ids)} papers from ArXiv (title map: {len(arxiv_title_map)}, arxiv_id map: {len(arxiv_id_map)})")

        # Add to candidate map with provenance
        for idx, (wid, (arxiv_id, meta)) in enumerate(zip(arxiv_work_ids, unique_arxiv_papers)):
            prov_entry = {
                "source": "arxiv",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),  # Rank-based score
                "arxiv_id": arxiv_id,
                "citations": meta.get("citations", 0),  # Include S2-enriched citation count
            }
            candidate_map.setdefault(wid, []).append(prov_entry)

    # 2d) Ingest Semantic Scholar papers AFTER ArXiv (can link to existing ArXiv entries)
    if s2_papers:
        # Deduplicate by S2 paper ID
        seen_s2_ids = set()
        unique_s2_papers = []
        for s2_id, meta in s2_papers:
            if s2_id not in seen_s2_ids:
                seen_s2_ids.add(s2_id)
                unique_s2_papers.append((s2_id, meta))

        # Ingest into DB and get work_ids + updated dedup maps
        # Pass existing maps to enable linking to freshly-inserted ArXiv papers
        s2_work_ids, arxiv_title_map, arxiv_id_map = _ingest_semantic_scholar_papers(
            conn, unique_s2_papers,
            arxiv_titles=arxiv_title_map,
            arxiv_ids=arxiv_id_map
        )
        logger.info(f"Ingested {len(s2_work_ids)} papers from Semantic Scholar (updated title map: {len(arxiv_title_map)}, arxiv_id map: {len(arxiv_id_map)})")

        # Add to candidate map with provenance (may merge with existing ArXiv entries)
        for idx, (wid, (s2_id, meta)) in enumerate(zip(s2_work_ids, unique_s2_papers)):
            prov_entry = {
                "source": "semantic_scholar",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),  # Rank-based score
                "citations": meta.get("citations", 0),
            }
            candidate_map.setdefault(wid, []).append(prov_entry)

    # 2d-ii) Ingest CrossRef papers
    if crossref_papers:
        seen_dois = set()
        unique_crossref = []
        for doi, meta in crossref_papers:
            if doi not in seen_dois:
                seen_dois.add(doi)
                unique_crossref.append((doi, meta))

        logger.info(f"Processing {len(unique_crossref)} unique CrossRef papers")
        for idx, (doi, meta) in enumerate(unique_crossref):
            title = meta.get("title", "")
            normalized_doi = normalize_doi_for_dedup(doi)

            # Multi-stage deduplication:
            # Stage 1: DOI match (primary)
            if normalized_doi and normalized_doi in doi_map:
                wid = doi_map[normalized_doi]
                logger.debug(f"CrossRef {doi[:30]} deduped to existing {wid} (DOI match)")
            # Stage 2: Title match (fallback)
            elif title and title.lower() in arxiv_title_map:
                wid = arxiv_title_map[title.lower()]
                logger.debug(f"CrossRef {doi[:30]} deduped to existing {wid} (title match)")
            # Stage 3: Create new work_id
            else:
                wid = f"DOI:{doi}"

            # Update dedup maps for subsequent sources
            if normalized_doi:
                doi_map[normalized_doi] = wid
            if title:
                arxiv_title_map[title.lower()] = wid

            prov_entry = {
                "source": "crossref",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),
                "citations": meta.get("citations", 0),
                "doi": doi,
            }
            candidate_map.setdefault(wid, []).append(prov_entry)

            # Upsert into works table with DOI
            try:
                conn.execute(
                    text("""
                        INSERT INTO works (work_id, title, year, cited_by_count, venue, doi)
                        VALUES (:wid, :title, :year, :citations, :venue, :doi)
                        ON CONFLICT (work_id) DO UPDATE SET
                            cited_by_count = GREATEST(COALESCE(works.cited_by_count, 0), COALESCE(EXCLUDED.cited_by_count, 0)),
                            title = COALESCE(works.title, EXCLUDED.title),
                            venue = COALESCE(works.venue, EXCLUDED.venue),
                            doi = COALESCE(works.doi, EXCLUDED.doi)
                    """),
                    {
                        "wid": wid,
                        "title": title,
                        "year": meta.get("year"),
                        "citations": meta.get("citations", 0),
                        "venue": meta.get("venue"),
                        "doi": normalized_doi,
                    },
                )
            except Exception as e:
                logger.debug(f"CrossRef upsert failed for {wid}: {e}")

    # 2d-iii) Ingest PubMed papers
    if pubmed_papers:
        seen_pmids = set()
        unique_pubmed = []
        for pmid, meta in pubmed_papers:
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_pubmed.append((pmid, meta))

        logger.info(f"Processing {len(unique_pubmed)} unique PubMed papers")
        for idx, (pmid, meta) in enumerate(unique_pubmed):
            title = meta.get("title", "")
            doi = meta.get("doi")
            normalized_doi = normalize_doi_for_dedup(doi) if doi else None

            # Multi-stage deduplication:
            # Stage 1: DOI match (primary)
            if normalized_doi and normalized_doi in doi_map:
                wid = doi_map[normalized_doi]
                logger.debug(f"PubMed {pmid} deduped to existing {wid} (DOI match)")
            # Stage 2: Title match (fallback)
            elif title and title.lower() in arxiv_title_map:
                wid = arxiv_title_map[title.lower()]
                logger.debug(f"PubMed {pmid} deduped to existing {wid} (title match)")
            # Stage 3: Create new work_id (prefer DOI over PMID)
            elif normalized_doi:
                wid = f"DOI:{doi}"
            else:
                wid = f"PMID:{pmid}"

            # Update dedup maps for subsequent sources
            if normalized_doi:
                doi_map[normalized_doi] = wid
            if title:
                arxiv_title_map[title.lower()] = wid

            prov_entry = {
                "source": "pubmed",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),
                "pmid": pmid,
            }
            candidate_map.setdefault(wid, []).append(prov_entry)

            # Upsert into works table with DOI
            try:
                conn.execute(
                    text("""
                        INSERT INTO works (work_id, title, year, venue, doi)
                        VALUES (:wid, :title, :year, :venue, :doi)
                        ON CONFLICT (work_id) DO UPDATE SET
                            title = COALESCE(works.title, EXCLUDED.title),
                            venue = COALESCE(works.venue, EXCLUDED.venue),
                            doi = COALESCE(works.doi, EXCLUDED.doi)
                    """),
                    {
                        "wid": wid,
                        "title": title,
                        "year": meta.get("year"),
                        "venue": meta.get("venue"),
                        "doi": normalized_doi,
                    },
                )
            except Exception as e:
                logger.debug(f"PubMed upsert failed for {wid}: {e}")

    # 2d-iv) Ingest DBLP papers
    if dblp_papers:
        seen_keys = set()
        unique_dblp = []
        for dblp_key, meta in dblp_papers:
            if dblp_key not in seen_keys:
                seen_keys.add(dblp_key)
                unique_dblp.append((dblp_key, meta))

        logger.info(f"Processing {len(unique_dblp)} unique DBLP papers")
        for idx, (dblp_key, meta) in enumerate(unique_dblp):
            title = meta.get("title", "")
            doi = meta.get("doi")
            # Check for existing paper by title or DOI
            if title and title.lower() in arxiv_title_map:
                wid = arxiv_title_map[title.lower()]
            elif doi:
                wid = f"DOI:{doi}"
            else:
                wid = f"DBLP:{dblp_key.replace('/', '_')}"

            if title:
                arxiv_title_map[title.lower()] = wid

            prov_entry = {
                "source": "dblp",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),
                "venue": meta.get("venue"),
            }
            candidate_map.setdefault(wid, []).append(prov_entry)

            # Upsert into works table
            try:
                conn.execute(
                    text("""
                        INSERT INTO works (work_id, title, year, venue)
                        VALUES (:wid, :title, :year, :venue)
                        ON CONFLICT (work_id) DO UPDATE SET
                            title = COALESCE(works.title, EXCLUDED.title),
                            venue = COALESCE(works.venue, EXCLUDED.venue)
                    """),
                    {
                        "wid": wid,
                        "title": title,
                        "year": meta.get("year"),
                        "venue": meta.get("venue"),
                    },
                )
            except Exception as e:
                logger.debug(f"DBLP upsert failed for {wid}: {e}")

    # 2e) Cross-source deduplication: Add OpenAlex results with multi-stage dedup
    # OpenAlex results were stored temporarily; now we check each against prior sources
    # Priority: DOI match → ArXiv ID match → Title match
    if openalex_results:
        # Collect unique OpenAlex work_ids to fetch metadata from DB
        oa_work_ids = list(set(wid for wid, _, _ in openalex_results))

        # Fetch titles, DOI, and arxiv_id from DB for deduplication (batch query)
        oa_metadata: Dict[str, Dict[str, Optional[str]]] = {}  # work_id -> {title, doi, arxiv_id}
        if oa_work_ids:
            try:
                rows = conn.execute(
                    text("SELECT work_id, title, doi, arxiv_id FROM works WHERE work_id = ANY(:ids)"),
                    {"ids": oa_work_ids},
                ).fetchall()
                for row in rows:
                    oa_metadata[row[0]] = {
                        "title": row[1],
                        "doi": row[2],
                        "arxiv_id": row[3],
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch OpenAlex metadata for dedup: {e}")

        # If metadata not in DB yet, we need to fetch from OpenAlex API
        # This happens for papers not yet ingested
        missing_wids = [wid for wid in oa_work_ids if wid not in oa_metadata]
        if missing_wids:
            # WorkStore.ensure_works_present will fetch these - do it now for dedup
            WorkStore.ensure_works_present(conn, missing_wids)
            # Re-fetch metadata
            try:
                rows = conn.execute(
                    text("SELECT work_id, title, doi, arxiv_id FROM works WHERE work_id = ANY(:ids)"),
                    {"ids": missing_wids},
                ).fetchall()
                for row in rows:
                    oa_metadata[row[0]] = {
                        "title": row[1],
                        "doi": row[2],
                        "arxiv_id": row[3],
                    }
            except Exception:
                pass

        # Now add OpenAlex results with multi-stage deduplication
        dedup_count = 0
        dedup_by_type = {"doi": 0, "arxiv_id": 0, "title": 0}
        for oa_wid, _, prov_entry in openalex_results:
            meta = oa_metadata.get(oa_wid, {})
            oa_title = meta.get("title")
            oa_doi = meta.get("doi")
            oa_arxiv = meta.get("arxiv_id")

            # Normalize external IDs
            normalized_doi = normalize_doi_for_dedup(oa_doi) if oa_doi else None
            normalized_arxiv = normalize_arxiv_id(oa_arxiv) if oa_arxiv else None

            canonical_wid = None

            # Stage 1: DOI match (primary)
            if normalized_doi and normalized_doi in doi_map:
                canonical_wid = doi_map[normalized_doi]
                if canonical_wid != oa_wid:
                    dedup_count += 1
                    dedup_by_type["doi"] += 1
                    logger.debug(f"Dedup: OpenAlex {oa_wid} → {canonical_wid} (DOI match)")

            # Stage 2: ArXiv ID match
            elif normalized_arxiv and normalized_arxiv in arxiv_id_map:
                canonical_wid = arxiv_id_map[normalized_arxiv]
                if canonical_wid != oa_wid:
                    dedup_count += 1
                    dedup_by_type["arxiv_id"] += 1
                    logger.debug(f"Dedup: OpenAlex {oa_wid} → {canonical_wid} (ArXiv ID match)")

            # Stage 3: Title match (fallback)
            elif oa_title and oa_title.lower() in arxiv_title_map:
                canonical_wid = arxiv_title_map[oa_title.lower()]
                if canonical_wid != oa_wid:
                    dedup_count += 1
                    dedup_by_type["title"] += 1
                    logger.debug(f"Dedup: OpenAlex {oa_wid} → {canonical_wid} (title match)")

            # Use canonical work_id if dedup match found, otherwise use OpenAlex work_id
            final_wid = canonical_wid if canonical_wid else oa_wid
            candidate_map.setdefault(final_wid, []).append(prov_entry)

            # Update dedup maps with OpenAlex papers
            if normalized_doi and not canonical_wid:
                doi_map[normalized_doi] = oa_wid
            if normalized_arxiv and not canonical_wid:
                arxiv_id_map[normalized_arxiv] = oa_wid
            if oa_title and not canonical_wid:
                arxiv_title_map[oa_title.lower()] = oa_wid

        if dedup_count > 0:
            logger.info(f"Cross-source dedup: {dedup_count} OpenAlex papers linked (DOI: {dedup_by_type['doi']}, ArXiv: {dedup_by_type['arxiv_id']}, Title: {dedup_by_type['title']})")

    # 2g) Highly-cited papers retrieval - ensures foundational papers are included
    # This retrieves the most-cited papers for the query, sorted by citation count
    # Search both original query AND key expanded terms to catch seminal works with specific terminology
    highly_cited_queries = [query_text]

    # Also search for review/survey papers explicitly - this ensures core_concepts has enough reviews
    # Many highly-cited reviews are only found when searching for "[topic] review/survey"
    if query_text and len(query_text.split()) <= 6:  # Only for short queries
        # Add multiple review-related search variants
        for review_term in ["review", "survey", "overview", "systematic review", "meta-analysis"]:
            review_query = f"{query_text} {review_term}"
            highly_cited_queries.append(review_query)

    # Collect foundational work titles separately - they need title.search filter
    # because they are exact paper titles like "Attention Is All You Need"
    foundational_work_titles: List[str] = []

    # Add foundational works and synonyms to highly-cited search
    # This ensures seminal papers are found via OpenAlex even when ArXiv/S2 fail
    if query_expansion.concepts:
        for concept in query_expansion.concepts:
            # Foundational works are PAPER TITLES - use title search for exact matching
            for fw in getattr(concept, 'foundational_works', []):
                if fw.lower() != query_text.lower() and len(fw) > 3:
                    foundational_work_titles.append(fw)
            # Add top synonyms as they represent direct terminology alternatives
            for syn in concept.synonyms[:1]:
                if syn.lower() != query_text.lower() and len(syn) > 3:
                    highly_cited_queries.append(syn)

    # Dedupe foundational work titles
    seen_fw = set()
    unique_fw_titles = []
    for fw in foundational_work_titles:
        fw_lower = fw.lower()
        if fw_lower not in seen_fw:
            seen_fw.add(fw_lower)
            unique_fw_titles.append(fw)
    # Limit to top 10 unique foundational works to avoid excessive API calls
    unique_fw_titles = unique_fw_titles[:10]

    highly_cited_added = 0
    highly_cited_deduped = 0

    # Helper function to add highly-cited paper with multi-stage deduplication
    def add_highly_cited_paper(wid: str, meta: Optional[Dict[str, Optional[str]]], prov_entry: Dict) -> bool:
        """Add a highly-cited paper with cross-source deduplication (DOI → ArXiv ID → Title). Returns True if added as new."""
        nonlocal highly_cited_added, highly_cited_deduped

        title = meta.get("title") if meta else None
        doi = meta.get("doi") if meta else None
        arxiv_id = meta.get("arxiv_id") if meta else None

        # Normalize external IDs
        normalized_doi = normalize_doi_for_dedup(doi) if doi else None
        normalized_arxiv = normalize_arxiv_id(arxiv_id) if arxiv_id else None

        canonical_wid = None

        # Stage 1: DOI match (primary)
        if normalized_doi and normalized_doi in doi_map:
            canonical_wid = doi_map[normalized_doi]
            if canonical_wid != wid:
                highly_cited_deduped += 1

        # Stage 2: ArXiv ID match
        elif normalized_arxiv and normalized_arxiv in arxiv_id_map:
            canonical_wid = arxiv_id_map[normalized_arxiv]
            if canonical_wid != wid:
                highly_cited_deduped += 1

        # Stage 3: Title match (fallback)
        elif title and title.lower() in arxiv_title_map:
            canonical_wid = arxiv_title_map[title.lower()]
            if canonical_wid != wid:
                highly_cited_deduped += 1

        # Use canonical work_id if dedup match found, otherwise use original work_id
        final_wid = canonical_wid if canonical_wid else wid

        if final_wid not in candidate_map:
            candidate_map.setdefault(final_wid, []).append(prov_entry)
            highly_cited_added += 1
            return True
        else:
            candidate_map[final_wid].append(prov_entry)
            return False

    # First: Search for foundational works using title.search filter
    # This is CRITICAL for finding THE papers like "Attention Is All You Need"
    for fw_title in unique_fw_titles:
        title_results = _search_openalex_by_title(fw_title, k=5)
        for idx, (wid, citations) in enumerate(title_results):
            prov_entry = {
                "source": "foundational_title",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),
                "citations": int(citations),
                "query": fw_title,
            }
            # Use the foundational work title itself for dedup (it's the paper title)
            add_highly_cited_paper(wid, {"title": fw_title}, prov_entry)

    # Second: Regular highly-cited search for original query and synonyms
    # For these, we need to fetch titles from DB for deduplication
    hc_work_ids_to_fetch: List[str] = []
    hc_results_pending: List[Tuple[str, int, str, Dict]] = []  # (wid, idx, query, prov)

    for hc_query in highly_cited_queries:
        highly_cited_results = _search_openalex_highly_cited(hc_query, OPENALEX_HIGHLY_CITED_LIMIT)
        for idx, (wid, citations) in enumerate(highly_cited_results):
            prov_entry = {
                "source": "highly_cited",
                "rank": idx + 1,
                "score": 1.0 / (idx + 1),
                "citations": int(citations),
                "query": hc_query,
            }
            hc_work_ids_to_fetch.append(wid)
            hc_results_pending.append((wid, idx, hc_query, prov_entry))

    # Fetch metadata (title, DOI, arxiv_id) for deduplication
    hc_metadata: Dict[str, Dict[str, Optional[str]]] = {}
    if hc_work_ids_to_fetch:
        WorkStore.ensure_works_present(conn, list(set(hc_work_ids_to_fetch)))
        try:
            rows = conn.execute(
                text("SELECT work_id, title, doi, arxiv_id FROM works WHERE work_id = ANY(:ids)"),
                {"ids": list(set(hc_work_ids_to_fetch))},
            ).fetchall()
            for row in rows:
                hc_metadata[row[0]] = {
                    "title": row[1],
                    "doi": row[2],
                    "arxiv_id": row[3],
                }
        except Exception:
            pass

    # Now add highly-cited results with deduplication
    for wid, idx, hc_query, prov_entry in hc_results_pending:
        meta = hc_metadata.get(wid)
        add_highly_cited_paper(wid, meta, prov_entry)

    logger.info(
        f"Added {highly_cited_added} new papers from highly-cited + title searches "
        f"({len(highly_cited_queries)} + {len(unique_fw_titles)} queries, {highly_cited_deduped} deduped)"
    )

    # 2h) Topic backstop retrieval (optional, for additional coverage)
    target_topic_id = context_json.get("target_topic_id")
    if target_topic_id:
        topic_results = _retrieve_topic_pool(conn, str(target_topic_id), k_topic)
        for idx, (wid, score) in enumerate(topic_results):
            if wid not in candidate_map:
                prov_entry = {"source": "topic_pool", "rank": idx + 1, "score": score, "topic_id": target_topic_id}
                candidate_map.setdefault(wid, []).append(prov_entry)

    # Ensure works exist in the DB
    all_wids = list(candidate_map.keys())
    logger.info(f"Total unique candidates: {len(all_wids)}")
    WorkStore.ensure_works_present(conn, all_wids)

    # Validate and correct suspicious metadata (future years, wrong papers, etc.)
    # This is a GENERAL fix that runs for ALL queries, not specific papers
    _validate_and_correct_metadata(conn, all_wids)

    # Apply year filters
    year_min = filters_json.get("year_min")
    year_max = filters_json.get("year_max")
    if year_min is not None or year_max is not None:
        loaded = WorkStore.load_many(conn, all_wids)
        to_remove: List[str] = []
        for wid in all_wids:
            w = loaded.get(wid)
            if not w:
                continue
            if w.is_retracted:
                to_remove.append(wid)
                continue
            y = w.year
            if year_min is not None and (y is None or int(y) < int(year_min)):
                to_remove.append(wid)
                continue
            if year_max is not None and (y is None or int(y) > int(year_max)):
                to_remove.append(wid)
                continue
        for wid in to_remove:
            candidate_map.pop(wid, None)

    # Cap pool size by aggregate provenance score (sum of all source scores)
    # This ensures highly-cited foundational papers are kept regardless of work_id prefix
    def compute_pool_score(wid: str) -> float:
        prov_list = candidate_map.get(wid, [])
        # Sum of all provenance scores
        total_score = sum(p.get("score", 0) for p in prov_list)
        # Bonus for citation count (if available)
        max_cites = max((p.get("citations", 0) for p in prov_list), default=0)
        # Normalize citations to a 0-1 scale (log scale, capped at 100k)
        import math
        cite_bonus = math.log10(max_cites + 1) / 5.0 if max_cites > 0 else 0
        return total_score + cite_bonus

    scored_ids = [(wid, compute_pool_score(wid)) for wid in candidate_map.keys()]
    scored_ids.sort(key=lambda x: -x[1])  # Sort by score descending
    trimmed_ids = [wid for wid, _ in scored_ids[:max_pool]]

    out: List[Dict[str, Any]] = []
    for wid in trimmed_ids:
        out.append({"work_id": wid, "provenance": candidate_map[wid]})

    return out, query_expansion


# Legacy function for backward compatibility
def generate_candidates_direct_legacy(
    conn: Connection,
    *,
    tenant_id: Any,
    query_text: str,
    context_json: Optional[Dict[str, Any]] = None,
    filters_json: Optional[Dict[str, Any]] = None,
    rank_params_json: Optional[Dict[str, Any]] = None,
    limit_pool: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Legacy candidate generation (OpenAlex lexical + topic backstop).

    This function is retained for backward compatibility. For new code,
    use generate_candidates_direct() which returns QueryExpansion.
    """
    context_json = context_json or {}
    filters_json = filters_json or {}
    rank_params_json = rank_params_json or {}

    k_lex = int(rank_params_json.get("k_lexical", DEFAULT_K_LEXICAL))
    k_topic = int(rank_params_json.get("k_topic", DEFAULT_K_TOPIC))
    max_pool = int(limit_pool or rank_params_json.get("limit_pool", DEFAULT_LIMIT_POOL))

    if k_lex < 0:
        k_lex = DEFAULT_K_LEXICAL
    if k_topic < 0:
        k_topic = DEFAULT_K_TOPIC

    candidate_map: Dict[str, List[Dict[str, Any]]] = {}

    # Lexical retrieval from OpenAlex
    lex_results = _search_openalex(query_text, k_lex)
    logger.info(f"Legacy lexical retrieval: {len(lex_results)} results")
    for idx, (wid, score) in enumerate(lex_results):
        prov_entry = {"source": "lexical_openalex", "rank": idx + 1, "score": score}
        candidate_map.setdefault(wid, []).append(prov_entry)

    # Topic backstop
    target_topic_id = context_json.get("target_topic_id")
    if target_topic_id:
        topic_results = _retrieve_topic_pool(conn, str(target_topic_id), k_topic)
        for idx, (wid, score) in enumerate(topic_results):
            prov_entry = {"source": "topic_pool", "rank": idx + 1, "score": score, "topic_id": target_topic_id}
            candidate_map.setdefault(wid, []).append(prov_entry)

    all_wids = list(candidate_map.keys())
    WorkStore.ensure_works_present(conn, all_wids)

    # Apply year filters
    year_min = filters_json.get("year_min")
    year_max = filters_json.get("year_max")
    if year_min is not None or year_max is not None:
        loaded = WorkStore.load_many(conn, all_wids)
        to_remove: List[str] = []
        for wid in all_wids:
            w = loaded.get(wid)
            if not w:
                continue
            if w.is_retracted:
                to_remove.append(wid)
                continue
            y = w.year
            if year_min is not None and (y is None or int(y) < int(year_min)):
                to_remove.append(wid)
                continue
            if year_max is not None and (y is None or int(y) > int(year_max)):
                to_remove.append(wid)
                continue
        for wid in to_remove:
            candidate_map.pop(wid, None)

    sorted_ids = sorted(candidate_map.keys())[:max_pool]
    return [{"work_id": wid, "provenance": candidate_map[wid]} for wid in sorted_ids]
