"""
Citation Map Service for Feature 1.

Two input modes:
1. Seed Paper Mode: Given a work_id, find top-k connections
2. NL Query Mode: Find the most influential seed paper from query, then expand

Search logic is copied from feature2/retrieval.py to keep this service self-contained.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from uuid import UUID, uuid4

import requests
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature1.schemas import (
    CitationEdge,
    CitationMapRequest,
    CitationMapResponse,
    CitationMapStats,
    CitationNode,
    SeedSelectionInfo,
)

# Ensure .env is loaded
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# API keys
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

# Search limits
OPENALEX_LIMIT = 100
OPENALEX_HIGHLY_CITED_LIMIT = 50
S2_LIMIT = 100
ARXIV_LIMIT = 100

# Rate limiting
SEMANTIC_SCHOLAR_DELAY = 1.0
OPENALEX_TIMEOUT = 15


# ============================================================================
# OpenAlex Search (copied from retrieval.py)
# ============================================================================

def _search_openalex(query: str, k: int = OPENALEX_LIMIT) -> List[Dict[str, Any]]:
    """Search OpenAlex for papers by query.

    Returns list of dicts with: work_id, title, year, cited_by_count
    """
    if not query:
        return []

    params = {
        "search": query,
        "per-page": min(k, 200),
        "select": "id,title,publication_year,cited_by_count",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    url = "https://api.openalex.org/works"

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            out = []
            for r in results:
                oa_id = r.get("id", "")
                if oa_id.startswith("https://openalex.org/"):
                    work_id = oa_id.replace("https://openalex.org/", "")
                    out.append({
                        "work_id": work_id,
                        "title": r.get("title"),
                        "year": r.get("publication_year"),
                        "cited_by_count": r.get("cited_by_count") or 0,
                        "source": "openalex",
                    })

            logger.info(f"OpenAlex search: {len(out)} papers for '{query[:30]}...'")
            return out

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"OpenAlex search HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenAlex search error: {e}")
            return []

    return []


def _search_openalex_highly_cited(query: str, k: int = OPENALEX_HIGHLY_CITED_LIMIT) -> List[Dict[str, Any]]:
    """Search OpenAlex for highly-cited papers sorted by citation count.

    Returns list of dicts with: work_id, title, year, cited_by_count
    """
    if not query:
        return []

    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "sort": "cited_by_count:desc",
        "per-page": min(k, 200),
        "select": "id,title,publication_year,cited_by_count",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            out = []
            for r in results:
                oa_id = r.get("id", "")
                if oa_id.startswith("https://openalex.org/"):
                    work_id = oa_id.replace("https://openalex.org/", "")
                    out.append({
                        "work_id": work_id,
                        "title": r.get("title"),
                        "year": r.get("publication_year"),
                        "cited_by_count": r.get("cited_by_count") or 0,
                        "source": "openalex_highly_cited",
                    })

            logger.info(f"OpenAlex highly-cited: {len(out)} papers for '{query[:30]}...'")
            return out

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"OpenAlex highly-cited HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"OpenAlex highly-cited error: {e}")
            return []

    return []


def _search_openalex_by_title(
    title_query: str,
    k: int = 10,
    year: Optional[int] = None,
    year_range: int = 2,
) -> List[Dict[str, Any]]:
    """Search OpenAlex for papers matching a title query.

    Uses title.search filter sorted by citation count.
    Also tries without year filter in case OpenAlex has wrong year data.
    """
    if not title_query:
        return []

    url = "https://api.openalex.org/works"
    all_results: Dict[str, Dict[str, Any]] = {}

    def do_search(filter_str: str) -> None:
        params = {
            "filter": filter_str,
            "sort": "cited_by_count:desc",
            "per-page": min(k * 2, 50),
            "select": "id,title,publication_year,cited_by_count",
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        try:
            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            resp.raise_for_status()
            for r in resp.json().get("results", []):
                oa_id = r.get("id", "")
                if oa_id.startswith("https://openalex.org/"):
                    work_id = oa_id.replace("https://openalex.org/", "")
                    if work_id not in all_results:
                        all_results[work_id] = {
                            "work_id": work_id,
                            "title": r.get("title"),
                            "year": r.get("publication_year"),
                            "cited_by_count": r.get("cited_by_count") or 0,
                            "source": "openalex_title",
                        }
        except Exception as e:
            logger.warning(f"OpenAlex title search error: {e}")

    # Search 1: Title search without year (catches papers with wrong year in OpenAlex)
    do_search(f'title.search:"{title_query}"')

    # Search 2: Title search with year filter if provided
    if year:
        year_filter = f",publication_year:{year - year_range}-{year + year_range}"
        do_search(f'title.search:"{title_query}"{year_filter}')

    # Sort by citation count
    out = list(all_results.values())
    out.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)

    logger.info(f"OpenAlex title search: {len(out)} papers for '{title_query[:30]}...'")
    return out[:k]


# ============================================================================
# Semantic Scholar Search (copied from retrieval.py)
# ============================================================================

def _search_semantic_scholar(query: str, k: int = S2_LIMIT) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for papers.

    Uses bulk endpoint with pagination.
    Returns list of dicts with: work_id, title, year, cited_by_count
    """
    if not query:
        return []

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    # Try bulk endpoint
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    out = []
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
                    return _search_s2_regular(query, min(k, 100), headers)

                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                continuation_token = data.get("token")

                if not results:
                    logger.info(f"S2 bulk search: {len(out)} papers for '{query[:30]}...'")
                    return out

                for paper in results:
                    paper_id = paper.get("paperId")
                    if not paper_id or paper_id in seen_ids:
                        continue

                    seen_ids.add(paper_id)
                    external_ids = paper.get("externalIds") or {}
                    openalex_id = external_ids.get("OpenAlex")

                    # Use OpenAlex ID if available, otherwise S2 ID
                    work_id = openalex_id if openalex_id else f"S2:{paper_id}"

                    out.append({
                        "work_id": work_id,
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "cited_by_count": paper.get("citationCount") or 0,
                        "source": "semantic_scholar",
                    })

                    if len(out) >= k:
                        logger.info(f"S2 bulk search: {len(out)} papers for '{query[:30]}...'")
                        return out

                if not continuation_token:
                    logger.info(f"S2 bulk search: {len(out)} papers for '{query[:30]}...'")
                    return out

                time.sleep(SEMANTIC_SCHOLAR_DELAY)
                break

            except requests.exceptions.RequestException as e:
                is_rate_limit = "429" in str(e)
                if attempt < MAX_RETRIES - 1:
                    backoff = (2 ** (attempt + 1)) if is_rate_limit else RETRY_BACKOFF_BASE * (2 ** attempt)
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


def _search_s2_regular(query: str, k: int, headers: dict) -> List[Dict[str, Any]]:
    """Fallback: Search S2 using regular endpoint with offset pagination."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    out = []
    seen_ids = set()
    offset = 0
    page_size = 100

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

                if not results:
                    return out

                for paper in results:
                    paper_id = paper.get("paperId")
                    if not paper_id or paper_id in seen_ids:
                        continue

                    seen_ids.add(paper_id)
                    external_ids = paper.get("externalIds") or {}
                    openalex_id = external_ids.get("OpenAlex")
                    work_id = openalex_id if openalex_id else f"S2:{paper_id}"

                    out.append({
                        "work_id": work_id,
                        "title": paper.get("title", ""),
                        "year": paper.get("year"),
                        "cited_by_count": paper.get("citationCount") or 0,
                        "source": "semantic_scholar",
                    })

                    if len(out) >= k:
                        return out

                offset += page_size
                time.sleep(SEMANTIC_SCHOLAR_DELAY)
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                logger.warning(f"S2 regular search failed: {e}")
                return out
        else:
            break

    return out


# ============================================================================
# ArXiv Search (copied from retrieval.py)
# ============================================================================

def _search_arxiv(query: str, k: int = ARXIV_LIMIT) -> List[Dict[str, Any]]:
    """Search ArXiv for papers.

    Returns list of dicts with: work_id, title, year, cited_by_count (always 0)
    """
    if not query:
        return []

    base_url = "https://export.arxiv.org/api/query"
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

            out = []
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

                out.append({
                    "work_id": f"AX:{arxiv_id}",
                    "title": title,
                    "year": year,
                    "cited_by_count": 0,  # ArXiv doesn't provide citation counts
                    "source": "arxiv",
                })

            logger.info(f"ArXiv search: {len(out)} papers for '{query[:30]}...'")
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


# ============================================================================
# DOI Bridge: OpenAlex <-> Semantic Scholar ID Mapping
# ============================================================================

def _get_doi_for_work(work_id: str) -> Optional[str]:
    """Get DOI for an OpenAlex work_id."""
    if not work_id or not work_id.startswith("W"):
        return None

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        params = {"select": "doi"}
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        doi = data.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            return doi.replace("https://doi.org/", "")
        return doi
    except Exception as e:
        logger.warning(f"Failed to get DOI for {work_id}: {e}")
        return None


def _lookup_s2_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Lookup a paper on Semantic Scholar by DOI."""
    if not doi:
        return None

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "paperId,title,year,citationCount,externalIds"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 404:
            logger.debug(f"S2 paper not found for DOI:{doi}")
            return None

        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to lookup S2 paper by DOI {doi}: {e}")
        return None


def _lookup_openalex_by_doi(doi: str) -> Optional[str]:
    """Lookup an OpenAlex work_id by DOI."""
    if not doi:
        return None

    try:
        url = "https://api.openalex.org/works"
        # Ensure DOI is clean (no https://doi.org/ prefix)
        clean_doi = doi.replace("https://doi.org/", "")
        params = {
            "filter": f"doi:{clean_doi}",
            "select": "id",
            "per-page": 1,
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if results:
            oa_id = results[0].get("id", "")
            if oa_id.startswith("https://openalex.org/"):
                return oa_id.replace("https://openalex.org/", "")
        return None
    except Exception as e:
        logger.warning(f"Failed to lookup OpenAlex by DOI {doi}: {e}")
        return None


def _batch_lookup_openalex_by_dois(dois: List[str]) -> Dict[str, str]:
    """Batch lookup OpenAlex work_ids by DOIs.

    Returns dict mapping DOI -> OpenAlex work_id.
    """
    if not dois:
        return {}

    # OpenAlex supports OR filters - batch lookup up to 50 at a time
    result = {}
    batch_size = 50

    for i in range(0, len(dois), batch_size):
        batch = dois[i:i + batch_size]
        clean_dois = [d.replace("https://doi.org/", "") for d in batch if d]
        if not clean_dois:
            continue

        try:
            # Use OR filter for batch lookup
            doi_filter = "|".join(clean_dois)
            url = "https://api.openalex.org/works"
            params = {
                "filter": f"doi:{doi_filter}",
                "select": "id,doi",
                "per-page": len(clean_dois),
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
            resp.raise_for_status()
            works = resp.json().get("results", [])

            for w in works:
                oa_id = w.get("id", "")
                doi = w.get("doi", "")
                if oa_id and doi:
                    # Normalize both
                    work_id = oa_id.replace("https://openalex.org/", "")
                    clean_doi = doi.replace("https://doi.org/", "")
                    result[clean_doi] = work_id

        except Exception as e:
            logger.warning(f"Batch DOI lookup failed: {e}")

    return result


# ============================================================================
# Semantic Scholar Citation Fetching
# ============================================================================

def _fetch_citing_papers_s2(doi: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch papers that cite a given work from Semantic Scholar using DOI.

    Returns list of dicts with: doi, s2_id, title, year, cited_by_count
    """
    if not doi:
        return []

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"
        params = {
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": min(limit, 1000),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 404:
            logger.debug(f"S2 citations not found for DOI:{doi}")
            return []

        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, dict):
            return []

        result = []
        for item in data.get("data") or []:
            if not item or not isinstance(item, dict):
                continue

            citing = item.get("citingPaper")
            # Skip if citingPaper is None or empty
            if not citing or not isinstance(citing, dict):
                continue

            external_ids = citing.get("externalIds") or {}
            citing_doi = external_ids.get("DOI")

            result.append({
                "doi": citing_doi,
                "s2_id": citing.get("paperId"),
                "title": citing.get("title"),
                "year": citing.get("year"),
                "cited_by_count": citing.get("citationCount") or 0,
            })

        logger.info(f"S2 citations: {len(result)} papers for DOI:{doi}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch S2 citations for DOI {doi}: {e}")
        return []


def _fetch_references_s2(doi: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch papers that a given work cites from Semantic Scholar using DOI.

    Returns list of dicts with: doi, s2_id, title, year, cited_by_count
    """
    if not doi:
        return []

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/references"
        params = {
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": min(limit, 1000),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 404:
            logger.debug(f"S2 references not found for DOI:{doi}")
            return []

        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, dict):
            return []

        result = []
        for item in data.get("data") or []:
            if not item or not isinstance(item, dict):
                continue

            ref = item.get("citedPaper")
            # Skip if citedPaper is None or empty
            if not ref or not isinstance(ref, dict):
                continue

            external_ids = ref.get("externalIds") or {}
            ref_doi = external_ids.get("DOI")

            result.append({
                "doi": ref_doi,
                "s2_id": ref.get("paperId"),
                "title": ref.get("title"),
                "year": ref.get("year"),
                "cited_by_count": ref.get("citationCount") or 0,
            })

        logger.info(f"S2 references: {len(result)} papers for DOI:{doi}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch S2 references for DOI {doi}: {e}")
        return []


# ============================================================================
# Citation Fetching (adapted from paper_cache.py)
# ============================================================================

def _backfill_title_from_s2(doi: Optional[str]) -> Optional[str]:
    """Fetch title from Semantic Scholar when OpenAlex has empty title."""
    if not doi:
        return None

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        clean_doi = doi.replace("https://doi.org/", "")
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}"
        params = {"fields": "title"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            title = resp.json().get("title")
            if title:
                logger.debug(f"Backfilled title from S2 for DOI:{clean_doi}")
                return title
    except Exception as e:
        logger.debug(f"S2 title backfill failed for {doi}: {e}")

    return None


def _is_citation_count_suspicious(cited_by_count: int, year: Optional[int], title: Optional[str]) -> bool:
    """Detect papers with suspiciously corrupted citation counts in OpenAlex.

    Known patterns of data corruption:
    - Recent papers (2018+) with >30K citations (not enough time to accumulate)
    - Papers with titles suggesting non-academic content with high cites
    - Known corrupted entries (health supplement, HISTORIAE, etc.)
    """
    if not cited_by_count or cited_by_count < 10000:
        return False  # Low citation counts are fine

    # Recent papers can't have mega citations
    current_year = 2026
    if year and year >= 2018 and cited_by_count > 30000:
        return True

    # Known corrupted titles
    if title:
        title_lower = title.lower()
        suspicious_patterns = [
            "health supplement",
            "historiae",
            "trustworthy",
            "dynamic generation",
        ]
        if any(pat in title_lower for pat in suspicious_patterns):
            return True

    return False


def _extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    """Extract abstract from OpenAlex work data."""
    abstract_inv = work.get("abstract_inverted_index")
    if not abstract_inv:
        return None

    try:
        word_positions = []
        for word, positions in abstract_inv.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(word for _, word in word_positions)
    except Exception:
        return None


def fetch_citing_papers(work_id: str, limit: int = 25, fetch_limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch papers that cite a given work from OpenAlex.

    Fetches more than needed (fetch_limit) to allow for filtering.
    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    """
    if not work_id:
        return []

    try:
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"cites:{work_id}",
            "sort": "cited_by_count:desc",
            "per-page": fetch_limit,
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

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
            title = w.get("title")
            year = w.get("publication_year")
            cited_by_count = w.get("cited_by_count") or 0

            # Backfill empty titles from S2
            if not title:
                doi = w.get("doi")
                title = _backfill_title_from_s2(doi)

            # Skip papers with suspiciously corrupted citation counts
            if _is_citation_count_suspicious(cited_by_count, year, title):
                logger.debug(f"Skipping suspicious paper: {title[:50]} ({cited_by_count:,} cites)")
                continue

            result.append({
                "work_id": wid,
                "title": title,
                "year": year,
                "cited_by_count": cited_by_count,
                "abstract": _extract_abstract(w),
            })

        logger.info(f"Fetched {len(result)} citing papers for {work_id}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch citing papers for {work_id}: {e}")
        return []


def fetch_references(work_id: str, limit: int = 25, fetch_limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch papers that a given work cites (its references) from OpenAlex.

    Fetches more than needed to allow for filtering.
    Returns list of dicts with: work_id, title, year, cited_by_count, abstract
    Sorted by citation count (most influential references first).
    """
    if not work_id:
        return []

    try:
        # First get the work to find its referenced_works
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        work_data = resp.json()
        referenced_works = work_data.get("referenced_works", [])

        if not referenced_works:
            return []

        # Extract work IDs - get more than limit for filtering
        ref_ids = []
        for ref_url in referenced_works[:fetch_limit]:
            if "/" in ref_url:
                ref_ids.append(ref_url.rsplit("/", 1)[-1])

        if not ref_ids:
            return []

        # Fetch details for referenced works
        filter_str = "|".join(ref_ids[:100])
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"openalex_id:{filter_str}",
            "per-page": 100,
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

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
            title = w.get("title")
            year = w.get("publication_year")
            cited_by_count = w.get("cited_by_count") or 0

            # Backfill empty titles from S2
            if not title:
                doi = w.get("doi")
                title = _backfill_title_from_s2(doi)

            # Skip papers with suspiciously corrupted citation counts
            if _is_citation_count_suspicious(cited_by_count, year, title):
                logger.debug(f"Skipping suspicious reference: {title[:50] if title else 'N/A'} ({cited_by_count:,} cites)")
                continue

            result.append({
                "work_id": wid,
                "title": title,
                "year": year,
                "cited_by_count": cited_by_count,
                "abstract": _extract_abstract(w),
            })

        # Sort by citation count to get most influential references
        result.sort(key=lambda x: x.get("cited_by_count") or 0, reverse=True)

        logger.info(f"Fetched {len(result)} references for {work_id}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch references for {work_id}: {e}")
        return []


def fetch_seed_paper_details(work_id: str) -> Optional[Dict[str, Any]]:
    """Fetch details for a single paper from OpenAlex."""
    if not work_id:
        return None

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        title = data.get("title")

        # Backfill empty titles from S2
        if not title:
            doi = data.get("doi")
            title = _backfill_title_from_s2(doi)

        return {
            "work_id": work_id,
            "title": title,
            "year": data.get("publication_year"),
            "cited_by_count": data.get("cited_by_count") or 0,
            "abstract": _extract_abstract(data),
        }

    except Exception as e:
        logger.warning(f"Failed to fetch seed paper details for {work_id}: {e}")
        return None


# ============================================================================
# LLM Seed Validation
# ============================================================================

# Tier to score mapping (same as feature2/llm_relevance.py)
TIER_SCORES = {
    "ESSENTIAL": 0.95,
    "HIGH": 0.75,
    "MEDIUM": 0.50,
    "LOW": 0.25,
    "NONE": 0.05,
}

# Minimum LLM score to be considered a valid seed (HIGH tier)
MIN_SEED_LLM_SCORE = 0.75

# Minimum LLM score for connection papers (MEDIUM+ for diversity)
MIN_CONNECTION_LLM_SCORE = 0.50


def _stratified_sample(
    papers: List[Dict[str, Any]],
    limit: int,
    high_ratio: float = 0.6,
    mid_ratio: float = 0.3,
    low_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """Sample papers with citation diversity: high, medium, and low cited.

    Ensures the graph has influential papers but also emerging/niche work.
    Papers should already be sorted by citation count (desc).
    """
    if not papers or limit <= 0:
        return []

    if len(papers) <= limit:
        return papers

    # Calculate tier sizes
    n_high = max(1, int(limit * high_ratio))
    n_mid = max(1, int(limit * mid_ratio))
    n_low = max(0, limit - n_high - n_mid)

    # Split into thirds by position (already sorted by citations)
    third = len(papers) // 3
    high_pool = papers[:third] if third > 0 else papers[:1]
    mid_pool = papers[third:2*third] if third > 0 else []
    low_pool = papers[2*third:] if third > 0 else []

    result = []

    # Take from high tier
    result.extend(high_pool[:n_high])

    # Take from mid tier (if available)
    if mid_pool:
        result.extend(mid_pool[:n_mid])
    elif high_pool:
        # Fallback to more high if no mid
        result.extend(high_pool[n_high:n_high + n_mid])

    # Take from low tier (if available)
    if low_pool and n_low > 0:
        result.extend(low_pool[:n_low])
    elif mid_pool and n_low > 0:
        # Fallback to more mid if no low
        result.extend(mid_pool[n_mid:n_mid + n_low])

    return result[:limit]


# ============================================================================
# Multi-hop Citation Network Expansion
# ============================================================================

def _merge_s2_citations_with_openalex(
    s2_papers: List[Dict[str, Any]],
    existing_work_ids: set,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Map S2 citation results back to OpenAlex IDs using DOI bridge.

    Returns:
        - papers: list of paper dicts with OpenAlex work_ids
        - doi_to_workid: mapping of DOI to OpenAlex work_id for edge creation
    """
    if not s2_papers:
        return [], {}

    # Collect DOIs that need mapping
    dois_to_lookup = []
    for p in s2_papers:
        doi = p.get("doi")
        if doi:
            clean_doi = doi.replace("https://doi.org/", "")
            dois_to_lookup.append(clean_doi)

    if not dois_to_lookup:
        return [], {}

    # Batch lookup DOIs -> OpenAlex IDs
    doi_to_workid = _batch_lookup_openalex_by_dois(dois_to_lookup)

    # Convert S2 papers to OpenAlex-compatible format
    result = []
    for p in s2_papers:
        doi = p.get("doi")
        if not doi:
            continue

        clean_doi = doi.replace("https://doi.org/", "")
        work_id = doi_to_workid.get(clean_doi)

        if work_id and work_id not in existing_work_ids:
            result.append({
                "work_id": work_id,
                "title": p.get("title"),
                "year": p.get("year"),
                "cited_by_count": p.get("cited_by_count") or 0,
                "abstract": None,  # S2 doesn't return abstracts in citation endpoints
                "source": "semantic_scholar",
            })

    logger.info(f"S2 DOI bridge: {len(s2_papers)} S2 papers -> {len(result)} mapped to OpenAlex")
    return result, doi_to_workid


def _expand_citation_network(
    seed_work_id: str,
    total_limit: int = 30,
    hop1_fetch: int = 50,
    hop2_fetch: int = 20,
    citation_weight: float = 0.6,
    connectivity_weight: float = 0.4,
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str]]]:
    """Expand citation network from seed using multi-hop exploration.

    Uses both OpenAlex AND Semantic Scholar as citation sources.
    S2 results are mapped back to OpenAlex IDs via DOI bridge.

    Returns:
        - papers: dict mapping work_id to paper data
        - edges: list of (from_id, to_id) tuples where from cites to
    """
    import math

    papers: Dict[str, Dict[str, Any]] = {}
    edges: List[Tuple[str, str]] = []
    edge_set: set = set()

    def add_edge(from_id: str, to_id: str):
        if (from_id, to_id) not in edge_set:
            edge_set.add((from_id, to_id))
            edges.append((from_id, to_id))

    # Get seed details (always add seed even if fetch fails)
    seed_data = fetch_seed_paper_details(seed_work_id)
    if seed_data:
        papers[seed_work_id] = {**seed_data, "hop": 0, "is_seed": True}
    else:
        # Placeholder if we can't fetch seed details
        papers[seed_work_id] = {
            "work_id": seed_work_id,
            "title": None,
            "year": None,
            "cited_by_count": 0,
            "hop": 0,
            "is_seed": True,
        }

    # Get DOI for seed to enable S2 lookup
    seed_doi = _get_doi_for_work(seed_work_id)
    logger.info(f"Seed DOI for S2 bridge: {seed_doi}")

    # Hop 1: Get seed's direct connections from OpenAlex
    hop1_citing = fetch_citing_papers(seed_work_id, limit=hop1_fetch, fetch_limit=hop1_fetch)
    hop1_refs = fetch_references(seed_work_id, limit=hop1_fetch, fetch_limit=hop1_fetch)

    # Also get from S2 if we have a DOI
    if seed_doi:
        s2_citing = _fetch_citing_papers_s2(seed_doi, limit=hop1_fetch)
        s2_refs = _fetch_references_s2(seed_doi, limit=hop1_fetch)

        # Map S2 results to OpenAlex IDs and merge
        s2_citing_mapped, doi_map_citing = _merge_s2_citations_with_openalex(
            s2_citing, set(papers.keys()) | {p.get("work_id") for p in hop1_citing}
        )
        s2_refs_mapped, doi_map_refs = _merge_s2_citations_with_openalex(
            s2_refs, set(papers.keys()) | {p.get("work_id") for p in hop1_refs}
        )

        hop1_citing.extend(s2_citing_mapped)
        hop1_refs.extend(s2_refs_mapped)

    for p in hop1_citing:
        wid = p.get("work_id")
        if wid and wid not in papers:
            papers[wid] = {**p, "hop": 1, "is_seed": False}
        if wid:
            add_edge(wid, seed_work_id)  # citing paper -> seed

    for p in hop1_refs:
        wid = p.get("work_id")
        if wid and wid not in papers:
            papers[wid] = {**p, "hop": 1, "is_seed": False}
        if wid:
            add_edge(seed_work_id, wid)  # seed -> reference

    # Hop 2: Expand from top hop-1 papers (by citations)
    hop1_papers = [(wid, papers[wid]) for wid in papers if papers[wid].get("hop") == 1]
    hop1_papers.sort(key=lambda x: x[1].get("cited_by_count", 0), reverse=True)

    # Only expand from top N hop-1 papers to limit API calls
    top_hop1 = hop1_papers[:min(10, len(hop1_papers))]

    for wid, paper_data in top_hop1:
        # Get this paper's connections from OpenAlex
        h2_citing = fetch_citing_papers(wid, limit=hop2_fetch, fetch_limit=hop2_fetch)
        h2_refs = fetch_references(wid, limit=hop2_fetch, fetch_limit=hop2_fetch)

        # Also try S2 for hop-2 expansion (only for high-cited hop-1 papers)
        hop1_doi = _get_doi_for_work(wid)
        if hop1_doi and paper_data.get("cited_by_count", 0) > 100:
            s2_h2_citing = _fetch_citing_papers_s2(hop1_doi, limit=hop2_fetch // 2)
            s2_h2_refs = _fetch_references_s2(hop1_doi, limit=hop2_fetch // 2)

            s2_h2_citing_mapped, _ = _merge_s2_citations_with_openalex(
                s2_h2_citing, set(papers.keys()) | {p.get("work_id") for p in h2_citing}
            )
            s2_h2_refs_mapped, _ = _merge_s2_citations_with_openalex(
                s2_h2_refs, set(papers.keys()) | {p.get("work_id") for p in h2_refs}
            )

            h2_citing.extend(s2_h2_citing_mapped)
            h2_refs.extend(s2_h2_refs_mapped)

        for p in h2_citing:
            pid = p.get("work_id")
            if pid and pid not in papers:
                papers[pid] = {**p, "hop": 2, "is_seed": False}
            if pid:
                add_edge(pid, wid)

        for p in h2_refs:
            pid = p.get("work_id")
            if pid and pid not in papers:
                papers[pid] = {**p, "hop": 2, "is_seed": False}
            if pid:
                add_edge(wid, pid)

    # Calculate connectivity (degree) for each paper in the local graph
    degree: Dict[str, int] = {wid: 0 for wid in papers}
    for from_id, to_id in edges:
        if from_id in degree:
            degree[from_id] += 1
        if to_id in degree:
            degree[to_id] += 1

    # Identify papers that cite seed vs papers seed cites
    citing_seed = {f for f, t in edges if t == seed_work_id}
    cited_by_seed = {t for f, t in edges if f == seed_work_id}

    # Normalize citations and connectivity for scoring
    max_cites = max((p.get("cited_by_count", 0) for p in papers.values()), default=1) or 1
    max_degree = max(degree.values(), default=1) or 1

    # Hop distance penalty (closer to seed = better)
    HOP_PENALTY = {0: 1.0, 1: 1.0, 2: 0.6}  # hop 2 gets 40% penalty

    # Minimum local density for hop-2 papers (filters out mega-cited generic papers)
    # density = local_degree / log(global_citations)
    # Lowered from 0.40 to 0.20 to avoid filtering legitimate related papers
    MIN_HOP2_DENSITY = 0.20

    # Score each paper: hybrid of citations + connectivity + hop proximity
    scores: Dict[str, float] = {}
    for wid, p in papers.items():
        if p.get("is_seed"):
            scores[wid] = float("inf")  # Seed always included
            continue

        cites = p.get("cited_by_count", 0)
        deg = degree.get(wid, 0)
        hop = p.get("hop", 2)

        # Log-scale citations to reduce dominance of mega-cited papers
        # Cap at 90th percentile to prevent mega-papers from dominating
        norm_cites = math.log1p(min(cites, 50000)) / math.log1p(50000)
        norm_deg = deg / max_degree

        # Apply hop penalty
        hop_mult = HOP_PENALTY.get(hop, 0.5)

        scores[wid] = hop_mult * (citation_weight * norm_cites + connectivity_weight * norm_deg)

    # Stratified selection: ensure balance between citing/cited/hop2
    selected_ids = {seed_work_id}

    # Reserve quota for papers that cite seed (forward-looking, newer work)
    citing_quota = max(3, total_limit // 4)  # ~25% citing papers
    citing_papers_sorted = sorted(
        [(wid, scores.get(wid, 0)) for wid in citing_seed if wid not in selected_ids],
        key=lambda x: -x[1]
    )
    for wid, _ in citing_papers_sorted[:citing_quota]:
        selected_ids.add(wid)

    # Reserve quota for papers seed cites (foundations)
    cited_quota = max(3, total_limit // 4)  # ~25% cited papers
    cited_papers_sorted = sorted(
        [(wid, scores.get(wid, 0)) for wid in cited_by_seed if wid not in selected_ids],
        key=lambda x: -x[1]
    )
    for wid, _ in cited_papers_sorted[:cited_quota]:
        selected_ids.add(wid)

    # Reserve quota for hop-2 papers (network expansion beyond direct connections)
    # IMPORTANT: Require hop-2 papers to have:
    #   1. degree >= 2 (connected to multiple papers in local graph)
    #   2. high local density (local_degree / log(global_citations + 1))
    # This filters out mega-cited generic papers (ResNet, LSTM) that connect to everything
    hop2_quota = max(2, total_limit // 5)  # ~20% hop-2 papers
    # Patterns indicating survey/review papers or domain-specific papers that cite broadly
    SURVEY_PATTERNS = ("survey", "review", "overview", "tutorial", "primer", "analysis")

    hop2_papers = []
    for wid, p in papers.items():
        if p.get("hop") != 2 or wid in selected_ids:
            continue

        # Skip survey/review papers in hop-2 (they cite broadly)
        title = (p.get("title") or "").lower()
        if any(pat in title for pat in SURVEY_PATTERNS):
            continue

        deg = degree.get(wid, 0)
        global_cites = p.get("cited_by_count", 0)

        # Scale degree requirement by citation count:
        # - Papers with >50K cites: need degree >= 5 (truly central to topic)
        # - Papers with >10K cites: need degree >= 4
        # - Papers with <10K cites: need degree >= 3
        if global_cites > 50000:
            min_deg = 5
        elif global_cites > 10000:
            min_deg = 4
        else:
            min_deg = 3

        if deg < min_deg:
            continue

        # Calculate local density: how connected is this paper relative to its global reach?
        local_density = deg / math.log1p(global_cites + 1)
        if local_density >= MIN_HOP2_DENSITY:
            hop2_papers.append((wid, scores.get(wid, 0), local_density))

    hop2_papers_sorted = sorted(hop2_papers, key=lambda x: -x[1])
    for wid, _, _ in hop2_papers_sorted[:hop2_quota]:
        selected_ids.add(wid)

    # Fill remainder with best-scored papers (any hop)
    # For hop-2 papers, apply same density filter to avoid generic mega-cited papers
    remaining = total_limit - len(selected_ids)
    if remaining > 0:
        candidates = []
        for wid in papers:
            if papers[wid].get("is_seed") or wid in selected_ids:
                continue
            hop = papers[wid].get("hop", 0)
            if hop <= 1:
                candidates.append((wid, scores[wid]))
            else:
                # Hop-2: scale degree requirement by citation count + density filter
                # Skip surveys (cite broadly)
                title = (papers[wid].get("title") or "").lower()
                if any(pat in title for pat in SURVEY_PATTERNS):
                    continue

                deg = degree.get(wid, 0)
                global_cites = papers[wid].get("cited_by_count", 0)

                if global_cites > 50000:
                    min_deg = 5
                elif global_cites > 10000:
                    min_deg = 4
                else:
                    min_deg = 3

                if deg < min_deg:
                    continue

                local_density = deg / math.log1p(global_cites + 1)
                if local_density >= MIN_HOP2_DENSITY:
                    candidates.append((wid, scores[wid]))
        sorted_papers = sorted(candidates, key=lambda x: -x[1])
        for wid, _ in sorted_papers[:remaining]:
            selected_ids.add(wid)

    # Filter papers and edges to selected set
    filtered_papers = {wid: papers[wid] for wid in selected_ids}
    filtered_edges = [
        (f, t) for f, t in edges
        if f in selected_ids and t in selected_ids
    ]

    logger.info(
        f"Citation network: {len(papers)} explored -> {len(filtered_papers)} selected, "
        f"{len(edges)} edges -> {len(filtered_edges)} kept"
    )

    return filtered_papers, filtered_edges


def _score_seed_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, float]:
    """Score seed candidates using LLM tier classification.

    Returns dict mapping work_id to relevance score (0.0 - 1.0).
    Only scores the top_k candidates to limit API calls.
    """
    if not candidates:
        return {}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping LLM validation")
        return {}

    # Limit to top_k candidates (already sorted by citation count)
    candidates_to_score = candidates[:top_k]

    # Prepare papers for prompt
    papers_for_prompt = []
    for c in candidates_to_score:
        papers_for_prompt.append({
            "id": c.get("work_id"),
            "title": c.get("title", "")[:200],
        })

    papers_json = json.dumps(papers_for_prompt, indent=2)

    prompt = f"""Classify papers by relevance to the query. Return ONLY a JSON object mapping paper_id to relevance tier.

STRICT RELEVANCE RULES:
1. Papers must be DIRECTLY about the query topic to score HIGH or ESSENTIAL
2. Generic methodology papers, tools, or techniques that just COULD be used → LOW or MEDIUM
3. Papers in a tangentially related field → LOW

INTERSECTION MATCHING (CRITICAL):
- Query has MULTIPLE components (e.g., "deep learning" + "drug discovery")
- Paper must match ALL components to score HIGH
- Paper matching only ONE component → LOW or MEDIUM
- Examples:
  * Query "deep learning drug discovery" + Paper "Natural products in drug discovery" → LOW (drug discovery but NOT deep learning)
  * Query "transformer attention NLP" + Paper "Transformers for time series" → LOW (transformers but NOT NLP)
  * Query "autonomous driving perception" + Paper "UAV computer vision" → LOW (CV but NOT autonomous driving)

DOMAIN MATCHING:
- If query specifies an application domain (e.g., "NLP", "robotics", "medical", "autonomous driving")
- Papers in a DIFFERENT domain must score LOW even if they use similar techniques
- Examples:
  * Query "reinforcement learning robotics" + Paper "RL for game playing" → LOW (RL but wrong domain)
  * Query "computer vision autonomous driving" + Paper "CV for drone navigation" → LOW (CV but wrong domain)

RELEVANCE TIERS:
- ESSENTIAL: Core focus DIRECTLY on ALL query components (rare - 1-2 per query)
- HIGH: Directly addresses ALL query components (not just some of them)
- MEDIUM: Matches most components but missing one, OR foundational work
- LOW: Matches only ONE component, wrong domain, or generic tool
- NONE: Unrelated

QUERY: {query}

PAPERS:
{papers_json}

OUTPUT (JSON only, no explanation):
{{"paper_id": "TIER", ...}}"""

    client = Groq(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            start = content.find("{")
            if start == -1:
                logger.warning(f"LLM seed validation: no JSON in response")
                continue

            # Find matching closing brace
            depth = 0
            end = start
            for i, c in enumerate(content[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            json_str = content[start:end]
            response_map = json.loads(json_str)

            # Convert tiers to scores
            result = {}
            for c in candidates_to_score:
                wid = c.get("work_id")
                tier = response_map.get(wid, "NONE").upper()
                if tier not in TIER_SCORES:
                    tier = "NONE"
                result[wid] = TIER_SCORES[tier]

            logger.info(f"LLM seed validation: scored {len(result)} candidates")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM seed validation JSON error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        except Exception as e:
            logger.warning(f"LLM seed validation error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

    return {}


# ============================================================================
# Seed Selection from NL Query
# ============================================================================

def _search_s2_by_title(title: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search Semantic Scholar by title.

    S2 often has better metadata (correct years, citations) than OpenAlex for some papers.
    Returns papers with identifiers (DOI, ArXiv) for mapping to OpenAlex.
    """
    if not title:
        return []

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": limit,
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 429:
            time.sleep(1)
            resp = requests.get(url, params=params, headers=headers, timeout=15)

        resp.raise_for_status()
        data = resp.json()

        result = []
        for paper in data.get("data", []):
            external_ids = paper.get("externalIds") or {}

            result.append({
                "s2_id": paper.get("paperId"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "cited_by_count": paper.get("citationCount") or 0,
                "doi": external_ids.get("DOI"),
                "arxiv_id": external_ids.get("ArXiv"),
            })

        return result
    except Exception as e:
        logger.warning(f"S2 title search error: {e}")
        return []


def _lookup_openalex_by_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Look up OpenAlex work by ArXiv ID.

    Note: OpenAlex ArXiv ID filter is inconsistent and often fails.
    This is a best-effort lookup.
    """
    if not arxiv_id:
        return None

    # ArXiv ID lookup is unreliable in OpenAlex - skip for now
    # The S2 -> OpenAlex title matching provides the same functionality
    return None


def _expand_search_queries(query: str) -> List[Dict[str, Any]]:
    """Use LLM to get titles + years of foundational papers.

    Returns list of dicts with: title, year, citations (approximate)
    We then match on title + year + high citations to find the exact paper.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return []

    prompt = f"""What are the 3 most famous/seminal academic papers for: {query}

For each paper provide: exact title, publication year, approximate citation count.

Output ONLY a JSON array:
[{{"title":"Paper Title Here","year":2017,"citations":50000}}]"""

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON array
        if "```" in content:
            start = content.find("```")
            end = content.rfind("```")
            if start < end:
                content = content[start:end]
                content = content.replace("```json", "").replace("```", "").strip()

        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            papers = json.loads(json_str)
            if isinstance(papers, list):
                result = []
                for p in papers[:3]:
                    if isinstance(p, dict) and p.get("title"):
                        result.append({
                            "title": p.get("title"),
                            "year": p.get("year"),
                            "citations": p.get("citations", 0),
                        })
                logger.info(f"LLM expansion: {[r.get('title', '')[:40] for r in result]}")
                return result

    except Exception as e:
        logger.warning(f"LLM query expansion failed: {e}")

    return []


def _find_paper_by_metadata(
    title: str,
    year: Optional[int] = None,
    expected_citations: int = 0,
) -> Optional[Dict[str, Any]]:
    """Find a paper by matching title + year + citation count.

    Uses multiple search strategies and picks the best match based on:
    1. Title similarity
    2. Year match (if provided)
    3. High citation count (seminal papers have many citations)
    """
    if not title:
        return None

    candidates = []
    s2_metadata = {}  # Store S2 metadata for papers (better year/citation data)

    # Strategy 1: Search S2 first (often has better metadata)
    s2_results = _search_s2_by_title(title, limit=5)
    for r in s2_results:
        oa_paper = None

        # Try DOI lookup first
        if r.get("doi"):
            oa_paper = _search_openalex_by_doi(r["doi"])

        # Try ArXiv ID lookup if no DOI
        if not oa_paper and r.get("arxiv_id"):
            oa_paper = _lookup_openalex_by_arxiv(r["arxiv_id"])

        if oa_paper:
            wid = oa_paper.get("work_id")
            # Store S2 metadata (more accurate year/citations)
            s2_metadata[wid] = {
                "title": r.get("title"),
                "year": r.get("year"),
                "cited_by_count": r.get("cited_by_count", 0),
            }
            # If OpenAlex has empty title or wrong year, use S2 data
            if not oa_paper.get("title") and r.get("title"):
                oa_paper["title"] = r["title"]
            # If OpenAlex year is clearly wrong (future year), use S2 year
            oa_year = oa_paper.get("year")
            s2_year = r.get("year")
            if oa_year and s2_year and oa_year > 2024 and s2_year <= 2024:
                oa_paper["year"] = s2_year
                oa_paper["cited_by_count"] = r.get("cited_by_count", 0)  # S2 citations are correct
            candidates.append(oa_paper)

    # Strategy 2: For high-cited S2 papers without DOI/ArXiv, match by title to OpenAlex
    # This handles cases like Transformer paper where OpenAlex has wrong year but S2 has correct data
    for r in s2_results:
        if r.get("cited_by_count", 0) >= 10000:  # Only for seminal papers
            s2_title = (r.get("title") or "").lower().strip()
            if not s2_title:
                continue

            # Search OpenAlex by title and find exact match
            oa_title_results = _search_openalex_by_title(r.get("title", ""), k=5)
            for oa_paper in oa_title_results:
                oa_title = (oa_paper.get("title") or "").lower().strip()
                wid = oa_paper.get("work_id")

                # Exact title match (case insensitive)
                if oa_title == s2_title and wid not in [c.get("work_id") for c in candidates]:
                    # Substitute S2's correct metadata
                    oa_paper["title"] = r.get("title")  # Use S2 title
                    oa_paper["year"] = r.get("year")  # Use S2 year (correct)
                    oa_paper["cited_by_count"] = r.get("cited_by_count", 0)  # Use S2 citations
                    oa_paper["metadata_source"] = "s2_override"
                    candidates.append(oa_paper)
                    logger.info(f"Matched S2 paper to OA {wid} with S2 metadata override")
                    break

    # Strategy 3: Search OpenAlex by title
    oa_results = _search_openalex_by_title(title, k=10, year=year, year_range=2)
    candidates.extend(oa_results)

    if not candidates:
        return None

    # Dedupe by work_id
    seen = set()
    unique = []
    for c in candidates:
        wid = c.get("work_id")
        if wid and wid not in seen:
            seen.add(wid)
            unique.append(c)
    candidates = unique

    # Simple matching: pick highest-cited paper within year range
    # Title + Year + High Citations = unique identification
    best = None
    best_score = -1

    for c in candidates:
        cites = c.get("cited_by_count", 0)
        c_year = c.get("year")

        # Year must match within 2 years if provided
        if year and c_year and abs(c_year - year) > 2:
            continue

        # Citations as primary score (seminal papers have high citations)
        score = cites

        if score > best_score:
            best_score = score
            best = c

    if not best:
        # No year match - just pick highest cited
        candidates.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
        best = candidates[0]

    logger.info(f"Found paper: '{best.get('title', '')[:50]}' ({best.get('year')}) - {best.get('cited_by_count', 0):,} citations")
    return best


def _search_openalex_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Search OpenAlex by DOI - most reliable way to find a specific paper."""
    if not doi:
        return None

    # Clean DOI
    clean_doi = doi.replace("https://doi.org/", "").strip()
    if not clean_doi:
        return None

    url = "https://api.openalex.org/works"
    params = {
        "filter": f"doi:{clean_doi}",
        "select": "id,title,publication_year,cited_by_count",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    try:
        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            r = results[0]
            oa_id = r.get("id", "")
            if oa_id.startswith("https://openalex.org/"):
                work_id = oa_id.replace("https://openalex.org/", "")
                return {
                    "work_id": work_id,
                    "title": r.get("title"),
                    "year": r.get("publication_year"),
                    "cited_by_count": r.get("cited_by_count") or 0,
                    "source": "openalex_doi",
                }
    except Exception as e:
        logger.warning(f"OpenAlex DOI search error for {doi}: {e}")

    return None


def select_seed_from_query(query: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select the best seed paper from a natural language query.

    Uses same retrieval sources as broad ranking:
    1. OpenAlex (relevance + highly-cited)
    2. Semantic Scholar bulk
    3. ArXiv
    4. Boosted searches for foundational papers (based on query keywords)

    Then validates topical relevance using LLM scoring:
    - Only papers with HIGH+ relevance (score >= 0.75) are considered
    - Among those, pick the highest cited

    Returns:
        - seed_work_id (or None if not found)
        - selection_info dict
    """
    if not query:
        return None, {"selection_strategy": "none", "selection_reason": "Empty query"}

    # Use dict to allow updating entries when better metadata is found
    papers_by_id: Dict[str, Dict[str, Any]] = {}

    # Use LLM to get titles + years of foundational papers
    expanded_queries = _expand_search_queries(query)

    # Run searches in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(_search_openalex, query, OPENALEX_LIMIT): "openalex",
            executor.submit(_search_openalex_highly_cited, query, OPENALEX_HIGHLY_CITED_LIMIT): "openalex_cited",
            executor.submit(_search_semantic_scholar, query, S2_LIMIT): "s2",
            executor.submit(_search_arxiv, query, ARXIV_LIMIT): "arxiv",
        }

        # Add LLM-generated searches - find by title + year + citation matching
        for exp in expanded_queries:
            if isinstance(exp, dict):
                title = exp.get("title", "")
                year = exp.get("year")
                citations = exp.get("citations", 0)
                if title:
                    # Use multi-field matching: title + year + citations
                    futures[executor.submit(
                        _find_paper_by_metadata, title, year, citations
                    )] = f"llm_match:{title[:30]}"

        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                # Handle both list and single dict results
                papers = [result] if isinstance(result, dict) else (result or [])
                for paper in papers:
                    if not paper:
                        continue
                    work_id = paper.get("work_id", "")
                    if not work_id:
                        continue

                    # Smart dedup: prefer higher citation counts (S2 metadata override)
                    # This handles OpenAlex data corruption where famous papers have wrong metadata
                    existing = papers_by_id.get(work_id)
                    if existing:
                        existing_cites = existing.get("cited_by_count", 0)
                        new_cites = paper.get("cited_by_count", 0)
                        # Replace if new entry has significantly more citations (5x or 10K+ more)
                        if new_cites > existing_cites * 5 or new_cites > existing_cites + 10000:
                            logger.debug(
                                f"Updating paper metadata: {work_id} "
                                f"({existing_cites:,} -> {new_cites:,} cites)"
                            )
                            papers_by_id[work_id] = paper
                    else:
                        papers_by_id[work_id] = paper
            except Exception as e:
                logger.warning(f"Search {source} failed: {e}")

    all_papers = list(papers_by_id.values())

    if not all_papers:
        return None, {
            "selection_strategy": "llm_validated_highest_cited",
            "selection_reason": "No papers found for query",
            "candidates_considered": 0,
        }

    # Filter to only OpenAlex IDs (W...) for seed - we need these for citation expansion
    openalex_papers = [p for p in all_papers if p.get("work_id", "").startswith("W")]
    if not openalex_papers:
        return None, {
            "selection_strategy": "llm_validated_highest_cited",
            "selection_reason": "No OpenAlex papers found (needed for citation expansion)",
            "candidates_considered": len(all_papers),
        }

    # Sort by citation count to prioritize highly-cited candidates
    openalex_papers.sort(key=lambda p: p.get("cited_by_count") or 0, reverse=True)

    # Run LLM validation on top candidates
    llm_scores = _score_seed_candidates(query, openalex_papers, top_k=20)

    if llm_scores:
        # Filter to HIGH+ relevance papers (score >= 0.75)
        relevant_papers = [
            p for p in openalex_papers
            if llm_scores.get(p.get("work_id"), 0) >= MIN_SEED_LLM_SCORE
        ]

        if relevant_papers:
            # Pick highest cited among relevant papers
            relevant_papers.sort(key=lambda p: p.get("cited_by_count") or 0, reverse=True)
            best_paper = relevant_papers[0]
            llm_score = llm_scores.get(best_paper.get("work_id"), 0)

            return best_paper["work_id"], {
                "selection_strategy": "llm_validated_highest_cited",
                "selection_reason": (
                    f"Highest cited ({best_paper.get('cited_by_count', 0):,}) "
                    f"among {len(relevant_papers)} LLM-validated papers "
                    f"(relevance score: {llm_score:.2f})"
                ),
                "candidates_considered": len(all_papers),
                "llm_validated_count": len(relevant_papers),
                "seed_title": best_paper.get("title"),
            }
        else:
            # No papers passed LLM validation - fall back to highest scored
            # (even if below threshold)
            scored_papers = [
                (p, llm_scores.get(p.get("work_id"), 0))
                for p in openalex_papers
                if p.get("work_id") in llm_scores
            ]
            if scored_papers:
                scored_papers.sort(key=lambda x: (-x[1], -(x[0].get("cited_by_count") or 0)))
                best_paper, best_score = scored_papers[0]

                return best_paper["work_id"], {
                    "selection_strategy": "llm_validated_highest_cited",
                    "selection_reason": (
                        f"Best available (LLM score: {best_score:.2f}, "
                        f"citations: {best_paper.get('cited_by_count', 0):,}) - "
                        f"no papers reached HIGH relevance threshold"
                    ),
                    "candidates_considered": len(all_papers),
                    "llm_validated_count": 0,
                    "seed_title": best_paper.get("title"),
                }

    # Fallback: LLM validation failed, use highest cited
    best_paper = openalex_papers[0]
    return best_paper["work_id"], {
        "selection_strategy": "highest_cited_fallback",
        "selection_reason": (
            f"Highest cited ({best_paper.get('cited_by_count', 0):,}) - "
            f"LLM validation unavailable"
        ),
        "candidates_considered": len(all_papers),
        "seed_title": best_paper.get("title"),
    }


# ============================================================================
# Graph Assembly
# ============================================================================

def _assemble_citation_graph(
    seed_work_id: str,
    seed_data: Dict[str, Any],
    citing_papers: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    min_citations: int = 0,
) -> Tuple[List[CitationNode], List[CitationEdge]]:
    """Assemble nodes and edges from citation data.

    Edge direction: from_work_id cites to_work_id
    - Citing papers: citing_paper → seed
    - References: seed → reference
    """
    nodes: List[CitationNode] = []
    edges: List[CitationEdge] = []
    seen_work_ids: set = set()

    # Seed node (always included)
    nodes.append(CitationNode(
        work_id=seed_work_id,
        title=seed_data.get("title"),
        year=seed_data.get("year"),
        cited_by_count=seed_data.get("cited_by_count") or 0,
        abstract=seed_data.get("abstract"),
        is_seed=True,
        hop=0,
        relationship="seed",
    ))
    seen_work_ids.add(seed_work_id)

    # Citing papers (they cite the seed)
    for paper in citing_papers:
        wid = paper.get("work_id")
        if not wid or wid in seen_work_ids:
            continue
        if (paper.get("cited_by_count") or 0) < min_citations:
            continue

        nodes.append(CitationNode(
            work_id=wid,
            title=paper.get("title"),
            year=paper.get("year"),
            cited_by_count=paper.get("cited_by_count") or 0,
            abstract=paper.get("abstract"),
            is_seed=False,
            hop=1,
            relationship="cites_seed",
        ))
        edges.append(CitationEdge(from_work_id=wid, to_work_id=seed_work_id))
        seen_work_ids.add(wid)

    # References (seed cites them)
    for paper in references:
        wid = paper.get("work_id")
        if not wid or wid in seen_work_ids:
            continue
        if (paper.get("cited_by_count") or 0) < min_citations:
            continue

        nodes.append(CitationNode(
            work_id=wid,
            title=paper.get("title"),
            year=paper.get("year"),
            cited_by_count=paper.get("cited_by_count") or 0,
            abstract=paper.get("abstract"),
            is_seed=False,
            hop=1,
            relationship="cited_by_seed",
        ))
        edges.append(CitationEdge(from_work_id=seed_work_id, to_work_id=wid))
        seen_work_ids.add(wid)

    return nodes, edges


def _assemble_multihop_graph(
    seed_work_id: str,
    papers_dict: Dict[str, Dict[str, Any]],
    edge_tuples: List[Tuple[str, str]],
    min_citations: int = 0,
) -> Tuple[List[CitationNode], List[CitationEdge]]:
    """Assemble nodes and edges from multi-hop expansion data.

    Edge direction: from_work_id cites to_work_id
    """
    nodes: List[CitationNode] = []
    edges: List[CitationEdge] = []

    for work_id, paper in papers_dict.items():
        # Apply minimum citations filter
        if (paper.get("cited_by_count") or 0) < min_citations and not paper.get("is_seed"):
            continue

        hop = paper.get("hop", 0)
        is_seed = paper.get("is_seed", False)

        # Determine relationship based on hop distance
        if is_seed:
            relationship = "seed"
        elif hop == 1:
            # Check edge direction to determine if it cites seed or is cited by seed
            cites_seed = any(f == work_id and t == seed_work_id for f, t in edge_tuples)
            relationship = "cites_seed" if cites_seed else "cited_by_seed"
        else:
            relationship = "network"  # 2+ hop papers

        nodes.append(CitationNode(
            work_id=work_id,
            title=paper.get("title"),
            year=paper.get("year"),
            cited_by_count=paper.get("cited_by_count") or 0,
            abstract=paper.get("abstract"),
            is_seed=is_seed,
            hop=hop,
            relationship=relationship,
        ))

    # Only include edges where both endpoints are in the filtered node set
    node_ids = {n.work_id for n in nodes}
    for from_id, to_id in edge_tuples:
        if from_id in node_ids and to_id in node_ids:
            edges.append(CitationEdge(from_work_id=from_id, to_work_id=to_id))

    logger.info(f"Assembled multi-hop graph: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ============================================================================
# Graph Draft Creation
# ============================================================================

def _create_graph_draft(
    conn: Connection,
    tenant_id: UUID,
    nodes: List[CitationNode],
    edges: List[CitationEdge],
) -> UUID:
    """Create a graph_draft from citation map data."""
    graph_draft_id = uuid4()

    # Insert header
    conn.execute(
        text("""
            INSERT INTO graph_drafts (graph_draft_id, tenant_id, candidate_set_id)
            VALUES (:gd_id, :t_id, NULL)
        """),
        {"gd_id": graph_draft_id, "t_id": tenant_id},
    )

    # Insert nodes
    if nodes:
        for node in nodes:
            conn.execute(
                text("""
                    INSERT INTO graph_draft_nodes (graph_draft_id, work_id)
                    VALUES (:graph_draft_id, :work_id)
                    ON CONFLICT DO NOTHING
                """),
                {"graph_draft_id": graph_draft_id, "work_id": node.work_id},
            )

    # Insert edges
    if edges:
        for edge in edges:
            conn.execute(
                text("""
                    INSERT INTO graph_draft_edges (graph_draft_id, from_work_id, to_work_id)
                    VALUES (:graph_draft_id, :from_work_id, :to_work_id)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "graph_draft_id": graph_draft_id,
                    "from_work_id": edge.from_work_id,
                    "to_work_id": edge.to_work_id,
                },
            )

    conn.commit()
    logger.info(f"Created graph_draft {graph_draft_id} with {len(nodes)} nodes and {len(edges)} edges")
    return graph_draft_id


# ============================================================================
# Main Service Function
# ============================================================================

def build_citation_map(
    engine: Engine,
    *,
    tenant_id: UUID,
    request: CitationMapRequest,
) -> CitationMapResponse:
    """
    Build a citation map around a seed paper.

    Two modes:
    1. Provide seed_work_id directly
    2. Provide query_text to find the best seed paper automatically
    """
    with engine.connect() as conn:
        # Step 1: Determine seed paper
        if request.seed_work_id:
            # Mode 1: Direct seed paper
            seed_work_id = request.seed_work_id
            seed_data = fetch_seed_paper_details(seed_work_id)
            if not seed_data:
                seed_data = {"work_id": seed_work_id, "title": None, "year": None, "cited_by_count": 0}

            seed_info = SeedSelectionInfo(
                seed_work_id=seed_work_id,
                seed_title=seed_data.get("title"),
                selection_strategy="direct",
                selection_reason="Seed paper provided directly",
                candidates_considered=0,
            )

        elif request.query_text:
            # Mode 2: Natural language query
            seed_work_id, selection_info = select_seed_from_query(request.query_text)
            if not seed_work_id:
                # No seed found - return empty response
                return CitationMapResponse(
                    seed_info=SeedSelectionInfo(
                        seed_work_id="",
                        seed_title=None,
                        selection_strategy=selection_info.get("selection_strategy", "none"),
                        selection_reason=selection_info.get("selection_reason", "No seed found"),
                        candidates_considered=selection_info.get("candidates_considered", 0),
                    ),
                    nodes=[],
                    edges=[],
                    graph_draft_id=None,
                    stats=CitationMapStats(),
                )

            seed_data = fetch_seed_paper_details(seed_work_id)
            if not seed_data:
                seed_data = {
                    "work_id": seed_work_id,
                    "title": selection_info.get("seed_title"),
                    "year": None,
                    "cited_by_count": 0,
                }

            seed_info = SeedSelectionInfo(
                seed_work_id=seed_work_id,
                seed_title=seed_data.get("title") or selection_info.get("seed_title"),
                selection_strategy=selection_info.get("selection_strategy", "highest_cited_from_query"),
                selection_reason=selection_info.get("selection_reason"),
                candidates_considered=selection_info.get("candidates_considered", 0),
            )

        else:
            raise ValueError("Either seed_work_id or query_text must be provided")

        # Step 2: Build citation network
        if request.use_multi_hop:
            # Compute effective total: use explicit value or derive from limits
            effective_total = request.total_nodes or (request.citing_limit + request.references_limit + 1)

            # Multi-hop exploration: explores 2 hops from seed with hybrid scoring
            papers_dict, edge_tuples = _expand_citation_network(
                seed_work_id=seed_work_id,
                total_limit=effective_total,
                hop1_fetch=max(request.citing_limit, request.references_limit) * 3,
                hop2_fetch=30,  # Increased to ensure hop-2 papers are available
                citation_weight=0.6,
                connectivity_weight=0.4,
            )

            # Convert to CitationNode and CitationEdge
            nodes, edges = _assemble_multihop_graph(
                seed_work_id=seed_work_id,
                papers_dict=papers_dict,
                edge_tuples=edge_tuples,
                min_citations=request.min_citations,
            )
        else:
            # Legacy 1-hop mode: fetch citing and references separately
            fetch_multiplier = 5
            raw_citing = fetch_citing_papers(
                seed_work_id,
                limit=request.citing_limit,
                fetch_limit=request.citing_limit * fetch_multiplier,
            )
            raw_references = fetch_references(
                seed_work_id,
                limit=request.references_limit,
                fetch_limit=request.references_limit * fetch_multiplier,
            )

            # Apply stratified sampling for citation diversity
            citing_papers = _stratified_sample(raw_citing, request.citing_limit)
            references = _stratified_sample(raw_references, request.references_limit)

            # Assemble 1-hop graph
            nodes, edges = _assemble_citation_graph(
                seed_work_id=seed_work_id,
                seed_data=seed_data,
                citing_papers=citing_papers,
                references=references,
                min_citations=request.min_citations,
            )

        # Step 4: Optionally create graph_draft
        graph_draft_id = None
        if request.create_graph_draft and nodes:
            graph_draft_id = _create_graph_draft(conn, tenant_id, nodes, edges)

        # Step 5: Build stats
        stats = CitationMapStats(
            total_nodes=len(nodes),
            citing_found=len([n for n in nodes if n.relationship == "cites_seed"]),
            references_found=len([n for n in nodes if n.relationship == "cited_by_seed"]),
            edges_count=len(edges),
            network_nodes=len([n for n in nodes if n.relationship == "network"]),
            max_hop=max((n.hop for n in nodes), default=0),
        )

        return CitationMapResponse(
            seed_info=seed_info,
            nodes=nodes,
            edges=edges,
            graph_draft_id=graph_draft_id,
            stats=stats,
        )
