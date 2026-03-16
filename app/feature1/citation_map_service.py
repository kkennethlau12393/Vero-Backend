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
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
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
from app.shared.s2_keys import get_s2_headers

# Ensure .env is loaded
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

DEBUG_LOG_PATH = "/Users/sami/project/vero/Vero-Backend/.cursor/debug-0431e8.log"
DEBUG_SESSION_ID = "0431e8"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    run_id: str = "post-fix",
) -> None:
    try:
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# ============================================================================
# Configuration
# ============================================================================

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# API keys
from app.shared.oa_keys import get_oa_api_key

# Search limits
OPENALEX_LIMIT = 100
OPENALEX_HIGHLY_CITED_LIMIT = 50
S2_LIMIT = 100
ARXIV_LIMIT = 100

# Rate limiting
SEMANTIC_SCHOLAR_DELAY = 1.0
OPENALEX_TIMEOUT = 15
GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "20"))

# S2 retry configuration
S2_MAX_RETRIES = 3
S2_RETRY_BASE_DELAY = 1.0  # seconds — S2 rate limit is 1 RPS for batch/search, 10 RPS for others
S2_RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("S2_RATE_LIMIT_COOLDOWN_SECONDS", "5"))
S2_MIN_REQUEST_INTERVAL = 1.2  # seconds between S2 requests (their limit is 1 RPS, with margin)
OA_MIN_REQUEST_INTERVAL = 0.11  # seconds between OpenAlex requests (their limit is 10 RPS)
ARXIV_MIN_REQUEST_INTERVAL = 3.1  # seconds between ArXiv requests (their limit is 1 req/3s)
_s2_rate_limited_until_ts: float = 0.0
_THROTTLE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
_THROTTLE_FILE = _THROTTLE_DIR / "api_throttle_timestamps.json"


def _read_throttle_ts(key: str) -> float:
    """Read last request timestamp from persistent file. Returns 0.0 if missing."""
    try:
        if _THROTTLE_FILE.exists():
            data = json.loads(_THROTTLE_FILE.read_text())
            return float(data.get(key, 0.0))
    except Exception:
        pass
    return 0.0


def _write_throttle_ts(key: str, ts: float) -> None:
    """Write last request timestamp to persistent file."""
    try:
        _THROTTLE_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        if _THROTTLE_FILE.exists():
            try:
                data = json.loads(_THROTTLE_FILE.read_text())
            except Exception:
                pass
        data[key] = ts
        _THROTTLE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def _s2_throttle():
    """Proactively rate-limit S2 requests to avoid 429s. Persists across runs."""
    last_ts = _read_throttle_ts("s2")
    now = time.time()
    elapsed = now - last_ts
    if elapsed < S2_MIN_REQUEST_INTERVAL:
        time.sleep(S2_MIN_REQUEST_INTERVAL - elapsed)
    _write_throttle_ts("s2", time.time())


def _oa_throttle():
    """Proactively rate-limit OpenAlex requests to avoid 429s. Persists across runs."""
    last_ts = _read_throttle_ts("oa")
    now = time.time()
    elapsed = now - last_ts
    if elapsed < OA_MIN_REQUEST_INTERVAL:
        time.sleep(OA_MIN_REQUEST_INTERVAL - elapsed)
    _write_throttle_ts("oa", time.time())


def _arxiv_throttle():
    """Proactively rate-limit ArXiv requests to avoid 429s. Persists across runs."""
    last_ts = _read_throttle_ts("arxiv")
    now = time.time()
    elapsed = now - last_ts
    if elapsed < ARXIV_MIN_REQUEST_INTERVAL:
        time.sleep(ARXIV_MIN_REQUEST_INTERVAL - elapsed)
    _write_throttle_ts("arxiv", time.time())


def _s2_get_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    max_retries: int = S2_MAX_RETRIES,
) -> requests.Response:
    """Make a GET request to S2 API with exponential backoff on 429/500.

    Returns the response object. Raises on non-retryable errors.
    On exhausted retries, returns the last retryable response (caller decides what to do).
    """
    global _s2_rate_limited_until_ts
    _s2_throttle()  # Proactive rate limiting — wait if needed before request
    now_ts = time.time()
    if now_ts < _s2_rate_limited_until_ts:
        # #region agent log
        _debug_log(
            "H16",
            "app/feature1/citation_map_service.py:_s2_get_with_retry.cooldown_skip",
            "S2 request skipped due to active cooldown",
            {
                "urlPrefix": url[:120],
                "cooldownRemainingMs": int((_s2_rate_limited_until_ts - now_ts) * 1000),
            },
        )
        # #endregion
        synthetic = requests.Response()
        synthetic.status_code = 429
        synthetic.url = url
        synthetic._content = b"S2 cooldown active"
        return synthetic

    last_resp = None
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code not in (429, 500):
            # Success or non-retryable error — clear any active cooldown
            _s2_rate_limited_until_ts = 0.0
            return resp
        last_resp = resp
        # #region agent log
        _debug_log(
            "H16",
            "app/feature1/citation_map_service.py:_s2_get_with_retry.rate_limited",
            f"S2 returned {resp.status_code}; retrying",
            {
                "urlPrefix": url[:120],
                "attempt": attempt + 1,
                "maxRetries": max_retries,
            },
        )
        # #endregion
        if attempt < max_retries:
            delay = S2_RETRY_BASE_DELAY * (2 ** attempt)  # 1s, 2s, 4s
            logger.info(f"S2 error ({resp.status_code}), retry {attempt + 1}/{max_retries} after {delay}s: {url[:80]}")
            time.sleep(delay)

    # All retries exhausted — activate cooldown to avoid hammering S2
    _s2_rate_limited_until_ts = time.time() + S2_RATE_LIMIT_COOLDOWN_SECONDS
    logger.warning(f"S2 rate limited after {max_retries} retries, cooldown {S2_RATE_LIMIT_COOLDOWN_SECONDS}s: {url[:80]}")
    return last_resp  # Return last 429 response if all retries exhausted


# ============================================================================
# Text normalization
# ============================================================================

def _normalize_title_text(title: str) -> str:
    """Normalize title text for API search and comparison.

    Handles PDF-extracted ligatures (ﬁ→fi, ﬂ→fl, etc.) and collapses whitespace.
    NFKC decomposition maps compatibility characters to their canonical forms.
    """
    if not title:
        return title
    normalized = unicodedata.normalize("NFKC", title)
    return " ".join(normalized.split())


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _strip_punct(word: str) -> str:
    """Strip punctuation from a word for Jaccard comparison."""
    return _PUNCT_RE.sub("", word)


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
    oa_key = get_oa_api_key()
    if oa_key:
        params["api_key"] = oa_key

    url = "https://api.openalex.org/works"

    for attempt in range(MAX_RETRIES):
        try:
            _oa_throttle()
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
    oa_key = get_oa_api_key()
    if oa_key:
        params["api_key"] = oa_key

    for attempt in range(MAX_RETRIES):
        try:
            _oa_throttle()
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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        try:
            _oa_throttle()
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

    # Strip commas — OpenAlex interprets them as filter value separators
    safe_title = title_query.replace(",", "")

    # Search 1: Title search without year (catches papers with wrong year in OpenAlex)
    do_search(f'title.search:"{safe_title}"')

    # Search 2: Title search with year filter if provided
    if year:
        year_filter = f",publication_year:{year - year_range}-{year + year_range}"
        do_search(f'title.search:"{safe_title}"{year_filter}')

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

    headers = get_s2_headers()

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

        try:
            resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=30)

            # 400 error = query too broad, fall back to regular search
            if resp.status_code == 400:
                return _search_s2_regular(query, min(k, 100), headers)

            if resp.status_code == 429:
                logger.warning("S2 bulk search rate limited after retries")
                return out

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

        except Exception as e:
            logger.warning(f"S2 bulk search error: {e}")
            return out

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

        try:
            resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 429:
                logger.warning("S2 regular search rate limited after retries")
                return out

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

        except Exception as e:
            logger.warning(f"S2 regular search failed: {e}")
            return out

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
    words = [w for w in words if w.upper() not in ("AND", "OR", "NOT")]
    if not words:
        return []
    search_terms = [f"all:{quote(word)}" for word in words]
    search_query = "+AND+".join(search_terms)
    url = f"{base_url}?search_query={search_query}&max_results={k}&sortBy=relevance"

    for attempt in range(MAX_RETRIES):
        try:
            _arxiv_throttle()
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
                    "doi": f"10.48550/arXiv.{arxiv_id}",
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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
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

    headers = get_s2_headers()

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "paperId,title,year,citationCount,externalIds"}
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)

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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
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


def _get_s2_paper_id(doi: Optional[str] = None, title: Optional[str] = None) -> Optional[str]:
    """Get S2 paper ID for a paper using any available identifier.

    Tries DOI first, then title search. Returns S2 paper ID or None.
    """
    # Try DOI first (most reliable)
    if doi:
        paper = _lookup_s2_paper_by_doi(doi)
        if paper and paper.get("paperId"):
            return paper["paperId"]

    # Try title search
    if title:
        results = _search_s2_by_title(title, limit=3)
        if results:
            return results[0].get("s2_id")

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
            oa_key = get_oa_api_key()
            if oa_key:
                params["api_key"] = oa_key

            _oa_throttle()
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

def _fetch_citing_papers_s2(identifier: str, limit: int = 50, id_type: str = "DOI") -> List[Dict[str, Any]]:
    """Fetch papers that cite a given work from Semantic Scholar.

    Supports multiple identifier types: DOI, S2 paper ID, ArXiv ID.
    Returns list of dicts with: doi, s2_id, title, year, cited_by_count
    """
    if not identifier:
        return []

    headers = get_s2_headers()

    try:
        if id_type == "S2":
            url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}/citations"
        elif id_type == "ArXiv":
            url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{identifier}/citations"
        else:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{identifier}/citations"
        params = {
            "fields": "paperId,title,year,citationCount,externalIds,abstract,authors,venue",
            "limit": min(limit, 1000),
        }
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 404:
            logger.debug(f"S2 citations not found for {id_type}:{identifier}")
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
            authors = [a.get("name") for a in citing.get("authors") or [] if a.get("name")]

            result.append({
                "doi": citing_doi,
                "s2_id": citing.get("paperId"),
                "title": citing.get("title"),
                "year": citing.get("year"),
                "cited_by_count": citing.get("citationCount") or 0,
                "abstract": citing.get("abstract"),
                "authors": authors,
                "venue": citing.get("venue") or None,
            })

        logger.info(f"S2 citations: {len(result)} papers for {id_type}:{identifier}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch S2 citations for {id_type}:{identifier}: {e}")
        return []


def _fetch_references_s2(identifier: str, limit: int = 50, id_type: str = "DOI") -> List[Dict[str, Any]]:
    """Fetch papers that a given work cites from Semantic Scholar.

    Supports multiple identifier types: DOI, S2 paper ID, ArXiv ID.
    Returns list of dicts with: doi, s2_id, title, year, cited_by_count
    """
    if not identifier:
        return []

    headers = get_s2_headers()

    try:
        if id_type == "S2":
            url = f"https://api.semanticscholar.org/graph/v1/paper/{identifier}/references"
        elif id_type == "ArXiv":
            url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{identifier}/references"
        else:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{identifier}/references"
        params = {
            "fields": "paperId,title,year,citationCount,externalIds,abstract,authors,venue",
            "limit": min(limit, 1000),
        }
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 404:
            logger.debug(f"S2 references not found for {id_type}:{identifier}")
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
            authors = [a.get("name") for a in ref.get("authors") or [] if a.get("name")]

            result.append({
                "doi": ref_doi,
                "s2_id": ref.get("paperId"),
                "title": ref.get("title"),
                "year": ref.get("year"),
                "cited_by_count": ref.get("citationCount") or 0,
                "abstract": ref.get("abstract"),
                "authors": authors,
                "venue": ref.get("venue") or None,
            })

        logger.info(f"S2 references: {len(result)} papers for {id_type}:{identifier}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch S2 references for {id_type}:{identifier}: {e}")
        return []


# ============================================================================
# Citation Fetching (adapted from paper_cache.py)
# ============================================================================

def _backfill_title_from_s2(doi: Optional[str]) -> Optional[str]:
    """Fetch title from Semantic Scholar when OpenAlex has empty title."""
    if not doi:
        return None

    headers = get_s2_headers()

    try:
        clean_doi = doi.replace("https://doi.org/", "")
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}"
        params = {"fields": "title"}
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=10)
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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
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

            doi_raw = w.get("doi")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            authors = [a["author"]["display_name"]
                       for a in w.get("authorships", [])
                       if a.get("author", {}).get("display_name")]
            venue = (w.get("primary_location") or {}).get("source", {})
            venue_name = venue.get("display_name") if isinstance(venue, dict) else None
            is_oa = (w.get("open_access") or {}).get("is_oa")

            result.append({
                "work_id": wid,
                "title": title,
                "year": year,
                "cited_by_count": cited_by_count,
                "abstract": _extract_abstract(w),
                "doi": doi,
                "authors": authors,
                "venue": venue_name,
                "is_open_access": is_oa,
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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
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
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
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

            doi_raw = w.get("doi")
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

            authors = [a["author"]["display_name"]
                       for a in w.get("authorships", [])
                       if a.get("author", {}).get("display_name")]
            venue = (w.get("primary_location") or {}).get("source", {})
            venue_name = venue.get("display_name") if isinstance(venue, dict) else None
            is_oa = (w.get("open_access") or {}).get("is_oa")

            result.append({
                "work_id": wid,
                "title": title,
                "year": year,
                "cited_by_count": cited_by_count,
                "abstract": _extract_abstract(w),
                "doi": doi,
                "authors": authors,
                "venue": venue_name,
                "is_open_access": is_oa,
            })

        # Sort by citation count to get most influential references
        result.sort(key=lambda x: x.get("cited_by_count") or 0, reverse=True)

        logger.info(f"Fetched {len(result)} references for {work_id}")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch references for {work_id}: {e}")
        return []


def _fetch_s2_paper_details(s2_id: str, work_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper details from Semantic Scholar by S2 paper ID."""
    headers = get_s2_headers()
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}"
        params = {"fields": "paperId,title,year,citationCount,abstract,externalIds,authors,venue"}
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        authors = [a.get("name") for a in data.get("authors") or [] if a.get("name")]
        external_ids = data.get("externalIds") or {}
        doi = external_ids.get("DOI")
        return {
            "work_id": work_id,
            "title": data.get("title"),
            "year": data.get("year"),
            "cited_by_count": data.get("citationCount") or 0,
            "abstract": data.get("abstract"),
            "authors": authors,
            "venue": data.get("venue") or None,
            "doi": doi,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch S2 paper details for {s2_id}: {e}")
        return None


def _fetch_arxiv_paper_details(arxiv_id: str, work_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper details for an ArXiv paper.

    Uses S2's ArXiv bridge first (has citations + abstract, 100 req/min with key).
    Falls back to ArXiv Atom API only if S2 fails (strict 1 req/3s rate limit).
    """
    # Primary: S2 ArXiv bridge (fast, has citation counts and abstracts)
    headers = get_s2_headers()
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
        params = {"fields": "paperId,title,year,citationCount,abstract,externalIds,authors,venue"}
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            authors = [a.get("name") for a in data.get("authors") or [] if a.get("name")]
            return {
                "work_id": work_id,
                "title": data.get("title"),
                "year": data.get("year"),
                "cited_by_count": data.get("citationCount") or 0,
                "abstract": data.get("abstract"),
                "authors": authors,
                "venue": data.get("venue") or None,
                "doi": f"10.48550/arXiv.{arxiv_id}",
            }
    except Exception as e:
        logger.debug(f"S2 ArXiv bridge failed for {arxiv_id}: {e}")

    # Fallback: ArXiv Atom API (no citations, no abstract, 1 req/3s limit)
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        _arxiv_throttle()
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else None
            published = entry.find("atom:published", ns)
            year = None
            if published is not None and published.text:
                try:
                    year = int(published.text[:4])
                except (ValueError, TypeError):
                    pass
            return {
                "work_id": work_id,
                "title": title,
                "year": year,
                "cited_by_count": 0,
                "abstract": None,
                "doi": f"10.48550/arXiv.{arxiv_id}",
            }
    except Exception as e:
        logger.warning(f"ArXiv API fallback failed for {arxiv_id}: {e}")
    return None


def fetch_seed_paper_details(work_id: str) -> Optional[Dict[str, Any]]:
    """Fetch details for a single paper from any source (OpenAlex, S2, ArXiv)."""
    if not work_id:
        return None

    # S2 paper
    if work_id.startswith("S2:"):
        return _fetch_s2_paper_details(work_id[3:], work_id)

    # ArXiv paper
    if work_id.startswith("AX:"):
        return _fetch_arxiv_paper_details(work_id[3:], work_id)

    # OpenAlex paper (W prefix or legacy)
    try:
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        _oa_throttle()
        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        title = data.get("title")

        # Backfill empty titles from S2
        if not title:
            doi = data.get("doi")
            title = _backfill_title_from_s2(doi)

        doi_raw = data.get("doi")
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
        authors = [a["author"]["display_name"]
                   for a in data.get("authorships", [])
                   if a.get("author", {}).get("display_name")]
        venue = (data.get("primary_location") or {}).get("source", {})
        venue_name = venue.get("display_name") if isinstance(venue, dict) else None
        is_oa = (data.get("open_access") or {}).get("is_oa")

        return {
            "work_id": work_id,
            "title": title,
            "year": data.get("publication_year"),
            "cited_by_count": data.get("cited_by_count") or 0,
            "abstract": _extract_abstract(data),
            "doi": doi,
            "authors": authors,
            "venue": venue_name,
            "is_open_access": is_oa,
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

# Stopwords for keyword overlap sanity check
_SEED_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "about", "above", "after", "again", "also", "any", "because",
    "before", "between", "during", "into", "new", "over", "own", "through", "under",
    "using", "based", "via", "approach", "method", "methods", "model", "models",
    "system", "systems", "paper", "study", "analysis", "results", "data",
    "learning", "deep", "neural", "network", "networks", "machine",
})


def _keyword_overlap_score(query: str, title: str, abstract: str = "") -> float:
    """Compute keyword overlap between query and a paper's title+abstract.

    Returns a score 0.0-1.0: fraction of query keywords found in the paper text.
    Used as a deterministic sanity check — catches LLM hallucinations where a paper
    from a completely unrelated field is rated HIGH.
    """
    import re
    query_words = set(re.findall(r'[a-z]{3,}', query.lower())) - _SEED_STOPWORDS
    if not query_words:
        return 1.0  # Can't check, assume ok

    paper_text = f"{title} {abstract}".lower()
    paper_words = set(re.findall(r'[a-z]{3,}', paper_text))

    matches = query_words & paper_words
    return len(matches) / len(query_words)


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

def _merge_s2_citations(
    s2_papers: List[Dict[str, Any]],
    existing_work_ids: set,
) -> List[Dict[str, Any]]:
    """Merge S2 citation results into the graph.

    All 3 sources are equal peers. Papers are mapped to OpenAlex IDs when possible
    via DOI bridge, but S2-only papers are included with S2: prefix.

    Returns list of paper dicts with work_ids (OpenAlex or S2: prefixed).
    """
    if not s2_papers:
        return []

    # Collect DOIs for batch mapping
    dois_to_lookup = []
    for p in s2_papers:
        doi = p.get("doi")
        if doi:
            clean_doi = doi.replace("https://doi.org/", "")
            dois_to_lookup.append(clean_doi)

    # Batch lookup DOIs -> OpenAlex IDs
    doi_to_workid = _batch_lookup_openalex_by_dois(dois_to_lookup) if dois_to_lookup else {}

    result = []
    mapped_count = 0
    s2_only_count = 0
    for p in s2_papers:
        doi = p.get("doi")
        s2_id = p.get("s2_id")
        clean_doi = doi.replace("https://doi.org/", "") if doi else None

        # Try OpenAlex mapping first
        work_id = doi_to_workid.get(clean_doi) if clean_doi else None

        if work_id:
            mapped_count += 1
        elif s2_id:
            # Keep as S2-only paper — sources are equal peers
            work_id = f"S2:{s2_id}"
            s2_only_count += 1
        else:
            continue  # No usable identifier

        if work_id not in existing_work_ids:
            result.append({
                "work_id": work_id,
                "title": p.get("title"),
                "year": p.get("year"),
                "cited_by_count": p.get("cited_by_count") or 0,
                "abstract": p.get("abstract"),
                "authors": p.get("authors") or [],
                "venue": p.get("venue"),
                "doi": clean_doi,
                "source": "semantic_scholar",
            })

    logger.info(f"S2 merge: {len(s2_papers)} S2 papers -> {mapped_count} mapped to OA + {s2_only_count} S2-only")
    return result


def _batch_resolve_s2_ids(papers_to_resolve: List[Dict[str, Any]]) -> Dict[str, str]:
    """Batch-resolve S2 paper IDs for multiple papers using POST /paper/batch.

    Takes a list of dicts with "work_id", "doi", "title" keys.
    Returns a dict mapping work_id -> S2 paper ID.

    Uses S2's batch endpoint (POST /paper/batch, up to 500 papers, 1 RPS)
    to resolve DOIs to S2 paper IDs in a single call instead of N individual calls.
    """
    if not papers_to_resolve:
        return {}

    headers = {"Content-Type": "application/json", **get_s2_headers()}

    result: Dict[str, str] = {}

    # Separate papers by identifier type
    doi_papers = []  # (work_id, doi)
    title_papers = []  # (work_id, title) — fallback for papers without DOI

    for p in papers_to_resolve:
        wid = p["work_id"]
        # S2: papers already have their S2 ID
        if wid.startswith("S2:"):
            result[wid] = wid[3:]
            continue

        doi = p.get("doi")
        if doi:
            clean_doi = doi.replace("https://doi.org/", "")
            doi_papers.append((wid, clean_doi))
        elif p.get("title"):
            title_papers.append((wid, p["title"]))

    # Batch resolve DOIs -> S2 paper IDs via POST /paper/batch
    if doi_papers:
        batch_ids = [f"DOI:{doi}" for _, doi in doi_papers]

        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            payload = {"ids": batch_ids}
            params = {"fields": "paperId"}

            # POST with retry on 429
            last_resp = None
            for attempt in range(S2_MAX_RETRIES + 1):
                _s2_throttle()
                resp = requests.post(url, json=payload, params=params, headers=headers, timeout=30)
                if resp.status_code != 429:
                    break
                last_resp = resp
                if attempt < S2_MAX_RETRIES:
                    delay = S2_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.info(f"S2 batch rate limited (429), retry {attempt + 1}/{S2_MAX_RETRIES} after {delay}s")
                    time.sleep(delay)
            else:
                resp = last_resp

            if resp.status_code == 200:
                data = resp.json()
                # Response is a list in the same order as input IDs
                # Entries can be null if paper not found
                for i, paper_data in enumerate(data):
                    if paper_data and isinstance(paper_data, dict):
                        s2_id = paper_data.get("paperId")
                        if s2_id:
                            wid = doi_papers[i][0]
                            result[wid] = s2_id
            else:
                logger.warning(f"S2 batch resolve failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"S2 batch resolve error: {e}")

    # Fallback: title search for papers without DOI (individual calls — unavoidable)
    for wid, title in title_papers:
        if wid not in result:
            s2_id = _get_s2_paper_id(title=title)
            if s2_id:
                result[wid] = s2_id

    logger.info(f"S2 batch resolve: {len(papers_to_resolve)} papers -> {len(result)} S2 IDs resolved")
    return result


def _backfill_metadata_cross_source(papers: Dict[str, Dict[str, Any]]) -> int:
    """Cross-reference metadata backfill between OpenAlex and Semantic Scholar.

    For papers missing abstract/authors/venue: batch-fetch from S2 via DOI.
    For S2-only papers missing metadata: individual S2 fetch.

    Modifies papers dict in-place. Returns count of fields filled.
    """
    # Collect papers missing any metadata that have a DOI we can look up in S2
    needs_backfill = []  # (work_id, doi)
    for wid, paper in papers.items():
        missing_abstract = not paper.get("abstract")
        missing_authors = not paper.get("authors")
        missing_venue = not paper.get("venue")
        if (missing_abstract or missing_authors or missing_venue):
            doi = paper.get("doi")
            if doi:
                needs_backfill.append((wid, doi))

    filled = 0

    # Batch fetch from S2 (up to 500 per call, like F2)
    if needs_backfill:
        headers = {"Content-Type": "application/json", **get_s2_headers()}

        BATCH_SIZE = 500
        for i in range(0, len(needs_backfill), BATCH_SIZE):
            batch = needs_backfill[i:i + BATCH_SIZE]
            s2_ids = [f"DOI:{doi}" for _, doi in batch]
            try:
                _s2_throttle()
                resp = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    json={"ids": s2_ids},
                    params={"fields": "abstract,authors,venue"},
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code != 200:
                    logger.warning(f"S2 metadata backfill failed: HTTP {resp.status_code}")
                    continue
                data = resp.json()
                for j, paper_data in enumerate(data):
                    if not paper_data or not isinstance(paper_data, dict):
                        continue
                    wid = batch[j][0]
                    paper = papers[wid]
                    # Backfill abstract
                    if not paper.get("abstract"):
                        abstract = paper_data.get("abstract")
                        if abstract:
                            paper["abstract"] = abstract
                            filled += 1
                    # Backfill authors
                    if not paper.get("authors"):
                        s2_authors = [a.get("name") for a in paper_data.get("authors") or [] if a.get("name")]
                        if s2_authors:
                            paper["authors"] = s2_authors
                            filled += 1
                    # Backfill venue
                    if not paper.get("venue"):
                        s2_venue = paper_data.get("venue") or None
                        if s2_venue:
                            paper["venue"] = s2_venue
                            filled += 1
            except Exception as e:
                logger.warning(f"S2 metadata backfill error: {e}")

    # For S2-only papers missing metadata, batch fetch from S2
    s2_missing = [wid for wid, p in papers.items()
                  if wid.startswith("S2:") and (not p.get("abstract") or not p.get("authors") or not p.get("venue"))]
    if s2_missing:
        s2_batch_ids = [wid[3:] for wid in s2_missing]  # raw S2 paper IDs
        try:
            _s2_throttle()
            resp = requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                json={"ids": s2_batch_ids},
                params={"fields": "abstract,authors,venue"},
                headers={"Content-Type": "application/json", **get_s2_headers()},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                for j, paper_data in enumerate(data):
                    if not paper_data or not isinstance(paper_data, dict):
                        continue
                    wid = s2_missing[j]
                    paper = papers[wid]
                    if not paper.get("abstract"):
                        abstract = paper_data.get("abstract")
                        if abstract:
                            paper["abstract"] = abstract
                            filled += 1
                    if not paper.get("authors"):
                        s2_authors = [a.get("name") for a in paper_data.get("authors") or [] if a.get("name")]
                        if s2_authors:
                            paper["authors"] = s2_authors
                            filled += 1
                    if not paper.get("venue"):
                        s2_venue = paper_data.get("venue") or None
                        if s2_venue:
                            paper["venue"] = s2_venue
                            filled += 1
            else:
                logger.warning(f"S2 batch backfill for S2-only papers failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"S2 batch backfill error: {e}")

    if filled > 0:
        logger.info(f"Metadata cross-reference backfill: filled {filled} missing fields")
    return filled


def _expand_citation_network(
    seed_work_id: str,
    total_limit: int = 30,
    hop1_fetch: int = 50,
    hop2_fetch: int = 20,
    citation_weight: float = 0.6,
    connectivity_weight: float = 0.4,
    structured_query: Optional[Dict[str, Any]] = None,
    expansion: Optional[str] = None,
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
        if from_id == to_id:
            return  # Skip self-loops
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

    # Resolve identifiers for all sources — all 3 sources are EQUAL peers
    seed_doi = _get_doi_for_work(seed_work_id) if seed_work_id.startswith("W") else None
    seed_title = papers[seed_work_id].get("title")

    # Get S2 paper ID — extract directly from prefix when available
    if seed_work_id.startswith("S2:"):
        seed_s2_id = seed_work_id[3:]
    elif seed_work_id.startswith("AX:"):
        # ArXiv seed: resolve via S2 ArXiv bridge to get S2 paper ID
        arxiv_id = seed_work_id[3:]
        seed_s2_id = None
        _s2_hdrs = get_s2_headers()
        try:
            _ax_resp = _s2_get_with_retry(
                f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}",
                params={"fields": "paperId"}, headers=_s2_hdrs, timeout=10,
            )
            if _ax_resp.status_code == 200:
                seed_s2_id = _ax_resp.json().get("paperId")
        except Exception:
            pass
        if not seed_s2_id:
            seed_s2_id = _get_s2_paper_id(doi=seed_doi, title=seed_title)
    else:
        # W prefix: resolve via DOI then title
        seed_s2_id = _get_s2_paper_id(doi=seed_doi, title=seed_title)
    logger.info(f"Seed identifiers — OA: {seed_work_id}, DOI: {seed_doi}, S2: {seed_s2_id}")

    # Hop 1: Get seed's direct connections from ALL sources in parallel
    hop1_citing = []
    hop1_refs = []

    # Source 1: OpenAlex (if we have an OpenAlex work_id)
    if seed_work_id.startswith("W"):
        hop1_citing = fetch_citing_papers(seed_work_id, limit=hop1_fetch, fetch_limit=hop1_fetch)
        hop1_refs = fetch_references(seed_work_id, limit=hop1_fetch, fetch_limit=hop1_fetch)

    # Source 2: Semantic Scholar (try S2 paper ID first, then DOI)
    s2_identifier = seed_s2_id or seed_doi
    s2_id_type = "S2" if seed_s2_id else "DOI"
    if s2_identifier:
        s2_citing = _fetch_citing_papers_s2(s2_identifier, limit=hop1_fetch, id_type=s2_id_type)
        s2_refs = _fetch_references_s2(s2_identifier, limit=hop1_fetch, id_type=s2_id_type)

        # Merge S2 results — includes both mapped-to-OA and S2-only papers
        s2_citing_mapped = _merge_s2_citations(
            s2_citing, set(papers.keys()) | {p.get("work_id") for p in hop1_citing}
        )
        s2_refs_mapped = _merge_s2_citations(
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

    # Apply hop filtering to hop-1 papers (structured query relevance)
    if structured_query and structured_query.get("topic") and expansion:
        try:
            from app.feature1.hop_filtering import filter_hop_papers
            hop1_list = [
                {**papers[wid], "work_id": wid}
                for wid in papers if papers[wid].get("hop") == 1
            ]
            filtered_hop1 = filter_hop_papers(hop1_list, structured_query, expansion, hop_level=1)
            filtered_wids = {p.get("work_id") for p in filtered_hop1}
            # Remove unfiltered hop-1 papers from graph
            removed = [wid for wid in list(papers.keys())
                        if papers[wid].get("hop") == 1 and wid not in filtered_wids]
            for wid in removed:
                del papers[wid]
            # Also remove edges involving removed papers
            edges = [(f, t) for f, t in edges if f not in removed and t not in removed]
            edge_set = set(edges)
            logger.info(f"Hop-1 filtering: kept {len(filtered_wids)}, removed {len(removed)}")
        except Exception as e:
            logger.warning(f"Hop-1 filtering failed: {e}")

    # Hop 2: Expand from top hop-1 papers (by citations)
    hop1_papers = [(wid, papers[wid]) for wid in papers if papers[wid].get("hop") == 1]
    hop1_papers.sort(key=lambda x: x[1].get("cited_by_count", 0), reverse=True)

    # Only expand from top N hop-1 papers to limit API calls
    top_hop1 = hop1_papers[:min(10, len(hop1_papers))]

    # Batch-resolve S2 paper IDs for all hop-1 papers at once
    # This replaces N individual _get_s2_paper_id calls with 1 batch call
    hop1_dois = {}  # wid -> doi (for W papers)
    papers_to_resolve = []
    for wid, paper_data in top_hop1:
        if wid.startswith("S2:"):
            # Already have S2 ID
            papers_to_resolve.append({"work_id": wid})
        elif wid.startswith("W"):
            doi = _get_doi_for_work(wid)
            hop1_dois[wid] = doi
            papers_to_resolve.append({
                "work_id": wid,
                "doi": doi,
                "title": paper_data.get("title"),
            })
        else:
            papers_to_resolve.append({
                "work_id": wid,
                "title": paper_data.get("title"),
            })

    # Single batch call to resolve all S2 IDs
    hop1_s2_ids = _batch_resolve_s2_ids(papers_to_resolve)

    def _expand_single_hop2(wid, paper_data):
        """Expand one hop-1 paper's citations+refs from OA and S2."""
        h2_citing = []
        h2_refs = []

        # OA
        if wid.startswith("W"):
            try:
                h2_citing.extend(fetch_citing_papers(wid, limit=hop2_fetch, fetch_limit=hop2_fetch))
            except Exception:
                pass
            try:
                h2_refs.extend(fetch_references(wid, limit=hop2_fetch, fetch_limit=hop2_fetch))
            except Exception:
                pass

        # S2
        hop1_s2_id = hop1_s2_ids.get(wid)
        hop1_doi = hop1_dois.get(wid)
        h2_s2_id = hop1_s2_id or hop1_doi
        h2_s2_type = "S2" if hop1_s2_id else "DOI"
        if h2_s2_id:
            try:
                s2_h2_citing = _fetch_citing_papers_s2(h2_s2_id, limit=hop2_fetch, id_type=h2_s2_type)
                s2_h2_citing_mapped = _merge_s2_citations(
                    s2_h2_citing, set(papers.keys()) | {p.get("work_id") for p in h2_citing}
                )
                h2_citing.extend(s2_h2_citing_mapped)
            except Exception:
                pass
            try:
                s2_h2_refs = _fetch_references_s2(h2_s2_id, limit=hop2_fetch, id_type=h2_s2_type)
                s2_h2_refs_mapped = _merge_s2_citations(
                    s2_h2_refs, set(papers.keys()) | {p.get("work_id") for p in h2_refs}
                )
                h2_refs.extend(s2_h2_refs_mapped)
            except Exception:
                pass

        return wid, h2_citing, h2_refs

    # Run hop-2 expansion in parallel (3 workers to leave S2 keys for other requests)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_expand_single_hop2, wid, pd): wid
            for wid, pd in top_hop1
        }
        for future in as_completed(futures):
            wid, h2_citing, h2_refs = future.result()
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

    # Apply hop filtering to hop-2 papers (structured query relevance)
    if structured_query and structured_query.get("topic") and expansion:
        try:
            from app.feature1.hop_filtering import filter_hop_papers
            hop2_list = [
                {**papers[wid], "work_id": wid}
                for wid in papers if papers[wid].get("hop") == 2
            ]
            filtered_hop2 = filter_hop_papers(hop2_list, structured_query, expansion, hop_level=2)
            filtered_wids = {p.get("work_id") for p in filtered_hop2}
            removed = [wid for wid in list(papers.keys())
                        if papers[wid].get("hop") == 2 and wid not in filtered_wids]
            for wid in removed:
                del papers[wid]
            edges = [(f, t) for f, t in edges if f not in removed and t not in removed]
            edge_set = set(edges)
            logger.info(f"Hop-2 filtering: kept {len(filtered_wids)}, removed {len(removed)}")
        except Exception as e:
            logger.warning(f"Hop-2 filtering failed: {e}")

    # Cross-reference metadata backfill: try S2 for papers missing abstracts/authors/venue
    _backfill_metadata_cross_source(papers)

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

    # Reserve slots for EMERGING papers (<100 citations) to ensure citation diversity
    # This ensures the graph has a mix of highly-cited and emerging/recent work
    emerging_quota = max(3, total_limit // 6)  # ~15% for emerging papers
    emerging_candidates = []
    for wid, p in papers.items():
        if wid in selected_ids or p.get("is_seed"):
            continue
        cites = p.get("cited_by_count", 0)
        hop = p.get("hop", 0)
        deg = degree.get(wid, 0)
        # Emerging: <100 citations, recent (2018+), has local connections
        year = p.get("year")
        if cites < 100 and deg >= 1 and hop <= 1:
            # Prefer recent papers with higher local connectivity
            emerging_score = deg * 2.0 + (0.5 if year and year >= 2020 else 0.0)
            emerging_candidates.append((wid, emerging_score))
        elif cites < 100 and deg >= 2 and hop == 2:
            emerging_score = deg * 1.5
            emerging_candidates.append((wid, emerging_score))

    emerging_candidates.sort(key=lambda x: -x[1])
    for wid, _ in emerging_candidates[:emerging_quota]:
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


def _expand_citation_network_v2(
    seed_work_ids: List[str],
    structured_query: Dict[str, Any],
    scope: str,
    drift: str,
    temporal: str,
    map_size: str,
    engine: Optional[Engine] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], Dict[str, Set]]:
    """V2 citation network expansion with score-and-select.

    1. Fetch hop-1 candidates (all citing + referenced) from OA + S2 for ALL seeds
    2. Score and filter by drift threshold
    3. Expand top hop-1 papers to hop-2
    4. Score and filter hop-2
    5. If drift=open and map_size=large: expand to hop-3
    6. Return all candidates + edges for greedy selection

    Returns
    -------
    tuple
        (candidates_list, edge_tuples, adjacency_map)
    """
    from app.feature1.candidate_scoring import (
        compute_relevance_score,
        get_drift_thresholds,
        get_fetch_limits,
    )

    drift_config = get_drift_thresholds(drift)
    fetch_config = get_fetch_limits(map_size)

    all_candidates: Dict[str, Dict[str, Any]] = {}  # work_id -> paper dict
    all_edges: Dict[str, Set] = {}  # work_id -> set of connected work_ids
    edge_tuples: List[Tuple[str, str]] = []
    edge_set: set = set()

    def add_edge(from_id: str, to_id: str):
        if from_id == to_id:
            return
        if (from_id, to_id) not in edge_set:
            edge_set.add((from_id, to_id))
            edge_tuples.append((from_id, to_id))
        # Build adjacency map (bidirectional)
        all_edges.setdefault(from_id, set()).add(to_id)
        all_edges.setdefault(to_id, set()).add(from_id)

    # Register all seeds
    for seed_work_id in seed_work_ids:
        seed_data = fetch_seed_paper_details(seed_work_id)
        if seed_data:
            all_candidates[seed_work_id] = {**seed_data, "hop": 0, "is_seed": True}
        else:
            all_candidates[seed_work_id] = {
                "work_id": seed_work_id,
                "title": None,
                "year": None,
                "cited_by_count": 0,
                "hop": 0,
                "is_seed": True,
            }

    hop1_fetch_per_seed = fetch_config["hop1_fetch"] // len(seed_work_ids)

    # ── HOP 1: Fetch citing + referenced for ALL seeds ──────────────────
    for seed_work_id in seed_work_ids:
        seed_doi = _get_doi_for_work(seed_work_id) if seed_work_id.startswith("W") else None
        seed_title = all_candidates[seed_work_id].get("title")

        if seed_work_id.startswith("S2:"):
            seed_s2_id = seed_work_id[3:]
        elif seed_work_id.startswith("AX:"):
            arxiv_id = seed_work_id[3:]
            seed_s2_id = None
            _s2_hdrs = get_s2_headers()
            try:
                _ax_resp = _s2_get_with_retry(
                    f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}",
                    params={"fields": "paperId"}, headers=_s2_hdrs, timeout=10,
                )
                if _ax_resp.status_code == 200:
                    seed_s2_id = _ax_resp.json().get("paperId")
            except Exception:
                pass
            if not seed_s2_id:
                seed_s2_id = _get_s2_paper_id(doi=seed_doi, title=seed_title)
        else:
            seed_s2_id = _get_s2_paper_id(doi=seed_doi, title=seed_title)

        logger.info(f"V2 expansion — seed: OA={seed_work_id}, DOI={seed_doi}, S2={seed_s2_id}")

        hop1_citing = []
        hop1_refs = []

        # Source 1: OpenAlex
        if seed_work_id.startswith("W"):
            hop1_citing = fetch_citing_papers(seed_work_id, limit=hop1_fetch_per_seed, fetch_limit=hop1_fetch_per_seed)
            hop1_refs = fetch_references(seed_work_id, limit=hop1_fetch_per_seed, fetch_limit=hop1_fetch_per_seed)

        # Source 2: Semantic Scholar
        s2_identifier = seed_s2_id or seed_doi
        s2_id_type = "S2" if seed_s2_id else "DOI"
        if s2_identifier:
            s2_citing = _fetch_citing_papers_s2(s2_identifier, limit=hop1_fetch_per_seed, id_type=s2_id_type)
            s2_refs = _fetch_references_s2(s2_identifier, limit=hop1_fetch_per_seed, id_type=s2_id_type)

            s2_citing_mapped = _merge_s2_citations(
                s2_citing, set(all_candidates.keys()) | {p.get("work_id") for p in hop1_citing}
            )
            s2_refs_mapped = _merge_s2_citations(
                s2_refs, set(all_candidates.keys()) | {p.get("work_id") for p in hop1_refs}
            )

            hop1_citing.extend(s2_citing_mapped)
            hop1_refs.extend(s2_refs_mapped)

        # Register hop-1 papers and edges
        for p in hop1_citing:
            wid = p.get("work_id")
            if wid and wid not in all_candidates:
                all_candidates[wid] = {**p, "hop": 1, "is_seed": False}
            if wid:
                add_edge(wid, seed_work_id)  # citing paper -> seed

        for p in hop1_refs:
            wid = p.get("work_id")
            if wid and wid not in all_candidates:
                all_candidates[wid] = {**p, "hop": 1, "is_seed": False}
            if wid:
                add_edge(seed_work_id, wid)  # seed -> reference

    # Add implicit edges between seeds if they share citations
    if len(seed_work_ids) > 1:
        for i, sid1 in enumerate(seed_work_ids):
            for sid2 in seed_work_ids[i + 1:]:
                neighbors_1 = all_edges.get(sid1, set())
                neighbors_2 = all_edges.get(sid2, set())
                if neighbors_1 & neighbors_2:
                    add_edge(sid1, sid2)

    # Filter hop-1 by drift threshold
    hop1_filtered = []
    for wid, p in list(all_candidates.items()):
        if p.get("hop") != 1:
            continue
        score = compute_relevance_score(p, structured_query, scope)
        if score >= drift_config["hop1_threshold"]:
            p["_relevance_score"] = score
            hop1_filtered.append(p)

    logger.info(
        f"V2 Hop-1: {sum(1 for p in all_candidates.values() if p.get('hop') == 1)} fetched, "
        f"{len(hop1_filtered)} passed threshold {drift_config['hop1_threshold']}"
    )

    # ── RETRIEVAL ENRICHMENT: LLM-expanded multi-source search ────────
    # Discover relevant papers not in the seed's citation neighborhood.
    # Enrichment candidates enter the same pool as hop-1 papers and are
    # scored by greedy_graph_select. Connectivity weight (0.35) ensures
    # only papers that actually connect to the graph are selected.
    try:
        from app.feature2.query_expansion import expand_query
        from app.feature2.retrieval import _search_openalex, _search_semantic_scholar

        # Build query text from structured query
        query_text = structured_query.get("topic", "")
        _enrich_domain = structured_query.get("domain")
        if _enrich_domain:
            query_text = f"{query_text} {_enrich_domain}"

        if query_text.strip() and engine is not None:
            # Get LLM-expanded concepts (cached)
            with engine.connect() as exp_conn:
                expansion = expand_query(exp_conn, query_text)

            # Build search queries from expansion
            search_queries = [query_text]
            foundational_titles = []

            if expansion.concepts:
                for concept in expansion.concepts:
                    for syn in concept.synonyms[:2]:
                        if syn and syn.lower() != query_text.lower() and len(syn) > 2:
                            search_queries.append(syn)
                    for fw in getattr(concept, "foundational_works", []):
                        if fw and len(fw) > 5 and fw.lower() != query_text.lower():
                            foundational_titles.append(fw)

            MAX_ENRICHMENT_QUERIES = 4
            search_queries = search_queries[:MAX_ENRICHMENT_QUERIES]

            logger.info(
                f"V2 retrieval enrichment: {len(search_queries)} queries, "
                f"{len(foundational_titles)} foundational titles"
            )

            # Run searches in parallel
            enrichment_oa_results = []
            enrichment_s2_results = []

            def _oa_enrich(q, limit=30):
                try:
                    return _search_openalex(q, limit)
                except Exception:
                    return []

            def _s2_enrich():
                try:
                    return _search_semantic_scholar(query_text, 50)
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=4) as enrich_executor:
                oa_futures = []
                for q in search_queries:
                    oa_futures.append(enrich_executor.submit(_oa_enrich, q, 30))
                for fw in foundational_titles[:5]:
                    oa_futures.append(enrich_executor.submit(_oa_enrich, fw, 5))

                s2_future = enrich_executor.submit(_s2_enrich)

                for f in as_completed(oa_futures):
                    enrichment_oa_results.extend(f.result())
                enrichment_s2_results = s2_future.result()

            # Process OA results: (work_id, score)
            enrichment_count = 0
            seed_ids_set = set(seed_work_ids)
            for wid, _score in enrichment_oa_results:
                if not wid or wid in all_candidates or wid in seed_ids_set:
                    continue
                all_candidates[wid] = {
                    "work_id": wid,
                    "title": None,
                    "year": None,
                    "cited_by_count": 0,
                    "hop": 1,
                    "is_seed": False,
                    "_source": "retrieval_enrichment",
                }
                enrichment_count += 1

            # Process S2 results: (s2_paper_id, metadata_dict)
            for s2_id, meta in enrichment_s2_results:
                oa_id = meta.get("openalex_id")
                wid = oa_id if oa_id else f"S2:{s2_id}"
                if not wid or wid in all_candidates or wid in seed_ids_set:
                    continue
                all_candidates[wid] = {
                    "work_id": wid,
                    "title": meta.get("title"),
                    "year": meta.get("year"),
                    "cited_by_count": meta.get("citations", 0),
                    "abstract": meta.get("abstract"),
                    "doi": meta.get("doi"),
                    "venue": meta.get("venue"),
                    "hop": 1,
                    "is_seed": False,
                    "_source": "retrieval_enrichment",
                }
                enrichment_count += 1

            logger.info(f"V2 retrieval enrichment: added {enrichment_count} new candidates")

            # Edge discovery: check which enrichment papers connect to graph
            enrichment_papers = [
                p for p in all_candidates.values()
                if p.get("_source") == "retrieval_enrichment"
            ]
            existing_wids = set(
                wid for wid, p in all_candidates.items()
                if p.get("_source") != "retrieval_enrichment"
            )

            # Only check papers that pass drift threshold
            enrichment_to_check = []
            for p in enrichment_papers:
                score = compute_relevance_score(p, structured_query, scope)
                if score >= drift_config["hop1_threshold"]:
                    p["_relevance_score"] = score
                    enrichment_to_check.append(p)

            def _check_enrichment_edges(wid):
                """Check if enrichment paper connects to existing graph."""
                found = []
                try:
                    refs = fetch_references(wid, limit=100, fetch_limit=100)
                    for r in refs:
                        rid = r.get("work_id")
                        if rid and rid in existing_wids:
                            found.append((wid, rid))
                except Exception:
                    pass
                try:
                    citing = fetch_citing_papers(wid, limit=100, fetch_limit=100)
                    for c in citing:
                        cid = c.get("work_id")
                        if cid and cid in existing_wids:
                            found.append((cid, wid))
                except Exception:
                    pass
                return found

            # Cap at 20 edge-discovery calls to limit API usage
            with ThreadPoolExecutor(max_workers=3) as edge_executor:
                edge_futures = {
                    edge_executor.submit(_check_enrichment_edges, p["work_id"]): p["work_id"]
                    for p in enrichment_to_check[:20]
                    if p["work_id"].startswith("W")  # Only OA IDs have fetchable refs
                }
                for future in as_completed(edge_futures):
                    for src, tgt in future.result():
                        add_edge(src, tgt)

            # Score enrichment papers that passed threshold and add to hop1_filtered
            for p in enrichment_to_check:
                if p["work_id"] in all_edges:
                    hop1_filtered.append(p)

            enrichment_connected = sum(
                1 for p in enrichment_to_check if p["work_id"] in all_edges
            )
            logger.info(
                f"V2 enrichment: {len(enrichment_to_check)} passed threshold, "
                f"{enrichment_connected} have graph connections"
            )

    except Exception as e:
        logger.warning(f"V2 retrieval enrichment failed (non-fatal): {e}")

    # ── HOP 2: Expand top hop-1 papers ───────────────────────────────────
    hop1_sorted = sorted(hop1_filtered, key=lambda p: p.get("_relevance_score", 0), reverse=True)
    hop2_sources = hop1_sorted[:fetch_config["hop2_expand_count"]]

    # Batch-resolve S2 paper IDs for hop-1 expansion sources
    hop1_dois = {}
    papers_to_resolve = []
    for p in hop2_sources:
        wid = p.get("work_id", "")
        if wid.startswith("S2:"):
            papers_to_resolve.append({"work_id": wid})
        elif wid.startswith("W"):
            doi = _get_doi_for_work(wid)
            hop1_dois[wid] = doi
            papers_to_resolve.append({
                "work_id": wid,
                "doi": doi,
                "title": p.get("title"),
            })
        else:
            papers_to_resolve.append({
                "work_id": wid,
                "title": p.get("title"),
            })

    hop1_s2_ids = _batch_resolve_s2_ids(papers_to_resolve)

    hop2_fetch = fetch_config["hop2_fetch"]

    def _expand_single_hop2_v2(wid, paper_data):
        """Expand one hop-1 paper's citations+refs from OA and S2."""
        h2_citing = []
        h2_refs = []

        if wid.startswith("W"):
            try:
                h2_citing.extend(fetch_citing_papers(wid, limit=hop2_fetch, fetch_limit=hop2_fetch))
            except Exception:
                pass
            try:
                h2_refs.extend(fetch_references(wid, limit=hop2_fetch, fetch_limit=hop2_fetch))
            except Exception:
                pass

        hop1_s2_id = hop1_s2_ids.get(wid)
        hop1_doi = hop1_dois.get(wid)
        h2_s2_id = hop1_s2_id or hop1_doi
        h2_s2_type = "S2" if hop1_s2_id else "DOI"
        if h2_s2_id:
            try:
                s2_h2_citing = _fetch_citing_papers_s2(h2_s2_id, limit=hop2_fetch, id_type=h2_s2_type)
                s2_h2_citing_mapped = _merge_s2_citations(
                    s2_h2_citing, set(all_candidates.keys()) | {p.get("work_id") for p in h2_citing}
                )
                h2_citing.extend(s2_h2_citing_mapped)
            except Exception:
                pass
            try:
                s2_h2_refs = _fetch_references_s2(h2_s2_id, limit=hop2_fetch, id_type=h2_s2_type)
                s2_h2_refs_mapped = _merge_s2_citations(
                    s2_h2_refs, set(all_candidates.keys()) | {p.get("work_id") for p in h2_refs}
                )
                h2_refs.extend(s2_h2_refs_mapped)
            except Exception:
                pass

        return wid, h2_citing, h2_refs

    # Run hop-2 expansion in parallel
    hop2_papers_raw = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_expand_single_hop2_v2, p.get("work_id", ""), p): p.get("work_id")
            for p in hop2_sources
        }
        seed_ids_set = set(seed_work_ids)
        for future in as_completed(futures):
            source_wid, h2_citing, h2_refs = future.result()
            for p in h2_citing:
                pid = p.get("work_id")
                if pid and pid not in all_candidates and pid not in seed_ids_set:
                    hop2_papers_raw.append(p)
                    all_candidates[pid] = {**p, "hop": 2, "is_seed": False}
                if pid:
                    add_edge(pid, source_wid)
            for p in h2_refs:
                pid = p.get("work_id")
                if pid and pid not in all_candidates and pid not in seed_ids_set:
                    hop2_papers_raw.append(p)
                    all_candidates[pid] = {**p, "hop": 2, "is_seed": False}
                if pid:
                    add_edge(source_wid, pid)

    # Deduplicate hop-2 by work_id (Fix 2: inline dict dedup)
    hop2_seen: Dict[str, Dict[str, Any]] = {}
    for p in hop2_papers_raw:
        hop2_seen.setdefault(p.get("work_id", ""), p)
    hop2_papers = list(hop2_seen.values())

    # Filter hop-2 by drift threshold
    hop2_filtered = []
    for p in hop2_papers:
        score = compute_relevance_score(p, structured_query, scope)
        if score >= drift_config["hop2_threshold"]:
            p["_relevance_score"] = score
            hop2_filtered.append(p)

    logger.info(
        f"V2 Hop-2: {len(hop2_papers)} fetched, {len(hop2_filtered)} passed "
        f"threshold {drift_config['hop2_threshold']}"
    )

    # ── HOP 3 (only for open + large) ────────────────────────────────────
    if drift_config["max_hops"] >= 3 and map_size == "large":
        hop2_sorted = sorted(hop2_filtered, key=lambda p: p.get("_relevance_score", 0), reverse=True)
        hop3_sources = hop2_sorted[:5]

        hop3_papers_raw = []
        for source_paper in hop3_sources:
            source_wid = source_paper.get("work_id", "")
            try:
                h3_citing = fetch_citing_papers(source_wid, limit=15, fetch_limit=15)
            except Exception:
                h3_citing = []
            try:
                h3_refs = fetch_references(source_wid, limit=15, fetch_limit=15)
            except Exception:
                h3_refs = []

            for p in h3_citing + h3_refs:
                wid = p.get("work_id")
                if wid and wid not in all_candidates and wid not in set(seed_work_ids):
                    hop3_papers_raw.append(p)
                    all_candidates[wid] = {**p, "hop": 3, "is_seed": False}
                if wid:
                    add_edge(source_wid, wid)

        # Deduplicate hop-3 (Fix 2)
        hop3_seen: Dict[str, Dict[str, Any]] = {}
        for p in hop3_papers_raw:
            hop3_seen.setdefault(p.get("work_id", ""), p)
        hop3_papers = list(hop3_seen.values())

        # Filter hop-3 by drift threshold
        for p in hop3_papers:
            score = compute_relevance_score(p, structured_query, scope)
            if score >= drift_config["hop3_threshold"]:
                p["_relevance_score"] = score

        logger.info(f"V2 Hop-3: {len(hop3_papers)} fetched")

    # Cross-reference metadata backfill
    _backfill_metadata_cross_source(all_candidates)

    # Return all candidates for greedy selection
    candidate_list = list(all_candidates.values())
    return candidate_list, edge_tuples, all_edges


def _score_seed_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 40,
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

    # Prepare papers for prompt — use short IDs (P0, P1...) to save tokens
    papers_for_prompt = []
    id_map: Dict[str, str] = {}  # "P0" -> work_id
    for i, c in enumerate(candidates_to_score):
        pid = f"P{i}"
        id_map[pid] = c.get("work_id", "")
        entry: Dict[str, Any] = {
            "id": pid,
            "title": c.get("title", "")[:200],
        }
        if c.get("year"):
            entry["year"] = c["year"]
        if c.get("cited_by_count"):
            entry["citations"] = c["cited_by_count"]
        abstract = c.get("abstract")
        if abstract:
            entry["abstract"] = abstract[:300]
        papers_for_prompt.append(entry)

    papers_json = json.dumps(papers_for_prompt)

    prompt = f"""Classify papers by relevance to the query for SEED PAPER selection. Return ONLY a JSON object mapping paper_id to relevance tier.

A good seed paper DEFINES or PIONEERED the research area — it is the paper others cite when working on this topic.

STEP 0 — WRONG FIELD (check FIRST, overrides EVERYTHING):
If the paper is from a COMPLETELY DIFFERENT scientific discipline than the query, it is NONE. Period.
High citation count does NOT make a paper relevant. Famous papers from unrelated fields are NONE.
Examples:
- Crystallography/biology software → NONE for "quantum error correction"
- Protein structure paper → NONE for "deep learning optimization"
- Medical imaging paper → NONE for "graph theory algorithms"
Ask: does this paper share ANY core concepts, methods, or phenomena with the query? If no → NONE.

STEP 1 — ABOUT vs USES (apply SECOND, overrides tier assignment):
A paper that USES or EVALUATES a technique but is primarily ABOUT a different domain → LOW.
This is the #1 mistake to avoid. Check: what is this paper's PRIMARY contribution about?
- Paper uses ChatGPT but is about medical exams → LOW for "LLM agents"
- Paper uses ML but is about soil mapping → LOW for "ML climate modeling"
- Paper uses transformers but is about time-series forecasting → LOW for "transformers NLP"
- Healthcare/education paper that evaluates an AI tool → LOW for queries about the AI tool itself

STEP 2 — QUERY TYPE:
- SINGLE TOPIC ("CRISPR", "deep learning", "capsule networks"):
  Papers about any core aspect of this topic can score HIGH/ESSENTIAL.
- INTERSECTION QUERY ("ML drug discovery", "NLP legal text", "RL robotics"):
  Decompose into [METHOD/TECHNIQUE] + [DOMAIN/APPLICATION].
  A paper primarily about applying METHOD to DOMAIN → HIGH (this IS the intersection).
  A paper about METHOD only (no DOMAIN) → HIGH if it's foundational to the method, else MEDIUM.
  A paper about DOMAIN only (no METHOD) → MEDIUM at best.
  KEY: a paper that applies NLP to legal texts IS "NLP legal text" — don't reject it for being "just an application".

STEP 3 — TOOL vs TOPIC:
A paper about a tool/technique commonly USED IN a field but not ABOUT the field itself → MEDIUM at best.
- t-SNE visualizes embeddings but is about dimensionality reduction → MEDIUM for "embedding"
- ADAM optimizer is used in deep learning but is about optimization → MEDIUM for "deep learning"

STEP 4 — RELEVANCE TIER:
- ESSENTIAL: Seminal/foundational work that DEFINED this specific field (rare — 1-2 per query)
- HIGH: Directly about the query topic; paper's primary contribution IS this topic
- MEDIUM: Same field, useful context, but not primarily about the query topic
- LOW: Wrong domain, uses but isn't about the topic, or tangentially related
- NONE: Completely different field, unrelated discipline, no shared concepts

QUERY: {query}

PAPERS:
{papers_json}

OUTPUT a JSON object mapping each paper_id to its tier. No commentary, no reasoning, no extra fields. Example format:
{{"P0": "HIGH", "P3": "MEDIUM"}}"""

    client = Groq(api_key=api_key)
    # #region agent log
    _debug_log(
        "H9",
        "app/feature1/citation_map_service.py:_score_seed_candidates.start",
        "LLM scoring started",
        {"candidateCount": len(candidates_to_score), "groqTimeoutSec": GROQ_TIMEOUT},
    )
    # #endregion

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a research paper classifier. Output only valid JSON. No markdown, no commentary, no code blocks. Respond in English only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=8192,
                timeout=GROQ_TIMEOUT,
                response_format={"type": "json_object"},
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

            # Convert tiers to scores — map short IDs back to work_ids
            result = {}
            for pid, wid in id_map.items():
                tier = response_map.get(pid, "NONE").upper()
                if tier not in TIER_SCORES:
                    tier = "NONE"
                result[wid] = TIER_SCORES[tier]

            logger.info(f"LLM seed validation: scored {len(result)} candidates")
            # #region agent log
            _debug_log(
                "H9",
                "app/feature1/citation_map_service.py:_score_seed_candidates.success",
                "LLM scoring succeeded",
                {"attempt": attempt + 1, "scoredCount": len(result)},
            )
            # #endregion
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM seed validation JSON error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        except Exception as e:
            logger.warning(f"LLM seed validation error (attempt {attempt + 1}): {e}")
            # #region agent log
            _debug_log(
                "H9",
                "app/feature1/citation_map_service.py:_score_seed_candidates.error",
                "LLM scoring attempt failed",
                {"attempt": attempt + 1, "error": str(e)},
            )
            # #endregion
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

    # #region agent log
    _debug_log(
        "H9",
        "app/feature1/citation_map_service.py:_score_seed_candidates.fallback",
        "LLM scoring fell back to deterministic path",
        {"candidateCount": len(candidates_to_score)},
    )
    # #endregion
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

    headers = get_s2_headers()

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "fields": "paperId,title,year,citationCount,externalIds",
            "limit": limit,
        }
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)

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

    Uses ArXiv DOI format (10.48550/arXiv.XXXX) for reliable lookup.
    Falls back to S2 bridge (ArXiv -> S2 -> DOI -> OpenAlex).
    """
    if not arxiv_id:
        return None

    # Strategy 1: ArXiv DOI format (most reliable)
    arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
    result = _search_openalex_by_doi(arxiv_doi)
    if result:
        return result

    # Strategy 2: S2 bridge — look up ArXiv paper in S2, get DOI, then OpenAlex
    headers = get_s2_headers()

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
        params = {"fields": "paperId,title,year,citationCount,externalIds"}
        resp = _s2_get_with_retry(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            s2_paper = resp.json()
            external_ids = s2_paper.get("externalIds") or {}

            # Try DOI from S2
            doi = external_ids.get("DOI")
            if doi:
                oa_result = _search_openalex_by_doi(doi)
                if oa_result:
                    return oa_result

            # Try OpenAlex ID directly from S2 external IDs
            oa_id = external_ids.get("OpenAlex")
            if oa_id:
                return {
                    "work_id": oa_id,
                    "title": s2_paper.get("title"),
                    "year": s2_paper.get("year"),
                    "cited_by_count": s2_paper.get("citationCount") or 0,
                    "source": "arxiv_s2_bridge",
                }
    except Exception as e:
        logger.debug(f"S2 ArXiv bridge lookup failed for {arxiv_id}: {e}")

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

Output ONLY a JSON object with a "papers" array. No comments, reasoning, or alternatives inside values. Use exact paper titles.
{{"papers":[{{"title":"Paper Title Here","year":2017,"citations":50000}}]}}"""

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a research paper classifier. Output only valid JSON. No markdown, no commentary, no code blocks. Respond in English only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        papers = parsed.get("papers", [])
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

    # Strategy 4: S2-only fallback — keep papers that couldn't bridge to OA
    if not candidates and s2_results:
        for r in s2_results:
            s2_id = r.get("s2_id")
            if s2_id:
                candidates.append({
                    "work_id": f"S2:{s2_id}",
                    "title": r.get("title"),
                    "year": r.get("year"),
                    "cited_by_count": r.get("cited_by_count") or 0,
                    "source": "s2_only",
                })
        if candidates:
            logger.info(f"_find_paper_by_metadata: using {len(candidates)} S2-only papers (no OA bridge found)")

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
    oa_key = get_oa_api_key()
    if oa_key:
        params["api_key"] = oa_key

    try:
        _oa_throttle()
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


def select_seed_from_query(
    query: str,
    scope: str = "broad",
    structured_query: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select the best seed paper from a natural language query.

    Uses same retrieval sources as broad ranking:
    1. OpenAlex (relevance + highly-cited)
    2. Semantic Scholar bulk
    3. ArXiv
    4. Boosted searches for foundational papers (based on query keywords)

    Then validates topical relevance using LLM scoring:
    - Only papers with HIGH+ relevance (score >= 0.75) are considered
    - Among those, pick the highest cited
    - When scope=intersection, additionally filters to papers matching
      BOTH topic AND domain in title/abstract

    Returns:
        - seed_work_id (or None if not found)
        - selection_info dict
    """
    if not query:
        return None, {"selection_strategy": "none", "selection_reason": "Empty query"}

    # #region agent log
    _debug_log(
        "H10",
        "app/feature1/citation_map_service.py:select_seed_from_query.start",
        "Seed selection started",
        {"queryLength": len(query)},
    )
    # #endregion

    # Use dict to allow updating entries when better metadata is found
    papers_by_id: Dict[str, Dict[str, Any]] = {}
    llm_expanded_ids: set = set()  # Track papers from LLM expansion (priority candidates)

    # Generate targeted queries from structured decomposition
    from app.common.query_generation import generate_citation_map_queries
    map_focus = "core_cluster"
    if structured_query and structured_query.get("topic"):
        targeted_queries = generate_citation_map_queries(structured_query, map_focus)
    else:
        targeted_queries = [query]

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

        # Additional targeted queries from decomposition
        for tq in targeted_queries:
            if tq.lower().strip() != query.lower().strip():
                futures[executor.submit(_search_openalex, tq, OPENALEX_LIMIT)] = f"openalex_tq:{tq[:30]}"
                futures[executor.submit(_search_semantic_scholar, tq, S2_LIMIT)] = f"s2_tq:{tq[:30]}"

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
                for rank, paper in enumerate(papers):
                    if not paper:
                        continue
                    work_id = paper.get("work_id", "")
                    if not work_id:
                        continue

                    # Track LLM-expanded papers (priority candidates for scoring)
                    if source.startswith("llm_match:"):
                        llm_expanded_ids.add(work_id)

                    # Track API relevance rank (position in search results)
                    paper["api_rank"] = rank

                    # Smart dedup: prefer higher citation counts (S2 metadata override)
                    # This handles OpenAlex data corruption where famous papers have wrong metadata
                    existing = papers_by_id.get(work_id)
                    if existing:
                        # Keep best (lowest) API rank across sources
                        existing["api_rank"] = min(existing.get("api_rank", 999), rank)
                        existing_cites = existing.get("cited_by_count", 0)
                        new_cites = paper.get("cited_by_count", 0)
                        # Replace if new entry has significantly more citations (5x or 10K+ more)
                        if new_cites > existing_cites * 5 or new_cites > existing_cites + 10000:
                            logger.debug(
                                f"Updating paper metadata: {work_id} "
                                f"({existing_cites:,} -> {new_cites:,} cites)"
                            )
                            paper["api_rank"] = existing["api_rank"]  # preserve best rank
                            papers_by_id[work_id] = paper
                    else:
                        papers_by_id[work_id] = paper
            except Exception as e:
                logger.warning(f"Search {source} failed: {e}")

    all_papers = list(papers_by_id.values())
    # #region agent log
    _debug_log(
        "H10",
        "app/feature1/citation_map_service.py:select_seed_from_query.candidates",
        "Seed candidates collected",
        {"candidateCount": len(all_papers), "llmExpandedCount": len(llm_expanded_ids)},
    )
    # #endregion

    if not all_papers:
        return None, {
            "selection_strategy": "llm_validated_highest_cited",
            "selection_reason": "No papers found for query",
            "candidates_considered": 0,
        }

    # All 3 sources are equal — no source-based filtering
    openalex_papers = [p for p in all_papers if p.get("work_id")]
    if not openalex_papers:
        return None, {
            "selection_strategy": "llm_validated_highest_cited",
            "selection_reason": "No papers found in any source",
            "candidates_considered": 0,
        }

    # Build diversified scoring batch: LLM-expanded first, then interleave
    # by citations and API relevance rank to get both famous AND relevant papers
    llm_priority = [p for p in openalex_papers if p.get("work_id") in llm_expanded_ids]
    llm_priority.sort(key=lambda p: p.get("cited_by_count") or 0, reverse=True)
    others = [p for p in openalex_papers if p.get("work_id") not in llm_expanded_ids]

    # Interleave: top by citations + top by API relevance (deduped)
    by_citations = sorted(others, key=lambda p: p.get("cited_by_count") or 0, reverse=True)
    by_relevance = sorted(others, key=lambda p: p.get("api_rank", 999))
    seen_ids = {p.get("work_id") for p in llm_priority}
    interleaved = []
    ci, ri = 0, 0
    while len(interleaved) < len(others):
        # Alternate: one from citations, one from relevance
        for lst, idx_ref in [(by_citations, 'ci'), (by_relevance, 'ri')]:
            idx = ci if idx_ref == 'ci' else ri
            while idx < len(lst):
                wid = lst[idx].get("work_id")
                if idx_ref == 'ci':
                    ci = idx + 1
                else:
                    ri = idx + 1
                if wid not in seen_ids:
                    seen_ids.add(wid)
                    interleaved.append(lst[idx])
                    break
                if idx_ref == 'ci':
                    ci = idx + 1
                else:
                    ri = idx + 1
                idx += 1

    openalex_papers = llm_priority + interleaved
    if llm_expanded_ids:
        logger.info(f"LLM-expanded priority candidates: {len(llm_priority)} papers")
    logger.info(f"Diversified scoring batch: {len(openalex_papers)} total candidates")

    # Fetch abstracts for top candidates to give LLM better context
    def _fetch_abstract_for_candidate(paper):
        wid = paper.get("work_id", "")
        if paper.get("abstract"):
            return

        if wid.startswith("W"):
            # OpenAlex
            try:
                url = f"https://api.openalex.org/works/{wid}"
                params = {"select": "abstract_inverted_index"}
                oa_key = get_oa_api_key()
                if oa_key:
                    params["api_key"] = oa_key
                _oa_throttle()
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    abstract = _extract_abstract(resp.json())
                    if abstract:
                        paper["abstract"] = abstract
            except Exception:
                pass

        elif wid.startswith("S2:"):
            # Semantic Scholar
            s2_id = wid[3:]
            s2_headers = get_s2_headers()
            try:
                url = f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}"
                resp = _s2_get_with_retry(url, params={"fields": "abstract"}, headers=s2_headers, timeout=10)
                if resp.status_code == 200:
                    abstract = resp.json().get("abstract")
                    if abstract:
                        paper["abstract"] = abstract
            except Exception:
                pass

        elif wid.startswith("AX:"):
            # ArXiv via S2 bridge (avoids ArXiv's 1 req/3s rate limit)
            arxiv_id = wid[3:]
            s2_headers = get_s2_headers()
            try:
                url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}"
                resp = _s2_get_with_retry(url, params={"fields": "abstract"}, headers=s2_headers, timeout=10)
                if resp.status_code == 200:
                    abstract = resp.json().get("abstract")
                    if abstract:
                        paper["abstract"] = abstract
            except Exception:
                pass

    # Fetch abstracts in parallel for top-80 candidates
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(_fetch_abstract_for_candidate, openalex_papers[:80])

    # Run LLM validation on top-80 candidates (interleaved by citations + relevance)
    llm_scores = _score_seed_candidates(query, openalex_papers, top_k=40)

    # Helper: pick best seed from candidate list with keyword sanity check
    def _pick_best_seed(candidates, scores, strategy_label, filter_desc):
        """Pick best seed with keyword overlap guard against wrong-field selections."""
        import math as _math
        if not candidates:
            return None
        max_cites = max((p.get("cited_by_count") or 0 for p in candidates), default=1) or 1

        def _composite(p):
            llm = scores.get(p.get("work_id"), 0) if scores else 0
            cites = p.get("cited_by_count") or 0
            norm_cites = _math.log1p(cites) / _math.log1p(max_cites)
            return 0.6 * llm + 0.4 * norm_cites

        candidates.sort(key=_composite, reverse=True)

        # Try each candidate — reject seeds with zero keyword overlap (wrong field)
        for paper in candidates:
            overlap = _keyword_overlap_score(
                query,
                paper.get("title", ""),
                paper.get("abstract", ""),
            )
            if overlap > 0:
                llm_score = scores.get(paper.get("work_id"), 0) if scores else 0
                return paper["work_id"], {
                    "selection_strategy": strategy_label,
                    "selection_reason": (
                        f"{filter_desc} "
                        f"(citations: {paper.get('cited_by_count', 0):,}, "
                        f"relevance: {llm_score:.2f}, keyword_overlap: {overlap:.2f})"
                    ),
                    "candidates_considered": len(all_papers),
                    "llm_validated_count": len(candidates),
                    "seed_title": paper.get("title"),
                }
            else:
                logger.warning(
                    f"Seed rejected (zero keyword overlap): "
                    f"{paper.get('title', '')[:80]} for query '{query}'"
                )
        return None

    if llm_scores:
        # Filter to HIGH+ relevance papers (score >= 0.75)
        relevant_papers = [
            p for p in openalex_papers
            if llm_scores.get(p.get("work_id"), 0) >= MIN_SEED_LLM_SCORE
        ]

        # Intersection-aware seed selection: prefer papers matching BOTH topic AND domain
        if relevant_papers and scope == "intersection" and structured_query and structured_query.get("domain"):
            from app.feature1.candidate_scoring import compute_relevance_score
            both_match = []
            either_match = []
            for p in relevant_papers:
                rel = compute_relevance_score(p, structured_query, "intersection")
                if rel >= 0.5:  # Both topic AND domain match → base 0.8
                    both_match.append(p)
                elif rel >= 0.3:  # At least one matches
                    either_match.append(p)
            if both_match:
                logger.info(
                    f"Intersection seed filter: {len(both_match)} papers match both topic+domain "
                    f"(from {len(relevant_papers)} LLM-validated)"
                )
                relevant_papers = both_match
            elif either_match:
                logger.info(
                    f"Intersection seed filter: no both-match, falling back to {len(either_match)} either-match"
                )
                relevant_papers = either_match
            # else: keep original relevant_papers (no intersection filtering)

        if relevant_papers:
            result = _pick_best_seed(
                relevant_papers, llm_scores,
                "llm_validated_highest_cited",
                f"Best of {len(relevant_papers)} LLM-validated papers",
            )
            if result:
                return result
            # All HIGH+ papers failed keyword check — fall through to below-threshold

        # No papers passed LLM threshold (or all failed keyword check)
        # Use highest LLM-scored paper (even if below threshold)
        scored_papers = [
            p for p in openalex_papers
            if p.get("work_id") in llm_scores and llm_scores.get(p.get("work_id"), 0) > 0.05
        ]
        if scored_papers:
            result = _pick_best_seed(
                scored_papers, llm_scores,
                "llm_validated_highest_cited",
                f"Best available (no papers reached HIGH threshold)",
            )
            if result:
                return result

    # Fallback: LLM validation failed or all candidates failed keyword check.
    # Use keyword-based heuristic instead of blindly picking highest cited.
    keyword_scored = []
    for p in openalex_papers[:80]:
        kw_score = _keyword_overlap_score(
            query,
            p.get("title", ""),
            p.get("abstract", ""),
        )
        if kw_score > 0:
            keyword_scored.append((p, kw_score))

    if keyword_scored:
        # Sort by keyword overlap (primary), then citations (secondary)
        keyword_scored.sort(key=lambda x: (-x[1], -(x[0].get("cited_by_count") or 0)))
        best_paper, best_kw = keyword_scored[0]
        return best_paper["work_id"], {
            "selection_strategy": "keyword_fallback",
            "selection_reason": (
                f"LLM validation unavailable; selected by keyword overlap "
                f"(overlap: {best_kw:.2f}, citations: {best_paper.get('cited_by_count', 0):,})"
            ),
            "candidates_considered": len(all_papers),
            "seed_title": best_paper.get("title"),
        }

    # Last resort: highest cited (should be extremely rare)
    best_paper = openalex_papers[0]
    return best_paper["work_id"], {
        "selection_strategy": "highest_cited_fallback",
        "selection_reason": (
            f"Highest cited ({best_paper.get('cited_by_count', 0):,}) - "
            f"no keyword matches found"
        ),
        "candidates_considered": len(all_papers),
        "seed_title": best_paper.get("title"),
    }


def select_multi_seeds(
    query: str,
    structured_query: Dict[str, Any],
    scope: str,
    max_seeds: int = 3,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Select multiple seed papers for multi-facet graph expansion.

    For intersection queries, selects up to 3 seeds:
    1. Intersection seed — matches both topic AND domain
    2. Topic seed — best paper matching topic only
    3. Domain seed — best paper matching domain only

    For non-intersection queries, falls back to single seed.

    Returns list of (work_id, paper_dict) tuples.
    """
    topic = structured_query.get("topic", "")
    domain = structured_query.get("domain", "")
    topic_aliases = structured_query.get("topic_aliases", [])
    domain_aliases = structured_query.get("domain_aliases", [])

    intent = structured_query.get("intent", "single_topic")
    if (scope != "intersection" and intent != "cross_domain") or not domain:
        # Non-intersection / non-cross-domain: single seed via existing function
        seed_id, info = select_seed_from_query(query, scope=scope, structured_query=structured_query)
        if seed_id:
            paper = fetch_seed_paper_details(seed_id) or {"work_id": seed_id}
            return [(seed_id, paper)]
        return []

    seeds = []
    seen_ids: set = set()

    # Seed 1: Intersection — search for papers matching both topic AND domain
    intersection_query = f"{topic} {domain}"
    intersection_id, _info = select_seed_from_query(
        intersection_query, scope="intersection", structured_query=structured_query,
    )
    if intersection_id:
        paper = fetch_seed_paper_details(intersection_id) or {"work_id": intersection_id}
        seeds.append((intersection_id, paper))
        seen_ids.add(intersection_id)

    # Seed 2: Topic-focused
    if intent == "cross_domain":
        topic_query = topic  # bare topic works for cross-domain
    else:
        topic_query = f"{topic} {domain}"  # full query prevents off-topic seeds
    topic_id, _ = select_seed_from_query(
        topic_query, scope="topic_focused", structured_query=structured_query,
    )
    if topic_id and topic_id not in seen_ids:
        paper = fetch_seed_paper_details(topic_id) or {"work_id": topic_id}
        seeds.append((topic_id, paper))
        seen_ids.add(topic_id)

    # Seed 3: Domain-focused
    if intent == "cross_domain":
        domain_query = domain  # bare domain works for cross-domain
    else:
        domain_query = f"{topic} {domain}"  # full query prevents off-topic seeds
    domain_id, _ = select_seed_from_query(
        domain_query, scope="domain_focused", structured_query=structured_query,
    )
    if domain_id and domain_id not in seen_ids:
        paper = fetch_seed_paper_details(domain_id) or {"work_id": domain_id}
        seeds.append((domain_id, paper))
        seen_ids.add(domain_id)

    logger.info(
        f"Multi-seed selection: {len(seeds)} seeds for scope={scope} "
        f"[{', '.join(s[1].get('title', '?')[:40] for s in seeds)}]"
    )

    return seeds[:max_seeds]


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
        authors=seed_data.get("authors") or [],
        venue=seed_data.get("venue"),
        doi=seed_data.get("doi"),
        is_open_access=seed_data.get("is_open_access"),
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
            authors=paper.get("authors") or [],
            venue=paper.get("venue"),
            doi=paper.get("doi"),
            is_open_access=paper.get("is_open_access"),
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
            authors=paper.get("authors") or [],
            venue=paper.get("venue"),
            doi=paper.get("doi"),
            is_open_access=paper.get("is_open_access"),
            is_seed=False,
            hop=1,
            relationship="cited_by_seed",
        ))
        edges.append(CitationEdge(from_work_id=seed_work_id, to_work_id=wid))
        seen_work_ids.add(wid)

    return nodes, edges


def _assemble_multihop_graph(
    seed_work_ids: Set[str],
    papers_dict: Dict[str, Dict[str, Any]],
    edge_tuples: List[Tuple[str, str]],
    min_citations: int = 0,
) -> Tuple[List[CitationNode], List[CitationEdge]]:
    """Assemble nodes and edges from multi-hop expansion data.

    Edge direction: from_work_id cites to_work_id
    """
    nodes: List[CitationNode] = []
    edges: List[CitationEdge] = []

    # Build edge index for O(1) lookup instead of scanning all tuples
    edge_from_index: Dict[str, Set[str]] = {}
    for f, t in edge_tuples:
        edge_from_index.setdefault(f, set()).add(t)

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
            # Check edge direction to determine if it cites any seed
            targets = edge_from_index.get(work_id, set())
            cites_seed = bool(targets & seed_work_ids)
            relationship = "cites_seed" if cites_seed else "cited_by_seed"
        else:
            relationship = "network"  # 2+ hop papers

        nodes.append(CitationNode(
            work_id=work_id,
            title=paper.get("title"),
            year=paper.get("year"),
            cited_by_count=paper.get("cited_by_count") or 0,
            abstract=paper.get("abstract"),
            authors=paper.get("authors") or [],
            venue=paper.get("venue"),
            doi=paper.get("doi"),
            is_open_access=paper.get("is_open_access"),
            is_seed=is_seed,
            hop=hop,
            relationship=relationship,
        ))

    # Only include edges where both endpoints are in the filtered node set (no self-loops)
    node_ids = {n.work_id for n in nodes}
    for from_id, to_id in edge_tuples:
        if from_id != to_id and from_id in node_ids and to_id in node_ids:
            edges.append(CitationEdge(from_work_id=from_id, to_work_id=to_id))

    logger.info(f"Assembled multi-hop graph: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ============================================================================
# Subtopic Clustering
# ============================================================================

def _cluster_papers_for_citation_map(
    papers: List[Dict[str, Any]],
    query_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """LLM-based paper clustering for citation maps.

    Unlike F2's subtopic service (which limits to 4-5), this creates as many
    subtopics as naturally emerge, with a minimum of 2 papers per subtopic.
    Every paper MUST be assigned to exactly one subtopic — no leftovers.

    Returns list of subtopic dicts with label, description, and work_ids.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, cannot cluster papers")
        return []

    sorted_papers = sorted(papers, key=lambda p: p.get("rank_index", 999))[:60]

    paper_list = []
    for i, p in enumerate(sorted_papers):
        paper_list.append({
            "id": i,
            "title": (p.get("title") or "Untitled")[:200],
            "work_id": p.get("work_id"),
        })

    query_context = f'Research area: "{query_text}"' if query_text else ""

    papers_json = json.dumps(
        [{"id": p["id"], "title": p["title"]} for p in paper_list], indent=2
    )

    prompt = f"""Cluster ALL of these papers into natural research subtopics.
{query_context}

Papers:
{papers_json}

RULES:
1. EVERY paper must be assigned to exactly ONE subtopic — no paper left out
2. Create as many subtopics as naturally emerge (typically 4-8 for ~30 papers)
3. Each subtopic needs at least 2 papers
4. Labels should be specific named methods, algorithms, or phenomena from the papers
5. Papers that don't fit neatly into a technical cluster should go into a subtopic like "Foundational Methods" or a domain-specific general category — NOT left unassigned

Return JSON only:
{{"subtopics": [{{"label": "Specific Name", "description": "One sentence", "paper_ids": [0,1,2]}}]}}

CRITICAL: Every paper ID (0 through {len(paper_list) - 1}) must appear in exactly one subtopic."""

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a research librarian. Return only valid JSON. Respond in English only. No markdown code blocks."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=60.0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()

            # Strip markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                start_idx = 1
                end_idx = len(lines)
                if lines[-1].strip().startswith("```"):
                    end_idx = -1
                content = "\n".join(lines[start_idx:end_idx])

            # Extract JSON object
            start = content.find("{")
            if start >= 0:
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
                content = content[start:end]

            result = json.loads(content)
            subtopics_raw = result.get("subtopics", [])

            if not subtopics_raw:
                logger.warning("LLM returned no subtopics for citation map")
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return []

            # Convert paper IDs to work_ids
            subtopics = []
            seen_work_ids: set = set()
            for st in subtopics_raw:
                paper_ids = st.get("paper_ids", [])
                work_ids = []
                for pid in paper_ids:
                    try:
                        idx_int = int(pid)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= idx_int < len(paper_list):
                        wid = paper_list[idx_int]["work_id"]
                        if wid not in seen_work_ids:
                            work_ids.append(wid)
                            seen_work_ids.add(wid)

                if len(work_ids) >= 2:
                    subtopics.append({
                        "label": st.get("label", "Subtopic"),
                        "description": st.get("description", ""),
                        "work_ids": work_ids,
                    })

            if len(subtopics) >= 2:
                subtopics.sort(key=lambda s: -len(s["work_ids"]))
                return subtopics

            logger.warning(f"Only {len(subtopics)} valid subtopics, retrying...")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse subtopic JSON: {e}")
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
        except Exception as e:
            logger.warning(f"Citation map clustering failed: {e}")
            break

    return []


def _subtopic_title_keywords(title: str, stopwords: set) -> set:
    """Extract meaningful keywords from a paper title for subtopic matching."""
    words = set(title.lower().replace("-", " ").split())
    return words - stopwords


def _match_to_closest_subtopic(
    node_title: str,
    subtopic_keywords: Dict[str, set],
    stopwords: set,
    min_overlap: int = 2,
) -> Optional[str]:
    """Match an unassigned paper to the closest subtopic by title keyword overlap.

    Returns the best matching subtopic label, or None if no good match.
    """
    node_kw = _subtopic_title_keywords(node_title, stopwords)
    if not node_kw:
        return None

    best_label = None
    best_score = 0
    for label, kw_set in subtopic_keywords.items():
        overlap = len(node_kw & kw_set)
        if overlap >= min_overlap and overlap > best_score:
            best_score = overlap
            best_label = label
    return best_label


def _assign_subtopics(nodes: List[CitationNode], query_text: Optional[str] = None) -> None:
    """Assign LLM-generated subtopic labels to citation map nodes.

    Uses F1-specific clustering (no subtopic count limit, all papers must be assigned).
    Falls back to keyword matching for any papers the LLM missed.
    Gracefully degrades: if LLM fails or GROQ_API_KEY is missing, nodes keep subtopic=None.
    """
    if not nodes:
        return

    # Build paper list sorted by citation count (most influential first)
    papers = []
    for i, node in enumerate(sorted(nodes, key=lambda n: n.cited_by_count, reverse=True)):
        papers.append({
            "title": node.title or "Untitled",
            "work_id": node.work_id,
            "rank_index": i,
        })

    try:
        subtopics = _cluster_papers_for_citation_map(papers, query_text)
    except Exception as e:
        logger.warning(f"Subtopic clustering failed: {e}")
        return

    if not subtopics:
        return

    # Build work_id -> label mapping from LLM results
    wid_to_label: Dict[str, str] = {}
    for st in subtopics:
        label = st.get("label", "")
        for wid in st.get("work_ids", []):
            wid_to_label[wid] = label

    # Build keyword profiles for each subtopic from assigned papers' titles
    stopwords = {"a", "an", "the", "of", "and", "for", "in", "on", "to", "with",
                 "is", "are", "by", "from", "at", "as", "or", "that", "this",
                 "its", "it", "be", "was", "were", "been", "being", "do", "does",
                 "did", "has", "have", "had", "not", "but", "if", "we", "our",
                 "via", "using", "based", "towards", "toward"}
    node_by_wid = {n.work_id: n for n in nodes}
    subtopic_keywords: Dict[str, set] = {}
    for label in set(wid_to_label.values()):
        kw_set: set = set()
        for wid, lbl in wid_to_label.items():
            if lbl == label:
                title = (node_by_wid.get(wid) or nodes[0]).title or ""
                kw_set |= _subtopic_title_keywords(title, stopwords)
        kw_set |= _subtopic_title_keywords(label, stopwords)
        subtopic_keywords[label] = kw_set

    # First pass: assign LLM-assigned labels
    assigned = 0
    unassigned_nodes: List[CitationNode] = []
    for node in nodes:
        label = wid_to_label.get(node.work_id)
        if label:
            node.subtopic = label
            assigned += 1
        else:
            unassigned_nodes.append(node)

    # Second pass: match unassigned papers to closest subtopic by title overlap
    matched = 0
    for node in unassigned_nodes:
        if not node.title:
            node.subtopic = "Other"
            continue
        title_words = len(_subtopic_title_keywords(node.title, stopwords))
        min_req = 2 if title_words >= 5 else 1
        best_label = _match_to_closest_subtopic(
            node.title, subtopic_keywords, stopwords, min_overlap=min_req
        )
        if best_label:
            node.subtopic = best_label
            matched += 1
        else:
            node.subtopic = "Other"

    # Third pass: if "Other" is too large (>20%), retry with min_overlap=1
    other_nodes = [n for n in nodes if n.subtopic == "Other"]
    if len(other_nodes) > len(nodes) * 0.2:
        for node in other_nodes:
            if not node.title:
                continue
            best_label = _match_to_closest_subtopic(
                node.title, subtopic_keywords, stopwords, min_overlap=1
            )
            if best_label:
                node.subtopic = best_label
                matched += 1

    total_assigned = assigned + matched
    other_count = len([n for n in nodes if n.subtopic == "Other"])
    logger.info(
        f"Subtopic assignment: {assigned} LLM-assigned + {matched} keyword-matched = "
        f"{total_assigned}/{len(nodes)} nodes in {len(subtopics)} subtopics, {other_count} as Other"
    )


# ============================================================================
# Graph Draft Creation
# ============================================================================

def _persist_paper_metadata(conn: Connection, nodes: List[CitationNode]) -> int:
    """Upsert paper metadata into the works table.

    Uses COALESCE on conflict so we never overwrite richer data from F2.
    Returns count of rows upserted.
    """
    if not nodes:
        return 0

    count = 0
    for node in nodes:
        authors_json = json.dumps(node.authors) if node.authors else None

        conn.execute(
            text("""
                INSERT INTO works (work_id, title, year, cited_by_count, abstract,
                                   authors_json, venue, doi, is_open_access)
                VALUES (:work_id, :title, :year, :cited_by_count, :abstract,
                        CAST(:authors_json AS jsonb), :venue, :doi, :is_open_access)
                ON CONFLICT (work_id) DO UPDATE SET
                    title = COALESCE(works.title, EXCLUDED.title),
                    year = COALESCE(works.year, EXCLUDED.year),
                    cited_by_count = GREATEST(COALESCE(works.cited_by_count, 0), COALESCE(EXCLUDED.cited_by_count, 0)),
                    abstract = COALESCE(works.abstract, EXCLUDED.abstract),
                    authors_json = COALESCE(works.authors_json, EXCLUDED.authors_json),
                    venue = COALESCE(works.venue, EXCLUDED.venue),
                    doi = COALESCE(works.doi, EXCLUDED.doi),
                    is_open_access = COALESCE(works.is_open_access, EXCLUDED.is_open_access)
            """),
            {
                "work_id": node.work_id,
                "title": node.title,
                "year": node.year,
                "cited_by_count": node.cited_by_count,
                "abstract": node.abstract,
                "authors_json": authors_json,
                "venue": node.venue,
                "doi": node.doi,
                "is_open_access": node.is_open_access,
            },
        )
        count += 1

    logger.info(f"Persisted metadata for {count} papers to works table")
    return count


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
# Citation Map Persistence & Retrieval
# ============================================================================

def _persist_citation_map(
    conn: Connection,
    tenant_id: UUID,
    request: "CitationMapRequest",
    response: "CitationMapResponse",
) -> UUID:
    """Persist the full citation map response as JSONB.

    Returns the citation_map_id.
    """
    citation_map_id = uuid4()
    response_json = response.model_dump(mode="json")

    conn.execute(
        text("""
            INSERT INTO citation_maps
                (citation_map_id, tenant_id, graph_draft_id, seed_work_id,
                 query_text, seed_doi, seed_title, response_json)
            VALUES
                (:citation_map_id, :tenant_id, :graph_draft_id, :seed_work_id,
                 :query_text, :seed_doi, :seed_title, CAST(:response_json AS jsonb))
        """),
        {
            "citation_map_id": citation_map_id,
            "tenant_id": tenant_id,
            "graph_draft_id": response.graph_draft_id,
            "seed_work_id": response.seed_info.seed_work_id,
            "query_text": request.query_text,
            "seed_doi": request.seed_doi,
            "seed_title": request.seed_title or response.seed_info.seed_title,
            "response_json": json.dumps(response_json),
        },
    )
    conn.commit()
    logger.info(f"Persisted citation map {citation_map_id} ({len(response.nodes)} nodes)")
    return citation_map_id


def get_citation_map(
    engine: Engine,
    *,
    tenant_id: UUID,
    citation_map_id: UUID,
) -> "CitationMapResponse":
    """Retrieve a previously built citation map by ID.

    Raises ValueError("citation_map_not_found") if not found or wrong tenant.
    """
    from app.feature1.schemas import CitationMapResponse

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT response_json
                FROM citation_maps
                WHERE citation_map_id = :cid AND tenant_id = :tid
            """),
            {"cid": citation_map_id, "tid": tenant_id},
        ).mappings().first()

    if not row:
        raise ValueError("citation_map_not_found")

    response = CitationMapResponse.model_validate(row["response_json"])
    response.citation_map_id = citation_map_id
    return response


def list_citation_maps(
    engine: Engine,
    *,
    tenant_id: UUID,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List saved citation maps for a tenant, most recent first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT citation_map_id, seed_work_id, seed_title, query_text,
                       (response_json->'stats'->>'total_nodes')::int AS node_count,
                       created_at
                FROM citation_maps
                WHERE tenant_id = :tid
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"tid": tenant_id, "lim": limit},
        ).mappings().all()

    return [dict(r) for r in rows]


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
    overall_start = time.time()
    # #region agent log
    _debug_log(
        "H15",
        "app/feature1/citation_map_service.py:build_citation_map.start",
        "Citation map build started",
        {
            "hasSeedWorkId": bool(request.seed_work_id),
            "hasSeedDoi": bool(request.seed_doi),
            "hasSeedTitle": bool(request.seed_title),
            "hasQueryText": bool(request.query_text),
        },
    )
    # #endregion

    with engine.connect() as conn:
        # Step 0: Parse V2 inputs (needed before seed selection for intersection-aware filtering)
        _scope = getattr(request, "scope", None) or "broad"
        _drift = getattr(request, "drift", None) or "moderate"
        _temporal = getattr(request, "temporal", None) or "all"
        _map_size = getattr(request, "map_size", None) or "medium"

        # Legacy backward compatibility mapping
        if getattr(request, "map_focus", None) and not getattr(request, "drift", None):
            _drift = {
                "core_cluster": "strict",
                "landscape": "open",
                "evolution": "moderate",
            }.get(request.map_focus, "moderate")

        if getattr(request, "expansion", None) and not getattr(request, "drift", None):
            _drift = {
                "narrow": "strict",
                "foundations": "moderate",
                "wide": "open",
            }.get(request.expansion, _drift)

            if request.expansion == "foundations" and not getattr(request, "temporal", None):
                _temporal = "seminal"
            elif request.expansion == "narrow" and not getattr(request, "temporal", None):
                _temporal = "recent"

        if getattr(request, "map_focus", None) == "evolution" and not getattr(request, "temporal", None):
            _temporal = "all"

        # Build structured query (if provided or auto-decompose)
        _sq = None
        if getattr(request, "topic", None):
            _sq = {
                "topic": request.topic,
                "domain": getattr(request, "domain", None),
                "aspect": getattr(request, "aspect", None),
                "topic_aliases": [],
                "domain_aliases": [],
                "aspect_aliases": [],
                "intent": "single_topic",
            }
        elif request.query_text:
            try:
                from app.common.query_decomposition import decompose_query as _decompose_q
                with engine.connect() as _dconn:
                    _sq = _decompose_q(_dconn, request.query_text)
            except Exception as _e:
                logger.warning(f"Citation map auto-decomposition failed: {_e}")

        if not _sq:
            _sq = {"topic": request.query_text or "", "domain": None, "aspect": None, "intent": "single_topic"}

        # If no domain, force scope to broad
        if not _sq.get("domain"):
            _scope = "broad"

        # Step 1: Determine seed paper (support multiple input modes)
        all_seed_ids = None  # Set by query_text mode (multi-seed), defaulted after selection

        if request.seed_doi:
            # Mode: DOI (from PDF metadata or user input)
            logger.info(f"Looking up paper by DOI: {request.seed_doi}")
            seed_work_id = _lookup_openalex_by_doi(request.seed_doi)

            if not seed_work_id:
                # Fallback: try Semantic Scholar by DOI
                logger.info(f"DOI not in OpenAlex, trying S2: {request.seed_doi}")
                s2_paper = _lookup_s2_paper_by_doi(request.seed_doi)
                if s2_paper and s2_paper.get("paperId"):
                    seed_work_id = f"S2:{s2_paper['paperId']}"
                    logger.info(f"Found via S2: {seed_work_id}")

            if not seed_work_id:
                raise ValueError(f"Paper with DOI {request.seed_doi} not found in OpenAlex or Semantic Scholar")
            seed_data = fetch_seed_paper_details(seed_work_id)
            seed_info = SeedSelectionInfo(
                seed_work_id=seed_work_id,
                seed_title=seed_data.get("title") if seed_data else None,
                selection_strategy="doi_lookup",
                selection_reason=f"Found via DOI: {request.seed_doi}",
                candidates_considered=1,
            )

        elif request.seed_title:
            # Mode: Title search (searches OpenAlex + S2)
            # Normalize ligatures (ﬁ→fi, ﬂ→fl) that PDF extraction often produces
            search_title = _normalize_title_text(request.seed_title)
            logger.info(f"Searching for paper by title: {search_title[:50]}...")
            # Search OpenAlex
            oa_results = _search_openalex_by_title(search_title, k=5)
            # Also search S2
            s2_raw = _search_s2_by_title(search_title, limit=5)

            # Map S2 results to OpenAlex work_ids via DOI or ArXiv bridge
            s2_results = []
            for s2_paper in s2_raw:
                doi = s2_paper.get("doi")
                arxiv_id = s2_paper.get("arxiv_id")
                work_id = None

                # Try DOI first (fastest)
                if doi:
                    work_id = _lookup_openalex_by_doi(doi)

                # Fallback to ArXiv ID bridge
                if not work_id and arxiv_id:
                    oa_data = _lookup_openalex_by_arxiv(arxiv_id)
                    if oa_data:
                        work_id = oa_data.get("work_id")

                if work_id:
                    s2_results.append({
                        "work_id": work_id,
                        "title": s2_paper.get("title"),
                        "year": s2_paper.get("year"),
                        "cited_by_count": s2_paper.get("cited_by_count") or 0,
                        "source": "s2_mapped",
                    })
                elif s2_paper.get("s2_id"):
                    # S2-only paper (no OA bridge) — keep as independent source
                    s2_results.append({
                        "work_id": f"S2:{s2_paper['s2_id']}",
                        "title": s2_paper.get("title"),
                        "year": s2_paper.get("year"),
                        "cited_by_count": s2_paper.get("cited_by_count") or 0,
                        "source": "s2_only",
                    })

            # Combine and dedupe by work_id — prefer higher-cited entry for duplicates
            all_candidates = {}
            for p in (oa_results + s2_results):
                wid = p.get("work_id")
                if not wid:
                    continue
                existing = all_candidates.get(wid)
                if not existing or (p.get("cited_by_count", 0) > existing.get("cited_by_count", 0)):
                    all_candidates[wid] = p

            if not all_candidates:
                raise ValueError(f"Paper with title '{request.seed_title}' not found in any source")

            # Use title similarity + citations to pick best match
            # This handles duplicate OpenAlex entries (e.g., pdf_svm, title_backprop)
            def _title_match_score(wid):
                p = all_candidates[wid]
                p_title = _normalize_title_text((p.get("title") or "")).lower()
                q_title = _normalize_title_text(search_title).lower()
                # Jaccard on punctuation-stripped words (so "BERT:" matches "BERT")
                p_words = {_strip_punct(w) for w in p_title.split() if _strip_punct(w)}
                q_words = {_strip_punct(w) for w in q_title.split() if _strip_punct(w)}
                jaccard = len(p_words & q_words) / max(len(p_words | q_words), 1)
                cites = p.get("cited_by_count", 0)
                import math as _m
                # Near-exact title match: let citations decide entirely
                # Handles OpenAlex duplicates where same paper has wildly different cite counts
                if jaccard > 0.85:
                    return 1.0 + _m.log1p(cites)
                return jaccard * 0.6 + _m.log1p(cites) / _m.log1p(100000) * 0.4

            seed_work_id = max(all_candidates.keys(), key=_title_match_score)
            seed_data = all_candidates[seed_work_id]
            seed_info = SeedSelectionInfo(
                seed_work_id=seed_work_id,
                seed_title=seed_data.get("title"),
                selection_strategy="title_search",
                selection_reason=f"Found via title search across OpenAlex + S2",
                candidates_considered=len(all_candidates),
            )

        elif request.seed_work_id:
            # Mode: Direct OpenAlex work ID
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
            # Mode 2: Natural language query — multi-seed for intersection
            seeds = select_multi_seeds(
                request.query_text,
                structured_query=_sq,
                scope=_scope,
                max_seeds=3,
            )
            if not seeds:
                logger.info("Citation map query had no seeds for query: %s", request.query_text)
                return CitationMapResponse(
                    seed_info=SeedSelectionInfo(
                        seed_work_id="",
                        seed_title=None,
                        selection_strategy="none",
                        selection_reason="No seed found for query",
                        candidates_considered=0,
                    ),
                    nodes=[],
                    edges=[],
                    graph_draft_id=None,
                    stats=CitationMapStats(),
                )

            seed_work_id = seeds[0][0]  # Primary seed for response metadata
            seed_data = seeds[0][1]
            all_seed_ids = [s[0] for s in seeds]

            seed_info = SeedSelectionInfo(
                seed_work_id=seed_work_id,
                seed_title=seed_data.get("title"),
                selection_strategy=f"multi_seed_{len(seeds)}",
                selection_reason=f"Multi-seed: {', '.join(s[1].get('title', '?')[:30] for s in seeds)}",
                candidates_considered=len(seeds),
            )
            logger.info(
                "Citation map query selected %d seeds: primary=%s strategy=%s",
                len(seeds),
                seed_info.seed_work_id,
                seed_info.selection_strategy,
            )

        else:
            raise ValueError("Either seed_work_id or query_text must be provided")

        # Normalize seed list: query_text mode sets all_seed_ids, others use single seed
        if all_seed_ids is None:
            all_seed_ids = [seed_work_id]

        # Step 2: Build citation network (V2 score-and-select)

        expand_start = time.time()
        # #region agent log
        _debug_log(
            "H15",
            "app/feature1/citation_map_service.py:build_citation_map.expand_start",
            "V2 citation network expansion started",
            {
                "seedWorkId": seed_work_id,
                "seedCount": len(all_seed_ids),
                "scope": _scope,
                "drift": _drift,
                "temporal": _temporal,
                "mapSize": _map_size,
            },
        )
        # #endregion

        candidates, edge_tuples, adjacency = _expand_citation_network_v2(
            seed_work_ids=all_seed_ids,
            structured_query=_sq,
            scope=_scope,
            drift=_drift,
            temporal=_temporal,
            map_size=_map_size,
            engine=engine,
        )

        # Greedy graph selection
        from app.feature1.candidate_scoring import greedy_graph_select, get_fetch_limits
        target = get_fetch_limits(_map_size)["target_nodes"]

        # Dedup by DOI — preprint vs published versions
        by_doi: Dict[str, Dict[str, Any]] = {}
        no_doi = []
        for c in candidates:
            doi = c.get("doi")
            if doi:
                existing = by_doi.get(doi)
                if existing is None or c.get("cited_by_count", 0) > existing.get("cited_by_count", 0):
                    by_doi[doi] = c
            else:
                no_doi.append(c)
        candidates = list(by_doi.values()) + no_doi

        # LLM relevance scoring — replaces keyword scores for top candidates
        from app.feature1.candidate_scoring import llm_relevance_score
        try:
            llm_scores = llm_relevance_score(
                candidates=candidates,
                structured_query=_sq,
                scope=_scope,
                max_candidates=75,
            )
            if llm_scores:
                for c in candidates:
                    wid = c.get("work_id")
                    if wid in llm_scores:
                        c["_llm_relevance"] = llm_scores[wid]
                logger.info(f"LLM relevance scoring: {len(llm_scores)} papers scored")
        except Exception as e:
            logger.warning(f"LLM relevance scoring failed (non-fatal): {e}")

        _relevance_floors = {"intersection": 0.3, "topic_focused": 0.2, "domain_focused": 0.2, "broad": 0.1}

        selected_ids = greedy_graph_select(
            candidates=candidates,
            edges=adjacency,
            seed_ids=all_seed_ids,
            target_size=target,
            structured_query=_sq,
            scope=_scope,
            temporal=_temporal,
            relevance_floor=_relevance_floors.get(_scope, 0.1),
        )

        # Build papers_dict from selected candidates
        selected_set = set(selected_ids)
        papers_dict = {}
        for c in candidates:
            wid = c.get("work_id")
            if wid and wid in selected_set:
                papers_dict[wid] = c

        # Filter edge_tuples to only selected nodes
        final_edge_tuples = [(s, t) for s, t in edge_tuples if s in selected_set and t in selected_set]

        # #region agent log
        _debug_log(
            "H15",
            "app/feature1/citation_map_service.py:build_citation_map.expand_done",
            "V2 citation network expansion finished",
            {
                "seedWorkId": seed_work_id,
                "seedCount": len(all_seed_ids),
                "totalCandidates": len(candidates),
                "selectedNodes": len(papers_dict),
                "edgeTupleCount": len(final_edge_tuples),
                "elapsedMs": int((time.time() - expand_start) * 1000),
            },
        )
        # #endregion

        # Convert to CitationNode and CitationEdge
        nodes, edges = _assemble_multihop_graph(
            seed_work_ids=set(all_seed_ids),
            papers_dict=papers_dict,
            edge_tuples=final_edge_tuples,
            min_citations=request.min_citations,
        )

        # Safety net: remove any isolated non-seed nodes post-assembly
        edge_endpoints = set()
        for e in edges:
            edge_endpoints.add(e.from_work_id)
            edge_endpoints.add(e.to_work_id)
        isolated = [n for n in nodes if not n.is_seed and n.work_id not in edge_endpoints]
        if isolated:
            logger.warning(f"Removing {len(isolated)} isolated nodes post-assembly")
            isolated_ids = {n.work_id for n in isolated}
            nodes = [n for n in nodes if n.work_id not in isolated_ids]

        # Step 4a: Assign subtopics via LLM clustering
        cluster_context = request.query_text or (seed_data.get("title") if seed_data else "") or ""
        _assign_subtopics(nodes, query_text=cluster_context)

        # Step 4b: Persist metadata and optionally create graph_draft
        graph_draft_id = None
        if nodes:
            _persist_paper_metadata(conn, nodes)
            if not request.create_graph_draft:
                conn.commit()  # graph_draft path commits inside _create_graph_draft
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

        logger.info(
            "Citation map built: seed=%s nodes=%s edges=%s graph_draft=%s",
            seed_info.seed_work_id,
            len(nodes),
            len(edges),
            graph_draft_id,
        )

        total_ms = int((time.time() - overall_start) * 1000)
        # #region agent log
        _debug_log(
            "H15",
            "app/feature1/citation_map_service.py:build_citation_map.success",
            "Citation map build finished",
            {
                "seedWorkId": seed_info.seed_work_id,
                "nodes": len(nodes),
                "edges": len(edges),
                "elapsedMs": total_ms,
            },
        )
        # #endregion
        response = CitationMapResponse(
            seed_info=seed_info,
            nodes=nodes,
            edges=edges,
            graph_draft_id=graph_draft_id,
            stats=stats,
        )

        # Step 6: Persist the full citation map response
        try:
            citation_map_id = _persist_citation_map(conn, tenant_id, request, response)
            response.citation_map_id = citation_map_id
        except Exception as e:
            logger.warning(f"Failed to persist citation map: {e}")

        return response
