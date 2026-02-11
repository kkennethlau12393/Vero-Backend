"""
Abstract validation and enrichment for Feature 3.

This module validates abstracts and fetches from fallback sources
(ArXiv, Semantic Scholar) when the primary abstract is corrupted or missing.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.paper_identity import (
    PaperIdentity,
    normalize_arxiv_id,
    normalize_doi,
    title_word_overlap,
    verify_paper_match,
)

logger = logging.getLogger(__name__)

# Rate limiters for external APIs
class RateLimiter:
    """Token bucket rate limiter for external APIs."""

    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


ARXIV_RATE_LIMITER = RateLimiter(calls_per_second=1.0)
S2_RATE_LIMITER = RateLimiter(calls_per_second=1.0)
OPENALEX_RATE_LIMITER = RateLimiter(calls_per_second=5.0)  # OpenAlex is generous

# API timeouts
API_TIMEOUT = 15


def detect_language_simple(text: str) -> Optional[str]:
    """
    Simple language detection without external dependencies.

    Uses common word patterns to detect English vs other languages.
    Returns 'en' for English, 'other' for non-English, None if uncertain.
    """
    if not text or len(text) < 50:
        return None

    text_lower = text.lower()

    # Common English words
    english_words = {
        'the', 'and', 'of', 'to', 'in', 'is', 'that', 'for', 'it', 'with',
        'as', 'are', 'on', 'we', 'this', 'by', 'from', 'which', 'an', 'be',
        'have', 'has', 'can', 'our', 'their', 'or', 'not', 'but', 'more',
        'these', 'such', 'been', 'also', 'its', 'may', 'than', 'between',
        'show', 'propose', 'method', 'results', 'paper', 'approach', 'model',
    }

    # Common Spanish words (for the ResNet case)
    spanish_words = {
        'de', 'la', 'el', 'en', 'que', 'y', 'los', 'se', 'del', 'las',
        'un', 'por', 'con', 'para', 'una', 'es', 'al', 'lo', 'como',
        'su', 'más', 'pero', 'sus', 'le', 'ya', 'este', 'entre', 'cuando',
    }

    words = set(re.findall(r'\b\w+\b', text_lower))

    english_count = len(words & english_words)
    spanish_count = len(words & spanish_words)

    # If significantly more Spanish words, it's likely Spanish
    if spanish_count > english_count * 1.5 and spanish_count >= 3:
        return 'other'

    # If we have reasonable English word count
    if english_count >= 5:
        return 'en'

    return None


def is_abstract_valid(
    title: str,
    abstract: Optional[str],
    expected_language: str = "en",
) -> Tuple[bool, str]:
    """
    Validate that an abstract is valid and matches the paper.

    Checks:
    1. Abstract exists and has reasonable length
    2. Language matches expected (detect language mismatch)
    3. Has minimal keyword overlap with title

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not abstract:
        return False, "Abstract is missing"

    if len(abstract.strip()) < 50:
        return False, "Abstract too short (<50 chars)"

    # Detect acknowledgments/boilerplate text masquerading as abstracts
    abstract_lower = abstract.lower()

    # Acknowledgments patterns - these indicate this is NOT an abstract
    acknowledgment_patterns = [
        "referees for help",
        "the authors thank",
        "we thank the",
        "we are grateful",
        "supported by the",
        "funded by the",
        "grant from the",
        "financial support from",
        "this work was supported",
        "this research was supported",
        "this study was funded",
        "acknowledges support from",
        "acknowledges funding from",
        "revision of work originally",
        "study commission",
    ]

    for pattern in acknowledgment_patterns:
        if pattern in abstract_lower:
            return False, f"Abstract appears to be acknowledgments text (contains '{pattern}')"

    # Detect if abstract is mostly about funding/support rather than research content
    # Count funding-related words vs research-content words
    funding_words = {'thank', 'thanks', 'grateful', 'supported', 'funded', 'grant',
                     'funding', 'support', 'commission', 'foundation', 'acknowledge',
                     'acknowledged', 'acknowledges', 'referee', 'referees', 'reviewer'}
    content_words = {'we', 'this', 'paper', 'study', 'show', 'propose', 'present',
                     'method', 'approach', 'results', 'demonstrate', 'find', 'analyze',
                     'analysis', 'model', 'data', 'evidence', 'effect', 'impact'}

    words = set(re.findall(r'\b\w+\b', abstract_lower))
    funding_count = len(words & funding_words)
    content_count = len(words & content_words)

    # If funding words dominate and there's little research content, reject
    if funding_count >= 3 and content_count <= 2:
        return False, f"Abstract appears to be acknowledgments (funding words: {funding_count}, content words: {content_count})"

    # Language check
    detected_lang = detect_language_simple(abstract)
    if detected_lang == 'other' and expected_language == 'en':
        return False, "Language mismatch: expected English, detected non-English"

    # Keyword overlap check - title words should appear in abstract
    if title:
        overlap = title_word_overlap(title, abstract)
        # Very low overlap suggests wrong abstract
        # But allow some tolerance for creative titles
        if overlap < 0.10 and len(title.split()) >= 4:
            return False, f"Very low title-abstract overlap ({overlap:.2f})"

    return True, "Valid"


# ============================================================================
# ArXiv API
# ============================================================================

def fetch_arxiv_by_id(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper details from ArXiv API by ID.

    Returns dict with: title, abstract, year, authors, arxiv_id
    """
    arxiv_id = normalize_arxiv_id(arxiv_id)
    if not arxiv_id:
        return None

    ARXIV_RATE_LIMITER.wait()

    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = requests.get(url, timeout=API_TIMEOUT)
        resp.raise_for_status()

        # Parse Atom XML response
        content = resp.text

        # Extract abstract
        abstract_match = re.search(r'<summary[^>]*>(.*?)</summary>', content, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else None

        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else None
        # Clean up ArXiv title (remove "Title: " prefix if present)
        if title and title.lower().startswith('arxiv query:'):
            return None  # No results

        # Extract year from published date
        published_match = re.search(r'<published>(\d{4})', content)
        year = int(published_match.group(1)) if published_match else None

        if not abstract:
            return None

        return {
            "title": title,
            "abstract": abstract,
            "year": year,
            "arxiv_id": arxiv_id,
            "source": "arxiv",
        }

    except Exception as e:
        logger.warning(f"ArXiv API error for {arxiv_id}: {e}")
        return None


def search_arxiv_by_title(title: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search ArXiv by title to find matching papers.

    Returns list of candidates for identity verification.
    """
    if not title or len(title) < 10:
        return []

    ARXIV_RATE_LIMITER.wait()

    try:
        # Clean title for search
        search_title = re.sub(r'[^\w\s]', ' ', title)
        search_title = re.sub(r'\s+', ' ', search_title).strip()

        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f'ti:"{search_title}"',
            "max_results": max_results,
        }

        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()

        content = resp.text
        results = []

        # Parse each entry
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        for entry in entries:
            abstract_match = re.search(r'<summary[^>]*>(.*?)</summary>', entry, re.DOTALL)
            title_match = re.search(r'<title[^>]*>(.*?)</title>', entry, re.DOTALL)
            id_match = re.search(r'<id>https?://arxiv\.org/abs/([^<]+)</id>', entry)
            published_match = re.search(r'<published>(\d{4})', entry)

            if abstract_match and title_match:
                results.append({
                    "title": title_match.group(1).strip(),
                    "abstract": abstract_match.group(1).strip(),
                    "year": int(published_match.group(1)) if published_match else None,
                    "arxiv_id": normalize_arxiv_id(id_match.group(1)) if id_match else None,
                    "source": "arxiv",
                })

        return results

    except Exception as e:
        logger.warning(f"ArXiv search error for '{title[:50]}...': {e}")
        return []


# ============================================================================
# OpenAlex API
# ============================================================================

def fetch_openalex_abstract(work_id: str, title: str = None) -> Optional[Dict[str, Any]]:
    """
    Fetch abstract from OpenAlex API by work_id.

    OpenAlex stores abstracts as inverted indexes, so we reconstruct them.
    """
    if not work_id or not work_id.startswith("W"):
        return None

    OPENALEX_RATE_LIMITER.wait()

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        headers = {"User-Agent": "Alexandria/1.0 (mailto:contact@example.com)"}

        resp = requests.get(url, headers=headers, timeout=API_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()

        # Reconstruct abstract from inverted index
        abstract_inv = data.get("abstract_inverted_index")
        if not abstract_inv:
            return None

        word_positions = []
        for word, positions in abstract_inv.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        abstract = " ".join(word for _, word in word_positions)

        if not abstract or len(abstract) < 50:
            return None

        # Verify title match if provided
        if title:
            api_title = data.get("title", "")
            overlap = title_word_overlap(title, api_title)
            if overlap < 0.5:
                logger.warning(f"OpenAlex title mismatch for {work_id}: {overlap:.2f} overlap")
                return None

        return {
            "title": data.get("title"),
            "abstract": abstract,
            "year": data.get("publication_year"),
            "doi": data.get("doi"),
            "source": "openalex",
        }

    except Exception as e:
        logger.warning(f"OpenAlex API error for {work_id}: {e}")
        return None


# ============================================================================
# Semantic Scholar API
# ============================================================================

def fetch_s2_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper from Semantic Scholar by DOI.
    """
    doi = normalize_doi(doi)
    if not doi:
        return None

    S2_RATE_LIMITER.wait()

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "title,abstract,year,externalIds"}

        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        if not data.get("abstract"):
            return None

        external_ids = data.get("externalIds", {})

        return {
            "title": data.get("title"),
            "abstract": data.get("abstract"),
            "year": data.get("year"),
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv"),
            "source": "semantic_scholar",
        }

    except Exception as e:
        logger.warning(f"Semantic Scholar API error for DOI {doi}: {e}")
        return None


def fetch_s2_by_arxiv(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper from Semantic Scholar by ArXiv ID.
    """
    arxiv_id = normalize_arxiv_id(arxiv_id)
    if not arxiv_id:
        return None

    S2_RATE_LIMITER.wait()

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        params = {"fields": "title,abstract,year,externalIds"}

        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        if not data.get("abstract"):
            return None

        external_ids = data.get("externalIds", {})

        return {
            "title": data.get("title"),
            "abstract": data.get("abstract"),
            "year": data.get("year"),
            "doi": external_ids.get("DOI"),
            "arxiv_id": external_ids.get("ArXiv"),
            "source": "semantic_scholar",
        }

    except Exception as e:
        logger.warning(f"Semantic Scholar API error for ArXiv {arxiv_id}: {e}")
        return None


def search_s2_by_title(title: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar by title.
    """
    if not title or len(title) < 10:
        return []

    S2_RATE_LIMITER.wait()

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": title,
            "limit": max_results,
            "fields": "title,abstract,year,externalIds",
        }

        resp = requests.get(url, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        results = []

        for paper in data.get("data", []):
            if not paper.get("abstract"):
                continue

            external_ids = paper.get("externalIds", {})
            results.append({
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "doi": external_ids.get("DOI"),
                "arxiv_id": external_ids.get("ArXiv"),
                "source": "semantic_scholar",
            })

        return results

    except Exception as e:
        logger.warning(f"Semantic Scholar search error for '{title[:50]}...': {e}")
        return []


# ============================================================================
# Main Enrichment Function
# ============================================================================

def ensure_valid_abstract(
    conn: Connection,
    work_id: str,
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Ensure work has a valid abstract, fetching from fallback sources if needed.

    Returns:
        (abstract, source)
        - abstract: The validated/enriched abstract
        - source: "cached" | "openalex" | "arxiv" | "unavailable"

    Flow:
    1. If abstract exists and is valid -> return (abstract, "cached")
    2. Try OpenAlex by work_id (most reliable - we have the ID)
    3. Try ArXiv by ID (if available)
    4. Try ArXiv search by title
    5. Return (None, "unavailable") if all fail
    """
    # Check if current abstract is valid
    is_valid, reason = is_abstract_valid(title, abstract)
    if is_valid:
        logger.debug(f"Abstract valid for {work_id}: {reason}")
        return abstract, "cached"

    logger.info(f"Abstract invalid for {work_id}: {reason}. Attempting enrichment...")

    # Build identity for verification
    identity = PaperIdentity(
        work_id=work_id,
        title=title,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
    )

    # Try OpenAlex first - most reliable since we have the work_id
    if work_id and work_id.startswith("W"):
        logger.debug(f"Trying OpenAlex by work_id: {work_id}")
        result = fetch_openalex_abstract(work_id, title)
        if result and result.get("abstract"):
            logger.info(f"Found valid abstract from OpenAlex for {work_id}")
            _cache_validated_abstract(conn, work_id, result["abstract"], "openalex")
            return result["abstract"], "openalex"

    # Try ArXiv by ID
    if arxiv_id:
        logger.debug(f"Trying ArXiv by ID: {arxiv_id}")
        result = fetch_arxiv_by_id(arxiv_id)
        if result and result.get("abstract"):
            is_match, confidence, match_reason = verify_paper_match(identity, result)
            if is_match:
                logger.info(f"Found valid abstract from ArXiv for {work_id} ({match_reason})")
                _cache_validated_abstract(conn, work_id, result["abstract"], "arxiv")
                return result["abstract"], "arxiv"

    # Try ArXiv search by title
    logger.debug(f"Trying ArXiv search for: {title[:50]}...")
    candidates = search_arxiv_by_title(title)
    for candidate in candidates:
        if candidate.get("abstract"):
            is_match, confidence, match_reason = verify_paper_match(identity, candidate, strict=True)
            if is_match:
                logger.info(f"Found valid abstract from ArXiv search for {work_id} ({match_reason})")
                _cache_validated_abstract(conn, work_id, candidate["abstract"], "arxiv")
                return candidate["abstract"], "arxiv"

    # NOTE: Semantic Scholar removed - was returning 429 rate limit errors without API key

    # All fallbacks failed - return None to signal no usable abstract
    # Don't return the invalid original - it would pollute LLM generation
    logger.warning(f"Could not find valid abstract for {work_id} from any source")
    return None, "unavailable"


def _cache_validated_abstract(
    conn: Connection,
    work_id: str,
    abstract: str,
    source: str,
) -> None:
    """
    Update works.abstract with validated abstract and track provenance.
    """
    try:
        conn.execute(
            text("""
                UPDATE works
                SET abstract = :abstract,
                    abstract_source = :source,
                    abstract_validated_at = now()
                WHERE work_id = :work_id
            """),
            {
                "work_id": work_id,
                "abstract": abstract,
                "source": source,
            },
        )
        conn.commit()
        logger.info(f"Cached validated abstract for {work_id} from {source}")
    except Exception as e:
        logger.warning(f"Failed to cache abstract for {work_id}: {e}")
