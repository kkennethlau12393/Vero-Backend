"""
Paper content fetching for Feature 4 (Methodology Comparison).

Fetches full paper text from Semantic Scholar, extracts the methods
section, and caches results. Falls back to abstracts when full text
is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os

import requests
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from app.shared.pdf_utils import download_and_extract_pdf as _download_and_extract_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,tldr,openAccessPdf,isOpenAccess,externalIds"
S2_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"

API_TIMEOUT = 15
MAX_METHODS_CHARS = 12_000  # Cap methods text at ~3000 words

# Section header patterns for extracting methods from full text
METHODS_HEADERS = re.compile(
    r"^#{0,4}\s*(?:\d+\.?\s*)?"
    r"(?:method(?:ology|s)?|experimental\s+(?:setup|design)|"
    r"approach|materials?\s+and\s+methods?|"
    r"(?:proposed\s+)?(?:model|framework|algorithm)|"
    r"research\s+design|study\s+design|data\s+(?:and\s+)?method)"
    r"\b",
    re.IGNORECASE | re.MULTILINE,
)

# Headers that signal the END of a methods section
END_HEADERS = re.compile(
    r"^#{0,4}\s*(?:\d+\.?\s*)?"
    r"(?:results?|discussion|conclusion|evaluation|"
    r"findings|analysis|limitations?|future\s+work|"
    r"related\s+work|acknowledgment)"
    r"\b",
    re.IGNORECASE | re.MULTILINE,
)

# Rate limiter
class _RateLimiter:
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


_s2_limiter = _RateLimiter(calls_per_second=10.0)  # S2 allows 100/s with API key; 10/s gives headroom

S2_MAX_RETRIES = 3
S2_RETRY_BACKOFF = 1.0  # seconds; doubles each retry


def _s2_get(url: str, params: Dict[str, Any]) -> Optional[requests.Response]:
    """GET with rate limiting and retry-on-429."""
    for attempt in range(S2_MAX_RETRIES):
        _s2_limiter.wait()
        try:
            resp = requests.get(
                url, params=params, headers=_s2_headers(), timeout=API_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = S2_RETRY_BACKOFF * (2 ** attempt)
                logger.info(f"S2 rate limited, retrying in {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as e:
            logger.warning(f"S2 request failed: {e}")
            return None
    logger.warning(f"S2 rate limit retries exhausted for {url}")
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PaperContent:
    """Content fetched for a single paper."""

    work_id: str
    text: str  # Methods section or abstract
    source_quality: str  # "full_text" | "abstract_only"


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def _get_cached_content(conn: Connection, work_id: str) -> Optional[PaperContent]:
    """Check paper_full_text_cache for previously fetched content."""
    row = conn.execute(
        sa_text("""
            SELECT methods_text, full_text_available
            FROM paper_full_text_cache
            WHERE work_id = :work_id
        """),
        {"work_id": work_id},
    ).mappings().first()

    if not row:
        return None

    if row["full_text_available"] and row["methods_text"]:
        return PaperContent(
            work_id=work_id,
            text=row["methods_text"],
            source_quality="full_text",
        )
    elif row["methods_text"]:
        return PaperContent(
            work_id=work_id,
            text=row["methods_text"],
            source_quality="abstract_only",
        )
    return None


def _cache_content(
    conn: Connection,
    work_id: str,
    s2_paper_id: Optional[str],
    methods_text: Optional[str],
    full_text_available: bool,
) -> None:
    """Store fetched content in cache."""
    try:
        conn.execute(
            sa_text("""
                INSERT INTO paper_full_text_cache
                    (work_id, s2_paper_id, methods_text, full_text_available)
                VALUES (:work_id, :s2_paper_id, :methods_text, :full_text_available)
                ON CONFLICT (work_id) DO UPDATE SET
                    s2_paper_id = EXCLUDED.s2_paper_id,
                    methods_text = EXCLUDED.methods_text,
                    full_text_available = EXCLUDED.full_text_available,
                    fetched_at = now()
            """),
            {
                "work_id": work_id,
                "s2_paper_id": s2_paper_id,
                "methods_text": methods_text,
                "full_text_available": full_text_available,
            },
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache paper content for {work_id}: {e}")


# ---------------------------------------------------------------------------
# Methods section extraction
# ---------------------------------------------------------------------------

def extract_methods_section(full_text: str) -> Optional[str]:
    """
    Extract the methods/methodology section from full paper text.

    Looks for common section headers and extracts text until the next
    major section header (Results, Discussion, etc.).
    """
    match = METHODS_HEADERS.search(full_text)
    if not match:
        return None

    start = match.start()
    end_match = END_HEADERS.search(full_text, pos=match.end() + 50)
    if end_match:
        methods = full_text[start : end_match.start()]
    else:
        # Take up to MAX_METHODS_CHARS from the start of methods
        methods = full_text[start : start + MAX_METHODS_CHARS]

    methods = methods.strip()
    if len(methods) < 100:
        return None  # Too short to be a real methods section

    return methods[:MAX_METHODS_CHARS]


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------

def _s2_headers() -> Dict[str, str]:
    """Build S2 API headers with API key if available."""
    headers = {}
    api_key = os.environ.get(S2_API_KEY_ENV)
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _fetch_from_s2_by_id(
    doi: Optional[str], arxiv_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Fetch paper data from S2 by DOI or arxiv_id.
    Returns the API response dict or None.
    """
    identifiers = []
    if doi:
        identifiers.append(f"DOI:{doi}")
    if arxiv_id:
        clean_arxiv = re.sub(r"v\d+$", "", arxiv_id)
        identifiers.append(f"ARXIV:{clean_arxiv}")

    for ident in identifiers:
        url = f"{S2_API_BASE}/paper/{ident}"
        resp = _s2_get(url, {"fields": S2_FIELDS})
        if resp is None:
            continue
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            logger.debug(f"S2 paper not found: {ident}")
            continue
        else:
            logger.warning(f"S2 API error {resp.status_code} for {ident}")
            continue

    return None


def _fetch_from_s2_by_title(title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Search S2 by paper title and return the best match.

    Uses the S2 paper search endpoint, then verifies the match
    by checking title similarity.
    """
    if not title or len(title) < 10:
        return None

    try:
        url = f"{S2_API_BASE}/paper/search"
        fields = S2_FIELDS + ",year"
        params = {"query": title, "fields": fields, "limit": 5}
        if year:
            params["year"] = f"{year - 1}-{year + 1}"
        resp = _s2_get(url, params)
        if resp is None or resp.status_code != 200:
            if resp is not None:
                logger.warning(f"S2 search error {resp.status_code} for title: {title[:60]}")
            return None

        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            logger.debug(f"S2 search returned no results for: {title[:60]}")
            return None

        # Find best title match
        title_lower = title.lower().strip()
        for paper in papers:
            s2_title = (paper.get("title") or "").lower().strip()
            if _titles_match(title_lower, s2_title):
                # Verify year if we have it (reject if off by >1 year)
                if year and paper.get("year") and abs(paper["year"] - year) > 1:
                    logger.debug(f"S2 title match rejected (year mismatch {paper['year']} vs {year}): {s2_title[:60]}")
                    continue
                logger.info(f"S2 title match found for: {title[:60]}")
                return paper

        # Fuzzy match: require higher overlap AND year match
        for paper in papers:
            s2_title = (paper.get("title") or "").lower().strip()
            overlap = _word_overlap(title_lower, s2_title)
            if overlap >= 0.85:
                if year and paper.get("year") and abs(paper["year"] - year) > 1:
                    logger.debug(f"S2 fuzzy match rejected (year mismatch): {s2_title[:60]}")
                    continue
                logger.info(f"S2 fuzzy title match ({overlap:.0%}) for: {title[:60]}")
                return paper

        logger.debug(f"S2 no good title match for: {title[:60]}")
        return None

    except requests.RequestException as e:
        logger.warning(f"S2 search request failed: {e}")
        return None


def _pdf_matches_paper(full_text: str, title: str) -> bool:
    """
    Verify that extracted PDF text actually belongs to the expected paper.
    Checks that significant title words appear in the first ~5000 chars of the PDF.
    This catches cases where S2 returns a thesis/report PDF that merely cites the paper.
    """
    if not title or not full_text:
        return False
    import string
    # Extract significant words from title (skip short/common words)
    stop_words = {"a", "an", "the", "of", "and", "in", "for", "on", "to", "with", "by", "from", "using", "based", "via"}
    title_words = [
        w.lower().translate(str.maketrans("", "", string.punctuation))
        for w in title.split()
    ]
    significant = [w for w in title_words if len(w) > 3 and w not in stop_words]
    if not significant:
        return True  # Can't verify, assume match

    # Check first ~5000 chars of PDF (title page + abstract area)
    header = full_text[:5000].lower()
    matches = sum(1 for w in significant if w in header)
    ratio = matches / len(significant)
    if ratio < 0.5:
        logger.debug(f"PDF title word match ratio: {ratio:.0%} ({matches}/{len(significant)})")
    return ratio >= 0.5


def _titles_match(a: str, b: str) -> bool:
    """Check if two titles are essentially the same."""
    # Normalize: remove punctuation, extra spaces
    import string
    def norm(s: str) -> str:
        return " ".join(s.translate(str.maketrans("", "", string.punctuation)).split())
    return norm(a) == norm(b)


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _fetch_from_s2(
    title: str,
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch paper data from Semantic Scholar.

    Strategy:
    1. Try by DOI/arxiv_id (exact match, fast)
    2. Fall back to title search (fuzzy match, with year verification)
    """
    # Try by identifier first
    if doi or arxiv_id:
        result = _fetch_from_s2_by_id(doi, arxiv_id)
        if result:
            return result

    # Fall back to title search
    return _fetch_from_s2_by_title(title, year=year)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_paper_content(conn: Connection, work_id: str) -> PaperContent:
    """
    Fetch methodology-relevant content for a single paper.

    Priority:
    1. Check cache
    2. Try Semantic Scholar for full text / extended abstract
    3. Fall back to abstract from works table

    Always caches the result.
    """
    # 1. Check cache
    cached = _get_cached_content(conn, work_id)
    if cached:
        logger.debug(f"Content cache hit for {work_id}: {cached.source_quality}")
        return cached

    # 2. Load paper metadata from works table
    row = conn.execute(
        sa_text("""
            SELECT title, abstract, doi, arxiv_id, year
            FROM works
            WHERE work_id = :work_id
        """),
        {"work_id": work_id},
    ).mappings().first()

    if not row:
        logger.warning(f"Work {work_id} not found in works table")
        return PaperContent(work_id=work_id, text="", source_quality="abstract_only")

    title = row.get("title") or ""
    doi = row.get("doi")
    arxiv_id = row.get("arxiv_id")
    year = row.get("year")
    abstract = row.get("abstract") or ""

    # 3. Try Semantic Scholar (by DOI/arxiv_id, then by title search)
    s2_data = _fetch_from_s2(title, doi, arxiv_id, year=year)
    s2_paper_id = None
    methods_text = None
    full_text_available = False

    if s2_data:
        s2_paper_id = s2_data.get("paperId")

        # S2 sometimes returns a richer abstract or TLDR
        s2_abstract = s2_data.get("abstract") or ""
        tldr = s2_data.get("tldr", {})
        tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""

        # Prefer S2 abstract (higher quality) when available
        if s2_abstract:
            abstract = s2_abstract

        # Append TLDR if available (gives methodology hints)
        if tldr_text:
            abstract = f"{abstract}\n\nTL;DR: {tldr_text}"

        # 4. Try to get full text via open-access PDF
        pdf_info = s2_data.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "")
        if pdf_url:
            logger.info(f"Attempting PDF download for {work_id}: {pdf_url[:80]}")
            full_text = _download_and_extract_pdf(pdf_url)
            if full_text and not _pdf_matches_paper(full_text, title):
                logger.warning(f"PDF content does not match paper title for {work_id}, skipping PDF")
                full_text = None
            if full_text:
                # Try to extract just the methods section
                methods = extract_methods_section(full_text)
                if methods:
                    logger.info(
                        f"Methods section extracted for {work_id}: "
                        f"{len(methods)} chars"
                    )
                    _cache_content(conn, work_id, s2_paper_id, methods, True)
                    return PaperContent(
                        work_id=work_id,
                        text=methods,
                        source_quality="full_text",
                    )
                else:
                    # No clear methods section — use full text (truncated)
                    truncated = full_text[:MAX_METHODS_CHARS]
                    logger.info(
                        f"No methods section found for {work_id}, "
                        f"using full text ({len(truncated)} chars)"
                    )
                    _cache_content(conn, work_id, s2_paper_id, truncated, True)
                    return PaperContent(
                        work_id=work_id,
                        text=truncated,
                        source_quality="full_text",
                    )

    # 5. Fall back to abstract
    _cache_content(conn, work_id, s2_paper_id, abstract, full_text_available=False)

    return PaperContent(
        work_id=work_id,
        text=abstract,
        source_quality="abstract_only",
    )


def fetch_papers_content(
    conn: Connection,
    work_ids: list[str],
) -> list[PaperContent]:
    """
    Fetch content for multiple papers concurrently.

    Uses ThreadPoolExecutor to parallelize S2 API calls.
    """
    results: dict[str, PaperContent] = {}

    def _fetch_one(wid: str) -> PaperContent:
        return fetch_paper_content(conn, wid)

    with ThreadPoolExecutor(max_workers=min(4, len(work_ids))) as pool:
        futures = {pool.submit(_fetch_one, wid): wid for wid in work_ids}
        for future in as_completed(futures):
            wid = futures[future]
            try:
                results[wid] = future.result()
            except Exception as e:
                logger.error(f"Failed to fetch content for {wid}: {e}")
                results[wid] = PaperContent(
                    work_id=wid, text="", source_quality="abstract_only"
                )

    # Return in original order
    return [results[wid] for wid in work_ids]
