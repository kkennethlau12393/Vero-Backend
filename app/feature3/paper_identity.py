"""
Paper identity verification utilities for Feature 3.

This module provides functions to verify that two papers are the same work,
critical for safely enriching paper data from multiple sources (ArXiv, Semantic Scholar).

Also provides shared utility functions like abstract decoding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class PaperIdentity:
    """Canonical identifiers for a paper."""

    work_id: str
    title: str
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Normalize DOI for comparison.

    - Lowercase
    - Remove URL prefix (https://doi.org/)
    - Strip whitespace
    """
    if not doi:
        return None

    doi = doi.lower().strip()

    # Remove common prefixes
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "doi.org/",
        "doi:",
    ]
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break

    return doi if doi else None


def normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    """
    Normalize arXiv ID for comparison.

    Handles formats:
    - 2301.12345
    - 2301.12345v2
    - arXiv:2301.12345
    - https://arxiv.org/abs/2301.12345
    - cs.LG/0601001 (old format)
    """
    if not arxiv_id:
        return None

    arxiv_id = arxiv_id.strip()

    # Remove URL prefix
    if "arxiv.org" in arxiv_id.lower():
        arxiv_id = re.sub(r'https?://arxiv\.org/(abs|pdf)/', '', arxiv_id, flags=re.IGNORECASE)

    # Remove arXiv: prefix
    arxiv_id = re.sub(r'^arxiv:', '', arxiv_id, flags=re.IGNORECASE)

    # Remove version suffix for comparison (2301.12345v2 -> 2301.12345)
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

    # Remove .pdf suffix
    arxiv_id = re.sub(r'\.pdf$', '', arxiv_id, flags=re.IGNORECASE)

    return arxiv_id.strip() if arxiv_id else None


def normalize_title(title: Optional[str]) -> str:
    """
    Normalize title for comparison.

    - Lowercase
    - Remove punctuation
    - Normalize unicode
    - Remove common prefixes/suffixes
    - Remove multiple spaces
    """
    if not title:
        return ""

    # Normalize unicode
    title = unicodedata.normalize('NFKD', title)

    # Lowercase
    title = title.lower()

    # Remove punctuation except spaces
    title = re.sub(r'[^\w\s]', ' ', title)

    # Remove common suffixes that vary between versions
    suffixes_to_remove = [
        r'\s+v\d+$',           # v1, v2, etc.
        r'\s+version\s+\d+$',  # version 1, etc.
        r'\s+arxiv$',
        r'\s+preprint$',
    ]
    for suffix in suffixes_to_remove:
        title = re.sub(suffix, '', title)

    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()

    return title


def title_word_overlap(title1: str, title2: str) -> float:
    """
    Calculate word-level Jaccard similarity between two titles.

    Returns: 0.0-1.0 (1.0 = identical word sets)
    """
    words1 = set(normalize_title(title1).split())
    words2 = set(normalize_title(title2).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def title_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles using multiple methods.

    Uses:
    1. Word-level Jaccard similarity
    2. Character-level similarity (for typos)

    Returns max of both approaches.
    """
    t1 = normalize_title(title1)
    t2 = normalize_title(title2)

    if not t1 or not t2:
        return 0.0

    # Exact match after normalization
    if t1 == t2:
        return 1.0

    # Word-level Jaccard
    jaccard = title_word_overlap(title1, title2)

    # Simple character-level similarity (ratio of common chars)
    # This helps catch minor typos
    chars1 = set(t1.replace(' ', ''))
    chars2 = set(t2.replace(' ', ''))
    if chars1 and chars2:
        char_sim = len(chars1 & chars2) / len(chars1 | chars2)
    else:
        char_sim = 0.0

    return max(jaccard, char_sim)


def verify_paper_match(
    source: PaperIdentity,
    candidate: Dict[str, Any],
    strict: bool = True,
) -> Tuple[bool, float, str]:
    """
    Verify that a candidate paper matches the source paper identity.

    Args:
        source: The paper we're trying to enrich
        candidate: A candidate paper from external source (dict with title, year, doi, arxiv_id)
        strict: If True, require high confidence; if False, allow lower confidence matches

    Returns:
        (is_match, confidence, reason)
        - is_match: True if papers are the same
        - confidence: 0.0-1.0 confidence score
        - reason: Explanation of match/mismatch
    """
    candidate_doi = normalize_doi(candidate.get("doi"))
    candidate_arxiv = normalize_arxiv_id(candidate.get("arxiv_id") or candidate.get("arxivId"))
    candidate_title = candidate.get("title", "")
    candidate_year = candidate.get("year") or candidate.get("publication_year")

    source_doi = normalize_doi(source.doi)
    source_arxiv = normalize_arxiv_id(source.arxiv_id)

    # 1. DOI match - definitive
    if source_doi and candidate_doi:
        if source_doi == candidate_doi:
            return True, 1.0, "DOI match"
        else:
            # Different DOIs = definitely different papers
            return False, 0.0, f"DOI mismatch: {source_doi} vs {candidate_doi}"

    # 2. ArXiv ID match - definitive
    if source_arxiv and candidate_arxiv:
        if source_arxiv == candidate_arxiv:
            return True, 0.98, "ArXiv ID match"
        # Different arXiv IDs might still be same paper (different versions handled above)
        # so don't reject outright

    # 3. Title + year matching
    sim = title_similarity(source.title, candidate_title)

    # Year comparison
    year_match = False
    year_close = False
    if source.year and candidate_year:
        year_diff = abs(int(source.year) - int(candidate_year))
        year_match = year_diff == 0
        year_close = year_diff <= 1  # Allow 1 year tolerance for preprints

    # High title similarity + year match
    if sim >= 0.90 and year_match:
        return True, 0.95, f"Title match ({sim:.2f}) with exact year"

    if sim >= 0.85 and year_close:
        return True, 0.88, f"Title match ({sim:.2f}) with year within 1 year"

    # In strict mode, require more evidence
    if strict:
        if sim >= 0.80 and year_match:
            return True, 0.80, f"Title match ({sim:.2f}) with exact year (strict)"
        return False, sim, f"Insufficient match: title similarity {sim:.2f}"

    # Non-strict mode - allow title-only matching for fallbacks
    if sim >= 0.85:
        return True, 0.75, f"Title match ({sim:.2f}) without year verification"

    return False, sim, f"No match: title similarity {sim:.2f}"


def extract_arxiv_id_from_url(url: Optional[str]) -> Optional[str]:
    """Extract arXiv ID from various URL formats."""
    if not url:
        return None

    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arxiv:(\d+\.\d+)',
        r'(\d{4}\.\d{4,5})',  # New format: YYMM.NNNNN
        r'([a-z-]+/\d+)',      # Old format: cs.LG/0601001
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return normalize_arxiv_id(match.group(1))

    return None


def extract_doi_from_url(url: Optional[str]) -> Optional[str]:
    """Extract DOI from various URL formats."""
    if not url:
        return None

    # Match DOI pattern: 10.XXXX/...
    match = re.search(r'(10\.\d{4,}/[^\s]+)', url)
    if match:
        return normalize_doi(match.group(1))

    return None


# ============================================================================
# OpenAlex Abstract Decoding
# ============================================================================

def decode_openalex_abstract(abstract_inverted_index: Any) -> Optional[str]:
    """
    Reconstruct abstract text from OpenAlex's inverted index format.

    OpenAlex stores abstracts as {word: [position1, position2, ...]} dicts.
    This function reconstructs the original text from that format.

    Args:
        abstract_inverted_index: Dict mapping words to position lists

    Returns:
        Reconstructed abstract text, or None if input is invalid
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return None

    word_positions = []
    for word, positions in abstract_inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            try:
                word_positions.append((int(pos), word))
            except (ValueError, TypeError):
                continue

    if not word_positions:
        return None

    word_positions.sort()
    return " ".join(w for _, w in word_positions).strip() or None
