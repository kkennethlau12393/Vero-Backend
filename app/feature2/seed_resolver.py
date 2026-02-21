"""
Seed resolver for Feature 2 (Ranking) - Convert DOI/work_id/PDF to title.

This module provides functions to resolve various paper identifiers to titles,
which are then fed into the existing `generate_topic_query_from_title()` workflow.

Reuses Feature 1's battle-tested infrastructure:
- DOI lookup (OpenAlex + Semantic Scholar)
- work_id fetch (supports W.../S2:.../AX:... formats)
- PDF title extraction (metadata + text parsing)

All functions raise ValueError on failure for consistent error handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_doi_to_title(doi: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a DOI to a paper title for ranking.

    Strategy:
    1. Try OpenAlex DOI lookup (primary - fastest, most complete)
    2. Fallback to Semantic Scholar DOI lookup
    3. Raise ValueError if not found in either source

    Args:
        doi: DOI string (e.g., "10.48550/arXiv.1706.03762")

    Returns:
        Tuple of (title, metadata)
        metadata contains: {work_id, year, cited_by_count, source, extraction_method}

    Raises:
        ValueError: If DOI not found in OpenAlex or Semantic Scholar

    Example:
        >>> title, meta = resolve_doi_to_title("10.48550/arXiv.1706.03762")
        >>> print(title)
        "Attention Is All You Need"
        >>> print(meta["source"])
        "openalex"
    """
    from app.feature1.citation_map_service import (
        _lookup_openalex_by_doi,
        _lookup_s2_paper_by_doi,
        fetch_seed_paper_details,
    )

    # Clean DOI (remove "doi:" prefix if present)
    doi = doi.strip()
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()

    logger.info(f"Resolving DOI to title: {doi}")

    # Try OpenAlex first (primary source)
    work_id = _lookup_openalex_by_doi(doi)

    if work_id:
        # Fetch full paper details
        try:
            paper = fetch_seed_paper_details(work_id)
            if paper and paper.get("title"):
                logger.info(f"DOI {doi} → OpenAlex work_id {work_id} → title: '{paper['title'][:50]}...'")
                return paper["title"], {
                    "work_id": work_id,
                    "year": paper.get("year"),
                    "cited_by_count": paper.get("cited_by_count", 0),
                    "source": "openalex",
                    "extraction_method": "doi_lookup",
                }
        except Exception as e:
            logger.warning(f"OpenAlex fetch failed for {work_id}: {e}, trying S2 fallback")

    # Fallback to Semantic Scholar
    logger.info(f"DOI not in OpenAlex or fetch failed, trying Semantic Scholar: {doi}")
    s2_paper = _lookup_s2_paper_by_doi(doi)

    if s2_paper and s2_paper.get("title"):
        logger.info(f"DOI {doi} → S2 paper → title: '{s2_paper['title'][:50]}...'")
        return s2_paper["title"], {
            "work_id": f"S2:{s2_paper.get('paperId', 'unknown')}",
            "year": s2_paper.get("year"),
            "cited_by_count": s2_paper.get("citationCount", 0),
            "source": "semantic_scholar",
            "extraction_method": "doi_lookup",
        }

    # Not found in either source
    raise ValueError(f"DOI {doi} not found in OpenAlex or Semantic Scholar")


def resolve_work_id_to_title(work_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a work_id to a paper title for ranking.

    Supports multiple work_id formats:
    - W... (OpenAlex)
    - S2:... (Semantic Scholar)
    - AX:... (ArXiv via S2 bridge)

    Args:
        work_id: Work ID string (e.g., "W2963663278", "S2:204e3073...", "AX:1706.03762")

    Returns:
        Tuple of (title, metadata)
        metadata contains: {work_id, year, cited_by_count, source, extraction_method}

    Raises:
        ValueError: If work_id not found or has no title

    Example:
        >>> title, meta = resolve_work_id_to_title("W2963663278")
        >>> print(title)
        "Attention Is All You Need"
    """
    from app.feature1.citation_map_service import fetch_seed_paper_details

    work_id = work_id.strip()
    logger.info(f"Resolving work_id to title: {work_id}")

    try:
        paper = fetch_seed_paper_details(work_id)
    except Exception as e:
        raise ValueError(f"Work ID {work_id} lookup failed: {str(e)}")

    if not paper:
        raise ValueError(f"Work ID {work_id} not found")

    if not paper.get("title"):
        raise ValueError(f"Work ID {work_id} has no title")

    source = _detect_source_from_work_id(work_id)
    logger.info(f"work_id {work_id} → title: '{paper['title'][:50]}...'")

    return paper["title"], {
        "work_id": work_id,
        "year": paper.get("year"),
        "cited_by_count": paper.get("cited_by_count", 0),
        "source": source,
        "extraction_method": "work_id_lookup",
    }


def resolve_pdf_to_title(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a PDF to a paper title for ranking.

    Strategy:
    1. Extract title + ArXiv ID from PDF metadata/text
    2. If ArXiv ID found: use ArXiv DOI → canonical title (more reliable)
    3. Otherwise: use extracted title directly

    Args:
        pdf_bytes: PDF file bytes

    Returns:
        Tuple of (title, metadata)
        metadata contains: {work_id, year, cited_by_count, source, extraction_method}

    Raises:
        ValueError: If title extraction fails

    Example:
        >>> with open("paper.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> title, meta = resolve_pdf_to_title(pdf_bytes)
    """
    from app.feature1.pdf_parser import extract_metadata_from_pdf

    logger.info("Extracting title from PDF")

    try:
        metadata = extract_metadata_from_pdf(pdf_bytes)
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {str(e)}")

    title = metadata.get("title")
    arxiv_id = metadata.get("arxiv_id")

    if not title:
        raise ValueError("Could not extract title from PDF")

    # If ArXiv ID found, use it to get canonical title
    if arxiv_id:
        logger.info(f"PDF has ArXiv ID {arxiv_id}, fetching canonical title via DOI")
        try:
            arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
            canonical_title, doi_metadata = resolve_doi_to_title(arxiv_doi)

            # Use canonical title from ArXiv (more reliable than extracted)
            logger.info(f"ArXiv DOI resolved to canonical title: '{canonical_title[:50]}...'")
            return canonical_title, {
                **doi_metadata,
                "extraction_method": "pdf_arxiv_doi",
                "pdf_extracted_title": title,  # Keep for debugging/comparison
            }
        except ValueError as e:
            # ArXiv DOI failed, fall back to extracted title
            logger.warning(f"ArXiv DOI {arxiv_doi} lookup failed: {e}, using extracted title")

    # No ArXiv ID or ArXiv lookup failed - use extracted title
    logger.info(f"Using PDF-extracted title: '{title[:50]}...'")
    return title, {
        "work_id": None,
        "year": metadata.get("year"),
        "cited_by_count": 0,
        "source": "pdf_extraction",
        "extraction_method": "pdf_metadata" if metadata.get("from_metadata") else "pdf_text",
    }


def _detect_source_from_work_id(work_id: str) -> str:
    """
    Detect source from work_id format.

    Args:
        work_id: Work ID string

    Returns:
        Source name: "openalex", "semantic_scholar", "arxiv", or "unknown"
    """
    work_id = work_id.strip().upper()

    if work_id.startswith("W"):
        return "openalex"
    elif work_id.startswith("S2:"):
        return "semantic_scholar"
    elif work_id.startswith("AX:"):
        return "arxiv"
    else:
        return "unknown"
