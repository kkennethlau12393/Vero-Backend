"""
Seed conversion utilities for Feature 2.

Converts various seed formats (work_id, DOI, PDF) to paper titles,
then generates search queries from those titles.

Reuses code from Feature 1 where possible to avoid duplication.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from pathlib import Path

# Import reusable functions from Feature 1
from app.feature1.pdf_parser import extract_title_from_pdf
from app.feature1.citation_map_service import _lookup_openalex_by_doi

# Import title-to-query converter from Feature 2
from app.feature2.title_to_query import generate_topic_query_from_title

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Constants for OpenAlex API
from app.shared.oa_keys import get_oa_api_key
OPENALEX_TIMEOUT = 15


def get_work_metadata(work_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for an OpenAlex work_id.

    This is a simplified version of Feature 1's fetch_seed_paper_details(),
    extracting only the fields needed for title extraction.

    Args:
        work_id: OpenAlex work ID (e.g., "W2964141474")

    Returns:
        Dict with: work_id, title, year, cited_by_count
        None if work not found or API error
    """
    if not work_id:
        return None

    # Handle prefixed IDs - Feature 2 only uses OpenAlex work_ids
    if not work_id.startswith("W"):
        logger.warning(f"Feature 2 only supports OpenAlex work_ids (W prefix), got: {work_id}")
        return None

    try:
        url = f"https://api.openalex.org/works/{work_id}"
        params = {}
        oa_key = get_oa_api_key()
        if oa_key:
            params["api_key"] = oa_key

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        return {
            "work_id": work_id,
            "title": data.get("title"),
            "year": data.get("publication_year"),
            "cited_by_count": data.get("cited_by_count") or 0,
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Work not found in OpenAlex: {work_id}")
        else:
            logger.warning(f"OpenAlex API error for {work_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {work_id}: {e}")
        return None


def work_id_to_title(work_id: str) -> Optional[str]:
    """
    Convert OpenAlex work_id to paper title.

    Args:
        work_id: OpenAlex work ID (e.g., "W2964141474")

    Returns:
        Paper title or None if not found
    """
    metadata = get_work_metadata(work_id)
    if metadata and metadata.get("title"):
        return metadata["title"]
    return None


def doi_to_title(doi: str) -> Optional[str]:
    """
    Convert DOI to paper title.

    Strategy:
    1. Lookup OpenAlex work_id by DOI (reuse F1 code)
    2. Fetch work metadata to get title

    Args:
        doi: DOI string (e.g., "10.48550/arXiv.1706.03762")

    Returns:
        Paper title or None if not found
    """
    if not doi:
        return None

    # Reuse Feature 1's DOI lookup (already handles S2 fallback)
    work_id = _lookup_openalex_by_doi(doi)
    if not work_id:
        logger.warning(f"DOI not found in OpenAlex: {doi}")
        return None

    # Get metadata to extract title
    metadata = get_work_metadata(work_id)
    if metadata and metadata.get("title"):
        return metadata["title"]

    logger.warning(f"Found work_id {work_id} for DOI {doi}, but no title available")
    return None


def pdf_to_title(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract title from PDF bytes.

    Reuses Feature 1's PDF parser (already handles metadata + text extraction).

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Extracted title or None if extraction failed
    """
    return extract_title_from_pdf(pdf_bytes)


def seed_to_query(
    *,
    work_id: Optional[str] = None,
    doi: Optional[str] = None,
    title: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
) -> Optional[str]:
    """
    Convert any seed format to a search query.

    Priority: title > work_id > doi > pdf_bytes (direct title is fastest)

    Flow:
    1. Extract/lookup title from seed
    2. Convert title → query via LLM (reuse title_to_query.py)

    Args:
        work_id: OpenAlex work ID
        doi: DOI
        title: Paper title (direct)
        pdf_bytes: PDF file bytes

    Returns:
        Generated search query or None if conversion failed
    """
    # Determine title from seed
    paper_title = None

    if title:
        # Direct title - fastest path
        paper_title = title
        logger.info(f"Using direct title: '{title[:50]}...'")
    elif work_id:
        paper_title = work_id_to_title(work_id)
        if paper_title:
            logger.info(f"Converted work_id {work_id} → title: '{paper_title[:50]}...'")
    elif doi:
        paper_title = doi_to_title(doi)
        if paper_title:
            logger.info(f"Converted DOI {doi} → title: '{paper_title[:50]}...'")
    elif pdf_bytes:
        paper_title = pdf_to_title(pdf_bytes)
        if paper_title:
            logger.info(f"Extracted title from PDF: '{paper_title[:50]}...'")

    if not paper_title:
        logger.warning("Failed to extract title from any seed format")
        return None

    # Convert title → query (reuse existing F2 logic)
    query = generate_topic_query_from_title(paper_title)
    return query
