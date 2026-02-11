"""
Resolve full-text access links for papers.

Priority:
1. LibKey (if configured) - institutional full-text link
2. OpenAlex OA PDF URL - direct open access PDF
3. Institutional proxy + DOI - proxied DOI link for closed access
4. Plain DOI link - fallback
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def resolve_access_link(
    *,
    doi: Optional[str],
    is_open_access: Optional[bool],
    oa_pdf_url: Optional[str],
    proxy_prefix: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    libkey_library_id: Optional[str] = None,
) -> dict:
    """
    Returns:
        {
            "access_status": "open" | "closed" | "unknown",
            "pdf_url": Optional[str],       # best available link
            "doi_url": Optional[str],        # plain DOI link
            "source": "oa_direct" | "libkey" | "proxy" | "doi" | None,
        }
    """
    doi_url = f"https://doi.org/{doi}" if doi else None

    # 1) Try LibKey if configured
    if libkey_api_key and libkey_library_id and doi:
        libkey_url = _try_libkey(doi, libkey_api_key, libkey_library_id)
        if libkey_url:
            return {
                "access_status": "open",
                "pdf_url": libkey_url,
                "doi_url": doi_url,
                "source": "libkey",
            }

    # 2) Open access PDF from OpenAlex
    if oa_pdf_url and is_open_access:
        return {
            "access_status": "open",
            "pdf_url": oa_pdf_url,
            "doi_url": doi_url,
            "source": "oa_direct",
        }

    # 3) Closed access with institutional proxy
    if doi and proxy_prefix:
        return {
            "access_status": "closed",
            "pdf_url": f"{proxy_prefix}{doi_url}",
            "doi_url": doi_url,
            "source": "proxy",
        }

    # 4) Fallback: plain DOI
    if doi:
        return {
            "access_status": "closed" if is_open_access is False else "unknown",
            "pdf_url": doi_url,
            "doi_url": doi_url,
            "source": "doi",
        }

    return {
        "access_status": "unknown",
        "pdf_url": None,
        "doi_url": None,
        "source": None,
    }


def _try_libkey(doi: str, api_key: str, library_id: str) -> Optional[str]:
    """Query LibKey Article API for a full-text link. Returns URL or None."""
    try:
        resp = requests.get(
            f"https://full.libkey.io/v1/libraries/{library_id}/articles/doi/{doi}",
            params={"key": api_key},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # LibKey returns fullTextFile (PDF) or contentLocation (publisher page)
        return data.get("fullTextFile") or data.get("contentLocation")
    except Exception:
        logger.debug("LibKey lookup failed for DOI %s", doi, exc_info=True)
        return None
