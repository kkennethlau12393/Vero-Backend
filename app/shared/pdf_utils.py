"""
Shared PDF download and text extraction utilities.

Used by Feature 3 (Novelty Assessment) and Feature 4 (Methodology Comparison)
for downloading open-access PDFs and extracting structured sections from
academic paper full text.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_TIMEOUT = 30
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB cap

# Section header pattern: matches numbered or unnumbered academic section headers.
# Examples: "1. Introduction", "2. Methods", "Introduction", "METHODS", "3 Methodology"
_SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.?\s+)?"  # optional number prefix like "1. " or "2 "
    r"("
    # Introduction / background
    r"introduction|background"
    r"|"
    # Methods
    r"method(?:ology|s)?|approach|experimental\s+(?:setup|design)|model"
    r"|"
    # Results / conclusion
    r"results?(?:\s+and\s+discussion)?|experiments?|evaluation|discussion|conclusions?"
    r"|"
    # References (stop marker)
    r"references|bibliography|acknowledgments?|acknowledgements?"
    r")"
    r"\s*$",  # header should be on its own line (possibly with trailing whitespace)
    re.IGNORECASE | re.MULTILINE,
)

# Classification of matched headers into target sections
_INTRO_PATTERNS = {"introduction", "background"}
_METHODS_PATTERNS = {"method", "methods", "methodology", "approach",
                     "experimental setup", "experimental design", "model"}
_RESULTS_PATTERNS = {"results", "result", "results and discussion",
                     "experiments", "experiment", "evaluation",
                     "discussion", "conclusion", "conclusions"}
_REFERENCES_PATTERNS = {"references", "bibliography",
                        "acknowledgments", "acknowledgements",
                        "acknowledgment", "acknowledgement"}

# Target truncation lengths per section (chars)
_INTRO_MAX = 1500
_METHODS_MAX = 1000
_RESULTS_MAX = 1000


# ---------------------------------------------------------------------------
# PDF download & text extraction
# ---------------------------------------------------------------------------

def download_and_extract_pdf(pdf_url: str) -> Optional[str]:
    """
    Download a PDF from a URL and extract full text using PyMuPDF.

    Returns the extracted text or None on failure.
    """
    if not pdf_url:
        return None

    try:
        import pymupdf
    except ImportError:
        logger.warning("pymupdf not installed -- cannot extract PDF text")
        return None

    try:
        resp = requests.get(
            pdf_url,
            timeout=PDF_TIMEOUT,
            stream=True,
            headers={"User-Agent": "Alexandria-Research-Tool/1.0"},
        )
        if resp.status_code != 200:
            logger.warning(f"PDF download failed ({resp.status_code}): {pdf_url[:80]}")
            return None

        # Read with size cap
        content = resp.content
        if len(content) > MAX_PDF_BYTES:
            logger.warning(f"PDF too large ({len(content)} bytes): {pdf_url[:80]}")
            return None

        if len(content) < 1000:
            logger.warning(f"PDF too small ({len(content)} bytes): {pdf_url[:80]}")
            return None

        # Extract text with PyMuPDF
        doc = pymupdf.open(stream=content, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()

        full_text = "\n".join(pages_text)
        # Strip NUL bytes (PostgreSQL TEXT columns cannot contain them)
        full_text = full_text.replace("\x00", "")
        if len(full_text) < 200:
            logger.warning(f"PDF text extraction yielded very little text: {pdf_url[:80]}")
            return None

        logger.info(
            f"PDF extracted: {len(full_text)} chars from {len(pages_text)} pages"
        )
        return full_text

    except Exception as e:
        logger.warning(f"PDF download/extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Section extraction from full text
# ---------------------------------------------------------------------------

def _classify_header(header_text: str) -> Optional[str]:
    """
    Classify a matched header into one of:
    'introduction', 'methods', 'results_conclusion', 'references', or None.
    """
    # Strip number prefix and normalize
    cleaned = re.sub(r"^\d+\.?\s*", "", header_text).strip().lower()

    if cleaned in _INTRO_PATTERNS:
        return "introduction"
    if cleaned in _METHODS_PATTERNS:
        return "methods"
    if cleaned in _RESULTS_PATTERNS:
        return "results_conclusion"
    if cleaned in _REFERENCES_PATTERNS:
        return "references"
    return None


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text to approximately max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    # Find the last space before or at max_chars
    cut = text.rfind(" ", 0, max_chars + 1)
    if cut == -1:
        # No space found; hard truncate
        return text[:max_chars]
    return text[:cut]


def extract_paper_sections(full_text: str) -> Dict[str, str]:
    """
    Extract key sections from academic paper full text.

    Returns a dict with keys: 'introduction', 'methods', 'results_conclusion'.
    Each value is the extracted section text, truncated to target length.
    Missing sections return empty strings.

    Section detection uses regex to find common academic section headers
    (numbered or unnumbered). Text between consecutive headers is extracted
    for the first match of each target section type.
    """
    result: Dict[str, str] = {
        "introduction": "",
        "methods": "",
        "results_conclusion": "",
    }

    if not full_text:
        return result

    # Find all section headers in the text
    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    if not matches:
        return result

    # Build list of (position, section_type) sorted by position
    headers = []
    for m in matches:
        section_type = _classify_header(m.group(0))
        if section_type:
            headers.append((m.start(), m.end(), section_type))

    if not headers:
        return result

    # Sort by position (should already be in order, but be safe)
    headers.sort(key=lambda h: h[0])

    # Extract text between consecutive headers
    # Track which target sections we've already found (take first match)
    found = set()

    for i, (start, end, section_type) in enumerate(headers):
        # Skip references -- we don't want to extract their content
        if section_type == "references":
            continue

        # Already found this section type? Skip.
        if section_type in found:
            continue

        # Determine end of this section's content
        if i + 1 < len(headers):
            section_end = headers[i + 1][0]
        else:
            # Last header -- take remaining text
            section_end = len(full_text)

        section_text = full_text[end:section_end].strip()

        if not section_text:
            continue

        # Truncate based on section type
        if section_type == "introduction":
            section_text = _truncate_at_word_boundary(section_text, _INTRO_MAX)
        elif section_type == "methods":
            section_text = _truncate_at_word_boundary(section_text, _METHODS_MAX)
        elif section_type == "results_conclusion":
            section_text = _truncate_at_word_boundary(section_text, _RESULTS_MAX)

        result[section_type] = section_text
        found.add(section_type)

        # If we've found all three, stop early
        if len(found) == 3:
            break

    return result
