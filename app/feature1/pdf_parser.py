"""PDF parsing utilities for Feature 1: Citation Map."""
import re
import fitz  # PyMuPDF
from typing import Optional

# Matches ArXiv IDs like "arXiv:1409.3215v3", "arXiv: 1409.3215", "arxiv:2006.11477"
_ARXIV_ID_RE = re.compile(r'arXiv[:\s]+(\d{4}\.\d{4,5})(?:v\d+)?', re.IGNORECASE)


def _extract_arxiv_id(doc: fitz.Document) -> Optional[str]:
    """Extract ArXiv ID from PDF metadata or first page text.

    Returns the base ArXiv ID (e.g., '1409.3215') without version suffix.
    """
    # Check metadata title (some ArXiv PDFs store the ID here)
    meta_title = doc.metadata.get("title", "") or ""
    m = _ARXIV_ID_RE.search(meta_title)
    if m:
        return m.group(1)

    # Check first page text (ArXiv stamp usually in header)
    if len(doc) > 0:
        first_page_text = doc[0].get_text()[:1000]  # Only need the top
        m = _ARXIV_ID_RE.search(first_page_text)
        if m:
            return m.group(1)

    return None


def _is_metadata_title_valid(title: str) -> bool:
    """Check if a PDF metadata title is an actual paper title (not an ArXiv stamp, etc.)."""
    if not title or len(title) <= 10 or len(title) >= 200:
        return False
    t = title.lower()
    # Reject ArXiv ID stamps (e.g., "arXiv:1409.3215v3  [cs.CL]  14 Dec 2014")
    if _ARXIV_ID_RE.search(title):
        return False
    # Reject submission/revision timestamps
    if re.search(r'submitted|revised|published|accepted|received', t):
        return False
    return True


def extract_title_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """Extract title from PDF.

    Strategy:
    1. Check PDF metadata first (with content validation)
    2. Extract first page text and find title (usually largest/bold text at top)
    3. Heuristic: first non-empty line that's not too long (< 200 chars)
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Strategy 1: Check PDF metadata (with validation)
        metadata_title = doc.metadata.get("title", "").strip()
        if _is_metadata_title_valid(metadata_title):
            doc.close()
            return metadata_title

        # Strategy 2: Extract from first page
        if len(doc) == 0:
            doc.close()
            return None

        first_page = doc[0]
        text = first_page.get_text()

        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Common header/footer patterns to skip
        skip_patterns = [
            'provided proper attribution',
            'published as a',
            'conference paper',
            'journal of',
            'proceedings of',
            'copyright',
            'all rights reserved',
            'biomedcentral',
            'biomed central',
            'springer',
            'elsevier',
            'ieee',
            'acm',
            'arxiv.org',
            'preprint',
            'reproduce the',
            'permission to',
            'granted',
            'solely for use',
            'scholarly works',
            'open access',
            'research article',
            'original article',
            'bmc ',   # BMC Bioinformatics, BMC Genomics, etc.
            'plos ',  # PLOS ONE, PLOS Biology, etc.
            'submitted',   # Journal submission lines (e.g., "Submitted 1/20; Revised 6/20")
            'revised',
            'accepted',
            'received',
        ]

        def _is_title_candidate(line: str) -> bool:
            """Check if a line could be part of a title."""
            line_lower = line.lower()
            if len(line) < 10:
                return False
            if len(line) > 200:
                return False
            if any(x in line_lower for x in ['http', '@', 'arxiv:', 'doi:']):
                return False
            if re.match(r'^[\d\s\-\/\.]+$', line):
                return False
            if any(pattern in line_lower for pattern in skip_patterns):
                return False
            if len(line.split()) < 2:
                return False
            # Skip page numbers like "Page 1 of 13", "1/13", etc.
            if re.match(r'^page\s+\d+\s+of\s+\d+', line_lower):
                return False
            # Skip lines that are mostly numbers with "of" (pagination)
            if re.match(r'^\d+\s+of\s+\d+', line_lower):
                return False
            # Skip short sentence fragments ending with period (e.g., "scholarly works.")
            words = line.split()
            if len(words) <= 3 and line.rstrip().endswith('.'):
                return False
            return True

        def _looks_like_authors(line: str) -> bool:
            """Detect author lines: names with academic markers (*, +, †, ‡)."""
            # Author markers: ∗, *, +, †, ‡, §, ¶, superscript numbers
            if any(c in line for c in '∗†‡§¶&'):
                return True
            # Pattern: "Name Name, Name Name" or "Name Name and Name Name"
            if re.match(r'^[A-Z][a-z]+\s+[A-Z]', line) and (' and ' in line or '&' in line):
                return True
            return False

        # Find title — handle multi-line titles (e.g., VGG paper)
        # Strategy: find first title-like line, then check if consecutive lines
        # continue the title (e.g., ALL CAPS lines that form a single title)
        for i, line in enumerate(lines[:20]):
            if not _is_title_candidate(line):
                continue
            if len(line) < 15 and len(line.split()) < 3:
                continue
            # Skip lines that look like author names
            if _looks_like_authors(line):
                continue

            # Check if this single line contains author names appended
            # e.g., "TITLE TEXT Karen Simonyan∗& Andrew Zisserman+"
            # Split at the first author marker and take the title part
            author_split = re.split(r'\s+(?=[A-Z][a-z]+[∗\*†‡§¶\+]+)', line)
            if len(author_split) > 1:
                line = author_split[0].strip()
                if len(line) >= 15:
                    doc.close()
                    return line

            # Found first title-like line — check for multi-line continuation
            # Only continue if the current title text is clearly incomplete
            # (ends with a connector word like "for", "of", "in", etc.)
            incomplete_endings = {'for', 'of', 'in', 'on', 'and', 'the', 'with', 'to', 'a', 'an', 'by', 'from', 'via', 'using'}

            title_parts = [line]
            for j in range(i + 1, min(i + 4, len(lines))):
                # Check if current title text is clearly incomplete
                last_word = title_parts[-1].split()[-1].lower().rstrip('.,;:')
                if last_word not in incomplete_endings:
                    # Title appears complete — only continue for ALL CAPS style
                    current_text = " ".join(title_parts)
                    upper_ratio = sum(1 for c in current_text if c.isupper()) / max(len(current_text), 1)
                    if upper_ratio <= 0.5:
                        break  # Not all-caps style, title is likely complete

                next_line = lines[j]
                # Stop if next line looks like authors
                if _looks_like_authors(next_line):
                    break
                if not _is_title_candidate(next_line):
                    break
                # Stop if next line looks like an author list (has commas + short words)
                if re.match(r'^[A-Z][a-z]+ [A-Z]', next_line) and ',' in next_line:
                    break
                # Stop if next line has email-like or affiliation patterns
                if any(x in next_line.lower() for x in ['university', 'department', 'institute', 'abstract']):
                    break
                # Continue if both lines are mostly uppercase (like VGG paper)
                current_text = " ".join(title_parts)
                current_upper_ratio = sum(1 for c in current_text if c.isupper()) / max(len(current_text), 1)
                next_upper_ratio = sum(1 for c in next_line if c.isupper()) / max(len(next_line), 1)
                if current_upper_ratio > 0.5 and next_upper_ratio > 0.5:
                    title_parts.append(next_line)
                    continue
                # Continue if title is incomplete (ends with connector) and combined is reasonable
                if last_word in incomplete_endings:
                    combined = " ".join(title_parts + [next_line])
                    if len(combined) <= 200:
                        title_parts.append(next_line)
                        continue
                break

            title = " ".join(title_parts)
            doc.close()
            return title

        doc.close()
        return None

    except Exception as e:
        print(f"Error extracting title from PDF: {e}")
        return None


def extract_metadata_from_pdf(pdf_bytes: bytes) -> dict:
    """Extract metadata from PDF for matching purposes.

    Returns dict with: title, year, arxiv_id (if available)
    """
    metadata = {}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract ArXiv ID (most reliable identifier if present)
        arxiv_id = _extract_arxiv_id(doc)
        if arxiv_id:
            metadata["arxiv_id"] = arxiv_id

        # Extract title
        title = extract_title_from_pdf(pdf_bytes)
        if title:
            metadata["title"] = title

        # Try to extract year from metadata or first page
        pdf_metadata = doc.metadata or {}
        creation_date = pdf_metadata.get("creationDate", "")

        # Parse year from creation date (format: D:20170612...)
        year_match = re.search(r'(\d{4})', creation_date)
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2030:
                metadata["year"] = year

        # Also check first page for year patterns
        if "year" not in metadata and len(doc) > 0:
            first_page_text = doc[0].get_text()
            # Look for year patterns like "2017", "(2017)", "2017.", etc.
            year_patterns = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', first_page_text)
            if year_patterns:
                # Take the first reasonable year found
                for year_str in year_patterns:
                    year = int(year_str)
                    if 1900 <= year <= 2030:
                        metadata["year"] = year
                        break

        doc.close()

    except Exception as e:
        print(f"Error extracting metadata from PDF: {e}")

    return metadata
