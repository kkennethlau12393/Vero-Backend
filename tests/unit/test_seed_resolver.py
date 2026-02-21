"""
Unit tests for seed_resolver.py - DOI/work_id/PDF to title extraction.

Tests pure logic with mocked Feature 1 functions. No database, no network.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ===== DOI Resolution Tests (10 tests) =====


def test_resolve_doi_to_title_success_openalex():
    """DOI found in OpenAlex - happy path."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:

        mock_oa.return_value = "W2963663278"
        mock_fetch.return_value = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "cited_by_count": 50000,
        }

        title, metadata = resolve_doi_to_title("10.48550/arXiv.1706.03762")

        assert title == "Attention Is All You Need"
        assert metadata["work_id"] == "W2963663278"
        assert metadata["year"] == 2017
        assert metadata["cited_by_count"] == 50000
        assert metadata["source"] == "openalex"
        assert metadata["extraction_method"] == "doi_lookup"


def test_resolve_doi_to_title_fallback_to_s2():
    """DOI not in OpenAlex, fallback to Semantic Scholar."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_oa.return_value = None  # Not found in OpenAlex
        mock_s2.return_value = {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "paperId": "abc123",
            "year": 2019,
            "citationCount": 40000,
        }

        title, metadata = resolve_doi_to_title("10.18653/v1/N19-1423")

        assert title == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert metadata["work_id"] == "S2:abc123"
        assert metadata["source"] == "semantic_scholar"


def test_resolve_doi_to_title_openalex_fetch_fails_fallback_to_s2():
    """OpenAlex DOI found but fetch fails, fallback to S2."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_oa.return_value = "W123"
        mock_fetch.side_effect = Exception("API timeout")
        mock_s2.return_value = {
            "title": "Some Paper",
            "paperId": "s2id",
            "year": 2020,
            "citationCount": 100,
        }

        title, metadata = resolve_doi_to_title("10.1234/test")

        assert title == "Some Paper"
        assert metadata["source"] == "semantic_scholar"


def test_resolve_doi_to_title_not_found():
    """DOI not found in OpenAlex or S2 - raises ValueError."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_oa.return_value = None
        mock_s2.return_value = None

        with pytest.raises(ValueError, match="not found in OpenAlex or Semantic Scholar"):
            resolve_doi_to_title("10.1234/does-not-exist")


def test_resolve_doi_to_title_strips_doi_prefix():
    """DOI with 'doi:' prefix is cleaned."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:

        mock_oa.return_value = "W123"
        mock_fetch.return_value = {"title": "Test", "year": 2020, "cited_by_count": 10}

        title, _ = resolve_doi_to_title("doi:10.1234/test")

        # Verify the DOI was cleaned before lookup
        mock_oa.assert_called_once_with("10.1234/test")


def test_resolve_doi_to_title_empty_doi():
    """Empty DOI raises ValueError."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with pytest.raises(ValueError):
        resolve_doi_to_title("")


def test_resolve_doi_to_title_whitespace_doi():
    """Whitespace-only DOI raises ValueError."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with pytest.raises(ValueError):
        resolve_doi_to_title("   ")


def test_resolve_doi_to_title_s2_no_title():
    """S2 returns paper but no title - raises ValueError."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_oa.return_value = None
        mock_s2.return_value = {"paperId": "abc", "year": 2020}  # No title

        with pytest.raises(ValueError, match="not found"):
            resolve_doi_to_title("10.1234/no-title")


def test_resolve_doi_to_title_openalex_no_title():
    """OpenAlex returns paper but no title - should fallback to S2."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    with patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_oa.return_value = "W123"
        mock_fetch.return_value = {"year": 2020, "cited_by_count": 10}  # No title
        mock_s2.return_value = {
            "title": "Fallback Title",
            "paperId": "s2id",
            "year": 2020,
            "citationCount": 100,
        }

        title, metadata = resolve_doi_to_title("10.1234/test")

        assert title == "Fallback Title"
        assert metadata["source"] == "semantic_scholar"


# ===== work_id Resolution Tests (10 tests) =====


def test_resolve_work_id_to_title_success_openalex():
    """OpenAlex work_id (W...) - happy path."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "cited_by_count": 50000,
        }

        title, metadata = resolve_work_id_to_title("W2963663278")

        assert title == "Attention Is All You Need"
        assert metadata["work_id"] == "W2963663278"
        assert metadata["source"] == "openalex"


def test_resolve_work_id_to_title_success_s2():
    """Semantic Scholar work_id (S2:...) - happy path."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {
            "title": "BERT",
            "year": 2019,
            "cited_by_count": 40000,
        }

        title, metadata = resolve_work_id_to_title("S2:204e3073...")

        assert title == "BERT"
        assert metadata["source"] == "semantic_scholar"


def test_resolve_work_id_to_title_success_arxiv():
    """ArXiv work_id (AX:...) - happy path."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {
            "title": "GPT-2",
            "year": 2019,
            "cited_by_count": 5000,
        }

        title, metadata = resolve_work_id_to_title("AX:1901.02860")

        assert title == "GPT-2"
        assert metadata["source"] == "arxiv"


def test_resolve_work_id_to_title_not_found():
    """work_id not found - raises ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = None

        with pytest.raises(ValueError, match="not found"):
            resolve_work_id_to_title("W999999999")


def test_resolve_work_id_to_title_no_title():
    """work_id found but no title - raises ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {"year": 2020, "cited_by_count": 10}  # No title

        with pytest.raises(ValueError, match="has no title"):
            resolve_work_id_to_title("W123")


def test_resolve_work_id_to_title_fetch_exception():
    """fetch_seed_paper_details raises exception - ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.side_effect = Exception("API error")

        with pytest.raises(ValueError, match="lookup failed"):
            resolve_work_id_to_title("W123")


def test_resolve_work_id_to_title_empty_id():
    """Empty work_id raises ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = None

        with pytest.raises(ValueError):
            resolve_work_id_to_title("")


def test_resolve_work_id_to_title_whitespace_id():
    """Whitespace-only work_id raises ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = None

        with pytest.raises(ValueError):
            resolve_work_id_to_title("   ")


def test_resolve_work_id_to_title_unknown_format():
    """Unknown work_id format returns 'unknown' source."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {
            "title": "Some Paper",
            "year": 2020,
            "cited_by_count": 10,
        }

        title, metadata = resolve_work_id_to_title("UNKNOWN123")

        assert metadata["source"] == "unknown"


def test_resolve_work_id_to_title_case_insensitive():
    """work_id format detection is case-insensitive."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    with patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:
        mock_fetch.return_value = {
            "title": "Test",
            "year": 2020,
            "cited_by_count": 10,
        }

        # Lowercase w should still be detected as OpenAlex
        title, metadata = resolve_work_id_to_title("w123")

        assert metadata["source"] == "openalex"


# ===== PDF Resolution Tests (10 tests) =====


def test_resolve_pdf_to_title_success_metadata():
    """PDF with metadata title - happy path."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "arxiv_id": None,
            "from_metadata": True,
        }

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert title == "Attention Is All You Need"
        assert metadata["source"] == "pdf_extraction"
        assert metadata["extraction_method"] == "pdf_metadata"


def test_resolve_pdf_to_title_success_arxiv_doi():
    """PDF with ArXiv ID - uses DOI lookup for canonical title."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract, \
         patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service.fetch_seed_paper_details") as mock_fetch:

        mock_extract.return_value = {
            "title": "Extracted Title (might be messy)",
            "year": 2017,
            "arxiv_id": "1706.03762",
            "from_metadata": False,
        }

        # Mock the ArXiv DOI lookup
        mock_oa.return_value = "W2963663278"
        mock_fetch.return_value = {
            "title": "Attention Is All You Need",
            "year": 2017,
            "cited_by_count": 50000,
        }

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert title == "Attention Is All You Need"
        assert metadata["extraction_method"] == "pdf_arxiv_doi"
        assert metadata["pdf_extracted_title"] == "Extracted Title (might be messy)"
        assert metadata["work_id"] == "W2963663278"


def test_resolve_pdf_to_title_arxiv_doi_fails_fallback():
    """PDF with ArXiv ID but DOI lookup fails - uses extracted title."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract, \
         patch("app.feature1.citation_map_service._lookup_openalex_by_doi") as mock_oa, \
         patch("app.feature1.citation_map_service._lookup_s2_paper_by_doi") as mock_s2:

        mock_extract.return_value = {
            "title": "Extracted Title",
            "year": 2017,
            "arxiv_id": "9999.99999",  # Doesn't exist
            "from_metadata": False,
        }

        # Mock DOI lookup failure
        mock_oa.return_value = None
        mock_s2.return_value = None

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert title == "Extracted Title"
        assert metadata["source"] == "pdf_extraction"


def test_resolve_pdf_to_title_no_title():
    """PDF with no extractable title - raises ValueError."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": None,
            "year": 2017,
            "arxiv_id": None,
        }

        with pytest.raises(ValueError, match="Could not extract title"):
            resolve_pdf_to_title(b"fake pdf content")


def test_resolve_pdf_to_title_empty_title():
    """PDF with empty string title - raises ValueError."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": "",
            "year": 2017,
            "arxiv_id": None,
        }

        with pytest.raises(ValueError, match="Could not extract title"):
            resolve_pdf_to_title(b"fake pdf content")


def test_resolve_pdf_to_title_parsing_fails():
    """PDF parsing fails - raises ValueError."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.side_effect = Exception("Corrupt PDF")

        with pytest.raises(ValueError, match="PDF parsing failed"):
            resolve_pdf_to_title(b"corrupt pdf")


def test_resolve_pdf_to_title_text_extraction():
    """PDF with title from text (not metadata)."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": "Title from Text",
            "year": None,
            "arxiv_id": None,
            "from_metadata": False,
        }

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert title == "Title from Text"
        assert metadata["extraction_method"] == "pdf_text"


def test_resolve_pdf_to_title_year_extracted():
    """PDF year is preserved in metadata."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": "Test Paper",
            "year": 2023,
            "arxiv_id": None,
            "from_metadata": True,
        }

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert metadata["year"] == 2023


def test_resolve_pdf_to_title_no_work_id():
    """PDF extraction without ArXiv ID has no work_id."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.return_value = {
            "title": "Regular Paper",
            "year": 2020,
            "arxiv_id": None,
            "from_metadata": True,
        }

        pdf_bytes = b"fake pdf content"
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert metadata["work_id"] is None
        assert metadata["cited_by_count"] == 0


def test_resolve_pdf_to_title_empty_bytes():
    """Empty PDF bytes - raises ValueError."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    with patch("app.feature1.pdf_parser.extract_metadata_from_pdf") as mock_extract:
        mock_extract.side_effect = Exception("Empty file")

        with pytest.raises(ValueError, match="PDF parsing failed"):
            resolve_pdf_to_title(b"")


# ===== Helper Function Tests (2 tests) =====


def test_detect_source_from_work_id_openalex():
    """Detect OpenAlex work_id format."""
    from app.feature2.seed_resolver import _detect_source_from_work_id

    assert _detect_source_from_work_id("W123") == "openalex"
    assert _detect_source_from_work_id("w456") == "openalex"  # Case insensitive


def test_detect_source_from_work_id_s2():
    """Detect Semantic Scholar work_id format."""
    from app.feature2.seed_resolver import _detect_source_from_work_id

    assert _detect_source_from_work_id("S2:abc123") == "semantic_scholar"
    assert _detect_source_from_work_id("s2:xyz") == "semantic_scholar"


def test_detect_source_from_work_id_arxiv():
    """Detect ArXiv work_id format."""
    from app.feature2.seed_resolver import _detect_source_from_work_id

    assert _detect_source_from_work_id("AX:1234.5678") == "arxiv"
    assert _detect_source_from_work_id("ax:9999") == "arxiv"


def test_detect_source_from_work_id_unknown():
    """Unknown work_id format returns 'unknown'."""
    from app.feature2.seed_resolver import _detect_source_from_work_id

    assert _detect_source_from_work_id("UNKNOWN123") == "unknown"
    assert _detect_source_from_work_id("123456") == "unknown"
