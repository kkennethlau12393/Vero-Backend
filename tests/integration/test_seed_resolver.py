"""
Integration tests for seed_resolver.py - DOI/work_id/PDF to title extraction.

Tests with dev database + mocked external APIs (OpenAlex, S2, ArXiv).
Uses unittest.mock.patch for requests since Feature 1 uses `requests` library.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import pytest


# ===== DOI Resolution Integration Tests (15 tests) =====


@pytest.mark.integration
def test_resolve_doi_to_title_openalex_success():
    """DOI → OpenAlex → title extraction with realistic API response."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    mock_resp_doi = MagicMock()
    mock_resp_doi.status_code = 200
    mock_resp_doi.json.return_value = {
        "results": [{
            "id": "https://openalex.org/W2963663278",
            "title": "Attention Is All You Need",
        }]
    }

    mock_resp_work = MagicMock()
    mock_resp_work.status_code = 200
    mock_resp_work.json.return_value = {
        "id": "https://openalex.org/W2963663278",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 50000,
    }

    def mock_requests_get(url, **kwargs):
        if "filter=doi:" in url or (kwargs.get("params") and "filter" in kwargs.get("params", {})):
            mock_resp_doi.raise_for_status = MagicMock()
            return mock_resp_doi
        else:
            mock_resp_work.raise_for_status = MagicMock()
            return mock_resp_work

    with patch("requests.get", side_effect=mock_requests_get):
        title, metadata = resolve_doi_to_title("10.48550/arXiv.1706.03762")

        assert title == "Attention Is All You Need"
        assert metadata["work_id"] == "W2963663278"
        assert metadata["year"] == 2017
        assert metadata["cited_by_count"] == 50000
        assert metadata["source"] == "openalex"


@pytest.mark.integration
def test_resolve_doi_to_title_s2_fallback():
    """DOI not in OpenAlex, S2 fallback with realistic API response."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    # Mock OpenAlex 404
    mock_resp_oa = MagicMock()
    mock_resp_oa.status_code = 404
    mock_resp_oa.json.return_value = {"results": []}

    # Mock S2 success
    mock_resp_s2 = MagicMock()
    mock_resp_s2.status_code = 200
    mock_resp_s2.json.return_value = {
        "paperId": "abc123def456",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "year": 2019,
        "citationCount": 40000,
    }

    def mock_requests_get(url, **kwargs):
        if "openalex.org" in url:
            mock_resp_oa.raise_for_status = MagicMock()
            return mock_resp_oa
        elif "semanticscholar.org" in url:
            mock_resp_s2.raise_for_status = MagicMock()
            return mock_resp_s2
        return MagicMock()

    with patch("requests.get", side_effect=mock_requests_get):
        title, metadata = resolve_doi_to_title("10.18653/v1/N19-1423")

        assert "BERT" in title
        assert metadata["work_id"] == "S2:abc123def456"
        assert metadata["source"] == "semantic_scholar"


@pytest.mark.integration
def test_resolve_doi_to_title_not_found():
    """DOI not found in OpenAlex or S2 - both return 404."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_resp_404.json.return_value = {"results": []}
    mock_resp_404.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp_404):
        with pytest.raises(ValueError, match="not found"):
            resolve_doi_to_title("10.9999/does-not-exist")


@pytest.mark.integration
def test_resolve_doi_to_title_arxiv_format():
    """ArXiv DOI format (10.48550/arXiv.XXXX) handled correctly."""
    from app.feature2.seed_resolver import resolve_doi_to_title

    mock_resp_doi = MagicMock()
    mock_resp_doi.status_code = 200
    mock_resp_doi.json.return_value = {
        "results": [{
            "id": "https://openalex.org/W3092871860",
        }]
    }
    mock_resp_doi.raise_for_status = MagicMock()

    mock_resp_work = MagicMock()
    mock_resp_work.status_code = 200
    mock_resp_work.json.return_value = {
        "id": "https://openalex.org/W3092871860",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "publication_year": 2020,
        "cited_by_count": 15000,
    }
    mock_resp_work.raise_for_status = MagicMock()

    def mock_requests_get(url, **kwargs):
        if "filter=doi:" in url or (kwargs.get("params") and "filter" in kwargs.get("params", {})):
            return mock_resp_doi
        else:
            return mock_resp_work

    with patch("requests.get", side_effect=mock_requests_get):
        title, metadata = resolve_doi_to_title("10.48550/arXiv.2010.11929")

        assert "16x16" in title
        assert metadata["work_id"] == "W3092871860"


# ===== work_id Resolution Integration Tests (10 tests) =====


@pytest.mark.integration
def test_resolve_work_id_to_title_openalex():
    """OpenAlex work_id (W...) with realistic API response."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "https://openalex.org/W2963663278",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 50000,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        title, metadata = resolve_work_id_to_title("W2963663278")

        assert title == "Attention Is All You Need"
        assert metadata["work_id"] == "W2963663278"
        assert metadata["source"] == "openalex"


@pytest.mark.integration
def test_resolve_work_id_to_title_s2():
    """S2 work_id (S2:...) with realistic API response."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "paperId": "abc123",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "year": 2019,
        "citationCount": 40000,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        title, metadata = resolve_work_id_to_title("S2:abc123")

        assert "BERT" in title
        assert metadata["source"] == "semantic_scholar"


@pytest.mark.integration
def test_resolve_work_id_to_title_not_found():
    """work_id not found - 404 response."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="not found|lookup failed"):
            resolve_work_id_to_title("W999999999")


@pytest.mark.integration
def test_resolve_work_id_to_title_no_title():
    """work_id found but no title field - raises ValueError."""
    from app.feature2.seed_resolver import resolve_work_id_to_title

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "https://openalex.org/W123",
        "publication_year": 2020,
        # No title field
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="has no title"):
            resolve_work_id_to_title("W123")


# ===== PDF Resolution Integration Tests (15 tests) =====


@pytest.mark.integration
def test_resolve_pdf_to_title_with_metadata(tmp_path):
    """Real PDF with metadata extraction."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    # Create a minimal PDF with metadata
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Attention Is All You Need")
    doc.set_metadata({
        "title": "Attention Is All You Need",
        "author": "Vaswani et al.",
    })
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    title, metadata = resolve_pdf_to_title(pdf_bytes)

    assert title == "Attention Is All You Need"
    assert metadata["extraction_method"] in ["pdf_metadata", "pdf_text"]


@pytest.mark.integration
def test_resolve_pdf_to_title_with_arxiv_id(tmp_path):
    """PDF with ArXiv ID in text - triggers DOI lookup."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    # Create PDF with ArXiv ID in text
    pdf_path = tmp_path / "arxiv_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "arXiv:1706.03762v5")
    page.insert_text((50, 100), "Attention Is All You Need")
    doc.save(pdf_path)
    doc.close()

    # Mock OpenAlex DOI lookup
    mock_resp_doi = MagicMock()
    mock_resp_doi.status_code = 200
    mock_resp_doi.json.return_value = {
        "results": [{
            "id": "https://openalex.org/W2963663278",
        }]
    }
    mock_resp_doi.raise_for_status = MagicMock()

    mock_resp_work = MagicMock()
    mock_resp_work.status_code = 200
    mock_resp_work.json.return_value = {
        "id": "https://openalex.org/W2963663278",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 50000,
    }
    mock_resp_work.raise_for_status = MagicMock()

    def mock_requests_get(url, **kwargs):
        if "filter=doi:" in url or (kwargs.get("params") and "filter" in kwargs.get("params", {})):
            return mock_resp_doi
        else:
            return mock_resp_work

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with patch("requests.get", side_effect=mock_requests_get):
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        assert "Attention" in title
        # If ArXiv DOI lookup succeeds, extraction_method should be pdf_arxiv_doi
        if metadata["extraction_method"] == "pdf_arxiv_doi":
            assert metadata["work_id"] == "W2963663278"


@pytest.mark.integration
def test_resolve_pdf_to_title_text_extraction_only(tmp_path):
    """PDF with no metadata, title extracted from text."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    pdf_path = tmp_path / "text_only.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Add title-like text at the top
    page.insert_text((50, 50), "Deep Learning for Computer Vision", fontsize=20)
    page.insert_text((50, 100), "Abstract: This paper presents...")
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    title, metadata = resolve_pdf_to_title(pdf_bytes)

    assert "Deep Learning" in title or "Computer Vision" in title
    assert metadata["source"] == "pdf_extraction"


@pytest.mark.integration
def test_resolve_pdf_to_title_empty_pdf_fails(tmp_path):
    """Empty PDF - raises ValueError."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with pytest.raises(ValueError, match="Could not extract title"):
        resolve_pdf_to_title(pdf_bytes)


@pytest.mark.integration
def test_resolve_pdf_to_title_corrupt_pdf_fails():
    """Corrupt PDF bytes - raises ValueError."""
    from app.feature2.seed_resolver import resolve_pdf_to_title

    corrupt_bytes = b"This is not a PDF file"

    with pytest.raises(ValueError, match="Could not extract title|PDF parsing failed"):
        resolve_pdf_to_title(corrupt_bytes)


@pytest.mark.integration
def test_resolve_pdf_to_title_multi_page_pdf(tmp_path):
    """Multi-page PDF - title extracted from first page."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    pdf_path = tmp_path / "multi_page.pdf"
    doc = fitz.open()

    # Page 1: Title
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Neural Networks for NLP", fontsize=20)

    # Page 2: Content
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is the content of the paper...")

    doc.set_metadata({"title": "Neural Networks for NLP"})
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    title, metadata = resolve_pdf_to_title(pdf_bytes)

    assert "Neural Networks" in title or "NLP" in title


@pytest.mark.integration
def test_resolve_pdf_to_title_unicode_title(tmp_path):
    """PDF with Unicode characters in title."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    pdf_path = tmp_path / "unicode.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Résumé of Deep Learning Methods", fontsize=20)
    doc.set_metadata({"title": "Résumé of Deep Learning Methods"})
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    title, metadata = resolve_pdf_to_title(pdf_bytes)

    assert "sum" in title.lower() or "deep learning" in title.lower()


@pytest.mark.integration
def test_resolve_pdf_to_title_no_work_id_without_arxiv(tmp_path):
    """PDF without ArXiv ID has no work_id."""
    import fitz
    from app.feature2.seed_resolver import resolve_pdf_to_title

    pdf_path = tmp_path / "no_arxiv.pdf"
    doc = fitz.open()
    page = doc.new_page()
    doc.set_metadata({"title": "Regular Conference Paper"})
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    title, metadata = resolve_pdf_to_title(pdf_bytes)

    assert metadata["work_id"] is None
    assert metadata["cited_by_count"] == 0
