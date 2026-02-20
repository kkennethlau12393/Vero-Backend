"""
Unit tests for app/feature3/paper_identity.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.paper_identity import (
    PaperIdentity,
    decode_openalex_abstract,
    extract_arxiv_id_from_url,
    extract_doi_from_url,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    title_similarity,
    title_word_overlap,
    verify_paper_match,
)


@pytest.mark.unit
class TestNormalizeDoi:
    def test_strips_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"

    def test_strips_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1234/test") == "10.1234/test"

    def test_strips_doi_org_prefix(self):
        assert normalize_doi("doi.org/10.1234/test") == "10.1234/test"

    def test_strips_doi_colon_prefix(self):
        assert normalize_doi("doi:10.1234/test") == "10.1234/test"

    def test_lowercase(self):
        assert normalize_doi("10.1234/TEST.Paper") == "10.1234/test.paper"

    def test_none_input(self):
        assert normalize_doi(None) is None

    def test_empty_string(self):
        assert normalize_doi("") is None

    def test_already_normalized(self):
        assert normalize_doi("10.1234/test") == "10.1234/test"

    def test_strips_whitespace(self):
        assert normalize_doi("  10.1234/test  ") == "10.1234/test"


@pytest.mark.unit
class TestNormalizeArxivId:
    def test_new_format(self):
        assert normalize_arxiv_id("2301.12345") == "2301.12345"

    def test_removes_version_suffix(self):
        assert normalize_arxiv_id("2301.12345v2") == "2301.12345"

    def test_removes_arxiv_prefix(self):
        assert normalize_arxiv_id("arXiv:2301.12345") == "2301.12345"

    def test_removes_abs_url(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2301.12345") == "2301.12345"

    def test_removes_pdf_url(self):
        assert normalize_arxiv_id("https://arxiv.org/pdf/2301.12345") == "2301.12345"

    def test_removes_pdf_suffix(self):
        assert normalize_arxiv_id("2301.12345.pdf") == "2301.12345"

    def test_none_input(self):
        assert normalize_arxiv_id(None) is None

    def test_empty_string(self):
        result = normalize_arxiv_id("")
        assert result is None or result == ""


@pytest.mark.unit
class TestNormalizeTitle:
    def test_lowercase(self):
        assert "attention" in normalize_title("Attention Is All You Need")

    def test_remove_punctuation(self):
        result = normalize_title("Hello, World! A test.")
        assert "," not in result
        assert "!" not in result

    def test_collapse_spaces(self):
        result = normalize_title("Too   many    spaces")
        assert "  " not in result

    def test_empty_returns_empty(self):
        assert normalize_title("") == ""

    def test_none_returns_empty(self):
        assert normalize_title(None) == ""

    def test_version_suffix_removed(self):
        result = normalize_title("My Paper v2")
        assert "v2" not in result


@pytest.mark.unit
class TestTitleWordOverlap:
    def test_identical_titles(self):
        assert title_word_overlap("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_no_overlap(self):
        assert title_word_overlap("Neural Networks", "Ocean Currents Study") == 0.0

    def test_partial_overlap(self):
        result = title_word_overlap("Deep Learning for NLP", "Deep Learning for Vision")
        assert 0.0 < result < 1.0

    def test_empty_titles(self):
        assert title_word_overlap("", "") == 0.0
        assert title_word_overlap("Test", "") == 0.0

    def test_case_insensitive(self):
        assert title_word_overlap("ATTENTION", "attention") == 1.0


@pytest.mark.unit
class TestTitleSimilarity:
    def test_exact_match(self):
        assert title_similarity("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_high_similarity(self):
        result = title_similarity(
            "Attention Is All You Need",
            "Attention Is All You Need: A Transformer Model"
        )
        assert result > 0.5

    def test_dissimilar_titles_lower_than_similar(self):
        similar = title_similarity("Deep Learning for NLP", "Deep Learning for NLP Tasks")
        dissimilar = title_similarity("Quantum Chromodynamics Field Theory", "Population Ecology of Amphibians")
        assert dissimilar < similar

    def test_empty_returns_zero(self):
        assert title_similarity("", "Test") == 0.0
        assert title_similarity("Test", "") == 0.0


@pytest.mark.unit
class TestVerifyPaperMatch:
    def test_doi_match(self):
        source = PaperIdentity(work_id="W1", title="Test", doi="10.1234/test")
        candidate = {"doi": "10.1234/test", "title": "Test Paper"}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is True
        assert conf == 1.0
        assert "DOI match" in reason

    def test_doi_mismatch(self):
        source = PaperIdentity(work_id="W1", title="Test", doi="10.1234/test1")
        candidate = {"doi": "10.1234/test2", "title": "Test"}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is False
        assert conf == 0.0

    def test_arxiv_match(self):
        source = PaperIdentity(work_id="W1", title="Test", arxiv_id="2301.12345")
        candidate = {"arxiv_id": "2301.12345", "title": "Test"}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is True
        assert conf == 0.98

    def test_title_year_exact_match(self):
        source = PaperIdentity(work_id="W1", title="Attention Is All You Need", year=2017)
        candidate = {"title": "Attention Is All You Need", "year": 2017}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is True
        assert conf >= 0.90

    def test_title_year_close(self):
        source = PaperIdentity(work_id="W1", title="Attention Is All You Need", year=2017)
        candidate = {"title": "Attention Is All You Need", "year": 2018}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is True

    def test_insufficient_match(self):
        source = PaperIdentity(work_id="W1", title="Neural Networks", year=2020)
        candidate = {"title": "Ocean Currents Study", "year": 2020}
        is_match, conf, reason = verify_paper_match(source, candidate)
        assert is_match is False

    def test_nonstrict_title_only(self):
        source = PaperIdentity(work_id="W1", title="Attention Is All You Need")
        candidate = {"title": "Attention Is All You Need"}
        is_match, conf, reason = verify_paper_match(source, candidate, strict=False)
        assert is_match is True


@pytest.mark.unit
class TestExtractArxivIdFromUrl:
    def test_abs_url(self):
        result = extract_arxiv_id_from_url("https://arxiv.org/abs/2301.12345")
        assert result == "2301.12345"

    def test_pdf_url(self):
        result = extract_arxiv_id_from_url("https://arxiv.org/pdf/2301.12345")
        assert result == "2301.12345"

    def test_none_input(self):
        assert extract_arxiv_id_from_url(None) is None

    def test_non_arxiv_url(self):
        assert extract_arxiv_id_from_url("https://google.com") is None


@pytest.mark.unit
class TestExtractDoiFromUrl:
    def test_doi_org_url(self):
        result = extract_doi_from_url("https://doi.org/10.1234/test.paper")
        assert result is not None
        assert "10.1234" in result

    def test_none_input(self):
        assert extract_doi_from_url(None) is None


@pytest.mark.unit
class TestDecodeOpenalexAbstract:
    def test_basic_inverted_index(self):
        index = {"We": [0], "propose": [1], "a": [2], "new": [3], "architecture": [4]}
        result = decode_openalex_abstract(index)
        assert result == "We propose a new architecture"

    def test_correct_word_ordering(self):
        index = {"world": [1], "hello": [0]}
        result = decode_openalex_abstract(index)
        assert result == "hello world"

    def test_empty_dict(self):
        assert decode_openalex_abstract({}) is None

    def test_none_input(self):
        assert decode_openalex_abstract(None) is None

    def test_non_dict_input(self):
        assert decode_openalex_abstract("not a dict") is None

    def test_invalid_positions_skipped(self):
        index = {"hello": [0], "world": ["bad"]}
        result = decode_openalex_abstract(index)
        assert result == "hello"
