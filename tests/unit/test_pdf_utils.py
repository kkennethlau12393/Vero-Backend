"""Unit tests for shared PDF utilities."""
import pytest
from app.shared.pdf_utils import extract_paper_sections


@pytest.mark.unit
class TestExtractPaperSections:
    def test_finds_introduction(self):
        text = "Abstract\nBlah blah\n\n1. Introduction\nThis paper presents a new method for image classification.\n\n2. Methods\nWe use a CNN."
        sections = extract_paper_sections(text)
        assert "This paper presents" in sections["introduction"]

    def test_finds_methods(self):
        text = "1. Introduction\nIntro text here.\n\n2. Methodology\nOur approach uses gradient descent and backpropagation.\n\n3. Results\nWe find improvements."
        sections = extract_paper_sections(text)
        assert "Our approach" in sections["methods"]

    def test_finds_results_conclusion(self):
        text = "2. Methods\nMethod text here.\n\n3. Results and Discussion\nOur findings show significant improvement over baselines.\n\nReferences\n[1] Smith et al."
        sections = extract_paper_sections(text)
        assert "findings show" in sections["results_conclusion"]

    def test_truncates_long_introduction(self):
        long_intro = "1. Introduction\n" + "This is a long introduction sentence. " * 100 + "\n\n2. Methods\nShort methods."
        sections = extract_paper_sections(long_intro)
        assert len(sections["introduction"]) <= 1600  # ~1500 + word boundary tolerance

    def test_truncates_long_methods(self):
        long_methods = "1. Introduction\nShort.\n\n2. Methods\n" + "Method detail sentence here. " * 100 + "\n\n3. Results\nDone."
        sections = extract_paper_sections(long_methods)
        assert len(sections["methods"]) <= 1100  # ~1000 + word boundary tolerance

    def test_handles_no_sections_found(self):
        sections = extract_paper_sections("Just some random text with no section headers at all.")
        assert sections["introduction"] == ""
        assert sections["methods"] == ""
        assert sections["results_conclusion"] == ""

    def test_handles_empty_text(self):
        sections = extract_paper_sections("")
        assert sections["introduction"] == ""
        assert sections["methods"] == ""
        assert sections["results_conclusion"] == ""

    def test_unnumbered_headers(self):
        text = "Abstract\nBlah.\n\nIntroduction\nThis paper does X.\n\nMethods\nWe do Y.\n\nConclusion\nWe found Z."
        sections = extract_paper_sections(text)
        assert "This paper does X" in sections["introduction"]
        assert "We do Y" in sections["methods"]
        assert "We found Z" in sections["results_conclusion"]

    def test_stops_at_references(self):
        text = "3. Results\nOur results are great.\n\nReferences\n[1] Foo. [2] Bar."
        sections = extract_paper_sections(text)
        assert "[1] Foo" not in sections["results_conclusion"]
