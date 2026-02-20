"""
Unit tests for pure validation functions in app/feature3/abstract_enrichment.py

Pure logic tests: no DB, no network.
"""
import pytest

from app.feature3.abstract_enrichment import (
    detect_language_simple,
    is_abstract_valid,
)


@pytest.mark.unit
class TestDetectLanguageSimple:
    def test_english_text(self):
        text = "We propose a new method for training deep neural networks that improves performance on standard benchmarks. The results show significant improvement over baseline approaches."
        assert detect_language_simple(text) == "en"

    def test_spanish_text(self):
        text = "En este trabajo se presenta un nuevo método para el análisis de datos que permite una mejor comprensión de los resultados experimentales."
        result = detect_language_simple(text)
        assert result == "other"

    def test_short_text(self):
        assert detect_language_simple("Too short") is None

    def test_empty_string(self):
        assert detect_language_simple("") is None

    def test_none_equivalent(self):
        # Function checks `not text`, empty string should return None
        assert detect_language_simple("") is None


@pytest.mark.unit
class TestIsAbstractValid:
    def test_valid_abstract(self):
        is_valid, reason = is_abstract_valid(
            "Deep Residual Learning for Image Recognition",
            "We propose a deep residual learning framework for training image recognition networks. Our approach uses skip connections to enable training of networks with hundreds of layers for image recognition tasks. Experiments show state-of-the-art results on ImageNet image classification."
        )
        assert is_valid is True
        assert reason == "Valid"

    def test_missing_abstract(self):
        is_valid, reason = is_abstract_valid("Test Paper", None)
        assert is_valid is False
        assert "missing" in reason.lower()

    def test_too_short(self):
        is_valid, reason = is_abstract_valid("Test Paper", "Too short.")
        assert is_valid is False
        assert "short" in reason.lower()

    def test_acknowledgments_detected(self):
        is_valid, reason = is_abstract_valid(
            "Test Paper",
            "The authors thank the anonymous referees for help with this manuscript. This work was supported by the National Science Foundation grant number 12345."
        )
        assert is_valid is False

    def test_funding_dominated(self):
        is_valid, reason = is_abstract_valid(
            "Test Paper",
            "We are grateful for funding from the National Institutes of Health. This work was supported by grant number 67890 from the foundation. The authors acknowledge support from the research council."
        )
        assert is_valid is False

    def test_language_mismatch(self):
        is_valid, reason = is_abstract_valid(
            "Deep Learning Paper",
            "En este trabajo se presenta un nuevo método para el análisis de datos que permite una mejor comprensión de los resultados experimentales obtenidos.",
            expected_language="en",
        )
        assert is_valid is False
        assert "language" in reason.lower() or "Language" in reason

    def test_low_title_overlap(self):
        is_valid, reason = is_abstract_valid(
            "Quantum Computing for Drug Discovery",
            "The ocean temperature patterns reveal significant changes in marine biodiversity across tropical regions, with implications for conservation biology and ecosystem management studies."
        )
        assert is_valid is False
        assert "overlap" in reason.lower()

    def test_acceptable_short_title(self):
        # Short titles (< 4 words) skip the overlap check
        is_valid, reason = is_abstract_valid(
            "BERT",
            "We introduce a new language representation model that obtains state-of-the-art results on eleven natural language processing tasks."
        )
        assert is_valid is True
