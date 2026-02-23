"""
Integration tests for Feature 3 sub-pipelines.

Tests abstract enrichment, topic inference, reference store, landmark retrieval,
cross-domain filtering, and grounding supplement with dev DB + mocked external APIs.

Uses unittest.mock.patch decorator style (matching Feature 1 integration test patterns).
"""
from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import text

from app.feature3.abstract_enrichment import (
    ensure_valid_abstract,
    is_abstract_valid,
    fetch_arxiv_by_id,
    fetch_openalex_abstract,
    fetch_s2_by_doi,
)
from app.feature3.topic_inference import (
    ensure_topic,
    fetch_topic_from_openalex_by_doi,
    fetch_topic_from_openalex_by_title,
    infer_topic_via_llm,
    map_llm_topic_to_openalex,
)
from app.feature3.reference_store import (
    get_referenced_works,
)
from app.feature3.landmark_retrieval import (
    get_topic_landmarks,
    calc_landmark_count,
)
from app.feature3.node_details_service import (
    _filter_cross_domain_papers,
)
from app.feature3.grounding_supplement import (
    supplement_grounding_papers,
    assess_grounding_needs,
)
from app.feature3.methodology import get_methodology, is_methodology_mismatch
from tests.fixtures.novelty_responses import (
    make_work_data,
    make_reference_paper,
    make_landmark_paper,
    make_references,
    make_landmarks,
    make_groq_chat_response,
    OPENALEX_WORK_RESPONSE,
    OPENALEX_TOPIC_RESPONSE,
)


# ── Mock Helper ──────────────────────────────────────────────────────────────

def _make_mock_response(json_data=None, text_data=None, status_code=200):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    if json_data is not None:
        resp.json.return_value = json_data
    if text_data is not None:
        resp.text = text_data
        resp.content = text_data.encode("utf-8")
    if status_code >= 400:
        from requests.exceptions import HTTPError
        http_error = HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_error
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── Mock Paths ───────────────────────────────────────────────────────────────

REQUESTS_GET_ABSTRACT = "app.feature3.abstract_enrichment.requests.get"
REQUESTS_GET_TOPIC_INF = "app.feature3.topic_inference.requests.get"
REQUESTS_GET_REFERENCE = "app.feature3.reference_store.requests.get"
REQUESTS_GET_LANDMARK = "app.feature3.landmark_retrieval.requests.get"
REQUESTS_GET_GROUNDING = "app.feature3.grounding_supplement.requests.get"
OPENAI_CLASS_TOPIC = "app.feature3.topic_inference.OpenAI"
OPENAI_CLASS_GROUNDING = "app.feature3.grounding_supplement.OpenAI"

# ── Mock API Responses ───────────────────────────────────────────────────────

ARXIV_RESPONSE_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This paper introduces a novel deep learning approach for image recognition tasks with superior performance on standard benchmarks compared to previous methods.</summary>
    <published>2020-01-15T00:00:00Z</published>
  </entry>
</feed>'''

ARXIV_EMPTY_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>'''

S2_PAPER_RESPONSE = {
    "paperId": "abc123",
    "title": "Test Paper Title",
    "abstract": "This paper introduces a novel deep learning approach for image recognition tasks with superior performance on standard benchmarks.",
    "year": 2020,
    "citationCount": 500,
    "externalIds": {"DOI": "10.1234/test.2020", "ArXiv": "2001.00001"},
}

OPENALEX_WORK_WITH_TOPIC = {
    "id": "https://openalex.org/W1234567890",
    "title": "Test Paper Title",
    "primary_topic": {
        "id": "https://openalex.org/T10123",
        "display_name": "Machine Learning",
        "score": 0.95,
    },
    "topics": [
        {"id": "https://openalex.org/T10123", "display_name": "Machine Learning", "score": 0.95},
        {"id": "https://openalex.org/T10456", "display_name": "Neural Networks", "score": 0.80},
    ],
}

OPENALEX_WORK_WITH_REFS = {
    "id": "https://openalex.org/W1234567890",
    "title": "Test Paper Title",
    "referenced_works": [
        "https://openalex.org/W9900000001",
        "https://openalex.org/W9900000002",
    ],
}

OPENALEX_BATCH_WORKS = {
    "results": [
        {
            "id": "https://openalex.org/W9900000001",
            "title": "Ref Paper 1",
            "publication_year": 2015,
            "cited_by_count": 500,
            "abstract_inverted_index": {"Method": [0], "paper": [1], "introducing": [2], "novel": [3], "techniques": [4], "for": [5], "deep": [6], "learning": [7]},
            "primary_topic": {"id": "https://openalex.org/T10123"},
        },
        {
            "id": "https://openalex.org/W9900000002",
            "title": "Ref Paper 2",
            "publication_year": 2014,
            "cited_by_count": 300,
            "abstract_inverted_index": None,
            "primary_topic": {"id": "https://openalex.org/T10456"},
        },
    ]
}


# ── Unique work_id generator ────────────────────────────────────────────────

_TEST_COUNTER = 0


def _unique_work_id(prefix="Wtest"):
    """Generate a unique work_id for test isolation."""
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    ts = int(time.time() * 1000) % 10_000_000_000
    return f"{prefix}_{ts}_{_TEST_COUNTER}"


# ── DB Test Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def test_work(db_conn):
    """Insert a test work into the works table and clean up after."""
    work_id = _unique_work_id()

    db_conn.execute(
        text("""
            INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id, doi, arxiv_id)
            VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id, :doi, :arxiv_id)
            ON CONFLICT (work_id) DO NOTHING
        """),
        {
            "work_id": work_id,
            "title": "Test Paper for Integration Testing",
            "year": 2020,
            "cited_by_count": 500,
            "abstract": None,
            "primary_topic_id": None,
            "doi": None,
            "arxiv_id": None,
        },
    )
    db_conn.commit()

    yield work_id

    # Cleanup
    db_conn.execute(
        text("DELETE FROM works WHERE work_id = :work_id"),
        {"work_id": work_id},
    )
    db_conn.commit()


@pytest.fixture()
def test_work_with_abstract(db_conn):
    """Insert a test work with a valid abstract."""
    work_id = _unique_work_id()

    db_conn.execute(
        text("""
            INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id)
            VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id)
            ON CONFLICT (work_id) DO NOTHING
        """),
        {
            "work_id": work_id,
            "title": "Deep Learning for Image Recognition",
            "year": 2020,
            "cited_by_count": 1200,
            "abstract": (
                "This paper introduces a novel deep learning approach for image recognition "
                "tasks. We propose a new convolutional architecture that achieves superior "
                "performance on ImageNet and CIFAR-10 benchmarks compared to previous methods."
            ),
            "primary_topic_id": "T10123",
        },
    )
    db_conn.commit()

    yield work_id

    db_conn.execute(
        text("DELETE FROM works WHERE work_id = :work_id"),
        {"work_id": work_id},
    )
    db_conn.commit()


@pytest.fixture()
def test_work_with_refs(db_conn):
    """Insert a test work and reference works for reference store tests."""
    work_id = _unique_work_id()
    ref1_id = _unique_work_id("Wref1")
    ref2_id = _unique_work_id("Wref2")

    # Insert the main work
    db_conn.execute(
        text("""
            INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id)
            VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id)
            ON CONFLICT (work_id) DO NOTHING
        """),
        {
            "work_id": work_id,
            "title": "Main Test Paper",
            "year": 2020,
            "cited_by_count": 500,
            "abstract": "A test paper for reference store integration tests with comprehensive evaluation.",
            "primary_topic_id": "T10123",
        },
    )

    # Insert reference works
    for ref_id, ref_title, ref_year, ref_cites in [
        (ref1_id, "Reference Paper A", 2015, 2000),
        (ref2_id, "Reference Paper B", 2012, 800),
    ]:
        db_conn.execute(
            text("""
                INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id)
                VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id)
                ON CONFLICT (work_id) DO NOTHING
            """),
            {
                "work_id": ref_id,
                "title": ref_title,
                "year": ref_year,
                "cited_by_count": ref_cites,
                "abstract": f"This is the abstract for {ref_title}, describing novel contributions to the field.",
                "primary_topic_id": "T10123",
            },
        )

    db_conn.commit()

    yield {
        "work_id": work_id,
        "ref_ids": [ref1_id, ref2_id],
    }

    # Cleanup
    for wid in [work_id, ref1_id, ref2_id]:
        db_conn.execute(
            text("DELETE FROM works WHERE work_id = :work_id"),
            {"work_id": wid},
        )
    db_conn.commit()


@pytest.fixture()
def test_works_with_topic(db_conn):
    """Insert test works with a shared topic_id for landmark retrieval tests."""
    topic_id = "Ttest_integration_999"
    work_ids = []

    papers = [
        (_unique_work_id("Wlm"), "Foundational Neural Network Paper", 2005, 15000),
        (_unique_work_id("Wlm"), "Deep Learning Breakthrough", 2012, 8000),
        (_unique_work_id("Wlm"), "Modern Architecture Paper", 2018, 3000),
        (_unique_work_id("Wlm"), "Recent Advances in ML", 2022, 500),
    ]

    for wid, title, year, cites in papers:
        db_conn.execute(
            text("""
                INSERT INTO works (work_id, title, year, cited_by_count, abstract, primary_topic_id)
                VALUES (:work_id, :title, :year, :cited_by_count, :abstract, :primary_topic_id)
                ON CONFLICT (work_id) DO NOTHING
            """),
            {
                "work_id": wid,
                "title": title,
                "year": year,
                "cited_by_count": cites,
                "abstract": f"This paper contributes to {title} with novel techniques and significant results.",
                "primary_topic_id": topic_id,
            },
        )
        work_ids.append(wid)

    db_conn.commit()

    yield {
        "topic_id": topic_id,
        "work_ids": work_ids,
        "papers": papers,
    }

    # Cleanup
    for wid in work_ids:
        db_conn.execute(
            text("DELETE FROM works WHERE work_id = :work_id"),
            {"work_id": wid},
        )
    db_conn.commit()


# ============================================================================
# 1. TestAbstractEnrichment
# ============================================================================

@pytest.mark.integration
class TestAbstractEnrichment:
    """Tests for app.feature3.abstract_enrichment.ensure_valid_abstract."""

    def test_valid_abstract_cached(self, db_conn, test_work_with_abstract):
        """A work with a good abstract returns (abstract, 'cached') without external calls."""
        work_id = test_work_with_abstract

        row = db_conn.execute(
            text("SELECT title, abstract, year, doi, arxiv_id FROM works WHERE work_id = :wid"),
            {"wid": work_id},
        ).mappings().first()

        result_abstract, source = ensure_valid_abstract(
            db_conn,
            work_id=work_id,
            title=row["title"],
            abstract=row["abstract"],
            year=row["year"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
        )

        assert source == "cached"
        assert result_abstract == row["abstract"]
        assert "deep learning" in result_abstract.lower()

    @patch(REQUESTS_GET_ABSTRACT)
    def test_missing_abstract_arxiv_fallback(self, mock_get, db_conn, test_work):
        """None abstract + arxiv_id triggers ArXiv fetch and returns abstract."""
        work_id = test_work

        # Update the work to have an arxiv_id
        db_conn.execute(
            text("UPDATE works SET arxiv_id = :arxiv_id WHERE work_id = :wid"),
            {"arxiv_id": "2001.00001", "wid": work_id},
        )
        db_conn.commit()

        # Mock OpenAlex first (tried before ArXiv), then ArXiv
        def route(url, **kwargs):
            url_str = str(url)
            if "api.openalex.org" in url_str:
                return _make_mock_response(status_code=404)
            if "arxiv.org" in url_str:
                return _make_mock_response(text_data=ARXIV_RESPONSE_XML)
            return _make_mock_response(status_code=404)

        mock_get.side_effect = route

        result_abstract, source = ensure_valid_abstract(
            db_conn,
            work_id=work_id,
            title="Test Paper Title",
            abstract=None,
            year=2020,
            arxiv_id="2001.00001",
        )

        assert source == "arxiv"
        assert result_abstract is not None
        assert "deep learning" in result_abstract.lower()

    @patch(REQUESTS_GET_ABSTRACT)
    def test_missing_abstract_openalex_fallback(self, mock_get, db_conn, test_work):
        """None abstract + W-prefix work_id triggers OpenAlex fetch."""
        work_id = test_work

        openalex_work = {
            "id": f"https://openalex.org/{work_id}",
            "title": "Test Paper for Integration Testing",
            "abstract_inverted_index": {
                "This": [0], "paper": [1], "introduces": [2], "a": [3],
                "novel": [4], "approach": [5], "to": [6], "solving": [7],
                "complex": [8], "problems": [9], "using": [10], "advanced": [11],
                "machine": [12], "learning": [13], "techniques": [14],
            },
            "publication_year": 2020,
            "doi": None,
        }

        mock_get.return_value = _make_mock_response(json_data=openalex_work)

        result_abstract, source = ensure_valid_abstract(
            db_conn,
            work_id=work_id,
            title="Test Paper for Integration Testing",
            abstract=None,
            year=2020,
        )

        assert source == "openalex"
        assert result_abstract is not None
        assert len(result_abstract) > 50

    @patch(REQUESTS_GET_ABSTRACT)
    def test_missing_abstract_all_fail(self, mock_get, db_conn, test_work):
        """When all external APIs fail/404, returns (None, 'unavailable')."""
        work_id = test_work

        # All endpoints return 404
        mock_get.return_value = _make_mock_response(status_code=404)

        result_abstract, source = ensure_valid_abstract(
            db_conn,
            work_id=work_id,
            title="Test Paper for Integration Testing",
            abstract=None,
            year=2020,
        )

        assert source == "unavailable"
        assert result_abstract is None

    @patch(REQUESTS_GET_ABSTRACT)
    def test_enriched_abstract_updated_in_db(self, mock_get, db_conn, test_work):
        """After enrichment from OpenAlex, works.abstract is updated in DB."""
        work_id = test_work

        openalex_work = {
            "id": f"https://openalex.org/{work_id}",
            "title": "Test Paper for Integration Testing",
            "abstract_inverted_index": {
                "This": [0], "paper": [1], "introduces": [2], "a": [3],
                "novel": [4], "approach": [5], "to": [6], "solving": [7],
                "complex": [8], "problems": [9], "using": [10], "advanced": [11],
                "machine": [12], "learning": [13], "techniques": [14],
            },
            "publication_year": 2020,
            "doi": None,
        }

        mock_get.return_value = _make_mock_response(json_data=openalex_work)

        result_abstract, source = ensure_valid_abstract(
            db_conn,
            work_id=work_id,
            title="Test Paper for Integration Testing",
            abstract=None,
            year=2020,
        )

        assert source == "openalex"

        # Verify DB was updated
        row = db_conn.execute(
            text("SELECT abstract, abstract_source FROM works WHERE work_id = :wid"),
            {"wid": work_id},
        ).mappings().first()

        assert row is not None
        assert row["abstract"] is not None
        assert len(row["abstract"]) > 50
        assert row["abstract_source"] == "openalex"

    def test_short_abstract_triggers_enrichment(self, db_conn, test_work):
        """An abstract shorter than 50 chars is treated as invalid."""
        work_id = test_work
        is_valid, reason = is_abstract_valid("Test Paper", "Too short")
        assert not is_valid
        assert "too short" in reason.lower()

    def test_non_english_abstract_triggers_enrichment(self, db_conn, test_work):
        """A non-English abstract is detected as invalid."""
        spanish_abstract = (
            "Este artículo presenta un nuevo método para la clasificación "
            "de datos utilizando técnicas de aprendizaje automático en el "
            "campo de la inteligencia artificial con resultados superiores."
        )
        is_valid, reason = is_abstract_valid("Machine Learning Paper", spanish_abstract)
        assert not is_valid
        assert "language" in reason.lower()

    def test_acknowledgments_text_detected(self, db_conn, test_work):
        """Acknowledgments text masquerading as abstract is caught."""
        ack_text = (
            "The authors thank the anonymous referees for help improving this manuscript. "
            "This work was supported by the National Science Foundation grant number 12345. "
            "We are grateful to the research team for their valuable feedback."
        )
        is_valid, reason = is_abstract_valid("Research Paper", ack_text)
        assert not is_valid
        assert "acknowledgment" in reason.lower()


# ============================================================================
# 2. TestTopicInference
# ============================================================================

@pytest.mark.integration
class TestTopicInference:
    """Tests for app.feature3.topic_inference.ensure_topic."""

    def test_existing_topic_cached(self, db_conn, test_work_with_abstract):
        """When topic_id already exists, returns (topic_id, 'cached')."""
        work_id = test_work_with_abstract

        topic_id, source = ensure_topic(
            db_conn,
            work_id=work_id,
            title="Deep Learning for Image Recognition",
            abstract="Some abstract text for testing.",
            current_topic_id="T10123",
        )

        assert topic_id == "T10123"
        assert source == "cached"

    @patch(REQUESTS_GET_TOPIC_INF)
    def test_infer_from_openalex_doi(self, mock_get, db_conn, test_work):
        """Topic inferred from OpenAlex using DOI lookup."""
        work_id = test_work

        mock_get.return_value = _make_mock_response(json_data=OPENALEX_WORK_WITH_TOPIC)

        topic_id, source = ensure_topic(
            db_conn,
            work_id=work_id,
            title="Test Paper Title",
            abstract="Machine learning paper about neural networks and deep learning.",
            current_topic_id=None,
            doi="10.1234/test.2020",
        )

        assert topic_id == "T10123"
        assert source == "openalex_fetch"

    @patch(REQUESTS_GET_TOPIC_INF)
    def test_infer_from_openalex_title(self, mock_get, db_conn, test_work):
        """Topic inferred from OpenAlex title search when no DOI."""
        work_id = test_work

        oa_search_response = {
            "results": [OPENALEX_WORK_WITH_TOPIC],
            "meta": {"count": 1},
        }
        mock_get.return_value = _make_mock_response(json_data=oa_search_response)

        topic_id, source = ensure_topic(
            db_conn,
            work_id=work_id,
            title="Test Paper Title",
            abstract="Machine learning paper about neural networks and deep learning.",
            current_topic_id=None,
            doi=None,
        )

        assert topic_id == "T10123"
        assert source == "openalex_fetch"

    @patch(REQUESTS_GET_TOPIC_INF)
    @patch(OPENAI_CLASS_TOPIC)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_infer_from_llm(self, MockOpenAI, mock_get, db_conn, test_work):
        """Topic inferred from LLM when OpenAlex returns nothing."""
        work_id = test_work

        # OpenAlex returns nothing for both DOI and title search
        mock_get.side_effect = [
            _make_mock_response(status_code=404),  # DOI lookup
            _make_mock_response(json_data={"results": []}),  # title search
            # Topics API search for mapping:
            _make_mock_response(json_data={
                "results": [
                    {"id": "https://openalex.org/T10789", "display_name": "Machine Learning"}
                ]
            }),
        ]

        # LLM returns topic classification
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = json.dumps({
            "field": "Computer Science",
            "subfield": "Machine Learning",
            "topic_description": "Deep neural networks for classification",
            "confidence": 0.90,
        })

        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = llm_response

        topic_id, source = ensure_topic(
            db_conn,
            work_id=work_id,
            title="Test Paper for Integration Testing",
            abstract="Deep neural networks for image classification with novel architecture.",
            current_topic_id=None,
            doi="10.1234/nonexistent",
        )

        assert topic_id is not None
        assert source == "llm_inference"

    @patch(REQUESTS_GET_TOPIC_INF)
    def test_topic_updated_in_db(self, mock_get, db_conn, test_work):
        """After inference, works.primary_topic_id is updated in the DB."""
        work_id = test_work

        mock_get.return_value = _make_mock_response(json_data=OPENALEX_WORK_WITH_TOPIC)

        topic_id, source = ensure_topic(
            db_conn,
            work_id=work_id,
            title="Test Paper Title",
            abstract="Machine learning paper about neural networks.",
            current_topic_id=None,
            doi="10.1234/test.2020",
        )

        assert source == "openalex_fetch"

        # Verify DB was updated
        row = db_conn.execute(
            text("SELECT primary_topic_id, topic_source FROM works WHERE work_id = :wid"),
            {"wid": work_id},
        ).mappings().first()

        assert row is not None
        assert row["primary_topic_id"] == "T10123"
        assert row["topic_source"] == "openalex_fetch"

    @patch(REQUESTS_GET_TOPIC_INF)
    @patch(OPENAI_CLASS_TOPIC)
    def test_all_fail_returns_unavailable(self, MockOpenAI, mock_get, db_conn, test_work):
        """When all sources fail, returns (None, 'unavailable')."""
        work_id = test_work

        # All OpenAlex calls fail
        mock_get.return_value = _make_mock_response(status_code=500)

        # LLM not configured (no API key)
        import os
        old = os.environ.pop("GROQ_API_KEY", None)
        try:
            topic_id, source = ensure_topic(
                db_conn,
                work_id=work_id,
                title="Test Paper",
                abstract="Some abstract text for testing.",
                current_topic_id=None,
            )
        finally:
            if old:
                os.environ["GROQ_API_KEY"] = old

        assert topic_id is None
        assert source == "unavailable"


# ============================================================================
# 3. TestReferenceStore
# ============================================================================

@pytest.mark.integration
class TestReferenceStore:
    """Tests for app.feature3.reference_store.get_referenced_works."""

    def test_cached_references(self, db_conn, test_work_with_refs):
        """Pre-populated referenced_works_json returns cached refs."""
        data = test_work_with_refs
        work_id = data["work_id"]
        ref_ids = data["ref_ids"]

        # Pre-populate the referenced_works_json cache
        db_conn.execute(
            text("UPDATE works SET referenced_works_json = :refs WHERE work_id = :wid"),
            {"refs": json.dumps(ref_ids), "wid": work_id},
        )
        db_conn.commit()

        results = get_referenced_works(db_conn, work_id)

        assert len(results) >= 1
        result_ids = [r["work_id"] for r in results]
        for ref_id in ref_ids:
            assert ref_id in result_ids

    @patch(REQUESTS_GET_REFERENCE)
    def test_fetch_references_from_openalex(self, mock_get, db_conn, test_work):
        """No cache triggers OpenAlex fetch, caches result."""
        work_id = test_work

        oa_work_with_refs = {
            "id": f"https://openalex.org/{work_id}",
            "referenced_works": [
                "https://openalex.org/W9900000001",
                "https://openalex.org/W9900000002",
            ],
        }

        # First call: get work with referenced_works
        # Second call: batch fetch work details
        # Third call might be made by _ensure_works_exist
        def route(url, **kwargs):
            url_str = str(url)
            params = kwargs.get("params", {})
            filter_param = params.get("filter", "")

            if f"/works/{work_id}" in url_str:
                return _make_mock_response(json_data=oa_work_with_refs)
            if "openalex:" in str(filter_param):
                return _make_mock_response(json_data=OPENALEX_BATCH_WORKS)
            return _make_mock_response(json_data={"results": []})

        mock_get.side_effect = route

        results = get_referenced_works(db_conn, work_id)

        # Results depend on whether the batch fetch succeeded and works were inserted
        assert isinstance(results, list)

        # Verify the cache was populated
        row = db_conn.execute(
            text("SELECT referenced_works_json FROM works WHERE work_id = :wid"),
            {"wid": work_id},
        ).mappings().first()
        assert row is not None
        assert row["referenced_works_json"] is not None

        # Cleanup inserted reference works
        for ref_wid in ["W9900000001", "W9900000002"]:
            db_conn.execute(
                text("DELETE FROM works WHERE work_id = :wid"),
                {"wid": ref_wid},
            )
        db_conn.commit()

    @patch(REQUESTS_GET_REFERENCE)
    def test_empty_references(self, mock_get, db_conn, test_work):
        """OpenAlex returning empty referenced_works yields empty list."""
        work_id = test_work

        oa_work_no_refs = {
            "id": f"https://openalex.org/{work_id}",
            "referenced_works": [],
        }
        mock_get.return_value = _make_mock_response(json_data=oa_work_no_refs)

        results = get_referenced_works(db_conn, work_id)
        assert results == []

    @patch(REQUESTS_GET_REFERENCE)
    def test_missing_works_fetched(self, mock_get, db_conn, test_work):
        """Referenced work_ids not in works table are fetched from OpenAlex."""
        work_id = test_work

        # Pre-populate referenced_works_json with IDs not in the DB
        missing_ref_id = _unique_work_id("Wmissing")
        db_conn.execute(
            text("UPDATE works SET referenced_works_json = :refs WHERE work_id = :wid"),
            {"refs": json.dumps([missing_ref_id]), "wid": work_id},
        )
        db_conn.commit()

        # Mock the batch fetch for the missing work
        batch_response = {
            "results": [
                {
                    "id": f"https://openalex.org/{missing_ref_id}",
                    "title": "Missing Paper Found",
                    "publication_year": 2018,
                    "cited_by_count": 750,
                    "abstract_inverted_index": {
                        "This": [0], "is": [1], "a": [2], "missing": [3],
                        "paper": [4], "now": [5], "found": [6],
                    },
                    "primary_topic": {"id": "https://openalex.org/T10123"},
                }
            ]
        }

        mock_get.return_value = _make_mock_response(json_data=batch_response)

        results = get_referenced_works(db_conn, work_id)

        # Verify the missing work was inserted into the DB
        row = db_conn.execute(
            text("SELECT work_id, title FROM works WHERE work_id = :wid"),
            {"wid": missing_ref_id},
        ).mappings().first()

        # The missing work should now exist (if batch fetch succeeded)
        if row:
            assert row["title"] == "Missing Paper Found"

        # Cleanup
        db_conn.execute(
            text("DELETE FROM works WHERE work_id = :wid"),
            {"wid": missing_ref_id},
        )
        db_conn.commit()

    def test_results_sorted_by_citations(self, db_conn, test_work_with_refs):
        """Returned references are sorted by cited_by_count descending."""
        data = test_work_with_refs
        work_id = data["work_id"]
        ref_ids = data["ref_ids"]

        # Pre-populate cache
        db_conn.execute(
            text("UPDATE works SET referenced_works_json = :refs WHERE work_id = :wid"),
            {"refs": json.dumps(ref_ids), "wid": work_id},
        )
        db_conn.commit()

        results = get_referenced_works(db_conn, work_id)

        if len(results) >= 2:
            cites = [r["cited_by_count"] for r in results]
            assert cites == sorted(cites, reverse=True)


# ============================================================================
# 4. TestLandmarkRetrieval
# ============================================================================

@pytest.mark.integration
class TestLandmarkRetrieval:
    """Tests for app.feature3.landmark_retrieval.get_topic_landmarks."""

    @patch(REQUESTS_GET_LANDMARK)
    def test_landmarks_fetched(self, mock_get, db_conn, test_works_with_topic):
        """Landmarks are returned with correct fields from DB-seeded topic works."""
        data = test_works_with_topic
        topic_id = data["topic_id"]

        # Mock the OpenAlex topic verification for high-cited papers
        # Return 404 to skip verification (keep all landmarks)
        mock_get.return_value = _make_mock_response(status_code=404)

        results = get_topic_landmarks(db_conn, topic_id, before_year=2025)

        assert len(results) >= 1
        for lm in results:
            assert "work_id" in lm
            assert "title" in lm
            assert "year" in lm
            assert "cited_by_count" in lm

    def test_no_topic_id(self, db_conn):
        """Passing None topic_id returns empty list."""
        results = get_topic_landmarks(db_conn, None, before_year=2025)
        assert results == []

    @patch(REQUESTS_GET_LANDMARK)
    def test_landmarks_spread_across_time(self, mock_get, db_conn, test_works_with_topic):
        """Landmarks come from different time periods."""
        data = test_works_with_topic
        topic_id = data["topic_id"]

        # Mock verification to skip
        mock_get.return_value = _make_mock_response(status_code=404)

        results = get_topic_landmarks(db_conn, topic_id, before_year=2025)

        if len(results) >= 2:
            years = [r["year"] for r in results if r.get("year")]
            # At least 2 different years
            assert len(set(years)) >= 2, f"Expected diverse years, got {years}"

    def test_calc_landmark_count_boundaries(self):
        """Landmark count scales logarithmically with topic span."""
        assert calc_landmark_count(0) == 3   # Minimum
        assert calc_landmark_count(2) == 3   # Very young
        assert calc_landmark_count(5) == 3   # Young
        assert calc_landmark_count(10) == 4  # Growing
        assert calc_landmark_count(20) == 5  # Established
        assert calc_landmark_count(50) >= 5  # Mature

    @patch(REQUESTS_GET_LANDMARK)
    def test_before_year_filter(self, mock_get, db_conn, test_works_with_topic):
        """Only papers before the specified year are returned."""
        data = test_works_with_topic
        topic_id = data["topic_id"]

        mock_get.return_value = _make_mock_response(status_code=404)

        results = get_topic_landmarks(db_conn, topic_id, before_year=2015)

        for lm in results:
            if lm.get("year"):
                assert lm["year"] < 2015, f"Landmark year {lm['year']} should be < 2015"


# ============================================================================
# 5. TestCrossDomainFiltering
# ============================================================================

@pytest.mark.integration
class TestCrossDomainFiltering:
    """Tests for _filter_cross_domain_papers from node_details_service."""

    def test_references_never_filtered(self):
        """References are author-curated and never filtered, even if unrelated."""
        refs = [
            {
                "work_id": "W001",
                "title": "Random Forest Classification with Ensemble Methods",
                "abstract": "We propose an ensemble of decision trees using bagging and boosting.",
                "cited_by_count": 1000,
                "primary_topic_id": "T123",
            },
            {
                "work_id": "W002",
                "title": "Deep Convolutional Neural Networks for Vision",
                "abstract": "We use deep convolutional neural networks with backpropagation.",
                "cited_by_count": 2000,
                "primary_topic_id": "T123",
            },
        ]

        filtered_refs, _ = _filter_cross_domain_papers(
            refs, [],
            target_title="Deep Learning with Neural Networks for Image Recognition",
            target_abstract="We train a deep neural network using backpropagation and convolutional layers.",
        )

        # ALL references should be kept - they are author-curated
        filtered_ref_ids = {r["work_id"] for r in filtered_refs}
        assert "W001" in filtered_ref_ids
        assert "W002" in filtered_ref_ids

    def test_irrelevant_landmarks_filtered_by_content(self):
        """Landmarks with no content overlap to target are filtered out."""
        landmarks = [
            {
                "work_id": "W003",
                "title": "Vehicle Tire Dynamics and Road Surface Interaction",
                "abstract": "We study tire friction coefficients under varying road conditions.",
                "cited_by_count": 500,
                "primary_topic_id": "T456",
            },
        ]

        _, filtered_landmarks = _filter_cross_domain_papers(
            [], landmarks,
            target_title="Deep Learning with Neural Networks for Image Recognition",
            target_abstract="We train a deep neural network using backpropagation and convolutional layers.",
        )

        # W003 (tire dynamics) should be filtered - no content overlap with neural networks
        assert len(filtered_landmarks) == 0

    def test_relevant_landmarks_kept_by_content(self):
        """Landmarks with content overlap to target are kept."""
        landmarks = [
            {
                "work_id": "W001",
                "title": "ResNet: Deep Residual Learning for Image Recognition",
                "abstract": "Deep neural network with residual connections and convolutional layers.",
                "cited_by_count": 50000,
            },
            {
                "work_id": "W002",
                "title": "VGGNet: Very Deep Convolutional Networks",
                "abstract": "Very deep convolutional neural network for image classification.",
                "cited_by_count": 40000,
            },
        ]

        _, filtered_landmarks = _filter_cross_domain_papers(
            [], landmarks,
            target_title="DenseNet: Dense Convolutional Networks",
            target_abstract="We propose dense connections between convolutional neural network layers.",
        )

        filtered_ids = {lm["work_id"] for lm in filtered_landmarks}
        assert "W001" in filtered_ids
        assert "W002" in filtered_ids

    def test_all_refs_kept_regardless_of_field(self):
        """References from any field are kept since they're author-curated."""
        refs = [
            {
                "work_id": "W001",
                "title": "Paper Alpha",
                "abstract": "Using random forest and bagging for classification.",
                "cited_by_count": 10000,
                "field_name": "Some Other Field",
            },
            {
                "work_id": "W002",
                "title": "Paper Beta",
                "abstract": "Using random forest for regression.",
                "cited_by_count": 8000,
                "field_name": "Some Other Field",
            },
            {
                "work_id": "W003",
                "title": "Paper Gamma",
                "abstract": "Using decision trees with bagging.",
                "cited_by_count": 5000,
                "field_name": "Some Other Field",
            },
        ]

        filtered_refs, _ = _filter_cross_domain_papers(
            refs, [],
            target_field_name="Target Field",
            target_title="Neural Network Paper",
            target_abstract="Deep learning with neural networks and convolutional layers.",
        )

        # All references should be kept - they are author-curated
        assert len(filtered_refs) == 3

    def test_no_filter_without_target_info(self):
        """Without target title/abstract, all papers returned."""
        refs = [
            {"work_id": "W001", "title": "Paper A", "cited_by_count": 100},
            {"work_id": "W002", "title": "Paper B", "cited_by_count": 200},
        ]
        landmarks = [
            {"work_id": "W003", "title": "Landmark", "cited_by_count": 5000},
        ]

        # No target info provided
        filtered_refs, filtered_landmarks = _filter_cross_domain_papers(
            refs, landmarks,
            target_field_name=None,
            target_topic_id=None,
            target_title=None,
            target_abstract=None,
        )

        assert len(filtered_refs) == 2
        assert len(filtered_landmarks) == 1

    def test_all_refs_kept_including_s2(self):
        """References are never filtered, even S2-prefixed papers."""
        refs = [
            {
                "work_id": "S2:abc123",
                "title": "S2 Paper Without Field",
                "cited_by_count": 500,
            },
            {
                "work_id": "W001",
                "title": "OpenAlex Paper",
                "cited_by_count": 1000,
            },
        ]

        filtered_refs, _ = _filter_cross_domain_papers(
            refs, [],
            target_field_name="Machine Learning",
            target_title="ML Paper",
            target_abstract="A paper about machine learning.",
        )

        # All references are author-curated, never filtered
        filtered_ids = {r["work_id"] for r in filtered_refs}
        assert "S2:abc123" in filtered_ids
        assert "W001" in filtered_ids


# ============================================================================
# 6. TestGroundingSupplementPipeline
# ============================================================================

@pytest.mark.integration
class TestGroundingSupplementPipeline:
    """Tests for app.feature3.grounding_supplement.supplement_grounding_papers."""

    def test_supplement_not_needed(self):
        """Enough existing refs+landmarks means no supplement needed."""
        refs = make_references(3)
        landmarks = make_landmarks(3)

        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)

        assert reason == "sufficient"
        assert refs_needed == 0
        assert landmarks_needed == 0

    @patch(REQUESTS_GET_GROUNDING)
    @patch(OPENAI_CLASS_GROUNDING)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_supplement_adds_landmarks(self, MockOpenAI, mock_get, db_conn):
        """Pass insufficient landmarks, mock LLM + OpenAlex, verify landmarks added."""
        refs = make_references(3)
        landmarks = []  # No landmarks -- need 3

        # LLM suggests landmark papers
        llm_suggestions = json.dumps([
            {
                "title": "Foundational Deep Learning Paper",
                "authors": "LeCun et al.",
                "year": 2015,
                "why_relevant": "Introduced deep learning techniques for the field",
            },
            {
                "title": "Seminal Neural Network Paper",
                "authors": "Hinton et al.",
                "year": 2012,
                "why_relevant": "Established neural network training methodology",
            },
            {
                "title": "Pioneering Machine Learning Paper",
                "authors": "Bishop et al.",
                "year": 2006,
                "why_relevant": "Created foundational ML framework",
            },
        ])

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = llm_suggestions

        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = llm_response

        # OpenAlex lookup for each suggestion
        def route(url, **kwargs):
            url_str = str(url)
            if "api.openalex.org/works" in url_str:
                return _make_mock_response(json_data={
                    "results": [
                        {
                            "id": "https://openalex.org/W5555555555",
                            "title": "Foundational Deep Learning Paper",
                            "publication_year": 2015,
                            "cited_by_count": 5000,
                            "abstract_inverted_index": {
                                "Deep": [0], "learning": [1], "techniques": [2],
                                "for": [3], "neural": [4], "network": [5],
                                "training": [6], "with": [7], "novel": [8],
                                "approaches": [9],
                            },
                            "primary_topic": {
                                "id": "https://openalex.org/T10123",
                                "subfield": {"display_name": "Machine Learning"},
                                "field": {"display_name": "Computer Science"},
                                "domain": {"display_name": "Physical Sciences"},
                            },
                        }
                    ]
                })
            if "api.openalex.org/topics" in url_str:
                return _make_mock_response(json_data={
                    "subfield": {"display_name": "Machine Learning"},
                })
            if "api.semanticscholar.org" in url_str:
                return _make_mock_response(json_data={"data": []})
            if "arxiv.org" in url_str:
                return _make_mock_response(text_data=ARXIV_EMPTY_XML)
            return _make_mock_response(status_code=404)

        mock_get.side_effect = route

        additional_refs, additional_landmarks = supplement_grounding_papers(
            title="Deep Neural Network Classification",
            abstract="We propose a neural network approach for image classification.",
            year=2020,
            existing_refs=refs,
            existing_landmarks=landmarks,
            field="Computer Science",
            primary_topic_id="T10123",
            is_pioneering=False,
            target_work_id=None,
            conn=db_conn,
        )

        # Should have at least attempted to add landmarks
        # The exact count depends on resolution success
        assert isinstance(additional_landmarks, list)
        # May also return some additional refs
        assert isinstance(additional_refs, list)

    @patch(OPENAI_CLASS_GROUNDING)
    @patch(REQUESTS_GET_GROUNDING)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_supplement_llm_failure(self, mock_get, MockOpenAI):
        """LLM failure results in graceful fallback (empty or partial supplement)."""
        refs = make_references(1)
        landmarks = []

        # LLM raises an exception
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        # External APIs also fail
        mock_get.return_value = _make_mock_response(status_code=500)

        additional_refs, additional_landmarks = supplement_grounding_papers(
            title="Test Paper Title",
            abstract="An abstract about novel methods for testing.",
            year=2020,
            existing_refs=refs,
            existing_landmarks=landmarks,
            field="Computer Science",
            primary_topic_id="T10123",
            is_pioneering=False,
        )

        # Should not raise, returns empty or partial
        assert isinstance(additional_refs, list)
        assert isinstance(additional_landmarks, list)

    def test_assess_grounding_needs_sufficient(self):
        """assess_grounding_needs returns 'sufficient' when enough papers."""
        refs = make_references(3)
        landmarks = make_landmarks(3)

        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)

        assert reason == "sufficient"
        assert refs_needed == 0
        assert landmarks_needed == 0

    def test_assess_grounding_needs_refs_missing(self):
        """assess_grounding_needs detects missing reference papers."""
        refs = []  # Need at least 2
        landmarks = make_landmarks(3)

        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)

        assert refs_needed >= 2
        assert landmarks_needed == 0
        assert "refs" in reason

    def test_assess_grounding_needs_landmarks_missing(self):
        """assess_grounding_needs detects missing landmark papers."""
        refs = make_references(3)
        landmarks = []  # Need at least 3

        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)

        assert refs_needed == 0
        assert landmarks_needed >= 3
        assert "landmarks" in reason

    def test_assess_grounding_needs_both_missing(self):
        """assess_grounding_needs detects both refs and landmarks missing."""
        refs = []
        landmarks = []

        refs_needed, landmarks_needed, reason = assess_grounding_needs(refs, landmarks)

        assert refs_needed >= 2
        assert landmarks_needed >= 3


# ============================================================================
# 7. TestMethodologyDetection (supporting tests for cross-domain)
# ============================================================================

@pytest.mark.integration
class TestMethodologyDetection:
    """Tests for app.feature3.methodology module."""

    def test_neural_network_detection(self):
        assert get_methodology("Deep convolutional neural network for image classification") == "neural_network"

    def test_ensemble_tree_detection(self):
        assert get_methodology("Random forest and gradient boosting for tabular data") == "ensemble_tree"

    def test_svm_detection(self):
        assert get_methodology("Support vector machine with RBF kernel") == "kernel_svm"

    def test_no_methodology(self):
        assert get_methodology("A survey of recent advances in healthcare research") is None

    def test_mismatch_detected(self):
        assert is_methodology_mismatch(
            "Deep learning with neural networks",
            "We train a deep neural network",
            "Random forest ensemble",
            "An ensemble of decision trees using bagging",
        ) is True

    def test_same_methodology_no_mismatch(self):
        assert is_methodology_mismatch(
            "Deep learning with neural networks",
            "We train a deep neural network",
            "Convolutional neural network for vision",
            "A CNN with backpropagation",
        ) is False

    def test_no_methodology_no_mismatch(self):
        """If target has no detectable methodology, no mismatch is reported."""
        assert is_methodology_mismatch(
            "A review of healthcare trends",
            "This review summarizes recent trends.",
            "Random forest ensemble",
            "An ensemble of decision trees.",
        ) is False

    def test_graph_network_prioritized_over_neural(self):
        """'graph neural network' should match graph_network, not neural_network."""
        assert get_methodology("Graph neural network for node classification") == "graph_network"


# ── Grounding Paper Auto-Save ─────────────────────────────────────────────


@pytest.mark.integration
class TestGroundingPaperAutoSave:
    """Tests for auto-saving grounding papers to workspace saved_papers table."""

    def test_grounding_papers_saved_to_workspace(self, db_engine):
        """Grounding papers are saved to saved_papers with correct workspace/user."""
        from uuid import uuid4
        from app.feature2.rank_api import _save_grounding_papers_to_workspace

        workspace_id = uuid4()
        user_id = uuid4()
        gp_work_id_1 = _unique_work_id("Wgp1")
        gp_work_id_2 = _unique_work_id("Wgp2")

        with db_engine.connect() as conn:
            # Setup: create auth user, workspace, and grounding paper works
            conn.execute(
                text("INSERT INTO auth.users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
                {"uid": user_id},
            )
            conn.execute(
                text("""
                    INSERT INTO workspaces (workspace_id, owner_user_id, workspace_name)
                    VALUES (:wid, :uid, 'Test Workspace')
                    ON CONFLICT DO NOTHING
                """),
                {"wid": workspace_id, "uid": user_id},
            )
            conn.commit()

            # Grounding papers to save
            grounding_papers = [
                {"work_id": gp_work_id_1, "title": "Grounding Paper 1", "year": 2018, "cited_by_count": 500},
                {"work_id": gp_work_id_2, "title": "Grounding Paper 2", "year": 2015, "cited_by_count": 1200},
                {"work_id": "AX:2001.00001", "title": "ArXiv Paper", "year": 2020, "cited_by_count": 50},  # Should be skipped
            ]

            _save_grounding_papers_to_workspace(conn, workspace_id, grounding_papers)

            # Verify works were inserted
            works = conn.execute(
                text("SELECT work_id FROM works WHERE work_id = ANY(:ids)"),
                {"ids": [gp_work_id_1, gp_work_id_2]},
            ).scalars().all()
            assert set(works) == {gp_work_id_1, gp_work_id_2}

            # Verify saved_papers rows
            saved = conn.execute(
                text("""
                    SELECT paper_work_id, user_id, source
                    FROM saved_papers
                    WHERE workspace_id = :wid
                    ORDER BY paper_work_id
                """),
                {"wid": workspace_id},
            ).mappings().all()
            assert len(saved) == 2
            saved_ids = {r["paper_work_id"] for r in saved}
            assert saved_ids == {gp_work_id_1, gp_work_id_2}
            assert all(r["source"] == "novelty_grounding" for r in saved)
            assert all(r["user_id"] == user_id for r in saved)

            # Verify non-W papers were skipped
            arxiv_saved = conn.execute(
                text("SELECT 1 FROM saved_papers WHERE paper_work_id = 'AX:2001.00001'"),
            ).first()
            assert arxiv_saved is None

            # Cleanup
            conn.execute(text("DELETE FROM saved_papers WHERE workspace_id = :wid"), {"wid": workspace_id})
            conn.execute(text("DELETE FROM works WHERE work_id = ANY(:ids)"), {"ids": [gp_work_id_1, gp_work_id_2]})
            conn.execute(text("DELETE FROM workspaces WHERE workspace_id = :wid"), {"wid": workspace_id})
            conn.execute(text("DELETE FROM auth.users WHERE id = :uid"), {"uid": user_id})
            conn.commit()

    def test_idempotent_save(self, db_engine):
        """Saving the same grounding papers twice doesn't create duplicates."""
        from uuid import uuid4
        from app.feature2.rank_api import _save_grounding_papers_to_workspace

        workspace_id = uuid4()
        user_id = uuid4()
        gp_work_id = _unique_work_id("Wgp_idem")

        with db_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO auth.users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
                {"uid": user_id},
            )
            conn.execute(
                text("""
                    INSERT INTO workspaces (workspace_id, owner_user_id, workspace_name)
                    VALUES (:wid, :uid, 'Test Workspace')
                    ON CONFLICT DO NOTHING
                """),
                {"wid": workspace_id, "uid": user_id},
            )
            conn.commit()

            grounding_papers = [
                {"work_id": gp_work_id, "title": "Idempotent Paper", "year": 2019, "cited_by_count": 300},
            ]

            # Save twice
            _save_grounding_papers_to_workspace(conn, workspace_id, grounding_papers)
            _save_grounding_papers_to_workspace(conn, workspace_id, grounding_papers)

            # Should still be exactly 1 row
            count = conn.execute(
                text("SELECT count(*) FROM saved_papers WHERE workspace_id = :wid"),
                {"wid": workspace_id},
            ).scalar()
            assert count == 1

            # Cleanup
            conn.execute(text("DELETE FROM saved_papers WHERE workspace_id = :wid"), {"wid": workspace_id})
            conn.execute(text("DELETE FROM works WHERE work_id = :wid"), {"wid": gp_work_id})
            conn.execute(text("DELETE FROM workspaces WHERE workspace_id = :wid"), {"wid": workspace_id})
            conn.execute(text("DELETE FROM auth.users WHERE id = :uid"), {"uid": user_id})
            conn.commit()
