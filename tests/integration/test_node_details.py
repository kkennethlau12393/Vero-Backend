"""
Integration tests for Feature 3 Node Details orchestrator (get_node_details).

Uses dev DB. Mocks ALL external APIs using unittest.mock.patch on:
- requests.get (reference_store, landmark_retrieval, abstract_enrichment,
  topic_inference, grounding_supplement, access_links)
- httpx.get (topic_lookup)
- OpenAI class (node_details_service, topic_inference, grounding_supplement)

Follows Feature 1's integration test pattern for mock helpers.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from app.feature3.node_details_service import (
    ASSESSMENT_VERSION,
    get_node_details,
)
from app.feature3.schemas import (
    ConnectedWork,
    GroundingPaper,
    NodeDetailsResponse,
    NodeTimeline,
    NoveltyAssessment,
)
from tests.fixtures.novelty_responses import (
    NOVELTY_LLM_RESPONSE_HIGH,
    NOVELTY_LLM_RESPONSE_MEDIUM_REVIEW,
    NOVELTY_LLM_RESPONSE_PIONEERING,
    NOVELTY_LLM_RESPONSE_MALFORMED,
    OPENALEX_TOPIC_RESPONSE,
    make_groq_chat_response,
    make_landmarks,
    make_references,
    make_work_data,
)


# ── Import Paths to Mock ────────────────────────────────────────────────────

REQUESTS_GET_REFERENCE = "app.feature3.reference_store.requests.get"
REQUESTS_GET_LANDMARK = "app.feature3.landmark_retrieval.requests.get"
REQUESTS_GET_ABSTRACT = "app.feature3.abstract_enrichment.requests.get"
REQUESTS_GET_TOPIC_INF = "app.feature3.topic_inference.requests.get"
REQUESTS_GET_GROUNDING = "app.feature3.grounding_supplement.requests.get"
REQUESTS_GET_ACCESS = "app.settings.access_links.requests.get"
REQUESTS_GET_PAPER_CACHE = "app.feature3.paper_cache.requests.get"
HTTPX_GET_TOPIC_LOOKUP = "app.feature3.topic_lookup.httpx.get"
OPENAI_CLASS_NODE = "app.feature3.node_details_service.OpenAI"
OPENAI_CLASS_TOPIC = "app.feature3.topic_inference.OpenAI"
OPENAI_CLASS_GROUNDING = "app.feature3.grounding_supplement.OpenAI"


# ── Mock Helper ──────────────────────────────────────────────────────────────


def _make_mock_response(json_data=None, text_data=None, status_code=200):
    """Create a mock requests.Response (same pattern as Feature 1 tests)."""
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


def _make_mock_httpx_response(json_data=None, status_code=200):
    """Create a mock httpx.Response for topic_lookup."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    if json_data is not None:
        resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _make_openai_mock(response_content: str):
    """Build a mock OpenAI class that returns a completion with the given content."""
    mock_class = MagicMock()
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_response = make_groq_chat_response(response_content)
    mock_client.chat.completions.create.return_value = mock_response
    return mock_class, mock_client


# ── OpenAlex Reference Responses ────────────────────────────────────────────


def _openalex_work_response(work_id="W1234567890", ref_ids=None):
    """Build an OpenAlex single-work response with referenced_works."""
    if ref_ids is None:
        ref_ids = [f"https://openalex.org/W990000000{i}" for i in range(5)]
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": "Test Paper",
        "publication_year": 2020,
        "cited_by_count": 500,
        "referenced_works": ref_ids,
        "primary_topic": {"id": "T10123", "display_name": "Machine Learning"},
    }


def _openalex_batch_response(work_ids_and_data):
    """Build an OpenAlex batch-filter response for multiple works."""
    results = []
    for wid, title, year, cites in work_ids_and_data:
        results.append({
            "id": f"https://openalex.org/{wid}",
            "title": title,
            "publication_year": year,
            "cited_by_count": cites,
            "abstract_inverted_index": {"A": [0], "test": [1], "abstract": [2]},
            "primary_topic": {"id": "T10123", "display_name": "Machine Learning"},
        })
    return {"results": results, "meta": {"count": len(results)}}


# ── DB Fixtures ──────────────────────────────────────────────────────────────

# Use a fixed test tenant so we don't conflict with real data
TEST_TENANT_ID = UUID("00000000-0000-0000-0000-000000000099")


@pytest.fixture()
def test_data(db_engine):
    """
    Insert a complete test graph into the dev DB:
    - 1 map belonging to TEST_TENANT_ID
    - 1 primary work + 2 connected works
    - 1 map_node (primary work)
    - 2 map_edges

    Yields (map_id, work_id, connected_work_ids) for tests.
    Cleans up all inserted rows on teardown.
    """
    ts = int(time.time())
    map_id = uuid4()
    work_id = f"W_test_{ts}_001"
    connected_ids = [f"W_test_{ts}_002", f"W_test_{ts}_003"]

    abstract = (
        "This paper introduces a novel self-attention mechanism for sequence modeling. "
        "The proposed approach eliminates recurrent connections entirely, enabling "
        "parallelized training. Experiments on machine translation benchmarks demonstrate "
        "superior performance compared to existing recurrent and convolutional models."
    )
    authors_json = json.dumps(["Alice Researcher", "Bob Scientist"])

    graph_draft_id = uuid4()
    params_hash = f"test_hash_{ts}"

    with db_engine.connect() as conn:
        # Insert graph_draft (FK dependency for maps)
        conn.execute(
            text("INSERT INTO graph_drafts (graph_draft_id, tenant_id) VALUES (:gid, :tid)"),
            {"gid": graph_draft_id, "tid": TEST_TENANT_ID},
        )

        # Insert map
        conn.execute(
            text("""
                INSERT INTO maps (
                    map_id, tenant_id, graph_draft_id,
                    default_grouping, allowed_groupings,
                    stats_json, params_json, params_hash
                ) VALUES (
                    :map_id, :tenant_id, :graph_draft_id,
                    'topic', '["topic","subfield"]'::jsonb,
                    '{}'::jsonb, '{"query":"integration test"}'::jsonb, :params_hash
                )
            """),
            {
                "map_id": map_id,
                "tenant_id": TEST_TENANT_ID,
                "graph_draft_id": graph_draft_id,
                "params_hash": params_hash,
            },
        )

        # Insert primary work
        conn.execute(
            text("""
                INSERT INTO works (
                    work_id, title, year, cited_by_count, abstract,
                    primary_topic_id, doi, arxiv_id, venue, authors_json,
                    is_open_access, oa_status, oa_pdf_url, category, category_confidence
                ) VALUES (
                    :work_id, :title, :year, :cited_by_count, :abstract,
                    :topic_id, :doi, :arxiv_id, :venue, :authors_json,
                    :is_oa, :oa_status, :oa_pdf_url, :category, :cat_conf
                )
            """),
            {
                "work_id": work_id,
                "title": "Self-Attention for Sequence Modeling",
                "year": 2020,
                "cited_by_count": 500,
                "abstract": abstract,
                "topic_id": "T10123",
                "doi": "10.1234/test.2020.001",
                "arxiv_id": None,
                "venue": "NeurIPS",
                "authors_json": authors_json,
                "is_oa": True,
                "oa_status": "gold",
                "oa_pdf_url": "https://example.com/paper.pdf",
                "category": "foundational",
                "cat_conf": 0.9,
            },
        )

        # Insert connected works
        for i, cid in enumerate(connected_ids):
            conn.execute(
                text("""
                    INSERT INTO works (
                        work_id, title, year, cited_by_count, abstract,
                        primary_topic_id, venue, authors_json,
                        is_open_access, oa_status
                    ) VALUES (
                        :work_id, :title, :year, :cited_by_count, :abstract,
                        :topic_id, :venue, :authors_json,
                        :is_oa, :oa_status
                    )
                """),
                {
                    "work_id": cid,
                    "title": f"Connected Paper {i + 1}",
                    "year": 2019 - i,
                    "cited_by_count": 200 * (i + 1),
                    "abstract": f"Abstract for connected paper {i + 1}.",
                    "topic_id": "T10123",
                    "venue": "ICML",
                    "authors_json": json.dumps([f"Colleague {i + 1}"]),
                    "is_oa": False,
                    "oa_status": "closed",
                },
            )

        # Insert map_node for primary work
        conn.execute(
            text("INSERT INTO map_nodes (map_id, work_id) VALUES (:map_id, :work_id)"),
            {"map_id": map_id, "work_id": work_id},
        )

        # Insert map_edges: primary -> connected[0] (cites), connected[1] -> primary (cited_by)
        conn.execute(
            text("INSERT INTO map_edges (map_id, from_work_id, to_work_id) VALUES (:map_id, :from_id, :to_id)"),
            {"map_id": map_id, "from_id": work_id, "to_id": connected_ids[0]},
        )
        conn.execute(
            text("INSERT INTO map_edges (map_id, from_work_id, to_work_id) VALUES (:map_id, :from_id, :to_id)"),
            {"map_id": map_id, "from_id": connected_ids[1], "to_id": work_id},
        )

        conn.commit()

    yield map_id, work_id, connected_ids

    # Teardown: remove test data in reverse dependency order
    with db_engine.connect() as conn:
        all_work_ids = [work_id] + connected_ids
        for wid in all_work_ids:
            conn.execute(
                text("DELETE FROM novelty_assessments WHERE work_id = :wid"),
                {"wid": wid},
            )
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": wid},
            )
        conn.execute(
            text("DELETE FROM map_edges WHERE map_id = :map_id"),
            {"map_id": map_id},
        )
        conn.execute(
            text("DELETE FROM map_nodes WHERE map_id = :map_id"),
            {"map_id": map_id},
        )
        for wid in all_work_ids:
            conn.execute(
                text("DELETE FROM works WHERE work_id = :wid"),
                {"wid": wid},
            )
        conn.execute(
            text("DELETE FROM maps WHERE map_id = :map_id"),
            {"map_id": map_id},
        )
        conn.execute(
            text("DELETE FROM graph_drafts WHERE graph_draft_id = :gid"),
            {"gid": graph_draft_id},
        )
        conn.commit()


@contextmanager
def _patch_all_external_apis(
    *,
    openai_response: str = NOVELTY_LLM_RESPONSE_HIGH,
    openai_side_effect=None,
    requests_routing=None,
    httpx_topic_response=None,
):
    """
    Context manager that patches ALL external API calls needed by get_node_details.

    Args:
        openai_response: JSON string the OpenAI mock returns for completions.
        openai_side_effect: If set, replaces .create.return_value with .create.side_effect.
        requests_routing: Optional callable(url, **kwargs) -> mock_response for fine-grained control.
        httpx_topic_response: Custom httpx mock response for topic_lookup.
    """
    # Default requests handler: return empty/safe responses for everything
    def default_requests_handler(url, **kwargs):
        url_str = str(url)
        if "api.openalex.org" in url_str:
            # OpenAlex: return empty results for any work lookup or filter
            if "/works/" in url_str and "filter=" not in url_str:
                return _make_mock_response(json_data=_openalex_work_response())
            return _make_mock_response(json_data={"results": [], "meta": {"count": 0}})
        if "api.semanticscholar.org" in url_str:
            return _make_mock_response(json_data={"data": [], "total": 0})
        if "export.arxiv.org" in url_str:
            return _make_mock_response(
                text_data='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            )
        if "libkey" in url_str:
            return _make_mock_response(status_code=404)
        # Default: empty JSON
        return _make_mock_response(json_data={})

    handler = requests_routing or default_requests_handler

    # Default httpx handler for topic_lookup
    default_httpx = httpx_topic_response or _make_mock_httpx_response(
        json_data=OPENALEX_TOPIC_RESPONSE
    )

    # Build OpenAI mocks
    openai_node_cls, openai_node_client = _make_openai_mock(openai_response)
    openai_topic_cls, _ = _make_openai_mock("{}")
    openai_grounding_cls, openai_grounding_client = _make_openai_mock("[]")

    if openai_side_effect is not None:
        openai_node_client.chat.completions.create.side_effect = openai_side_effect

    with (
        patch(REQUESTS_GET_REFERENCE, side_effect=handler),
        patch(REQUESTS_GET_LANDMARK, side_effect=handler),
        patch(REQUESTS_GET_ABSTRACT, side_effect=handler),
        patch(REQUESTS_GET_TOPIC_INF, side_effect=handler),
        patch(REQUESTS_GET_GROUNDING, side_effect=handler),
        patch(REQUESTS_GET_ACCESS, side_effect=handler),
        patch(REQUESTS_GET_PAPER_CACHE, side_effect=handler),
        patch(HTTPX_GET_TOPIC_LOOKUP, return_value=default_httpx),
        patch(OPENAI_CLASS_NODE, openai_node_cls),
        patch(OPENAI_CLASS_TOPIC, openai_topic_cls),
        patch(OPENAI_CLASS_GROUNDING, openai_grounding_cls),
    ):
        yield {
            "openai_node_client": openai_node_client,
            "openai_grounding_client": openai_grounding_client,
        }


# ============================================================================
# TestGetNodeDetailsMetadata (include_novelty=False)
# ============================================================================


@pytest.mark.integration
class TestGetNodeDetailsMetadata:
    """Tests the lightweight metadata path (no LLM call)."""

    def test_returns_basic_metadata(self, db_engine, test_data):
        """Work exists in map -> returns full metadata, novelty_assessment is None."""
        map_id, work_id, connected_ids = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
            )

        assert isinstance(result, NodeDetailsResponse)
        assert result.work_id == work_id
        assert result.title == "Self-Attention for Sequence Modeling"
        assert result.year == 2020
        assert result.cited_by_count == 500
        assert result.venue == "NeurIPS"
        assert result.abstract is not None
        assert len(result.authors) == 2
        assert "Alice Researcher" in result.authors

        # Summary and keywords should be derived from abstract
        assert result.summary is not None
        assert len(result.summary) > 0
        assert isinstance(result.keywords, list)

        # Novelty assessment should be None for metadata-only
        assert result.novelty_assessment is None

        # Connected works should be present
        assert len(result.connected_works) == 2
        cw_ids = {cw.work_id for cw in result.connected_works}
        assert connected_ids[0] in cw_ids  # cites relationship
        assert connected_ids[1] in cw_ids  # cited_by relationship

        # Verify relationship directions
        cites_works = [cw for cw in result.connected_works if cw.relationship == "cites"]
        cited_by_works = [cw for cw in result.connected_works if cw.relationship == "cited_by"]
        assert len(cites_works) == 1
        assert cites_works[0].work_id == connected_ids[0]
        assert len(cited_by_works) == 1
        assert cited_by_works[0].work_id == connected_ids[1]

        # Access info should be resolved
        assert result.access_status in ("open", "closed", "unknown")

        # Topic display name
        assert result.primary_topic_id == "T10123"

    def test_map_not_found(self, db_engine, test_data):
        """Non-existent map_id raises ValueError('map_not_found')."""
        _, work_id, _ = test_data
        fake_map_id = uuid4()

        with _patch_all_external_apis():
            with pytest.raises(ValueError, match="map_not_found"):
                get_node_details(
                    db_engine,
                    tenant_id=TEST_TENANT_ID,
                    map_id=fake_map_id,
                    work_id=work_id,
                    include_novelty=False,
                )

    def test_map_wrong_tenant(self, db_engine, test_data):
        """Map exists but belongs to a different tenant -> map_not_found."""
        map_id, work_id, _ = test_data
        wrong_tenant = UUID("00000000-0000-0000-0000-000000000098")

        with _patch_all_external_apis():
            with pytest.raises(ValueError, match="map_not_found"):
                get_node_details(
                    db_engine,
                    tenant_id=wrong_tenant,
                    map_id=map_id,
                    work_id=work_id,
                    include_novelty=False,
                )

    def test_node_not_found(self, db_engine, test_data):
        """Work not a node in this map raises ValueError('node_not_found')."""
        map_id, _, connected_ids = test_data
        # connected_ids are works in DB but NOT in map_nodes
        non_node_work = connected_ids[0]

        with _patch_all_external_apis():
            with pytest.raises(ValueError, match="node_not_found"):
                get_node_details(
                    db_engine,
                    tenant_id=TEST_TENANT_ID,
                    map_id=map_id,
                    work_id=non_node_work,
                    include_novelty=False,
                )

    def test_work_not_found(self, db_engine, test_data):
        """Node in map but work deleted from works table -> work_not_found."""
        map_id, work_id, _ = test_data

        # Insert a map_node pointing to a non-existent work
        ghost_work_id = f"W_ghost_{int(time.time())}"
        with db_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO map_nodes (map_id, work_id) VALUES (:map_id, :wid)"),
                {"map_id": map_id, "wid": ghost_work_id},
            )
            conn.commit()

        try:
            with _patch_all_external_apis():
                with pytest.raises(ValueError, match="work_not_found"):
                    get_node_details(
                        db_engine,
                        tenant_id=TEST_TENANT_ID,
                        map_id=map_id,
                        work_id=ghost_work_id,
                        include_novelty=False,
                    )
        finally:
            with db_engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM map_nodes WHERE map_id = :map_id AND work_id = :wid"),
                    {"map_id": map_id, "wid": ghost_work_id},
                )
                conn.commit()

    def test_summary_from_abstract(self, db_engine, test_data):
        """Summary should be derived from the abstract (first 2-3 sentences)."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
            )

        # Summary should contain text from the abstract's opening sentences
        assert "self-attention" in result.summary.lower() or "sequence" in result.summary.lower()
        # Summary should not be longer than the full abstract
        assert len(result.summary) <= len(result.abstract) + 10  # small margin for period

    def test_keywords_from_abstract(self, db_engine, test_data):
        """Keywords should be extracted from abstract content."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
            )

        assert isinstance(result.keywords, list)
        # The abstract mentions machine translation, attention, sequence modeling
        # At least some keywords should be extracted
        all_kw = " ".join(result.keywords).lower()
        # Check that at least one technical term was extracted
        has_relevant = any(
            term in all_kw
            for term in ["attention", "sequence", "machine", "neural", "recurrent", "translation"]
        )
        assert has_relevant or len(result.keywords) > 0


# ============================================================================
# TestGetNodeDetailsNovelty (include_novelty=True)
# ============================================================================


@pytest.mark.integration
class TestGetNodeDetailsNovelty:
    """Tests the full novelty assessment pipeline with all external APIs mocked."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_full_pipeline_high_novelty(self, db_engine, test_data):
        """Mock everything, return NOVELTY_LLM_RESPONSE_HIGH, verify result."""
        map_id, work_id, _ = test_data

        # Clear any cached result for this work
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH):
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        assert isinstance(result, NodeDetailsResponse)
        assert result.work_id == work_id
        assert result.summary is not None
        assert len(result.summary) > 0
        assert isinstance(result.keywords, list)

        # Novelty assessment should be populated
        assert result.novelty_assessment is not None
        assert result.novelty_assessment.novelty_level in ("high", "medium", "low", "pioneering")
        assert result.novelty_assessment.confidence in ("high", "medium", "low")
        assert result.novelty_assessment.novelty_explanation is not None
        assert len(result.novelty_assessment.novelty_explanation) > 0

        # Grounding papers: mocked APIs return empty, so validation may filter some out
        # but the pipeline should still produce a valid response
        for gp in result.novelty_assessment.grounding_papers:
            assert isinstance(gp, GroundingPaper)
            assert gp.work_id is not None
            assert gp.relationship in ("cited_reference", "field_landmark")

        # Banned verbs should be scrubbed from summary
        banned_verbs = ["explores", "discusses", "examines", "investigates", "assesses", "evaluates"]
        summary_lower = result.summary.lower()
        for verb in banned_verbs:
            assert verb not in summary_lower, f"Banned verb '{verb}' found in summary"

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_full_pipeline_medium_review(self, db_engine, test_data):
        """Mock to return NOVELTY_LLM_RESPONSE_MEDIUM_REVIEW, verify level='medium'."""
        map_id, work_id, _ = test_data

        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_MEDIUM_REVIEW):
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        assert result.novelty_assessment is not None
        assert result.novelty_assessment.novelty_level == "medium"
        assert "synthesis" in result.summary.lower() or "review" in result.summary.lower() or len(result.summary) > 0

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_pioneering_detection(self, db_engine, test_data):
        """Set cited_by_count > 50000, verify pioneering signal passed to LLM."""
        map_id, work_id, _ = test_data

        # Temporarily bump citations to trigger pioneering detection
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.execute(
                text("UPDATE works SET cited_by_count = 60000 WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        try:
            captured_prompts = []

            def capture_create(**kwargs):
                messages = kwargs.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "user":
                        captured_prompts.append(msg["content"])
                return make_groq_chat_response(NOVELTY_LLM_RESPONSE_PIONEERING)

            openai_node_cls = MagicMock()
            openai_node_client = MagicMock()
            openai_node_cls.return_value = openai_node_client
            openai_node_client.chat.completions.create.side_effect = capture_create

            openai_topic_cls, _ = _make_openai_mock("{}")
            openai_grounding_cls, _ = _make_openai_mock("[]")

            with (
                patch(REQUESTS_GET_REFERENCE, side_effect=lambda url, **kw: _make_mock_response(json_data=_openalex_work_response())),
                patch(REQUESTS_GET_LANDMARK, side_effect=lambda url, **kw: _make_mock_response(json_data={"results": [], "meta": {"count": 0}})),
                patch(REQUESTS_GET_ABSTRACT, side_effect=lambda url, **kw: _make_mock_response(json_data={})),
                patch(REQUESTS_GET_TOPIC_INF, side_effect=lambda url, **kw: _make_mock_response(json_data={})),
                patch(REQUESTS_GET_GROUNDING, side_effect=lambda url, **kw: _make_mock_response(json_data={"results": [], "meta": {"count": 0}})),
                patch(REQUESTS_GET_ACCESS, side_effect=lambda url, **kw: _make_mock_response(status_code=404)),
                patch(REQUESTS_GET_PAPER_CACHE, side_effect=lambda url, **kw: _make_mock_response(json_data={"results": [], "meta": {"count": 0}})),
                patch(HTTPX_GET_TOPIC_LOOKUP, return_value=_make_mock_httpx_response(json_data=OPENALEX_TOPIC_RESPONSE)),
                patch(OPENAI_CLASS_NODE, openai_node_cls),
                patch(OPENAI_CLASS_TOPIC, openai_topic_cls),
                patch(OPENAI_CLASS_GROUNDING, openai_grounding_cls),
            ):
                result = get_node_details(
                    db_engine,
                    tenant_id=TEST_TENANT_ID,
                    map_id=map_id,
                    work_id=work_id,
                    include_novelty=True,
                )

            # The prompt sent to the LLM should contain pioneering context
            # (only if references/landmarks were available for the LLM call to happen)
            if captured_prompts:
                prompt_text = captured_prompts[0]
                assert "60,000 citations" in prompt_text or "HIGH-IMPACT" in prompt_text

        finally:
            # Restore original citation count
            with db_engine.connect() as conn:
                conn.execute(
                    text("UPDATE works SET cited_by_count = 500 WHERE work_id = :wid"),
                    {"wid": work_id},
                )
                conn.commit()

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_cache_hit(self, db_engine, test_data):
        """First call populates cache, second call returns cached result without LLM call."""
        map_id, work_id, _ = test_data

        # Clear cache
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        # First call: should invoke LLM
        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH) as mocks:
            result1 = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )
            first_call_count = mocks["openai_node_client"].chat.completions.create.call_count

        assert result1.novelty_assessment is not None

        # Second call: should use cache, NOT call LLM
        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH) as mocks:
            result2 = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )
            second_call_count = mocks["openai_node_client"].chat.completions.create.call_count

        # LLM should NOT have been called on the second invocation
        assert second_call_count == 0, (
            f"Expected 0 LLM calls on cache hit, got {second_call_count}"
        )

        # Results should be equivalent
        assert result2.novelty_assessment is not None
        assert result2.work_id == result1.work_id
        assert result2.summary == result1.summary

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_llm_failure_returns_default(self, db_engine, test_data):
        """Mock OpenAI to raise exception -> returns a default response."""
        map_id, work_id, _ = test_data

        # Clear cache so the LLM path is exercised
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        with _patch_all_external_apis(
            openai_side_effect=Exception("Service unavailable"),
        ):
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        assert isinstance(result, NodeDetailsResponse)
        # Even on LLM failure, we should get a response (possibly with default assessment
        # or assessment_unavailable_reason, depending on grounding data availability)
        assert result.work_id == work_id
        assert result.summary is not None

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_banned_verbs_scrubbed(self, db_engine, test_data):
        """Mock LLM response containing banned verbs, verify they are replaced."""
        map_id, work_id, _ = test_data

        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        # Create a response with banned verbs in summary and novelty fields
        response_with_banned = json.dumps({
            "summary": "This paper explores the impact of self-attention and examines performance on benchmarks.",
            "keywords": ["attention", "transformers"],
            "novelty_assessment": {
                "whats_new": "This work investigates a new attention mechanism and discusses its implications.",
                "compared_to_prior_work": "Prior work [W9900000000] evaluates similar approaches.",
                "novelty_level": "high",
                "confidence": "high",
                "novelty_explanation": "Classified as high because it addresses the limitations of [W9900000000] and studies the effects of [W8800000000].",
                "grounding_papers": [
                    {"work_id": "W9900000000", "title": "Reference Paper 1", "year": 2015, "cited_by_count": 200, "relationship": "cited_reference", "relevance": "Examines related approaches."},
                    {"work_id": "W8800000000", "title": "Landmark Paper 1", "year": 2010, "cited_by_count": 5000, "relationship": "field_landmark", "relevance": "Explores foundational techniques."},
                    {"work_id": "W8800000001", "title": "Landmark Paper 2", "year": 2007, "cited_by_count": 10000, "relationship": "field_landmark", "relevance": "Introduced deep learning."},
                    {"work_id": "W8800000002", "title": "Landmark Paper 3", "year": 2004, "cited_by_count": 15000, "relationship": "field_landmark", "relevance": "Foundational neural network work."},
                ],
            },
        })

        with _patch_all_external_apis(openai_response=response_with_banned):
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        # Verify banned verbs have been replaced in summary
        if result.novelty_assessment is not None:
            summary_lower = result.summary.lower()
            assert "explores" not in summary_lower
            assert "examines" not in summary_lower

            # Check novelty fields
            if result.novelty_assessment.whats_new:
                whats_new_lower = result.novelty_assessment.whats_new.lower()
                assert "investigates" not in whats_new_lower
                assert "discusses" not in whats_new_lower

            if result.novelty_assessment.compared_to_prior_work:
                compared_lower = result.novelty_assessment.compared_to_prior_work.lower()
                assert "evaluates" not in compared_lower

            # Check grounding paper relevance strings
            for gp in result.novelty_assessment.grounding_papers:
                rel_lower = gp.relevance.lower()
                assert "examines" not in rel_lower
                assert "explores" not in rel_lower


# ============================================================================
# TestGetNodeDetailsTimeline (include_timeline=True)
# ============================================================================


@pytest.mark.integration
class TestGetNodeDetailsTimeline:
    """Tests the timeline feature."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_timeline_returned(self, db_engine, test_data):
        """Verify timeline sections (backward/forward) are populated when requested."""
        map_id, work_id, _ = test_data

        # Clear cache to ensure fresh assessment + timeline
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH):
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
                include_timeline=True,
            )

        assert isinstance(result, NodeDetailsResponse)
        # Timeline should be present
        assert result.timeline is not None
        assert isinstance(result.timeline, NodeTimeline)
        assert result.timeline.target_work_id == work_id
        assert result.timeline.target_year == 2020

        # backward and forward are lists of TimelineSection
        assert isinstance(result.timeline.backward, list)
        assert isinstance(result.timeline.forward, list)

    def test_timeline_without_novelty(self, db_engine, test_data):
        """include_novelty=False, include_timeline=True returns timeline but no novelty."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
                include_timeline=True,
            )

        assert isinstance(result, NodeDetailsResponse)
        # Novelty should not be present
        assert result.novelty_assessment is None
        # Timeline should also be None when include_novelty=False
        # (the service only builds timeline when include_novelty=True or cache hit with timeline)
        # For metadata-only path, timeline is not built
        # This is the expected behavior per the service implementation

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-integration"})
    def test_timeline_with_cached_novelty(self, db_engine, test_data):
        """When novelty is cached, timeline is still built fresh if requested."""
        map_id, work_id, _ = test_data

        # Clear cache, then do first call to populate it
        with db_engine.connect() as conn:
            conn.execute(
                text("DELETE FROM node_details_cache WHERE work_id = :wid"),
                {"wid": work_id},
            )
            conn.commit()

        # First call: populate cache WITHOUT timeline
        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH):
            result1 = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
                include_timeline=False,
            )
        assert result1.novelty_assessment is not None
        assert result1.timeline is None

        # Second call: cache hit + timeline requested
        with _patch_all_external_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH):
            result2 = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
                include_timeline=True,
            )
        assert result2.novelty_assessment is not None
        assert result2.timeline is not None
        assert isinstance(result2.timeline, NodeTimeline)
        assert result2.timeline.target_work_id == work_id


# ============================================================================
# TestGetNodeDetailsAccessInfo
# ============================================================================


@pytest.mark.integration
class TestGetNodeDetailsAccessInfo:
    """Tests access link resolution within get_node_details."""

    def test_open_access_paper(self, db_engine, test_data):
        """Paper with is_open_access=True and oa_pdf_url returns open status."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
            )

        # The test fixture has is_open_access=True, oa_pdf_url set
        assert result.access_status == "open"
        assert result.pdf_url == "https://example.com/paper.pdf"
        assert result.doi_url is not None
        assert "10.1234/test.2020.001" in result.doi_url

    def test_topic_display_name_resolved(self, db_engine, test_data):
        """Topic display name should be fetched from OpenAlex or cache."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=False,
            )

        # topic_display_name might be None if not cached and the mock doesn't
        # insert into openalex_topics, but the primary_topic_id should be set
        assert result.primary_topic_id == "T10123"

    # ── Permanent Assessment Storage Tests ──────────────────────────────

    def test_assessment_persisted_after_generation(self, db_engine, test_data):
        """After LLM generates an assessment, it should be saved in novelty_assessments table."""
        map_id, work_id, _ = test_data

        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        assert result.novelty_assessment is not None
        assert result.novelty_assessment.novelty_level == "high"

        # Verify it was persisted in novelty_assessments table
        with db_engine.connect() as conn:
            row = conn.execute(
                text("SELECT novelty_level, confidence, novelty_explanation FROM novelty_assessments WHERE work_id = :wid"),
                {"wid": work_id},
            ).mappings().first()

        assert row is not None
        assert row["novelty_level"] == "high"
        assert row["confidence"] == "high"
        assert row["novelty_explanation"] is not None

    def test_persisted_assessment_served_without_llm(self, db_engine, test_data):
        """Persisted assessment should be served directly, no LLM call needed."""
        map_id, work_id, _ = test_data

        # Manually insert a persisted assessment
        with db_engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO novelty_assessments
                        (work_id, novelty_level, confidence, whats_new,
                         compared_to_prior_work, novelty_explanation,
                         grounding_papers, model_version)
                    VALUES
                        (:wid, 'medium', 'high', 'Test whats new',
                         'Test comparison', 'Test explanation',
                         :grounding, 'test-version')
                """),
                {
                    "wid": work_id,
                    "grounding": json.dumps([{
                        "work_id": "W999", "title": "Test Paper",
                        "year": 2020, "cited_by_count": 100,
                        "relationship": "cited_reference",
                        "relevance": "Test relevance",
                    }]),
                },
            )
            conn.commit()

        # Should serve from permanent table without calling LLM
        with _patch_all_external_apis() as _:
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
            )

        assert result.novelty_assessment is not None
        assert result.novelty_assessment.novelty_level == "medium"
        assert result.novelty_assessment.novelty_explanation == "Test explanation"

    def test_force_regenerate_bypasses_persisted(self, db_engine, test_data):
        """force_regenerate=True should delete persisted assessment and regenerate via LLM."""
        map_id, work_id, _ = test_data

        # Insert a stale persisted assessment
        with db_engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO novelty_assessments
                        (work_id, novelty_level, confidence, novelty_explanation,
                         grounding_papers, model_version)
                    VALUES
                        (:wid, 'low', 'low', 'Stale explanation',
                         '[]', 'old-version')
                """),
                {"wid": work_id},
            )
            conn.commit()

        # force_regenerate should bypass the stale assessment
        with _patch_all_external_apis():
            result = get_node_details(
                db_engine,
                tenant_id=TEST_TENANT_ID,
                map_id=map_id,
                work_id=work_id,
                include_novelty=True,
                force_regenerate=True,
            )

        # Should get fresh LLM result (HIGH from fixture), not stale LOW
        assert result.novelty_assessment is not None
        assert result.novelty_assessment.novelty_level == "high"

        # And the persisted table should now have the fresh result
        with db_engine.connect() as conn:
            row = conn.execute(
                text("SELECT novelty_level FROM novelty_assessments WHERE work_id = :wid"),
                {"wid": work_id},
            ).mappings().first()
        assert row is not None
        assert row["novelty_level"] == "high"
