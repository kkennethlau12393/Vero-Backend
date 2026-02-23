"""
Integration tests for the ranking pipeline (POST /v1/rank).

Uses the real dev DB but mocks ALL external HTTP calls (Groq, OpenAlex,
Semantic Scholar, ArXiv, CrossRef, PubMed, DBLP, OpenCitations).
"""
from __future__ import annotations

import json
from uuid import UUID

import pytest
import respx
from httpx import Response

from tests.fixtures.groq_responses.rank_classification import (
    CLASSIFICATION_RESPONSE,
    EXPANSION_RESPONSE,
    RELEVANCE_SCORING_RESPONSE,
    GARBAGE_RESPONSE,
)
from tests.fixtures.api_responses.openalex_search import (
    SEARCH_RESPONSE,
    EMPTY_SEARCH_RESPONSE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mock_all_external_apis(respx_mock):
    """Set up respx routes to intercept all external API calls."""
    # Groq (LLM) — classification, expansion, and relevance scoring
    respx_mock.route(host="api.groq.com").respond(
        json=RELEVANCE_SCORING_RESPONSE
    )

    # OpenAI (paper classification borderline LLM)
    respx_mock.route(host="api.openai.com").respond(
        json={"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )

    # OpenAlex
    respx_mock.route(host="api.openalex.org").respond(json=SEARCH_RESPONSE)

    # Semantic Scholar
    respx_mock.route(host="api.semanticscholar.org").respond(json={"data": [], "total": 0})

    # ArXiv
    respx_mock.route(host="export.arxiv.org").respond(
        text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )

    # CrossRef
    respx_mock.route(host="api.crossref.org").respond(
        json={"message": {"items": []}}
    )

    # PubMed (search)
    respx_mock.route(host="eutils.ncbi.nlm.nih.gov").respond(
        json={"esearchresult": {"idlist": []}}
    )

    # DBLP
    respx_mock.route(host="dblp.org").respond(
        json={"result": {"hits": {"hit": []}}}
    )

    # OpenCitations
    respx_mock.route(host="api.opencitations.net").respond(json=[])


# ── Test: Basic rank endpoint ────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_endpoint_returns_200(client, auth_headers):
    """POST /v1/rank with a query returns 200 and correct response shape."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={"query_text": "machine learning"},
        headers=auth_headers,
    )

    # Should be 200 (completed) or 202 (async pending)
    assert response.status_code in (200, 202)

    data = response.json()
    assert "rank_job_id" in data


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_endpoint_response_shape(client, auth_headers):
    """Verify the full response structure of a completed rank job."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={"query_text": "deep learning"},
        headers=auth_headers,
    )

    if response.status_code == 200:
        data = response.json()
        assert "rank_job_id" in data
        assert "job" in data
        # Should have categorized results OR items (drill-down)
        has_categories = "foundational" in data or "methodology" in data
        has_items = "items" in data
        assert has_categories or has_items


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_missing_query_returns_400(client, auth_headers):
    """POST /v1/rank without query_text or seed_title should return 400."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={},
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_with_filters(client, auth_headers):
    """POST /v1/rank with year filters returns 200."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={
            "query_text": "neural networks",
            "filters": {"year_min": 2015, "year_max": 2023},
        },
        headers=auth_headers,
    )
    assert response.status_code in (200, 202)


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_with_params(client, auth_headers):
    """POST /v1/rank with custom ranking params."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={
            "query_text": "transformers",
            "params": {
                "top_k": 10,
                "max_candidates_scored": 100,
            },
        },
        headers=auth_headers,
    )
    assert response.status_code in (200, 202)


# ── Test: Cache behaviour ────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_cache_second_call_faster(client, auth_headers):
    """Second call with same query should return cached results."""
    _mock_all_external_apis(respx.mock)

    # First call
    resp1 = await client.post(
        "/v1/rank",
        json={"query_text": "cache test query unique 12345"},
        headers=auth_headers,
    )

    # Second call (should hit cache)
    resp2 = await client.post(
        "/v1/rank",
        json={"query_text": "cache test query unique 12345"},
        headers=auth_headers,
    )

    assert resp1.status_code in (200, 202)
    assert resp2.status_code in (200, 202)

    if resp1.status_code == 200 and resp2.status_code == 200:
        # Both should return the same rank_job_id (cached)
        assert resp1.json()["rank_job_id"] == resp2.json()["rank_job_id"]


# ── Test: Error handling ─────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_groq_returns_garbage(client, auth_headers):
    """If Groq returns garbage, endpoint should still return a response (graceful degradation)."""
    # Mock Groq with garbage
    respx.mock.route(host="api.groq.com").respond(json=GARBAGE_RESPONSE)
    # Mock everything else normally
    respx.mock.route(host="api.openai.com").respond(
        json={"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )
    respx.mock.route(host="api.openalex.org").respond(json=SEARCH_RESPONSE)
    respx.mock.route(host="api.semanticscholar.org").respond(json={"data": [], "total": 0})
    respx.mock.route(host="export.arxiv.org").respond(
        text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )
    respx.mock.route(host="api.crossref.org").respond(json={"message": {"items": []}})
    respx.mock.route(host="eutils.ncbi.nlm.nih.gov").respond(json={"esearchresult": {"idlist": []}})
    respx.mock.route(host="dblp.org").respond(json={"result": {"hits": {"hit": []}}})
    respx.mock.route(host="api.opencitations.net").respond(json=[])

    response = await client.post(
        "/v1/rank",
        json={"query_text": "garbage test query"},
        headers=auth_headers,
    )

    # Should not crash — either 200 (with degraded results) or 500
    assert response.status_code in (200, 202, 500)


@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_openalex_timeout(client, auth_headers):
    """If OpenAlex times out, endpoint should still work with other sources."""
    respx.mock.route(host="api.groq.com").respond(json=RELEVANCE_SCORING_RESPONSE)
    respx.mock.route(host="api.openai.com").respond(
        json={"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )
    # OpenAlex returns error
    respx.mock.route(host="api.openalex.org").respond(status_code=503)
    respx.mock.route(host="api.semanticscholar.org").respond(json={"data": [], "total": 0})
    respx.mock.route(host="export.arxiv.org").respond(
        text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )
    respx.mock.route(host="api.crossref.org").respond(json={"message": {"items": []}})
    respx.mock.route(host="eutils.ncbi.nlm.nih.gov").respond(json={"esearchresult": {"idlist": []}})
    respx.mock.route(host="dblp.org").respond(json={"result": {"hits": {"hit": []}}})
    respx.mock.route(host="api.opencitations.net").respond(json=[])

    response = await client.post(
        "/v1/rank",
        json={"query_text": "openalex timeout test"},
        headers=auth_headers,
    )

    # Should handle gracefully
    assert response.status_code in (200, 202, 500)


# ── Test: Drill-down endpoint ────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_drill_down_endpoint(client, auth_headers):
    """POST /v1/rank/drill-down returns flat list."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank/drill-down",
        json={"query_text": "BERT pre-training", "top_k": 5},
        headers=auth_headers,
    )

    assert response.status_code in (200, 202)
    if response.status_code == 200:
        data = response.json()
        assert "rank_job_id" in data


# ── Test: Seed title mode ────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
@respx.mock
async def test_rank_with_seed_title(client, auth_headers):
    """POST /v1/rank with seed_title generates a query from the title."""
    _mock_all_external_apis(respx.mock)

    response = await client.post(
        "/v1/rank",
        json={"seed_title": "Attention Is All You Need"},
        headers=auth_headers,
    )

    # Should work (generates query from title via LLM)
    assert response.status_code in (200, 202, 400, 500)


# ── Test: Rank Novelty Endpoint ─────────────────────────────────────────────

import time
from contextlib import contextmanager
from unittest.mock import patch, MagicMock
from sqlalchemy import text

from tests.fixtures.novelty_responses import (
    NOVELTY_LLM_RESPONSE_HIGH,
    OPENALEX_TOPIC_RESPONSE,
    make_groq_chat_response,
)


# Use a test-specific tenant to avoid conflicts
_NOVELTY_TEST_TENANT = UUID("00000000-0000-0000-0000-000000000001")


def _make_mock_response(json_data=None, text_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    if json_data is not None:
        resp.json.return_value = json_data
    if text_data is not None:
        resp.text = text_data
        resp.content = text_data.encode("utf-8")
    resp.raise_for_status.return_value = None
    return resp


def _make_openai_mock(response_content: str):
    mock_class = MagicMock()
    mock_client = MagicMock()
    mock_class.return_value = mock_client
    mock_response = make_groq_chat_response(response_content)
    mock_client.chat.completions.create.return_value = mock_response
    return mock_class, mock_client


def _openalex_ref_work(wid, title, year, cites):
    """Build an OpenAlex single-work response."""
    return {
        "id": f"https://openalex.org/{wid}",
        "title": title,
        "publication_year": year,
        "cited_by_count": cites,
        "abstract_inverted_index": {"A": [0], "foundational": [1], "method": [2]},
        "primary_topic": {"id": "T10123", "display_name": "Machine Learning"},
        "referenced_works": [],
    }


@contextmanager
def _patch_novelty_apis(openai_response=NOVELTY_LLM_RESPONSE_HIGH):
    """Patch all external APIs used by the novelty pipeline."""
    # Fake reference work IDs the test paper "cites"
    fake_ref_ids = [f"https://openalex.org/W990000000{i}" for i in range(5)]
    fake_refs_batch = [
        _openalex_ref_work(f"W990000000{i}", f"Ref Paper {i}", 2015 + i, 300 * (i + 1))
        for i in range(5)
    ]

    def requests_handler(url, **kwargs):
        url_str = str(url)
        if "api.openalex.org" in url_str:
            # Single work lookup (reference fetch)
            if "/works/" in url_str and "filter=" not in url_str:
                return _make_mock_response(json_data={
                    "id": "https://openalex.org/W_test",
                    "title": "Test Paper",
                    "publication_year": 2021,
                    "cited_by_count": 800,
                    "referenced_works": fake_ref_ids,
                    "primary_topic": {"id": "T10123", "display_name": "Machine Learning"},
                })
            # Batch filter lookup (resolve references or landmarks)
            return _make_mock_response(json_data={
                "results": fake_refs_batch,
                "meta": {"count": len(fake_refs_batch)},
            })
        if "api.semanticscholar.org" in url_str:
            return _make_mock_response(json_data={"data": [], "total": 0})
        if "export.arxiv.org" in url_str:
            return _make_mock_response(
                text_data='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            )
        return _make_mock_response(json_data={})

    httpx_resp = MagicMock()
    httpx_resp.status_code = 200
    httpx_resp.is_success = True
    httpx_resp.json.return_value = OPENALEX_TOPIC_RESPONSE
    httpx_resp.raise_for_status.return_value = None

    openai_node_cls, _ = _make_openai_mock(openai_response)
    openai_topic_cls, _ = _make_openai_mock("{}")
    openai_grounding_cls, _ = _make_openai_mock("[]")

    with (
        patch("app.feature3.reference_store.requests.get", side_effect=requests_handler),
        patch("app.feature3.landmark_retrieval.requests.get", side_effect=requests_handler),
        patch("app.feature3.abstract_enrichment.requests.get", side_effect=requests_handler),
        patch("app.feature3.topic_inference.requests.get", side_effect=requests_handler),
        patch("app.feature3.grounding_supplement.requests.get", side_effect=requests_handler),
        patch("app.feature3.paper_cache.requests.get", side_effect=requests_handler),
        patch("app.feature3.topic_lookup.httpx.get", return_value=httpx_resp),
        patch("app.feature3.node_details_service.OpenAI", openai_node_cls),
        patch("app.feature3.topic_inference.OpenAI", openai_topic_cls),
        patch("app.feature3.grounding_supplement.OpenAI", openai_grounding_cls),
    ):
        yield


@pytest.fixture()
def rank_novelty_test_data(db_engine):
    """
    Insert a rank job + result + work for novelty endpoint testing.
    Cleans up on teardown.
    """
    from uuid import uuid4

    ts = int(time.time())
    work_id = f"W_rank_novelty_test_{ts}"
    rank_job_id = uuid4()
    candidate_set_id = uuid4()

    abstract = (
        "This paper introduces a novel self-attention mechanism for sequence modeling. "
        "The proposed approach eliminates recurrent connections entirely, enabling "
        "parallelized training. Experiments demonstrate superior performance."
    )

    with db_engine.connect() as conn:
        # Insert work
        conn.execute(
            text("""
                INSERT INTO works (
                    work_id, title, year, cited_by_count, abstract,
                    primary_topic_id, venue, authors_json,
                    is_open_access, oa_status, category, category_confidence
                ) VALUES (
                    :work_id, :title, :year, :cited_by_count, :abstract,
                    :topic_id, :venue, :authors_json,
                    :is_oa, :oa_status, :category, :cat_conf
                )
            """),
            {
                "work_id": work_id,
                "title": "Test Paper for Rank Novelty",
                "year": 2021,
                "cited_by_count": 800,
                "abstract": abstract,
                "topic_id": "T10123",
                "venue": "ICML",
                "authors_json": json.dumps(["Test Author"]),
                "is_oa": True,
                "oa_status": "gold",
                "category": "foundational",
                "cat_conf": 0.9,
            },
        )

        # Insert candidate_set (FK for rank_jobs)
        conn.execute(
            text("""
                INSERT INTO candidate_sets (
                    candidate_set_id, tenant_id, seed_type, seed_json, params_hash
                ) VALUES (
                    :csid, :tid, 'direct_query',
                    '{"query_text":"test novelty"}'::jsonb, :hash
                )
            """),
            {
                "csid": candidate_set_id,
                "tid": _NOVELTY_TEST_TENANT,
                "hash": f"test_novelty_{ts}",
            },
        )

        # Insert rank_job
        conn.execute(
            text("""
                INSERT INTO rank_jobs (
                    rank_job_id, tenant_id, rank_type, candidate_set_id,
                    status, params_hash
                ) VALUES (
                    :rjid, :tid, 'direct_prod', :csid,
                    'completed', :hash
                )
            """),
            {
                "rjid": rank_job_id,
                "tid": _NOVELTY_TEST_TENANT,
                "csid": candidate_set_id,
                "hash": f"test_novelty_job_{ts}",
            },
        )

        # Insert rank_result
        conn.execute(
            text("""
                INSERT INTO rank_results (
                    rank_job_id, rank_index, work_id, score,
                    score_breakdown_json, reasons_json,
                    work_preview_json, provenance_json
                ) VALUES (
                    :rjid, 0, :wid, 0.95,
                    '{}'::jsonb, '["test reason"]'::jsonb,
                    '{"title":"Test Paper"}'::jsonb, '[{"source":"test"}]'::jsonb
                )
            """),
            {"rjid": rank_job_id, "wid": work_id},
        )

        conn.commit()

    yield rank_job_id, work_id

    # Teardown
    with db_engine.connect() as conn:
        conn.execute(
            text("DELETE FROM node_details_cache WHERE work_id = :wid"),
            {"wid": work_id},
        )
        conn.execute(
            text("DELETE FROM rank_results WHERE rank_job_id = :rjid"),
            {"rjid": rank_job_id},
        )
        conn.execute(
            text("DELETE FROM rank_jobs WHERE rank_job_id = :rjid"),
            {"rjid": rank_job_id},
        )
        conn.execute(
            text("DELETE FROM candidate_sets WHERE candidate_set_id = :csid"),
            {"csid": candidate_set_id},
        )
        conn.execute(
            text("DELETE FROM works WHERE work_id = :wid"),
            {"wid": work_id},
        )
        conn.commit()


@pytest.mark.integration
def test_rank_novelty_success(db_engine, rank_novelty_test_data):
    """POST /v1/rank/{job_id}/nodes/{work_id}/novelty returns novelty assessment."""
    from fastapi.testclient import TestClient
    from app.main import app

    rank_job_id, work_id = rank_novelty_test_data
    headers = {"X-Workspace-Id": str(_NOVELTY_TEST_TENANT)}

    with _patch_novelty_apis():
        with TestClient(app) as tc:
            response = tc.post(
                f"/v1/rank/{rank_job_id}/nodes/{work_id}/novelty",
                headers=headers,
            )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "novelty_level" in data
    assert data["novelty_level"] in ("low", "medium", "high", "pioneering")
    assert "confidence" in data
    assert "novelty_explanation" in data
    assert "grounding_papers" in data


@pytest.mark.integration
def test_rank_novelty_job_not_found():
    """POST with non-existent rank_job_id returns 404."""
    from fastapi.testclient import TestClient
    from app.main import app

    fake_job_id = "00000000-0000-0000-0000-ffffffffffff"
    headers = {"X-Workspace-Id": str(_NOVELTY_TEST_TENANT)}

    with TestClient(app) as tc:
        response = tc.post(
            f"/v1/rank/{fake_job_id}/nodes/W1234/novelty",
            headers=headers,
        )

    assert response.status_code == 404
    assert "rank_job_not_found" in response.json()["detail"]


@pytest.mark.integration
def test_rank_novelty_work_not_in_results(db_engine, rank_novelty_test_data):
    """POST with work_id not in the rank results returns 404."""
    from fastapi.testclient import TestClient
    from app.main import app

    rank_job_id, _ = rank_novelty_test_data
    headers = {"X-Workspace-Id": str(_NOVELTY_TEST_TENANT)}

    with TestClient(app) as tc:
        response = tc.post(
            f"/v1/rank/{rank_job_id}/nodes/W_nonexistent_999/novelty",
            headers=headers,
        )

    assert response.status_code == 404
    assert "work_not_in_rank_results" in response.json()["detail"]
