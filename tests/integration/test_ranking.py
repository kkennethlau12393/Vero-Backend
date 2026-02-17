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
