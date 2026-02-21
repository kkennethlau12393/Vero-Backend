"""
Integration tests for rank_api.py seed modes (seed_doi, seed_work_id, PDF upload).

Tests the full workflow: DOI/work_id/PDF → title → query → ranking.
Uses dev DB + mocked external APIs.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_rank_with_seed_doi_success():
    """seed_doi → title → query → ranking pipeline works end-to-end."""
    from app.main import app

    client = TestClient(app)

    # Mock OpenAlex DOI lookup
    mock_resp_doi = MagicMock()
    mock_resp_doi.status_code = 200
    mock_resp_doi.json.return_value = {
        "results": [{
            "id": "https://openalex.org/W2963663278",
        }]
    }
    mock_resp_doi.raise_for_status = MagicMock()

    # Mock OpenAlex work fetch
    mock_resp_work = MagicMock()
    mock_resp_work.status_code = 200
    mock_resp_work.json.return_value = {
        "id": "https://openalex.org/W2963663278",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 50000,
    }
    mock_resp_work.raise_for_status = MagicMock()

    # Mock Groq LLM for query generation
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [MagicMock()]
    mock_groq_response.choices[0].message.content = "transformer architecture attention mechanisms"

    def mock_requests_get(url, **kwargs):
        if "filter=doi:" in url or (kwargs.get("params") and "filter" in kwargs.get("params", {})):
            return mock_resp_doi
        else:
            return mock_resp_work

    with patch("requests.get", side_effect=mock_requests_get), \
         patch("groq.Groq") as mock_groq_class:

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_groq_client

        response = client.post(
            "/v1/rank",
            json={"seed_doi": "10.48550/arXiv.1706.03762"},
            headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
        )

    # Should succeed (202 or 200)
    assert response.status_code in [200, 202], f"Unexpected status: {response.status_code}, body: {response.text}"

    result = response.json()

    # If async (202), should have rank_job_id
    if response.status_code == 202:
        assert "rank_job_id" in result
        assert result["status"] in ["pending", "running"]
    else:
        # If sync (200), should have ranked items
        assert "items" in result or "ranked_items" in result


@pytest.mark.integration
def test_rank_with_seed_work_id_success():
    """seed_work_id → title → query → ranking pipeline works end-to-end."""
    from app.main import app

    client = TestClient(app)

    # Mock OpenAlex work fetch
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "https://openalex.org/W2963663278",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "cited_by_count": 50000,
    }
    mock_resp.raise_for_status = MagicMock()

    # Mock Groq LLM
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [MagicMock()]
    mock_groq_response.choices[0].message.content = "transformer architecture attention mechanisms"

    with patch("requests.get", return_value=mock_resp), \
         patch("groq.Groq") as mock_groq_class:

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_groq_client

        response = client.post(
            "/v1/rank",
            json={"seed_work_id": "W2963663278"},
            headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
        )

    assert response.status_code in [200, 202], f"Unexpected status: {response.status_code}, body: {response.text}"

    result = response.json()
    if response.status_code == 202:
        assert "rank_job_id" in result


@pytest.mark.integration
def test_rank_with_invalid_doi_fails():
    """Invalid DOI raises 400 error."""
    from app.main import app

    client = TestClient(app)

    # Mock 404 responses from both APIs
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_resp_404.json.return_value = {"results": []}
    mock_resp_404.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp_404):
        response = client.post(
            "/v1/rank",
            json={"seed_doi": "10.9999/does-not-exist"},
            headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
        )

    assert response.status_code == 400
    assert "not found" in response.text.lower()


@pytest.mark.integration
def test_rank_with_multiple_inputs_fails():
    """Providing multiple input modes raises 400 error."""
    from app.main import app

    client = TestClient(app)

    response = client.post(
        "/v1/rank",
        json={
            "seed_doi": "10.48550/arXiv.1706.03762",
            "seed_work_id": "W2963663278",  # Can't provide both
        },
        headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
    )

    assert response.status_code == 400
    assert "only ONE" in response.text or "Provide only ONE" in response.text


@pytest.mark.integration
def test_rank_with_no_input_fails():
    """Providing no input mode raises 400 error."""
    from app.main import app

    client = TestClient(app)

    response = client.post(
        "/v1/rank",
        json={},  # No query_text, seed_doi, seed_work_id, or seed_title
        headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
    )

    assert response.status_code == 400
    assert "Must provide ONE" in response.text


@pytest.mark.integration
def test_rank_backward_compat_seed_title():
    """seed_title still works for backward compatibility (deprecated)."""
    from app.main import app

    client = TestClient(app)

    # Mock Groq LLM
    mock_groq_response = MagicMock()
    mock_groq_response.choices = [MagicMock()]
    mock_groq_response.choices[0].message.content = "transformer architecture attention mechanisms"

    with patch("groq.Groq") as mock_groq_class:
        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_groq_client

        response = client.post(
            "/v1/rank",
            json={"seed_title": "Attention Is All You Need"},
            headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
        )

    # Should still work (deprecated but functional)
    assert response.status_code in [200, 202], f"Unexpected status: {response.status_code}, body: {response.text}"
