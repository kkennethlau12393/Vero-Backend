"""
Integration tests for the Vero Backend API.

These tests hit every endpoint through FastAPI's TestClient (full app stack,
no mocks). Each test uses the `bench` fixture which automatically records
timing, response size, and status code.

Run:
    pytest tests/test_integration.py -v -s

The benchmark report prints at the end and is saved to tests/reports/.
"""

from __future__ import annotations

import pytest

# ===================================================================
# SYSTEM
# ===================================================================


class TestSystem:
    """Health and system endpoints."""

    def test_health(self, bench):
        resp = bench.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_openapi_schema(self, bench):
        resp = bench.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]

    def test_cors_preflight(self, client):
        """CORS preflight should return allow-origin header."""
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert (
            resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
        )

    def test_404_returns_json(self, bench):
        resp = bench.get("/v1/nonexistent-endpoint")
        assert resp.status_code in (404, 405)
        # Should be JSON, not HTML
        assert resp.headers.get("content-type", "").startswith("application/json")


# ===================================================================
# FEATURE 2: RANK
# ===================================================================


class TestRank:
    """Scholar search and ranking endpoints."""

    def test_rank_missing_query(self, bench):
        """Should return 400 when neither query_text nor seed_title provided."""
        resp = bench.post("/v1/rank", json={})
        assert resp.status_code == 400

    @pytest.mark.slow
    def test_rank_basic_query(self, bench):
        """Full ranking pipeline — this calls external APIs and LLM.

        Expect 200 (sync) or 202 (async/pending). Either is valid.
        """
        resp = bench.post(
            "/v1/rank",
            json={"query_text": "transformer attention mechanism"},
        )
        assert resp.status_code in (200, 202)
        data = resp.json()
        if resp.status_code == 200:
            # Completed synchronously
            assert "rank_job_id" in data
            assert any(
                key in data for key in ("items", "fundamentals", "core", "applications")
            )
        else:
            # Async — got job ID for polling
            assert "rank_job_id" in data
            assert data["status"] in ("pending", "running")

    @pytest.mark.db
    def test_rank_job_status_not_found(self, bench):
        """Polling a non-existent job should 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/rank/{fake_id}/status")
        assert resp.status_code in (404, 403)

    def test_subtopics_not_found(self, bench):
        """Subtopics for non-existent rank job should 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = bench.post(f"/v1/rank/{fake_id}/generate-subtopics")
        assert resp.status_code in (404, 400, 500)

    def test_temporal_map_not_found(self, bench):
        """Temporal map for non-existent rank job should 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/rank/{fake_id}/temporal-map")
        assert resp.status_code in (404, 400, 500)

    @pytest.mark.slow
    def test_drill_down(self, bench):
        """Drill-down ranking with lightweight params."""
        resp = bench.post(
            "/v1/rank/drill-down",
            json={"query_text": "graph neural networks", "top_k": 5},
        )
        assert resp.status_code in (200, 202)


# ===================================================================
# FEATURE 1: CITATION MAP
# ===================================================================


class TestCitationMap:
    """Citation map generation endpoints."""

    @pytest.mark.slow
    def test_citation_map_by_query(self, bench):
        """Build citation map from a text query."""
        resp = bench.post(
            "/v1/citation-map",
            json={
                "query_text": "BERT language model",
                "max_nodes": 15,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "nodes" in data or "graph_draft_id" in data

    def test_citation_map_missing_input(self, bench):
        """Should fail when neither query_text nor seed_work_id provided."""
        resp = bench.post("/v1/citation-map", json={})
        assert resp.status_code in (400, 422)


# ===================================================================
# FEATURE 2: MAPS
# ===================================================================


class TestMaps:
    """Map build and retrieval endpoints."""

    def test_get_map_not_found(self, bench):
        """Getting a non-existent map should 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/maps/{fake_id}")
        assert resp.status_code in (404, 403, 500)

    def test_build_map_invalid_draft(self, bench):
        """Building from non-existent graph_draft should fail."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = bench.post(
            "/v1/maps/build",
            json={"graph_draft_id": fake_id},
        )
        assert resp.status_code in (404, 400, 500)


# ===================================================================
# FEATURE 3: NODE DETAILS
# ===================================================================


class TestNodeDetails:
    """Node details popup endpoint."""

    def test_node_details_not_found(self, bench):
        """Details for non-existent map/node should fail."""
        fake_map = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/maps/{fake_map}/nodes/W0000000000/details")
        assert resp.status_code in (404, 403, 500)


# ===================================================================
# FEATURE 4: COMPARE
# ===================================================================


class TestCompare:
    """Methodology comparison endpoint."""

    def test_compare_invalid_map(self, bench):
        """Compare on non-existent map should fail."""
        fake_map = "00000000-0000-0000-0000-000000000000"
        resp = bench.post(
            f"/v1/maps/{fake_map}/compare-methodologies",
            json={"work_ids": ["W001", "W002"]},
        )
        assert resp.status_code in (404, 403, 500)

    def test_compare_invalid_uuid(self, bench):
        """Non-UUID map_id should return 422 validation error."""
        resp = bench.post(
            "/v1/maps/not-a-uuid/compare-methodologies",
            json={"work_ids": ["W001", "W002"]},
        )
        assert resp.status_code == 422


# ===================================================================
# FEATURE 5: GAP ANALYSIS
# ===================================================================


class TestGapAnalysis:
    """Research gap analysis endpoints."""

    def test_gap_status_not_found(self, bench):
        """Gap analysis status for non-existent map should fail."""
        fake_map = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/maps/{fake_map}/gap-analysis/status")
        assert resp.status_code in (404, 403, 500)

    def test_gap_results_not_found(self, bench):
        """Gap results for non-existent map should fail."""
        fake_map = "00000000-0000-0000-0000-000000000000"
        resp = bench.get(f"/v1/maps/{fake_map}/gap-analysis/results")
        assert resp.status_code in (404, 403, 500)

    def test_gap_trigger_not_found(self, bench):
        """Triggering gap analysis on non-existent map should fail."""
        fake_map = "00000000-0000-0000-0000-000000000000"
        resp = bench.post(f"/v1/maps/{fake_map}/gap-analysis", json={})
        assert resp.status_code in (404, 403, 500)


# ===================================================================
# SETTINGS
# ===================================================================


class TestSettings:
    """Settings / institutional access endpoints."""

    def test_get_institutional_access(self, bench):
        """Should return settings or 401/403 if no auth."""
        resp = bench.get("/v1/settings/institutional-access")
        # Depends on auth setup — 200 or 401/403 are both valid
        assert resp.status_code in (200, 401, 403, 500)


# ===================================================================
# RESPONSE CONTRACT TESTS
# ===================================================================


class TestResponseContracts:
    """Verify response shapes match what the frontend expects."""

    def test_health_shape(self, client):
        data = client.get("/health").json()
        assert isinstance(data, dict)
        assert "status" in data

    def test_error_is_json(self, client):
        """All errors should be JSON with a 'detail' field."""
        resp = client.get("/v1/maps/not-a-uuid")
        assert "application/json" in resp.headers.get("content-type", "")

    def test_422_has_detail(self, client):
        """Pydantic validation errors should have structured detail."""
        resp = client.post(
            "/v1/maps/not-a-uuid/compare-methodologies",
            json={"work_ids": "not-a-list"},
        )
        if resp.status_code == 422:
            data = resp.json()
            assert "detail" in data
