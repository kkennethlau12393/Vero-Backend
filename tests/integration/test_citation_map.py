"""
Integration tests for Feature 1 Citation Map service.

Uses dev DB. Mocks ALL external APIs using unittest.mock.patch on requests.get
and Groq client since the service uses the `requests` library (not httpx).
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, PropertyMock
from uuid import UUID, uuid4

import pytest

from app.feature1.schemas import CitationMapRequest
from app.feature1.citation_map_service import (
    build_citation_map,
    get_citation_map,
    list_citation_maps,
    select_seed_from_query,
    fetch_citing_papers,
    fetch_references,
    fetch_seed_paper_details,
    _search_openalex,
    _search_openalex_highly_cited,
    _search_semantic_scholar,
    _search_arxiv,
    _fetch_citing_papers_s2,
    _fetch_references_s2,
    _get_doi_for_work,
    _batch_lookup_openalex_by_dois,
    _score_seed_candidates,
    _expand_search_queries,
    _backfill_title_from_s2,
    _search_openalex_by_title,
)
from tests.fixtures.citation_map_responses import (
    SEED_WORK_RESPONSE,
    CITING_PAPERS_RESPONSE,
    REFERENCES_WORK_RESPONSE,
    REFERENCES_DETAILS_RESPONSE,
    S2_SEARCH_RESPONSE,
    S2_CITATIONS_RESPONSE,
    S2_REFERENCES_RESPONSE,
    S2_EMPTY_RESPONSE,
    ARXIV_SEARCH_RESPONSE,
    ARXIV_EMPTY_RESPONSE,
    SEED_SCORING_RESPONSE,
    make_openalex_search_response,
    make_openalex_work,
    make_s2_paper,
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


REQUESTS_GET = "app.feature1.citation_map_service.requests.get"
GROQ_CLASS = "app.feature1.citation_map_service.Groq"


# ============================================================================
# OpenAlex Search Tests
# ============================================================================

@pytest.mark.integration
class TestSearchOpenAlex:
    @patch(REQUESTS_GET)
    def test_basic_search(self, mock_get):
        response = make_openalex_search_response()
        mock_get.return_value = _make_mock_response(json_data=response)
        results = _search_openalex("machine learning")
        assert len(results) == 3
        assert all(r["work_id"].startswith("W") for r in results)
        assert all(r["source"] == "openalex" for r in results)

    def test_empty_query(self):
        assert _search_openalex("") == []

    @patch(REQUESTS_GET)
    def test_empty_results(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data={"results": []})
        assert _search_openalex("nonexistent") == []

    @patch(REQUESTS_GET)
    def test_http_error(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=500)
        assert _search_openalex("test") == []

    @patch(REQUESTS_GET)
    def test_highly_cited_search(self, mock_get):
        response = make_openalex_search_response()
        mock_get.return_value = _make_mock_response(json_data=response)
        results = _search_openalex_highly_cited("transformers")
        assert len(results) > 0
        assert all(r["source"] == "openalex_highly_cited" for r in results)


# ============================================================================
# Semantic Scholar Search Tests
# ============================================================================

@pytest.mark.integration
class TestSearchSemanticScholar:
    @patch(REQUESTS_GET)
    def test_basic_search(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=S2_SEARCH_RESPONSE)
        results = _search_semantic_scholar("transformers", k=10)
        assert len(results) > 0

    def test_empty_query(self):
        assert _search_semantic_scholar("") == []

    @patch(REQUESTS_GET)
    def test_400_falls_back_to_regular(self, mock_get):
        """400 on bulk should fall back to regular endpoint."""
        # First call (bulk): 400, second call (regular): success
        bulk_resp = _make_mock_response(status_code=400)
        regular_resp = _make_mock_response(json_data=S2_SEARCH_RESPONSE)
        mock_get.side_effect = [bulk_resp, regular_resp]
        results = _search_semantic_scholar("broad query", k=10)
        assert isinstance(results, list)

    @patch(REQUESTS_GET)
    def test_openalex_id_used_when_available(self, mock_get):
        data = {
            "data": [make_s2_paper("s1", "Test", 2020, 100, openalex_id="W12345")],
            "total": 1,
        }
        mock_get.return_value = _make_mock_response(json_data=data)
        results = _search_semantic_scholar("test", k=10)
        assert results[0]["work_id"] == "W12345"

    @patch(REQUESTS_GET)
    def test_s2_prefix_when_no_openalex(self, mock_get):
        data = {
            "data": [make_s2_paper("s1", "Test", 2020, 100)],
            "total": 1,
        }
        mock_get.return_value = _make_mock_response(json_data=data)
        results = _search_semantic_scholar("test", k=10)
        assert results[0]["work_id"] == "S2:s1"


# ============================================================================
# ArXiv Search Tests
# ============================================================================

@pytest.mark.integration
class TestSearchArxiv:
    @patch(REQUESTS_GET)
    def test_basic_search(self, mock_get):
        mock_get.return_value = _make_mock_response(text_data=ARXIV_SEARCH_RESPONSE)
        results = _search_arxiv("attention transformers")
        assert len(results) == 2
        assert all(r["work_id"].startswith("AX:") for r in results)
        assert all(r["cited_by_count"] == 0 for r in results)

    def test_empty_query(self):
        assert _search_arxiv("") == []

    @patch(REQUESTS_GET)
    def test_empty_results(self, mock_get):
        mock_get.return_value = _make_mock_response(text_data=ARXIV_EMPTY_RESPONSE)
        assert _search_arxiv("nothing") == []

    @patch(REQUESTS_GET)
    def test_http_error(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=503)
        assert _search_arxiv("test") == []


# ============================================================================
# Seed Paper Details
# ============================================================================

@pytest.mark.integration
class TestFetchSeedPaperDetails:
    @patch(REQUESTS_GET)
    def test_basic_fetch(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=SEED_WORK_RESPONSE)
        result = fetch_seed_paper_details("W999")
        assert result is not None
        assert result["work_id"] == "W999"
        assert result["title"] == "Attention Is All You Need"
        assert result["year"] == 2017

    def test_empty_work_id(self):
        assert fetch_seed_paper_details("") is None
        assert fetch_seed_paper_details(None) is None

    @patch(REQUESTS_GET)
    def test_http_error(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=404)
        assert fetch_seed_paper_details("W999") is None

    @patch(REQUESTS_GET)
    def test_abstract_extracted(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=SEED_WORK_RESPONSE)
        result = fetch_seed_paper_details("W999")
        assert result["abstract"] is not None
        assert "propose" in result["abstract"]


# ============================================================================
# Citing Papers & References
# ============================================================================

@pytest.mark.integration
class TestFetchCitingPapers:
    @patch(REQUESTS_GET)
    def test_basic_fetch(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=CITING_PAPERS_RESPONSE)
        results = fetch_citing_papers("W999")
        assert len(results) > 0

    def test_empty_work_id(self):
        assert fetch_citing_papers("") == []

    @patch(REQUESTS_GET)
    def test_filters_suspicious_citations(self, mock_get):
        """Papers with suspiciously corrupted citation data should be filtered."""
        bad_paper = make_openalex_work("W400", "Health Supplement Guide", 2022, 50000)
        response = make_openalex_search_response([bad_paper])
        mock_get.return_value = _make_mock_response(json_data=response)
        results = fetch_citing_papers("W999")
        ids = {r["work_id"] for r in results}
        assert "W400" not in ids

    @patch(REQUESTS_GET)
    def test_error_returns_empty(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=500)
        assert fetch_citing_papers("W999") == []


@pytest.mark.integration
class TestFetchReferences:
    @patch(REQUESTS_GET)
    def test_basic_fetch(self, mock_get):
        # First call: get work with referenced_works; second: details
        mock_get.side_effect = [
            _make_mock_response(json_data=REFERENCES_WORK_RESPONSE),
            _make_mock_response(json_data=REFERENCES_DETAILS_RESPONSE),
        ]
        results = fetch_references("W999")
        assert len(results) > 0

    def test_empty_work_id(self):
        assert fetch_references("") == []

    @patch(REQUESTS_GET)
    def test_sorted_by_citations(self, mock_get):
        mock_get.side_effect = [
            _make_mock_response(json_data=REFERENCES_WORK_RESPONSE),
            _make_mock_response(json_data=REFERENCES_DETAILS_RESPONSE),
        ]
        results = fetch_references("W999")
        if len(results) >= 2:
            cites = [r["cited_by_count"] for r in results]
            assert cites == sorted(cites, reverse=True)

    @patch(REQUESTS_GET)
    def test_error_returns_empty(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=500)
        assert fetch_references("W999") == []


# ============================================================================
# S2 Citation/Reference Fetching
# ============================================================================

@pytest.mark.integration
class TestS2Citations:
    @patch(REQUESTS_GET)
    def test_fetch_citing_papers(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=S2_CITATIONS_RESPONSE)
        results = _fetch_citing_papers_s2("10.1234/test")
        assert len(results) >= 2

    @patch(REQUESTS_GET)
    def test_fetch_references(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=S2_REFERENCES_RESPONSE)
        results = _fetch_references_s2("10.1234/test")
        assert len(results) == 2

    def test_empty_doi(self):
        assert _fetch_citing_papers_s2("") == []
        assert _fetch_references_s2("") == []

    @patch(REQUESTS_GET)
    def test_404_returns_empty(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=404)
        assert _fetch_citing_papers_s2("10.1234/missing") == []


# ============================================================================
# DOI Bridge
# ============================================================================

@pytest.mark.integration
class TestDOIBridge:
    @patch(REQUESTS_GET)
    def test_get_doi_for_work(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data={"doi": "https://doi.org/10.1234/test"})
        result = _get_doi_for_work("W999")
        assert result == "10.1234/test"

    def test_get_doi_non_w_id(self):
        assert _get_doi_for_work("S2:abc") is None
        assert _get_doi_for_work("") is None

    @patch(REQUESTS_GET)
    def test_batch_lookup_openalex_by_dois(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data={
            "results": [
                {"id": "https://openalex.org/W500", "doi": "https://doi.org/10.1234/a"},
                {"id": "https://openalex.org/W501", "doi": "https://doi.org/10.1234/b"},
            ]
        })
        result = _batch_lookup_openalex_by_dois(["10.1234/a", "10.1234/b"])
        assert result["10.1234/a"] == "W500"
        assert result["10.1234/b"] == "W501"

    def test_batch_lookup_empty(self):
        assert _batch_lookup_openalex_by_dois([]) == {}


# ============================================================================
# LLM Seed Scoring
# ============================================================================

@pytest.mark.integration
class TestScoreSeedCandidates:
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_basic_scoring(self):
        candidates = [
            {"work_id": "W999", "title": "Attention Is All You Need"},
            {"work_id": "W100", "title": "Some Paper"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"W999": "ESSENTIAL", "W100": "HIGH"}'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            scores = _score_seed_candidates("transformers", candidates)

        assert scores["W999"] == 0.95
        assert scores["W100"] == 0.75

    def test_no_api_key_returns_empty(self):
        import os
        old = os.environ.pop("GROQ_API_KEY", None)
        try:
            result = _score_seed_candidates("test", [{"work_id": "W1", "title": "T"}])
            assert result == {}
        finally:
            if old:
                os.environ["GROQ_API_KEY"] = old

    def test_empty_candidates(self):
        assert _score_seed_candidates("test", []) == {}

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_unknown_tier_defaults_to_none(self):
        candidates = [{"work_id": "W1", "title": "Paper"}]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"W1": "INVALID_TIER"}'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            scores = _score_seed_candidates("test", candidates)

        assert scores["W1"] == 0.05  # NONE tier

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_llm_exception_returns_empty(self):
        candidates = [{"work_id": "W1", "title": "Paper"}]
        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API error")
            scores = _score_seed_candidates("test", candidates)

        assert scores == {}


# ============================================================================
# LLM Query Expansion
# ============================================================================

@pytest.mark.integration
class TestExpandSearchQueries:
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_basic_expansion(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"title": "Attention Is All You Need", "year": 2017, "citations": 90000}]'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            result = _expand_search_queries("transformer attention")

        assert len(result) == 1
        assert result[0]["title"] == "Attention Is All You Need"

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_garbage_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I cannot help with that."

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            result = _expand_search_queries("test")

        assert result == []

    def test_no_api_key(self):
        import os
        old = os.environ.pop("GROQ_API_KEY", None)
        try:
            result = _expand_search_queries("test")
            assert result == []
        finally:
            if old:
                os.environ["GROQ_API_KEY"] = old

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_max_3_results(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            {"title": f"Paper {i}", "year": 2020, "citations": 1000}
            for i in range(5)
        ])

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            result = _expand_search_queries("test")

        assert len(result) <= 3


# ============================================================================
# select_seed_from_query
# ============================================================================

@pytest.mark.integration
class TestSelectSeedFromQuery:
    @patch(REQUESTS_GET)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_finds_seed_with_llm_validation(self, mock_get):
        """Full pipeline: search → LLM score → pick best."""
        oa_response = make_openalex_search_response([
            make_openalex_work("W999", "Attention Is All You Need", 2017, 90000),
            make_openalex_work("W100", "Some Other Paper", 2020, 500),
        ])
        mock_get.return_value = _make_mock_response(json_data=oa_response)

        mock_scoring = MagicMock()
        mock_scoring.choices = [MagicMock()]
        mock_scoring.choices[0].message.content = '{"W999": "ESSENTIAL", "W100": "LOW"}'

        mock_expansion = MagicMock()
        mock_expansion.choices = [MagicMock()]
        mock_expansion.choices[0].message.content = '[]'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.side_effect = [mock_expansion, mock_scoring]
            seed_id, info = select_seed_from_query("transformer attention mechanism")

        assert seed_id == "W999"
        assert info["selection_strategy"] == "llm_validated_highest_cited"

    def test_empty_query(self):
        seed_id, info = select_seed_from_query("")
        assert seed_id is None
        assert info["selection_strategy"] == "none"

    @patch(REQUESTS_GET)
    def test_no_results_found(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data={"results": [], "meta": {"count": 0}, "data": []})

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '[]'
            mock_client.chat.completions.create.return_value = mock_response

            seed_id, info = select_seed_from_query("completely nonexistent topic xyz")

        assert seed_id is None

    @patch(REQUESTS_GET)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_fallback_when_no_llm_high_papers(self, mock_get):
        """When no papers pass LLM HIGH threshold, fall back to best scored."""
        oa_response = make_openalex_search_response([
            make_openalex_work("W100", "Tangential Paper", 2020, 5000),
        ])
        mock_get.return_value = _make_mock_response(json_data=oa_response)

        mock_scoring = MagicMock()
        mock_scoring.choices = [MagicMock()]
        mock_scoring.choices[0].message.content = '{"W100": "MEDIUM"}'

        mock_expansion = MagicMock()
        mock_expansion.choices = [MagicMock()]
        mock_expansion.choices[0].message.content = '[]'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.side_effect = [mock_expansion, mock_scoring]
            seed_id, info = select_seed_from_query("tangential topic")

        assert seed_id == "W100"
        assert "no papers reached HIGH" in info.get("selection_reason", "")

    @patch(REQUESTS_GET)
    def test_highest_cited_fallback_no_groq(self, mock_get):
        """When GROQ_API_KEY is missing, fall back to highest cited."""
        oa_response = make_openalex_search_response([
            make_openalex_work("W100", "Paper A", 2020, 5000),
            make_openalex_work("W101", "Paper B", 2020, 10000),
        ])
        mock_get.return_value = _make_mock_response(json_data=oa_response)

        import os
        old = os.environ.pop("GROQ_API_KEY", None)
        try:
            seed_id, info = select_seed_from_query("some topic")
        finally:
            if old:
                os.environ["GROQ_API_KEY"] = old

        assert seed_id == "W101"  # highest cited
        assert info["selection_strategy"] == "highest_cited_fallback"


# ============================================================================
# build_citation_map (full pipeline)
# ============================================================================

@pytest.mark.integration
class TestBuildCitationMap:
    @patch(REQUESTS_GET)
    def test_seed_mode_1hop(self, mock_get, db_engine):
        """Direct seed mode with 1-hop expansion."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=5,
            references_limit=5,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.seed_info.seed_work_id == "W999"
        assert response.seed_info.selection_strategy == "direct"
        assert len(response.nodes) > 0
        assert response.stats.total_nodes > 0
        seed_nodes = [n for n in response.nodes if n.is_seed]
        assert len(seed_nodes) == 1

    @patch(REQUESTS_GET)
    def test_seed_mode_multihop(self, mock_get, db_engine):
        """Direct seed mode with multi-hop expansion."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            total_nodes=10,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.seed_info.seed_work_id == "W999"
        assert len(response.nodes) > 0

    @patch(REQUESTS_GET)
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_query_mode(self, mock_get, db_engine):
        """NL query mode: find seed then expand."""
        oa_response = make_openalex_search_response([
            make_openalex_work("W999", "Attention Is All You Need", 2017, 90000),
        ])

        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            if "api.openalex.org" in url_str:
                return _make_mock_response(json_data=oa_response)
            if "semanticscholar" in url_str:
                return _make_mock_response(json_data=S2_EMPTY_RESPONSE)
            if "arxiv" in url_str:
                return _make_mock_response(text_data=ARXIV_EMPTY_RESPONSE)
            return _make_mock_response(json_data={"results": []})

        mock_get.side_effect = route

        mock_scoring = MagicMock()
        mock_scoring.choices = [MagicMock()]
        mock_scoring.choices[0].message.content = '{"W999": "ESSENTIAL"}'

        mock_expansion = MagicMock()
        mock_expansion.choices = [MagicMock()]
        mock_expansion.choices[0].message.content = '[]'

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_client.chat.completions.create.side_effect = [mock_expansion, mock_scoring]

            request = CitationMapRequest(
                query_text="transformer attention",
                citing_limit=5,
                references_limit=5,
                create_graph_draft=False,
            )
            tenant_id = UUID("00000000-0000-0000-0000-000000000001")
            response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.seed_info.seed_work_id != ""
        assert len(response.nodes) >= 1

    @patch(REQUESTS_GET)
    def test_query_mode_no_results(self, mock_get, db_engine):
        """NL query with no results returns empty response."""
        mock_get.return_value = _make_mock_response(json_data={"results": [], "data": [], "meta": {"count": 0}})

        with patch(GROQ_CLASS) as MockGroq:
            mock_client = MagicMock()
            MockGroq.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '[]'
            mock_client.chat.completions.create.return_value = mock_response

            request = CitationMapRequest(
                query_text="completely nonexistent xyz topic",
                create_graph_draft=False,
            )
            tenant_id = UUID("00000000-0000-0000-0000-000000000001")
            response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert len(response.nodes) == 0
        assert len(response.edges) == 0

    def test_missing_both_seed_and_query(self, db_engine):
        """Should raise ValueError if neither seed_work_id nor query_text provided."""
        request = CitationMapRequest(create_graph_draft=False)
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        with pytest.raises(ValueError, match="Either seed_work_id or query_text"):
            build_citation_map(db_engine, tenant_id=tenant_id, request=request)

    @patch(REQUESTS_GET)
    def test_graph_draft_creation(self, mock_get, db_engine):
        """Test that graph_draft is created in DB when requested."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=True,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.graph_draft_id is not None
        assert isinstance(response.graph_draft_id, UUID)

    @patch(REQUESTS_GET)
    def test_stats_populated(self, mock_get, db_engine):
        """Verify stats are populated correctly."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=5,
            references_limit=5,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.stats.total_nodes == len(response.nodes)
        assert response.stats.edges_count == len(response.edges)


# ============================================================================
# Title Backfill
# ============================================================================

@pytest.mark.integration
class TestBackfillTitle:
    @patch(REQUESTS_GET)
    def test_backfill_from_s2(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data={"title": "Backfilled Title"})
        result = _backfill_title_from_s2("10.1234/test")
        assert result == "Backfilled Title"

    def test_backfill_empty_doi(self):
        assert _backfill_title_from_s2("") is None
        assert _backfill_title_from_s2(None) is None

    @patch(REQUESTS_GET)
    def test_backfill_s2_error(self, mock_get):
        mock_get.return_value = _make_mock_response(status_code=500)
        assert _backfill_title_from_s2("10.1234/fail") is None


# ============================================================================
# OpenAlex Title Search
# ============================================================================

@pytest.mark.integration
class TestSearchOpenAlexByTitle:
    @patch(REQUESTS_GET)
    def test_basic_title_search(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=make_openalex_search_response([
            make_openalex_work("W500", "Attention Is All You Need", 2017, 90000),
        ]))
        results = _search_openalex_by_title("Attention Is All You Need")
        assert len(results) > 0

    def test_empty_query(self):
        assert _search_openalex_by_title("") == []

    @patch(REQUESTS_GET)
    def test_with_year_filter(self, mock_get):
        mock_get.return_value = _make_mock_response(json_data=make_openalex_search_response([
            make_openalex_work("W500", "Test Paper", 2020, 1000),
        ]))
        results = _search_openalex_by_title("Test Paper", year=2020)
        assert isinstance(results, list)

    @patch(REQUESTS_GET)
    def test_deduplicates_results(self, mock_get):
        """Same work_id from multiple searches should be deduped."""
        mock_get.return_value = _make_mock_response(json_data=make_openalex_search_response([
            make_openalex_work("W500", "Test", 2020, 1000),
            make_openalex_work("W500", "Test", 2020, 1000),  # duplicate
        ]))
        results = _search_openalex_by_title("Test", k=10)
        ids = [r["work_id"] for r in results]
        assert len(ids) == len(set(ids))


# ============================================================================
# Citation Map Persistence & Retrieval
# ============================================================================

def _citation_maps_table_exists(engine) -> bool:
    """Check if citation_maps table exists in the DB."""
    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        row = conn.execute(sa_text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'citation_maps')"
        )).scalar()
    return bool(row)


@pytest.mark.integration
class TestCitationMapPersistence:
    """Test persisting, retrieving, and listing citation maps."""

    @pytest.fixture(autouse=True)
    def _require_table(self, db_engine):
        if not _citation_maps_table_exists(db_engine):
            pytest.skip("citation_maps table not yet created — run migration first")

    @patch(REQUESTS_GET)
    def test_build_returns_citation_map_id(self, mock_get, db_engine):
        """build_citation_map should persist and return a citation_map_id."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        response = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        assert response.citation_map_id is not None
        assert isinstance(response.citation_map_id, UUID)

    @patch(REQUESTS_GET)
    def test_get_citation_map_roundtrip(self, mock_get, db_engine):
        """Build a map, then GET it by ID — response should match."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        built = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        # Retrieve it
        retrieved = get_citation_map(
            db_engine, tenant_id=tenant_id, citation_map_id=built.citation_map_id,
        )

        assert retrieved.citation_map_id == built.citation_map_id
        assert retrieved.seed_info.seed_work_id == built.seed_info.seed_work_id
        assert len(retrieved.nodes) == len(built.nodes)
        assert len(retrieved.edges) == len(built.edges)
        assert retrieved.stats.total_nodes == built.stats.total_nodes

    @patch(REQUESTS_GET)
    def test_get_citation_map_wrong_tenant(self, mock_get, db_engine):
        """GET with a different tenant_id should raise not found."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=False,
        )
        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        built = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        other_tenant = UUID("00000000-0000-0000-0000-000000000099")
        with pytest.raises(ValueError, match="citation_map_not_found"):
            get_citation_map(db_engine, tenant_id=other_tenant, citation_map_id=built.citation_map_id)

    def test_get_citation_map_nonexistent(self, db_engine):
        """GET with a random ID should raise not found."""
        with pytest.raises(ValueError, match="citation_map_not_found"):
            get_citation_map(
                db_engine,
                tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
                citation_map_id=uuid4(),
            )

    @patch(REQUESTS_GET)
    def test_list_citation_maps(self, mock_get, db_engine):
        """list_citation_maps should include recently built maps."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=False,
        )
        built = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        maps = list_citation_maps(db_engine, tenant_id=tenant_id)
        assert len(maps) > 0
        ids = [m["citation_map_id"] for m in maps]
        assert built.citation_map_id in ids

        # Check fields present
        first = maps[0]
        assert "seed_work_id" in first
        assert "created_at" in first

    @patch(REQUESTS_GET)
    def test_list_citation_maps_tenant_isolation(self, mock_get, db_engine):
        """list_citation_maps should only return maps for the given tenant."""
        def route(url, **kwargs):
            url_str = str(url)
            if "/works/W999" in url_str:
                return _make_mock_response(json_data=SEED_WORK_RESPONSE)
            return _make_mock_response(json_data=CITING_PAPERS_RESPONSE)

        mock_get.side_effect = route

        tenant_id = UUID("00000000-0000-0000-0000-000000000001")
        request = CitationMapRequest(
            seed_work_id="W999",
            citing_limit=3,
            references_limit=3,
            create_graph_draft=False,
        )
        built = build_citation_map(db_engine, tenant_id=tenant_id, request=request)

        other_tenant = UUID("00000000-0000-0000-0000-000000000099")
        other_maps = list_citation_maps(db_engine, tenant_id=other_tenant)
        other_ids = [m["citation_map_id"] for m in other_maps]
        assert built.citation_map_id not in other_ids
