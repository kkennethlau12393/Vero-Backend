"""
Unit tests for Feature 1 Citation Map — pure logic functions.

NO DB, NO network. Tests pure functions only.
"""
from __future__ import annotations

import pytest

from unittest.mock import patch, MagicMock
import time

from app.feature1.citation_map_service import (
    _stratified_sample,
    _is_citation_count_suspicious,
    _extract_abstract,
    _assemble_citation_graph,
    _assemble_multihop_graph,
    _s2_throttle,
    _s2_get_with_retry,
    _oa_throttle,
    _arxiv_throttle,
    S2_MIN_REQUEST_INTERVAL,
    OA_MIN_REQUEST_INTERVAL,
    ARXIV_MIN_REQUEST_INTERVAL,
)
from app.feature1.schemas import CitationNode, CitationEdge
from tests.fixtures.citation_map_responses import (
    make_paper_dict,
    make_papers_sorted_by_cites,
)


# ============================================================================
# _stratified_sample
# ============================================================================

class TestStratifiedSample:
    @pytest.mark.unit
    def test_empty_input(self):
        assert _stratified_sample([], 10) == []

    @pytest.mark.unit
    def test_zero_limit(self):
        papers = make_papers_sorted_by_cites(10)
        assert _stratified_sample(papers, 0) == []

    @pytest.mark.unit
    def test_fewer_papers_than_limit(self):
        papers = make_papers_sorted_by_cites(5)
        result = _stratified_sample(papers, 20)
        assert len(result) == 5

    @pytest.mark.unit
    def test_exact_limit(self):
        papers = make_papers_sorted_by_cites(10)
        result = _stratified_sample(papers, 10)
        assert len(result) == 10

    @pytest.mark.unit
    def test_stratified_produces_correct_count(self):
        papers = make_papers_sorted_by_cites(30)
        result = _stratified_sample(papers, 10)
        assert len(result) == 10

    @pytest.mark.unit
    def test_includes_high_cited_papers(self):
        papers = make_papers_sorted_by_cites(30, base_cites=30000)
        result = _stratified_sample(papers, 10)
        # The first paper (highest cited) should be included
        result_ids = {p["work_id"] for p in result}
        assert papers[0]["work_id"] in result_ids

    @pytest.mark.unit
    def test_includes_low_cited_papers(self):
        """Stratified sample should include some low-cited (emerging) papers."""
        papers = make_papers_sorted_by_cites(30, base_cites=30000)
        result = _stratified_sample(papers, 10)
        result_ids = {p["work_id"] for p in result}
        # At least one paper from the bottom third should be included
        bottom_third = papers[20:]
        bottom_in_result = [p for p in bottom_third if p["work_id"] in result_ids]
        assert len(bottom_in_result) >= 1

    @pytest.mark.unit
    def test_custom_ratios(self):
        papers = make_papers_sorted_by_cites(30)
        result = _stratified_sample(papers, 10, high_ratio=0.8, mid_ratio=0.1, low_ratio=0.1)
        assert len(result) == 10

    @pytest.mark.unit
    def test_single_paper(self):
        papers = [make_paper_dict(work_id="W1", cited_by_count=100)]
        result = _stratified_sample(papers, 5)
        assert len(result) == 1

    @pytest.mark.unit
    def test_two_papers_limit_one(self):
        papers = [
            make_paper_dict(work_id="W1", cited_by_count=1000),
            make_paper_dict(work_id="W2", cited_by_count=10),
        ]
        result = _stratified_sample(papers, 1)
        assert len(result) == 1

    @pytest.mark.unit
    def test_preserves_paper_data(self):
        papers = make_papers_sorted_by_cites(20)
        result = _stratified_sample(papers, 5)
        for p in result:
            assert "work_id" in p
            assert "title" in p
            assert "cited_by_count" in p

    @pytest.mark.unit
    def test_no_duplicates(self):
        papers = make_papers_sorted_by_cites(30)
        result = _stratified_sample(papers, 15)
        ids = [p["work_id"] for p in result]
        assert len(ids) == len(set(ids))

    @pytest.mark.unit
    def test_limit_larger_than_input_returns_all(self):
        papers = make_papers_sorted_by_cites(3)
        result = _stratified_sample(papers, 100)
        assert len(result) == 3


# ============================================================================
# _is_citation_count_suspicious
# ============================================================================

class TestIsCitationCountSuspicious:
    @pytest.mark.unit
    def test_low_citations_not_suspicious(self):
        assert _is_citation_count_suspicious(100, 2020, "Normal Paper") is False

    @pytest.mark.unit
    def test_below_10k_never_suspicious(self):
        assert _is_citation_count_suspicious(9999, 2024, "Any Title") is False

    @pytest.mark.unit
    def test_recent_paper_mega_citations(self):
        """Papers from 2018+ with >30K citations are suspicious."""
        assert _is_citation_count_suspicious(50000, 2020, "Some Paper") is True

    @pytest.mark.unit
    def test_old_paper_mega_citations_ok(self):
        """Old papers with many citations are NOT suspicious (had time to accumulate)."""
        assert _is_citation_count_suspicious(50000, 2005, "Classic Paper") is False

    @pytest.mark.unit
    def test_health_supplement_title(self):
        assert _is_citation_count_suspicious(15000, 2010, "Best Health Supplement Review") is True

    @pytest.mark.unit
    def test_historiae_title(self):
        assert _is_citation_count_suspicious(12000, 2000, "Naturalis HISTORIAE") is True

    @pytest.mark.unit
    def test_trustworthy_title(self):
        assert _is_citation_count_suspicious(20000, 2015, "Trustworthy AI Systems") is True

    @pytest.mark.unit
    def test_dynamic_generation_title(self):
        assert _is_citation_count_suspicious(11000, 2010, "Dynamic Generation of Content") is True

    @pytest.mark.unit
    def test_none_year_not_suspicious(self):
        """If year is None, the year-based check doesn't trigger."""
        assert _is_citation_count_suspicious(50000, None, "Normal Paper") is False

    @pytest.mark.unit
    def test_none_title_not_suspicious(self):
        """If title is None, the title-based check doesn't trigger."""
        assert _is_citation_count_suspicious(50000, 2000, None) is False

    @pytest.mark.unit
    def test_zero_citations(self):
        assert _is_citation_count_suspicious(0, 2020, "Any") is False

    @pytest.mark.unit
    def test_none_citations(self):
        # 0/None treated as low
        assert _is_citation_count_suspicious(0, 2024, "health supplement") is False

    @pytest.mark.unit
    def test_exactly_30k_recent_not_suspicious(self):
        """30K exactly at 2018 is NOT suspicious (check is >30K)."""
        assert _is_citation_count_suspicious(30000, 2018, "Normal") is False

    @pytest.mark.unit
    def test_30001_recent_is_suspicious(self):
        assert _is_citation_count_suspicious(30001, 2018, "Normal") is True

    @pytest.mark.unit
    def test_case_insensitive_title_match(self):
        assert _is_citation_count_suspicious(15000, 2010, "HEALTH SUPPLEMENT Guide") is True

    @pytest.mark.unit
    def test_year_2017_mega_cites_ok(self):
        """Year 2017 is before 2018 threshold, so mega citations are OK."""
        assert _is_citation_count_suspicious(50000, 2017, "Normal Paper") is False


# ============================================================================
# _extract_abstract
# ============================================================================

class TestExtractAbstract:
    @pytest.mark.unit
    def test_basic_inverted_index(self):
        work = {"abstract_inverted_index": {"Hello": [0], "world": [1]}}
        assert _extract_abstract(work) == "Hello world"

    @pytest.mark.unit
    def test_out_of_order_positions(self):
        work = {"abstract_inverted_index": {"world": [1], "Hello": [0], "beautiful": [2]}}
        assert _extract_abstract(work) == "Hello world beautiful"

    @pytest.mark.unit
    def test_word_at_multiple_positions(self):
        work = {"abstract_inverted_index": {"the": [0, 3], "cat": [1], "sat": [2], "mat": [4]}}
        assert _extract_abstract(work) == "the cat sat the mat"

    @pytest.mark.unit
    def test_none_inverted_index(self):
        assert _extract_abstract({"abstract_inverted_index": None}) is None

    @pytest.mark.unit
    def test_missing_key(self):
        assert _extract_abstract({}) is None

    @pytest.mark.unit
    def test_empty_inverted_index(self):
        work = {"abstract_inverted_index": {}}
        result = _extract_abstract(work)
        # Empty dict is falsy → function returns None
        assert result is None

    @pytest.mark.unit
    def test_single_word(self):
        work = {"abstract_inverted_index": {"Abstract": [0]}}
        assert _extract_abstract(work) == "Abstract"

    @pytest.mark.unit
    def test_realistic_abstract(self):
        work = {
            "abstract_inverted_index": {
                "We": [0],
                "propose": [1],
                "a": [2, 5],
                "new": [3],
                "simple": [4],
                "network": [6],
                "architecture": [7],
            }
        }
        result = _extract_abstract(work)
        assert result == "We propose a new simple a network architecture"


# ============================================================================
# _assemble_citation_graph (1-hop)
# ============================================================================

class TestAssembleCitationGraph:
    @pytest.mark.unit
    def test_seed_only(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        nodes, edges = _assemble_citation_graph("W1", seed_data, [], [])
        assert len(nodes) == 1
        assert len(edges) == 0
        assert nodes[0].is_seed is True
        assert nodes[0].work_id == "W1"
        assert nodes[0].hop == 0
        assert nodes[0].relationship == "seed"

    @pytest.mark.unit
    def test_citing_papers_create_edges_to_seed(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [
            {"work_id": "W2", "title": "Citer 1", "year": 2021, "cited_by_count": 50, "abstract": None},
            {"work_id": "W3", "title": "Citer 2", "year": 2022, "cited_by_count": 30, "abstract": None},
        ]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, [])
        assert len(nodes) == 3
        assert len(edges) == 2
        # Edges go from citing paper TO seed (citer -> seed)
        for e in edges:
            assert e.to_work_id == "W1"

    @pytest.mark.unit
    def test_references_create_edges_from_seed(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        refs = [
            {"work_id": "W4", "title": "Ref 1", "year": 2015, "cited_by_count": 5000, "abstract": None},
        ]
        nodes, edges = _assemble_citation_graph("W1", seed_data, [], refs)
        assert len(nodes) == 2
        assert len(edges) == 1
        # Edge goes from seed TO reference (seed -> reference)
        assert edges[0].from_work_id == "W1"
        assert edges[0].to_work_id == "W4"

    @pytest.mark.unit
    def test_dedup_by_work_id(self):
        """Same work_id appearing in both citing and references should only appear once."""
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [{"work_id": "W2", "title": "Paper", "year": 2021, "cited_by_count": 50, "abstract": None}]
        refs = [{"work_id": "W2", "title": "Paper", "year": 2021, "cited_by_count": 50, "abstract": None}]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, refs)
        # W2 should appear only once (first seen as citing)
        work_ids = [n.work_id for n in nodes]
        assert work_ids.count("W2") == 1
        assert len(nodes) == 2  # seed + W2

    @pytest.mark.unit
    def test_min_citations_filter(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [
            {"work_id": "W2", "title": "High", "year": 2021, "cited_by_count": 500, "abstract": None},
            {"work_id": "W3", "title": "Low", "year": 2022, "cited_by_count": 5, "abstract": None},
        ]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, [], min_citations=100)
        # Only seed and W2 (500 >= 100), W3 filtered out
        assert len(nodes) == 2
        node_ids = {n.work_id for n in nodes}
        assert "W3" not in node_ids

    @pytest.mark.unit
    def test_skip_papers_without_work_id(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [
            {"work_id": None, "title": "No ID", "year": 2021, "cited_by_count": 500, "abstract": None},
            {"work_id": "", "title": "Empty ID", "year": 2021, "cited_by_count": 500, "abstract": None},
        ]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, [])
        assert len(nodes) == 1  # only seed

    @pytest.mark.unit
    def test_relationship_labels(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [{"work_id": "W2", "title": "Citer", "year": 2021, "cited_by_count": 50, "abstract": None}]
        refs = [{"work_id": "W3", "title": "Ref", "year": 2015, "cited_by_count": 5000, "abstract": None}]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, refs)
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W1"].relationship == "seed"
        assert node_map["W2"].relationship == "cites_seed"
        assert node_map["W3"].relationship == "cited_by_seed"

    @pytest.mark.unit
    def test_hop_values(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [{"work_id": "W2", "title": "Citer", "year": 2021, "cited_by_count": 50, "abstract": None}]
        nodes, _ = _assemble_citation_graph("W1", seed_data, citing, [])
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W1"].hop == 0
        assert node_map["W2"].hop == 1

    @pytest.mark.unit
    def test_seed_not_duplicated_if_in_citing(self):
        """If seed work_id appears in citing list, it should not create a duplicate."""
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [{"work_id": "W1", "title": "Seed again", "year": 2020, "cited_by_count": 100, "abstract": None}]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, [])
        assert len(nodes) == 1  # just the seed

    @pytest.mark.unit
    def test_large_graph(self):
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [
            {"work_id": f"WC{i}", "title": f"Citer {i}", "year": 2021, "cited_by_count": i * 10, "abstract": None}
            for i in range(50)
        ]
        refs = [
            {"work_id": f"WR{i}", "title": f"Ref {i}", "year": 2010, "cited_by_count": i * 100, "abstract": None}
            for i in range(50)
        ]
        nodes, edges = _assemble_citation_graph("W1", seed_data, citing, refs)
        assert len(nodes) == 101  # seed + 50 citing + 50 refs
        assert len(edges) == 100


# ============================================================================
# _assemble_multihop_graph
# ============================================================================

class TestAssembleMultihopGraph:
    @pytest.mark.unit
    def test_seed_only(self):
        papers = {"W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0}}
        nodes, edges = _assemble_multihop_graph("W1", papers, [])
        assert len(nodes) == 1
        assert nodes[0].is_seed is True
        assert nodes[0].relationship == "seed"

    @pytest.mark.unit
    def test_hop1_cites_seed(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "Citer", "year": 2021, "cited_by_count": 50, "is_seed": False, "hop": 1},
        }
        edge_tuples = [("W2", "W1")]  # W2 cites W1
        nodes, edges = _assemble_multihop_graph("W1", papers, edge_tuples)
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W2"].relationship == "cites_seed"

    @pytest.mark.unit
    def test_hop1_cited_by_seed(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W3": {"title": "Ref", "year": 2015, "cited_by_count": 5000, "is_seed": False, "hop": 1},
        }
        edge_tuples = [("W1", "W3")]  # W1 cites W3 (seed -> reference)
        nodes, edges = _assemble_multihop_graph("W1", papers, edge_tuples)
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W3"].relationship == "cited_by_seed"

    @pytest.mark.unit
    def test_hop2_is_network(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "Hop1", "year": 2021, "cited_by_count": 50, "is_seed": False, "hop": 1},
            "W3": {"title": "Hop2", "year": 2022, "cited_by_count": 30, "is_seed": False, "hop": 2},
        }
        edge_tuples = [("W2", "W1"), ("W3", "W2")]
        nodes, edges = _assemble_multihop_graph("W1", papers, edge_tuples)
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W3"].relationship == "network"

    @pytest.mark.unit
    def test_min_citations_filter(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "High", "year": 2021, "cited_by_count": 500, "is_seed": False, "hop": 1},
            "W3": {"title": "Low", "year": 2022, "cited_by_count": 5, "is_seed": False, "hop": 1},
        }
        edge_tuples = [("W2", "W1"), ("W3", "W1")]
        nodes, edges = _assemble_multihop_graph("W1", papers, edge_tuples, min_citations=100)
        node_ids = {n.work_id for n in nodes}
        assert "W3" not in node_ids
        assert "W2" in node_ids
        assert "W1" in node_ids  # seed always included

    @pytest.mark.unit
    def test_edges_filtered_to_included_nodes(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "Included", "year": 2021, "cited_by_count": 500, "is_seed": False, "hop": 1},
            "W3": {"title": "Excluded", "year": 2022, "cited_by_count": 5, "is_seed": False, "hop": 1},
        }
        edge_tuples = [("W2", "W1"), ("W3", "W1"), ("W3", "W2")]
        nodes, edges = _assemble_multihop_graph("W1", papers, edge_tuples, min_citations=100)
        # W3 is excluded, so edges involving W3 should be filtered out
        for e in edges:
            assert e.from_work_id != "W3"
            assert e.to_work_id != "W3"

    @pytest.mark.unit
    def test_empty_graph(self):
        nodes, edges = _assemble_multihop_graph("W1", {}, [])
        assert len(nodes) == 0
        assert len(edges) == 0

    @pytest.mark.unit
    def test_hop_values_preserved(self):
        papers = {
            "W1": {"title": "Seed", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "H1", "year": 2021, "cited_by_count": 50, "is_seed": False, "hop": 1},
            "W3": {"title": "H2", "year": 2022, "cited_by_count": 30, "is_seed": False, "hop": 2},
        }
        nodes, _ = _assemble_multihop_graph("W1", papers, [("W2", "W1"), ("W3", "W2")])
        node_map = {n.work_id: n for n in nodes}
        assert node_map["W1"].hop == 0
        assert node_map["W2"].hop == 1
        assert node_map["W3"].hop == 2


# ============================================================================
# Edge direction semantics
# ============================================================================

class TestEdgeDirection:
    """Verify the contract: from_work_id CITES to_work_id."""

    @pytest.mark.unit
    def test_citing_edge_direction(self):
        """If W2 cites the seed W1, edge should be (W2, W1)."""
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        citing = [{"work_id": "W2", "title": "C", "year": 2021, "cited_by_count": 50, "abstract": None}]
        _, edges = _assemble_citation_graph("W1", seed_data, citing, [])
        assert edges[0].from_work_id == "W2"
        assert edges[0].to_work_id == "W1"

    @pytest.mark.unit
    def test_reference_edge_direction(self):
        """If seed W1 cites W3, edge should be (W1, W3)."""
        seed_data = {"title": "Seed", "year": 2020, "cited_by_count": 100, "abstract": None}
        refs = [{"work_id": "W3", "title": "R", "year": 2015, "cited_by_count": 500, "abstract": None}]
        _, edges = _assemble_citation_graph("W1", seed_data, [], refs)
        assert edges[0].from_work_id == "W1"
        assert edges[0].to_work_id == "W3"

    @pytest.mark.unit
    def test_multihop_edge_direction_preserved(self):
        papers = {
            "W1": {"title": "S", "year": 2020, "cited_by_count": 100, "is_seed": True, "hop": 0},
            "W2": {"title": "A", "year": 2021, "cited_by_count": 50, "is_seed": False, "hop": 1},
        }
        edge_tuples = [("W2", "W1")]
        _, edges = _assemble_multihop_graph("W1", papers, edge_tuples)
        assert edges[0].from_work_id == "W2"
        assert edges[0].to_work_id == "W1"


# ============================================================================
# TIER_SCORES mapping
# ============================================================================

class TestTierScores:
    @pytest.mark.unit
    def test_tier_scores_values(self):
        from app.feature1.citation_map_service import TIER_SCORES
        assert TIER_SCORES["ESSENTIAL"] == 0.95
        assert TIER_SCORES["HIGH"] == 0.75
        assert TIER_SCORES["MEDIUM"] == 0.50
        assert TIER_SCORES["LOW"] == 0.25
        assert TIER_SCORES["NONE"] == 0.05

    @pytest.mark.unit
    def test_tier_scores_ordering(self):
        from app.feature1.citation_map_service import TIER_SCORES
        assert TIER_SCORES["ESSENTIAL"] > TIER_SCORES["HIGH"]
        assert TIER_SCORES["HIGH"] > TIER_SCORES["MEDIUM"]
        assert TIER_SCORES["MEDIUM"] > TIER_SCORES["LOW"]
        assert TIER_SCORES["LOW"] > TIER_SCORES["NONE"]


# ============================================================================
# Schema validation
# ============================================================================

class TestSchemaDefaults:
    @pytest.mark.unit
    def test_citation_map_request_defaults(self):
        from app.feature1.schemas import CitationMapRequest
        req = CitationMapRequest()
        assert req.citing_limit == 15
        assert req.references_limit == 15
        assert req.min_citations == 0
        assert req.create_graph_draft is True
        assert req.seed_work_id is None
        assert req.query_text is None

    @pytest.mark.unit
    def test_citation_node_defaults(self):
        node = CitationNode(work_id="W1", relationship="seed")
        assert node.hop == 0
        assert node.is_seed is False
        assert node.cited_by_count == 0

    @pytest.mark.unit
    def test_citation_edge_fields(self):
        edge = CitationEdge(from_work_id="W1", to_work_id="W2")
        assert edge.from_work_id == "W1"
        assert edge.to_work_id == "W2"

    @pytest.mark.unit
    def test_citation_map_stats_defaults(self):
        from app.feature1.schemas import CitationMapStats
        stats = CitationMapStats()
        assert stats.total_nodes == 0
        assert stats.edges_count == 0
        assert stats.max_hop == 0


# ============================================================================
# Scoring constants
# ============================================================================

class TestScoringConstants:
    @pytest.mark.unit
    def test_min_seed_llm_score(self):
        from app.feature1.citation_map_service import MIN_SEED_LLM_SCORE
        assert MIN_SEED_LLM_SCORE == 0.75

    @pytest.mark.unit
    def test_min_connection_llm_score(self):
        from app.feature1.citation_map_service import MIN_CONNECTION_LLM_SCORE
        assert MIN_CONNECTION_LLM_SCORE == 0.50


# ============================================================================
# Rate Limiting (S2, OpenAlex, ArXiv)
# ============================================================================

class TestS2RateLimiting:
    """Verify proactive S2 throttle prevents 429s."""

    @pytest.mark.unit
    def test_throttle_sleeps_when_called_rapidly(self):
        """Back-to-back _s2_throttle calls should sleep to enforce interval."""
        recent_ts = time.time()
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=recent_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = recent_ts + 0.1
            _s2_throttle()
            mock_time.sleep.assert_called_once()
            sleep_arg = mock_time.sleep.call_args[0][0]
            assert 0.8 < sleep_arg < S2_MIN_REQUEST_INTERVAL

    @pytest.mark.unit
    def test_throttle_no_sleep_when_interval_elapsed(self):
        """No sleep needed if enough time has passed since last request."""
        old_ts = time.time() - 10.0
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=old_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = time.time()
            _s2_throttle()
            mock_time.sleep.assert_not_called()

    @pytest.mark.unit
    def test_throttle_updates_timestamp(self):
        """_s2_throttle must persist timestamp after running."""
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts") as mock_write:
            _s2_throttle()
            mock_write.assert_called_once()
            written_ts = mock_write.call_args[0][1]
            assert written_ts >= time.time() - 1.0

    @pytest.mark.unit
    def test_get_with_retry_calls_throttle(self):
        """_s2_get_with_retry must call _s2_throttle before making request."""
        import app.feature1.citation_map_service as svc
        svc._s2_rate_limited_until_ts = 0.0
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("app.feature1.citation_map_service._s2_throttle") as mock_throttle, \
             patch("app.feature1.citation_map_service.requests.get", return_value=mock_resp):
            _s2_get_with_retry("https://api.semanticscholar.org/test")
            mock_throttle.assert_called_once()

    @pytest.mark.unit
    def test_consecutive_requests_are_spaced(self):
        """Two real _s2_throttle calls back-to-back take >= interval seconds."""
        timestamps = []
        with patch("app.feature1.citation_map_service._read_throttle_ts", side_effect=lambda k: timestamps[-1] if timestamps else 0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts", side_effect=lambda k, ts: timestamps.append(ts)):
            _s2_throttle()
            _s2_throttle()
            assert len(timestamps) == 2
            assert timestamps[1] - timestamps[0] >= S2_MIN_REQUEST_INTERVAL - 0.05


class TestOARateLimiting:
    """Verify proactive OpenAlex throttle prevents 429s (10 RPS limit)."""

    @pytest.mark.unit
    def test_throttle_sleeps_when_called_rapidly(self):
        recent_ts = time.time()
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=recent_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = recent_ts + 0.02
            _oa_throttle()
            mock_time.sleep.assert_called_once()
            sleep_arg = mock_time.sleep.call_args[0][0]
            assert 0.05 < sleep_arg < OA_MIN_REQUEST_INTERVAL

    @pytest.mark.unit
    def test_throttle_no_sleep_when_interval_elapsed(self):
        old_ts = time.time() - 5.0
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=old_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = time.time()
            _oa_throttle()
            mock_time.sleep.assert_not_called()

    @pytest.mark.unit
    def test_throttle_updates_timestamp(self):
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts") as mock_write:
            _oa_throttle()
            mock_write.assert_called_once()
            written_ts = mock_write.call_args[0][1]
            assert written_ts >= time.time() - 1.0

    @pytest.mark.unit
    def test_consecutive_requests_are_spaced(self):
        timestamps = []
        with patch("app.feature1.citation_map_service._read_throttle_ts", side_effect=lambda k: timestamps[-1] if timestamps else 0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts", side_effect=lambda k, ts: timestamps.append(ts)):
            _oa_throttle()
            _oa_throttle()
            assert len(timestamps) == 2
            assert timestamps[1] - timestamps[0] >= OA_MIN_REQUEST_INTERVAL - 0.02


class TestArXivRateLimiting:
    """Verify proactive ArXiv throttle prevents 429s (1 req/3s limit)."""

    @pytest.mark.unit
    def test_throttle_sleeps_when_called_rapidly(self):
        recent_ts = time.time()
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=recent_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = recent_ts + 0.5
            _arxiv_throttle()
            mock_time.sleep.assert_called_once()
            sleep_arg = mock_time.sleep.call_args[0][0]
            assert 2.0 < sleep_arg < ARXIV_MIN_REQUEST_INTERVAL

    @pytest.mark.unit
    def test_throttle_no_sleep_when_interval_elapsed(self):
        old_ts = time.time() - 10.0
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=old_ts), \
             patch("app.feature1.citation_map_service._write_throttle_ts"), \
             patch("app.feature1.citation_map_service.time") as mock_time:
            mock_time.time.return_value = time.time()
            _arxiv_throttle()
            mock_time.sleep.assert_not_called()

    @pytest.mark.unit
    def test_throttle_updates_timestamp(self):
        with patch("app.feature1.citation_map_service._read_throttle_ts", return_value=0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts") as mock_write:
            _arxiv_throttle()
            mock_write.assert_called_once()
            written_ts = mock_write.call_args[0][1]
            assert written_ts >= time.time() - 1.0

    @pytest.mark.unit
    def test_consecutive_requests_are_spaced(self):
        timestamps = []
        with patch("app.feature1.citation_map_service._read_throttle_ts", side_effect=lambda k: timestamps[-1] if timestamps else 0.0), \
             patch("app.feature1.citation_map_service._write_throttle_ts", side_effect=lambda k, ts: timestamps.append(ts)):
            _arxiv_throttle()
            _arxiv_throttle()
            assert len(timestamps) == 2
            assert timestamps[1] - timestamps[0] >= ARXIV_MIN_REQUEST_INTERVAL - 0.1
