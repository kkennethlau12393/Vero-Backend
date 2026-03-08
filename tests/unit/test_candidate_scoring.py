"""
Unit tests for citation map V2 candidate scoring.

Tests pure scoring functions and greedy graph selection.
NO DB, NO network.
"""
from __future__ import annotations

import pytest

from app.feature1.candidate_scoring import (
    compute_relevance_score,
    compute_temporal_score,
    compute_connectivity_score,
    get_drift_thresholds,
    get_fetch_limits,
    greedy_graph_select,
)


def _make_sq(**kwargs):
    """Build a structured query dict with defaults."""
    return {
        "topic": kwargs.get("topic", "natural language processing"),
        "domain": kwargs.get("domain", "legal"),
        "aspect": kwargs.get("aspect", ""),
        "topic_aliases": kwargs.get("topic_aliases", ["NLP"]),
        "domain_aliases": kwargs.get("domain_aliases", ["law"]),
        "aspect_aliases": kwargs.get("aspect_aliases", []),
    }


def _make_paper(title, abstract="", year=2020, cited_by_count=100, work_id="W1"):
    return {
        "work_id": work_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "cited_by_count": cited_by_count,
    }


# ── compute_relevance_score ──────────────────────────────────────────────────

class TestComputeRelevanceScore:
    @pytest.mark.unit
    def test_intersection_both_match(self):
        sq = _make_sq()
        paper = _make_paper("NLP for Legal Text Analysis", "natural language processing law")
        score = compute_relevance_score(paper, sq, "intersection")
        assert score >= 0.8

    @pytest.mark.unit
    def test_intersection_one_match(self):
        sq = _make_sq()
        paper = _make_paper("NLP Methods for Text", "natural language processing")
        score = compute_relevance_score(paper, sq, "intersection")
        assert 0.3 <= score <= 0.6

    @pytest.mark.unit
    def test_intersection_no_match(self):
        sq = _make_sq()
        paper = _make_paper("Quantum Computing Basics", "quantum entanglement physics")
        score = compute_relevance_score(paper, sq, "intersection")
        assert score <= 0.15

    @pytest.mark.unit
    def test_broad_either_match(self):
        sq = _make_sq()
        paper = _make_paper("Legal Document Analysis", "law text classification")
        score = compute_relevance_score(paper, sq, "broad")
        assert score >= 0.7

    @pytest.mark.unit
    def test_topic_focused_topic_match(self):
        sq = _make_sq()
        paper = _make_paper("NLP for Healthcare", "natural language processing medical")
        score = compute_relevance_score(paper, sq, "topic_focused")
        assert score >= 0.8

    @pytest.mark.unit
    def test_domain_focused_domain_match(self):
        sq = _make_sq()
        paper = _make_paper("Machine Learning in Law", "legal classification")
        score = compute_relevance_score(paper, sq, "domain_focused")
        assert score >= 0.8

    @pytest.mark.unit
    def test_aspect_bonus(self):
        sq = _make_sq(aspect="sentiment analysis", aspect_aliases=["opinion mining"])
        paper = _make_paper("NLP for Legal Opinion Mining", "natural language processing law")
        score_with = compute_relevance_score(paper, sq, "intersection")

        sq_no_aspect = _make_sq(aspect="", aspect_aliases=[])
        score_without = compute_relevance_score(paper, sq_no_aspect, "intersection")
        assert score_with > score_without

    @pytest.mark.unit
    def test_title_double_match_bonus(self):
        sq = _make_sq()
        paper_title_match = _make_paper("NLP for Legal Analysis")
        paper_abstract_match = _make_paper("Generic Paper", "NLP for legal analysis")
        score_title = compute_relevance_score(paper_title_match, sq, "intersection")
        score_abstract = compute_relevance_score(paper_abstract_match, sq, "intersection")
        assert score_title >= score_abstract

    @pytest.mark.unit
    def test_alias_matching(self):
        sq = _make_sq(topic_aliases=["NLP", "computational linguistics"])
        paper = _make_paper("Computational Linguistics in Courts", "legal applications")
        score = compute_relevance_score(paper, sq, "intersection")
        assert score >= 0.4  # alias match on topic

    @pytest.mark.unit
    def test_no_terms_returns_base(self):
        sq = _make_sq(topic="", domain="", topic_aliases=[], domain_aliases=[])
        paper = _make_paper("Any Paper", "any content")
        score = compute_relevance_score(paper, sq, "broad")
        assert score == 0.1  # no terms → no match


# ── compute_temporal_score ───────────────────────────────────────────────────

class TestComputeTemporalScore:
    @pytest.mark.unit
    def test_seminal_old_highly_cited(self):
        paper = _make_paper("Classic Paper", year=2000, cited_by_count=5000)
        score = compute_temporal_score(paper, "seminal", current_year=2026)
        assert score >= 0.8

    @pytest.mark.unit
    def test_seminal_new_few_cites(self):
        paper = _make_paper("New Paper", year=2025, cited_by_count=10)
        score = compute_temporal_score(paper, "seminal", current_year=2026)
        assert score < 0.2

    @pytest.mark.unit
    def test_recent_new(self):
        paper = _make_paper("Brand New", year=2025, cited_by_count=10)
        score = compute_temporal_score(paper, "recent", current_year=2026)
        assert score == 1.0

    @pytest.mark.unit
    def test_recent_old(self):
        paper = _make_paper("Ancient Paper", year=2000, cited_by_count=5000)
        score = compute_temporal_score(paper, "recent", current_year=2026)
        assert score <= 0.1

    @pytest.mark.unit
    def test_recent_moderate_age(self):
        paper = _make_paper("Mid Paper", year=2022, cited_by_count=100)
        score = compute_temporal_score(paper, "recent", current_year=2026)
        assert score == 0.8

    @pytest.mark.unit
    def test_all_neutral(self):
        paper = _make_paper("Any Paper", year=2020, cited_by_count=100)
        score = compute_temporal_score(paper, "all", current_year=2026)
        assert score == 0.5

    @pytest.mark.unit
    def test_missing_year_uses_current(self):
        paper = {"work_id": "W1", "title": "No Year", "cited_by_count": 100}
        score = compute_temporal_score(paper, "recent", current_year=2026)
        assert score == 1.0  # treated as current year


# ── compute_connectivity_score ───────────────────────────────────────────────

class TestComputeConnectivityScore:
    @pytest.mark.unit
    def test_isolated_paper(self):
        edges = {"W2": {"W3"}}
        score = compute_connectivity_score("W1", {"W2", "W3"}, edges)
        assert score == 0.0

    @pytest.mark.unit
    def test_one_connection(self):
        edges = {"W1": {"W2"}}
        score = compute_connectivity_score("W1", {"W2", "W3"}, edges)
        assert score == 0.3

    @pytest.mark.unit
    def test_multiple_connections(self):
        edges = {"W1": {"W2", "W3", "W4"}}
        score = compute_connectivity_score("W1", {"W2", "W3", "W4"}, edges)
        assert score == 0.6

    @pytest.mark.unit
    def test_highly_connected(self):
        edges = {"W1": {"W2", "W3", "W4", "W5", "W6", "W7", "W8"}}
        score = compute_connectivity_score("W1", {"W2", "W3", "W4", "W5", "W6", "W7", "W8"}, edges)
        assert score == 1.0

    @pytest.mark.unit
    def test_empty_graph(self):
        score = compute_connectivity_score("W1", set(), {})
        assert score == 0.5  # neutral for first papers


# ── get_drift_thresholds ─────────────────────────────────────────────────────

class TestGetDriftThresholds:
    @pytest.mark.unit
    def test_strict_higher_than_open(self):
        strict = get_drift_thresholds("strict")
        open_ = get_drift_thresholds("open")
        assert strict["hop1_threshold"] > open_["hop1_threshold"]
        assert strict["hop2_threshold"] > open_["hop2_threshold"]

    @pytest.mark.unit
    def test_moderate_between(self):
        strict = get_drift_thresholds("strict")
        moderate = get_drift_thresholds("moderate")
        open_ = get_drift_thresholds("open")
        assert strict["hop1_threshold"] > moderate["hop1_threshold"] > open_["hop1_threshold"]

    @pytest.mark.unit
    def test_open_allows_hop3(self):
        open_ = get_drift_thresholds("open")
        assert open_["max_hops"] == 3

    @pytest.mark.unit
    def test_strict_limits_hop2(self):
        strict = get_drift_thresholds("strict")
        assert strict["max_hops"] == 2

    @pytest.mark.unit
    def test_unknown_defaults_to_open(self):
        result = get_drift_thresholds("unknown")
        assert result["max_hops"] == 3  # falls through to open


# ── get_fetch_limits ─────────────────────────────────────────────────────────

class TestGetFetchLimits:
    @pytest.mark.unit
    def test_small_values(self):
        limits = get_fetch_limits("small")
        assert limits["target_nodes"] == 25
        assert limits["hop1_fetch"] == 50

    @pytest.mark.unit
    def test_medium_values(self):
        limits = get_fetch_limits("medium")
        assert limits["target_nodes"] == 40
        assert limits["hop1_fetch"] == 80

    @pytest.mark.unit
    def test_large_values(self):
        limits = get_fetch_limits("large")
        assert limits["target_nodes"] == 60
        assert limits["hop1_fetch"] == 120

    @pytest.mark.unit
    def test_scale_order(self):
        small = get_fetch_limits("small")
        medium = get_fetch_limits("medium")
        large = get_fetch_limits("large")
        assert small["target_nodes"] < medium["target_nodes"] < large["target_nodes"]
        assert small["hop1_fetch"] < medium["hop1_fetch"] < large["hop1_fetch"]
        assert small["hop2_fetch"] < medium["hop2_fetch"] < large["hop2_fetch"]
        assert small["hop2_expand_count"] < medium["hop2_expand_count"] < large["hop2_expand_count"]

    @pytest.mark.unit
    def test_unknown_defaults_to_medium(self):
        limits = get_fetch_limits("unknown")
        assert limits["target_nodes"] == 40


# ── greedy_graph_select ──────────────────────────────────────────────────────

class TestGreedyGraphSelect:
    @pytest.mark.unit
    def test_seed_always_first(self):
        candidates = [
            _make_paper("Seed Paper", work_id="SEED"),
            _make_paper("NLP for Law", "natural language processing legal", work_id="W1"),
        ]
        edges = {"W1": {"SEED"}, "SEED": {"W1"}}
        sq = _make_sq()
        result = greedy_graph_select(candidates, edges, "SEED", 2, sq, "broad", "all")
        assert result[0] == "SEED"

    @pytest.mark.unit
    def test_respects_target_size(self):
        candidates = [_make_paper("Seed", work_id="SEED")]
        candidates += [
            _make_paper(f"NLP Paper {i}", "natural language processing", work_id=f"W{i}")
            for i in range(20)
        ]
        edges = {f"W{i}": {"SEED"} for i in range(20)}
        edges["SEED"] = {f"W{i}" for i in range(20)}
        sq = _make_sq()
        result = greedy_graph_select(candidates, edges, "SEED", 5, sq, "broad", "all")
        assert len(result) == 5

    @pytest.mark.unit
    def test_prefers_connected_over_isolated(self):
        candidates = [
            _make_paper("Seed", work_id="SEED"),
            _make_paper("Connected NLP Law", "natural language processing legal", work_id="W1"),
            _make_paper("Isolated NLP Law", "natural language processing legal", work_id="W2"),
        ]
        # W1 is connected to seed, W2 is isolated
        edges = {"W1": {"SEED"}, "SEED": {"W1"}}
        sq = _make_sq()
        result = greedy_graph_select(candidates, edges, "SEED", 2, sq, "broad", "all")
        assert "W1" in result
        # W2 should not be selected when W1 is equally relevant but connected
        if len(result) == 2:
            assert result[1] == "W1"

    @pytest.mark.unit
    def test_candidate_pool_cap(self):
        """With many candidates, greedy select still returns correct count."""
        candidates = [_make_paper("Seed", work_id="SEED")]
        candidates += [
            _make_paper(f"Paper {i}", "natural language processing", work_id=f"W{i}")
            for i in range(500)
        ]
        edges = {f"W{i}": {"SEED"} for i in range(500)}
        edges["SEED"] = {f"W{i}" for i in range(500)}
        sq = _make_sq()
        # target=10, candidates=500. Pool should be capped to 30 (3*10)
        result = greedy_graph_select(candidates, edges, "SEED", 10, sq, "broad", "all")
        assert len(result) == 10
        assert result[0] == "SEED"

    @pytest.mark.unit
    def test_empty_candidates(self):
        result = greedy_graph_select([], {}, "SEED", 5, _make_sq(), "broad", "all")
        assert result == ["SEED"]

    @pytest.mark.unit
    def test_temporal_preference_affects_selection(self):
        """Seminal temporal should prefer old, highly-cited papers."""
        candidates = [
            _make_paper("Seed", work_id="SEED"),
            _make_paper("Classic NLP", "natural language processing", year=2000, cited_by_count=5000, work_id="OLD"),
            _make_paper("New NLP", "natural language processing", year=2025, cited_by_count=10, work_id="NEW"),
        ]
        edges = {"OLD": {"SEED"}, "NEW": {"SEED"}, "SEED": {"OLD", "NEW"}}
        sq = _make_sq()

        # With seminal temporal, old paper should be preferred
        result_seminal = greedy_graph_select(candidates, edges, "SEED", 2, sq, "broad", "seminal")
        assert result_seminal[1] == "OLD"

        # With recent temporal, new paper should be preferred
        result_recent = greedy_graph_select(candidates, edges, "SEED", 2, sq, "broad", "recent")
        assert result_recent[1] == "NEW"

    @pytest.mark.unit
    def test_relevance_floor_filters_low_relevance(self):
        """Papers below relevance floor are excluded even with high connectivity."""
        candidates = [
            _make_paper("Seed", work_id="SEED"),
            # Low relevance (no topic/domain match) but highly connected
            _make_paper("Quantum Physics", "quantum entanglement research", work_id="LOW_REL"),
            # High relevance (topic match) with same connectivity
            _make_paper("NLP for Law", "natural language processing legal", work_id="HIGH_REL"),
        ]
        edges = {
            "LOW_REL": {"SEED", "HIGH_REL"},
            "HIGH_REL": {"SEED", "LOW_REL"},
            "SEED": {"LOW_REL", "HIGH_REL"},
        }
        sq = _make_sq()  # topic=NLP, domain=legal

        # With floor=0.3, quantum paper (relevance ~0.1) should be excluded
        result = greedy_graph_select(
            candidates, edges, "SEED", 2, sq, "broad", "all",
            relevance_floor=0.3,
        )
        assert "HIGH_REL" in result
        assert "LOW_REL" not in result


# ── Two-phase backbone + enrichment ─────────────────────────────────────────

class TestGreedyGraphSelectTwoPhase:
    """Test two-phase backbone + enrichment selection."""

    @pytest.mark.unit
    def test_backbone_preserves_graph_structure(self):
        """Graph-traversal papers selected first, enrichment fills remaining."""
        seed = "SEED"
        candidates = []
        edges = {seed: set()}

        # 5 graph papers connected to seed and each other
        for i in range(5):
            wid = f"GRAPH_{i}"
            candidates.append({
                "work_id": wid,
                "title": f"graph neural network paper {i}",
                "year": 2020,
                "cited_by_count": 100,
            })
            edges.setdefault(seed, set()).add(wid)
            edges.setdefault(wid, set()).add(seed)
            if i > 0:
                prev = f"GRAPH_{i-1}"
                edges.setdefault(wid, set()).add(prev)
                edges.setdefault(prev, set()).add(wid)

        # 5 enrichment papers — highly relevant but isolated
        for i in range(5):
            candidates.append({
                "work_id": f"ENRICH_{i}",
                "title": f"graph neural networks for drug discovery molecule {i}",
                "year": 2023,
                "cited_by_count": 50,
                "_source": "retrieval_enrichment",
            })

        sq = {"topic": "graph neural networks", "domain": "drug discovery"}
        result = greedy_graph_select(
            candidates, edges, seed, target_size=8,
            structured_query=sq, scope="intersection", temporal="all",
            backbone_ratio=0.6,
        )

        assert result[0] == seed
        graph_in_result = [r for r in result if r.startswith("GRAPH_")]
        enrich_in_result = [r for r in result if r.startswith("ENRICH_")]
        # backbone_target = int(8 * 0.6) = 4 (includes seed), so 3 graph papers in backbone
        assert len(graph_in_result) >= 3, "Backbone should preserve most graph papers"
        assert len(enrich_in_result) >= 1, "Enrichment should fill remaining slots"

    @pytest.mark.unit
    def test_enrichment_papers_need_source_tag(self):
        """Papers without _source tag are treated as graph candidates."""
        candidates = [
            {"work_id": "A", "title": "test", "year": 2020, "cited_by_count": 10},
            {"work_id": "B", "title": "test", "year": 2020, "cited_by_count": 10,
             "_source": "retrieval_enrichment"},
        ]
        edges = {"SEED": {"A", "B"}, "A": {"SEED"}, "B": {"SEED"}}
        sq = {"topic": "test"}

        result = greedy_graph_select(
            candidates, edges, "SEED", target_size=3,
            structured_query=sq, scope="broad", temporal="all",
        )
        assert len(result) == 3

    @pytest.mark.unit
    def test_no_enrichment_papers_works_normally(self):
        """When no enrichment papers exist, all slots go to graph papers."""
        candidates = [
            {"work_id": f"G{i}", "title": "graph paper", "year": 2020, "cited_by_count": 100}
            for i in range(10)
        ]
        edges = {"SEED": set(f"G{i}" for i in range(10))}
        for i in range(10):
            edges[f"G{i}"] = {"SEED"}
        sq = {"topic": "graph"}

        result = greedy_graph_select(
            candidates, edges, "SEED", target_size=6,
            structured_query=sq, scope="broad", temporal="all",
        )
        assert len(result) == 6
        assert result[0] == "SEED"
