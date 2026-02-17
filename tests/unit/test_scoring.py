"""
Unit tests for Feature 2 scoring / normalisation functions.

Tests the real functions from:
  - app.feature2.features (robust_norm, bayesian_impact_rate, compute_recency, etc.)
  - app.feature2.rerank   (build_reasons, mmr_diversify)
"""
from __future__ import annotations

import math
import pytest

from app.feature2.features import (
    robust_norm,
    bayesian_impact_rate,
    compute_recency,
    compute_topic_relevance,
    compute_completeness,
    compute_age,
    compute_llm_relevance_feature,
)
from app.feature2.rerank import build_reasons, mmr_diversify
from tests.fixtures.seed_data import make_work


# ── robust_norm ──────────────────────────────────────────────────────────────

class TestRobustNorm:
    def test_empty_dict(self):
        assert robust_norm({}) == {}

    def test_single_value(self):
        # p05 == p95 → all zeros
        assert robust_norm({"a": 5.0}) == {"a": 0.0}

    def test_two_equal_values(self):
        assert robust_norm({"a": 3.0, "b": 3.0}) == {"a": 0.0, "b": 0.0}

    def test_linear_spread(self):
        vals = {str(i): float(i) for i in range(20)}
        normed = robust_norm(vals)
        # Extreme low should be 0, extreme high should be 1
        assert normed["0"] == 0.0
        assert normed["19"] == 1.0
        # Monotonically non-decreasing
        prev = -1.0
        for i in range(20):
            assert normed[str(i)] >= prev
            prev = normed[str(i)]

    def test_values_clipped_to_0_1(self):
        normed = robust_norm({str(i): float(i) for i in range(100)})
        for v in normed.values():
            assert 0.0 <= v <= 1.0


# ── bayesian_impact_rate ─────────────────────────────────────────────────────

class TestBayesianImpactRate:
    def test_zero_citations_positive_rate(self):
        rate = bayesian_impact_rate(0, 5.0)
        # With prior alpha=1, beta=2: (1+0)/(2+6) = 0.125
        assert rate == pytest.approx(1.0 / 8.0)

    def test_high_citations(self):
        rate = bayesian_impact_rate(1000, 10.0)
        # (1+1000)/(2+11) = 1001/13 ≈ 77.0
        assert rate > 50.0

    def test_zero_age(self):
        # age=0 → exposure = 0+1 = 1
        rate = bayesian_impact_rate(10, 0.0)
        assert rate == pytest.approx((1 + 10) / (2 + 1))

    def test_uncertainty_penalty_reduces_rate(self):
        base = bayesian_impact_rate(50, 5.0, uncertainty_gamma=0.0)
        penalised = bayesian_impact_rate(50, 5.0, uncertainty_gamma=2.0)
        assert penalised < base

    def test_base_rate_normalises(self):
        raw = bayesian_impact_rate(100, 5.0)
        normed = bayesian_impact_rate(100, 5.0, base_rate=raw)
        assert normed == pytest.approx(1.0)

    def test_never_negative(self):
        rate = bayesian_impact_rate(0, 100.0, uncertainty_gamma=100.0)
        assert rate >= 0.0


# ── compute_recency ──────────────────────────────────────────────────────────

class TestComputeRecency:
    def test_zero_age_is_one(self):
        assert compute_recency(0.0, 5.0) == pytest.approx(1.0)

    def test_half_life(self):
        # At age == half_life, recency = exp(-1) ≈ 0.368
        assert compute_recency(5.0, 5.0) == pytest.approx(math.exp(-1.0))

    def test_negative_age_is_zero(self):
        assert compute_recency(-1.0, 5.0) == 0.0

    def test_none_age_is_zero(self):
        assert compute_recency(None, 5.0) == 0.0

    def test_zero_half_life_uses_default(self):
        # half_life 0 → treated as 1
        val = compute_recency(2.0, 0.0)
        assert val == pytest.approx(math.exp(-2.0))


# ── compute_topic_relevance ──────────────────────────────────────────────────

class TestComputeTopicRelevance:
    def test_primary_match(self):
        w = make_work(primary_topic_id="T100")
        assert compute_topic_relevance(w, "T100") == 1.0

    def test_secondary_match(self):
        from app.feature2.work_topic_store import TopicScore
        w = make_work(
            primary_topic_id="T100",
            topics=[TopicScore("T100", 0.95), TopicScore("T200", 0.6)],
        )
        assert compute_topic_relevance(w, "T200") == pytest.approx(0.6)

    def test_no_match(self):
        w = make_work(primary_topic_id="T100")
        assert compute_topic_relevance(w, "T999") == 0.0

    def test_none_target(self):
        w = make_work()
        assert compute_topic_relevance(w, None) == 0.0


# ── compute_completeness ─────────────────────────────────────────────────────

class TestComputeCompleteness:
    def test_full_metadata(self):
        w = make_work()
        assert compute_completeness(w) == 1.0

    def test_missing_venue(self):
        w = make_work(venue=None)
        assert compute_completeness(w) == pytest.approx(4.0 / 5.0)

    def test_none_work(self):
        assert compute_completeness(None) == 0.0

    def test_empty_authors(self):
        w = make_work(authors_json=[])
        # missing authors → 4/5
        assert compute_completeness(w) == pytest.approx(4.0 / 5.0)


# ── compute_age ──────────────────────────────────────────────────────────────

class TestComputeAge:
    def test_none_year_default(self):
        assert compute_age(None) == 10.0

    def test_current_year(self):
        from datetime import date
        assert compute_age(date.today().year) == 0.0

    def test_past_year(self):
        from datetime import date
        age = compute_age(2000)
        assert age == pytest.approx(float(date.today().year - 2000))


# ── compute_llm_relevance_feature ────────────────────────────────────────────

class TestLLMRelevanceFeature:
    def test_missing_papers_get_zero(self):
        scores = {"W1": 0.8}
        result = compute_llm_relevance_feature(scores, ["W1", "W2"])
        assert result["W1"] == 0.8
        assert result["W2"] == 0.0


# ── build_reasons ────────────────────────────────────────────────────────────

class TestBuildReasons:
    def test_high_impact_gets_reason(self):
        reasons = build_reasons(
            rel_lex={"W1": 0.0},
            rel_llm={"W1": 0.0},
            rel_topic={"W1": 0.0},
            impact={"W1": 0.9},
            recency={"W1": 0.0},
            completeness={"W1": 0.0},
        )
        assert "High impact for its age" in reasons["W1"]

    def test_highly_relevant(self):
        reasons = build_reasons(
            rel_lex={"W1": 0.0},
            rel_llm={"W1": 0.8},
            rel_topic={"W1": 0.0},
            impact={"W1": 0.0},
            recency={"W1": 0.0},
            completeness={"W1": 0.0},
        )
        assert "Highly relevant" in reasons["W1"]

    def test_no_reasons_below_thresholds(self):
        reasons = build_reasons(
            rel_lex={"W1": 0.1},
            rel_llm={"W1": 0.1},
            rel_topic={"W1": 0.1},
            impact={"W1": 0.1},
            recency={"W1": 0.1},
            completeness={"W1": 0.1},
        )
        assert reasons.get("W1", []) == []


# ── mmr_diversify ────────────────────────────────────────────────────────────

class TestMMRDiversify:
    def test_empty_input(self):
        assert mmr_diversify([], {}, 5) == []

    def test_k_larger_than_pool(self):
        scored = [("W1", 1.0), ("W2", 0.5)]
        result = mmr_diversify(scored, {}, 10)
        assert len(result) == 2

    def test_pure_relevance_lambda_1(self):
        scored = [("W1", 1.0), ("W2", 0.8), ("W3", 0.5)]
        result = mmr_diversify(scored, {}, 3, lambda_param=1.0)
        # With lambda=1, pure relevance order
        assert result[0] == "W1"

    def test_diversity_with_identical_embeddings(self):
        emb = [1.0, 0.0, 0.0]
        scored = [("W1", 1.0), ("W2", 0.99), ("W3", 0.5)]
        embeddings = {"W1": emb, "W2": emb, "W3": [0.0, 1.0, 0.0]}
        result = mmr_diversify(scored, embeddings, 3, lambda_param=0.5)
        # W3 should be promoted due to diversity
        assert "W3" in result
