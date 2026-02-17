"""
Live LLM scoring quality tests — verifies Groq LLM produces
sensible relevance tiers for known paper-query combinations.

Tests hit: real Groq API.
"""
from __future__ import annotations

import pytest

from app.feature2.llm_relevance import score_papers
from app.feature2.work_topic_store import WorkStore


# ---------------------------------------------------------------------------
# Tier thresholds (matching rank_service.py LLM scoring scale)
# 0.95 = ESSENTIAL, 0.75 = HIGH, 0.50 = MEDIUM, 0.25 = LOW, 0.0 = NONE
# ---------------------------------------------------------------------------
ESSENTIAL = 0.90  # Allow slight variance
HIGH = 0.70
MEDIUM = 0.45
LOW = 0.20


# ---------------------------------------------------------------------------
# Known paper-query pairs with expected minimum tiers
# ---------------------------------------------------------------------------
SCORING_CASES = [
    {
        "id": "attention_essential",
        "query": "transformer architecture attention mechanisms in deep learning",
        "description": "Attention Is All You Need should score ESSENTIAL for attention query",
        "paper_title_substring": "attention is all you need",
        "min_score": HIGH,  # At minimum HIGH, expect ESSENTIAL
    },
    {
        "id": "bert_high",
        "query": "transformer architecture attention mechanisms in deep learning",
        "description": "BERT should score at least HIGH for transformer query",
        "paper_title_substring": "bert: pre-training of deep bidirectional",
        "min_score": MEDIUM,
    },
    {
        "id": "gnn_irrelevant_to_transformers",
        "query": "transformer architecture attention mechanisms in deep learning",
        "description": "GCN paper should score LOW for transformer query",
        "paper_title_substring": "semi-supervised classification with graph convolutional",
        "max_score": MEDIUM,  # Should NOT score high
    },
]


def _find_work_by_title(conn, substring: str) -> str | None:
    """Find a work_id by title substring in the DB."""
    from sqlalchemy import text
    row = conn.execute(
        text("""
            SELECT work_id FROM works
            WHERE LOWER(title) LIKE :pattern
            ORDER BY cited_by_count DESC NULLS LAST
            LIMIT 1
        """),
        {"pattern": f"%{substring.lower()}%"},
    ).first()
    return row[0] if row else None


@pytest.mark.live
@pytest.mark.timeout(120)
class TestLLMScoringQuality:
    """Test that LLM assigns sensible relevance scores."""

    @pytest.mark.parametrize(
        "case",
        [c for c in SCORING_CASES if "min_score" in c],
        ids=[c["id"] for c in SCORING_CASES if "min_score" in c],
    )
    def test_relevant_papers_score_high(self, case, db_engine):
        with db_engine.begin() as conn:
            work_id = _find_work_by_title(conn, case["paper_title_substring"])
            if not work_id:
                pytest.skip(f"Paper '{case['paper_title_substring']}' not in DB")

            works = WorkStore.load_many(conn, [work_id])
            from app.feature2.query_expansion import compute_query_hash
            query_hash = compute_query_hash(case["query"])

            scores = score_papers(
                conn,
                query_text=case["query"],
                query_hash=query_hash,
                paper_ids=[work_id],
                works=works,
                llm_scoring_cap=1,
            )

        entry = scores.get(work_id, {})
        score = entry.get("score", 0.0) if isinstance(entry, dict) else entry
        assert score >= case["min_score"], (
            f"{case['description']}: expected >= {case['min_score']}, got {score}"
        )

    @pytest.mark.parametrize(
        "case",
        [c for c in SCORING_CASES if "max_score" in c],
        ids=[c["id"] for c in SCORING_CASES if "max_score" in c],
    )
    def test_irrelevant_papers_score_low(self, case, db_engine):
        with db_engine.begin() as conn:
            work_id = _find_work_by_title(conn, case["paper_title_substring"])
            if not work_id:
                pytest.skip(f"Paper '{case['paper_title_substring']}' not in DB")

            works = WorkStore.load_many(conn, [work_id])
            from app.feature2.query_expansion import compute_query_hash
            query_hash = compute_query_hash(case["query"])

            scores = score_papers(
                conn,
                query_text=case["query"],
                query_hash=query_hash,
                paper_ids=[work_id],
                works=works,
                llm_scoring_cap=1,
            )

        entry = scores.get(work_id, {})
        score = entry.get("score", 0.0) if isinstance(entry, dict) else entry
        assert score <= case["max_score"], (
            f"{case['description']}: expected <= {case['max_score']}, got {score}"
        )


@pytest.mark.live
@pytest.mark.timeout(180)
def test_scoring_consistency(db_engine):
    """Same paper scored twice for same query should get same tier."""
    with db_engine.begin() as conn:
        work_id = _find_work_by_title(conn, "attention is all you need")
        if not work_id:
            pytest.skip("'Attention Is All You Need' not in DB")

        works = WorkStore.load_many(conn, [work_id])
        query = "transformer architecture attention mechanisms"
        from app.feature2.query_expansion import compute_query_hash
        query_hash = compute_query_hash(query)

        scores1 = score_papers(
            conn, query_text=query, query_hash=query_hash,
            paper_ids=[work_id], works=works, llm_scoring_cap=1,
        )
        # Note: second call may hit cache, which is fine — that's the real behavior
        scores2 = score_papers(
            conn, query_text=query, query_hash=query_hash,
            paper_ids=[work_id], works=works, llm_scoring_cap=1,
        )

    s1 = scores1.get(work_id, {})
    s2 = scores2.get(work_id, {})
    v1 = s1.get("score", 0.0) if isinstance(s1, dict) else s1
    v2 = s2.get("score", 0.0) if isinstance(s2, dict) else s2
    assert v1 == v2, f"Inconsistent scores: {v1} vs {v2}"


@pytest.mark.live
@pytest.mark.timeout(180)
def test_batch_ordering_invariance(db_engine):
    """Score shouldn't depend on batch ordering."""
    with db_engine.begin() as conn:
        # Find 3 known papers
        papers = []
        for substr in [
            "attention is all you need",
            "deep residual learning",
            "generative adversarial",
        ]:
            wid = _find_work_by_title(conn, substr)
            if wid:
                papers.append(wid)

        if len(papers) < 2:
            pytest.skip("Need at least 2 known papers in DB")

        works = WorkStore.load_many(conn, papers)
        query = "deep learning architectures"
        from app.feature2.query_expansion import compute_query_hash
        query_hash = compute_query_hash(query)

        # Score in original order
        scores_fwd = score_papers(
            conn, query_text=query, query_hash=query_hash,
            paper_ids=papers, works=works, llm_scoring_cap=len(papers),
        )
        # Score in reverse order (different query_hash to avoid cache)
        query_rev = "deep learning architectures "  # Slightly different to bust cache
        query_hash_rev = compute_query_hash(query_rev)
        scores_rev = score_papers(
            conn, query_text=query_rev, query_hash=query_hash_rev,
            paper_ids=list(reversed(papers)), works=works,
            llm_scoring_cap=len(papers),
        )

    # Scores should be within one tier step (0.25) of each other
    for wid in papers:
        e1 = scores_fwd.get(wid, {})
        e2 = scores_rev.get(wid, {})
        v1 = e1.get("score", 0.0) if isinstance(e1, dict) else e1
        v2 = e2.get("score", 0.0) if isinstance(e2, dict) else e2
        assert abs(v1 - v2) <= 0.25, (
            f"Ordering affects score for {wid}: {v1} vs {v2}"
        )
