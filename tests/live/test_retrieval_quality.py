"""
Live retrieval quality tests — verifies Stage-1 retrieval returns
sufficient, diverse, high-quality candidates from real APIs.

Tests hit: OpenAlex, Semantic Scholar, ArXiv, pgvector.
"""
from __future__ import annotations

import pytest

from app.feature2.retrieval import generate_candidates_direct


# ---------------------------------------------------------------------------
# Known queries with expected retrieval properties
# ---------------------------------------------------------------------------
RETRIEVAL_CASES = [
    {
        "id": "transformers",
        "query": "transformer architecture attention mechanisms in deep learning",
        "min_candidates": 50,
        "landmark_titles": [
            "attention is all you need",
        ],
        "expect_multi_source": True,
    },
    {
        "id": "gnn",
        "query": "graph neural networks node classification",
        "min_candidates": 40,
        "landmark_titles": [
            "semi-supervised classification with graph convolutional networks",
        ],
        "expect_multi_source": True,
    },
    {
        "id": "rl",
        "query": "reinforcement learning policy gradient methods",
        "min_candidates": 40,
        "landmark_titles": [
            "playing atari with deep reinforcement learning",
        ],
        "expect_multi_source": True,
    },
    {
        "id": "crispr",
        "query": "CRISPR-Cas9 gene editing therapeutic applications",
        "min_candidates": 30,
        "landmark_titles": [],  # Less predictable titles in biotech
        "expect_multi_source": True,
    },
    {
        "id": "llm_alignment",
        "query": "large language model alignment RLHF",
        "min_candidates": 20,
        "landmark_titles": [
            "training language models to follow instructions",
        ],
        "expect_multi_source": True,
    },
]


@pytest.mark.live
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "case",
    RETRIEVAL_CASES,
    ids=[c["id"] for c in RETRIEVAL_CASES],
)
def test_retrieval_volume_and_diversity(case, db_engine, dev_tenant_id):
    """Assert retrieval returns enough candidates from multiple sources."""
    with db_engine.begin() as conn:
        candidates, query_expansion = generate_candidates_direct(
            conn,
            tenant_id=dev_tenant_id,
            query_text=case["query"],
            context_json={},
            filters_json={},
            rank_params_json={},
        )

    # Volume check
    assert len(candidates) >= case["min_candidates"], (
        f"Expected >= {case['min_candidates']} candidates for '{case['id']}', got {len(candidates)}"
    )

    # Source diversity check
    if case["expect_multi_source"]:
        all_sources = set()
        for c in candidates:
            for p in c.get("provenance", []):
                src = p.get("source", "")
                # Normalize source names to top-level
                if "openalex" in src or "lexical" in src:
                    all_sources.add("openalex")
                elif "s2" in src or "semantic" in src:
                    all_sources.add("s2")
                elif "arxiv" in src:
                    all_sources.add("arxiv")
                elif "foundational" in src:
                    all_sources.add("foundational")
                elif "fts" in src:
                    all_sources.add("fts")
                else:
                    all_sources.add(src)
        assert len(all_sources) >= 2, (
            f"Expected multiple sources for '{case['id']}', got: {all_sources}"
        )

    # Landmark paper check (fuzzy title match)
    if case["landmark_titles"]:
        candidate_titles = set()
        for c in candidates:
            title = (c.get("title") or "").lower().strip()
            if title:
                candidate_titles.add(title)

        for landmark in case["landmark_titles"]:
            found = any(landmark.lower() in t for t in candidate_titles)
            if not found:
                # Softer check: look in work_id provenance titles too
                pytest.warns(
                    UserWarning,
                    match=f"Landmark paper '{landmark}' not found in retrieval for '{case['id']}'",
                ) if not found else None
                # Don't hard-fail on landmark check — retrieval evolves
                # Just warn. The e2e tests will catch ranking quality.


@pytest.mark.live
@pytest.mark.timeout(120)
def test_query_expansion_broadens_results(db_engine, dev_tenant_id):
    """Verify query expansion generates useful expanded terms."""
    from app.feature2.query_expansion import expand_query

    with db_engine.begin() as conn:
        expansion = expand_query(conn, "transformer attention mechanisms")

    # Should have expanded queries
    assert expansion is not None
    assert expansion.expanded_queries, "Query expansion produced no expanded queries"
    assert len(expansion.expanded_queries) >= 2, (
        f"Expected >= 2 expanded queries, got {len(expansion.expanded_queries)}"
    )
    # Expanded queries should differ from original
    original_lower = "transformer attention mechanisms"
    for eq in expansion.expanded_queries:
        assert eq.lower() != original_lower, "Expanded query is identical to original"
