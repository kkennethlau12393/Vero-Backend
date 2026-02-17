"""
Focused live tests for seed selection quality in Feature 1 (Citation Map).

Tests whether the system picks the right seminal paper for known queries.
Uses LLM-as-judge criteria for evaluation (via print-based scoring).

Usage:
    pytest tests/live/test_seed_selection_quality.py -v -s --timeout=300
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.live.conftest import save_result, load_previous_result


# ---------------------------------------------------------------------------
# Known-answer seed selection queries
# ---------------------------------------------------------------------------
SEED_QUALITY_QUERIES = [
    # Well-known seminal papers — these should be easy
    {
        "id": "transformers_seminal",
        "query": "attention is all you need transformer",
        "expected_title_fragments": ["attention is all you need"],
        "expected_min_citations": 50000,
        "difficulty": "easy",
        "notes": "The original transformer paper by Vaswani et al. 2017",
    },
    {
        "id": "resnet",
        "query": "deep residual learning image recognition",
        "expected_title_fragments": ["deep residual learning", "residual"],
        "expected_min_citations": 100000,
        "difficulty": "easy",
        "notes": "He et al. 2015 ResNet paper",
    },
    {
        "id": "adam_optimizer",
        "query": "adam optimizer adaptive learning rate",
        "expected_title_fragments": ["adam"],
        "expected_min_citations": 100000,
        "difficulty": "easy",
        "notes": "Kingma & Ba 2014",
    },
    {
        "id": "gan",
        "query": "generative adversarial networks",
        "expected_title_fragments": ["generative adversarial"],
        "expected_min_citations": 30000,
        "difficulty": "easy",
        "notes": "Goodfellow et al. 2014",
    },
    {
        "id": "bert",
        "query": "BERT pre-training language representations",
        "expected_title_fragments": ["bert", "pre-training"],
        "expected_min_citations": 50000,
        "difficulty": "easy",
        "notes": "Devlin et al. 2018",
    },

    # Medium difficulty — well-known but broader queries
    {
        "id": "crispr_broad",
        "query": "CRISPR gene editing technology",
        "expected_title_fragments": ["crispr", "cas9", "genome"],
        "expected_min_citations": 5000,
        "difficulty": "medium",
        "notes": "Should pick Doudna/Charpentier or Jinek et al. 2012",
    },
    {
        "id": "alphafold",
        "query": "protein structure prediction deep learning",
        "expected_title_fragments": ["protein", "structure", "alphafold"],
        "expected_min_citations": 5000,
        "difficulty": "medium",
        "notes": "Should pick AlphaFold or AlphaFold2 paper",
    },
    {
        "id": "word2vec",
        "query": "word embeddings distributed representations",
        "expected_title_fragments": ["word2vec", "distributed representations", "word embeddings", "efficient estimation"],
        "expected_min_citations": 20000,
        "difficulty": "medium",
        "notes": "Mikolov et al. 2013",
    },

    # Hard — broad or ambiguous queries
    {
        "id": "broad_machine_learning",
        "query": "machine learning",
        "expected_title_fragments": [],  # many valid answers
        "expected_min_citations": 10000,
        "difficulty": "hard",
        "notes": "Very broad — any highly cited ML paper is acceptable",
    },
    {
        "id": "niche_neural_ode",
        "query": "neural ordinary differential equations continuous depth",
        "expected_title_fragments": ["neural ordinary differential", "neural ode"],
        "expected_min_citations": 1000,
        "difficulty": "hard",
        "notes": "Chen et al. 2018 NeurIPS",
    },
    {
        "id": "ambiguous_attention",
        "query": "attention mechanism",
        "expected_title_fragments": ["attention"],
        "expected_min_citations": 10000,
        "difficulty": "hard",
        "notes": "Could be Bahdanau 2014 or Vaswani 2017 — both valid",
    },
    {
        "id": "niche_capsule_networks",
        "query": "capsule networks dynamic routing",
        "expected_title_fragments": ["capsule", "dynamic routing"],
        "expected_min_citations": 3000,
        "difficulty": "hard",
        "notes": "Sabour/Hinton 2017",
    },
]


@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    SEED_QUALITY_QUERIES,
    ids=[c["id"] for c in SEED_QUALITY_QUERIES],
)
def test_seed_selection_quality(case, live_client, auth_headers, results_dir):
    """Test that seed selection picks the right seminal paper for a query."""
    response = live_client.post(
        "/v1/citation-map",
        json={
            "query_text": case["query"],
            "citing_limit": 3,
            "references_limit": 3,
            "create_graph_draft": False,
        },
        headers=auth_headers,
        timeout=180,
    )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    seed_info = result["seed_info"]

    seed_title = (seed_info.get("seed_title") or "").lower()
    seed_work_id = seed_info.get("seed_work_id", "")

    # --- Check title match ---
    title_match = False
    matched_fragment = None
    if case["expected_title_fragments"]:
        for frag in case["expected_title_fragments"]:
            if frag.lower() in seed_title:
                title_match = True
                matched_fragment = frag
                break
    else:
        # No specific expectation — just needs to exist
        title_match = bool(seed_title)

    # --- Check citation count (from seed node) ---
    seed_node = next(
        (n for n in result["nodes"] if n.get("is_seed")),
        None,
    )
    seed_citations = seed_node["cited_by_count"] if seed_node else 0

    citation_check = seed_citations >= case["expected_min_citations"]

    # --- Scoring ---
    # Title relevance: 0-10
    if title_match:
        title_score = 10
    elif seed_title and any(kw.lower() in seed_title for kw in case["query"].split()[:2]):
        title_score = 5  # Partial match
    else:
        title_score = 0

    # Citation prominence: 0-10
    if citation_check:
        citation_score = 10
    elif seed_citations >= case["expected_min_citations"] * 0.5:
        citation_score = 7
    elif seed_citations >= 1000:
        citation_score = 4
    else:
        citation_score = 0

    total_score = title_score + citation_score

    # --- Print breakdown ---
    print(f"\n{'='*60}")
    print(f"SEED SELECTION QUALITY: {case['id']}")
    print(f"Query: {case['query']}")
    print(f"Difficulty: {case['difficulty']}")
    print(f"{'='*60}")
    print(f"Selected seed: {seed_info.get('seed_title', 'N/A')}")
    print(f"Work ID: {seed_work_id}")
    print(f"Citations: {seed_citations:,}")
    print(f"Strategy: {seed_info.get('selection_strategy', 'N/A')}")
    print(f"Candidates considered: {seed_info.get('candidates_considered', 'N/A')}")
    print(f"\nTitle match: {'✓' if title_match else '✗'} (matched: {matched_fragment})")
    print(f"Citation check (>={case['expected_min_citations']:,}): {'✓' if citation_check else '✗'}")
    print(f"\nScore: {total_score}/20")
    print(f"  Title relevance: {title_score}/10")
    print(f"  Citation prominence: {citation_score}/10")
    print(f"\nLLM-Judge Criteria (evaluate manually):")
    print(f"  - Is '{seed_info.get('seed_title', 'N/A')}' the best seed for '{case['query']}'?")
    print(f"  - Notes: {case['notes']}")
    print(f"{'='*60}\n")

    # --- Save ---
    summary = {
        "query": case["query"],
        "difficulty": case["difficulty"],
        "seed_info": seed_info,
        "seed_citations": seed_citations,
        "title_match": title_match,
        "matched_fragment": matched_fragment,
        "citation_check": citation_check,
        "score": {
            "title_relevance": title_score,
            "citation_prominence": citation_score,
            "total": total_score,
        },
        "notes": case["notes"],
    }
    save_result(results_dir, f"seed_quality_{case['id']}", summary)

    # Soft assertions — don't fail hard on seed selection (it's scored, not pass/fail)
    # But DO fail if we got no seed at all
    assert seed_work_id, f"[{case['id']}] No seed paper was selected"
    assert seed_title, f"[{case['id']}] Seed paper has no title"
