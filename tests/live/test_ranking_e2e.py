"""
End-to-end ranking quality tests — formalized benchmarks.

Hits the real /v1/rank endpoint with known queries, asserts:
- Known papers appear in results
- Category distribution is reasonable
- Score thresholds met
- Results saved for manual review + regression detection

Derived from the former ranking_benchmark*.py scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.live.conftest import save_result, load_previous_result


# ---------------------------------------------------------------------------
# Benchmark queries — parametrized test cases
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    # -- Core CS/ML (from ranking_benchmark.py) --
    {
        "id": "transformers",
        "query": "transformer architecture attention mechanisms in deep learning",
        "expected_papers": [
            "attention is all you need",
            "bert",
        ],
        "min_foundational": 3,
        "min_total": 10,
    },
    {
        "id": "graph_neural_networks",
        "query": "graph neural networks node classification",
        "expected_papers": [
            "graph convolutional",
            "inductive representation learning",  # GraphSAGE paper
        ],
        "min_foundational": 2,
        "min_total": 8,
    },
    {
        "id": "reinforcement_learning",
        "query": "reinforcement learning policy gradient methods",
        "expected_papers": [
            "policy gradient",
        ],
        "min_foundational": 2,
        "min_total": 8,
    },
    # -- Diverse domains (from ranking_benchmark_diverse.py) --
    {
        "id": "crispr_gene_editing",
        "query": "CRISPR-Cas9 gene editing therapeutic applications",
        "expected_papers": [
            "crispr",
        ],
        "min_foundational": 2,
        "min_total": 8,
    },
    {
        "id": "quantum_computing",
        "query": "quantum computing error correction fault tolerance",
        "expected_papers": [
            "quantum error",
        ],
        "min_foundational": 2,
        "min_total": 6,
    },
    {
        "id": "climate_modeling",
        "query": "climate change modeling global circulation models projections",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 6,
    },
    # -- Diverse 2 (from ranking_benchmark_diverse2.py) --
    {
        "id": "federated_learning",
        "query": "federated learning privacy-preserving machine learning",
        "expected_papers": [
            "federated",
        ],
        "min_foundational": 2,
        "min_total": 8,
    },
    {
        "id": "diffusion_models",
        "query": "diffusion models denoising score matching image generation",
        "expected_papers": [
            "denoising diffusion",
        ],
        "min_foundational": 2,
        "min_total": 8,
    },
    # -- Diverse 3 (from ranking_benchmark_diverse3.py) --
    {
        "id": "protein_folding",
        "query": "protein structure prediction deep learning AlphaFold",
        "expected_papers": [
            "alphafold",
        ],
        "min_foundational": 1,
        "min_total": 6,
    },
    {
        "id": "llm_alignment",
        "query": "large language model alignment RLHF safety",
        "expected_papers": [
            "large language model",
        ],
        "min_foundational": 1,
        "min_total": 6,
    },
    # -- Intersection / specific queries (from ranking_benchmark_failing3.py) --
    {
        "id": "attention_medical_imaging",
        "query": "attention mechanisms in medical image segmentation",
        "expected_papers": [],
        "min_foundational": 0,
        "min_total": 5,
    },
    {
        "id": "gnn_drug_discovery",
        "query": "graph neural networks for drug discovery molecular property prediction",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "rl_robotics",
        "query": "reinforcement learning sim-to-real transfer robotic manipulation",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
]


def _all_items(result: dict) -> list[dict]:
    """Extract all ranked items from a categorized or flat response."""
    # Prefer non-empty "items" list (drill-down mode); otherwise collect from categories
    if result.get("items"):
        return result["items"]
    items = []
    for cat in ("foundational", "methodology", "reviews", "applications", "textbooks"):
        items.extend(result.get(cat, []))
    return items


def _titles_lower(items: list[dict]) -> list[str]:
    """Extract lowered titles from result items."""
    titles = []
    for item in items:
        title = (item.get("preview", {}).get("title") or "").lower()
        if title:
            titles.append(title)
    return titles


@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_QUERIES,
    ids=[c["id"] for c in BENCHMARK_QUERIES],
)
def test_ranking_quality(case, live_client, auth_headers, results_dir):
    """End-to-end ranking benchmark for a single query."""
    response = live_client.post(
        "/v1/rank",
        json={"query_text": case["query"]},
        headers=auth_headers,
        timeout=240,
    )

    # Accept 200 (completed) or 202 (async — skip if async)
    if response.status_code == 202:
        pytest.skip(f"Rank job for '{case['id']}' returned 202 (async), re-run to get cached result")

    assert response.status_code == 200, (
        f"Expected 200 for '{case['id']}', got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    all_items = _all_items(result)
    titles = _titles_lower(all_items)

    # --- Save results FIRST (before assertions, so we never lose data) ---
    summary = {
        "query": case["query"],
        "total_items": len(all_items),
        "categories": {
            cat: len(result.get(cat, []))
            for cat in ("foundational", "methodology", "reviews", "applications", "textbooks")
        },
        "top_10_titles": titles[:10],
        "top_10_scores": [item.get("score", 0) for item in all_items[:10]],
        "query_classification": result.get("query_classification"),
        "rank_job_id": str(result.get("rank_job_id", "")),
    }
    save_result(results_dir, f"e2e_{case['id']}", summary)

    # --- Regression detection ---
    previous = load_previous_result(results_dir, f"e2e_{case['id']}")
    if previous and previous.get("total_items", 0) > 0:
        prev_total = previous["total_items"]
        if len(all_items) < prev_total * 0.5:
            pytest.warns(
                UserWarning,
                match=f"Regression: {case['id']} dropped from {prev_total} to {len(all_items)} items",
            )

    # --- Assertions ---

    # 1. Minimum total results
    assert len(all_items) >= case["min_total"], (
        f"[{case['id']}] Expected >= {case['min_total']} total items, got {len(all_items)}"
    )

    # 2. Minimum foundational papers (if categorized)
    foundational = result.get("foundational", [])
    if foundational is not None and case["min_foundational"] > 0:
        assert len(foundational) >= case["min_foundational"], (
            f"[{case['id']}] Expected >= {case['min_foundational']} foundational, got {len(foundational)}"
        )

    # 3. Expected papers appear somewhere in results (fuzzy title match)
    for expected in case["expected_papers"]:
        found = any(expected.lower() in t for t in titles)
        assert found, (
            f"[{case['id']}] Expected paper containing '{expected}' not found in results. "
            f"Got titles: {titles[:10]}"
        )

    # 4. Score sanity: top items should have score > 0
    if all_items:
        top_score = all_items[0].get("score", 0)
        assert top_score > 0, f"[{case['id']}] Top item has score {top_score}"

    # 5. Category distribution: should have at least 2 non-empty categories
    non_empty_cats = sum(
        1 for cat in ("foundational", "methodology", "reviews", "applications")
        if len(result.get(cat, [])) > 0
    )
    if len(all_items) >= 10:
        assert non_empty_cats >= 2, (
            f"[{case['id']}] Expected >= 2 non-empty categories, got {non_empty_cats}"
        )


@pytest.mark.live
@pytest.mark.timeout(300)
def test_drill_down_endpoint(live_client, auth_headers, results_dir):
    """Test the /v1/rank/drill-down lightweight endpoint."""
    response = live_client.post(
        "/v1/rank/drill-down",
        json={"query_text": "vision transformers image classification", "top_k": 5},
        headers=auth_headers,
        timeout=180,
    )

    if response.status_code == 202:
        pytest.skip("Drill-down returned 202, re-run for cached")

    assert response.status_code == 200
    result = response.json()

    # Drill-down returns flat "items" list
    items = result.get("items", [])
    assert len(items) >= 3, f"Expected >= 3 drill-down items, got {len(items)}"
    assert len(items) <= 10, f"Drill-down should be compact, got {len(items)}"

    save_result(results_dir, "drill_down_vit", {
        "total": len(items),
        "titles": [(item.get("preview", {}).get("title") or "")[:80] for item in items],
        "scores": [item.get("score", 0) for item in items],
    })
