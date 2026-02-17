"""
Batch ranking e2e tests — 17 new queries covering domain & type gaps.

Fills gaps identified in TESTED_QUERIES.md:
- New domains: Mathematics, Economics, Materials Science, Neuroscience,
  NLP/Linguistics, Energy, Astronomy, Epidemiology, Statistics,
  Information Retrieval, Engineering
- New query types: methodological, broad, niche, comparison

Hits the real /v1/rank endpoint. Burns API credits.
"""
from __future__ import annotations

import pytest

from tests.live.conftest import save_result, load_previous_result


# ---------------------------------------------------------------------------
# 17 new benchmark queries — filling coverage gaps
# ---------------------------------------------------------------------------
NEW_BATCH_QUERIES = [
    # --- New domains ---
    {
        "id": "number_theory",
        "query": "prime number distribution Riemann hypothesis analytic number theory",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "behavioral_economics",
        "query": "behavioral economics nudge theory decision making under uncertainty",
        "expected_papers": ["prospect theory"],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "superconductors",
        "query": "high temperature superconductivity cuprate materials mechanisms",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "neural_plasticity",
        "query": "synaptic plasticity long-term potentiation memory formation hippocampus",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "nlp_sentiment",
        "query": "sentiment analysis opinion mining natural language processing",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 6,
    },
    {
        "id": "solar_cells",
        "query": "perovskite solar cells efficiency photovoltaic materials",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "exoplanet_detection",
        "query": "exoplanet detection methods transit spectroscopy habitability",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "covid_epidemiology",
        "query": "COVID-19 epidemiological modeling SIR transmission dynamics",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    # --- New query types ---
    {
        "id": "optimization_methods",
        "query": "what optimization methods exist for training deep neural networks",
        "expected_papers": ["adam"],
        "min_foundational": 2,
        "min_total": 6,
    },
    {
        "id": "cnn_evolution",
        "query": "convolutional neural networks evolution from LeNet to modern architectures",
        "expected_papers": [],
        "min_foundational": 2,
        "min_total": 6,
    },
    {
        "id": "gan_vs_diffusion",
        "query": "generative adversarial networks versus diffusion models image synthesis comparison",
        "expected_papers": ["generative adversarial"],
        "min_foundational": 2,
        "min_total": 6,
    },
    {
        "id": "photosynthesis",
        "query": "photosynthesis",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "capsule_networks",
        "query": "capsule networks dynamic routing agreement Hinton",
        "expected_papers": ["capsule"],
        "min_foundational": 1,
        "min_total": 3,
    },
    # --- More diversity ---
    {
        "id": "bayesian_optimization",
        "query": "Bayesian optimization hyperparameter tuning surrogate models",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "transfer_learning",
        "query": "transfer learning domain adaptation deep neural networks",
        "expected_papers": [],
        "min_foundational": 2,
        "min_total": 6,
    },
    {
        "id": "autonomous_driving",
        "query": "autonomous driving perception lidar point cloud deep learning",
        "expected_papers": [],
        "min_foundational": 1,
        "min_total": 5,
    },
    {
        "id": "recommender_systems",
        "query": "collaborative filtering recommender systems matrix factorization",
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
    NEW_BATCH_QUERIES,
    ids=[c["id"] for c in NEW_BATCH_QUERIES],
)
def test_new_batch_ranking(case, live_client, auth_headers, results_dir):
    """E2E ranking benchmark for new batch queries."""
    response = live_client.post(
        "/v1/rank",
        json={"query_text": case["query"]},
        headers=auth_headers,
        timeout=240,
    )

    if response.status_code == 202:
        pytest.skip(f"Rank job for '{case['id']}' returned 202 (async)")

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
        "all_items": [
            {
                "rank": i + 1,
                "title": (item.get("preview", {}).get("title") or "")[:120],
                "year": item.get("preview", {}).get("year"),
                "score": item.get("score", 0),
            }
            for i, item in enumerate(all_items)
        ],
        "top_10_titles": titles[:10],
        "top_10_scores": [item.get("score", 0) for item in all_items[:10]],
        "query_classification": result.get("query_classification"),
        "rank_job_id": str(result.get("rank_job_id", "")),
    }
    save_result(results_dir, f"new_batch_{case['id']}", summary)

    # --- Regression detection ---
    previous = load_previous_result(results_dir, f"new_batch_{case['id']}")
    if previous and previous.get("total_items", 0) > 0:
        prev_total = previous["total_items"]
        if len(all_items) < prev_total * 0.5:
            import warnings
            warnings.warn(
                f"Regression: {case['id']} dropped from {prev_total} to {len(all_items)} items",
                UserWarning,
            )

    # --- Assertions ---

    # 1. Minimum total results
    assert len(all_items) >= case["min_total"], (
        f"[{case['id']}] Expected >= {case['min_total']} total items, got {len(all_items)}"
    )

    # 2. Minimum foundational papers
    foundational = result.get("foundational", [])
    if foundational is not None and case["min_foundational"] > 0:
        assert len(foundational) >= case["min_foundational"], (
            f"[{case['id']}] Expected >= {case['min_foundational']} foundational, got {len(foundational)}"
        )

    # 3. Expected papers (fuzzy title match)
    for expected in case["expected_papers"]:
        found = any(expected.lower() in t for t in titles)
        assert found, (
            f"[{case['id']}] Expected paper containing '{expected}' not found. "
            f"Got titles: {titles[:10]}"
        )

    # 4. Score sanity
    if all_items:
        top_score = all_items[0].get("score", 0)
        assert top_score > 0, f"[{case['id']}] Top item has score {top_score}"

    # 5. Category distribution
    non_empty_cats = sum(
        1 for cat in ("foundational", "methodology", "reviews", "applications")
        if len(result.get(cat, [])) > 0
    )
    if len(all_items) >= 10:
        assert non_empty_cats >= 2, (
            f"[{case['id']}] Expected >= 2 non-empty categories, got {non_empty_cats}"
        )
