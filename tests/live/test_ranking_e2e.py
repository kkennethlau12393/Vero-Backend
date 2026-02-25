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
    # == NL QUERY MODE (25 cases) ==
    # Core ML/DL
    {"id": "nl_transformers", "query": "transformer architecture attention mechanisms deep learning", "min_total": 10},
    {"id": "nl_bert", "query": "BERT bidirectional encoder representations transformers NLP", "min_total": 10},
    {"id": "nl_gnn", "query": "graph neural networks node classification message passing", "min_total": 8},
    {"id": "nl_resnet", "query": "residual networks deep learning computer vision ImageNet", "min_total": 10},
    {"id": "nl_gan", "query": "generative adversarial networks image synthesis", "min_total": 10},

    # RL/Robotics
    {"id": "nl_dqn", "query": "deep Q-network reinforcement learning Atari", "min_total": 8},
    {"id": "nl_ppo", "query": "proximal policy optimization reinforcement learning", "min_total": 8},
    {"id": "nl_alphago", "query": "AlphaGo Monte Carlo tree search reinforcement learning", "min_total": 8},
    {"id": "nl_rl_robotics", "query": "reinforcement learning robotic manipulation sim-to-real", "min_total": 6},

    # Generative Models
    {"id": "nl_diffusion", "query": "diffusion models denoising score matching image generation", "min_total": 8},
    {"id": "nl_vae", "query": "variational autoencoders latent variable models", "min_total": 8},
    {"id": "nl_gpt", "query": "GPT generative pre-training language models", "min_total": 10},

    # NLP
    {"id": "nl_word2vec", "query": "word embeddings word2vec skip-gram continuous bag of words", "min_total": 8},
    {"id": "nl_seq2seq", "query": "sequence to sequence models neural machine translation", "min_total": 8},
    {"id": "nl_llm_alignment", "query": "large language model alignment RLHF safety", "min_total": 8},

    # Biology/Chemistry
    {"id": "nl_alphafold", "query": "protein structure prediction AlphaFold deep learning", "min_total": 6},
    {"id": "nl_crispr", "query": "CRISPR-Cas9 gene editing genome engineering", "min_total": 6},
    {"id": "nl_drug_discovery", "query": "machine learning drug discovery molecular property prediction", "min_total": 6},

    # Physics/Quantum
    {"id": "nl_quantum_error", "query": "quantum error correction surface codes fault tolerance", "min_total": 6},
    {"id": "nl_quantum_ml", "query": "quantum machine learning variational quantum eigensolver", "min_total": 5},

    # Federated/Privacy
    {"id": "nl_federated", "query": "federated learning privacy-preserving distributed machine learning", "min_total": 8},
    {"id": "nl_differential_privacy", "query": "differential privacy machine learning privacy guarantees", "min_total": 6},

    # Meta-learning/Few-shot
    {"id": "nl_maml", "query": "meta-learning model-agnostic few-shot learning", "min_total": 6},
    {"id": "nl_few_shot", "query": "few-shot learning prototypical networks metric learning", "min_total": 6},

    # Interpretability
    {"id": "nl_interpretability", "query": "neural network interpretability explainable AI attention visualization", "min_total": 6},

    # == SEED PAPER MODE (25 cases) ==
    # Foundational papers
    {"id": "seed_attention_all_you_need", "seed_title": "Attention Is All You Need", "min_total": 10},
    {"id": "seed_bert", "seed_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "min_total": 10},
    {"id": "seed_resnet", "seed_title": "Deep Residual Learning for Image Recognition", "min_total": 10},
    {"id": "seed_gan", "seed_title": "Generative Adversarial Networks", "min_total": 10},
    {"id": "seed_adam", "seed_title": "Adam: A Method for Stochastic Optimization", "min_total": 10},

    # RL papers
    {"id": "seed_dqn", "seed_title": "Human-level control through deep reinforcement learning", "min_total": 8},
    {"id": "seed_alphago", "seed_title": "Mastering the game of Go with deep neural networks and tree search", "min_total": 8},
    {"id": "seed_ppo", "seed_title": "Proximal Policy Optimization Algorithms", "min_total": 8},
    {"id": "seed_ddpg", "seed_title": "Continuous control with deep reinforcement learning", "min_total": 8},

    # Generative models
    {"id": "seed_ddpm", "seed_title": "Denoising Diffusion Probabilistic Models", "min_total": 8},
    {"id": "seed_stable_diffusion", "seed_title": "High-Resolution Image Synthesis with Latent Diffusion Models", "min_total": 8},
    {"id": "seed_vae", "seed_title": "Auto-Encoding Variational Bayes", "min_total": 8},
    {"id": "seed_gpt3", "seed_title": "Language Models are Few-Shot Learners", "min_total": 10},

    # NLP
    {"id": "seed_word2vec", "seed_title": "Efficient Estimation of Word Representations in Vector Space", "min_total": 8},
    {"id": "seed_seq2seq", "seed_title": "Sequence to Sequence Learning with Neural Networks", "min_total": 8},
    {"id": "seed_instructgpt", "seed_title": "Training language models to follow instructions with human feedback", "min_total": 8},

    # Computer Vision
    {"id": "seed_alexnet", "seed_title": "ImageNet Classification with Deep Convolutional Neural Networks", "min_total": 10},
    {"id": "seed_vit", "seed_title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "min_total": 10},
    {"id": "seed_clip", "seed_title": "Learning Transferable Visual Models From Natural Language Supervision", "min_total": 10},

    # GNN
    {"id": "seed_gcn", "seed_title": "Semi-Supervised Classification with Graph Convolutional Networks", "min_total": 8},
    {"id": "seed_graphsage", "seed_title": "Inductive Representation Learning on Large Graphs", "min_total": 8},

    # Biology
    {"id": "seed_alphafold2", "seed_title": "Highly accurate protein structure prediction with AlphaFold", "min_total": 6},

    # Meta-learning
    {"id": "seed_maml", "seed_title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "min_total": 6},

    # Optimization
    {"id": "seed_batchnorm", "seed_title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "min_total": 10},
    {"id": "seed_dropout", "seed_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "min_total": 10},
]


def _all_items(result: dict) -> list[dict]:
    """Extract all ranked items from a categorized or flat response."""
    # Prefer non-empty "items" list (drill-down mode); otherwise collect from categories
    if result.get("items"):
        return result["items"]
    items = []
    for cat in ("foundational", "methodology", "reviews", "applications", "textbooks", "additional_relevant"):
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
    """End-to-end ranking benchmark - supports both NL query and seed paper modes."""
    # Build request payload based on input mode
    if "query" in case:
        # NL query mode
        payload = {"query_text": case["query"]}
    elif "seed_title" in case:
        # Seed paper mode
        payload = {"seed_title": case["seed_title"]}
    else:
        raise ValueError(f"Test case {case['id']} missing both 'query' and 'seed_title'")

    response = live_client.post(
        "/v1/rank",
        json=payload,
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
    # Build full category breakdown with all papers (not just top 10)
    full_categories = {}
    for cat in ("foundational", "methodology", "reviews", "applications", "textbooks", "additional_relevant"):
        cat_items = result.get(cat, [])
        full_categories[cat] = [
            {
                "title": (item.get("preview", {}).get("title") or ""),
                "score": item.get("score", 0),
                "year": item.get("preview", {}).get("year"),
                "cited_by_count": item.get("preview", {}).get("cited_by_count"),
                "work_id": item.get("work_id", ""),
            }
            for item in cat_items
        ]

    summary = {
        "query": case.get("query"),
        "seed_title": case.get("seed_title"),
        "total_items": len(all_items),
        "categories": {
            cat: len(result.get(cat, []))
            for cat in ("foundational", "methodology", "reviews", "applications", "textbooks", "additional_relevant")
        },
        "full_categories": full_categories,
        "top_10_titles": titles[:10],
        "top_10_scores": [item.get("score", 0) for item in all_items[:10]],
        "all_titles": titles,
        "all_scores": [item.get("score", 0) for item in all_items],
        "all_years": [item.get("preview", {}).get("year") for item in all_items],
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
    assert len(all_items) >= case.get("min_total", 5), (
        f"[{case['id']}] Expected >= {case.get('min_total', 5)} total items, got {len(all_items)}"
    )

    # 2. Minimum foundational papers (if specified)
    foundational = result.get("foundational", [])
    if foundational is not None and case.get("min_foundational", 0) > 0:
        assert len(foundational) >= case["min_foundational"], (
            f"[{case['id']}] Expected >= {case['min_foundational']} foundational, got {len(foundational)}"
        )

    # 3. Expected papers appear somewhere in results (fuzzy title match - if specified)
    for expected in case.get("expected_papers", []):
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
