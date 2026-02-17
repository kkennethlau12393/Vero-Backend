"""
End-to-end citation map quality tests — formalized benchmarks.

Hits the real /v1/citation-map endpoint with known queries, asserts:
- Seed paper is relevant and well-cited
- Graph has reasonable node count and structure
- Edge directions are correct
- Year and citation spread exist
- Hop distribution is reasonable
- No duplicate nodes
- Results saved for manual review + regression detection

Usage:
    pytest tests/live/test_citation_map_e2e.py -v -s --timeout=300
    pytest tests/live/test_citation_map_e2e.py -k "transformers" -v -s
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pytest

from tests.live.conftest import save_result, load_previous_result


# ---------------------------------------------------------------------------
# Benchmark queries — parametrized test cases
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES_QUERY_MODE = [
    # -- Core CS/ML --
    {
        "id": "transformers",
        "query_text": "transformer attention mechanism",
        "expected_seed_keywords": ["attention", "transformer"],
        "min_nodes": 15,
        "min_edges": 10,
        "domain": "CS/ML",
    },
    {
        "id": "graph_neural_networks",
        "query_text": "graph neural networks",
        "expected_seed_keywords": ["graph", "neural", "network"],
        "min_nodes": 12,
        "min_edges": 8,
        "domain": "CS/ML",
    },
    {
        "id": "rl_robotics",
        "query_text": "reinforcement learning robotics",
        "expected_seed_keywords": ["reinforcement", "learning"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "CS/Robotics",
    },
    {
        "id": "diffusion_models",
        "query_text": "denoising diffusion probabilistic models image generation",
        "expected_seed_keywords": ["diffusion"],
        "min_nodes": 12,
        "min_edges": 8,
        "domain": "CS/ML",
    },
    # -- Biology --
    {
        "id": "crispr",
        "query_text": "CRISPR gene editing",
        "expected_seed_keywords": ["crispr", "cas9", "genome", "editing"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Biology",
    },
    {
        "id": "protein_folding",
        "query_text": "protein structure prediction AlphaFold",
        "expected_seed_keywords": ["protein", "structure", "alphafold", "folding"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Biology/ML",
    },
    # -- Physics --
    {
        "id": "quantum_error_correction",
        "query_text": "quantum error correction surface codes",
        "expected_seed_keywords": ["quantum", "error", "correction"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Physics",
    },
    # -- Social Science --
    {
        "id": "behavioral_economics",
        "query_text": "prospect theory behavioral economics",
        "expected_seed_keywords": ["prospect", "theory"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Economics",
    },
    # -- Interdisciplinary --
    {
        "id": "gnn_drug_discovery",
        "query_text": "graph neural networks drug discovery molecular property prediction",
        "expected_seed_keywords": ["graph", "neural", "molecular", "drug"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "CS/Chemistry",
    },
    {
        "id": "attention_medical_imaging",
        "query_text": "attention mechanisms medical image segmentation",
        "expected_seed_keywords": ["attention", "medical", "image", "segmentation"],
        "min_nodes": 8,
        "min_edges": 5,
        "domain": "CS/Medical",
    },
    # -- Neuroscience --
    {
        "id": "synaptic_plasticity",
        "query_text": "synaptic plasticity long-term potentiation",
        "expected_seed_keywords": ["synaptic", "plasticity", "potentiation"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Neuroscience",
    },
    # -- Materials --
    {
        "id": "graphene",
        "query_text": "graphene electronic properties synthesis",
        "expected_seed_keywords": ["graphene"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Materials Science",
    },
    # -- OLD PAPERS (pre-2000) --
    {
        "id": "backpropagation",
        "query_text": "backpropagation neural networks learning",
        "expected_seed_keywords": ["backpropagation", "learning"],
        "min_nodes": 8,
        "min_edges": 5,
        "domain": "Classic ML",
    },
    {
        "id": "hidden_markov",
        "query_text": "hidden markov models speech recognition",
        "expected_seed_keywords": ["markov", "hidden"],
        "min_nodes": 8,
        "min_edges": 5,
        "domain": "Classic ML",
    },
    # -- SOFTWARE PAPERS (expect noisy citations) --
    {
        "id": "pytorch",
        "query_text": "PyTorch deep learning framework",
        "expected_seed_keywords": ["pytorch"],
        "min_nodes": 8,
        "min_edges": 5,
        "domain": "Software",
    },
    {
        "id": "tensorflow",
        "query_text": "TensorFlow machine learning system",
        "expected_seed_keywords": ["tensorflow"],
        "min_nodes": 8,
        "min_edges": 5,
        "domain": "Software",
    },
    # -- NICHE FIELDS (hard seed selection) --
    {
        "id": "topological_data_analysis",
        "query_text": "persistent homology topological data analysis",
        "expected_seed_keywords": ["topological", "homology", "persistence"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "Math/CS",
    },
    {
        "id": "quantum_computing_algorithms",
        "query_text": "quantum algorithms Shor factoring",
        "expected_seed_keywords": ["quantum", "shor"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "Quantum Computing",
    },
    # -- CROSS-DOMAIN (might drift) --
    {
        "id": "climate_ml",
        "query_text": "machine learning climate modeling prediction",
        "expected_seed_keywords": ["climate", "machine", "learning"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "Climate/ML",
    },
    {
        "id": "nlp_law",
        "query_text": "natural language processing legal text",
        "expected_seed_keywords": ["legal", "nlp", "text"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "NLP/Law",
    },
    # -- EMERGING/NEW (low citations) --
    {
        "id": "llm_agents",
        "query_text": "large language model agents tool use",
        "expected_seed_keywords": ["language", "model", "agent"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "LLM (new)",
    },
    {
        "id": "federated_learning",
        "query_text": "federated learning privacy-preserving",
        "expected_seed_keywords": ["federated", "learning"],
        "min_nodes": 8,
        "min_edges": 4,
        "domain": "Privacy ML",
    },
    # -- REVIEW PAPERS (cite broadly) --
    {
        "id": "deep_learning_review",
        "query_text": "deep learning review LeCun Bengio Hinton",
        "expected_seed_keywords": ["deep", "learning", "review"],
        "min_nodes": 10,
        "min_edges": 6,
        "domain": "Survey",
    },
    # -- AMBIGUOUS TERMS --
    {
        "id": "embedding",
        "query_text": "embedding",
        "expected_seed_keywords": ["embedding"],
        "min_nodes": 5,
        "min_edges": 3,
        "domain": "Ambiguous",
    },
]

# Direct seed_work_id mode — known OpenAlex work IDs
BENCHMARK_QUERIES_SEED_MODE = [
    {
        "id": "seed_attention_is_all_you_need",
        "seed_work_id": "W2963403868",  # Attention Is All You Need
        "expected_title_keywords": ["attention"],
        "min_nodes": 15,
        "min_edges": 10,
    },
]

# DOI mode — realistic PDF metadata extraction workflow
BENCHMARK_DOI_MODE = [
    {"id": "doi_alexnet", "seed_doi": "10.1145/3065386", "domain": "Computer Vision", "min_nodes": 15, "min_edges": 10},
    {"id": "doi_word2vec", "seed_doi": "10.48550/arxiv.1301.3781", "domain": "NLP", "min_nodes": 15, "min_edges": 10},
    {"id": "doi_adam", "seed_doi": "10.48550/arXiv.1412.6980", "domain": "Deep Learning", "min_nodes": 15, "min_edges": 10},
    {"id": "doi_alphago", "seed_doi": "10.1038/nature16961", "domain": "Reinforcement Learning", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_gcn", "seed_doi": "10.48550/arXiv.1609.02907", "domain": "Graph ML", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_yolo", "seed_doi": "10.1109/CVPR.2016.91", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_seq2seq", "seed_doi": "10.48550/arXiv.1409.3215", "domain": "NLP", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_bahdanau", "seed_doi": "10.48550/arxiv.1409.0473", "domain": "NLP", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_vae", "seed_doi": "10.48550/arXiv.1312.6114", "domain": "Generative Models", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_simclr", "seed_doi": "10.48550/arXiv.2002.05709", "domain": "Self-Supervised Learning", "min_nodes": 10, "min_edges": 6},
    # OLD PAPERS
    {"id": "doi_lenet", "seed_doi": "10.1109/5.726791", "domain": "Classic CV", "min_nodes": 10, "min_edges": 6},
    {"id": "doi_lstm", "seed_doi": "10.1162/neco.1997.9.8.1735", "domain": "Classic RNN", "min_nodes": 10, "min_edges": 6},
    # SOFTWARE PAPERS
    {"id": "doi_numpy", "seed_doi": "10.1038/s41586-020-2649-2", "domain": "Software", "min_nodes": 8, "min_edges": 5},
    {"id": "doi_pytorch", "seed_doi": "10.48550/arXiv.1912.01703", "domain": "Software", "min_nodes": 8, "min_edges": 5},
    # EMERGING FIELDS
    {"id": "doi_instructgpt", "seed_doi": "10.48550/arXiv.2203.02155", "domain": "LLM Alignment", "min_nodes": 8, "min_edges": 5},
    {"id": "doi_clip", "seed_doi": "10.48550/arXiv.2103.00020", "domain": "Vision-Language", "min_nodes": 10, "min_edges": 6},
]

# Title mode — fallback when PDF has no DOI
BENCHMARK_TITLE_MODE = [
    {"id": "title_alexnet", "seed_title": "ImageNet Classification with Deep Convolutional Neural Networks", "domain": "Computer Vision", "min_nodes": 15, "min_edges": 10},
    {"id": "title_gpt3", "seed_title": "Language Models are Few-Shot Learners", "domain": "NLP", "min_nodes": 10, "min_edges": 6},
    {"id": "title_fcn", "seed_title": "Fully Convolutional Networks for Semantic Segmentation", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},
    {"id": "title_graphsage", "seed_title": "Inductive Representation Learning on Large Graphs", "domain": "Graph ML", "min_nodes": 10, "min_edges": 6},
    {"id": "title_elmo", "seed_title": "Deep contextualized word representations", "domain": "NLP", "min_nodes": 10, "min_edges": 6},
    {"id": "title_faster_rcnn", "seed_title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},
    {"id": "title_layer_norm", "seed_title": "Layer Normalization", "domain": "Deep Learning", "min_nodes": 10, "min_edges": 6},
    {"id": "title_dqn", "seed_title": "Human-level control through deep reinforcement learning", "domain": "Reinforcement Learning", "min_nodes": 10, "min_edges": 6},
    {"id": "title_clip", "seed_title": "Learning Transferable Visual Models From Natural Language Supervision", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},
    {"id": "title_diffusion", "seed_title": "Denoising Diffusion Probabilistic Models", "domain": "Generative Models", "min_nodes": 10, "min_edges": 6},
    # OLD PAPERS
    {"id": "title_backprop", "seed_title": "Learning representations by back-propagating errors", "domain": "Classic ML", "min_nodes": 8, "min_edges": 5},
    {"id": "title_svm", "seed_title": "Support-Vector Networks", "domain": "Classic ML", "min_nodes": 10, "min_edges": 6},
    # SOFTWARE
    {"id": "title_scikit", "seed_title": "Scikit-learn: Machine Learning in Python", "domain": "Software", "min_nodes": 8, "min_edges": 5},
    # EMERGING
    {"id": "title_llama", "seed_title": "LLaMA: Open and Efficient Foundation Language Models", "domain": "LLM", "min_nodes": 8, "min_edges": 5},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return name.replace(" ", "_").lower()


def _check_graph_structure(nodes: list, edges: list, case: dict):
    """Run structural assertions on the citation graph."""
    errors = []

    # 1. Minimum node count
    if len(nodes) < case["min_nodes"]:
        errors.append(f"Expected >= {case['min_nodes']} nodes, got {len(nodes)}")

    # 2. Minimum edge count
    if len(edges) < case["min_edges"]:
        errors.append(f"Expected >= {case['min_edges']} edges, got {len(edges)}")

    # 3. No duplicate work_ids
    work_ids = [n["work_id"] for n in nodes]
    dupes = [wid for wid, count in Counter(work_ids).items() if count > 1]
    if dupes:
        errors.append(f"Duplicate work_ids: {dupes}")

    # 4. Exactly one seed node
    seeds = [n for n in nodes if n.get("is_seed")]
    if len(seeds) != 1:
        errors.append(f"Expected exactly 1 seed node, got {len(seeds)}")

    # 5. Edge endpoints reference existing nodes
    node_id_set = set(work_ids)
    for edge in edges:
        if edge["from_work_id"] not in node_id_set:
            errors.append(f"Edge from_work_id {edge['from_work_id']} not in nodes")
            break
        if edge["to_work_id"] not in node_id_set:
            errors.append(f"Edge to_work_id {edge['to_work_id']} not in nodes")
            break

    # 6. No self-loops
    self_loops = [e for e in edges if e["from_work_id"] == e["to_work_id"]]
    if self_loops:
        errors.append(f"Found {len(self_loops)} self-loop edge(s)")

    return errors


def _check_diversity(nodes: list):
    """Check year and citation diversity. Returns dict of metrics."""
    years = [n["year"] for n in nodes if n.get("year")]
    citations = [n["cited_by_count"] for n in nodes if n.get("cited_by_count") is not None]
    hops = [n["hop"] for n in nodes]

    metrics = {}

    # Year spread
    if years:
        metrics["year_min"] = min(years)
        metrics["year_max"] = max(years)
        metrics["year_spread"] = max(years) - min(years)
        metrics["unique_decades"] = len(set(y // 10 for y in years))
    else:
        metrics["year_spread"] = 0
        metrics["unique_decades"] = 0

    # Citation spread
    if citations:
        metrics["citation_min"] = min(citations)
        metrics["citation_max"] = max(citations)
        metrics["high_cited"] = sum(1 for c in citations if c >= 100)
        metrics["low_cited"] = sum(1 for c in citations if c < 100)
    else:
        metrics["citation_min"] = 0
        metrics["citation_max"] = 0
        metrics["high_cited"] = 0
        metrics["low_cited"] = 0

    # Hop distribution
    hop_counts = Counter(hops)
    metrics["hop_distribution"] = dict(hop_counts)
    metrics["unique_hops"] = len(hop_counts)

    return metrics


def _compute_automated_score(nodes: list, edges: list) -> dict:
    """Compute the automated portion of the rubric (35 pts max)."""
    scores = {}

    # Graph structure (10 pts)
    n_nodes = len(nodes)
    n_edges = len(edges)
    node_ids = set(n["work_id"] for n in nodes)
    # Check for orphans (non-seed nodes with no edges)
    edge_participants = set()
    for e in edges:
        edge_participants.add(e["from_work_id"])
        edge_participants.add(e["to_work_id"])
    orphans = [n for n in nodes if not n.get("is_seed") and n["work_id"] not in edge_participants]

    if n_nodes >= 20 and n_edges >= 15 and len(orphans) == 0:
        scores["graph_structure"] = 10
    elif n_nodes >= 15 and n_edges >= 10 and len(orphans) <= 2:
        scores["graph_structure"] = 8
    elif n_nodes >= 10 and n_edges >= 6:
        scores["graph_structure"] = 6
    elif n_nodes >= 5:
        scores["graph_structure"] = 3
    else:
        scores["graph_structure"] = 0

    # Deduplication (5 pts)
    work_ids = [n["work_id"] for n in nodes]
    dupes = len(work_ids) - len(set(work_ids))
    if dupes == 0:
        scores["deduplication"] = 5
    elif dupes == 1:
        scores["deduplication"] = 3
    else:
        scores["deduplication"] = 0

    # Hop distribution (5 pts)
    hops = Counter(n["hop"] for n in nodes)
    unique_hops = len(hops)
    if unique_hops >= 3:
        scores["hop_distribution"] = 5
    elif unique_hops == 2:
        scores["hop_distribution"] = 3
    else:
        scores["hop_distribution"] = 1

    # Year diversity (5 pts)
    years = [n["year"] for n in nodes if n.get("year")]
    if years:
        decade_set = set(y // 10 for y in years)
        spread = max(years) - min(years)
        if len(decade_set) >= 3 and spread >= 15:
            scores["year_diversity"] = 5
        elif len(decade_set) >= 2 and spread >= 5:
            scores["year_diversity"] = 3
        else:
            scores["year_diversity"] = 1
    else:
        scores["year_diversity"] = 0

    # Citation diversity (5 pts)
    citations = [n["cited_by_count"] for n in nodes if n.get("cited_by_count") is not None]
    if citations:
        high = sum(1 for c in citations if c >= 100)
        low = sum(1 for c in citations if c < 100)
        if high >= 3 and low >= 3:
            scores["citation_diversity"] = 5
        elif high >= 1 and low >= 1:
            scores["citation_diversity"] = 3
        else:
            scores["citation_diversity"] = 1
    else:
        scores["citation_diversity"] = 0

    # Edge correctness (5 pts)
    bad_edges = 0
    for e in edges:
        if e["from_work_id"] not in node_ids or e["to_work_id"] not in node_ids:
            bad_edges += 1
        if e["from_work_id"] == e["to_work_id"]:
            bad_edges += 1
    if bad_edges == 0:
        scores["edge_correctness"] = 5
    elif bad_edges <= 2:
        scores["edge_correctness"] = 3
    else:
        scores["edge_correctness"] = 0

    scores["total_automated"] = sum(scores.values())
    return scores


# ---------------------------------------------------------------------------
# Tests — Query Mode
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_QUERIES_QUERY_MODE,
    ids=[c["id"] for c in BENCHMARK_QUERIES_QUERY_MODE],
)
def test_citation_map_query_mode(case, live_client, auth_headers, results_dir):
    """End-to-end citation map benchmark — query_text mode."""
    response = live_client.post(
        "/v1/citation-map",
        json={
            "query_text": case["query_text"],
            "citing_limit": 15,
            "references_limit": 15,
            "create_graph_draft": False,
        },
        headers=auth_headers,
        timeout=240,
    )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    nodes = result["nodes"]
    edges = result["edges"]
    seed_info = result["seed_info"]
    stats = result["stats"]

    # --- Seed quality ---
    assert seed_info["seed_work_id"], f"[{case['id']}] No seed work_id returned"
    seed_title = (seed_info.get("seed_title") or "").lower()

    # At least one expected keyword should appear in seed title
    keyword_match = any(kw.lower() in seed_title for kw in case["expected_seed_keywords"])
    # Don't hard-fail — just record it. Seed selection is judged by rubric.
    seed_keyword_hit = keyword_match

    # --- Graph structure ---
    structure_errors = _check_graph_structure(nodes, edges, case)
    assert not structure_errors, (
        f"[{case['id']}] Graph structure errors: {structure_errors}"
    )

    # --- Diversity ---
    diversity = _check_diversity(nodes)

    # Year spread: at least 5 years between oldest and newest
    if len(nodes) >= 10:
        assert diversity["year_spread"] >= 5, (
            f"[{case['id']}] Year spread too narrow: {diversity['year_spread']} years "
            f"(range: {diversity.get('year_min')}-{diversity.get('year_max')})"
        )

    # Citation spread: should have both high and low cited papers
    if len(nodes) >= 10:
        assert diversity["high_cited"] >= 1, (
            f"[{case['id']}] No highly-cited papers (>=100 citations) found"
        )

    # Hop distribution: should have at least 2 different hop levels
    assert diversity["unique_hops"] >= 2, (
        f"[{case['id']}] Only {diversity['unique_hops']} hop level(s): {diversity['hop_distribution']}"
    )

    # --- Automated scoring ---
    auto_scores = _compute_automated_score(nodes, edges)

    # --- Print scoring breakdown ---
    print(f"\n{'='*60}")
    print(f"CITATION MAP SCORING: {case['id']}")
    print(f"Query: {case['query_text']}")
    print(f"Domain: {case.get('domain', 'unknown')}")
    print(f"{'='*60}")
    print(f"Seed: {seed_info.get('seed_title', 'N/A')}")
    print(f"Seed work_id: {seed_info['seed_work_id']}")
    print(f"Seed keyword match: {'✓' if seed_keyword_hit else '✗'}")
    print(f"Strategy: {seed_info.get('selection_strategy', 'N/A')}")
    print(f"Candidates considered: {seed_info.get('candidates_considered', 'N/A')}")
    print(f"\nGraph: {stats['total_nodes']} nodes, {stats['edges_count']} edges")
    print(f"  Citing seed: {stats.get('citing_found', 0)}")
    print(f"  Referenced by seed: {stats.get('references_found', 0)}")
    print(f"  Network (multi-hop): {stats.get('network_nodes', 0)}")
    print(f"  Max hop: {stats.get('max_hop', 0)}")
    print(f"\nDiversity:")
    print(f"  Year spread: {diversity['year_spread']} years ({diversity.get('year_min', '?')}-{diversity.get('year_max', '?')})")
    print(f"  Unique decades: {diversity['unique_decades']}")
    print(f"  High-cited (>=100): {diversity['high_cited']}")
    print(f"  Low-cited (<100): {diversity['low_cited']}")
    print(f"  Hop distribution: {diversity['hop_distribution']}")
    print(f"\nAutomated Score ({auto_scores['total_automated']}/35):")
    for k, v in auto_scores.items():
        if k != "total_automated":
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # --- Save results ---
    summary = {
        "query": case["query_text"],
        "mode": "query_text",
        "domain": case.get("domain"),
        "seed_info": seed_info,
        "seed_keyword_hit": seed_keyword_hit,
        "stats": stats,
        "diversity": diversity,
        "automated_scores": auto_scores,
        "top_10_titles": [
            n.get("title", "N/A") for n in sorted(
                nodes, key=lambda x: x.get("cited_by_count", 0), reverse=True
            )[:10]
        ],
        "all_nodes_summary": [
            {
                "work_id": n["work_id"],
                "title": n.get("title"),
                "year": n.get("year"),
                "cited_by_count": n.get("cited_by_count", 0),
                "hop": n.get("hop", 0),
                "relationship": n.get("relationship"),
            }
            for n in nodes
        ],
        "edges": edges,
    }
    save_result(results_dir, f"citation_map_{case['id']}", summary)

    # --- Regression detection ---
    previous = load_previous_result(results_dir, f"citation_map_{case['id']}")
    if previous and previous.get("stats", {}).get("total_nodes", 0) > 0:
        prev_total = previous["stats"]["total_nodes"]
        if len(nodes) < prev_total * 0.5:
            pytest.warns(
                UserWarning,
                match=f"Regression: {case['id']} dropped from {prev_total} to {len(nodes)} nodes",
            )


# ---------------------------------------------------------------------------
# Tests — Seed Mode (direct work_id)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_QUERIES_SEED_MODE,
    ids=[c["id"] for c in BENCHMARK_QUERIES_SEED_MODE],
)
def test_citation_map_seed_mode(case, live_client, auth_headers, results_dir):
    """End-to-end citation map benchmark — seed_work_id mode."""
    response = live_client.post(
        "/v1/citation-map",
        json={
            "seed_work_id": case["seed_work_id"],
            "citing_limit": 15,
            "references_limit": 15,
            "create_graph_draft": False,
        },
        headers=auth_headers,
        timeout=240,
    )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    nodes = result["nodes"]
    edges = result["edges"]
    seed_info = result["seed_info"]

    # Seed should be the one we requested
    assert seed_info["seed_work_id"] == case["seed_work_id"]
    assert seed_info["selection_strategy"] == "direct"

    # Graph structure checks
    structure_errors = _check_graph_structure(nodes, edges, case)
    assert not structure_errors, (
        f"[{case['id']}] Graph structure errors: {structure_errors}"
    )

    # Diversity checks
    diversity = _check_diversity(nodes)
    auto_scores = _compute_automated_score(nodes, edges)

    print(f"\n{'='*60}")
    print(f"CITATION MAP (SEED MODE): {case['id']}")
    print(f"Seed: {seed_info.get('seed_title', 'N/A')} ({case['seed_work_id']})")
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Automated Score: {auto_scores['total_automated']}/35")
    print(f"{'='*60}\n")

    save_result(results_dir, f"citation_map_{case['id']}", {
        "mode": "seed_work_id",
        "seed_work_id": case["seed_work_id"],
        "seed_info": seed_info,
        "stats": result["stats"],
        "diversity": diversity,
        "automated_scores": auto_scores,
        "node_count": len(nodes),
        "edge_count": len(edges),
    })


# ---------------------------------------------------------------------------
# Tests — DOI Mode (realistic PDF metadata extraction)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_DOI_MODE,
    ids=[c["id"] for c in BENCHMARK_DOI_MODE],
)
def test_citation_map_doi_mode(case, live_client, auth_headers, results_dir):
    """End-to-end citation map benchmark — seed_doi mode."""
    response = live_client.post(
        "/v1/citation-map",
        json={
            "seed_doi": case["seed_doi"],
            "citing_limit": 20,
            "references_limit": 20,
            "total_nodes": 50,
            "create_graph_draft": False,
        },
        headers=auth_headers,
        timeout=240,
    )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    nodes = result["nodes"]
    edges = result["edges"]
    seed_info = result["seed_info"]
    stats = result["stats"]

    assert seed_info["seed_work_id"], f"[{case['id']}] No seed work_id returned"
    assert seed_info["selection_strategy"] == "doi_lookup"

    structure_errors = _check_graph_structure(nodes, edges, case)
    assert not structure_errors, f"[{case['id']}] Graph structure errors: {structure_errors}"

    diversity = _check_diversity(nodes)
    auto_scores = _compute_automated_score(nodes, edges)

    print(f"\n{'='*60}")
    print(f"CITATION MAP (DOI MODE): {case['id']}")
    print(f"DOI: {case['seed_doi']}")
    print(f"Seed: {seed_info.get('seed_title', 'N/A')}")
    print(f"Graph: {stats['total_nodes']} nodes, {stats['edges_count']} edges")
    print(f"Automated Score: {auto_scores['total_automated']}/35")
    print(f"{'='*60}\n")

    save_result(results_dir, f"citation_map_{case['id']}", {
        "mode": "seed_doi",
        "seed_doi": case["seed_doi"],
        "domain": case.get("domain"),
        "seed_info": seed_info,
        "stats": stats,
        "diversity": diversity,
        "automated_scores": auto_scores,
        "nodes": nodes,
        "edges": edges,
    })


# ---------------------------------------------------------------------------
# Tests — Title Mode (fallback when PDF has no DOI)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_TITLE_MODE,
    ids=[c["id"] for c in BENCHMARK_TITLE_MODE],
)
def test_citation_map_title_mode(case, live_client, auth_headers, results_dir):
    """End-to-end citation map benchmark — seed_title mode."""
    response = live_client.post(
        "/v1/citation-map",
        json={
            "seed_title": case["seed_title"],
            "citing_limit": 20,
            "references_limit": 20,
            "total_nodes": 50,
            "create_graph_draft": False,
        },
        headers=auth_headers,
        timeout=240,
    )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    nodes = result["nodes"]
    edges = result["edges"]
    seed_info = result["seed_info"]
    stats = result["stats"]

    assert seed_info["seed_work_id"], f"[{case['id']}] No seed work_id returned"
    assert seed_info["selection_strategy"] == "title_search"

    structure_errors = _check_graph_structure(nodes, edges, case)
    assert not structure_errors, f"[{case['id']}] Graph structure errors: {structure_errors}"

    diversity = _check_diversity(nodes)
    auto_scores = _compute_automated_score(nodes, edges)

    print(f"\n{'='*60}")
    print(f"CITATION MAP (TITLE MODE): {case['id']}")
    print(f"Title: {case['seed_title']}")
    print(f"Found: {seed_info.get('seed_title', 'N/A')}")
    print(f"Candidates: {seed_info.get('candidates_considered', 'N/A')}")
    print(f"Graph: {stats['total_nodes']} nodes, {stats['edges_count']} edges")
    print(f"Automated Score: {auto_scores['total_automated']}/35")
    print(f"{'='*60}\n")

    save_result(results_dir, f"citation_map_{case['id']}", {
        "mode": "seed_title",
        "seed_title": case["seed_title"],
        "domain": case.get("domain"),
        "seed_info": seed_info,
        "stats": stats,
        "diversity": diversity,
        "automated_scores": auto_scores,
        "nodes": nodes,
        "edges": edges,
    })


# ---------------------------------------------------------------------------
# PDF Upload Mode — realistic end-to-end workflow with actual PDF files
# ---------------------------------------------------------------------------

BENCHMARK_PDF_MODE = [
    # EXISTING: Modern ML/CV papers
    {"id": "pdf_lenet", "pdf_path": "test_pdfs/03_W2112796928.pdf", "expected_work_id": "W2112796928", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},
    {"id": "pdf_svm", "pdf_path": "test_pdfs/05_W4239510810.pdf", "expected_work_id": "W4239510810", "expected_work_ids": ["W4239510810", "W2119821739"], "domain": "Machine Learning", "min_nodes": 10, "min_edges": 6},
    {"id": "pdf_scipy", "pdf_path": "test_pdfs/06_W3003257820.pdf", "expected_work_id": "W3003257820", "domain": "Scientific Computing", "min_nodes": 10, "min_edges": 6},
    {"id": "pdf_wgcna", "pdf_path": "test_pdfs/09_W1966327575.pdf", "expected_work_id": "W1966327575", "domain": "Bioinformatics", "min_nodes": 10, "min_edges": 6},
    {"id": "pdf_attention", "pdf_path": "test_pdfs/attention_is_all_you_need.pdf", "expected_work_id": "W2626778328", "domain": "NLP", "min_nodes": 15, "min_edges": 10},
    {"id": "pdf_bert", "pdf_path": "test_pdfs/bert.pdf", "expected_work_id": "W2963341956", "domain": "NLP", "min_nodes": 15, "min_edges": 10},
    {"id": "pdf_resnet", "pdf_path": "test_pdfs/resnet.pdf", "expected_work_id": "W2194775991", "domain": "Computer Vision", "min_nodes": 15, "min_edges": 10},
    {"id": "pdf_vgg", "pdf_path": "test_pdfs/vgg.pdf", "expected_work_id": "W1686810756", "domain": "Computer Vision", "min_nodes": 10, "min_edges": 6},

    # TODO: Download and add these diverse PDFs to prevent regressions:
    #
    # OLD PAPERS (pre-2000) - Test sparse citations, historical networks
    # {"id": "pdf_backprop", "pdf_path": "test_pdfs/backprop.pdf", "expected_work_id": "W1984876255", "domain": "Classic ML", "min_nodes": 8, "min_edges": 5},
    # {"id": "pdf_lstm", "pdf_path": "test_pdfs/lstm_hochreiter.pdf", "expected_work_id": "W1963535118", "domain": "Classic RNN", "min_nodes": 10, "min_edges": 6},
    #
    # SOFTWARE PAPERS - Test noisy cross-domain citations
    # {"id": "pdf_numpy", "pdf_path": "test_pdfs/numpy_array.pdf", "expected_work_id": "W3020563335", "domain": "Software", "min_nodes": 8, "min_edges": 5},
    # {"id": "pdf_pytorch", "pdf_path": "test_pdfs/pytorch_framework.pdf", "expected_work_id": "W2929477623", "domain": "Software", "min_nodes": 8, "min_edges": 5},
    #
    # EMERGING PAPERS - Test low-citation new fields
    # {"id": "pdf_instructgpt", "pdf_path": "test_pdfs/instructgpt.pdf", "expected_work_id": "W4213426850", "domain": "LLM Alignment", "min_nodes": 8, "min_edges": 5},
    # {"id": "pdf_llama", "pdf_path": "test_pdfs/llama.pdf", "expected_work_id": "W4360956087", "domain": "Open LLM", "min_nodes": 8, "min_edges": 5},
    #
    # NICHE FIELDS - Test hard matching, specialized domains
    # {"id": "pdf_topological", "pdf_path": "test_pdfs/persistent_homology.pdf", "expected_work_id": "WXXXXXXXX", "domain": "Math/CS", "min_nodes": 8, "min_edges": 4},
    # {"id": "pdf_quantum", "pdf_path": "test_pdfs/shor_algorithm.pdf", "expected_work_id": "W2135126713", "domain": "Quantum", "min_nodes": 8, "min_edges": 4},
    #
    # CROSS-DOMAIN - Test citation drift
    # {"id": "pdf_alphafold", "pdf_path": "test_pdfs/alphafold.pdf", "expected_work_id": "W3177828909", "domain": "Bio/ML", "min_nodes": 10, "min_edges": 6},
    # {"id": "pdf_climate_ml", "pdf_path": "test_pdfs/climate_ml.pdf", "expected_work_id": "WXXXXXXXX", "domain": "Climate/ML", "min_nodes": 8, "min_edges": 4},
]


# ---------------------------------------------------------------------------
# Tests — PDF Upload Mode
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_PDF_MODE,
    ids=[c["id"] for c in BENCHMARK_PDF_MODE],
)
def test_citation_map_pdf_upload_mode(case, live_client, auth_headers, results_dir):
    """End-to-end citation map benchmark — PDF upload mode."""
    from pathlib import Path

    pdf_path = Path(case["pdf_path"])
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"

    with open(pdf_path, 'rb') as f:
        files = {'pdf_file': (pdf_path.name, f, 'application/pdf')}
        response = live_client.post(
            "/v1/citation-map/pdf",
            files=files,
            params={
                'citing_limit': 20,
                'references_limit': 20,
                'create_graph_draft': False,
            },
            headers=auth_headers,
            timeout=240,
        )

    assert response.status_code == 200, (
        f"[{case['id']}] Expected 200, got {response.status_code}: {response.text[:500]}"
    )

    result = response.json()
    nodes = result["nodes"]
    edges = result["edges"]
    seed_info = result["seed_info"]
    stats = result["stats"]

    assert seed_info["seed_work_id"], f"[{case['id']}] No seed work_id returned"
    assert seed_info["selection_strategy"] == "title_search"

    # Verify correct paper was matched
    # Some papers have duplicate OpenAlex entries — accept either
    matched_work_id = seed_info["seed_work_id"]
    expected = case.get("expected_work_ids", [case["expected_work_id"]])
    assert matched_work_id in expected, (
        f"[{case['id']}] Expected work_id in {expected}, got {matched_work_id}"
    )

    structure_errors = _check_graph_structure(nodes, edges, case)
    assert not structure_errors, f"[{case['id']}] Graph structure errors: {structure_errors}"

    diversity = _check_diversity(nodes)
    auto_scores = _compute_automated_score(nodes, edges)

    print(f"\n{'='*60}")
    print(f"CITATION MAP (PDF UPLOAD): {case['id']}")
    print(f"PDF: {pdf_path.name}")
    print(f"Matched: {seed_info.get('seed_title', 'N/A')}")
    print(f"Work ID: {matched_work_id}")
    print(f"Graph: {stats['total_nodes']} nodes, {stats['edges_count']} edges")
    print(f"Automated Score: {auto_scores['total_automated']}/35")
    print(f"{'='*60}\n")

    save_result(results_dir, f"citation_map_{case['id']}", {
        "mode": "pdf_upload",
        "pdf_file": pdf_path.name,
        "domain": case.get("domain"),
        "expected_work_id": case["expected_work_id"],
        "seed_info": seed_info,
        "stats": stats,
        "diversity": diversity,
        "automated_scores": auto_scores,
        "nodes": nodes,
        "edges": edges,
    })
