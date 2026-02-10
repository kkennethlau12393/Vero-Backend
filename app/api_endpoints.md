# Alexandria Backend — API Endpoints Reference

Complete documentation of all API endpoints, their roles, pipelines, request/response structures, and frontend integration points.

---

## Table of Contents

- [Feature 1: Citation Map Retrieval](#feature-1-citation-map-retrieval)
- [Feature 2: Ranking & Temporal Analysis](#feature-2-ranking--temporal-analysis)
- [Feature 3: Node Details & Novelty Assessment](#feature-3-node-details--novelty-assessment)
- [Feature 4: Methodology Comparison](#feature-4-methodology-comparison)
- [Feature 5: Research Gap Analysis](#feature-5-research-gap-analysis)
- [Settings: Institutional Access](#settings-institutional-access)
- [Cross-Feature Dependencies](#cross-feature-dependencies)
- [Authentication](#authentication)

---

## Authentication

All endpoints require a valid tenant token passed via the `Authorization` header:

```
Authorization: Bearer <tenant_token>
```

Tenant context is resolved from the token and scoped to all database operations.

---

## Feature 1: Citation Map Retrieval

**Prefix:** `/v1`
**Role:** Builds a citation graph around a seed paper. This is the entry point for the entire application — users either provide a paper ID directly or a natural-language query, and Feature 1 returns a citation network that feeds into all downstream features.

### `POST /v1/citation-map`

**When to call on the frontend:** When the user initiates a new exploration — either by pasting a paper ID/URL or typing a research query.

#### Pipeline / Workflow

```
User Input (seed_work_id OR query_text)
  │
  ├─ Mode 1: seed_work_id provided → use directly
  │
  └─ Mode 2: query_text provided
       │
       ├─ Search OpenAlex, Semantic Scholar, ArXiv
       ├─ Collect candidate papers
       ├─ Sort by citation count
       ├─ LLM tier classification on top-20 candidates
       ├─ Filter to HIGH+ relevance (score ≥ 0.75)
       └─ Select highest-cited from filtered set
  │
  ▼
Expand citation graph
  ├─ Fetch citing papers (up to citing_limit)
  ├─ Fetch referenced papers (up to references_limit)
  └─ Multi-hop expansion (if use_multi_hop=true)
  │
  ▼
Return nodes, edges, graph_draft_id
```

#### Request

```json
{
  "seed_work_id": "W2100837269",         // Mode 1: direct paper ID (optional)
  "query_text": "deep reinforcement learning for robotics",  // Mode 2: NL query (optional)
  "citing_limit": 15,                     // max papers citing the seed
  "references_limit": 15,                 // max papers referenced by seed
  "min_citations": 0,                     // minimum citation threshold
  "use_multi_hop": true,                  // enable multi-hop graph expansion
  "total_nodes": null,                    // target node count for multi-hop
  "create_graph_draft": true              // create draft for Feature 2
}
```

> **Note:** Exactly one of `seed_work_id` or `query_text` must be provided.

#### Response

```json
{
  "seed_info": {
    "seed_work_id": "W2100837269",
    "title": "Continuous control with deep reinforcement learning",
    "selection_method": "llm_validated",
    "llm_relevance_tier": "ESSENTIAL",
    "cited_by_count": 8500
  },
  "nodes": [
    {
      "work_id": "W2100837269",
      "title": "Continuous control with deep reinforcement learning",
      "year": 2015,
      "authors": ["Timothy Lillicrap", "..."],
      "venue": "ICLR",
      "cited_by_count": 8500,
      "hop_distance": 0,
      "access_status": "open",
      "pdf_url": "https://..."
    }
  ],
  "edges": [
    {
      "from_work_id": "W2100837269",
      "to_work_id": "W2076063813",
      "relationship": "cites"
    }
  ],
  "graph_draft_id": "a1b2c3d4-...",
  "stats": {
    "node_count": 45,
    "edge_count": 120,
    "max_hop_distance": 2
  }
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `missing_input` | Neither `seed_work_id` nor `query_text` provided |
| 400 | `invalid_input` | Both `seed_work_id` and `query_text` provided |
| 500 | `internal_error` | Upstream API or LLM failure |

---

## Feature 2: Ranking & Temporal Analysis

**Prefix:** `/v1/maps`, `/v1/rank`
**Role:** Builds interactive maps from citation graphs, ranks papers by relevance, generates subtopic clusters, and provides temporal analysis. This is the core analytical layer of the application.

---

### `POST /v1/maps/build`

**When to call on the frontend:** Immediately after receiving a `graph_draft_id` from Feature 1. This transforms the raw citation graph into a browsable, grouped map.

#### Pipeline / Workflow

```
graph_draft_id (from Feature 1)
  │
  ├─ Validate ownership
  ├─ Normalize graph (remove duplicates, self-loops, missing endpoints)
  ├─ Enrich nodes with topic info from OpenAlex
  ├─ Compute connector scores (degree or PageRank)
  ├─ Determine default grouping (topic vs subfield)
  └─ Persist map to database
  │
  ▼
Return map_id + metadata
```

#### Request

```json
{
  "graph_draft_id": "a1b2c3d4-...",
  "connector_score_mode": "degree",     // "degree" | "pagerank" | "none"
  "layout_mode": "none",
  "grouping_policy_version": 1
}
```

#### Response

```json
{
  "map_id": "e5f6g7h8-...",
  "default_grouping": "topic",
  "allowed_groupings": ["topic", "subfield"],
  "node_count": 45,
  "edge_count": 120,
  "stats": {}
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 403 | `permission_error` | Map not owned by tenant |
| 404 | `graph_draft_not_found` | Invalid graph_draft_id |
| 400 | `graph_draft_empty` | No nodes in graph draft |
| 413 | `graph_too_large_for_sync` | >3000 nodes or >20000 edges |

---

### `GET /v1/maps/{map_id}`

**When to call on the frontend:** To render the citation map visualization. Call after `maps/build` returns, and again when the user switches groupings.

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Map to retrieve |
| `group_by` | string | query | Optional: `"topic"` or `"subfield"` |

#### Response

```json
{
  "map_id": "e5f6g7h8-...",
  "field_contexts": [
    { "field_id": "...", "display_name": "Machine Learning" }
  ],
  "active_grouping": "topic",
  "allowed_groupings": ["topic", "subfield"],
  "group_summaries": [
    {
      "group_id": "...",
      "label": "Reinforcement Learning",
      "paper_count": 12
    }
  ],
  "nodes": [
    {
      "work_id": "W2100837269",
      "title": "Continuous control with deep reinforcement learning",
      "year": 2015,
      "authors": ["Timothy Lillicrap"],
      "venue": "ICLR",
      "cited_by_count": 8500,
      "access_status": "open",
      "pdf_url": "https://...",
      "connector_score": 0.85,
      "x": 120.5,
      "y": 340.2
    }
  ],
  "edges": [
    {
      "from_work_id": "W2100837269",
      "to_work_id": "W2076063813"
    }
  ]
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 403 | `permission_error` | Map not owned by tenant |
| 404 | `map_not_found` | Invalid map_id |
| 400 | `invalid_group_by` | Unrecognized grouping value |

---

### `POST /v1/rank`

**When to call on the frontend:** When the user wants a ranked reading list for a research topic. This is the main ranking endpoint — it can be called standalone (without a citation map) or as a companion to map exploration. Also used when a user clicks a topic from the map to explore it as a ranked list.

#### Pipeline / Workflow

```
query_text OR seed_title
  │
  ├─ If seed_title: LLM generates query from title
  │
  ▼
Stage 1: Retrieval
  ├─ OpenAlex keyword search
  ├─ Semantic Scholar search
  ├─ pgvector KNN (embedding similarity)
  └─ Merge into candidate pool (up to 5000)
  │
  ▼
Stage 2: Scoring (wave-based convergence)
  ├─ LLM relevance (ESSENTIAL/HIGH/MEDIUM/LOW/NONE tiers)
  ├─ Provenance score (retrieval source quality)
  ├─ Citation impact (log-scaled, Bayesian)
  ├─ Recency (exponential decay, configurable half-life)
  ├─ Topic relevance
  └─ Methodological & domain alignment
  │
  ▼
Stage 3: Weighting & Normalization
  ├─ Weighted combination: 60% relevance, 30% impact, 8% recency, 2% completeness
  ├─ Minimum LLM relevance threshold: 0.60
  └─ Robust normalization across features
  │
  ▼
Stage 4: Categorization
  ├─ Fundamentals (~20%) — field-defining foundational papers
  ├─ Core Concepts & Theory (~50%) — core methods and theory
  └─ Applications (~30%) — clinical/case studies
  │
  ▼
Return ranked results + query classification
```

#### Request

```json
{
  "query_text": "transformer architectures for NLP",   // option 1
  "seed_title": null,                                   // option 2 (mutually exclusive)
  "context": {
    "topic_id": "T12345"                                // optional: OpenAlex topic filter
  },
  "filters": {
    "year_min": 2015,
    "year_max": 2025,
    "topic_id": null
  },
  "params": {
    "top_k": 100,
    "max_candidates_scored": 5000,
    "recency_half_life_years": 5.0,
    "w_rel": 0.60,
    "w_imp": 0.30,
    "w_rec": 0.08,
    "w_comp": 0.02
  }
}
```

> **Note:** Exactly one of `query_text` or `seed_title` must be provided.

#### Response (200 — Completed)

```json
{
  "rank_job_id": "f9a8b7c6-...",
  "job": {
    "status": "completed"
  },
  "query_classification": {
    "type": "topical",
    "confidence": 0.92,
    "query_specificity": "broad"
  },
  "convergence": {
    "waves_used": 3,
    "total_scored": 250,
    "converged": true
  },
  "items": [ /* flat ranked list — all papers */ ],
  "fundamentals": [
    {
      "rank_index": 1,
      "work_id": "W2100837269",
      "score": 0.94,
      "reasons": ["Highly cited foundational work", "Directly addresses query topic"],
      "score_breakdown": {
        "llm_relevance": 0.95,
        "citation_impact": 0.88,
        "recency": 0.45,
        "provenance": 0.90
      },
      "preview": {
        "title": "Attention Is All You Need",
        "year": 2017,
        "authors": ["Ashish Vaswani", "..."],
        "venue": "NeurIPS",
        "cited_by_count": 95000,
        "abstract": "..."
      },
      "provenance": ["openalex", "semantic_scholar"]
    }
  ],
  "core_concepts_and_theory": [ /* same structure */ ],
  "applications": [ /* same structure */ ]
}
```

#### Response (202 — Job Pending)

```json
{
  "rank_job_id": "f9a8b7c6-...",
  "job": { "status": "running" },
  "query_classification": null,
  "convergence": null,
  "items": [],
  "fundamentals": [],
  "core_concepts_and_theory": [],
  "applications": []
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `missing_input` | Neither `query_text` nor `seed_title` provided |
| 403 | `permission_error` | Tenant mismatch |
| 404 | `candidate_set_not_found` | Internal candidate set missing |
| 500 | `internal_error` | Upstream failure |

---

### `POST /v1/rank/drill-down`

**When to call on the frontend:** When the user clicks a subtopic from the subtopic list to explore it in detail. Uses the `drill_down_query` from the subtopics endpoint.

#### Pipeline / Workflow

Same as the main ranking pipeline but optimized for speed:
- Reduced candidate pool: **300 max** (vs 5000)
- LLM scoring cap: **30 papers**
- Skips slow external sources (DBLP, PubMed, CrossRef)
- Higher LLM relevance threshold: **0.35**
- Weights: LLM 50%, Impact 35%, Methodological 5%, Domain 5%
- Returns flat ranked list (no categorization into fundamentals/core/applications)

#### Request

```json
{
  "query_text": "attention mechanisms in transformer architectures",
  "top_k": 10
}
```

#### Response

Same structure as `POST /v1/rank` but with only the `items` list populated (no `fundamentals`/`core_concepts_and_theory`/`applications` categorization).

---

### `POST /v1/rank/{rank_job_id}/generate-subtopics`

**When to call on the frontend:** After receiving ranking results for a **broad** query (`query_specificity: "broad"`). Displays subtopic chips/cards that the user can click to drill down.

#### Pipeline / Workflow

```
rank_job_id (from ranking endpoint)
  │
  ├─ Fetch original query + ranked papers
  ├─ LLM clusters papers by theme
  ├─ Generate human-readable labels per cluster
  ├─ Identify representative papers per cluster
  └─ Create drill_down_query for each subtopic
  │
  ▼
Return subtopic list with drill-down queries
```

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `rank_job_id` | UUID | path | From the ranking response |

#### Response

```json
{
  "rank_job_id": "f9a8b7c6-...",
  "query_specificity": "broad",
  "subtopics": [
    {
      "subtopic_id": "subtopic_1",
      "label": "Transformers & Attention Mechanisms",
      "description": "Papers focusing on self-attention architectures and their variations",
      "representative_work_ids": ["W2100837269", "W3456789012"],
      "paper_count": 15,
      "drill_down_query": "self-attention mechanisms in transformer neural networks"
    },
    {
      "subtopic_id": "subtopic_2",
      "label": "Pre-training & Transfer Learning",
      "description": "Large-scale pre-training strategies and domain adaptation",
      "representative_work_ids": ["W1234567890"],
      "paper_count": 22,
      "drill_down_query": "pre-training strategies and transfer learning for language models"
    }
  ]
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_input` | ValueError |
| 404 | `rank_job_not_found` | Invalid rank_job_id |
| 500 | `internal_error` | LLM failure |

---

### `GET /v1/rank/{rank_job_id}/temporal-map`

**When to call on the frontend:** When the user opens the "Timeline" or "Temporal Map" view for a set of ranked results. Can also be scoped to a single subtopic.

#### Pipeline / Workflow

```
rank_job_id + optional subtopic_id
  │
  ├─ Load ranked results
  ├─ Group papers by decade (eras)
  ├─ Identify milestone papers (high citations within era)
  │
  ├─ If include_analytics=true:
  │    ├─ Breakthrough detection (paradigm shift analysis via LLM)
  │    └─ Evolution trend computation per era
  │
  ▼
Return eras with papers, milestones, and optional analytics
```

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `rank_job_id` | UUID | path | From ranking response |
| `subtopic_id` | string | query | Optional subtopic filter |
| `include_analytics` | bool | query | Include breakthrough/evolution analysis (default: false) |

#### Response

```json
{
  "rank_job_id": "f9a8b7c6-...",
  "scope": "broad",
  "scope_label": "transformer architectures for NLP",
  "topic_id": "T12345",
  "subtopic_id": null,
  "eras": [
    {
      "label": "2010s",
      "start_year": 2010,
      "end_year": 2019,
      "papers": [
        {
          "work_id": "W2100837269",
          "title": "Attention Is All You Need",
          "year": 2017,
          "cited_by_count": 95000,
          "is_milestone": true,
          "rank_in_results": 1
        }
      ],
      "milestone_count": 3,
      "is_breakthrough_era": true
    }
  ],
  "analytics": {
    "breakthrough": {
      "breakthrough_year": 2017,
      "from_era": "2010s",
      "breakthrough_era": "2010s",
      "breakthrough_papers": [
        { "work_id": "W2100837269", "title": "Attention Is All You Need", "year": 2017 }
      ],
      "pre_breakthrough_approach": "RNN/LSTM-based sequence models with attention add-ons",
      "post_breakthrough_approach": "Pure self-attention transformer architectures",
      "shift_description": "The introduction of the Transformer eliminated recurrence...",
      "shift_citations": ["W2100837269"],
      "rejection_reason": null,
      "rejection_citations": null,
      "confidence": 0.95
    },
    "evolution": [
      {
        "era": "2000s",
        "dominant_approach": "Statistical NLP and n-gram models",
        "key_themes": ["statistical parsing", "machine translation"],
        "representative_paper_id": "W9876543210"
      },
      {
        "era": "2010s",
        "dominant_approach": "Neural sequence-to-sequence with attention",
        "key_themes": ["word embeddings", "attention mechanisms", "transformers"],
        "representative_paper_id": "W2100837269"
      }
    ]
  }
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_input` | ValueError |
| 404 | `rank_job_not_found` | Invalid rank_job_id |
| 500 | `internal_error` | LLM or upstream failure |

---

## Feature 3: Node Details & Novelty Assessment

**Prefix:** `/v1`
**Role:** Provides deep-dive information about a single paper within a citation map — including LLM-generated summaries, keyword extraction, novelty assessment with grounding evidence, and per-node timeline analysis.

### `GET /v1/maps/{map_id}/nodes/{work_id}/details`

**When to call on the frontend:** When the user clicks on a node in the citation map to open the detail panel/sidebar.

#### Pipeline / Workflow

```
map_id + work_id
  │
  ├─ Validate map ownership & node membership
  ├─ Check cache for existing details
  │
  ├─ If not cached:
  │    ├─ Fetch paper metadata (OpenAlex / S2)
  │    ├─ LLM: Generate summary
  │    ├─ LLM: Extract keywords
  │    ├─ Novelty Assessment:
  │    │    ├─ Fetch cited references
  │    │    ├─ Retrieve field landmarks
  │    │    ├─ LLM: Compare paper to prior work
  │    │    └─ Ground claims in specific papers
  │    └─ Cache results
  │
  ├─ If include_timeline=true:
  │    ├─ Build backward timeline (references + landmarks before paper)
  │    ├─ Build forward timeline (citing papers after)
  │    └─ Impact analysis (paradigm shift detection)
  │
  ▼
Return full node details
```

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Citation map ID |
| `work_id` | string | path | Paper ID (e.g., `"W2100837269"`) |
| `include_timeline` | bool | query | Include per-node timeline (default: false) |

#### Response

```json
{
  "work_id": "W2100837269",
  "title": "Continuous control with deep reinforcement learning",
  "year": 2015,
  "authors": ["Timothy Lillicrap", "Jonathan J. Hunt", "..."],
  "venue": "ICLR",
  "cited_by_count": 8500,
  "abstract": "We adapt the ideas underlying...",
  "summary": "This paper introduces DDPG, a model-free off-policy actor-critic algorithm that learns continuous control policies directly from raw pixels...",
  "keywords": ["deep reinforcement learning", "actor-critic", "continuous control", "DDPG"],
  "novelty_assessment": {
    "whats_new": "First successful application of deep function approximation to continuous action spaces using deterministic policy gradients",
    "compared_to_prior_work": "Prior work (DQN) was limited to discrete actions; DPG lacked deep function approximators",
    "novelty_level": "high",
    "confidence": "high",
    "novelty_explanation": "DDPG combines insights from DQN (experience replay, target networks) with DPG to enable continuous control from pixels — a novel and impactful combination.",
    "grounding_papers": [
      {
        "work_id": "W2076063813",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "year": 2013,
        "cited_by_count": 12000,
        "relationship": "cited_reference",
        "relevance": "DQN provided the experience replay and target network techniques adapted by DDPG"
      },
      {
        "work_id": "W2165150801",
        "title": "Deterministic policy gradient algorithms",
        "year": 2014,
        "cited_by_count": 3500,
        "relationship": "cited_reference",
        "relevance": "DPG provided the theoretical foundation for deterministic policy gradients extended by DDPG"
      }
    ]
  },
  "assessment_unavailable_reason": null,
  "connected_works": [
    {
      "work_id": "W2076063813",
      "title": "Playing Atari with Deep Reinforcement Learning",
      "relationship": "cites"
    },
    {
      "work_id": "W3456789012",
      "title": "Soft Actor-Critic",
      "relationship": "cited_by"
    }
  ],
  "timeline": {
    "target_work_id": "W2100837269",
    "target_year": 2015,
    "backward": [
      {
        "section_label": "References",
        "papers": [
          { "work_id": "W2076063813", "title": "...", "year": 2013, "cited_by_count": 12000 }
        ]
      }
    ],
    "forward": [
      {
        "section_label": "Citing Papers",
        "papers": [
          { "work_id": "W3456789012", "title": "Soft Actor-Critic", "year": 2018, "cited_by_count": 5000 }
        ]
      }
    ],
    "impact_analysis": {
      "is_paradigm_shift": true,
      "impact_score": 0.88,
      "before_approach": "Discrete action RL with DQN",
      "after_approach": "Continuous control with actor-critic methods",
      "shift_description": "Enabled a new line of continuous control RL research"
    }
  },
  "primary_topic_id": "T12345",
  "topic_display_name": "Reinforcement Learning",
  "access_status": "open",
  "pdf_url": "https://arxiv.org/pdf/1509.02971",
  "doi_url": "https://doi.org/10.48550/arXiv.1509.02971",
  "oa_status": "gold"
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_input` | ValueError |
| 403 | `permission_error` | Map not owned by tenant |
| 404 | `map_not_found` | Invalid map_id |
| 404 | `node_not_found` | work_id not in map |
| 404 | `work_not_found` | Paper not found in any data source |
| 500 | `internal_error` | LLM or upstream failure |

---

## Feature 4: Methodology Comparison

**Prefix:** `/v1`
**Role:** Provides structured side-by-side comparison of research methodologies across 2–4 papers. Extracts methodology fingerprints, analyzes citation lineage, identifies convergence/divergence, and generates actionable recommendations.

### `POST /v1/maps/{map_id}/compare-methodologies`

**When to call on the frontend:** When the user selects 2–4 papers from the citation map and clicks "Compare Methodologies."

#### Pipeline / Workflow

```
map_id + work_ids (2-4 papers)
  │
  ├─ Validate: papers in map, unique, 2-4 count
  ├─ Check comparison cache
  │
  ├─ If not cached:
  │    ├─ Fetch paper content (S2 full text or abstract fallback)
  │    ├─ Build citation lineage
  │    │    ├─ Direct citations between selected papers
  │    │    └─ Shared references
  │    │
  │    ├─ LLM Call 1: Extract methodology fingerprints
  │    │    ├─ Approach, data requirements, validation method
  │    │    ├─ Domain, novelty over prior work
  │    │    └─ Assumptions, limitations, key components
  │    │
  │    ├─ LLM Call 2: Synthesize comparison
  │    │    ├─ Convergence/divergence analysis
  │    │    ├─ Strengths/weaknesses matrix
  │    │    └─ Recommendation with decision matrix
  │    │
  │    └─ Cache results
  │
  ▼
Return full methodology comparison
```

#### Request

```json
{
  "work_ids": ["W2100837269", "W2165150801", "W2963864421"]
}
```

> **Constraints:** 2–4 unique work_ids, all must exist in the specified map.

#### Response

```json
{
  "work_ids": ["W2100837269", "W2165150801", "W2963864421"],
  "papers": [
    {
      "work_id": "W2100837269",
      "title": "Continuous control with deep reinforcement learning",
      "year": 2015,
      "source_quality": "abstract_only",
      "methodology_fingerprint": {
        "approach": "Off-policy actor-critic with experience replay and target networks",
        "data_requirements": "Simulated environments with continuous action spaces",
        "validation_method": "Benchmark tasks in MuJoCo physics simulator",
        "domain": "Continuous control, robotics",
        "novelty_over_prior": "Combines DQN stability techniques with deterministic policy gradients",
        "assumptions": ["Stationary environment", "Dense reward signals"],
        "limitations": ["Sensitive to hyperparameters", "Sample inefficient"],
        "key_components": ["Actor network", "Critic network", "Experience replay buffer", "Target networks"]
      }
    }
  ],
  "lineage": {
    "direct_citations": [
      {
        "from_work_id": "W2100837269",
        "to_work_id": "W2165150801",
        "context": "Builds upon deterministic policy gradient theorem"
      }
    ],
    "shared_references": [
      {
        "reference_work_id": "W2076063813",
        "shared_by": ["W2100837269", "W2963864421"],
        "context": "Both cite DQN as foundational architecture"
      }
    ],
    "evolution_chain": "W2165150801 (2014) → W2100837269 (2015) → W2963864421 (2015)"
  },
  "convergence_divergence": {
    "common_problem": {
      "domain": "Reinforcement Learning",
      "challenge": "Learning control policies for continuous action spaces",
      "why_hard": "High-dimensional continuous action spaces make Q-learning intractable"
    },
    "paradigms": [
      {
        "name": "Deterministic Policy Gradient",
        "mechanism": "Learn a deterministic mapping from states to actions",
        "philosophy": "Policy gradient in continuous spaces via chain rule",
        "papers": ["W2165150801", "W2100837269"]
      }
    ],
    "divergence_summary": "While all three papers address continuous control, they diverge in..."
  },
  "strengths_weaknesses_matrix": [
    {
      "work_id": "W2100837269",
      "title": "Continuous control with deep reinforcement learning",
      "handles_well": [
        { "capability": "High-dimensional continuous control", "evidence": "..." }
      ],
      "struggles_with": [
        { "limitation": "Sample efficiency", "evidence": "..." }
      ],
      "assumptions": [
        { "assumption": "Stationary dynamics", "risk": "Fails under distribution shift" }
      ],
      "complemented_by": [
        { "work_id": "W2963864421", "how": "Addresses sample efficiency via..." }
      ]
    }
  ],
  "recommendation": {
    "summary": "DDPG is best for simple continuous control; SAC for robustness...",
    "decision_matrix": [
      {
        "scenario": "Simple continuous control with known dynamics",
        "recommended_work_id": "W2100837269",
        "reasoning": "Simpler architecture, sufficient for well-defined environments"
      }
    ],
    "can_combine": true,
    "combination_notes": "DDPG's replay buffer can be combined with..."
  },
  "confidence": "high"
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_input` | <2 or >4 work_ids, or duplicates |
| 403 | `permission_error` | Map not owned by tenant |
| 404 | `not_found` | Paper not in map |
| 500 | `internal_error` | LLM or upstream failure |

---

## Feature 5: Research Gap Analysis

**Prefix:** `/v1`
**Role:** Detects research gaps by synthesizing insights from all other features (citation graph, ranking, timeline, methodology, novelty). Requires the user to have explored the map sufficiently (≥50% coverage) before unlocking. Validates detected gaps against external literature using web search.

---

### `GET /v1/maps/{map_id}/gap-analysis/status`

**When to call on the frontend:** When rendering the gap analysis section/tab to determine if it's unlocked or show a progress bar toward the 50% threshold.

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Map ID |

#### Response

```json
{
  "unlocked": false,
  "coverage_pct": 35.0,
  "breakdown": {
    "map_wide_timeline": 25.0,
    "methodology_comparisons": 10.0,
    "per_node_exploration": 0.0,
    "methodology_comparison_count": 1,
    "nodes_explored": 0,
    "total_nodes": 45
  },
  "message": "Explore more of the map to unlock gap analysis. Current coverage: 35%."
}
```

---

### `POST /v1/maps/{map_id}/gap-analysis`

**When to call on the frontend:** When the user clicks "Run Gap Analysis" (only available when `unlocked: true`).

#### Pipeline / Workflow

```
map_id (coverage ≥ 50%)
  │
  ├─ Check cache (return if force_refresh=false)
  │
  ├─ Gap Detection (internal heuristics):
  │    ├─ Citation graph → structural gaps (disconnected clusters)
  │    ├─ Ranked list → coverage gaps (underrepresented topics)
  │    ├─ Timeline → temporal gaps (dormant research areas)
  │    ├─ Methodology comparisons → methodological gaps (untried combinations)
  │    └─ Novelty assessments → novelty gaps (unexplored directions)
  │
  ├─ Gap Synthesis (LLM):
  │    ├─ Generate descriptions with inline citations
  │    └─ Suggest research directions
  │
  ├─ External Validation (GPT-5.2 + web search):
  │    ├─ For each gap: search literature
  │    ├─ Classify as "open", "partial", or "addressed"
  │    └─ Return sources and coverage estimate
  │
  └─ Cache results
  │
  ▼
Return validated gap cards
```

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Map ID |
| `force_refresh` | bool | query | Bypass cache (default: false) |

#### Response

```json
{
  "job_id": "c6c022d8-...",
  "gaps": [
    {
      "gap_id": "gap_1",
      "type": "methodological",
      "type_explanation": "Methods that haven't been combined or applied to certain domains",
      "title": "Empirical Validation of Theoretical Bounds for Sample Efficiency in RL",
      "description": "Multiple foundational papers (W2076063813, W2119717200) propose theoretical models...",
      "evidence": [
        {
          "work_id": "W2076063813",
          "authors": "Jürgen Schmidhuber",
          "year": 2014,
          "title": "Deep learning in neural networks: An overview",
          "role": "Historical overview without addressing sample efficiency in RL"
        }
      ],
      "suggested_direction": "Conduct controlled experiments comparing theoretical sample complexity bounds...",
      "confidence": 0.64,
      "validation_result": {
        "status": "partial",
        "reasoning": "Some theory papers include empirical results, but the specific program described...",
        "sources": [
          {
            "url": "https://arxiv.org/abs/2312.08369",
            "year": 2023,
            "title": "The Effective Horizon Explains Deep RL Performance",
            "relevance": "Closest match in spirit..."
          }
        ],
        "coverage_pct": 35.0
      },
      "detection_score": 0.79,
      "data_sources_used": ["abstracts", "methodology_fingerprints", "novelty_assessments"]
    }
  ],
  "coverage_pct": 50.6,
  "data_sources_used": ["citation_graph", "timeline", "methodology", "novelty"],
  "total_candidates_detected": 3,
  "candidates_validated": 2
}
```

#### Gap Types

| Type | Description |
|------|-------------|
| `structural` | Disconnected clusters or missing citation links in the graph |
| `coverage` | Topics mentioned in the area but underrepresented in papers |
| `temporal` | Research areas that went dormant or lack recent follow-up |
| `methodological` | Methods not yet combined or applied to certain domains |
| `novelty` | Unexplored research directions identified from novelty assessments |

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_input` | ValueError |
| 403 | `not_unlocked` | Coverage < 50% |
| 404 | `map_not_found` | Invalid map_id |
| 500 | `internal_error` | LLM or web search failure |

---

### `GET /v1/maps/{map_id}/gap-analysis/results`

**When to call on the frontend:** To retrieve previously computed gap analysis results without re-running. Use this when the user navigates back to the gap analysis view.

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Map ID |

#### Response

Same structure as `POST .../gap-analysis`. Returns `null` if no analysis has been run.

---

### `POST /v1/maps/{map_id}/track-feature`

**When to call on the frontend:** Automatically, whenever the user uses a feature that contributes to gap analysis coverage. Call in the background (fire-and-forget) after:
- Viewing the map-wide timeline → `feature_type=timeline_map_wide`
- Viewing a per-node timeline → `feature_type=timeline_per_node`
- Viewing node details/ranked list → `feature_type=ranked_list_per_node`
- Viewing citation graph for a node → `feature_type=citation_graph_per_node`

#### Request

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `map_id` | UUID | path | Map ID |
| `feature_type` | string | query | One of: `timeline_map_wide`, `timeline_per_node`, `ranked_list_per_node`, `citation_graph_per_node` |
| `work_id` | string | query | Required for per-node features |

#### Coverage Weights

| Feature | Max Contribution |
|---------|-----------------|
| `timeline_map_wide` | 25% |
| `methodology_comparisons` | 30% (tracked via Feature 4 calls) |
| `per_node_exploration` | 30% (sum of per-node features) |

#### Response

```json
{
  "tracked": true,
  "coverage_pct": 45.0,
  "unlocked": false,
  "message": "Coverage updated. 5% more to unlock gap analysis."
}
```

#### Errors

| Status | Code | Cause |
|--------|------|-------|
| 400 | `invalid_feature_type` | Unrecognized feature_type |
| 400 | `missing_work_id` | Per-node feature without work_id |
| 404 | `map_not_found` | Invalid map_id |

---

## Settings: Institutional Access

**Prefix:** `/v1/settings`
**Role:** Allows users to configure their institutional proxy and LibKey settings for accessing paywalled papers.

---

### `GET /v1/settings/institutional-access`

**When to call on the frontend:** When rendering the settings page to populate current institutional access configuration.

#### Response

```json
{
  "institutional_proxy_prefix": "https://proxy.university.edu/login?url=",
  "has_libkey": true,
  "libkey_library_id": "1234"
}
```

> **Note:** The `libkey_api_key` is never returned — only a boolean `has_libkey` flag.

---

### `PUT /v1/settings/institutional-access`

**When to call on the frontend:** When the user saves their institutional access settings.

#### Request

```json
{
  "institutional_proxy_prefix": "https://proxy.university.edu/login?url=",
  "libkey_api_key": "sk-...",
  "libkey_library_id": "1234"
}
```

#### Response

```json
{
  "status": "ok"
}
```

---

## Cross-Feature Dependencies

The features form a directed pipeline — understanding this flow is critical for frontend state management:

```
Feature 1: Citation Map
    │
    ├── graph_draft_id ──→ Feature 2: Build Map (maps/build)
    │                           │
    │                           ├── map_id ──→ Feature 2: Get Map Render (maps/{map_id})
    │                           ├── map_id ──→ Feature 3: Node Details
    │                           ├── map_id ──→ Feature 4: Methodology Comparison
    │                           └── map_id ──→ Feature 5: Gap Analysis
    │
    └── (standalone) ──→ Feature 2: Direct Ranking (rank)
                              │
                              ├── rank_job_id ──→ Feature 2: Generate Subtopics
                              │                       │
                              │                       └── drill_down_query ──→ Feature 2: Drill-Down Ranking
                              │
                              └── rank_job_id ──→ Feature 2: Temporal Map
```

### Frontend Call Sequence (Typical Flow)

1. **User enters query** → `POST /v1/citation-map` → get `graph_draft_id`
2. **Build map** → `POST /v1/maps/build` with `graph_draft_id` → get `map_id`
3. **Render map** → `GET /v1/maps/{map_id}` → render nodes/edges
4. **User clicks node** → `GET /v1/maps/{map_id}/nodes/{work_id}/details` → show detail panel
5. **Track exploration** → `POST /v1/maps/{map_id}/track-feature` (fire-and-forget)
6. **User selects papers** → `POST /v1/maps/{map_id}/compare-methodologies` → show comparison
7. **User opens ranked list** → `POST /v1/rank` with query → get `rank_job_id`
8. **If broad query** → `POST /v1/rank/{rank_job_id}/generate-subtopics` → show subtopic chips
9. **User clicks subtopic** → `POST /v1/rank/drill-down` with `drill_down_query`
10. **User opens timeline** → `GET /v1/rank/{rank_job_id}/temporal-map`
11. **Check gap readiness** → `GET /v1/maps/{map_id}/gap-analysis/status`
12. **Run gap analysis** → `POST /v1/maps/{map_id}/gap-analysis`

---

## Summary

| # | Feature | Endpoint | Method | Path |
|---|---------|----------|--------|------|
| 1 | Citation Map | Build Citation Map | POST | `/v1/citation-map` |
| 2 | Maps | Build Map | POST | `/v1/maps/build` |
| 3 | Maps | Get Map Render | GET | `/v1/maps/{map_id}` |
| 4 | Ranking | Direct Ranking | POST | `/v1/rank` |
| 5 | Ranking | Drill-Down Ranking | POST | `/v1/rank/drill-down` |
| 6 | Ranking | Generate Subtopics | POST | `/v1/rank/{rank_job_id}/generate-subtopics` |
| 7 | Ranking | Temporal Map | GET | `/v1/rank/{rank_job_id}/temporal-map` |
| 8 | Node Details | Get Node Details | GET | `/v1/maps/{map_id}/nodes/{work_id}/details` |
| 9 | Methodology | Compare Methodologies | POST | `/v1/maps/{map_id}/compare-methodologies` |
| 10 | Gap Analysis | Get Status | GET | `/v1/maps/{map_id}/gap-analysis/status` |
| 11 | Gap Analysis | Run Analysis | POST | `/v1/maps/{map_id}/gap-analysis` |
| 12 | Gap Analysis | Get Results | GET | `/v1/maps/{map_id}/gap-analysis/results` |
| 13 | Gap Analysis | Track Feature | POST | `/v1/maps/{map_id}/track-feature` |
| 14 | Settings | Get Institutional Access | GET | `/v1/settings/institutional-access` |
| 15 | Settings | Update Institutional Access | PUT | `/v1/settings/institutional-access` |
