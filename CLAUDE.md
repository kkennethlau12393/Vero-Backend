# Vero Backend — Project Context

## What This Is

A FastAPI backend for a research paper discovery platform. Users search for academic topics, get ranked results, visualize citation networks as maps, compare paper methodologies, and find research gaps.

## Tech Stack

- **Framework:** FastAPI (Python 3.11+, sync — zero async/await in codebase)
- **Database:** PostgreSQL via Supabase, SQLAlchemy 2.0 (sync engine, `pool_pre_ping=True`)
- **LLMs:** Groq API (Llama-4 Maverick, Qwen3-32B) via both `groq` and `openai` SDKs
- **External APIs:** OpenAlex, Semantic Scholar, ArXiv, CrossRef, PubMed, DBLP
- **Auth:** Tenant-based (`get_tenant_id()` dependency)
- **Config:** `.env` file loaded independently by every feature module

## Project Structure

```
app/
  main.py                    # FastAPI app, CORS, global exception handler, routers
  db.py                      # SQLAlchemy engine factory (make_engine)
  auth/tenant.py             # Tenant ID extraction dependency
  feature1/                  # Citation Map — graph-based paper network
    api.py, schemas.py, citation_map_service.py (2314 lines)
  feature2/                  # Scholar Search & Ranking — LLM-scored paper retrieval
    rank_api.py, maps_api.py, schemas.py
    rank_service.py (937 lines), retrieval.py (2525 lines)
    llm_relevance.py, query_expansion.py, repos.py, ...
  feature3/                  # Node Details — paper analysis with novelty assessment
    node_details_api.py, schemas.py, node_details_service.py (1747 lines)
    grounding_supplement.py (1629 lines), paper_cache.py, ...
  feature4/                  # Paper Compare — methodology comparison
    compare_api.py, schemas.py, compare_service.py (1151 lines)
  feature5/                  # Research Gaps — gap detection and validation
    gap_api.py, schemas.py, gap_service.py (1190 lines)
  settings/                  # Institutional access, LibKey integration
    api.py, store.py, access_links.py
tests/
  conftest.py                # Benchmark infrastructure + TestClient fixtures
  test_integration.py        # Integration tests for all endpoints
  reports/                   # Auto-generated benchmark JSON reports
```

## API Endpoints

All endpoints use `/v1/` prefix. Key routes:

| Method | Path | Feature | Notes |
|--------|------|---------|-------|
| GET | `/health` | System | Added in Phase 1 |
| POST | `/v1/citation-map` | Feature1 | Builds citation graph from query or seed paper |
| POST | `/v1/maps/build` | Feature2 | Builds map from graph_draft_id |
| GET | `/v1/maps/{map_id}` | Feature2 | Full map payload (nodes, edges, groups) |
| POST | `/v1/rank` | Feature2 | LLM-ranked paper search (can return 202 async) |
| GET | `/v1/rank/{id}/status` | Feature2 | Poll async rank job status (added Phase 1) |
| POST | `/v1/rank/drill-down` | Feature2 | Lightweight subtopic ranking |
| POST | `/v1/rank/{id}/generate-subtopics` | Feature2 | Cluster ranked results |
| GET | `/v1/rank/{id}/temporal-map` | Feature2 | Papers by era |
| GET | `/v1/maps/{id}/nodes/{wid}/details` | Feature3 | Paper analysis + novelty |
| POST | `/v1/maps/{id}/compare-methodologies` | Feature4 | Compare 2-4 papers |
| POST | `/v1/maps/{id}/gap-analysis` | Feature5 | Trigger gap detection |
| GET | `/v1/maps/{id}/gap-analysis/status` | Feature5 | Poll gap analysis |
| GET | `/v1/maps/{id}/gap-analysis/results` | Feature5 | Get gap results |

## Critical Architecture: Rank vs Citation Maps Are Disconnected

**The biggest architectural issue.** Rank results have NO path to a map:

```
Citation Map → graph_draft → map → node details / compare / gaps  ✅
Rank         → rank_results → ???  (dead end)                     ❌
```

All downstream features (Feature3-5) require a `map_id`, but the rank pipeline never creates a `graph_draft`. A bridge endpoint (`POST /v1/rank/{id}/build-graph-draft`) is needed. See `AUDIT.md` Section 1 for full analysis.

## Known Code Quality Issues

Detailed in `AUDIT.md`. Key points:

- **6 god-files** over 1000 lines (worst: `retrieval.py` at 2525)
- **~1500 LOC duplicated** — OpenAlex/S2/ArXiv search code copied across features. `citation_map_service.py` line 8: "Search logic is copied from feature2/retrieval.py"
- **All sync, zero async** — blocking I/O everywhere, ~5 concurrent users max
- **Inconsistent error handling** — 140 broad `except Exception` catches, mixed return values ([], None, {})
- **New ThreadPoolExecutor per request** (up to 12 threads)
- **New LLM client per call** (18+ locations)
- **No cache on citation maps** — every call re-fetches from APIs
- **Features are well-isolated** — zero cross-feature imports (good)

## Caching Architecture

**Rank (Feature2)** — 4-layer DB-backed cache:
1. `rank_jobs` — dedup by params_hash (includes `RANKING_VERSION=rank-v84`)
2. `llm_relevance_cache` — per (paper_id, query_hash, model_version=llm-type-v27)
3. `query_expansion_cache` — per query_hash
4. `query_classification_cache` — per query_hash

**Citation Map (Feature1)** — zero caching. Every call runs the full pipeline.

**Other features** cache LLM results in DB: `node_details_cache`, `methodology_fingerprint_cache`, `methodology_comparison_cache`, `paper_full_text_cache`.

No TTL on any cache. No eviction. Tables grow forever.

## Changes Made This Session

### Phase 1: Frontend Integration Fixes (COMPLETED, architect-approved)

**`app/main.py`:**
- Added `CORSMiddleware` (allow_origins=["*"], TODO: restrict for production)
- Added global `@app.exception_handler(Exception)` returning JSON
- Added `GET /health` endpoint

**`app/feature2/rank_api.py`:**
- Added `RankJobStatusResponse` model
- Added `GET /v1/rank/{rank_job_id}/status` endpoint (tenant-validated)

**`app/feature4/compare_api.py`:**
- Changed `map_id: str` → `map_id: UUID` with `str(map_id)` at service boundary

### Hop-2 Disabled in Citation Maps

**`app/feature1/citation_map_service.py`:**
- Hop-2 expansion disabled via `enable_hop2 = False` flag (line ~1202)
- Reduces API calls from ~25 to ~5, papers fetched from ~1500 to ~200
- Latency drops from 30-60s to 5-10s
- Code preserved behind flag for re-enabling later

### Integration Test Suite Created

**`tests/conftest.py`:** Benchmark infrastructure — auto-records timing, response size, status code per test. Prints table + writes JSON to `tests/reports/`.

**`tests/test_integration.py`:** 21 tests covering all endpoints. Uses `bench` fixture for automatic benchmarking. Markers: `@pytest.mark.slow` for external API tests, `@pytest.mark.db` for tests requiring live DB.

**`pyproject.toml`:** Added `[tool.pytest.ini_options]` with testpaths and markers.

Run: `pytest tests/test_integration.py -m "not slow" -v -s`

## Prioritized Action Plan (from AUDIT.md)

1. ~~Phase 1: Frontend integration fixes~~ ✅ DONE
2. **Phase 2: Extract shared infrastructure** — HTTP clients, LLM client, JSON utils, retry decorator, config centralization (~1 week)
3. **Phase 3: Bridge rank and maps** — `POST /v1/rank/{id}/build-graph-draft` endpoint (~2-3 days)
4. **Phase 4: Split god files** — break down the 6 largest files (~2-3 weeks incremental)
5. **Phase 5: Production hardening** — restrict CORS, rate limiting, request tracing, DB pool config, query timeouts
6. **Phase 6: API v2** — unified sessions concept, evaluate async migration

## Pre-existing Pyright Errors

The codebase has many pre-existing Pyright type errors (especially in `citation_map_service.py` — Optional subscript access, None member access, type mismatches). These are NOT from our changes. Don't fix them unless specifically working on those files.

## Running the App

```bash
# Ensure .env has DATABASE_URL, GROQ_API_KEY, OPENALEX_API_KEY, SEMANTIC_SCHOLAR_API_KEY
uvicorn app.main:app --reload

# Run tests (fast, no DB/API required)
pytest tests/test_integration.py -m "not slow" -v -s

# Run all tests (requires DB + API keys)
pytest tests/test_integration.py -v -s
```

## Key Files to Read First

1. `app/main.py` — entry point, middleware, routers (small, clean)
2. `app/feature2/rank_api.py` — rank endpoints (the main user-facing API)
3. `app/feature1/api.py` — citation map endpoint
4. `app/feature2/schemas.py` — Pydantic models for rank responses
5. `AUDIT.md` — full code quality audit with findings and recommendations

---

## Deep Codebase Context (for future sessions)

This section provides enough detail so future Claude instances can work on any feature without re-reading thousands of lines. Generated via 4-agent parallel exploration on 2026-02-17.

### Feature1: Citation Maps — Deep Dive

**Entry point:** `build_citation_map()` at `citation_map_service.py:2175`

**Two modes:**
1. **Seed Paper Mode** (`seed_work_id` provided) — fetches paper details via `fetch_seed_paper_details()` (:960), then builds network
2. **NL Query Mode** (`query_text` provided) — calls `select_seed_from_query()` (:1822) which searches OpenAlex + S2 + ArXiv in parallel via ThreadPoolExecutor(max_workers=12) (:1849), then uses `_score_seed_candidates()` (:1408) with LLM scoring via Groq to pick the best seed

**Network building (two paths):**
- **Multi-hop** (`use_multi_hop=True`): `_expand_citation_network()` (:1121) — fetches hop-1 citations/refs via S2 API, optionally hop-2 (currently disabled via `enable_hop2 = False` at ~:1202). Uses hybrid scoring: `citation_weight=0.6, connectivity_weight=0.4`. Assembles via `_assemble_multihop_graph()` (:2064)
- **1-hop** (legacy): `fetch_citing_papers()` (:807) + `fetch_references()` (:869) via OpenAlex API, then `_stratified_sample()` (:1015) for diversity, assembled via `_assemble_citation_graph()` (:1989)

**Graph draft creation:** `_create_graph_draft()` (:2120) — writes nodes/edges to DB tables `graph_drafts`, `graph_draft_nodes`, `graph_draft_edges` via raw SQL. Returns UUID. This is what connects citation maps to downstream features (Feature3-5).

**Key external APIs used:**
- OpenAlex: `_search_openalex()` (:71), `_search_openalex_highly_cited()` (:125), `_search_openalex_by_title()` (:179)
- Semantic Scholar: `_search_semantic_scholar()` (:244), `_fetch_citing_papers_s2()` (:612), `_fetch_references_s2()` (:671)
- ArXiv: `_search_arxiv()` (:405)
- LLM (Groq): `_score_seed_candidates()` (:1408) uses `Groq()` client directly, `_expand_search_queries()` (:1601) uses Groq for query expansion

**Constants:** `OPENALEX_LIMIT=100`, `S2_LIMIT=100`, `ARXIV_LIMIT=100`, `MAX_RETRIES=3`, `SEMANTIC_SCHOLAR_DELAY=1.0s`

**Gotchas:**
- Search logic is COPIED from `feature2/retrieval.py` (line 8 comment confirms this) — ~1500 LOC of duplication
- No caching — every call hits external APIs
- New Groq client created per LLM call (:1477, :1618)
- ThreadPoolExecutor(12) created per request in seed selection

### Feature2: Rank Pipeline — Deep Dive

**Entry point:** `direct_rank_prod()` at `rank_service.py:128`

**Pipeline stages:**
1. **Params hashing** — SHA256 of `(tenant_id, RETRIEVAL_PIPELINE_VERSION, query, context, filters, params)` for cache dedup
2. **Candidate set** — `CandidateSetRepo.insert_or_get_candidate_set()` checks for existing set by `params_hash`
3. **Retrieval** — `generate_candidates_direct()` at `retrieval.py:1649` — the main retrieval orchestrator

**Retrieval sources (all co-equal, run in parallel via ThreadPoolExecutor):**
- OpenAlex: `_search_openalex()` (:1552) — ~200 papers/query × 10-15 expanded queries
- OpenAlex Recent (2018+): same function with year filter
- OpenAlex Highly-Cited: `_search_openalex_highly_cited()` (:828)
- Semantic Scholar: `_search_semantic_scholar()` (:106) — uses bulk API (:136) with pagination (1000/page), falls back to regular API (:214)
- S2 Highly-Cited: `_search_s2_highly_cited()` (:287)
- ArXiv: `_search_arxiv()` (:936) with CrossRef enrichment (:1022) for citation counts
- CrossRef: `_search_crossref()` (:358)
- PubMed: `_search_pubmed()` (:433) — uses E-utilities API (esearch + efetch)
- DBLP: `_search_dblp()` (:532)

**Retrieval constants:** `SEMANTIC_SCHOLAR_LIMIT=2000`, `ARXIV_LIMIT=500`, `CROSSREF_LIMIT=100`, `PUBMED_LIMIT=100`, `DBLP_LIMIT=100`, `DEFAULT_LIMIT_POOL=3000` (cap after dedup)

**Post-retrieval pipeline (back in rank_service.py):**
4. **Query expansion** — `query_expansion.py` uses Groq LLM to generate expanded queries
5. **Query classification** — `query_classification.py` classifies query type and specificity
6. **LLM relevance scoring** — `llm_relevance.py` scores papers via Groq in waves (`score_wave()`) with convergence detection (`convergence.py`)
7. **Feature computation** — `features.py` computes: semantic similarity, lexical match, topic relevance, impact (Bayesian), recency, completeness
8. **Reranking** — `rerank.py` assembles final ranked results with reasons and categories
9. **Paper classification** — `paper_classification.py` categorizes as foundational/methodology/application/tool/textbook
10. **Methodological alignment** — `methodological_alignment.py` partitions by category with limits

**Version strings:**
- `RANKING_VERSION = "rank-v84"` (rank_service.py:80) — invalidates rank caches
- `PIPELINE_VERSION` in retrieval.py — invalidates candidate sets
- LLM prompt version embedded in ranking version comments (currently v27)
- `MODEL_VERSION` varies per module (Llama-4 Maverick or Qwen3-32B)

**Caching (4 layers, all DB-backed, no TTL):**
1. `rank_jobs` table — keyed by `params_hash` (includes RANKING_VERSION)
2. `llm_relevance_cache` — keyed by `(paper_id, query_hash, model_version)`
3. `query_expansion_cache` — keyed by `query_hash`
4. `query_classification_cache` — keyed by `query_hash`

**Async job handling:** `rank_api.py` returns 202 with `rank_job_id` for long-running queries. Status polled via `GET /v1/rank/{id}/status`.

**Feature2 auxiliary files:**
- `repos.py` — `CandidateSetRepo`, `RankRepo` (raw SQL data access)
- `work_topic_store.py` — `WorkStore` (paper storage/dedup), `WorkForMap` (dataclass), `TopicHierarchyStore`
- `maps_api.py` / `maps_service.py` — build maps from graph_drafts
- `subtopic_service.py` — cluster ranked results into subtopics
- `temporal_analytics.py` / `temporal_map_service.py` — papers-by-era analysis
- `tfidf_similarity.py` — TF-IDF vector computation for diversity
- `title_to_query.py` — convert paper title to search query
- `domain_filters.py` — domain-specific filtering rules
- `lexical_retrieval.py` — lexical search utilities

### Feature3: Node Details — Deep Dive

**Purpose:** Generate detailed analysis for a single paper node in the citation map.

**Entry:** `node_details_service.py` — `get_node_details()` function
- Requires `map_id` + `work_id` (paper identifier)
- Fetches paper metadata from map's graph_draft nodes
- Uses LLM (Llama-4 Maverick via Groq/OpenAI SDK) for summary, keywords, novelty assessment
- `MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"`
- `ASSESSMENT_VERSION = "maverick-v4"` — cache invalidation key

**Key sub-modules:**
- `grounding_supplement.py` (1629 lines) — fetches additional grounding papers for novelty claims, uses ThreadPoolExecutor(5) and ThreadPoolExecutor(3)
- `paper_cache.py` — `paper_full_text_cache` table for caching fetched paper content
- `abstract_enrichment.py` — ensures papers have valid abstracts
- `methodology.py` — extracts methodology fingerprints
- `landmark_retrieval.py` — finds landmark/seminal papers in the field
- `node_timeline.py` — builds chronological timeline for a paper's context
- `topic_inference.py` — infers topic from paper when not available
- `paper_impact_analytics.py` — citation impact analysis
- `reference_store.py` — manages paper reference relationships

**Caching:** `node_details_cache` table (DB-backed, no TTL)

### Feature4: Paper Compare — Deep Dive

**Purpose:** Compare methodologies of 2-4 papers side by side.

**Entry:** `compare_service.py` — `compare_methodologies()` function
- Requires `map_id` + list of `work_ids` (2-4 papers)
- Uses LLM (Groq via OpenAI SDK) for methodology extraction and comparison
- `get_groq_client()` (:71) creates OpenAI client with Groq base URL
- Uses ThreadPoolExecutor(2) for parallel paper content fetching (:1005)

**Key sub-modules:**
- `paper_content.py` — fetches full paper content, uses ThreadPoolExecutor(4) for parallel S2 API calls
- `citation_lineage.py` — traces citation relationships between compared papers

**Caching:** `methodology_fingerprint_cache`, `methodology_comparison_cache` (DB-backed, no TTL)

### Feature5: Research Gaps — Deep Dive

**Purpose:** Detect research gaps in a citation map's coverage area.

**Entry:** `gap_service.py` — `run_gap_analysis()` function (async job pattern)
- Requires `map_id`
- Phase 1: `gap_detection.py` — heuristic gap detection from paper metadata patterns
- Phase 2: LLM synthesis via Groq (Qwen3-32B model: `"qwen/qwen3-32b"`)
- Phase 3: `external_validation.py` — validates gaps against external sources, uses ThreadPoolExecutor(MAX_VALIDATION_CONCURRENCY)

**Special LLM handling:** Strips `<think>...</think>` tags from Qwen3-32B thinking model responses (:76-79)

**Constants:** `MIN_CONFIDENCE_THRESHOLD=0.50`, `MIN_DETECTION_SCORE=0.40`, `MAX_COVERAGE_PCT=60.0`, `CACHE_TTL_HOURS=24` (the ONLY feature with TTL!)

**Caching:** Gap analysis results stored in DB with status tracking (pending/running/complete/failed)

### Infrastructure Patterns

**Database (`app/db.py`):**
- `make_engine()` — creates SQLAlchemy engine with `NullPool` (no connection pooling, new connection per request)
- Uses `DATABASE_URL_POOLER` env var (Supabase pgBouncer URL)
- Every feature module calls `make_engine()` independently

**Auth (`app/auth/tenant.py`):**
- `get_tenant_id()` — dev stub, reads `DEV_TENANT_ID` from env, returns UUID
- Single-tenant only, no real auth middleware

**LLM Client Pattern (18+ locations):**
- Two patterns: `Groq(api_key=...)` directly (feature1, feature2/llm_relevance) or `OpenAI(api_key=..., base_url="https://api.groq.com/openai/v1")` (everything else)
- New client created per function call — no shared client
- Models used: `meta-llama/llama-4-maverick-17b-128e-instruct` (fast, most features) and `qwen/qwen3-32b` (feature5 gaps, feature2 query classification)

**ThreadPoolExecutor Pattern:**
- New executor per request, never shared/pooled
- Max workers vary: 2 (compare), 3-5 (grounding), 10 (retrieval enrichment), 12 (citation map seed selection)
- All features use this for parallel API calls

**.env Loading:**
- `load_dotenv()` called independently in every module: main.py, citation_map_service.py, retrieval.py, node_details_service.py, gap_service.py, etc.
- Pattern: `load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")`

**Error Handling:**
- ~140 broad `except Exception` catches across codebase
- Mixed return patterns: `[]`, `None`, `{}` on failure
- Recent fix: nested transaction savepoints added to retrieval.py and work_topic_store.py to prevent `PendingRollbackError` cascading

**Complete file inventory (67 Python files):**
- Feature1: 3 files (api.py, schemas.py, citation_map_service.py)
- Feature2: 18 files (rank_api.py, maps_api.py, rank_service.py, retrieval.py, llm_relevance.py, query_expansion.py, query_classification.py, repos.py, work_topic_store.py, schemas.py, maps_service.py, features.py, rerank.py, convergence.py, tfidf_similarity.py, subtopic_service.py, temporal_analytics.py, temporal_map_service.py, title_to_query.py, domain_filters.py, lexical_retrieval.py, paper_classification.py, methodological_alignment.py)
- Feature3: 13 files (node_details_api.py, node_details_service.py, schemas.py, grounding_supplement.py, paper_cache.py, abstract_enrichment.py, methodology.py, json_utils.py, landmark_retrieval.py, node_timeline.py, reference_store.py, topic_inference.py, topic_lookup.py, paper_identity.py, paper_impact_analytics.py)
- Feature4: 5 files (compare_api.py, compare_service.py, schemas.py, paper_content.py, citation_lineage.py)
- Feature5: 6 files (gap_api.py, gap_service.py, schemas.py, gap_detection.py, external_validation.py, coverage_tracker.py, prompts.py)
- Infrastructure: 5 files (main.py, db.py, auth/tenant.py, settings/api.py, settings/store.py, settings/access_links.py)
