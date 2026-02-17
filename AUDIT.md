# Vero Backend — Code Quality & Architecture Audit

**Date:** 2026-02-16
**Codebase:** ~27,663 lines across 67 Python files (FastAPI)
**Reviewers:** Architecture, API Contract, Performance, Code Patterns, and Architecture (rank/maps coupling)

---

## Executive Summary

The Vero Backend is a capable research paper discovery platform with well-isolated feature modules and good observability. However, it has significant technical debt that will slow frontend integration and future development:

- **6 god-files** exceeding 1,000 lines (worst: 2,525 lines)
- **~1,500 lines of duplicated code** across features (HTTP clients, LLM init, JSON parsing)
- **Zero CORS or global error handling** *(fixed in Phase 1)*
- **No async/await anywhere** — all I/O is blocking
- **Rank and citation map pipelines are completely disconnected** — rank results can't reach maps, compare, or gap analysis
- **Inconsistent error handling**, retry logic, and caching patterns across features

The feature modules themselves are well-isolated (zero cross-feature imports), logging is thorough (458 logger calls), and type hints/Pydantic schemas are used throughout. The foundation is solid — it needs infrastructure extraction and a bridge between its two core pipelines.

---

## Table of Contents

1. [Architecture: Rank vs Citation Maps](#1-architecture-rank-vs-citation-maps)
2. [API Contract & Frontend Integration](#2-api-contract--frontend-integration)
3. [Architecture & Code Quality](#3-architecture--code-quality)
4. [Performance & Reliability](#4-performance--reliability)
5. [Code Patterns & Duplication](#5-code-patterns--duplication)
6. [Prioritized Action Plan](#6-prioritized-action-plan)
7. [Phase 1 Changes (Completed)](#7-phase-1-changes-completed)

---

## 1. Architecture: Rank vs Citation Maps

### The Problem

The ranking system (feature2) and citation map system (feature1) are **completely disconnected at runtime**. They serve different purposes but share no data path:

```
Citation Map → graph_draft → map → node details / compare / gaps  ✅
Rank         → rank_results → ???  (dead end, no map_id)          ❌
```

All downstream features (node details, methodology comparison, gap analysis) require a `map_id`. But the rank pipeline never creates a `graph_draft`, so rank results are stranded — users get a flat list with no access to the rich visualization and analysis features.

### Current Data Flow

**Path A — Citation Map (Feature1):**
```
POST /v1/citation-map  (query_text or seed_work_id)
  → builds citation graph (nodes + edges from OpenAlex/S2/ArXiv)
  → creates graph_draft (graph_draft_id)
  → returns { nodes, edges, graph_draft_id }
```

**Path B — Rank (Feature2):**
```
POST /v1/rank  (query_text)
  → retrieves candidates from OpenAlex/S2/ArXiv
  → LLM-scores and ranks them
  → returns { rank_job_id, items[], fundamentals[], core[], applications[] }
  ** Does NOT create a graph_draft **
```

**Convergence — Map Builder (Feature2):**
```
POST /v1/maps/build  (graph_draft_id)
  → loads graph_draft nodes + edges
  → enriches with topic/subfield taxonomy
  → returns { map_id }
```

The entity relationship in the database confirms this gap:
```
candidate_sets <── rank_jobs <── rank_results
candidate_sets <─? graph_drafts <── maps <── map_nodes/edges
                                         <── gap_analysis_results
```

The `graph_drafts.candidate_set_id` FK is nullable and **never populated** by Feature1.

### Why They Should Stay Separate

| Feature | Purpose | Latency | Output |
|---------|---------|---------|--------|
| Citation Map | Citation neighborhood around a paper (graph structure) | Fast (~5s) | Nodes + edges |
| Rank | Most relevant papers for a query (scored list) | Slow (30s+ with LLM) | Categorized ranked items |

Merging them into one endpoint would be the wrong abstraction — different latency profiles, different user intents, different data shapes.

### Recommended Fix

**Phase 1 (now): Bridge endpoint**
Add `POST /v1/rank/{rank_job_id}/build-graph-draft` that:
1. Takes the top-N ranked papers from a rank job
2. Fetches citation edges between them (from OpenAlex)
3. Creates a `graph_draft` linked to the rank's `candidate_set_id`
4. Returns `{ graph_draft_id }` → frontend calls `/v1/maps/build`

Frontend flow becomes:
```
POST /v1/rank                           → rank_job_id + ranked list
POST /v1/rank/{id}/build-graph-draft    → graph_draft_id  (NEW)
POST /v1/maps/build                     → map_id
GET  /v1/maps/{map_id}                  → visualization + all downstream features
```

**Phase 2 (later): Unified sessions**
```
POST /v2/sessions { query_text: "..." }
  → { session_id }  (kicks off rank + citation map simultaneously)

GET /v2/sessions/{id}/ranked-list
GET /v2/sessions/{id}/map
GET /v2/sessions/{id}/map/nodes/{wid}
POST /v2/sessions/{id}/compare
POST /v2/sessions/{id}/gaps
```

### Shared Code Duplication

Feature1's `citation_map_service.py` line 8: *"Search logic is copied from feature2/retrieval.py to keep this service self-contained."*

~500 lines of OpenAlex/S2/ArXiv search code are duplicated between the two pipelines. This must be extracted to a shared module regardless of the API design.

---

## 2. API Contract & Frontend Integration

### CRITICAL Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **No CORS middleware** | `main.py` | Frontend cannot call any endpoint |
| 2 | **Inconsistent error response formats** | All API files | Frontend needs per-endpoint error parsing |
| 3 | **Missing request validation on path parameters** | `compare_api.py:37` | `map_id: str` causes 500 instead of 400 on invalid UUID |
| 4 | **No global exception handler** | `main.py` | Frontend receives HTML instead of JSON on errors |
| 5 | **Inconsistent 404 handling** | Multiple APIs | `"map_not_found"` vs `"Map not found"` vs `"candidate_set_not_found"` |
| 6 | **Missing tenant authorization checks** | Service layers | Most services don't verify map ownership consistently |
| 7 | **No rate limiting** | `main.py` | Expensive LLM operations can be spammed |
| 8 | **Async endpoint inconsistency** | `rank_api.py:90-152` | Returns 202 for async jobs but no polling endpoint exists |

> Issues 1, 3, 4, and 8 were **fixed in Phase 1** (see Section 7).

### HIGH Issues

| # | Issue | Impact |
|---|-------|--------|
| 9 | **No pagination on large result sets** | Large maps (>1000 nodes) return >10MB JSON |
| 10 | **Missing field-level documentation** | Frontend devs must read service code to understand fields |
| 11 | **Inconsistent naming conventions** | `work_id` (string) vs `map_id` (UUID); snake_case vs camelCase |
| 12 | **No OpenAPI tags/descriptions** | Auto-generated docs are hard to navigate |
| 13 | **Optional fields without defaults** | Frontend must check null vs undefined vs missing |
| 14 | **No health endpoint** | Frontend can't check if backend is up |

> Issue 14 was **fixed in Phase 1**.

### MEDIUM Issues

| # | Issue | Impact |
|---|-------|--------|
| 15 | Verbose error details in production | Exposes SQL errors and stack traces |
| 16 | No request ID tracing | Hard to debug frontend-reported errors |
| 17 | Missing Content-Type validation | Malformed requests cause unexpected errors |
| 18 | No timestamp fields on responses | Frontend can't show "Last updated" or cache |
| 19 | No API versioning strategy documented | Breaking changes will break frontends |

### Positive Findings

- Good input validation with Pydantic models
- Consistent UUID usage for resource IDs
- Proper HTTP status codes in most places
- Type hints everywhere — schema generation works well
- Modular router structure is clean
- Tenant isolation dependency consistently applied

---

## 3. Architecture & Code Quality

### CRITICAL: God-File Anti-Pattern

| File | Lines | Factor over 500-line max |
|------|-------|-------------------------|
| `feature2/retrieval.py` | 2,525 | 5.0x |
| `feature1/citation_map_service.py` | 2,314 | 4.6x |
| `feature3/node_details_service.py` | 1,747 | 3.5x |
| `feature3/grounding_supplement.py` | 1,629 | 3.3x |
| `feature5/gap_service.py` | 1,190 | 2.4x |
| `feature4/compare_service.py` | 1,151 | 2.3x |

**Root cause:** Service files mix HTTP calls + business logic + caching + LLM prompting + database access + error handling.

**Example:** `feature2/retrieval.py` (2,525 lines) contains:
- OpenAlex API calls (8+ functions)
- Semantic Scholar API calls (4+ functions)
- ArXiv, CrossRef, PubMed, DBLP API calls
- Metadata validation logic
- Citation enrichment
- Database operations
- Query expansion integration

**Recommended split:**
```
feature2/
  retrieval_service.py      # Orchestration only (~300 lines)
  search_executor.py        # Parallel search coordination
  result_merger.py          # Dedup and merge logic
  metadata_enricher.py      # Citation/metadata enrichment
```

### HIGH: Inconsistent Error Handling

- 195 `try:` blocks across 41 files
- 140 `except Exception as e:` catches (overly broad)
- Inconsistent return values on failure: some return `[]`, others `None`, others `{}`
- Inconsistent retry constants across features:

| Feature | MAX_RETRIES | RETRY_BACKOFF |
|---------|-------------|---------------|
| feature1/2 | 3 | 0.5 |
| feature3 | 2 | 0.5 |
| feature4 | 4 | 0.5 |
| feature5 | 3 | 2.0 |

### Module Coupling

**Good news:** Zero cross-feature imports. Features are properly isolated modules.

**Bad news:** The isolation was achieved through code duplication rather than shared abstractions. Within features, coupling is high (feature3's `node_details_service` imports 10+ other feature3 modules).

### Maintainability Assessment

| Metric | Rating |
|--------|--------|
| Cohesion | LOW (god files doing too much) |
| Coupling | MEDIUM (features isolated, internal coupling high) |
| Testability | LOW (giant functions, mixed concerns) |
| Duplication | HIGH (search functions copied across features) |
| Complexity | HIGH (2,500-line files) |

---

## 4. Performance & Reliability

### CRITICAL: All Sync, Zero Async

The entire FastAPI application uses **synchronous handlers**. Search for `async def|await|asyncio\.` returns **zero matches**.

- All external API calls use `requests.get()` (blocking)
- All database operations use synchronous `engine.connect()`
- Each request blocks a worker thread during I/O waits
- With default Uvicorn workers (1-4), concurrent capacity is ~5 users

### CRITICAL: Database Pool Misconfiguration

`app/db.py:9` creates the engine without explicit pool configuration:
```python
return create_engine(url, pool_pre_ping=True, future=True)
```

Uses SQLAlchemy's default `QueuePool` (pool_size=5, max_overflow=10). With ThreadPoolExecutor usage up to 12 workers per request, 15 total connections are insufficient.

**Fix:** If using PgBouncer, configure `NullPool`. Otherwise, increase to `pool_size=20, max_overflow=30`.

### HIGH: ThreadPoolExecutor Per Request

Multiple `ThreadPoolExecutor` instances created per request:
- `feature2/retrieval.py:1884` — 10 workers
- `feature1/citation_map_service.py:1849` — 12 workers
- `feature2/llm_relevance.py:392` — 6 workers

5 concurrent API requests × 10 threads = 50 threads competing for 15 DB connections.

**Fix:** Create module-level singleton executors with bounded pools.

### HIGH: LLM Client Created on Every Call

18+ locations create new `Groq()` or `OpenAI()` client instances per call:
```python
client = Groq(api_key=api_key)  # NEW CLIENT ON EVERY CALL
```

**Fix:** Module-level singleton clients (both are thread-safe).

### HIGH: No Database Query Timeouts

Database queries have no explicit timeouts. Slow queries block threads indefinitely.

**Fix:** `SET statement_timeout = 30000` per connection.

### MEDIUM: In-Memory Dict Caching Without Eviction

Module-level dictionaries used for caching with no size limits or TTL. Memory grows unbounded.

### MEDIUM: Inconsistent Rate Limiting

Feature4 has a custom `_RateLimiter` class using `time.sleep()` (blocks threads). Other features have no rate limiting.

### Positive Observations

- Good timeout configuration on most external API calls (10-30s)
- Database cache tables for LLM results
- Proper context manager usage for connection cleanup
- `pool_pre_ping=True` detects stale connections
- Rate limit detection (429) with backoff

---

## 5. Code Patterns & Duplication

### HTTP Client Duplication (SEVERE)

| API Client | Duplicated In | Est. Lines |
|------------|--------------|------------|
| OpenAlex search | feature1, feature2, feature3 (reference_store, paper_cache), feature5 | ~500 |
| Semantic Scholar | feature1, feature2, feature4 | ~300 |
| ArXiv | feature1, feature2 | ~150 |

Feature1 line 8: *"Search logic is copied from feature2/retrieval.py to keep this service self-contained."*

### LLM Client Duplication (SEVERE)

Every feature initializes its own LLM client differently:

| Feature | Client | Model | Missing Key Behavior |
|---------|--------|-------|---------------------|
| feature1 | `Groq()` | varies | logs warning |
| feature2 | `Groq()` | varies | logs warning |
| feature3 | `OpenAI(base_url=GROQ_URL)` | llama-4-maverick | logs warning |
| feature4 | `OpenAI(base_url=GROQ_URL)` | llama-4-maverick | returns None |
| feature5 | `OpenAI(base_url=GROQ_URL)` | qwen3-32b | **raises RuntimeError** |

### JSON Extraction Duplication (SEVERE)

- `feature3/json_utils.py` — dedicated utility
- `feature5/gap_service.py:70-128` — nearly identical copy
- `feature1` — inline JSON extraction (lines 1490-1509)

All three implement: markdown fence stripping, think-tag removal, balanced bracket finding, fallback parsing.

### Environment Variable Loading

Every feature loads `.env` independently:
```python
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
```

And reads API keys individually:
```python
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
```

### Recommended Shared Module Structure

```
app/
  services/
    external_apis/
      openalex.py              # One client, used everywhere
      semantic_scholar.py
      arxiv.py
    llm/
      client.py                # Singleton Groq/OpenAI client
      json_extraction.py       # One copy of JSON parsing
    retry.py                   # Unified retry decorator
    cache.py                   # Generic DB-backed cache
  config.py                    # Centralized env var loading
```

**Estimated reduction:** ~1,500-2,000 lines (15-20% of codebase).

---

## 6. Prioritized Action Plan

### Phase 1: Make Frontend Work (2-3 days) ✅ COMPLETED

1. ~~Add CORS middleware to `main.py`~~ ✅
2. ~~Add global JSON exception handler~~ ✅
3. ~~Add `/health` endpoint~~ ✅
4. ~~Add `GET /v1/rank/{rank_job_id}/status` for polling~~ ✅
5. ~~Fix `map_id` type (str → UUID) in compare endpoint~~ ✅

### Phase 2: Extract Shared Infrastructure (1 week)

6. Extract HTTP clients to `app/services/external_apis/`
7. Centralize LLM client initialization to `app/services/llm/client.py`
8. Unify JSON extraction to `app/services/llm/json_extraction.py`
9. Create retry decorator in `app/services/retry.py`
10. Centralize config/env loading in `app/config.py`

### Phase 3: Bridge Rank and Maps (2-3 days)

11. Add `POST /v1/rank/{rank_job_id}/build-graph-draft` endpoint
12. Implement citation edge fetching for ranked papers
13. Wire graph_draft creation with `candidate_set_id` linkage

### Phase 4: Split God Files (2-3 weeks, incremental)

14. Split `feature2/retrieval.py` (2,525 lines → ~5 files)
15. Split `feature1/citation_map_service.py` (2,314 lines → ~4 files)
16. Split `feature3/node_details_service.py` (1,747 lines → ~3 files)
17. Standardize error handling with custom exception hierarchy
18. Create module-level singleton ThreadPoolExecutors and LLM clients

### Phase 5: Production Hardening

19. Restrict CORS to actual frontend origins
20. Add rate limiting for expensive LLM operations
21. Add request ID tracing middleware
22. Configure database pool properly (NullPool if PgBouncer)
23. Add database query timeouts
24. Standardize error response format across all endpoints

### Phase 6: API v2 (when product matures)

25. Introduce `research_session` concept
26. Unified entry point running rank + citation map in parallel
27. Session-scoped downstream features
28. Evaluate async migration for high-traffic endpoints

---

## 7. Phase 1 Changes (Completed)

The following changes have been implemented and verified:

### Files Modified

**`app/main.py`**
- Added `CORSMiddleware` with `allow_origins=["*"]` (TODO: restrict in production)
- Added `@app.exception_handler(Exception)` returning JSON `{"detail": "Internal server error"}`
- Added `GET /health` returning `{"status": "ok"}`

**`app/feature2/rank_api.py`**
- Added `RankJobStatusResponse` model
- Added `GET /v1/rank/{rank_job_id}/status` endpoint with tenant validation, 404 handling, and items_count on completion

**`app/feature4/compare_api.py`**
- Changed `map_id: str` → `map_id: UUID` in path parameter
- Added `str(map_id)` conversion at service boundary

### Verification Evidence

- App loads with 21 routes
- CORS preflight returns `access-control-allow-origin` header
- Exception handler registered for `Exception` class
- `GET /health` → 200 `{"status": "ok"}`
- Rank status endpoint route registered
- No new Pyright errors introduced
- Architect review: **APPROVED**

---

## Appendix: Technical Debt Metrics

| Metric | Value |
|--------|-------|
| Total lines | 27,663 |
| Files > 1,000 lines | 6 |
| Duplicated LOC | ~1,500-2,000 |
| `try/except` blocks | 195 |
| Broad `except Exception` catches | 140 |
| Logger calls | 458 |
| `async def` usage | 0 |
| Cross-feature imports | 0 (good) |
| Test files | 595 |
| Estimated refactor effort | 4-6 developer-weeks |
