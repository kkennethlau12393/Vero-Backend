# CLAUDE.md — Alexandria Backend

## Testing

**Always use the existing test infrastructure.** Do not create ad-hoc test files or one-off scripts.

### Structure
```
tests/
├── conftest.py              # DB connection, test client, shared fixtures
├── fixtures/                # saved API/LLM responses for mocking
│   ├── groq_responses/      # saved Groq LLM scoring responses
│   ├── api_responses/       # saved OpenAlex/S2/ArXiv responses
│   └── seed_data.py         # factory functions for test papers
├── unit/                    # pure logic, no DB, no network (<5s)
├── integration/             # uses dev DB, mocks external APIs (~30s)
├── quality/                 # ranking regression tests (~1-2min)
│   └── golden/              # snapshot baselines
└── live/                    # REAL API tests — burns credits! (~5-15min)
    ├── conftest.py          # live-only fixtures, no mocks, real DB + APIs
    ├── results/             # saved JSON results for manual review
    ├── test_retrieval_quality.py   # Stage-1 retrieval volume & diversity
    ├── test_prompt_quality.py      # LLM scoring tier accuracy
    └── test_ranking_e2e.py         # Full /v1/rank endpoint benchmarks
```

### Rules
1. **Unit tests** (`tests/unit/`): No DB, no network. Test pure functions only (scoring math, normalization, dedup, MMR, classification, categorization, temporal logic).
2. **Integration tests** (`tests/integration/`): Use the existing dev Postgres DB. **Mock ALL external API calls** (Groq, OpenAlex, Semantic Scholar, ArXiv, CrossRef, PubMed, DBLP) using `respx` with saved fixtures from `tests/fixtures/`.
3. **Quality tests** (`tests/quality/`): Ranking regression tests. Assert known papers appear in top N, category distribution is reasonable, temporal spread is diverse, scores meet thresholds. Compare against golden snapshots in `tests/quality/golden/`.
4. **Never burn API credits in mocked tests.** Always mock Groq and external academic APIs in unit/integration/quality.
5. **Use `pytest` markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.quality`, `@pytest.mark.live`.
6. **Live tests** (`tests/live/`): Hit real Groq, OpenAlex, S2, ArXiv. **Burns API credits.** Only run when changing retrieval logic, LLM prompts, or ranking weights. Never included in `pytest --tb=short` (require explicit `pytest tests/live/` or `-m live`). Results saved to `tests/live/results/` for manual review and regression detection.

### ⚠️ DO NOT Create Redundant Test Files
**NEVER create new test files that duplicate or overlap with existing ones.** The test files below are the canonical, authoritative test files for each feature. If you need to add, modify, or fix tests, **edit the existing files** — do not create new ones.

**Feature 1 (Citation Map):**
- `tests/unit/test_citation_map.py`
- `tests/integration/test_citation_map.py`
- `tests/quality/test_citation_map_quality.py`
- `tests/live/test_citation_map_e2e.py`
- `tests/live/test_seed_selection_quality.py`
- `tests/fixtures/citation_map_responses.py`

**Feature 2 (Ranking):**
- `tests/unit/test_scoring.py`, `test_dedup.py`, `test_query_expansion.py`, `test_classification.py`, `test_categorization.py`, `test_temporal.py`
- `tests/integration/test_ranking.py`, `test_maps.py`
- `tests/quality/test_ranking_quality.py`, `test_diversity.py`
- `tests/live/test_ranking_e2e.py`, `test_retrieval_quality.py`, `test_prompt_quality.py`, `test_new_queries_batch.py`
- `tests/fixtures/groq_responses/`, `tests/fixtures/api_responses/`, `tests/fixtures/seed_data.py`

**If a test file already exists for a feature, use it. Do not create alternatives, copies, or "v2" files. Keep the codebase clean.**

### Running Tests — Feature 2 (Ranking)
```bash
pytest tests/unit/test_scoring.py tests/unit/test_dedup.py tests/unit/test_query_expansion.py tests/unit/test_classification.py tests/unit/test_categorization.py tests/unit/test_temporal.py -x -v   # while coding
pytest tests/integration/test_ranking.py tests/integration/test_maps.py -x -v    # before committing
pytest tests/quality/test_ranking_quality.py tests/quality/test_diversity.py -v   # before shipping

# Live tests — BURNS API CREDITS
pytest tests/live/test_ranking_e2e.py -v -s --timeout=300
pytest tests/live/test_ranking_e2e.py -k "transformers" -v -s   # single query
```

### Running Tests — Feature 1 (Citation Map)
```bash
pytest tests/unit/test_citation_map.py -x -v           # while coding (~2s)
pytest tests/integration/test_citation_map.py -x -v    # before committing (~15s)
pytest tests/quality/test_citation_map_quality.py -v   # before shipping (~30s)

# Live tests — BURNS API CREDITS
pytest tests/live/test_citation_map_e2e.py -v -s --timeout=300          # full e2e
pytest tests/live/test_seed_selection_quality.py -v -s --timeout=300     # seed quality
pytest tests/live/test_citation_map_e2e.py -k "transformers" -v -s      # single query
```

### Running All Tests (except live)
```bash
pytest --tb=short                  # everything (except live)
```

### Adding New Tests
- New pure logic function? → `tests/unit/`
- New endpoint or pipeline change? → `tests/integration/`
- Changed ranking behaviour? → Update `tests/quality/` + golden snapshots
- Changed retrieval sources, LLM prompts, or ranking weights? → Run `tests/live/` and compare results

### Fixtures
When mocking external APIs, save real responses once to `tests/fixtures/` and load them in tests. This ensures deterministic, reproducible results.

## Code Style

- **No hard-coded fixes.** Always find the elegant, general solution — whether deterministic logic or better prompt engineering.
- **No ORM models** — raw SQL via `sqlalchemy.text()` is the convention. Keep it.
- **Cache everything** — LLM calls, API responses, computed results. Use version-gated cache keys.
- **Feature-based modules** — `app/feature1/`, `app/feature2/`, etc. Keep API routes in `*_api.py`, business logic in `*_service.py`.

## Ranking Output Quality Scoring

After any live test run or ranking pipeline change, score results using `tests/live/ranking_output_rubric.md` (100 points total).

### Automated Scoring (35 pts) — run programmatically:
- **Deduplication** (5 pts) — check for duplicate titles in results
- **Category Distribution** (10 pts) — are relevant categories populated and balanced?
- **Temporal Diversity** (10 pts) — check year spread across results
- **Source Diversity** (5 pts) — check provenance across retrieval sources
- **Score Calibration** (5 pts) — check score distribution/spread is meaningful

### LLM-Judged Scoring (65 pts) — use your own LLn knowledge/judgement to evaluate:
- **Relevance** (30 pts) — are the top 10 results relevant to the query?
- **Foundational Coverage** (15 pts) — are the landmark/seminal papers present?
- **Ranking Order** (15 pts) — does the ordering make sense? Best papers on top?
- **Paper Type Accuracy** (5 pts) — are papers in the correct categories?

### Workflow
1. Run live tests → results saved to `tests/live/results/`
2. Automated scoring runs on objective dimensions (35 pts)
3. LLM scoring evaluates subjective dimensions (65 pts)
4. Combined score logged with breakdown per query
5. Flag anything below 70/100 for investigation
6. Compare scores across pipeline versions to catch regressions

### When to Score
- After changing the ranking pipeline version (v84 → v85)
- After changing the LLM prompt version (v27 → v28)
- After changing retrieval logic or weights
- Before shipping a new version

### Query Tracker
Read `tests/live/TESTED_QUERIES.md` before generating new test queries. It tracks all tested queries, domains covered, and coverage gaps. Avoid duplicates, fill gaps.

## Citation Map Output Quality Scoring (Feature 1)

After any live test run or citation map pipeline change, score results using `tests/live/citation_map_rubric.md` (100 points total).

### Automated Scoring (35 pts) — computed programmatically on every test run:
- **Graph Structure** (10 pts) — node count, edge count, connectivity, no orphans
- **Deduplication** (5 pts) — no duplicate work_ids
- **Hop Distribution** (5 pts) — mix of hop-0, hop-1, hop-2 nodes
- **Year Diversity** (5 pts) — temporal spread across decades
- **Citation Diversity** (5 pts) — mix of highly-cited and emerging papers
- **Edge Correctness** (5 pts) — valid endpoints, correct direction, no self-loops

### LLM-Judged Scoring (65 pts) — **REQUIRED on every test run**, use your own knowledge/judgement to evaluate:
- **Seed Relevance** (20 pts) — is the seed paper the best choice for this query?
- **Network Relevance** (20 pts) — are connected papers topically relevant, not citation noise?
- **Foundational Coverage** (10 pts) — are the key seminal papers in the graph?
- **Discovery Value** (10 pts) — does the graph surface interesting/non-obvious papers?
- **Relationship Quality** (5 pts) — do edges represent meaningful citation relationships?

### Workflow (BOTH automated AND LLM-judged scoring required every time)
1. Run live tests → results saved to `tests/live/results/`
2. Automated scoring runs automatically (35 pts) — computed by `_compute_automated_score()` in test file
3. **LLM-judged scoring MUST be performed manually on all results (65 pts)** — review saved results and score each dimension
4. Combined score (100 pts total) logged with breakdown per query
5. Flag anything below 70/100 for investigation
6. Compare scores across pipeline versions to catch regressions

### When to Score
- After changing seed selection logic or multi-hop expansion
- After changing citation API sources or weights
- After changing `_score_seed_candidates()` or `_expand_citation_network()`
- Before shipping a new version

### Running F1 Live Tests
```bash
pytest tests/live/test_citation_map_e2e.py -v -s --timeout=300          # full e2e
pytest tests/live/test_seed_selection_quality.py -v -s --timeout=300     # seed quality
pytest tests/live/test_citation_map_e2e.py -k "transformers" -v -s      # single query
```

## Feature 1: Citation Map Pipeline

### Input Modes (4 supported)

The citation map endpoint accepts multiple input modes for maximum flexibility:

1. **Natural Language Query** (`query_text`): "attention mechanisms in transformers"
   - Searches OpenAlex, Semantic Scholar, ArXiv in parallel
   - LLM validates topical relevance of candidates (score >= 0.75)
   - Picks highest-cited paper among relevant candidates
   - Best for: Exploratory research, finding the "seminal paper" for a topic

2. **DOI** (`seed_doi`): "10.48550/arXiv.1706.03762"
   - Direct lookup in OpenAlex by DOI
   - Fast, deterministic matching
   - Best for: When you have the exact paper identifier

3. **Title** (`seed_title`): "Attention Is All You Need"
   - Searches both OpenAlex AND Semantic Scholar by title
   - Maps S2 results to OpenAlex work_ids via DOI or title matching
   - Uses Jaccard similarity + citation count for best match selection
   - Best for: When you have the paper title but not the DOI

4. **PDF Upload** (`/v1/citation-map/pdf`):
   - Extracts title from PDF metadata or first page text
   - Searches both OpenAlex AND Semantic Scholar by extracted title
   - Same matching logic as Title mode (similarity + citations)
   - Best for: User-facing workflows where papers are uploaded

### PDF Upload Implementation

**Endpoint**: `POST /v1/citation-map/pdf`

**Pipeline**:
1. Extract title from PDF:
   - Try PDF metadata first (`fitz.open().metadata`)
   - Fallback to first page text extraction (skip headers/footers)
   - Heuristics: 15-200 chars, 3+ words, not a URL/email/date
2. Match PDF to work_id:
   - Search OpenAlex by title (top 5)
   - Search Semantic Scholar by title (top 5)
   - Map S2 → OpenAlex via DOI or exact title match
   - Pick best match: highest citation count among high-similarity candidates (Jaccard > 0.5)
3. Build citation map around matched paper (same as other modes)

**Dependencies**:
- `PyMuPDF>=1.23` - PDF parsing
- `python-multipart>=0.0.6` - FastAPI file upload

**Files**:
- `app/feature1/api.py` - PDF upload endpoint
- `app/feature1/pdf_parser.py` - Title extraction from PDF
- `app/feature1/citation_map_service.py` - Dual-source matching (OpenAlex + S2)

### Citation Network Sources

All input modes use **both OpenAlex AND Semantic Scholar** for citation network expansion:

- **OpenAlex**: Primary source for work_ids, metadata, citations
- **Semantic Scholar**: Complementary coverage, better metadata for some papers
- **Bridge**: S2 papers mapped to OpenAlex work_ids via DOI or title matching
- **Why dual-source**: Better coverage, S2 catches papers OpenAlex misses (and vice versa)

### Baseline Establishment & Regression Detection

**First-time setup** (do this once):

```bash
# 1. Run full live test suite
pytest tests/live/test_citation_map_e2e.py -v -s > baseline_2026-02-12.txt

# 2. Perform LLM-judged scoring on all results (65 pts) - review saved JSON outputs
# 3. Calculate averages:
#    - Automated: should be ~30-33/35
#    - LLM-judged: should be ~55-60/65
#    - Total: should be ~85-93/100
# 4. Save these as your baseline reference
```

**Before every commit**:

```bash
# 1. Run live tests
pytest tests/live/test_citation_map_e2e.py -v -s

# 2. Perform LLM-judged scoring on all results (REQUIRED - don't skip!)

# 3. Only commit if:
# - All tests pass
# - Average automated score >= 30/35 (within 5 points of baseline)
# - Average LLM-judged score >= 50/65 (within 10 points of baseline)
# - Average total score >= 80/100
# - No new seed matching failures (DOI/PDF modes)
```

**Why run FULL scoring (100 pts) before every commit:**
- Prevents the "fix → good → wait → broken" regression cycle
- Catches quality regressions immediately (not days later)
- Ensures real API behavior is tested (not just mocked fixtures)
- Validates LLM prompt effectiveness, seed selection, multi-hop quality
- **Automated scoring (35 pts) catches structural issues**
- **LLM-judged scoring (65 pts) catches semantic quality regressions** — wrong seed selection, noisy citation networks, missing foundational papers

**Cost**: ~62 test cases × 2-3 API calls = ~150-200 calls per commit. Time: ~15-20 min (5-10 min for tests + 10 min for manual LLM scoring). Essential for quality.

## Scoring & Iteration

When iterating on a feature, score it against the relevant **grading rubric .md** file if one exists. Use Ralph Wiggum loops with the **test spec .md** for consistent iteration.
