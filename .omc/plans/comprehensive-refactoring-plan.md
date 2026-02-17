# Comprehensive Refactoring, Documentation, and Testing Plan

**Created:** 2026-02-13
**Scope:** Module renaming (feature3/4/5), CLAUDE.md enhancement, test infrastructure
**Estimated complexity:** MEDIUM-HIGH (incremental, each phase is independently shippable)

---

## Context

The Alexandria Backend has five core modules. Two have been properly named (`scholar_search`, `citation_map`), and one support module (`paper_access`) has good test coverage. Three modules still use placeholder names (`feature3`, `feature4`, `feature5`) and have zero test coverage. The CLAUDE.md has detailed documentation for `scholar_search` but only placeholder stubs for the other three.

### Current State Summary

| Module | Current Name | Proper Name | Files | Tests | CLAUDE.md Coverage |
|--------|-------------|-------------|-------|-------|--------------------|
| Scholar Search | `app/scholar_search/` | (done) | ~20 | 0 | Detailed |
| Citation Map | `app/citation_map/` | (done) | ~5 | 0 | 2 lines |
| Node Details | `app/feature3/` | `app/node_details/` | 16 | 0 | 2 lines |
| Paper Compare | `app/feature4/` | `app/paper_compare/` | 6 | 0 | 1 line |
| Research Gaps | `app/feature5/` | `app/research_gaps/` | 8 | 0 | 1 line |
| Paper Access | `app/paper_access/` | (done) | ~10 | 122 | Moderate |

### Cross-Module Dependencies (Critical for Rename Order)

```
feature4/compare_service.py --> feature3/json_utils.py  (extract_json_from_llm_response)
feature5/gap_service.py     --> (no cross-feature imports, only DB tables shared)
main.py                     --> feature3, feature4, feature5 (router imports)
```

The `feature4 -> feature3` dependency means feature3 MUST be renamed first or simultaneously with feature4.

### Database Tables Referenced (Not Renamed -- SQL strings only)

- `node_details_cache` (feature3)
- `methodology_comparison_cache`, `methodology_fingerprint_cache`, `paper_full_text_cache` (feature4)
- `gap_analysis_results`, `gap_feature_usage` (feature5)
- `works`, `map_nodes`, `map_edges`, `maps`, `openalex_topics`, `rank_results` (shared)
- `tenant_settings` (settings)
- `graph_drafts`, `graph_draft_nodes`, `graph_draft_edges` (citation_map)

---

## Phase 1: Module Renaming (feature3 -> node_details, feature4 -> paper_compare, feature5 -> research_gaps)

### Objective
Rename all three feature modules to descriptive names, following the same pattern used in the `feature2 -> scholar_search` refactor.

### Proposed Names

| Current | New Name | Rationale |
|---------|----------|-----------|
| `feature3` | `node_details` | Module provides detailed node pop-up information (summary, novelty, timeline) |
| `feature4` | `paper_compare` | Module compares methodologies across 2-4 papers |
| `feature5` | `research_gaps` | Module detects and validates research gaps |

### Task 1.1: Rename `app/feature3/` to `app/node_details/`

**What to change:**
- Rename directory: `app/feature3/` -> `app/node_details/`
- Update all internal imports within node_details (16 files reference `app.feature3.*`)
- Update `app/main.py` import: `from app.feature3.node_details_api import router` -> `from app.node_details.api import router`
- Update docstrings: replace "Feature 3" with "Node Details" in all files
- Rename `node_details_api.py` -> `api.py` and `node_details_service.py` -> `service.py` for consistency with other modules

**Files to modify (18 total):**
- `app/main.py` (1 import)
- All 16 files in `app/feature3/` (directory move + internal imports)
- `app/feature4/compare_service.py` (1 cross-module import: `from app.feature3.json_utils`)

**Acceptance criteria:**
- `from app.node_details.api import router` works
- `from app.node_details.service import get_node_details` works
- `from app.node_details.json_utils import extract_json_from_llm_response` works (cross-module)
- Server starts without import errors: `python -c "from app.main import app"`
- All existing paper_access tests still pass: `pytest tests/`

### Task 1.2: Rename `app/feature4/` to `app/paper_compare/`

**What to change:**
- Rename directory: `app/feature4/` -> `app/paper_compare/`
- Update all internal imports within paper_compare (6 files)
- Update `app/main.py` import
- Update the cross-module import in `compare_service.py`: `from app.feature3.json_utils` -> `from app.node_details.json_utils`
- Update docstrings: replace "Feature 4" with "Paper Compare"
- Rename `compare_api.py` -> `api.py` and `compare_service.py` -> `service.py`

**Files to modify (7 total):**
- `app/main.py` (1 import)
- All 6 files in `app/feature4/` (directory move + internal imports)

**Acceptance criteria:**
- `from app.paper_compare.api import router` works
- `from app.paper_compare.service import compare_methodologies` works
- Cross-module import to `app.node_details.json_utils` resolves
- Server starts without import errors
- All existing tests still pass

### Task 1.3: Rename `app/feature5/` to `app/research_gaps/`

**What to change:**
- Rename directory: `app/feature5/` -> `app/research_gaps/`
- Update all internal imports within research_gaps (8 files)
- Update `app/main.py` import
- Update docstrings: replace "Feature 5" with "Research Gaps"
- Rename `gap_api.py` -> `api.py` and `gap_service.py` -> `service.py`

**Files to modify (9 total):**
- `app/main.py` (1 import)
- All 8 files in `app/feature5/` (directory move + internal imports)

**Acceptance criteria:**
- `from app.research_gaps.api import router` works
- `from app.research_gaps.service import run_gap_analysis` works
- Server starts without import errors
- All existing tests still pass

### Task 1.4: Update CLAUDE.md Module References

**What to change:**
- Replace all `feature3` references with `node_details`
- Replace all `feature4` references with `paper_compare`
- Replace all `feature5` references with `research_gaps`
- Update the "Recent Changes" section to document this refactor

**Acceptance criteria:**
- No remaining references to `feature3`, `feature4`, or `feature5` in CLAUDE.md
- New module names appear in architecture overview

---

## Phase 2: CLAUDE.md Enhancement

### Objective
Expand CLAUDE.md from 402 lines to ~700-800 lines with content that enables autonomous agents to work effectively for hours without asking questions.

### Task 2.1: Add Node Details Module Documentation

**Content to add (following the scholar_search documentation pattern):**

```markdown
### `app/node_details/` -- Node Details Pop-up

**Purpose**: Generate detailed pop-up information for citation map nodes, including
LLM-generated summaries, keyword extraction, and grounded novelty assessments.

**Key Features**:
- LLM-generated paper summary (Groq Llama-4 Maverick)
- Keyword extraction (LLM + heuristic fallback)
- Grounded novelty assessment with 4 tiers: low/medium/high/pioneering
- Cross-domain filtering (methodology-based + subfield-based)
- Per-node timeline with impact analysis
- Abstract enrichment from ArXiv/Semantic Scholar
- Topic inference for papers missing OpenAlex topic IDs

**Architecture** (ranked by importance):
1. `service.py` (1565 lines) -- Main orchestration, entry point: `get_node_details()`
2. `api.py` -- HTTP endpoint: GET /v1/maps/{map_id}/nodes/{work_id}/details
3. `schemas.py` -- Pydantic models: NodeDetailsResponse, NoveltyAssessment, GroundingPaper
4. `grounding_supplement.py` -- Supplements grounding papers when refs/landmarks sparse
5. `node_timeline.py` -- Builds backward/forward timeline with impact analysis
6. `abstract_enrichment.py` -- Fetches abstracts from ArXiv/S2 when missing/invalid
7. `topic_inference.py` -- Infers OpenAlex topic ID via API or LLM fallback
8. `landmark_retrieval.py` -- Retrieves field landmark papers by topic ID
9. `reference_store.py` -- Loads referenced works from DB
10. `methodology.py` -- Methodology detection for cross-domain filtering
11. `json_utils.py` -- Robust JSON extraction from LLM responses (SHARED with paper_compare)
12. `paper_cache.py` -- Citing paper cache
13. `paper_identity.py` -- DOI/ArXiv ID normalization
14. `paper_impact_analytics.py` -- Paradigm shift detection
15. `topic_lookup.py` -- Topic display name resolution

**Data Flow**:
Request -> Verify map/node ownership -> Load work metadata -> Enrich abstract ->
Infer topic -> Check cache -> Load refs + landmarks -> Cross-domain filter ->
Supplement grounding -> LLM assessment -> Validate + cache -> Response

**Key Tables**: node_details_cache, works, map_nodes, map_edges, maps, openalex_topics

**Entry Point**:
from app.node_details.service import get_node_details
```

**Acceptance criteria:**
- Architecture section lists all 15 files with line counts and purpose
- Data flow diagram matches actual code in `service.py`
- Entry points and key tables are documented

### Task 2.2: Add Paper Compare Module Documentation

**Content to add:**

```markdown
### `app/paper_compare/` -- Methodology Comparison

**Purpose**: Compare methodologies of 2-4 papers with structured LLM analysis.

**Key Features**:
- 3-stage LLM pipeline: content fetch -> fingerprint extraction -> comparative synthesis
- Full-text methodology extraction via Semantic Scholar + PDF download
- Citation lineage analysis (direct citations, shared references, evolution chains)
- Convergence/divergence paradigm analysis
- Strengths/weaknesses matrix with cross-references
- Decision matrix recommendations

**Architecture**:
1. `service.py` (~1150 lines) -- Main pipeline: `compare_methodologies()`
2. `api.py` -- HTTP endpoint: POST /v1/maps/{map_id}/compare-methodologies
3. `schemas.py` (~199 lines) -- Rich schema: MethodologyComparisonResponse, paradigms, etc.
4. `paper_content.py` -- S2 full-text fetch, PDF extraction, methods section parsing
5. `citation_lineage.py` -- Direct citations, shared refs, evolution chain builder

**Pipeline Stages**:
1. Validate work_ids belong to map
2. Check comparison cache (hash of sorted work_ids)
3. Stage 1A+1B (parallel): Fetch paper content + Build citation lineage
4. Stage 2: Extract methodology fingerprints (LLM call 1, with validation loop)
5. Stage 3: Synthesize comparison (LLM call 2, with validation loop)
6. Cache and return

**Key Tables**: methodology_comparison_cache, methodology_fingerprint_cache,
               paper_full_text_cache, works, map_nodes, map_edges

**Cross-Module Dependency**: Imports json_utils from node_details module.

**Entry Point**:
from app.paper_compare.service import compare_methodologies
```

**Acceptance criteria:**
- Pipeline stages documented with stage numbering matching code comments
- Cross-module dependency explicitly noted
- All 5 files described with purpose

### Task 2.3: Add Research Gaps Module Documentation

**Content to add:**

```markdown
### `app/research_gaps/` -- Research Gap Analysis

**Purpose**: Detect, synthesize, and externally validate research gaps in a map.

**Key Features**:
- 5 heuristic gap detectors: structural, coverage, temporal, methodological, novelty
- LLM-direct fallback when heuristics produce sparse results
- LLM synthesis with accept/reject filtering
- External validation via GPT-5.2 + web_search
- Coverage-gated unlock (50% exploration required)
- Feature usage tracking for coverage calculation

**Architecture**:
1. `service.py` (~1190 lines) -- Main pipeline: `run_gap_analysis()`
2. `api.py` -- HTTP endpoints:
   - GET /v1/maps/{map_id}/gap-analysis/status
   - POST /v1/maps/{map_id}/gap-analysis
   - GET /v1/maps/{map_id}/gap-analysis/results
   - POST /v1/maps/{map_id}/track-feature
3. `schemas.py` -- GapCard, Evidence, ExternalValidationResult, gap candidate types
4. `gap_detection.py` (~870 lines) -- 5 heuristic detectors + orchestrator
5. `coverage_tracker.py` -- Coverage calculation and feature usage tracking
6. `external_validation.py` -- GPT-5.2 + web_search validation
7. `prompts.py` -- LLM prompt templates

**Pipeline**:
1. Check unlock status (>= 50% coverage required)
2. Get available data sources for this map
3. Run heuristic gap detectors (structural, coverage, temporal, methodological, novelty)
4. If < 3 candidates: supplement with LLM-direct detection
5. LLM synthesis: evaluate, accept/reject, describe gaps
6. Create gap cards with evidence and roles
7. Filter weak candidates (detection_score < 0.40)
8. Deduplicate by evidence overlap + title similarity
9. External validation via GPT-5.2 + web_search
10. Filter high-coverage gaps (> 60% already addressed)
11. Cache and return

**Key Tables**: gap_analysis_results, gap_feature_usage, node_details_cache,
               methodology_fingerprint_cache, methodology_comparison_cache,
               rank_results, works, map_nodes, map_edges, openalex_topics

**Entry Points**:
from app.research_gaps.service import run_gap_analysis, get_cached_gap_analysis
from app.research_gaps.coverage_tracker import get_gap_analysis_status
```

**Acceptance criteria:**
- All 7 files described with line counts
- All 4 API endpoints listed
- 11-step pipeline matches code in `service.py`
- Coverage unlock mechanism explained

### Task 2.4: Add Testing Guidelines Section

**Content to add:**

```markdown
## Testing Guidelines

### Running Tests
pytest tests/                           # All tests
pytest tests/paper_access/              # Single module
pytest tests/ -k "test_openalex"        # Pattern match
pytest tests/ --cov=app --cov-report=term-missing  # With coverage

### Test Patterns

**Unit test structure** (all modules follow this pattern):
tests/
  {module_name}/
    __init__.py
    conftest.py          # Module-specific fixtures
    test_{feature}.py    # One file per logical feature

**Mocking external APIs** (use `responses` library):
@responses.activate
def test_openalex_search():
    responses.add(responses.GET, "https://api.openalex.org/works", json={...})
    result = search_openalex("query")
    assert result[0].title == "Expected"

**Mocking database** (use in-memory SQLite or mock Connection):
def mock_connection(query_results):
    conn = MagicMock(spec=Connection)
    conn.execute.return_value.mappings.return_value.all.return_value = query_results
    return conn

**Mocking LLM calls** (patch OpenAI client):
@patch("app.node_details.service.OpenAI")
def test_llm_generation(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = ...

### Test Priorities (by risk)
1. scholar_search/ranking -- Core ranking logic, high user impact
2. node_details/service -- LLM grounding, novelty assessment accuracy
3. paper_compare/service -- Multi-stage LLM pipeline
4. research_gaps/gap_detection -- Heuristic algorithms
5. citation_map -- Graph construction

### What NOT to Test
- Direct LLM output quality (non-deterministic)
- External API availability (use mocks)
- Database schema (tested implicitly through integration)
```

**Acceptance criteria:**
- Test commands documented
- Mocking patterns for APIs, DB, and LLM calls shown with code examples
- Test priorities listed with rationale

### Task 2.5: Add Database Schema Reference

**Content to add:**

```markdown
## Database Schema Reference

### Core Tables
- `works` -- Normalized paper metadata (work_id PK, title, year, cited_by_count,
  authors_json, venue, abstract, primary_topic_id, doi, arxiv_id, is_open_access,
  oa_status, oa_pdf_url, referenced_works_json, topics_json, category, category_confidence)
- `maps` -- Citation maps (map_id PK, tenant_id FK, query_text, created_at)
- `map_nodes` -- Papers in a map (map_id + work_id composite PK, best_topic_id, best_subfield_id)
- `map_edges` -- Citation links (map_id + from_work_id + to_work_id)
- `openalex_topics` -- Topic metadata (topic_id PK, display_name)
- `tenant_settings` -- Per-tenant config (tenant_id PK, institutional_proxy_prefix,
  libkey_api_key, libkey_library_id)

### Module-Specific Cache Tables
- `node_details_cache` -- LLM assessment cache (work_id PK, summary, keywords,
  novelty_assessment JSONB, model_version, assessment_unavailable_reason)
- `methodology_comparison_cache` -- Comparison results (comparison_hash PK, tenant_id,
  work_ids[], result_json JSONB, model_version)
- `methodology_fingerprint_cache` -- Per-paper fingerprints (work_id PK,
  fingerprint_json JSONB, source_quality, model_version)
- `paper_full_text_cache` -- S2 full text (work_id PK, s2_paper_id,
  methods_text, full_text_available)
- `gap_analysis_results` -- Gap analysis output (id PK, map_id, tenant_id,
  gaps JSONB, data_sources_used, coverage_pct)
- `gap_feature_usage` -- Feature usage tracking (map_id + feature_type + work_id,
  metadata, updated_at)
- `rank_jobs`, `rank_results` -- Scholar search ranking output

### Graph Tables
- `graph_drafts` -- Draft citation graphs (graph_draft_id PK, tenant_id, candidate_set_id)
- `graph_draft_nodes` -- Nodes in draft graphs
- `graph_draft_edges` -- Edges in draft graphs
```

**Acceptance criteria:**
- All tables referenced in code are listed
- Column names match actual SQL queries in the codebase
- Tables grouped by purpose (core, cache, graph)

### Task 2.6: Add Error Handling Patterns and Agent Navigation Aids

**Content to add:**

```markdown
## Error Handling Patterns

All API endpoints follow this structure:
try:
    result = service_function(...)
    return result
except PermissionError:       # 403 - tenant mismatch
    raise HTTPException(403)
except ValueError as e:       # 400/404 depending on message
    if "not_found" in str(e):
        raise HTTPException(404)
    raise HTTPException(400)
except Exception:             # 500 - unexpected
    logger.exception(...)
    raise HTTPException(500)

### LLM Call Patterns
All LLM calls use:
- Groq Llama-4 Maverick (node_details, paper_compare) or Qwen3-32B (research_gaps)
- Retry with exponential backoff (3-4 attempts)
- JSON extraction with robust parsing (handles code fences, trailing text)
- Validation loops (compare_service sends violations back for correction)
- Graceful fallback on failure (default responses, not errors)

### Common Pitfalls
- json_utils.py is in node_details but shared with paper_compare -- consider extracting to app/shared/ if more modules need it
- LLM model constants are defined per-module (MODEL_VERSION) -- update all when changing models
- Cache tables use ON CONFLICT DO UPDATE -- idempotent by design
- Advisory locks use PostgreSQL pg_advisory_lock -- not available in SQLite test DBs
- dotenv is loaded in multiple service files -- the main.py load_dotenv should suffice

## Quick Navigation for Agents

**I want to...**
- **...add a new API endpoint** -> Copy pattern from any `api.py`, add router to `main.py`
- **...modify LLM prompts** -> node_details: `service.py:_build_grounded_prompt`, paper_compare: `service.py:_build_extraction_prompt`/`_build_synthesis_prompt`, research_gaps: `prompts.py`
- **...change scoring/ranking** -> `scholar_search/ranking/scoring.py` + `ranking/pipeline.py`
- **...add a new gap detector** -> `research_gaps/gap_detection.py`, add to `detect_all_gaps()`
- **...modify novelty assessment** -> `node_details/service.py:_build_grounded_prompt` (the prompt IS the assessment logic)
- **...update external API clients** -> `paper_access/` (OpenAlex, S2, ArXiv, etc.)
- **...change database queries** -> Look at `repos.py` files or inline SQL in service files
- **...understand cross-module data flow** -> scholar_search populates works/rank_results, citation_map populates maps/map_nodes/map_edges, node_details/paper_compare/research_gaps consume this data
```

**Acceptance criteria:**
- Error handling pattern matches all 3 API modules
- LLM patterns cover retry, parsing, validation, fallback
- Common pitfalls list includes the json_utils sharing issue
- Navigation aids cover the 8 most common agent tasks

---

## Phase 3: Test Infrastructure

### Objective
Establish test infrastructure and write priority unit tests that enable confident autonomous development.

### Task 3.1: Create Test Fixtures and Conftest

**What to create:**
- `tests/conftest.py` -- Enhance with shared fixtures:
  - `mock_engine` / `mock_connection` fixtures (SQLAlchemy mocks)
  - `sample_work_data` fixture (realistic paper metadata dict)
  - `sample_map_data` fixture (map_id, nodes, edges)
  - `mock_groq_client` fixture (patched OpenAI client returning valid JSON)
  - Environment variable safety (already exists, expand)

**Acceptance criteria:**
- `mock_connection` fixture returns configurable query results
- `mock_groq_client` fixture returns valid LLM JSON for each module's expected format
- All fixtures are documented with docstrings
- Existing 122 paper_access tests still pass

### Task 3.2: Unit Tests for Node Details Module

**Priority tests (target: 25-35 tests):**

1. **service.py -- Core logic (15-20 tests)**
   - `test_get_cached_details_hit` -- Returns cached data without LLM call
   - `test_get_cached_details_miss` -- Falls through to LLM generation
   - `test_cache_invalidation_on_enrichment` -- Skips cache when abstract/topic changed
   - `test_load_work_data_returns_all_fields` -- All expected keys present
   - `test_load_connected_works_both_directions` -- Cites and cited_by relationships
   - `test_truncate_text_preserves_word_boundaries` -- Edge cases: empty, exact length, mid-word
   - `test_generate_summary_from_abstract_fallback` -- No abstract, short abstract, normal
   - `test_extract_keywords_from_abstract` -- Quoted terms, capitalized phrases, patterns
   - `test_build_grounded_prompt_with_pioneering_context` -- High-citation papers get context
   - `test_build_grounded_prompt_without_landmarks` -- Handles empty landmarks
   - `test_validate_llm_response_filters_invalid_grounding` -- Invalid work_ids removed
   - `test_validate_llm_response_removes_self_citation` -- Target work_id excluded
   - `test_validate_llm_response_supplements_landmarks` -- Minimum 3 landmarks enforced
   - `test_validate_llm_response_enum_normalization` -- Invalid novelty_level defaults to medium
   - `test_filter_cross_domain_papers_methodology` -- Methodology mismatch filtered
   - `test_filter_cross_domain_papers_fallback` -- Keeps top refs when strict filter too aggressive

2. **json_utils.py (5-8 tests)**
   - `test_extract_json_from_code_block` -- Handles ```json ... ``` wrapping
   - `test_extract_json_from_trailing_text` -- JSON followed by explanation text
   - `test_extract_json_nested_braces` -- Deeply nested objects
   - `test_extract_json_invalid_returns_none` -- Garbage input returns None
   - `test_extract_json_expected_type_mismatch` -- Object when expecting array

3. **schemas.py (3-5 tests)**
   - `test_node_details_response_serialization` -- Full round-trip
   - `test_novelty_assessment_optional_fields` -- Pioneering works with null whats_new
   - `test_grounding_paper_relationship_literal` -- Only cited_reference/field_landmark

**Acceptance criteria:**
- All tests pass with `pytest tests/node_details/`
- No real API calls or database connections
- Tests cover the main service.py branching paths (cache hit, cache miss, enrichment, no grounding data)

### Task 3.3: Unit Tests for Paper Compare Module

**Priority tests (target: 15-25 tests):**

1. **service.py (10-15 tests)**
   - `test_comparison_hash_deterministic` -- Same work_ids (any order) produce same hash
   - `test_comparison_hash_different_for_different_sets` -- Different work_ids produce different hash
   - `test_validate_extraction_short_approach` -- Violation for < 200 char approach
   - `test_validate_extraction_generic_components` -- Violation for "deep neural networks"
   - `test_validate_extraction_missing_assumptions` -- Violation for < 2 assumptions
   - `test_validate_extraction_hedging_language` -- Detects "may", "might", "could"
   - `test_validate_extraction_limitation_format` -- Requires arrow structure
   - `test_validate_synthesis_missing_sections` -- Missing convergence_divergence
   - `test_validate_synthesis_insufficient_scenarios` -- < 3 decision scenarios
   - `test_validate_synthesis_invalid_work_id_in_use` -- work_id not in paper set
   - `test_dimensions_overlap_jaccard` -- Name similarity detection
   - `test_dimensions_overlap_semantic_cluster` -- Equivalent terms detection
   - `test_is_shallow_cross_task` -- Negation pattern detection

2. **paper_content.py (5-8 tests)**
   - `test_extract_methods_section_standard` -- "## Methods" header
   - `test_extract_methods_section_variant_headers` -- "Materials and Methods", "Approach"
   - `test_extract_methods_section_too_short` -- < 100 chars returns None
   - `test_titles_match_ignoring_punctuation` -- Normalized comparison
   - `test_word_overlap_calculation` -- Jaccard score correctness
   - `test_pdf_matches_paper_title_words` -- Significant word matching

3. **schemas.py (2-3 tests)**
   - `test_methodology_compare_request_min_ids` -- Rejects < 2 work_ids
   - `test_methodology_compare_request_max_ids` -- Rejects > 4 work_ids
   - `test_methodology_compare_request_unique_ids` -- Rejects duplicates

**Acceptance criteria:**
- All tests pass with `pytest tests/paper_compare/`
- Validation logic thoroughly tested (this is the quality gate for LLM output)
- Schema validators tested for all edge cases

### Task 3.4: Unit Tests for Research Gaps Module

**Priority tests (target: 15-25 tests):**

1. **gap_detection.py (8-12 tests)**
   - `test_detect_structural_gaps_basic` -- Two clusters with high similarity, low density
   - `test_detect_structural_gaps_small_clusters_skipped` -- Clusters < 2 papers ignored
   - `test_compute_topic_similarity_jaccard` -- Correct Jaccard calculation
   - `test_detect_coverage_gaps_z_score` -- Underrepresented topics detected
   - `test_detect_coverage_gaps_singleton_low_impact_skipped` -- Single low-cite paper ignored
   - `test_detect_temporal_gaps_rolling_window` -- Uses local context not flat average
   - `test_detect_temporal_gaps_sparse_data_skipped` -- < 15 papers skips detection
   - `test_detect_temporal_gaps_comeback_pattern` -- active-stagnant-active detected
   - `test_categorize_method_keywords` -- Maps free text to categories
   - `test_detect_methodological_gaps_both_axes_required` -- Single-axis gaps rejected
   - `test_detect_novelty_gaps_active_clusters_only` -- Old clusters with low novelty skipped

2. **service.py (5-8 tests)**
   - `test_extract_json_from_llm_code_fences` -- Strips markdown code blocks
   - `test_extract_json_from_llm_think_tags` -- Strips Qwen3 think tags
   - `test_create_gap_cards_filters_integration_titles` -- "Integration of X and Y" rejected
   - `test_clean_gap_title_strips_citations` -- "(Author, Year)" removed
   - `test_deduplicate_gap_cards_evidence_overlap` -- > 50% shared evidence deduped
   - `test_deduplicate_gap_cards_title_similarity` -- > 50% Jaccard deduped
   - `test_title_word_similarity_stop_words` -- Stop words excluded from comparison

3. **coverage_tracker.py (3-5 tests)**
   - `test_coverage_calculation_all_sources` -- Timeline + methodology + per-node
   - `test_unlock_threshold_50_percent` -- Unlocked when >= 50%
   - `test_coverage_empty_map` -- Returns 0% for map with no nodes

**Acceptance criteria:**
- All tests pass with `pytest tests/research_gaps/`
- Gap detection heuristics tested with constructed datasets
- Deduplication and filtering logic covered

### Task 3.5: Integration Test Fixtures and Smoke Tests

**What to create:**
- `tests/integration/conftest.py` -- Shared integration fixtures:
  - Mock FastAPI TestClient (using `from fastapi.testclient import TestClient`)
  - Dependency overrides for engine and tenant_id
  - Pre-populated mock data for a complete map (nodes, edges, works)

- `tests/integration/test_api_smoke.py` -- Basic endpoint reachability:
  - Each endpoint returns expected status codes (200, 400, 404)
  - Request/response schemas validate correctly
  - Error handling returns proper HTTP codes

**Priority integration tests (target: 10-15 tests):**
- `test_rank_endpoint_returns_200` -- POST /v1/rank with valid body
- `test_node_details_endpoint_returns_200` -- GET /v1/maps/{id}/nodes/{id}/details
- `test_node_details_endpoint_404_missing_map` -- Invalid map_id
- `test_compare_endpoint_returns_200` -- POST /v1/maps/{id}/compare-methodologies
- `test_compare_endpoint_400_single_id` -- Only 1 work_id
- `test_gap_analysis_status_endpoint` -- GET /v1/maps/{id}/gap-analysis/status
- `test_gap_analysis_403_not_unlocked` -- POST gap-analysis before 50% coverage
- `test_track_feature_400_invalid_type` -- Invalid feature_type

**Acceptance criteria:**
- Integration tests use FastAPI TestClient (no real HTTP server)
- Database mocked at the engine level
- Tests verify HTTP status codes and response schema shapes
- All tests pass with `pytest tests/integration/`

---

## Phase 4: CLAUDE.md Final Assembly and Verification

### Task 4.1: Assemble Complete CLAUDE.md

**What to do:**
- Integrate all content from Phase 2 tasks into CLAUDE.md
- Ensure consistent formatting and section ordering
- Update the "Recent Changes" section with Phase 1 rename details
- Verify all import paths reference new module names
- Add a "Testing" section with test commands and coverage targets

**Acceptance criteria:**
- CLAUDE.md is 700-800 lines
- All 6 modules documented at similar detail level
- No stale references to feature3/4/5
- Testing section with commands, patterns, and priorities
- Database schema reference complete

### Task 4.2: Verification Pass

**What to verify:**
- `python -c "from app.main import app; print('OK')"` -- Server starts
- `pytest tests/` -- All tests pass (existing + new)
- `pytest tests/ --cov=app --cov-report=term-missing` -- Coverage report
- No remaining references to `feature3`, `feature4`, `feature5` in any `.py` file
- CLAUDE.md contains no broken references

**Acceptance criteria:**
- Zero import errors
- All tests pass
- No references to old module names in Python files
- Coverage report shows new test files covering node_details, paper_compare, research_gaps

---

## Implementation Order and Dependencies

```
Phase 1 (Rename) -----> Phase 2 (Docs) -----> Phase 4 (Assembly)
    |                       |
    |                       v
    +-----------------> Phase 3 (Tests) ---> Phase 4 (Verification)
```

**Phase 1** must complete first (imports change).
**Phase 2** and **Phase 3** can run in parallel after Phase 1.
**Phase 4** depends on both Phase 2 and Phase 3.

### Estimated Effort by Phase

| Phase | Tasks | Estimated Complexity | Files Modified | Files Created |
|-------|-------|---------------------|----------------|---------------|
| 1 | 4 | MEDIUM | ~35 | 0 |
| 2 | 6 | LOW-MEDIUM | 1 (CLAUDE.md) | 0 |
| 3 | 5 | MEDIUM-HIGH | 2 (conftest) | ~8 test files |
| 4 | 2 | LOW | 1 (CLAUDE.md) | 0 |

---

## Success Criteria

### Quantitative
- [ ] 0 references to feature3/4/5 in Python files
- [ ] CLAUDE.md >= 700 lines with all modules documented
- [ ] >= 60 new unit tests across 3 modules
- [ ] >= 10 integration smoke tests
- [ ] All tests pass (existing 122 + new ~70-80)
- [ ] Server starts without errors after rename

### Qualitative
- [ ] An autonomous agent can read CLAUDE.md and understand how to modify any module
- [ ] Test fixtures enable writing new tests without understanding DB schema
- [ ] Module names communicate purpose (no "featureN" ambiguity)
- [ ] Cross-module dependencies are documented and explicit

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed import during rename | MEDIUM | HIGH (server won't start) | Grep for all `feature[345]` references; verify with import check |
| Cross-module dependency breaks | LOW | HIGH | feature4->feature3 import identified; rename together |
| Test fixtures too coupled to implementation | MEDIUM | MEDIUM | Use interface-level mocking, not internal details |
| CLAUDE.md becomes stale | LOW | MEDIUM | Add "last updated" dates per section |
| Database table names confused with module names | LOW | LOW | Tables are NOT renamed (SQL strings only) |

---

## Guardrails

### Must Have
- Backward-compatible API contracts (URL paths unchanged)
- All existing 122 tests continue to pass at every phase
- Server starts after each phase completes
- No secrets or credentials in test fixtures

### Must NOT Have
- Database migrations (table names stay the same)
- API URL changes (only Python module paths change)
- Architecture redesign (only renaming and documenting)
- New runtime dependencies
