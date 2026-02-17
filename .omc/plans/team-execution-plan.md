# Team Execution Plan: Test-First Refactoring

**Strategy**: Write tests BEFORE renaming modules to ensure refactoring safety

---

## Phase A: Test Infrastructure (Foundation)

### Task A1: Create Shared Test Fixtures
**Owner**: TBD
**Priority**: CRITICAL (blocks all other test tasks)
**Estimated Effort**: 2-3 hours

**Objectives**:
- Enhance `tests/conftest.py` with reusable fixtures
- Add mock_engine, mock_connection, sample_work_data, sample_map_data
- Add mock_groq_client for LLM mocking
- Verify existing 122 paper_access tests still pass

**Acceptance Criteria**:
- `pytest tests/paper_access/` passes (all 122 tests)
- All fixtures have docstrings
- mock_groq_client returns valid JSON for each module

**Files to Create/Modify**:
- `tests/conftest.py` (enhance)

---

## Phase B: Unit Tests (Write BEFORE Renaming)

These tasks write tests for the CURRENT module names (feature3/4/5).

### Task B1: Unit Tests for Feature3 (node_details)
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task A1 (shared fixtures)
**Estimated Effort**: 4-6 hours

**Objectives**:
- Write 25-35 unit tests for `app/feature3/` module
- Test service.py core logic (15-20 tests)
- Test json_utils.py (5-8 tests)
- Test schemas.py (3-5 tests)
- All tests use current import paths: `from app.feature3 import ...`

**Acceptance Criteria**:
- `pytest tests/test_feature3.py` passes all tests
- No real API calls or database connections
- Tests cover main service.py branching paths

**Files to Create**:
- `tests/test_feature3_service.py`
- `tests/test_feature3_json_utils.py`
- `tests/test_feature3_schemas.py`

### Task B2: Unit Tests for Feature4 (paper_compare)
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task A1 (shared fixtures)
**Estimated Effort**: 3-5 hours

**Objectives**:
- Write 15-25 unit tests for `app/feature4/` module
- Test service.py validation logic (10-15 tests)
- Test paper_content.py (5-8 tests)
- Test schemas.py (2-3 tests)
- All tests use current import paths: `from app.feature4 import ...`

**Acceptance Criteria**:
- `pytest tests/test_feature4.py` passes all tests
- Validation logic thoroughly tested
- Schema validators tested for edge cases

**Files to Create**:
- `tests/test_feature4_service.py`
- `tests/test_feature4_paper_content.py`
- `tests/test_feature4_schemas.py`

### Task B3: Unit Tests for Feature5 (research_gaps)
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task A1 (shared fixtures)
**Estimated Effort**: 3-5 hours

**Objectives**:
- Write 15-25 unit tests for `app/feature5/` module
- Test gap_detection.py heuristics (8-12 tests)
- Test service.py (5-8 tests)
- Test coverage_tracker.py (3-5 tests)
- All tests use current import paths: `from app.feature5 import ...`

**Acceptance Criteria**:
- `pytest tests/test_feature5.py` passes all tests
- Gap detection heuristics tested with constructed datasets
- Deduplication logic covered

**Files to Create**:
- `tests/test_feature5_gap_detection.py`
- `tests/test_feature5_service.py`
- `tests/test_feature5_coverage.py`

**Phase B Verification**: Run `pytest tests/` - all 122 + ~60 new tests should pass

---

## Phase C: Module Renaming (With Test Safety Net)

These tasks rename modules while tests ensure nothing breaks.

### Task C1: Rename feature3 → node_details
**Owner**: TBD
**Priority**: CRITICAL
**Dependencies**: Tasks B1, B2, B3 (all unit tests passing)
**Estimated Effort**: 2-3 hours

**Objectives**:
- Rename directory: `app/feature3/` → `app/node_details/`
- Update all 16 internal imports within node_details
- Update `app/main.py` import and router
- Update cross-module import in `app/feature4/compare_service.py`
- Update docstrings: "Feature 3" → "Node Details"
- Rename `node_details_api.py` → `api.py`, `node_details_service.py` → `service.py`

**Acceptance Criteria**:
- `python -c "from app.main import app"` succeeds
- `grep -rn "from app\.feature3" app/` returns 0 matches
- All existing tests still pass (tests will need import updates in Task C4)

**Files to Modify**: 18 files (main.py + 16 in feature3 + 1 in feature4)

### Task C2: Rename feature4 → paper_compare
**Owner**: TBD
**Priority**: CRITICAL
**Dependencies**: Task C1 (feature3 renamed first due to dependency)
**Estimated Effort**: 1-2 hours

**Objectives**:
- Rename directory: `app/feature4/` → `app/paper_compare/`
- Update all 6 internal imports
- Update `app/main.py` import
- Update cross-module import to use `app.node_details.json_utils`
- Update docstrings: "Feature 4" → "Paper Compare"
- Rename `compare_api.py` → `api.py`, `compare_service.py` → `service.py`

**Acceptance Criteria**:
- `python -c "from app.main import app"` succeeds
- `grep -rn "from app\.feature4" app/` returns 0 matches
- Cross-module import resolves correctly

**Files to Modify**: 7 files (main.py + 6 in feature4)

### Task C3: Rename feature5 → research_gaps
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task C1 (can run in parallel with C2)
**Estimated Effort**: 1-2 hours

**Objectives**:
- Rename directory: `app/feature5/` → `app/research_gaps/`
- Update all 8 internal imports
- Update `app/main.py` import
- Update docstrings: "Feature 5" → "Research Gaps"
- Rename `gap_api.py` → `api.py`, `gap_service.py` → `service.py`

**Acceptance Criteria**:
- `python -c "from app.main import app"` succeeds
- `grep -rn "from app\.feature5" app/` returns 0 matches

**Files to Modify**: 9 files (main.py + 8 in feature5)

### Task C4: Update Test Imports
**Owner**: TBD
**Priority**: CRITICAL
**Dependencies**: Tasks C1, C2, C3 (all renames complete)
**Estimated Effort**: 30 minutes

**Objectives**:
- Update all test files to use new module names
- `from app.feature3` → `from app.node_details`
- `from app.feature4` → `from app.paper_compare`
- `from app.feature5` → `from app.research_gaps`
- Clean up `__pycache__` directories

**Acceptance Criteria**:
- `pytest tests/` passes all tests (~182 tests)
- `grep -rn "from app\.feature[345]" tests/` returns 0 matches
- Server starts cleanly: `uvicorn app.main:app --reload` (30 second test)

**Files to Modify**: ~9 test files

**Phase C Verification**: `pytest tests/ && python -c "from app.main import app"` succeeds

---

## Phase D: CLAUDE.md Enhancement

These tasks can run in parallel after renaming is complete.

### Task D1: Add Module Documentation (node_details)
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task C1 (node_details renamed)
**Estimated Effort**: 1-2 hours

**Objectives**:
- Add detailed node_details module documentation to CLAUDE.md
- Include purpose, key features, architecture (15 files ranked), data flow
- Document entry points and key database tables

**Acceptance Criteria**:
- Architecture section lists all 15 files with line counts
- Data flow diagram matches code in service.py
- Entry points and tables documented

**Files to Modify**: `CLAUDE.md` (section addition)

### Task D2: Add Module Documentation (paper_compare + research_gaps)
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Tasks C2, C3 (both modules renamed)
**Estimated Effort**: 1-2 hours

**Objectives**:
- Add paper_compare module documentation
- Add research_gaps module documentation
- Include architecture, pipeline stages, cross-module dependencies

**Acceptance Criteria**:
- Both modules have complete documentation
- Pipeline stages match code comments
- Cross-module dependencies explicitly noted

**Files to Modify**: `CLAUDE.md` (section additions)

### Task D3: Add Testing Guidelines
**Owner**: TBD
**Priority**: MEDIUM
**Dependencies**: Tasks B1, B2, B3 (test patterns established)
**Estimated Effort**: 1 hour

**Objectives**:
- Document test commands (pytest usage)
- Provide mocking patterns (API, DB, LLM)
- List test priorities by risk

**Acceptance Criteria**:
- Test commands documented
- Mocking patterns with code examples
- Priorities listed with rationale

**Files to Modify**: `CLAUDE.md` (section addition)

### Task D4: Add Database Schema + Error Handling + Navigation
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: None (can start anytime)
**Estimated Effort**: 1-2 hours

**Objectives**:
- Add database schema reference (all tables with columns)
- Document error handling patterns (try/except structure)
- Add agent navigation aids ("I want to do X" guide)
- Document common pitfalls

**Acceptance Criteria**:
- All tables documented with column names matching SQL
- Error handling pattern matches all API modules
- Navigation aids cover 8 common agent tasks

**Files to Modify**: `CLAUDE.md` (section additions)

**Phase D Verification**: CLAUDE.md expanded from 402 to 700-800 lines

---

## Phase E: Integration & E2E Testing

### Task E1: Integration Test Infrastructure
**Owner**: TBD
**Priority**: MEDIUM
**Dependencies**: Task C4 (all renames + test updates complete)
**Estimated Effort**: 2-3 hours

**Objectives**:
- Create `tests/integration/conftest.py` with FastAPI TestClient fixtures
- Create dependency overrides for engine and tenant_id
- Add pre-populated mock data for complete map
- Write 10-15 API smoke tests

**Acceptance Criteria**:
- TestClient fixture works with all endpoints
- All API endpoints return expected status codes
- Request/response schemas validate correctly

**Files to Create**:
- `tests/integration/conftest.py`
- `tests/integration/test_api_smoke.py`

### Task E2: End-to-End Test Suite
**Owner**: TBD
**Priority**: HIGH
**Dependencies**: Task E1 (integration infrastructure)
**Estimated Effort**: 2-4 hours

**Objectives**:
- Write E2E test for core user journey:
  1. Search for papers (scholar_search ranking)
  2. Get node details (node_details module)
  3. Compare papers (paper_compare module)
  4. Analyze research gaps (research_gaps module)
- Use realistic data and mock external APIs
- Test multi-module data flow

**Acceptance Criteria**:
- E2E test passes with full user journey
- All module interactions tested
- External APIs properly mocked

**Files to Create**:
- `tests/e2e/test_user_journey.py`

**Phase E Verification**: `pytest tests/` passes all ~200+ tests

---

## Final Verification

After all phases complete, verify:

1. **All tests pass**: `pytest tests/ -v`
2. **Server starts cleanly**: `uvicorn app.main:app --reload`
3. **No stale references**: `grep -rn "feature[345]" app/ tests/`
4. **CLAUDE.md expanded**: Check line count (~700-800 lines)
5. **Test coverage**: `pytest tests/ --cov=app --cov-report=term-missing`

**Success Criteria**:
- [ ] ~200+ tests passing (122 original + ~80 new)
- [ ] Zero import errors
- [ ] CLAUDE.md comprehensive for autonomous work
- [ ] Ready for autonomous multi-hour development sessions

---

## Task Dependencies Graph

```
A1 (fixtures)
├─ B1 (feature3 tests)
├─ B2 (feature4 tests)
└─ B3 (feature5 tests)
    ├─ C1 (rename feature3)
    │   ├─ C2 (rename feature4)
    │   └─ C3 (rename feature5)
    │       └─ C4 (update test imports)
    │           ├─ D1-D4 (CLAUDE.md enhancement)
    │           └─ E1 (integration tests)
    │               └─ E2 (E2E tests)
```

---

## Team Coordination Notes

**Parallelization Opportunities**:
- Phase B: Tasks B1, B2, B3 can run in parallel (3 agents)
- Phase C: Task C3 can run in parallel with C2 (after C1)
- Phase D: Tasks D1-D4 can run in parallel (3-4 agents)

**Critical Path**:
A1 → B1 → C1 → C2 → C4 → E1 → E2

**Estimated Total Time** (with 5 parallel agents):
- Phase A: 2-3 hours
- Phase B: 4-6 hours (parallel)
- Phase C: 3-4 hours (mostly parallel)
- Phase D: 1-2 hours (parallel)
- Phase E: 3-5 hours (sequential)
- **Total**: ~13-20 hours (single-threaded), ~8-12 hours (with 5 agents)
