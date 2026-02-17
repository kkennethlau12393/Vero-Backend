# Feature 2 Test Suite — Grading Rubric

## Scoring Dimensions

| Dimension | Weight | Current Score | Max | Notes |
|---|---|---|---|---|
| **Unit Coverage** | 25% | 22/25 | 25 | Strong coverage of pure functions: scoring, dedup, classification, temporal, query expansion. Missing: convergence.py unit tests, methodological_alignment.py |
| **Integration Coverage** | 25% | 18/25 | 25 | Ranking endpoint, maps endpoints, drill-down, error handling, cache behaviour. Missing: end-to-end with real DB data, temporal map with analytics, subtopic generation with real rank job |
| **Quality/Regression Coverage** | 20% | 17/20 | 20 | Score distributions, category balance, known paper classification, MMR diversity, golden baselines. Missing: full benchmark formalization (no benchmark scripts found in repo) |
| **Edge Cases** | 10% | 8/10 | 10 | Empty inputs, None values, boundary years, single elements, invalid enums. Missing: unicode titles, extremely long abstracts, concurrent rank jobs |
| **Mock Completeness** | 10% | 9/10 | 10 | All external APIs mocked (Groq, OpenAlex, S2, ArXiv, CrossRef, PubMed, DBLP, OpenCitations, OpenAI). Missing: mock for title_to_query LLM call |
| **Documentation** | 10% | 9/10 | 10 | Clear docstrings, follows CLAUDE.md conventions, markers used correctly. This rubric exists. |

**Total: 83/100**

## Current State

### ✅ What's Covered
- **4 unit test files**: classification, categorization, temporal, query expansion
- **2 integration test files**: ranking pipeline, maps endpoints
- **2 quality test files**: ranking quality regression, diversity regression
- **Fixtures**: Saved Groq responses, OpenAlex responses, seed data factories
- **Golden baselines**: Auto-generated on first run for robust_norm, impact_rate, heuristic classification

### 🔲 Gaps to Address

1. **convergence.py unit tests** — The Weibull curve fitting has complex math (fit_curve, evaluate_convergence) that should have dedicated unit tests with known inputs/outputs.

2. **domain_filters.py deeper tests** — Current tests cover detect_domain and compute_domain_alignment. Could add: batch alignment, is_off_domain, edge cases with empty abstracts.

3. **Integration tests with seeded DB data** — Current integration tests rely on the pipeline finding papers via mocked APIs. Would be stronger with pre-seeded works in the test DB.

4. **Temporal map with analytics integration test** — Need a test that creates a rank job, then calls the temporal-map endpoint with `include_analytics=true`.

5. **Subtopic generation integration test** — Need a test that creates a rank job with enough papers, then calls generate-subtopics.

6. **Concurrent request handling** — What happens when two identical rank requests arrive simultaneously? The advisory lock mechanism should be tested.

7. **LLM response edge cases** — More varied garbage responses: empty JSON, partial JSON, wrong schema, rate limit responses (429).

8. **Benchmark formalization** — No benchmark scripts (ranking_benchmark.py etc.) found in repo root. When added, formalize as pytest quality tests.

### 📋 Test Count Summary

| Directory | Files | Test Classes | Approx Tests |
|---|---|---|---|
| tests/unit/ | 6 (existing 2 + new 4) | ~25 | ~95 |
| tests/integration/ | 2 (new) | ~6 | ~15 |
| tests/quality/ | 2 (new) | ~6 | ~20 |
| **Total** | **10** | **~37** | **~130** |

### 🏃 Running

```bash
# Unit tests only (fast, <5s)
pytest tests/unit/ -x -v -m unit

# Integration tests (needs DB, ~30s)
pytest tests/integration/ -x -v -m integration

# Quality/regression tests (<10s)
pytest tests/quality/ -x -v -m quality

# Everything
pytest tests/ --tb=short
```

### 🎯 Priority Improvements

1. **High**: Add convergence.py unit tests (complex math, high regression risk)
2. **High**: Add seeded DB integration tests (currently relies on empty mocked responses)
3. **Medium**: Add concurrent rank job test
4. **Medium**: Add more LLM garbage response variations
5. **Low**: Unicode/i18n title handling tests
