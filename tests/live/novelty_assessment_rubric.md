# Novelty Assessment Output Quality Rubric (100 points)

> Use this to score novelty assessment results from live tests.
> Complements the 10-point grading spec (`docs/novelty_assessment_grading_spec.md`).

## Automated Scoring (35 pts) — computed programmatically

### 1. Schema Completeness (5 pts)
| Score | Criteria |
|---|---|
| 5 | All required fields present, valid enums, grounding papers 5-7 |
| 3 | 1-2 missing/invalid fields |
| 0 | 3+ missing/invalid fields |

### 2. Grounding Paper Structure (10 pts)
| Score | Criteria |
|---|---|
| 10 | 5-7 grounding papers, mix of cited_reference + field_landmark, min 3 landmarks, all work_ids valid (W-prefixed), no self-citation, relevance >20 chars |
| 7 | 4-5 grounding papers, some mix, no major issues |
| 4 | 3-4 grounding papers, or missing landmarks, or boilerplate relevance |
| 0 | <3 grounding papers or self-citation present |

### 3. Novelty Level Accuracy (5 pts)
| Score | Criteria |
|---|---|
| 5 | Exact match with expected level |
| 3 | Adjacent level (off by 1: low<->medium, medium<->high, high<->pioneering) |
| 1 | Off by 2 levels |
| 0 | Off by 3+ or review/software rated pioneering |

### 4. Banned Verb Absence (5 pts)
| Score | Criteria |
|---|---|
| 5 | Zero banned verbs in summary, whats_new, compared_to, explanation, relevance |
| 3 | 1 banned verb found |
| 0 | 2+ banned verbs found |

### 5. Grounding Consistency (5 pts)
| Score | Criteria |
|---|---|
| 5 | All work_ids cited in explanation appear in grounding_papers, no orphaned citations |
| 3 | 1 orphaned citation |
| 0 | 2+ orphaned citations or consistency_score < 0.5 |

### 6. Cross-Domain Absence (5 pts)
| Score | Criteria |
|---|---|
| 5 | All grounding papers from same domain/field as target |
| 3 | 1 cross-domain paper |
| 0 | 2+ cross-domain papers or completely unrelated field |

---

## LLM-Judged Scoring (65 pts) — REQUIRED on every test run

### 7. Summary Quality (15 pts)
| Score | Criteria |
|---|---|
| 15 | States specific contribution with technical detail, no hedging language |
| 10 | States contribution but somewhat generic or vague |
| 5 | Identifies topic but vague on actual contribution |
| 0 | Fails to identify contribution, uses hedging ("likely", "might"), or echoes title only |

### 8. Novelty Explanation Quality (15 pts)
| Score | Criteria |
|---|---|
| 15 | Cites 2+ work_ids with specific technical claims explaining WHY this novelty level |
| 10 | Cites work_ids OR makes specific claims (not both) |
| 5 | Generic explanation with some substance |
| 0 | Restates Q0-Q3 decision tree without substance, or no technical claims |

### 9. Grounding Paper Relevance (15 pts)
| Score | Criteria |
|---|---|
| 15 | All papers directly relevant to the target's domain and methodology |
| 10 | 1 tangentially related paper (same broad field but different subarea) |
| 5 | 2 irrelevant papers |
| 0 | 3+ irrelevant papers or completely wrong domain |

### 10. Whats-New / Compared-To Quality (10 pts)
| Score | Criteria |
|---|---|
| 10 | Specific, actionable descriptions with work_id citations, explains novelty clearly |
| 7 | Good but could be more specific or is missing citations |
| 4 | Generic, vague, or null when shouldn't be |
| 0 | Null when required (non-pioneering), or completely wrong |

### 11. Overall Coherence (10 pts)
| Score | Criteria |
|---|---|
| 10 | Novelty level, explanation, grounding, and summary tell a consistent story |
| 7 | Mostly consistent with minor contradictions |
| 3 | Some contradictions (e.g., "medium" but explanation describes major breakthrough) |
| 0 | Internally contradictory assessment |

---

## Grading Scale

| Total | Grade | Meaning |
|---|---|---|
| 90-100 | A | Production ready, ship it |
| 80-89 | B | Good with minor issues |
| 70-79 | C | Acceptable but needs improvement |
| 60-69 | D | Below standard, investigate |
| <60 | F | Major problems, requires rework |

---

## Workflow

1. Run live tests → results saved to `tests/live/results/`
2. Automated scoring runs automatically (35 pts) — computed by `_compute_automated_score()` in test file
3. **LLM-judged scoring MUST be performed manually on all results (65 pts)** — review saved results and score each dimension
4. Combined score (100 pts total) logged with breakdown per paper
5. Flag anything below 70/100 for investigation
6. Compare scores across pipeline versions to catch regressions

## When to Score
- After changing novelty assessment logic or LLM prompts
- After changing grounding supplement or cross-domain filtering
- After changing `ASSESSMENT_VERSION` (cache invalidation)
- Before shipping a new version

## Relationship to 10-Point Grading Spec
- The 10-point grading spec (`docs/novelty_assessment_grading_spec.md`) is used during **iteration** (Ralph Wiggum loops)
- This 100-point rubric is used during **regression testing** (live test runs)
- Both should agree: a paper scoring 9+/10 should score 85+/100
