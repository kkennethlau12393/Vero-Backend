# Ranking Output Quality Rubric

> Use this to score the quality of ranking results for any query. Score each dimension, sum for total.

## Scoring Dimensions (100 points)

### 1. Relevance (30 pts)

| Score | Criteria |
|---|---|
| 30 | All top 10 results are directly relevant to the query. No off-topic papers. |
| 25 | 8-9 of top 10 are relevant. 1-2 tangentially related. |
| 20 | 6-7 of top 10 are relevant. Some noise. |
| 15 | 4-5 of top 10 are relevant. Significant noise. |
| 10 | Fewer than 4 relevant in top 10. |
| 0 | Results are mostly irrelevant. |

**Check:** Read top 10 titles. Would a researcher searching this query be satisfied?

### 2. Foundational Coverage (15 pts)

| Score | Criteria |
|---|---|
| 15 | All landmark/seminal papers for this topic are present. The papers any expert would expect to see. |
| 12 | Most landmarks present. Missing 1 notable paper. |
| 9 | Some landmarks present but missing 2-3 key papers. |
| 5 | Only 1-2 landmarks found. Major gaps. |
| 0 | No foundational papers found. |

**Check:** What are the 3-5 papers everyone in this field cites? Are they all here?

### 3. Category Distribution (10 pts)

| Score | Criteria |
|---|---|
| 10 | All relevant categories populated (foundational, methodology, reviews, applications). Distribution feels natural for query type. |
| 8 | Most categories populated. One category empty but shouldn't be. |
| 5 | 2+ categories empty or severely imbalanced (e.g. 90% foundational, nothing else). |
| 2 | Only 1 category has results. |
| 0 | Categories make no sense for the query. |

**Check:** For a broad query, expect all categories. For a methods query, methodology should be well-populated. For a niche query, fewer categories is OK.

### 4. Temporal Diversity (10 pts)

| Score | Criteria |
|---|---|
| 10 | Results span the full relevant timeline. Mix of foundational (older) and recent work. |
| 8 | Good spread but missing either very recent (<2 years) or foundational (>10 years). |
| 5 | Skewed — mostly old or mostly recent. |
| 2 | All results from a single time period. |
| 0 | Temporal distribution makes no sense. |

**Check:** Are there both seminal older papers AND recent advances? Does the timeline match the field's history?

### 5. Source Diversity (5 pts)

| Score | Criteria |
|---|---|
| 5 | Papers from multiple retrieval sources (OpenAlex, S2, ArXiv, etc). Not all from one source. |
| 3 | Mostly one source but some diversity. |
| 0 | All from a single source. |

**Check:** Look at provenance. Are we actually leveraging multi-source retrieval?

### 6. Ranking Order (15 pts)

| Score | Criteria |
|---|---|
| 15 | Top papers are clearly the most important/relevant. Ordering feels right to a domain expert. Score decreases monotonically (no low-relevance paper above a high-relevance one). |
| 12 | Mostly correct ordering. 1-2 papers feel out of place. |
| 9 | Ordering is roughly right but several papers feel misranked. |
| 5 | Ordering is inconsistent. Important papers buried below mediocre ones. |
| 0 | Ordering seems random. |

**Check:** Does #1 deserve to be #1? Would you swap any top 5 papers with ones lower down?

### 7. Paper Type Accuracy (5 pts)

| Score | Criteria |
|---|---|
| 5 | Papers are in the correct categories. Reviews are actually reviews. Methods papers are actually methods. Foundational papers are actually foundational. |
| 3 | Mostly correct. 1-2 miscategorized. |
| 1 | Multiple miscategorizations. |
| 0 | Categories seem random. |

**Check:** Pick 3-5 papers. Are they in the right category?

### 8. Score Calibration (5 pts)

| Score | Criteria |
|---|---|
| 5 | Top scores (>0.8) are genuinely excellent papers. Low scores (<0.4) are genuinely less relevant. Score spread is meaningful — not all bunched at one value. |
| 3 | Scores roughly correlate with quality but some outliers. |
| 1 | Scores don't correlate well with actual relevance. |
| 0 | Scores seem arbitrary. |

**Check:** Is a 0.9 paper clearly better than a 0.5 paper? Or are they basically the same quality?

### 9. Deduplication (5 pts)

| Score | Criteria |
|---|---|
| 5 | No duplicate papers. Each result is a unique work. |
| 3 | 1 duplicate found. |
| 0 | Multiple duplicates. |

**Check:** Scan titles. Any paper appear twice (possibly with slightly different titles across sources)?

## Grading Scale

| Total | Grade | Meaning |
|---|---|---|
| 90-100 | A | Ship it. Results are excellent. |
| 80-89 | B | Good. Minor issues to address. |
| 70-79 | C | Acceptable but needs improvement. |
| 60-69 | D | Below standard. Significant issues. |
| <60 | F | Major problems. Don't ship. |

## How to Use

1. Pick a query
2. Run it through the ranking pipeline
3. Score each dimension
4. Sum the scores
5. Log the result:

```
| Date | Query | Relevance | Foundational | Categories | Temporal | Sources | Order | Types | Calibration | Dedup | Total | Grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-02-11 | attention mechanisms | 28 | 15 | 10 | 8 | 5 | 12 | 5 | 4 | 5 | 92 | A |
```

## When to Use

- After changing the ranking pipeline (v84 → v85)
- After changing the LLM prompt (v27 → v28)  
- After changing retrieval logic
- After changing weights or normalization
- Before shipping a new version
- When running live tests — score a sample of results

## Query Difficulty Tiers

Not all queries are equal. Note the difficulty when scoring:

- **Easy:** Well-known broad field (e.g. "deep learning", "CRISPR")
- **Medium:** Specific subfield (e.g. "graph neural networks node classification")
- **Hard:** Cross-domain intersection (e.g. "attention mechanisms in medical imaging")
- **Adversarial:** Ambiguous, niche, or deliberately tricky queries

A score of 75 on a hard query might be better than 85 on an easy one.
