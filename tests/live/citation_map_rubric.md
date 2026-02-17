# Citation Map Output Quality Rubric

> Use this to score the quality of citation map results for any query. Score each dimension, sum for total.

## Scoring Dimensions (100 points)

### Automated Scoring (~35 pts)

#### 1. Graph Structure (10 pts)

| Score | Criteria |
|---|---|
| 10 | ≥20 nodes, ≥15 edges, no orphan nodes, graph is well-connected. |
| 8 | ≥15 nodes, ≥10 edges, ≤2 orphans. |
| 6 | ≥10 nodes, ≥6 edges. Some orphans. |
| 3 | ≥5 nodes but sparse edges or many orphans. |
| 0 | <5 nodes or empty graph. |

**Check:** Is the graph rich enough to be useful for exploration? Are all non-seed nodes connected via edges?

#### 2. Deduplication (5 pts)

| Score | Criteria |
|---|---|
| 5 | No duplicate work_ids. Every node is a unique paper. |
| 3 | 1 duplicate found. |
| 0 | Multiple duplicates. |

**Check:** Scan work_ids. Any paper appear twice?

#### 3. Hop Distribution (5 pts)

| Score | Criteria |
|---|---|
| 5 | Mix of hop-0 (seed), hop-1 (direct), and hop-2 (network) nodes. At least 3 distinct hop levels. |
| 3 | 2 hop levels present (seed + 1-hop, or seed + 2-hop). |
| 1 | All nodes at same hop level (besides seed). |
| 0 | No hop information or broken hop values. |

**Check:** Is the network exploring beyond direct citations? Are there discovery papers at hop-2?

#### 4. Year Diversity (5 pts)

| Score | Criteria |
|---|---|
| 5 | Papers span ≥3 decades, spread ≥15 years. Mix of foundational (older) and recent work. |
| 3 | Papers span ≥2 decades, spread ≥5 years. |
| 1 | All papers from roughly the same time period (<5 year spread). |
| 0 | No year information available. |

**Check:** Does the graph show the evolution of a research area over time?

#### 5. Citation Diversity (5 pts)

| Score | Criteria |
|---|---|
| 5 | Mix of highly-cited papers (≥100) and emerging/newer papers (<100). At least 3 of each. |
| 3 | Some mix but skewed (mostly high or mostly low cited). |
| 1 | All papers in same citation tier. |
| 0 | No citation count information. |

**Check:** Does the graph include both seminal high-impact papers AND newer contributions?

#### 6. Edge Correctness (5 pts)

| Score | Criteria |
|---|---|
| 5 | All edges have valid endpoints (both from_work_id and to_work_id exist in nodes). No self-loops. Edge direction (from cites to) is consistent. |
| 3 | 1-2 edges with missing endpoints or self-loops. |
| 0 | Multiple broken edges or inconsistent direction semantics. |

**Check:** Do all edges connect real nodes? Are there self-referential edges?

---

### LLM-Judged Scoring (~65 pts)

#### 7. Seed Relevance (20 pts)

| Score | Criteria |
|---|---|
| 20 | The seed paper is THE seminal/most important paper for this query. A domain expert would pick the same paper. |
| 16 | Seed is a strong, relevant paper but not the single best choice. Top-3 candidate. |
| 12 | Seed is relevant to the topic but not a top choice. A related but not central paper. |
| 8 | Seed is tangentially related. Same broad field but not the right subarea. |
| 4 | Seed is barely related. Wrong field or very marginal connection. |
| 0 | Seed is completely irrelevant to the query. |

**Check:** Given the query, is this the paper you'd start a citation exploration from? What are the 3 papers everyone in this field cites — is the seed one of them?

#### 8. Network Relevance (20 pts)

| Score | Criteria |
|---|---|
| 20 | All connected papers are topically relevant. The graph tells a coherent story about the research area. No citation noise. |
| 16 | Most papers (>80%) are relevant. 1-3 tangentially related papers. |
| 12 | About 60-80% relevant. Some noise from citation chains that drift off-topic. |
| 8 | About 40-60% relevant. Significant noise. |
| 4 | Mostly noise. Few relevant papers beyond the seed. |
| 0 | Network is entirely off-topic. |

**Check:** Read the titles of all nodes. Would a researcher exploring this topic find them useful? Or are many just "citation noise" — papers that happen to cite the seed but aren't about the same topic?

#### 9. Foundational Coverage (10 pts)

| Score | Criteria |
|---|---|
| 10 | The key seminal papers in this research area are present in the graph. The "must-cite" papers are all there. |
| 8 | Most landmarks present. Missing 1 notable paper that should be there. |
| 5 | Some important papers present but 2-3 key ones missing. |
| 2 | Only 1-2 important papers found. Major gaps in foundational coverage. |
| 0 | No foundational papers in the graph (besides possibly the seed). |

**Check:** What are the 3-5 papers everyone in this field cites? Are they in the graph as nodes? If studying "transformers", is the original Attention paper there? BERT? GPT?

#### 10. Discovery Value (10 pts)

| Score | Criteria |
|---|---|
| 10 | The graph surfaces interesting, non-obvious papers. A researcher would discover papers they didn't know about. Hop-2 papers add genuine value. |
| 8 | Some discovery value. 1-2 surprising/useful papers found through indirect connections. |
| 5 | Mostly obvious papers. The graph shows what you'd find with a simple search. |
| 2 | No discovery value. All papers are the most obvious results. |
| 0 | Graph is too small or too noisy for any discovery. |

**Check:** Would this graph teach a PhD student something they couldn't find with a Google Scholar search? Are there interesting 2-hop connections?

#### 11. Relationship Quality (5 pts)

| Score | Criteria |
|---|---|
| 5 | Edges represent meaningful citation relationships. The citation chains make sense (paper A cites paper B because B's work is foundational to A). |
| 3 | Most edges are meaningful. 1-2 seem like coincidental citations. |
| 1 | Many edges feel like noise — papers citing each other for tangential reasons. |
| 0 | Edges don't represent meaningful intellectual relationships. |

**Check:** Pick 3-5 edges. Does the citing paper actually build on the cited paper's work?

---

## Grading Scale

| Total | Grade | Meaning |
|---|---|---|
| 90-100 | A | Excellent citation map. Ship it. |
| 80-89 | B | Good. Minor issues to address. |
| 70-79 | C | Acceptable but needs improvement. |
| 60-69 | D | Below standard. Significant issues. |
| <60 | F | Major problems. Don't ship. |

## How to Use

1. Pick a query (or run live tests)
2. Run it through the citation map pipeline
3. Score each dimension (automated scores computed by `_compute_automated_score()`)
4. Use your own knowledge to evaluate LLM-judged dimensions
5. Sum the scores
6. Log the result:

```
| Date | Query | Structure | Dedup | Hops | Years | Citations | Edges | Seed | Network | Foundational | Discovery | Relationships | Total | Grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-02-12 | transformers | 10 | 5 | 5 | 5 | 5 | 5 | 20 | 16 | 10 | 8 | 5 | 94 | A |
```

## When to Use

- After changing the citation map pipeline or seed selection logic
- After changing multi-hop expansion parameters
- After changing retrieval sources or citation APIs
- Before shipping a new version
- When running live tests — score a sample of results

## Query Difficulty Tiers

Not all queries are equal. Note the difficulty when scoring:

- **Easy:** Well-known field with clear seminal paper (e.g. "attention is all you need")
- **Medium:** Specific subfield with multiple valid seeds (e.g. "graph neural networks")
- **Hard:** Cross-domain, broad, or niche queries (e.g. "attention mechanisms in medical imaging")
- **Adversarial:** Ambiguous terms, very new fields, or single-word queries

A score of 75 on a hard query might be better than 85 on an easy one.
