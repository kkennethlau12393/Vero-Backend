# Review: Undermind Search Assistant Paper — Applicability to Vero

**Paper**: "Benchmarking the Undermind Search Assistant" by Thomas Hartke & Joshua Ramette (Jan 2024, Undermind.ai)

---

## Paper Summary

Undermind is an academic literature search system that uses LLMs (GPT-4) as a reasoning engine and relevance classifier within a structured, iterative search process over ArXiv (2.3M papers). Compared to Google Scholar's top 5 pages (~50 results), Undermind finds **10x more relevant papers** for the median query and misses virtually none that Google Scholar surfaces.

### Undermind's Four-Step Algorithm

1. **Basic search** — Semantic vector embeddings + citations + LLM reasoning to find initial candidates
2. **Relevance classification** — GPT-4 reads full paper text and classifies each candidate into three tiers: *highly relevant*, *closely related*, or *ignorable*
3. **Adaptation and exploration** — The algorithm iterates: it adapts search strategy based on what it has found, follows citation trails, and re-searches, mimicking a human researcher's discovery process
4. **Estimating comprehensiveness** — Tracks discovery rate over time. The rate of finding new relevant papers follows an exponential decay curve (`f = 1 - e^(-n/τ)`), which lets Undermind estimate when it has found nearly all relevant work

### Key Results

- 10x concentration of relevant results in top hits vs. Google Scholar
- Three-tier classification accuracy: ~98% (highly relevant papers are almost never classified as irrelevant)
- Exhaustiveness: >97% of Google Scholar's highly relevant papers are also found by Undermind
- Convergence detection: system knows when to stop searching

---

## Applicable Methods for Vero

### 1. LLM-Based Relevance Classification (HIGH PRIORITY)

**What Undermind does**: Uses GPT-4 to read the full text of each candidate paper and classify it into three tiers (highly relevant / closely related / ignorable) given the user's query and context.

**How to apply to Vero**: Your `CandidateItem` already has a `relevance_score` (float 0-1). Consider replacing or augmenting this with:

- A **discrete classification tier** (highly relevant / closely related / not relevant) in addition to the continuous score. This maps well to your existing `ScoreBreakdown` but would add a user-facing label.
- **Full-text classification** — rather than scoring papers on metadata alone (title, abstract, citation count), send the abstract (and ideally full text when available) plus the user's query to an LLM for classification.
- **Transparent reasoning** — Undermind explains *why* each paper is relevant. Your `RankedItem.reasons` field is already designed for this. Populate it with LLM-generated explanations.

**Concrete model changes**:
- Add a `relevance_tier: Literal["highly_relevant", "closely_related", "not_relevant"]` to `CandidateItem` or `RankedItem`
- Add an `llm_explanation: Optional[str]` field for the LLM's reasoning about relevance

### 2. Iterative Adaptive Search (HIGH PRIORITY)

**What Undermind does**: After an initial search, it examines what it found, then adapts its search strategy — following citation trails from relevant papers, adjusting search terms, exploring related areas. This is the core differentiator.

**How to apply to Vero**: Your current flow is `Seed → CandidateSet → Graph → Map → RankedList`. This is a single-pass pipeline. To adopt Undermind's approach:

- **Multi-round discovery**: After the first `CandidateSet` is built, run LLM classification on the top candidates. Then use the *highly relevant* results as new seeds for a second round of discovery. Repeat until diminishing returns.
- **Citation trail following**: Your `GraphDraft` already models citation edges. Use these edges actively during discovery — if paper A is highly relevant, fetch everything it cites and everything that cites it, then classify those too.
- **The discovery loop** could be modeled as multiple `CandidateSet` iterations linked together, or as progressive expansion of a single set.

**Concrete model changes**:
- Add `iteration: int` and `parent_candidate_set_id: Optional[str]` to `CandidateSet` to track multi-round discovery
- Add `discovery_source: Literal["initial_search", "citation_forward", "citation_backward", "adapted_query"]` to `CandidateItem.reason` vocabulary

### 3. Exponential Convergence Detection (MEDIUM PRIORITY)

**What Undermind does**: Tracks how frequently new relevant papers are discovered over time. This follows an exponential decay: `f = 1 - e^(-n/τ)`. By fitting τ (time constant), Undermind can estimate the total number of relevant papers and know when the search is ~85-98% complete.

**How to apply to Vero**: This is valuable for knowing when a search/graph-build is "done."

- Track the **discovery curve** as papers are evaluated: for every N papers examined, how many new relevant ones were found?
- Fit the exponential model to estimate total relevant papers and current completeness percentage.
- Surface this as a **completeness indicator** to users (e.g., "We've likely found ~90% of relevant papers").

**Concrete model changes**:
- Add to `GraphStats` or create a new model:
  ```
  class DiscoveryProgress(BaseModel):
      papers_evaluated: int
      relevant_found: int
      estimated_total: Optional[int]
      estimated_completeness: Optional[float]  # 0.0 to 1.0
      time_constant_tau: Optional[float]
  ```

### 4. Three-Tier Result Presentation (MEDIUM PRIORITY)

**What Undermind does**: Clearly separates results into "highly relevant" (flagged at top, with explanations) and "closely related" (relevant but not central). This is much more useful than a flat ranked list.

**How to apply to Vero**: Your `RankedList` currently returns a flat list of `RankedItem`s. Consider:

- Grouping results by tier in the API response
- Putting "highly relevant" papers at the top with detailed explanations
- Making "closely related" papers available but visually separated
- This directly improves the user experience of your Maps and ranked results

### 5. Complex Query Understanding (LOWER PRIORITY)

**What Undermind does**: Accepts verbose, natural-language queries with multiple constraints (e.g., "experimental papers on tapered optical fibers coupling light into microfabricated waveguides in the visible spectrum"). The LLM decomposes this into search strategies that a keyword engine cannot handle.

**How to apply to Vero**: Your `TopicQueryRef` and `seed_type: "topic_text"` already support text queries. To go further:

- Use an LLM to decompose complex queries into multiple sub-queries
- Generate multiple search strategies from a single user query (Undermind essentially does what a skilled human researcher would: try several keyword angles)
- Your `params_json` on `CandidateSet` could store the decomposed search strategy

---

## Methods That Are Less Applicable

| Undermind Method | Why Less Relevant to Vero |
|---|---|
| ArXiv full-text search | Vero uses OpenAlex, which has different coverage. Full-text search would require a separate corpus. Could be relevant if you add PDF ingestion. |
| Google Scholar comparison methodology | Useful for their benchmarking, but not for product features. |
| Exhaustiveness estimation via cross-method comparison (Eq. 1) | Interesting statistical technique but primarily useful for evaluating the system, not as a runtime feature. |

---

## Recommended Implementation Priority

1. **LLM relevance classification** — Biggest bang for effort. Add GPT-4/Claude-based classification of candidates using abstract + user query. Populate `reasons` and add a tier label.
2. **Iterative adaptive search** — Transform single-pass discovery into a multi-round loop. Use citation edges and highly-relevant results as seeds for subsequent rounds.
3. **Convergence detection** — Add exponential curve fitting to know when to stop expanding the graph.
4. **Tiered result presentation** — Group ranked results by relevance tier instead of flat scores.
5. **Query decomposition** — Use LLM to break complex queries into multiple search strategies.

---

## Key Takeaway

Undermind's core insight is that **LLMs should be embedded throughout the search pipeline** — not just for query understanding, but for classifying every result, adapting search strategy, and explaining relevance. Vero's architecture already has the right data models (score breakdowns, reasons, candidate metadata), but the actual intelligence layer that uses LLMs at classification and adaptation time is the critical missing piece that Undermind demonstrates works extremely well.
