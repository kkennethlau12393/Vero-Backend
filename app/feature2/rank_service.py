"""
Ranking services for Feature 2 (Direct Ranking).

This module implements both the original v1 direct ranking and an
experimental production‑grade ranking pipeline.  The production
pipeline uses the Stage‑1 retrieval layer to build a candidate pool
from OpenAlex and pgvector, then computes graded relevance features
(semantic, lexical, topic, impact, recency and completeness),
normalises them robustly, blends them with configurable weights and
diversifies the top‑K results using Max‑Marginal‑Relevance (MMR).

The original `direct_rank` function has been retained for backwards
compatibility and continues to score candidates using a simple topic
match, logarithmic citation impact and exponential recency.  The new
`direct_rank_prod` function performs end‑to‑end retrieval, ranking
and explanation based on the new feature functions defined in
`features.py` and `rerank.py`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional
from uuid import UUID
from datetime import date, datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

from .repos import CandidateSetRepo, RankRepo
from .work_topic_store import WorkStore, WorkForMap, TopicHierarchyStore
from .retrieval import generate_candidates_direct, PIPELINE_VERSION as RETRIEVAL_PIPELINE_VERSION
from .features import (
    robust_norm,
    bayesian_impact_rate,
    compute_recency,
    compute_topic_relevance,
    compute_completeness,
    compute_age,
    compute_llm_relevance_feature,
)
from .rerank import build_reasons, build_reasons_with_categories, assemble_ranked_results
from .llm_relevance import score_papers, score_wave
from .convergence import (
    WAVE_SIZE, MAX_WAVES, evaluate_convergence, WaveResult, ConvergenceState,
)
from .tfidf_similarity import compute_tfidf_vectors
from .query_classification import classify_query, QueryType, QuerySpecificity
from .methodological_alignment import (
    partition_results_by_category,
    DEFAULT_CATEGORY_LIMITS,
)


# Ranking pipeline version - increment to invalidate rank caches when logic changes
# v70: Applications sorted by 65% citations + 35% recency (recent applications matter)
# v71: Audit fixes - citation-floor boost requires lexical evidence, store paper_type
#      in cache, raise foundational LLM threshold to 0.40, zero dead topic weight,
#      graceful empty-candidate handling
# v72: LLM prompt v19 - intersection-aware scoring for compound queries
# v73: Raised MIN_LLM_RELEVANCE_THRESHOLD 0.50→0.60, FOUNDATIONAL_MIN_RELEVANCE 0.40→0.60
# v74: LLM prompt v20 - stricter 2-of-3 concept capping at MEDIUM
# v75: LLM prompt v21 - single-topic vs intersection, sharper MEDIUM/LOW,
#      FOUNDATIONAL_MIN_RELEVANCE 0.60→0.70, removed dead citation-floor boost
# v76: LLM prompt v22 - technique≠domain rule, sub-task boundaries, stronger modality
# v77: LLM prompt v23 - research-contribution vs application-use, generic-theory cap,
#      temperature 0.0 for deterministic scoring
# v78: Programmatic modality boundary enforcement - catches LLM false positives where
#      same technique name (attention, transformer, RL) is used in different domain
# v81: LLM prompt v26 - system-boundary principle, intersection METHOD+DOMAIN decomposition
# v82: Programmatic boundary enforcement expanded - scientific system groups (bacterial vs
#      oncology vs parasitology) and generative method groups (GAN vs diffusion vs VAE)
# v83: Tool paper detection (cap at methodology), textbook category,
#      foundational sort by LLM relevance then citations
# v84: LLM prompt v27 - adjacent-phenomena specificity, cause-effect topic constraint
RANKING_VERSION = "rank-v84"


def _stable_rank_hash(
    *,
    tenant_id: UUID,
    candidate_set_id: UUID,
    context_json: Dict[str, Any],
    filters_json: Dict[str, Any],
    rank_params_json: Dict[str, Any],
) -> str:
    """Compute a deterministic hash for ranking jobs.

    This helper produces a SHA‑256 hash of the sorted JSON inputs and
    identifiers to ensure idempotent rank job creation.  It is used
    internally by both the v1 and production ranking pipelines.
    """
    s = "|".join(
        [
            RANKING_VERSION,  # Include version to invalidate caches on logic changes
            str(tenant_id),
            str(candidate_set_id),
            json.dumps(context_json or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(filters_json or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(rank_params_json or {}, sort_keys=True, separators=(",", ":")),
        ]
    )
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pick_best_topic_id(w: WorkForMap) -> Optional[str]:
    """Return the best topic ID for a work.

    If a work has a primary topic ID, use it.  Otherwise choose the
    topic with the highest score from the `topics` list, breaking
    ties by lexicographically smallest topic_id.  Returns None if no
    topics are available.
    """
    if w.primary_topic_id is not None:
        return w.primary_topic_id
    if not w.topics:
        return None
    best = sorted(w.topics, key=lambda t: (-t.score, t.topic_id))[0]
    return best.topic_id




def direct_rank_prod(
    engine: Engine,
    *,
    tenant_id: UUID,
    query_text: str,
    context_json: Optional[Dict[str, Any]] = None,
    filters_json: Optional[Dict[str, Any]] = None,
    rank_params_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Production‑grade direct ranking pipeline.

    This function orchestrates Stage‑1 retrieval, candidate set
    persistence, graded feature computation, robust normalisation,
    weighted scoring, diversification and explanation.  It does not
    require a pre‑existing candidate_set_id; instead it accepts a
    query string and context/filters and builds or reuses a
    deterministic candidate set via its `params_hash`.

    The ranking results are persisted in the `rank_results` table with
    provenance and breakdowns for post‑hoc analysis.
    """
    context_json = context_json or {}
    filters_json = filters_json or {}
    rank_params_json = rank_params_json or {}

    # Build a stable params hash keyed on the query and filters
    s = "|".join(
        [
            str(tenant_id),
            RETRIEVAL_PIPELINE_VERSION,
            query_text,
            json.dumps(context_json or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(filters_json or {}, sort_keys=True, separators=(",", ":")),
            json.dumps(rank_params_json or {}, sort_keys=True, separators=(",", ":")),
        ]
    )
    params_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Create or reuse candidate set
    with engine.begin() as tx:
        candidate_set_id = CandidateSetRepo.insert_or_get_candidate_set(
            tx,
            tenant_id=tenant_id,
            seed_type="direct_query",
            seed_json={"query_text": query_text},
            params_hash=params_hash,
        )
        logger.info(f"Candidate set ID: {candidate_set_id}")
        # If candidate set is empty, generate candidates
        existing_rows = CandidateSetRepo.load_work_ids_and_provenance(
            tx, tenant_id, candidate_set_id
        )
        logger.info(f"Existing rows in candidate set: {len(existing_rows)}")
        query_expansion = None
        if existing_rows:
            legacy_sources = {"semantic_knn", "lexical_openalex"}
            has_legacy = any(
                any(item.get("source") in legacy_sources for item in (row.get("provenance") or []))
                for row in existing_rows
            )
            if has_legacy:
                # Clear legacy candidates so the new pipeline can regenerate.
                tx.execute(
                    text(
                        "DELETE FROM candidate_set_items WHERE candidate_set_id = :cs_id"
                    ),
                    {"cs_id": candidate_set_id},
                )
                existing_rows = []
                logger.info("Cleared legacy candidate_set_items for regeneration")
        if not existing_rows:
            candidates, query_expansion = generate_candidates_direct(
                tx,
                tenant_id=tenant_id,
                query_text=query_text,
                context_json=context_json,
                filters_json=filters_json,
                rank_params_json=rank_params_json,
            )
            logger.info(f"Generated {len(candidates)} candidates")
            CandidateSetRepo.bulk_insert_candidate_items(tx, candidate_set_id, candidates)
            logger.info(f"Inserted candidates into DB")

    # Build a rank job using a separate params hash for ranking (ties
    # ranking to candidate_set_id plus parameters)
    params_hash_rank = _stable_rank_hash(
        tenant_id=tenant_id,
        candidate_set_id=candidate_set_id,
        context_json=context_json,
        filters_json=filters_json,
        rank_params_json=rank_params_json,
    )
    with engine.begin() as tx:
        rank_job_id, created_new, status = RankRepo.insert_pending_or_get_existing(
            tx,
            tenant_id=tenant_id,
            rank_type="direct_prod",
            candidate_set_id=candidate_set_id,
            context_json=context_json,
            filters_json=filters_json,
            rank_params_json=rank_params_json,
            params_hash=params_hash_rank,
        )
        if not created_new and status == "completed":
            loaded = RankRepo.load_results(tx, tenant_id, rank_job_id)
            # The RankRepo.load_results returns "items" list format
            # We need to re-categorize these items for the response
            # This is faster than re-running the entire pipeline
            if "items" in loaded:
                # Re-categorize cached items using their stored score_breakdown
                items = loaded.get("items", [])
                if not items:
                    # No items cached - need to re-run
                    tx.execute(
                        text("DELETE FROM rank_results WHERE rank_job_id = :job_id"),
                        {"job_id": rank_job_id},
                    )
                    logger.info(f"Empty cached rank job {rank_job_id}, re-running")
                    created_new = True
                else:
                    # Return cached results
                    logger.info(f"Returning cached rank job {rank_job_id} with {len(items)} items")

                    # Check if skip_categorization is set (drill-down endpoint)
                    skip_categorization = rank_params_json.get("skip_categorization", False)
                    if skip_categorization:
                        # Return flat list of items for drill-down (no categorization)
                        # Get query classification for the response
                        cached_query_class = classify_query(tx, query_text)
                        top_k = rank_params_json.get("top_k", 10)
                        return {
                            "rank_job_id": rank_job_id,
                            "job": loaded.get("job", {}),
                            "query_classification": {
                                "type": cached_query_class.query_type.value,
                                "confidence": cached_query_class.confidence,
                                "query_specificity": cached_query_class.query_specificity.value if cached_query_class.query_specificity else None,
                            },
                            "items": [
                                {
                                    "rank_index": idx,
                                    "work_id": item["work_id"],
                                    "score": item["score"],
                                    "reasons": item.get("reasons", []),
                                    "score_breakdown": item.get("score_breakdown", {}),
                                    "preview": item.get("preview", {}),
                                    "provenance": item.get("provenance", []),
                                }
                                for idx, item in enumerate(items[:top_k])
                            ],
                        }

                    # Re-categorize cached results using stored paper_type + citations
                    cached_query_class = classify_query(tx, query_text)

                    # Reconstruct llm_scores from cached breakdowns
                    cached_llm_scores = {}
                    for item in items:
                        wid = item["work_id"]
                        breakdown = item.get("score_breakdown", {})
                        norm = breakdown.get("norm", {})
                        cached_llm_scores[wid] = {
                            "score": norm.get("llm_relevance", 0.0),
                            "paper_type": breakdown.get("paper_type", "other"),
                        }

                    # Use partition_results_by_category for proper categorization
                    categorized = partition_results_by_category(
                        items,
                        cached_llm_scores,
                        category_limits=None,
                        query_text=query_text,
                    )

                    def _fmt_cached(item, idx):
                        return {
                            "rank_index": idx,
                            "work_id": item["work_id"],
                            "score": item["score"],
                            "reasons": item.get("reasons", []),
                            "score_breakdown": item.get("score_breakdown", {}),
                            "preview": item.get("preview", {}),
                            "provenance": item.get("provenance", []),
                        }

                    return {
                        "rank_job_id": rank_job_id,
                        "job": loaded.get("job", {}),
                        "query_classification": {
                            "type": cached_query_class.query_type.value,
                            "confidence": cached_query_class.confidence,
                            "query_specificity": cached_query_class.query_specificity.value if cached_query_class.query_specificity else None,
                        },
                        "foundational": [
                            _fmt_cached(item, idx) for idx, item in enumerate(categorized["foundational"])
                        ],
                        "methodology": [
                            _fmt_cached(item, idx) for idx, item in enumerate(categorized["methodology"])
                        ],
                        "reviews": [
                            _fmt_cached(item, idx) for idx, item in enumerate(categorized["reviews"])
                        ],
                        "applications": [
                            _fmt_cached(item, idx) for idx, item in enumerate(categorized["applications"])
                        ],
                        "textbooks": [
                            _fmt_cached(item, idx) for idx, item in enumerate(categorized.get("textbooks", []))
                        ],
                    }
            else:
                return {"rank_job_id": rank_job_id, **loaded}
        if not created_new and status in ("pending", "running"):
            return {
                "rank_job_id": rank_job_id,
                "job": {"rank_job_id": rank_job_id, "status": status},
                "foundational": [],
                "methodology": [],
                "reviews": [],
                "applications": [],
                "textbooks": [],
            }
        if not created_new and status == "failed":
            # Failed jobs should be retried - delete old results and job, then re-run
            tx.execute(
                text("DELETE FROM rank_results WHERE rank_job_id = :job_id"),
                {"job_id": rank_job_id},
            )
            tx.execute(
                text("DELETE FROM rank_jobs WHERE rank_job_id = :job_id"),
                {"job_id": rank_job_id},
            )
            logger.info(f"Deleted failed rank job {rank_job_id} for retry")
            # Recursively call to create a new job
            return direct_rank_prod(
                engine,
                tenant_id=tenant_id,
                query_text=query_text,
                context_json=context_json,
                filters_json=filters_json,
                rank_params_json=rank_params_json,
            )
        RankRepo.mark_running(tx, rank_job_id)

    try:
        with engine.begin() as conn:
            # Load candidate IDs and provenance for scoring
            rows = CandidateSetRepo.load_work_ids_and_provenance(
                conn, tenant_id, candidate_set_id
            )
            work_ids = [r["work_id"] for r in rows]
            provenance_map: Dict[str, list] = {r["work_id"]: r.get("provenance", []) for r in rows}

            if not work_ids:
                raise ValueError("candidate_set_empty")

            # Hard filters and limits
            def safe_int(val, default=None):
                if val is None:
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            year_min = safe_int(filters_json.get("year_min"))
            year_max = safe_int(filters_json.get("year_max"))
            topic_id_filter = filters_json.get("topic_id")  # Strict topic filter for "Research this topic"
            # Category limits can be overridden via rank_params_json
            # Default: Fundamentals 15 + Core 15 + Recent 5 + Applications 3 + Specific 2 = 40
            category_limits = rank_params_json.get("category_limits") or DEFAULT_CATEGORY_LIMITS
            total_target = sum(v for k, v in category_limits.items() if k != "implementation_resources")
            # top_k should be large enough to fill all categories with buffer for distribution variance
            top_k = safe_int(rank_params_json.get("top_k"), total_target * 3)
            max_candidates_scored = safe_int(rank_params_json.get("max_candidates_scored"), 3000)
            if max_candidates_scored < 1:
                max_candidates_scored = 3000

            # Smart candidate selection: prioritize by combined provenance + citation score
            # This ensures the LLM scoring cap gets the best candidates - both those
            # with high retrieval scores AND foundational papers with high citations.
            # Also reserve slots for recent papers (2018+) to ensure temporal diversity.

            # Custom LLM scoring cap (for lightweight operations like drill-down)
            llm_scoring_cap = safe_int(rank_params_json.get("llm_scoring_cap"))

            # Query classification for methodological awareness and specificity detection
            # Run early since we need specificity for candidate selection parameters
            query_classification = classify_query(conn, query_text)
            logger.info(
                f"Query classified as: {query_classification.query_type.value} "
                f"(confidence={query_classification.confidence:.2f}), "
                f"specificity={query_classification.query_specificity.value}"
            )

            # Specific query handling: tighter precision, smaller pool, higher threshold
            # Only apply if not explicitly overridden by rank_params_json
            is_specific_query = (
                query_classification.query_specificity == QuerySpecificity.SPECIFIC
                and not rank_params_json.get("force_broad_mode", False)
            )

            if is_specific_query:
                logger.info("SPECIFIC QUERY MODE: Using efficiency optimizations")
                # SPECIFIC mode uses same quality threshold as BROAD (0.50) but smaller candidate pools
                # This maintains quality while being more efficient for focused queries
                # Note: Don't use stricter threshold - causes sparse results for intersection topics
                if llm_scoring_cap is None:
                    llm_scoring_cap = 150  # Smaller than BROAD but enough for categories
                # NOTE: Removed skip_categorization for specific queries since we always want
                # categorized output (foundational/methodology/reviews/applications)
                # Use single S2 bulk search (2000 papers) instead of multiple searches
                # This avoids rate limiting issues and is sufficient for specific queries
                if "single_s2_search" not in rank_params_json:
                    rank_params_json = {**rank_params_json, "single_s2_search": True}

            # Pre-load works to check years and citations for selection
            works_preload = WorkStore.load_many(conn, work_ids)

            # =================================================================
            # THREE-TIER CANDIDATE SELECTION
            # Simple, guaranteed coverage approach:
            #   Tier 1: Top by citations (foundational coverage)
            #   Tier 2: Top recent papers by citations (temporal coverage)
            #   Tier 3: Top by lexical match (query-specific coverage)
            # =================================================================

            current_year = datetime.now().year
            recent_cutoff_year = current_year - 3  # Last 3 years

            def get_lexical_score(wid: str) -> float:
                """Get best lexical match score from provenance."""
                prov = provenance_map.get(wid, [])
                lexical_scores = [
                    float(p.get("score", 0.0))
                    for p in prov
                    if p.get("source", "").startswith("lexical")
                ]
                return max(lexical_scores) if lexical_scores else 0.0

            # Tier sizes - adjust for SPECIFIC vs BROAD
            # Larger tiers ensure we get enough papers of each type (target: 6+ per category)
            if is_specific_query:
                tier1_size = 60   # Citations (foundational)
                tier2_size = 40   # Recent
                tier3_size = 50   # Lexical (query-matched)
            else:
                tier1_size = 120  # Citations (foundational)
                tier2_size = 60   # Recent (captures newer reviews/applications)
                tier3_size = 80   # Lexical (query-matched)

            # Tier 1: Top papers by citation count (foundational coverage)
            by_citations = sorted(
                work_ids,
                key=lambda wid: -(works_preload.get(wid).cited_by_count or 0) if works_preload.get(wid) else 0
            )
            tier1_papers = by_citations[:tier1_size]

            # Tier 2: Top recent papers by citations (temporal coverage)
            recent_papers = [
                wid for wid in work_ids
                if works_preload.get(wid) and works_preload.get(wid).year
                and works_preload.get(wid).year >= recent_cutoff_year
            ]
            recent_by_citations = sorted(
                recent_papers,
                key=lambda wid: -(works_preload.get(wid).cited_by_count or 0) if works_preload.get(wid) else 0
            )
            tier2_papers = recent_by_citations[:tier2_size]

            # Tier 3: Top papers by lexical match score (query-specific coverage)
            by_lexical = sorted(work_ids, key=lambda wid: -get_lexical_score(wid))
            tier3_papers = by_lexical[:tier3_size]

            # Union of all tiers (deduplicated)
            selected_set = set(tier1_papers) | set(tier2_papers) | set(tier3_papers)
            work_ids = list(selected_set)

            logger.info(
                f"Three-tier selection: {len(tier1_papers)} by-citations + "
                f"{len(tier2_papers)} recent + {len(tier3_papers)} lexical = "
                f"{len(work_ids)} unique (from {len(works_preload)} candidates)"
            )

            # Reuse preloaded works (filter to selected IDs only)
            works = {wid: works_preload[wid] for wid in work_ids if wid in works_preload}
            previews = WorkStore.load_previews_many(conn, work_ids)

            # Determine target topic if not provided
            target_topic_id = context_json.get("target_topic_id")
            if not target_topic_id:
                counts: Dict[str, int] = {}
                for wid in work_ids:
                    w = works.get(wid)
                    if not w:
                        continue
                    tid = _pick_best_topic_id(w)
                    if not tid:
                        continue
                    counts[tid] = counts.get(tid, 0) + 1
                if counts:
                    target_topic_id = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

            # Prepare raw feature dictionaries
            lex_raw: Dict[str, float] = {}
            llm_raw: Dict[str, float] = {}
            topic_raw: Dict[str, float] = {}
            impact_raw: Dict[str, float] = {}
            recency_raw: Dict[str, float] = {}
            completeness_raw: Dict[str, float] = {}

            def safe_float(val, default: float) -> float:
                try:
                    return float(val) if val is not None else default
                except (ValueError, TypeError):
                    return default

            half_life = float(rank_params_json.get("recency_half_life_years", 5.0))
            alpha = float(rank_params_json.get("impact_alpha", 1.0))
            beta = float(rank_params_json.get("impact_beta", 2.0))
            gamma = float(rank_params_json.get("impact_gamma", 0.0))
            base_rate_param = rank_params_json.get("impact_base_rate")
            # Convert None/negative/zero to None for the base_rate; otherwise cast to float
            base_rate = float(base_rate_param) if base_rate_param and float(base_rate_param) > 0.0 else None

            # Get query hash for LLM scoring cache
            # If we have query_expansion from earlier, use it; otherwise compute fresh
            from .query_expansion import compute_query_hash
            query_hash = query_expansion.query_hash if query_expansion else compute_query_hash(query_text)

            # LLM relevance scoring — wave-based with convergence detection
            # For lightweight operations (drill-down) with explicit cap, use old path
            # llm_scores now contains {"score": float, "paper_type": str} for each paper
            if llm_scoring_cap is not None:
                llm_scores = score_papers(
                    conn,
                    query_text=query_text,
                    query_hash=query_hash,
                    paper_ids=work_ids,
                    works=works,
                    llm_scoring_cap=llm_scoring_cap,
                )
                convergence_state = None
            else:
                # Score all selected candidates (three-tier selection already limits count)
                llm_scores = score_papers(
                    conn,
                    query_text=query_text,
                    query_hash=query_hash,
                    paper_ids=work_ids,
                    works=works,
                    llm_scoring_cap=len(work_ids),  # No additional cap
                )
                convergence_state = None

            logger.info(f"LLM scored {len(llm_scores)} papers")

            min_llm_relevance = safe_float(
                rank_params_json.get("min_llm_relevance"), 0.50
            )

            # Build feature values
            valid_work_ids = []
            llm_filter_stats = {"passed": 0, "filtered": 0}
            for wid in work_ids:
                w = works.get(wid)
                if not w:
                    continue
                # Hard drop retracted works (check both flag and title)
                if w.is_retracted:
                    continue
                # Also filter papers with "RETRACTED" in title (journal-marked retractions)
                if w.title and "RETRACTED" in w.title.upper():
                    continue
                y = w.year
                # Year filter (already applied in retrieval, but double-check)
                if year_min is not None and (y is None or int(y) < year_min):
                    continue
                if year_max is not None and (y is None or int(y) > year_max):
                    continue

                # Topic filter (strict, for "Research this topic" from node details)
                # Exclude papers that don't match the target topic
                if topic_id_filter:
                    if not w.primary_topic_id or w.primary_topic_id != topic_id_filter:
                        continue

                # LLM relevance from scoring (use as a hard gate)
                # Papers from foundational_works expansion get a lower threshold
                # because the query expansion LLM already validated their relevance
                # EXCEPTION: For SPECIFIC queries, use strict threshold for ALL papers
                # to ensure high topic precision
                llm_entry = llm_scores.get(wid, {"score": 0.0, "paper_type": "other"})
                llm_val = llm_entry.get("score", 0.0) if isinstance(llm_entry, dict) else llm_entry
                prov = provenance_map.get(wid, [])
                is_foundational = any(
                    entry.get("source") == "foundational_title" for entry in prov
                )
                # For specific queries: no lower threshold for foundational papers
                # All papers must meet the same precision bar
                if is_specific_query:
                    effective_threshold = min_llm_relevance
                else:
                    effective_threshold = 0.40 if is_foundational else min_llm_relevance
                if llm_val < effective_threshold:
                    llm_filter_stats["filtered"] = llm_filter_stats.get("filtered", 0) + 1
                    continue

                llm_filter_stats["passed"] = llm_filter_stats.get("passed", 0) + 1
                valid_work_ids.append(wid)

                # Lexical raw: use maximum lexical score from provenance
                prov = provenance_map.get(wid, [])
                lex_val = 0.0
                for entry in prov:
                    src = entry.get("source")
                    score = float(entry.get("score", 0.0))
                    if src in ("fts_lexical", "lexical_openalex") and score > lex_val:
                        lex_val = score
                lex_raw[wid] = lex_val

                llm_raw[wid] = llm_val

                # Topic relevance raw
                if target_topic_id:
                    topic_raw[wid] = compute_topic_relevance(w, str(target_topic_id))
                else:
                    topic_raw[wid] = 0.0

                # Impact: Bayesian citation rate
                age = compute_age(y)
                impact_raw[wid] = bayesian_impact_rate(
                    int(w.cited_by_count or 0),
                    age,
                    alpha,
                    beta,
                    uncertainty_gamma=gamma,
                    base_rate=base_rate,
                )

                # Recency: exponential decay
                recency_raw[wid] = compute_recency(age, half_life)

                # Completeness
                completeness_raw[wid] = compute_completeness(w)

            # Log LLM filter stats
            logger.info(f"LLM filter: {llm_filter_stats['passed']} passed, {llm_filter_stats['filtered']} filtered (threshold={min_llm_relevance})")

            # If no candidates remain, return empty categorized result
            if not valid_work_ids:
                logger.warning("No candidates remain after LLM filtering")
                with engine.begin() as tx:
                    RankRepo.mark_completed(tx, rank_job_id)
                return {
                    "rank_job_id": rank_job_id,
                    "job": {
                        "rank_job_id": rank_job_id,
                        "rank_type": "direct_prod",
                        "candidate_set_id": candidate_set_id,
                        "status": "completed",
                    },
                    "query_classification": {
                        "type": query_classification.query_type.value,
                        "confidence": query_classification.confidence,
                        "query_specificity": query_classification.query_specificity.value,
                    },
                    "foundational": [],
                    "methodology": [],
                    "reviews": [],
                    "applications": [],
                    "textbooks": [],
                }

            # Normalise features
            lex_norm = robust_norm(lex_raw)
            # LLM relevance: use raw scores directly (already on 0-1 scale: 0.25/0.50/0.75/0.95)
            # Percentile normalization destroys the meaningful scale
            llm_norm = llm_raw
            topic_norm = robust_norm(topic_raw)
            impact_norm = robust_norm(impact_raw)
            recency_norm = robust_norm(recency_raw)
            comp_norm = robust_norm(completeness_raw)

            # Note: Citation-floor boost removed (v73). The global LLM filter (min_llm_relevance=0.50)
            # already excludes papers with LLM < 0.50, so the boost condition (llm < 0.2) was
            # unreachable. The improved LLM prompt (v21) handles famous-paper scoring correctly.

            # Compose features dictionary for reranker
            # Simplified: LLM relevance is primary, supplemented by lexical, topic, impact, recency
            features_dict = {
                "llm_relevance": llm_norm,
                "lexical": lex_norm,
                "topic": topic_norm,
                "impact": impact_norm,
                "recency": recency_norm,
            }

            # Weights: LLM is dominant signal (0.55), others provide secondary ranking
            # Topic relevance weight zeroed: the LLM already judges topical relevance
            # far better than the OpenAlex topic-ID matching heuristic, which returns 0
            # for ~90% of papers. Its 5% weight is redistributed to LLM.
            weights_dict = {
                "llm_relevance": safe_float(rank_params_json.get("w_llm_relevance"), 0.55),
                "lexical": safe_float(rank_params_json.get("w_lexical"), 0.15),
                "topic": safe_float(rank_params_json.get("w_topic"), 0.0),
                "impact": safe_float(rank_params_json.get("w_impact"), 0.20),
                "recency": safe_float(rank_params_json.get("w_recency"), 0.10),
            }

            # Adjust weights for SPECIFIC queries: prioritize LLM relevance (precision)
            if is_specific_query:
                weights_dict["llm_relevance"] = 0.65
                weights_dict["impact"] = 0.10
                weights_dict["recency"] = 0.10
                weights_dict["lexical"] = 0.15
                weights_dict["topic"] = 0.0
                logger.info(
                    f"SPECIFIC QUERY weights: llm={weights_dict['llm_relevance']:.2f}, "
                    f"lexical={weights_dict['lexical']:.2f}, impact={weights_dict['impact']:.2f}"
                )

            # Build reasons for each paper
            reasons_map = build_reasons(
                rel_lex=lex_norm,
                rel_llm=llm_norm,
                rel_topic=topic_norm,
                impact=impact_norm,
                recency=recency_norm,
                completeness=comp_norm,
            )

            # Compute TF-IDF vectors for MMR diversification (replaces embeddings)
            tfidf_vectors = compute_tfidf_vectors(works, valid_work_ids)
            logger.info(f"Computed TF-IDF vectors for {len(tfidf_vectors)} papers")

            # Build diversity metadata for MMR
            # Years: extract publication year from works
            years_map = {wid: works[wid].year for wid in valid_work_ids if works.get(wid) and works[wid].year}

            # Subfields: map papers to subfield_id via topic hierarchy
            primary_topic_ids = [
                works[wid].primary_topic_id
                for wid in valid_work_ids
                if works.get(wid) and works[wid].primary_topic_id
            ]
            topic_hierarchy = TopicHierarchyStore.load_topic_to_subfield_and_field(conn, primary_topic_ids)
            subfields_map = {}
            for wid in valid_work_ids:
                w = works.get(wid)
                if w and w.primary_topic_id:
                    hier = topic_hierarchy.get(w.primary_topic_id)
                    if hier:
                        subfields_map[wid] = hier.get("subfield_id")

            logger.info(f"Diversity metadata: {len(years_map)} years, {len(subfields_map)} subfields")

            # Assemble ranked results with MMR diversification
            ranked_items = assemble_ranked_results(
                candidate_ids=valid_work_ids,
                features=features_dict,
                weights=weights_dict,
                k=top_k,
                tfidf_vectors=tfidf_vectors,
                reasons=reasons_map,
                previews=previews,
                provenance=provenance_map,
                years=years_map,
                subfields=subfields_map,
                impact_scores=impact_norm,
            )

            # Add raw LLM scores and paper_type to breakdown for:
            # 1. Specific query testing (verify topic precision via actual LLM judgment)
            # 2. Cache re-categorization (paper_type needed to re-categorize cached results)
            for item in ranked_items:
                wid = item["work_id"]
                if "breakdown" in item:
                    llm_entry = llm_scores.get(wid, {})
                    item["breakdown"]["raw"] = {
                        "llm_relevance": llm_raw.get(wid, 0.0),
                    }
                    item["breakdown"]["paper_type"] = (
                        llm_entry.get("paper_type", "other")
                        if isinstance(llm_entry, dict) else "other"
                    )

            # Partition results into 4 categories using LLM classifications
            categorized = partition_results_by_category(
                ranked_items,
                llm_scores,
                category_limits,
                query_text=query_text,
            )

            # Prepare result rows for persistence (all items together)
            result_rows: list[Dict[str, Any]] = []
            for idx, item in enumerate(ranked_items):
                # The "breakdown" from assemble_ranked_results contains "norm" and "weights"; keep as is
                result_rows.append(
                    {
                        "rank_job_id": rank_job_id,
                        "rank_index": idx,
                        "work_id": item["work_id"],
                        "score": item["score"],
                        "score_breakdown_json": item.get("breakdown", {}),
                        "reasons_json": item.get("reasons", []),
                        "work_preview_json": item.get("preview", {}),
                        "provenance_json": item.get("provenance", []),
                    }
                )

        # Persist results and mark completion
        with engine.begin() as tx:
            RankRepo.bulk_insert_results(tx, result_rows)
            RankRepo.mark_completed(tx, rank_job_id)

        # Build convergence info for response
        convergence_info = None
        if convergence_state is not None and convergence_state.waves:
            convergence_info = {
                "total_predicted": round(convergence_state.N, 1),
                "found": round(convergence_state.waves[-1].cumulative_relevance, 1),
                "completeness": round(convergence_state.completeness, 3),
                "confidence": round(convergence_state.confidence, 3),
                "tau": round(convergence_state.tau, 1),
                "k": round(convergence_state.k, 2),
                "papers_scored": convergence_state.waves[-1].papers_scored,
                "waves_completed": len(convergence_state.waves),
            }

        # Helper to format items for response
        def _format_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
            return {
                "rank_index": idx,
                "work_id": item["work_id"],
                "score": item["score"],
                "reasons": item.get("reasons", []),
                "score_breakdown": item.get("breakdown", {}),
                "preview": item.get("preview", {}),
                "provenance": item.get("provenance", []),
            }

        # Check if skip_categorization is set (for drill-down: flat list only)
        skip_categorization = rank_params_json.get("skip_categorization", False)

        if skip_categorization:
            # Return flat list of top papers (for drill-down endpoint)
            # Just return the top_k items without categorization
            return {
                "rank_job_id": rank_job_id,
                "job": {
                    "rank_job_id": rank_job_id,
                    "rank_type": "direct_prod",
                    "candidate_set_id": candidate_set_id,
                    "status": "completed",
                    "context_json": {"target_topic_id": target_topic_id},
                    "filters_json": filters_json,
                    "rank_params_json": rank_params_json,
                },
                "query_classification": {
                    "type": query_classification.query_type.value,
                    "confidence": query_classification.confidence,
                    "query_specificity": query_classification.query_specificity.value,
                },
                "convergence": convergence_info,
                "items": [
                    _format_item(item, idx) for idx, item in enumerate(ranked_items[:top_k])
                ],
            }

        # Assemble API response with categorized results
        # FOUR CATEGORIES: Foundational, Methodology, Reviews, Applications
        return {
            "rank_job_id": rank_job_id,
            "job": {
                "rank_job_id": rank_job_id,
                "rank_type": "direct_prod",
                "candidate_set_id": candidate_set_id,
                "status": "completed",
                "context_json": {"target_topic_id": target_topic_id},
                "filters_json": filters_json,
                "rank_params_json": rank_params_json,
            },
            "query_classification": {
                "type": query_classification.query_type.value,
                "confidence": query_classification.confidence,
                "query_specificity": query_classification.query_specificity.value,
            },
            "foundational": [
                _format_item(item, idx) for idx, item in enumerate(categorized["foundational"])
            ],
            "methodology": [
                _format_item(item, idx) for idx, item in enumerate(categorized["methodology"])
            ],
            "reviews": [
                _format_item(item, idx) for idx, item in enumerate(categorized["reviews"])
            ],
            "applications": [
                _format_item(item, idx) for idx, item in enumerate(categorized["applications"])
            ],
            "textbooks": [
                _format_item(item, idx) for idx, item in enumerate(categorized.get("textbooks", []))
            ],
            "convergence": convergence_info,
        }
    except Exception as e:
        # On failure, record the error and re‑raise
        with engine.begin() as tx:
            RankRepo.mark_failed(tx, rank_job_id, {"error": str(e)})
        raise
