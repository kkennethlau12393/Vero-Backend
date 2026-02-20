"""
Feature 2 reranking utilities for production‑grade ranking.

This module assembles candidate works into a final ranked list by
combining multiple normalised signals (LLM relevance, lexical, topic, impact,
recency, completeness) with configurable weights, then applying
deterministic diversification (Max‑Marginal‑Relevance) to reduce
duplicate results.  It also constructs user‑facing reasons and a
breakdown of contributions for transparency.

The MMR algorithm selects documents iteratively, balancing the overall
score with similarity to already selected items. Similarity is computed
using TF-IDF vectors on paper abstracts (replacing the previous
embedding-based approach).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import math
from collections import defaultdict


def reciprocal_rank_fusion(
    *rank_lists: Dict[str, float],
    k: int = 60,
) -> Dict[str, float]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF(paper) = Σ 1/(k + rank_i(paper)) for each ranker i.

    Papers must be consistently good across all signals to rank high.
    If one ranker ranks a paper #1 but others rank it #50+, RRF demotes it.
    No single ranker can dominate — natural protection against LLM errors.

    Parameters
    ----------
    rank_lists : variable number of Dict[str, float]
        Each dict maps paper_id to a raw score from one ranker.
        Scores are only used for ordering (converted to ranks internally).
    k : int
        Smoothing constant (default 60, standard in literature).
        Higher k reduces the influence of top-ranked items.

    Returns
    -------
    Dict[str, float]
        Combined RRF scores for all papers across all rankers.
    """
    if not rank_lists:
        return {}

    all_paper_ids: set = set()
    for scores in rank_lists:
        all_paper_ids.update(scores.keys())

    if not all_paper_ids:
        return {}

    num_rankers = len(rank_lists)
    total_papers = len(all_paper_ids)

    # For each ranker, sort papers by score descending and assign ranks
    ranker_ranks: List[Dict[str, int]] = []
    for scores in rank_lists:
        sorted_papers = sorted(scores.keys(), key=lambda p: -scores.get(p, 0.0))
        ranks = {pid: rank for rank, pid in enumerate(sorted_papers, start=1)}
        ranker_ranks.append(ranks)

    # Compute RRF score for each paper
    rrf_scores: Dict[str, float] = {}
    for pid in all_paper_ids:
        rrf = 0.0
        for ranks in ranker_ranks:
            # Papers not in a ranker's list get worst rank
            rank = ranks.get(pid, total_papers + 1)
            rrf += 1.0 / (k + rank)
        rrf_scores[pid] = rrf

    return rrf_scores


def mmr_diversify(
    scored: List[Tuple[str, float]],
    embeddings: Dict[str, List[float]],
    k: int,
    lambda_param: float = 0.8,
) -> List[str]:
    """Apply Max‑Marginal‑Relevance (MMR) to diversify the top results.

    Parameters
    ----------
    scored : list of (work_id, score)
        Candidates sorted by descending score.
    embeddings : dict mapping work_id to embedding vector
        Embeddings used to compute similarity between candidates.
    k : int
        Desired number of results after diversification.
    lambda_param : float, optional
        Balance parameter between relevance and diversity.  1.0 means
        pure relevance, 0.0 means pure diversity.

    Returns
    -------
    list of work_id
        Diversified ranking of length <= k.
    """
    if not scored:
        return []
    k = min(k, len(scored))
    # Normalise embeddings to unit length for cosine similarity
    def coerce_embedding(vec: Any) -> Optional[List[float]]:
        if vec is None:
            return None
        if isinstance(vec, list):
            try:
                return [float(x) for x in vec]
            except Exception:
                return None
        if isinstance(vec, str):
            cleaned = vec.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            if not cleaned:
                return None
            try:
                return [float(x) for x in cleaned.split(",") if x]
            except Exception:
                return None
        try:
            return [float(x) for x in list(vec)]
        except Exception:
            return None

    def norm(vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec)) or 1.0
    normalised = {}
    for wid, emb in embeddings.items():
        emb_list = coerce_embedding(emb)
        if not emb_list:
            continue
        normalised[wid] = [x / norm(emb_list) for x in emb_list]
    # Precompute pairwise cosine similarity function
    def cos_sim(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
    # Selected set
    selected: List[str] = []
    # Set of candidate IDs left to choose from
    remaining = [wid for (wid, _score) in scored]
    # Iteratively select k items
    while remaining and len(selected) < k:
        mmr_best = None
        mmr_val_best = float("-inf")
        for wid in remaining:
            relevance = next((s for (w, s) in scored if w == wid), 0.0)
            max_sim = 0.0
            if selected and wid in normalised:
                vec = normalised.get(wid)
                # Compute similarity to the closest selected item
                for sel in selected:
                    if sel in normalised:
                        sim = cos_sim(vec, normalised[sel])
                        if sim > max_sim:
                            max_sim = sim
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > mmr_val_best:
                mmr_val_best = mmr_score
                mmr_best = wid
        if mmr_best is None:
            break
        selected.append(mmr_best)
        remaining.remove(mmr_best)
    return selected


def build_reasons(
    rel_lex: Dict[str, float],
    rel_llm: Dict[str, float],
    rel_topic: Dict[str, float],
    impact: Dict[str, float],
    recency: Dict[str, float],
    completeness: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """Generate short reason strings for each work based on feature thresholds.

    The `thresholds` dict can define custom cutoffs for when to trigger
    particular reasons.  Defaults are used if thresholds are absent.
    Returns a dict mapping work_id to a list of human‑readable reasons.
    """
    thresholds = thresholds or {}
    t_lex = thresholds.get("lex", 0.5)
    t_llm = thresholds.get("llm", 0.5)
    t_topic = thresholds.get("topic", 0.5)
    t_imp = thresholds.get("impact", 0.75)
    t_rec = thresholds.get("recency", 0.75)
    t_comp = thresholds.get("completeness", 0.75)

    reasons: Dict[str, List[str]] = defaultdict(list)
    for wid in set(list(rel_lex.keys()) + list(rel_llm.keys()) + list(rel_topic.keys())):
        if rel_topic.get(wid, 0.0) >= t_topic:
            reasons[wid].append("Strong topic match")
        elif 0.25 <= rel_topic.get(wid, 0.0) < t_topic:
            reasons[wid].append("Topic match")
        if rel_llm.get(wid, 0.0) >= t_llm:
            reasons[wid].append("Highly relevant")
        elif 0.25 <= rel_llm.get(wid, 0.0) < t_llm:
            reasons[wid].append("Relevant")
        if rel_lex.get(wid, 0.0) >= t_lex:
            reasons[wid].append("Lexical match")
        if impact.get(wid, 0.0) >= t_imp:
            reasons[wid].append("High impact for its age")
        if recency.get(wid, 0.0) >= t_rec:
            reasons[wid].append("Recent work")
        if completeness.get(wid, 0.0) >= t_comp:
            reasons[wid].append("Complete metadata")
    return reasons


# Mapping from paper category to human-readable reason string
CATEGORY_REASONS = {
    "foundational": "Foundational work",
    "methodological": "Methodology paper",
    "applied": "Applied study",
    "recent": "Recent development",
    "implementation": "Implementation resource",
    "handbook": "Review/Handbook",
}


def build_reasons_with_categories(
    rel_lex: Dict[str, float],
    rel_llm: Dict[str, float],
    rel_topic: Dict[str, float],
    impact: Dict[str, float],
    recency: Dict[str, float],
    completeness: Dict[str, float],
    method_align: Dict[str, float],
    paper_classifications: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """Generate reason strings including paper category information.

    Extends build_reasons() with category-based reasons and methodological
    alignment information.

    Parameters
    ----------
    paper_classifications : dict
        Mapping from work_id to PaperClassification objects.
    method_align : dict
        Mapping from work_id to methodological alignment score.
    """
    # Start with standard reasons
    reasons = build_reasons(
        rel_lex=rel_lex,
        rel_llm=rel_llm,
        rel_topic=rel_topic,
        impact=impact,
        recency=recency,
        completeness=completeness,
        thresholds=thresholds,
    )

    thresholds = thresholds or {}
    t_method = thresholds.get("method_align", 0.7)

    # Add category-based reasons (prepend to front)
    for wid, paper_class in paper_classifications.items():
        category_value = paper_class.category.value if hasattr(paper_class.category, 'value') else str(paper_class.category)
        category_reason = CATEGORY_REASONS.get(category_value)

        if category_reason:
            # Prepend category reason to the front
            if wid not in reasons:
                reasons[wid] = []
            if category_reason not in reasons[wid]:
                reasons[wid].insert(0, category_reason)

        # Add methodological alignment reason if score is high
        if method_align.get(wid, 0.0) >= t_method:
            if "Strong methodological fit" not in reasons[wid]:
                reasons[wid].append("Strong methodological fit")

    return reasons


def assemble_ranked_results(
    candidate_ids: List[str],
    features: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    k: int,
    tfidf_vectors: Dict[str, Dict[str, float]],
    reasons: Dict[str, List[str]],
    previews: Dict[str, Any],
    provenance: Dict[str, Any],
    years: Optional[Dict[str, int]] = None,
    subfields: Optional[Dict[str, str]] = None,
    impact_scores: Optional[Dict[str, float]] = None,
    precomputed_scores: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Combine features into final scores, diversify and produce ranking items.

    Parameters
    ----------
    candidate_ids : list of str
        Work IDs in the candidate pool.
    features : dict of feature_name -> dict work_id -> value
        Normalised feature values (used for breakdown display).
    weights : dict of feature_name -> float
        Weight for each feature; missing keys default to 0.
        Ignored when precomputed_scores is provided.
    k : int
        Number of results to return.
    tfidf_vectors : dict work_id -> TF-IDF vector (sparse dict)
        TF-IDF vectors used for diversification (replaces embeddings).
    reasons : dict work_id -> list of str
        Precomputed reason strings.
    previews : dict work_id -> WorkPreview
        Loaded previews for rendering.
    provenance : dict work_id -> list of provenance entries
        Source/provenance for each candidate.
    years : dict work_id -> publication year, optional
        Used for soft temporal diversity in MMR.
    subfields : dict work_id -> subfield_id, optional
        Used for paradigm diversity in MMR (spread across sub-topics).
    impact_scores : dict work_id -> normalized impact score, optional
        High-impact papers get reduced diversity penalties.
    precomputed_scores : dict work_id -> float, optional
        Pre-computed combined scores (e.g. from RRF). If provided, skips
        the weighted sum computation and uses these directly.

    Returns
    -------
    list of dict
        List of ranked result items including breakdowns and reasons.
    """
    from .tfidf_similarity import mmr_diversify_tfidf, dedupe_by_title_similarity

    # Use precomputed scores (RRF) or fall back to weighted sum
    combined: Dict[str, float] = {}
    if precomputed_scores is not None:
        for wid in candidate_ids:
            combined[wid] = precomputed_scores.get(wid, 0.0)
    else:
        for wid in candidate_ids:
            score = 0.0
            for fname, fvals in features.items():
                weight = weights.get(fname, 0.0)
                score += weight * fvals.get(wid, 0.0)
            combined[wid] = score
    # Sort by combined score then by work_id for determinism
    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    # Dedupe similar titles before MMR (keeps highest-scored per title cluster)
    titles = {wid: p.title for wid, p in previews.items() if p and p.title}
    ordered = dedupe_by_title_similarity(ordered, titles, threshold=0.4)
    # Apply diversification using TF-IDF similarity with temporal and paradigm diversity
    diversified_ids = mmr_diversify_tfidf(
        ordered,
        tfidf_vectors,
        k,
        years=years,
        subfields=subfields,
        impact_scores=impact_scores,
    )
    results: List[Dict[str, Any]] = []
    for rank_idx, wid in enumerate(diversified_ids):
        item = {
            "work_id": wid,
            "score": float(combined.get(wid, 0.0)),
            "reasons": reasons.get(wid, []),
            "provenance": provenance.get(wid, []),
        }
        # Build score breakdown for transparency
        breakdown_raw = {fname: features[fname].get(wid, 0.0) for fname in features.keys()}
        item["breakdown"] = {
            "norm": breakdown_raw,
            "weights": weights,
        }
        # Attach preview if available
        p = previews.get(wid)
        if p:
            item["preview"] = {
                "work_id": wid,
                "title": p.title,
                "year": p.year,
                "cited_by_count": int(p.cited_by_count or 0),
                "authors": p.authors,
                "venue": p.venue,
            }
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# User-facing scoring rubric
# ---------------------------------------------------------------------------

# Stars mapping: 0-10 score → 1-5 stars
def _score_to_stars(score: float) -> int:
    if score >= 8.0:
        return 5
    if score >= 6.0:
        return 4
    if score >= 4.0:
        return 3
    if score >= 2.0:
        return 2
    return 1


# Labels for relevance/influence/textual_match dimensions
_STANDARD_LABELS = {5: "Excellent", 4: "Strong", 3: "Moderate", 2: "Low", 1: "Minimal"}
# Labels for recency dimension (context-aware: low ≠ bad, it means "classic")
_RECENCY_LABELS = {5: "Very Recent", 4: "Recent", 3: "Moderate", 2: "Established", 1: "Classic"}
# Labels for overall
_OVERALL_LABELS = {5: "Excellent", 4: "Strong", 3: "Good", 2: "Fair", 1: "Low"}


def _make_dimension(score_10: float, labels: Dict[str, str]) -> Dict[str, Any]:
    """Build a single scoring dimension dict."""
    stars = _score_to_stars(score_10)
    return {
        "score": round(score_10, 1),
        "stars": stars,
        "label": labels[stars],
    }


def build_user_scoring(
    llm_norm: Dict[str, float],
    impact_norm: Dict[str, float],
    lex_norm: Dict[str, float],
    recency_norm: Dict[str, float],
    combined_scores: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Convert normalized scores (0-1) to user-facing 0-10 rubric with stars and labels.

    Pure computation — no API calls. Maps existing pipeline values to a
    user-friendly quantitative rubric.

    Parameters
    ----------
    llm_norm : dict work_id → float (0-1)
        LLM relevance score (normalized).
    impact_norm : dict work_id → float (0-1)
        Impact/influence score (normalized).
    lex_norm : dict work_id → float (0-1)
        Lexical/textual match score (normalized).
    recency_norm : dict work_id → float (0-1)
        Recency score (normalized).
    combined_scores : dict work_id → float (0-1)
        Final weighted combined score.

    Returns
    -------
    dict work_id → scoring dict with "overall" and "dimensions"
    """
    result: Dict[str, Dict[str, Any]] = {}

    all_ids = set(combined_scores.keys())

    for wid in all_ids:
        relevance_10 = llm_norm.get(wid, 0.0) * 10.0
        influence_10 = impact_norm.get(wid, 0.0) * 10.0
        textual_10 = lex_norm.get(wid, 0.0) * 10.0
        recency_10 = recency_norm.get(wid, 0.0) * 10.0
        overall_10 = combined_scores.get(wid, 0.0) * 10.0

        result[wid] = {
            "overall": _make_dimension(overall_10, _OVERALL_LABELS),
            "dimensions": {
                "relevance": _make_dimension(relevance_10, _STANDARD_LABELS),
                "influence": _make_dimension(influence_10, _STANDARD_LABELS),
                "textual_match": _make_dimension(textual_10, _STANDARD_LABELS),
                "recency": _make_dimension(recency_10, _RECENCY_LABELS),
            },
        }

    return result
