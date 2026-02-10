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
) -> List[Dict[str, Any]]:
    """Combine features into final scores, diversify and produce ranking items.

    Parameters
    ----------
    candidate_ids : list of str
        Work IDs in the candidate pool.
    features : dict of feature_name -> dict work_id -> value
        Normalised feature values.
    weights : dict of feature_name -> float
        Weight for each feature; missing keys default to 0.
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

    Returns
    -------
    list of dict
        List of ranked result items including breakdowns and reasons.
    """
    from .tfidf_similarity import mmr_diversify_tfidf, dedupe_by_title_similarity

    # Compute raw combined score (weighted sum of features)
    combined: Dict[str, float] = {}
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
    ordered = dedupe_by_title_similarity(ordered, titles, threshold=0.5)
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
