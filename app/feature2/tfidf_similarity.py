"""
TF-IDF similarity module for MMR diversification.

This module computes TF-IDF vectors for paper abstracts and provides
cosine similarity functions for the MMR algorithm, replacing embedding-based
similarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from .work_topic_store import WorkForMap


# Stopwords for title deduplication (smaller set focused on academic filler)
TITLE_STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'or', 'with',
    'using', 'based', 'via', 'towards', 'from', 'by'
}


def title_word_set(title: str) -> set:
    """Extract content words from title as a set for deduplication."""
    if not title:
        return set()
    cleaned = re.sub(r'[^\w\s]', '', title.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in TITLE_STOPWORDS}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |intersection| / |union|."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def dedupe_by_title_similarity(
    scored: List[Tuple[str, float]],
    titles: Dict[str, str],
    threshold: float = 0.4,
) -> List[Tuple[str, float]]:
    """Remove papers with highly similar titles, keeping highest-scored.

    Parameters
    ----------
    scored : list of (work_id, score)
        Candidates sorted by descending score.
    titles : dict mapping work_id to title string
        Title text for each candidate.
    threshold : float
        Jaccard similarity threshold. Papers with similarity >= threshold
        to any already-selected paper are considered duplicates.
        Default 0.4 catches reformatted titles (e.g., IPCC AR4 variants).

    Returns
    -------
    list of (work_id, score)
        Deduplicated list preserving score order.
    """
    selected: List[Tuple[str, float, set]] = []

    for wid, score in scored:
        title = titles.get(wid, "")
        words = title_word_set(title)

        # Empty title → keep (can't compare)
        if not words:
            selected.append((wid, score, words))
            continue

        # Check similarity to all selected
        is_duplicate = False
        for _, _, existing_words in selected:
            if jaccard_similarity(words, existing_words) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append((wid, score, words))

    return [(wid, score) for wid, score, _ in selected]


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization: lowercase, remove punctuation, split on whitespace.

    Also removes common stop words to improve TF-IDF quality.
    """
    if not text:
        return []

    # Common English stop words
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'as', 'if', 'then', 'than', 'so',
        'such', 'no', 'not', 'only', 'own', 'same', 'into', 'over', 'under',
        'again', 'further', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'any',
        'both', 'during', 'before', 'after', 'above', 'below', 'between',
        'through', 'about', 'against', 'up', 'down', 'out', 'off', 'which',
        'who', 'whom', 'what', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
        'their', 'i', 'me', 'my', 'his', 'her', 'him', 'them', 'us',
    }

    # Remove punctuation and lowercase
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())

    # Split and filter
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return tokens


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for a document.

    Uses raw frequency normalized by document length.
    """
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)

    return {term: count / total for term, count in counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency across corpus.

    IDF(t) = log(N / df(t)) where df(t) is the number of documents containing term t.
    """
    if not documents:
        return {}

    n_docs = len(documents)
    doc_freq: Dict[str, int] = Counter()

    for doc_tokens in documents:
        unique_terms = set(doc_tokens)
        for term in unique_terms:
            doc_freq[term] += 1

    idf = {}
    for term, df in doc_freq.items():
        # Add 1 to avoid division by zero and smooth IDF
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1

    return idf


def compute_tfidf_vectors(
    works: Dict[str, WorkForMap],
    paper_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute TF-IDF vectors for all papers.

    Args:
        works: Dict mapping work_id to WorkForMap
        paper_ids: List of paper IDs to compute vectors for

    Returns:
        Dict mapping work_id to sparse TF-IDF vector (term -> weight)
    """
    if not paper_ids:
        return {}

    # Tokenize all documents
    tokenized_docs: Dict[str, List[str]] = {}
    all_docs: List[List[str]] = []

    for pid in paper_ids:
        w = works.get(pid)
        if not w:
            tokenized_docs[pid] = []
            all_docs.append([])
            continue

        # Combine title and abstract for better similarity
        title = w.title or ""
        abstract = getattr(w, 'abstract', None) or ""
        text = f"{title} {abstract}"

        tokens = tokenize_text(text)
        tokenized_docs[pid] = tokens
        all_docs.append(tokens)

    # Compute IDF across corpus
    idf = compute_idf(all_docs)

    # Compute TF-IDF for each document
    tfidf_vectors: Dict[str, Dict[str, float]] = {}

    for pid in paper_ids:
        tokens = tokenized_docs.get(pid, [])
        tf = compute_tf(tokens)

        # TF-IDF = TF * IDF
        tfidf = {}
        for term, tf_val in tf.items():
            idf_val = idf.get(term, 1.0)
            tfidf[term] = tf_val * idf_val

        tfidf_vectors[pid] = tfidf

    return tfidf_vectors


def cosine_similarity_tfidf(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float],
) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors.

    Returns a value in [0, 1] where 1 means identical.
    """
    if not vec_a or not vec_b:
        return 0.0

    # Find common terms
    common_terms = set(vec_a.keys()) & set(vec_b.keys())

    if not common_terms:
        return 0.0

    # Compute dot product
    dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)

    # Compute magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def compute_tfidf_query_similarity(
    query_text: str,
    works: Dict[str, WorkForMap],
    paper_ids: List[str],
) -> Dict[str, float]:
    """Compute TF-IDF cosine similarity between query and each paper.

    Embeds the query alongside all papers in the same TF-IDF space, then
    computes cosine similarity between the query vector and each paper vector.
    ~200ms for 1000 papers, no API calls.

    Returns raw cosine similarity scores. Caller uses ranks for RRF.
    """
    if not paper_ids or not query_text:
        return {}

    # Tokenize all documents + query
    all_docs: List[List[str]] = []
    tokenized_docs: Dict[str, List[str]] = {}

    # Query goes first (index 0)
    query_tokens = tokenize_text(query_text)
    all_docs.append(query_tokens)

    for pid in paper_ids:
        w = works.get(pid)
        if not w:
            tokenized_docs[pid] = []
            all_docs.append([])
            continue
        title = w.title or ""
        abstract = getattr(w, 'abstract', None) or ""
        text = f"{title} {abstract}"
        tokens = tokenize_text(text)
        tokenized_docs[pid] = tokens
        all_docs.append(tokens)

    # Compute IDF across entire corpus (query + papers)
    idf = compute_idf(all_docs)

    # Build query TF-IDF vector
    query_tf = compute_tf(query_tokens)
    query_vec = {term: tf_val * idf.get(term, 1.0) for term, tf_val in query_tf.items()}

    # Compute similarity for each paper
    result: Dict[str, float] = {}
    for pid in paper_ids:
        tokens = tokenized_docs.get(pid, [])
        tf = compute_tf(tokens)
        paper_vec = {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}
        result[pid] = cosine_similarity_tfidf(query_vec, paper_vec)

    return result


def mmr_diversify_tfidf(
    scored: List[Tuple[str, float]],
    tfidf_vectors: Dict[str, Dict[str, float]],
    k: int,
    lambda_param: float = 0.8,
    years: Dict[str, int] = None,
    subfields: Dict[str, str] = None,
    impact_scores: Dict[str, float] = None,
    temporal_penalty: float = 0.08,
    paradigm_penalty: float = 0.12,
    era_window: int = 5,
) -> List[str]:
    """Apply Max-Marginal-Relevance (MMR) using TF-IDF similarity.

    Enhanced with soft temporal and paradigm diversity penalties.

    Parameters
    ----------
    scored : list of (work_id, score)
        Candidates sorted by descending score.
    tfidf_vectors : dict mapping work_id to TF-IDF vector
        Sparse TF-IDF vectors for similarity computation.
    k : int
        Desired number of results after diversification.
    lambda_param : float, optional
        Balance parameter between relevance and diversity.
        1.0 = pure relevance, 0.0 = pure diversity.
    years : dict mapping work_id to publication year, optional
        Used for soft temporal diversity (avoid era clustering).
    subfields : dict mapping work_id to subfield_id, optional
        Used for paradigm diversity (spread across sub-topics).
    impact_scores : dict mapping work_id to normalized impact score, optional
        High-impact papers get reduced temporal/paradigm penalties.
    temporal_penalty : float, optional
        Penalty multiplier for same-era papers (within era_window years).
        Applied softly - reduced for high-impact papers.
    paradigm_penalty : float, optional
        Penalty multiplier for same-subfield papers.
        Applied softly - reduced for high-impact papers.
    era_window : int, optional
        Years defining an "era" for temporal diversity (default 5).

    Returns
    -------
    list of work_id
        Diversified ranking of length <= k.
    """
    if not scored:
        return []

    k = min(k, len(scored))
    years = years or {}
    subfields = subfields or {}
    impact_scores = impact_scores or {}

    # Build score lookup
    score_map = {wid: score for wid, score in scored}

    # Selected set
    selected: List[str] = []

    # Track selected years and subfields for diversity
    selected_years: List[int] = []
    selected_subfields: List[str] = []

    # Set of candidate IDs left to choose from
    remaining = [wid for (wid, _) in scored]

    # Iteratively select k items
    while remaining and len(selected) < k:
        best_id = None
        best_mmr = float("-inf")

        for wid in remaining:
            relevance = score_map.get(wid, 0.0)

            # Compute max TF-IDF similarity to already selected items
            max_sim = 0.0
            if selected:
                vec_wid = tfidf_vectors.get(wid, {})
                for sel_id in selected:
                    vec_sel = tfidf_vectors.get(sel_id, {})
                    sim = cosine_similarity_tfidf(vec_wid, vec_sel)
                    if sim > max_sim:
                        max_sim = sim

            # Soft temporal diversity penalty
            # Penalize if this paper is within era_window years of already-selected papers
            # Penalty is reduced for high-impact papers (foundational work shouldn't be penalized)
            temporal_sim = 0.0
            if years and selected_years:
                paper_year = years.get(wid)
                if paper_year:
                    # Count how many selected papers are in the same era window
                    same_era_count = sum(
                        1 for sy in selected_years if abs(sy - paper_year) <= era_window
                    )
                    if same_era_count > 0:
                        # Normalize by number of selected papers
                        era_ratio = same_era_count / len(selected_years)
                        # Reduce penalty for high-impact papers (but less reduction now: 50%)
                        impact = impact_scores.get(wid, 0.5)
                        impact_discount = 1.0 - (impact * 0.5)  # High impact = 50% reduction
                        temporal_sim = era_ratio * impact_discount

            # Soft paradigm diversity penalty
            # Penalize if this paper shares subfield with already-selected papers
            # Penalty is reduced for high-impact papers within their paradigm
            paradigm_sim = 0.0
            if subfields and selected_subfields:
                paper_subfield = subfields.get(wid)
                if paper_subfield:
                    # Count how many selected papers share this subfield
                    same_subfield_count = sum(
                        1 for sf in selected_subfields if sf == paper_subfield
                    )
                    if same_subfield_count > 0:
                        # Normalize by number of selected papers
                        subfield_ratio = same_subfield_count / len(selected_subfields)
                        # Reduce penalty for high-impact papers (50% reduction)
                        impact = impact_scores.get(wid, 0.5)
                        impact_discount = 1.0 - (impact * 0.5)  # High impact = 50% reduction
                        paradigm_sim = subfield_ratio * impact_discount

            # Combined MMR score with temporal and paradigm penalties
            mmr_score = (
                lambda_param * relevance
                - (1.0 - lambda_param) * max_sim
                - temporal_penalty * temporal_sim
                - paradigm_penalty * paradigm_sim
            )

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = wid

        if best_id is None:
            break

        selected.append(best_id)
        remaining.remove(best_id)

        # Track diversity metadata for next iteration
        if years.get(best_id):
            selected_years.append(years[best_id])
        if subfields.get(best_id):
            selected_subfields.append(subfields[best_id])

    return selected
