"""
Intersection scoring for structured query candidates.

Scores candidate papers based on how well they match the structured
query components (topic, domain, aspect) using text-based matching.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def term_match_score(terms: str, text: str, aliases: Optional[List[str]] = None) -> float:
    """Score how well search terms appear in text. Returns 0.0-1.0.

    When aliases are provided, checks all forms (original + aliases)
    and returns the best match. This handles abbreviation mismatches
    like "NLP" vs "natural language processing".
    """
    if not terms or not text:
        return 0.0

    text_normalized = normalize_text(text)

    # Build all forms to check: original + aliases
    all_forms = [normalize_text(terms)]
    if aliases:
        all_forms.extend(normalize_text(a) for a in aliases if a)

    best_score = 0.0
    for form in all_forms:
        # Exact phrase match
        if form in text_normalized:
            return 1.0

        # Word-level matching
        form_words = set(form.split())
        text_words = set(text_normalized.split())

        if not form_words:
            continue

        matched = form_words & text_words
        score = len(matched) / len(form_words)
        best_score = max(best_score, score)

    return best_score


def score_candidate(paper: Dict[str, Any], structured: Dict[str, Optional[str]], scope: str) -> float:
    """Score a candidate paper against the structured query.

    Parameters
    ----------
    paper : dict
        Paper with at least 'title' and optionally 'abstract'.
    structured : dict
        Decomposed query with keys: topic, domain, aspect.
    scope : str
        One of: intersection, topic_focused, domain_focused, broad.

    Returns
    -------
    float
        Relevance score between 0.0 and 1.0.
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract}"

    topic = structured.get("topic", "")
    domain = structured.get("domain")
    aspect = structured.get("aspect")

    # Extract aliases from decomposition (empty lists if not present)
    topic_aliases = structured.get("topic_aliases", [])
    domain_aliases = structured.get("domain_aliases", [])
    aspect_aliases = structured.get("aspect_aliases", [])

    topic_score = term_match_score(topic, text, aliases=topic_aliases)
    domain_score = term_match_score(domain, text, aliases=domain_aliases) if domain else 1.0
    aspect_score = term_match_score(aspect, text, aliases=aspect_aliases) if aspect else 1.0

    if scope == "intersection":
        # Must match BOTH topic and domain — multiplicative
        return topic_score * domain_score * (0.5 + 0.5 * aspect_score)

    elif scope == "topic_focused":
        return topic_score * 0.8 + domain_score * 0.2

    elif scope == "domain_focused":
        return topic_score * 0.2 + domain_score * 0.8

    else:  # broad
        return (topic_score + domain_score) / 2


def filter_candidates(
    papers: List[Dict[str, Any]],
    structured: Dict[str, Optional[str]],
    scope: str,
    min_score: float = 0.2,
) -> List[Dict[str, Any]]:
    """Filter and sort candidates by relevance score.

    Parameters
    ----------
    papers : list[dict]
        Candidate papers.
    structured : dict
        Decomposed query.
    scope : str
        Scope mode.
    min_score : float
        Minimum relevance score for inclusion.

    Returns
    -------
    list[dict]
        Filtered and sorted papers with '_relevance_score' added.
    """
    scored = []
    for paper in papers:
        score = score_candidate(paper, structured, scope)
        if score >= min_score:
            paper["_relevance_score"] = score
            scored.append(paper)

    scored.sort(key=lambda p: p["_relevance_score"], reverse=True)
    return scored
