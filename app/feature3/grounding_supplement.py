"""
Grounding paper supplement for Feature 3.

This module ensures novelty assessments have sufficient grounding papers
of BOTH types: reference-type (methodological) and landmark-type (field context).

Target distribution:
- At least 2-3 reference-type papers (direct methodology comparison)
- At least 2-3 landmark-type papers (historical field context)
- Total: 5-7 grounding papers
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Use same model as feature2
MODEL_VERSION = "gpt-4o-mini"

# Grounding requirements
MIN_REFS = 2          # Minimum reference-type papers
MIN_LANDMARKS = 3     # Minimum landmark-type papers (3 for robust field context)
MIN_TOTAL = 5         # Minimum total grounding papers
MAX_TOTAL = 7         # Maximum total grounding papers

# API settings
OPENALEX_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5


# ============================================================================
# LLM Prompts for Different Paper Types
# ============================================================================

SUGGEST_REFS_PROMPT = """You are an expert academic researcher. Suggest papers that are METHODOLOGICALLY SIMILAR to the target paper - papers that use similar techniques, approaches, or solve similar problems.

Target Paper:
Title: {title}
Abstract: {abstract}
Year: {year}

Papers I already have:
{existing_papers}

Suggest {num_needed} papers that:
1. Use SIMILAR METHODS or techniques as the target paper
2. Were published BEFORE {year}
3. Are directly comparable in methodology
4. Are NOT already in my list

Return a JSON array:
[
    {{
        "title": "Exact paper title",
        "authors": "First Author et al.",
        "year": 2015,
        "why_relevant": "Uses similar [specific technique] for [specific task]"
    }}
]

Focus on methodological similarity, not just topic similarity."""


SUGGEST_LANDMARKS_PROMPT = """You are an expert academic researcher. Suggest SEMINAL/FOUNDATIONAL papers in the field - highly influential papers that established key concepts the target paper builds upon.

Target Paper:
Title: {title}
Abstract: {abstract}
Year: {year}
Field/Topic: {field}

Papers I already have:
{existing_papers}

Suggest {num_needed} LANDMARK papers that:
1. Are highly cited (1000+ citations typically)
2. Introduced foundational concepts/techniques used in this field
3. Were published BEFORE {year}
4. Are NOT already in my list

Return a JSON array:
[
    {{
        "title": "Exact paper title",
        "authors": "First Author et al.",
        "year": 2012,
        "why_relevant": "Introduced [foundational concept] that this work builds upon"
    }}
]

Focus on SEMINAL papers - the classics everyone in this field cites."""


# ============================================================================
# Core Functions
# ============================================================================

def assess_grounding_needs(
    refs: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
) -> Tuple[int, int, str]:
    """
    Assess what additional grounding papers are needed.

    ALWAYS requires at least MIN_LANDMARKS (2) landmarks for proper field context,
    regardless of total count.

    Returns:
        (refs_needed, landmarks_needed, reason)
    """
    num_refs = len(refs)
    num_landmarks = len(landmarks)
    total = num_refs + num_landmarks

    if total == 0:
        return 0, 0, "no_grounding_unavailable"

    # ALWAYS require minimum landmarks for field context (even if total is high)
    refs_needed = max(0, MIN_REFS - num_refs)
    landmarks_needed = max(0, MIN_LANDMARKS - num_landmarks)

    # If we have enough of each type but total is low, add more refs (not landmarks)
    # since landmarks are the historical context and refs are for methodology
    if refs_needed == 0 and landmarks_needed == 0 and total < MIN_TOTAL:
        refs_needed = MIN_TOTAL - total

    reason = "sufficient"
    if refs_needed > 0 or landmarks_needed > 0:
        parts = []
        if refs_needed > 0:
            parts.append(f"need {refs_needed} more refs")
        if landmarks_needed > 0:
            parts.append(f"need {landmarks_needed} more landmarks")
        reason = ", ".join(parts)

    return refs_needed, landmarks_needed, reason


def suggest_papers_llm(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    existing_papers: List[Dict[str, Any]],
    num_needed: int,
    paper_type: str,  # "reference" or "landmark"
    field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use LLM to suggest papers of a specific type.

    Args:
        paper_type: "reference" for methodologically similar, "landmark" for seminal papers
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, skipping paper suggestion")
        return []

    if num_needed <= 0:
        return []

    # Format existing papers
    existing_str = "\n".join([
        f"- {p.get('title', 'Untitled')} ({p.get('year', '?')})"
        for p in existing_papers[:10]
    ]) if existing_papers else "(none)"

    # Request MORE papers than needed to compensate for lookup failures
    # Request 2x what we need, capped at 6
    request_count = min(num_needed * 2, 6)

    # Select prompt based on paper type
    if paper_type == "landmark":
        prompt = SUGGEST_LANDMARKS_PROMPT.format(
            title=title,
            abstract=(abstract or "No abstract available.")[:1500],
            year=year or 2024,
            field=field or "this research area",
            existing_papers=existing_str,
            num_needed=request_count,
        )
    else:  # reference
        prompt = SUGGEST_REFS_PROMPT.format(
            title=title,
            abstract=(abstract or "No abstract available.")[:1500],
            year=year or 2024,
            existing_papers=existing_str,
            num_needed=request_count,
        )

    client = OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic researcher. Return only valid JSON arrays.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=30.0,
            )

            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )

            suggestions = json.loads(content)

            if not isinstance(suggestions, list):
                logger.warning(f"LLM returned non-list for {paper_type} suggestions")
                return []

            valid = []
            for s in suggestions:
                if isinstance(s, dict) and s.get("title"):
                    valid.append({
                        "title": s["title"],
                        "authors": s.get("authors", ""),
                        "year": s.get("year"),
                        "why_relevant": s.get("why_relevant", ""),
                    })

            logger.info(f"LLM suggested {len(valid)} {paper_type} papers")
            return valid[:request_count]  # Return all suggestions - _resolve_suggestions will cap to num_needed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM {paper_type} suggestions: {e}")
            if attempt == MAX_RETRIES - 1:
                return []
        except Exception as e:
            logger.warning(f"LLM {paper_type} suggestion error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                return []

    return []


def lookup_paper_in_openalex(
    title: str,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up a paper in OpenAlex by title.

    Returns dict with: work_id, title, year, cited_by_count, abstract
    """
    if not title or len(title) < 10:
        return None

    try:
        search_title = title.replace('"', '').strip()

        url = "https://api.openalex.org/works"
        params = {
            "search": search_title,
            "per-page": 5,
        }
        if year:
            params["filter"] = f"publication_year:{year-1}-{year+1}"

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        if not results:
            return None

        # Find best match by title similarity
        search_lower = search_title.lower()
        best_match = None
        best_score = 0

        for r in results:
            result_title = (r.get("title") or "").lower()
            words1 = set(search_lower.split())
            words2 = set(result_title.split())
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > best_score:
                    best_score = overlap
                    best_match = r

        # Require at least 50% word overlap AND the paper must have significant citations
        # to avoid false positive matches on low-quality results
        if not best_match or best_score < 0.50:
            logger.debug(f"No good match for '{title[:50]}...' (best: {best_score:.2f})")
            return None

        # For landmark papers, verify it has substantial citations
        cited_count = best_match.get("cited_by_count", 0)
        if cited_count < 100 and best_score < 0.7:
            # Low citation + low match = likely wrong paper
            logger.debug(f"Rejecting low-confidence match: '{title[:50]}...' ({cited_count} cites, {best_score:.2f} match)")
            return None

        work_id = best_match.get("id", "")
        if work_id and "/" in work_id:
            work_id = work_id.rsplit("/", 1)[-1]

        return {
            "work_id": work_id,
            "title": best_match.get("title"),
            "year": best_match.get("publication_year"),
            "cited_by_count": best_match.get("cited_by_count", 0),
            "abstract": _extract_abstract(best_match),
        }

    except Exception as e:
        logger.warning(f"OpenAlex lookup error for '{title[:50]}...': {e}")
        return None


def _extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    """Extract abstract from OpenAlex work data."""
    abstract_inv = work.get("abstract_inverted_index")
    if not abstract_inv:
        return None

    try:
        word_positions = []
        for word, positions in abstract_inv.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(word for _, word in word_positions)
    except Exception:
        return None


def _resolve_suggestions(
    suggestions: List[Dict[str, Any]],
    existing_titles: set,
    relationship: str,
    max_papers: int,
) -> List[Dict[str, Any]]:
    """Look up suggestions in OpenAlex and return resolved papers."""
    resolved = []

    for suggestion in suggestions:
        if len(resolved) >= max_papers:
            break

        if suggestion["title"].lower() in existing_titles:
            continue

        paper = lookup_paper_in_openalex(
            title=suggestion["title"],
            year=suggestion.get("year"),
        )

        if paper:
            paper["relationship"] = relationship
            paper["why_relevant"] = suggestion.get("why_relevant", "")
            resolved.append(paper)
            existing_titles.add(paper["title"].lower())
            logger.info(
                f"Resolved {relationship}: {paper['title'][:50]}... "
                f"({paper['work_id']}, {paper['cited_by_count']} cites)"
            )

    return resolved


def fetch_top_cited_papers_openalex(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fallback: Fetch highly-cited papers directly from OpenAlex by search terms.

    This is used when LLM suggestions can't be resolved - we query OpenAlex
    directly for seminal papers in the field.
    """
    if not search_terms or len(search_terms) < 5:
        return []

    try:
        # Search for highly-cited papers matching the terms
        # Use 300 citation threshold - high enough for quality but not too restrictive
        url = "https://api.openalex.org/works"
        params = {
            "search": search_terms[:100],  # Limit query length
            "filter": f"publication_year:<{before_year},cited_by_count:>300",
            "sort": "cited_by_count:desc",
            "per-page": limit * 2,  # Fetch extra in case some are duplicates
        }

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        papers = []
        for r in results:
            if len(papers) >= limit:
                break

            paper_title = r.get("title", "")
            if not paper_title or paper_title.lower() in existing_titles:
                continue

            work_id = r.get("id", "")
            if work_id and "/" in work_id:
                work_id = work_id.rsplit("/", 1)[-1]

            papers.append({
                "work_id": work_id,
                "title": paper_title,
                "year": r.get("publication_year"),
                "cited_by_count": r.get("cited_by_count", 0),
                "abstract": _extract_abstract(r),
                "relationship": "field_landmark",  # Use valid enum value
                "why_relevant": f"Highly-cited foundational paper in this field",
            })
            existing_titles.add(paper_title.lower())

        logger.info(f"OpenAlex fallback found {len(papers)} highly-cited papers")
        return papers

    except Exception as e:
        logger.warning(f"OpenAlex fallback search error: {e}")
        return []


def _extract_search_terms(title: str, abstract: Optional[str]) -> str:
    """Extract key search terms from title and abstract."""
    # Use first ~50 chars of title + key abstract terms
    text = title
    if abstract:
        # Get first sentence of abstract
        first_sentence = abstract.split('.')[0] if '.' in abstract else abstract[:200]
        text += " " + first_sentence

    # Remove common words and punctuation, keep substantive terms
    import re
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'which',
                 'their', 'there', 'these', 'those', 'about', 'into', 'more', 'some',
                 'such', 'than', 'they', 'will', 'would', 'could', 'should', 'also',
                 'paper', 'method', 'approach', 'results', 'using', 'based', 'novel'}
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered[:8])  # Top 8 substantive words


# ============================================================================
# Main Entry Point
# ============================================================================

def supplement_grounding_papers(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    existing_refs: List[Dict[str, Any]],
    existing_landmarks: List[Dict[str, Any]],
    field: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Supplement grounding papers to ensure balanced coverage.

    Args:
        title: Target paper title
        abstract: Target paper abstract
        year: Target paper year
        existing_refs: Referenced works already found
        existing_landmarks: Landmark papers already found
        field: Optional field/topic description for better landmark suggestions

    Returns:
        (additional_refs, additional_landmarks) - may be empty lists
    """
    refs_needed, landmarks_needed, reason = assess_grounding_needs(
        existing_refs, existing_landmarks
    )

    if reason == "no_grounding_unavailable":
        logger.debug("No existing grounding papers - UNAVAILABLE case, not supplementing")
        return [], []

    if reason == "sufficient":
        logger.debug(f"Grounding sufficient: {len(existing_refs)} refs, {len(existing_landmarks)} landmarks")
        return [], []

    logger.info(f"Supplementing grounding: {reason}")

    # Track existing titles to avoid duplicates
    # IMPORTANT: Include the target paper's title to prevent self-citation
    existing_titles = {
        p.get("title", "").lower()
        for p in existing_refs + existing_landmarks
    }
    existing_titles.add(title.lower())  # Exclude target paper from suggestions

    additional_refs = []
    additional_landmarks = []

    # Suggest reference-type papers if needed
    if refs_needed > 0:
        logger.info(f"Requesting {refs_needed} reference-type papers from LLM")
        suggestions = suggest_papers_llm(
            title=title,
            abstract=abstract,
            year=year,
            existing_papers=existing_refs + existing_landmarks,
            num_needed=refs_needed,
            paper_type="reference",
        )
        additional_refs = _resolve_suggestions(
            suggestions, existing_titles, "cited_reference", refs_needed
        )

    # Suggest landmark-type papers if needed
    if landmarks_needed > 0:
        logger.info(f"Requesting {landmarks_needed} landmark-type papers from LLM")
        suggestions = suggest_papers_llm(
            title=title,
            abstract=abstract,
            year=year,
            existing_papers=existing_refs + existing_landmarks + additional_refs,
            num_needed=landmarks_needed,
            paper_type="landmark",
            field=field,
        )
        additional_landmarks = _resolve_suggestions(
            suggestions, existing_titles, "field_landmark", landmarks_needed
        )

        # FALLBACK: If LLM suggestions didn't resolve enough landmarks,
        # query OpenAlex directly for highly-cited papers in the field
        still_needed = landmarks_needed - len(additional_landmarks)
        if still_needed > 0:
            logger.info(f"LLM resolved {len(additional_landmarks)} landmarks, need {still_needed} more - trying OpenAlex fallback")
            search_terms = _extract_search_terms(title, abstract)
            fallback_papers = fetch_top_cited_papers_openalex(
                search_terms=search_terms,
                before_year=year or 2024,
                existing_titles=existing_titles,
                limit=still_needed,
            )
            additional_landmarks.extend(fallback_papers)

    logger.info(
        f"Supplemented: +{len(additional_refs)} refs, +{len(additional_landmarks)} landmarks"
    )
    return additional_refs, additional_landmarks
