"""
External prior art validation for novelty level classification.

When the LLM classifies a paper as "high" or "pioneering", this module
searches externally for papers that applied the SAME method to the SAME
problem before. Uses a focused LLM comparison call (adversarial reviewer)
to evaluate overlap — the LLM does comparison, not self-assessment.

Flow:
  LLM says "high"/"pioneering" → search_prior_art() → results found?
    ├─ NO  → keep classification
    └─ YES → evaluate_prior_art_overlap() (focused LLM comparison)
               ├─ has_prior_art=true → downgrade (pioneering→high, high→medium)
               └─ has_prior_art=false → keep classification
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from app.common.id_mapping import IdMapper
from app.feature3.grounding_supplement import _extract_search_terms
from app.feature3.landmark_retrieval import (
    _search_openalex_topic_landmarks,
    OPENALEX_API_KEY,
    OPENALEX_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
)
from app.feature3.paper_identity import content_word_overlap, decode_openalex_abstract
from app.feature3.json_utils import extract_json_from_llm_response

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# LLM settings (same model as grounding_supplement)
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Prior art search settings
MAX_PRIOR_ART_RESULTS = 10
SELF_SIMILARITY_THRESHOLD = 0.6  # Skip papers too similar to target (likely same paper)


# ---------------------------------------------------------------------------
# 1) search_prior_art  — two parallel OpenAlex searches
# ---------------------------------------------------------------------------

def search_prior_art(
    title: str,
    abstract: Optional[str],
    year: int,
    topic_id: Optional[str],
    whats_new: str,
    known_work_ids: set[str],
) -> List[Dict[str, Any]]:
    """
    Search for papers that may have done the same method+problem before.

    Two parallel OpenAlex searches:
    1. Full-text keyword search using terms from whats_new + title
    2. Topic-based search for highly-cited papers in the same topic

    Args:
        title: Target paper title
        abstract: Target paper abstract
        year: Target paper publication year (search only BEFORE this year)
        topic_id: OpenAlex primary_topic_id for topic-based search
        whats_new: The LLM's description of what the paper introduces
        known_work_ids: Work IDs to exclude (target paper + grounding papers)

    Returns:
        Deduplicated list of prior papers with abstracts, sorted by citations.
    """
    before_year = year  # Only papers published BEFORE the target paper

    # Extract search terms from whats_new (the novelty claim) + title
    search_terms = _extract_search_terms(whats_new, abstract)

    if not search_terms or len(search_terms) < 5:
        logger.info("Prior art search: insufficient search terms, skipping")
        return []

    keyword_results: List[Dict[str, Any]] = []
    topic_results: List[Dict[str, Any]] = []

    def _keyword_search():
        nonlocal keyword_results
        try:
            keyword_results = _search_openalex_fulltext(
                search_terms=search_terms,
                before_year=before_year,
                limit=MAX_PRIOR_ART_RESULTS,
            )
        except Exception as e:
            logger.warning(f"Prior art keyword search failed: {e}")

    def _topic_search():
        nonlocal topic_results
        if not topic_id:
            return
        try:
            topic_results = _search_openalex_topic_landmarks(
                topic_id=topic_id,
                before_year=before_year,
                limit=MAX_PRIOR_ART_RESULTS,
            )
        except Exception as e:
            logger.warning(f"Prior art topic search failed: {e}")

    # Run both searches in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_keyword_search),
            executor.submit(_topic_search),
        ]
        for f in as_completed(futures):
            f.result()  # Propagate exceptions if any

    # Merge and deduplicate
    all_papers = _merge_and_dedup(
        keyword_results, topic_results, known_work_ids, title
    )

    logger.info(
        f"Prior art search: {len(keyword_results)} keyword + "
        f"{len(topic_results)} topic → {len(all_papers)} deduped"
    )

    return all_papers[:MAX_PRIOR_ART_RESULTS]


def _search_openalex_fulltext(
    search_terms: str,
    before_year: int,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Full-text search on OpenAlex for prior art candidates.

    Unlike the grounding search, this has a LOWER citation threshold (>50)
    because we're looking for ANY prior work, not just landmarks.
    """
    for attempt in range(MAX_RETRIES):
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": search_terms[:100],
                "filter": f"publication_year:<{before_year},cited_by_count:>50",
                "sort": "cited_by_count:desc",
                "per-page": limit * 2,
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)

            if resp.status_code == 429:
                backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"OpenAlex 429 for prior art, waiting {backoff}s")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            results = resp.json().get("results", [])

            papers = []
            for w in results:
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue

                wid = wid_full.rsplit("/", 1)[-1]
                papers.append({
                    "work_id": wid,
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract": decode_openalex_abstract(
                        w.get("abstract_inverted_index")
                    ),
                })

            return papers

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
            else:
                logger.warning(f"OpenAlex prior art fulltext search failed: {e}")

    return []


def _merge_and_dedup(
    keyword_papers: List[Dict[str, Any]],
    topic_papers: List[Dict[str, Any]],
    known_work_ids: set[str],
    target_title: str,
) -> List[Dict[str, Any]]:
    """Merge two lists, deduplicate by work_id, exclude known papers and self."""
    seen_ids: set[str] = set()
    merged: List[Dict[str, Any]] = []

    for paper in keyword_papers + topic_papers:
        wid = paper.get("work_id", "")
        if not wid or wid in seen_ids:
            continue
        # Skip known papers (target + grounding)
        if wid in known_work_ids:
            continue
        # Skip papers that are too similar to target (likely the same paper)
        paper_title = paper.get("title") or ""
        if paper_title and content_word_overlap(target_title, paper_title) > SELF_SIMILARITY_THRESHOLD:
            continue
        # Must have abstract for meaningful comparison
        if not paper.get("abstract"):
            continue

        seen_ids.add(wid)
        merged.append(paper)

    # Sort by citation count descending
    merged.sort(key=lambda p: p.get("cited_by_count", 0), reverse=True)
    return merged


# ---------------------------------------------------------------------------
# 2) evaluate_prior_art_overlap  — adversarial LLM comparison
# ---------------------------------------------------------------------------

def evaluate_prior_art_overlap(
    target_title: str,
    target_abstract: Optional[str],
    target_whats_new: str,
    prior_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Use LLM as an adversarial prior art reviewer to check if any prior paper
    applies the SAME method to the SAME problem.

    The LLM is positioned as a critical reviewer (counters sycophancy).
    It does comparison/matching, NOT classification.

    Args:
        target_title: Title of the paper being assessed
        target_abstract: Abstract of the paper being assessed
        target_whats_new: LLM's description of what the paper introduces
        prior_papers: List of candidate prior art papers (with abstracts)

    Returns:
        {
            "has_prior_art": bool,
            "matching_paper_ids": [...],
            "reasoning": "..."
        }
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("No GROQ_API_KEY, skipping prior art evaluation")
        return {"has_prior_art": False, "matching_paper_ids": [], "reasoning": "no_api_key"}

    if not prior_papers:
        return {"has_prior_art": False, "matching_paper_ids": [], "reasoning": "no_candidates"}

    # Build paper list for the prompt with short IDs
    mapper = IdMapper("P")
    paper_descriptions = []
    for i, p in enumerate(prior_papers[:10], 1):
        short_id = mapper.add(p['work_id'])
        paper_descriptions.append(
            f"[{i}] {short_id} ({p.get('year', '?')}): \"{p['title']}\"\n"
            f"    Abstract: {(p.get('abstract') or 'N/A')[:300]}"
        )
    papers_text = "\n\n".join(paper_descriptions)

    prompt = f"""You are an adversarial prior art reviewer. Your job is to find PRIOR WORK
that invalidates a novelty claim. Be critical and skeptical.

TARGET PAPER: "{target_title}"
{f'Abstract: {target_abstract}' if target_abstract else ''}

CLAIMED NOVELTY: {target_whats_new}

CANDIDATE PRIOR PAPERS (all published BEFORE the target):
{papers_text}

TASK: Do ANY of these prior papers apply the SAME method/approach to the SAME problem
as the target paper? Look for:
- Same algorithmic technique applied to the same domain/task
- Same architectural design used for the same purpose
- Same theoretical framework applied to the same problem class

Be STRICT: "both use neural networks" is NOT a match. The method AND problem must
both overlap substantially. General techniques applied to different problems don't count.
Papers that use a DIFFERENT method for the same problem don't count either.

Respond in JSON:
{{
  "has_prior_art": true/false,
  "matching_paper_ids": ["W...", ...],
  "reasoning": "One sentence explaining your decision"
}}"""

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_VERSION,
            messages=[
                {"role": "system", "content": "You are a critical prior art reviewer. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        content = response.choices[0].message.content or ""
        parsed, error = extract_json_from_llm_response(content, expected_type="object")

        if parsed and isinstance(parsed, dict):
            result = {
                "has_prior_art": bool(parsed.get("has_prior_art", False)),
                "matching_paper_ids": mapper.resolve_list(parsed.get("matching_paper_ids", [])),
                "reasoning": parsed.get("reasoning", ""),
            }
            logger.info(
                f"Prior art evaluation: has_prior_art={result['has_prior_art']}, "
                f"matches={result['matching_paper_ids']}, reason={result['reasoning']}"
            )
            return result
        else:
            logger.warning(f"Failed to parse prior art LLM response: {error}")
            return {"has_prior_art": False, "matching_paper_ids": [], "reasoning": f"parse_error: {error}"}

    except Exception as e:
        logger.warning(f"Prior art LLM evaluation failed: {e}")
        return {"has_prior_art": False, "matching_paper_ids": [], "reasoning": f"error: {e}"}


# ---------------------------------------------------------------------------
# 3) validate_novelty_level  — entry point (replaces _enforce_novelty_level)
# ---------------------------------------------------------------------------

def validate_novelty_level(
    novelty_assessment: Dict[str, Any],
    work_data: Dict[str, Any],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
) -> None:
    """
    Validate and potentially downgrade novelty level using external prior art search.

    Only triggers for "high" or "pioneering" — medium/low skip entirely.
    Fail-open: any error keeps the original level.

    Mutates novelty_assessment in place (same pattern as _enforce_novelty_level).

    Args:
        novelty_assessment: The LLM's novelty assessment dict (mutated in place)
        work_data: Paper metadata (title, abstract, year, primary_topic_id, etc.)
        referenced_works: Grounding reference papers
        landmarks: Grounding landmark papers
    """
    current_level = (novelty_assessment.get("novelty_level") or "").lower().strip()

    # Only validate high and pioneering — medium/low pass through
    if current_level not in ("high", "pioneering"):
        logger.info(f"Novelty validation: level={current_level}, skipping (not high/pioneering)")
        return

    title = work_data.get("title", "")
    abstract = work_data.get("abstract")
    year = work_data.get("year")
    topic_id = work_data.get("primary_topic_id")
    whats_new = novelty_assessment.get("whats_new", "")

    if not title or not year:
        logger.warning("Novelty validation: missing title or year, keeping original level")
        return

    # Collect known work_ids to exclude from search
    known_ids = set()
    for p in referenced_works + landmarks:
        wid = p.get("work_id") or p.get("id", "")
        if wid:
            known_ids.add(wid)
    # Also exclude the target paper itself
    target_wid = work_data.get("work_id") or work_data.get("id", "")
    if target_wid:
        known_ids.add(target_wid)

    try:
        # Step 1: Search for prior art
        prior_papers = search_prior_art(
            title=title,
            abstract=abstract,
            year=year,
            topic_id=topic_id,
            whats_new=whats_new,
            known_work_ids=known_ids,
        )

        if not prior_papers:
            logger.info(
                f"Novelty validation: no prior art found for '{title[:60]}', "
                f"keeping {current_level}"
            )
            return

        # Step 2: LLM evaluation of overlap
        evaluation = evaluate_prior_art_overlap(
            target_title=title,
            target_abstract=abstract,
            target_whats_new=whats_new,
            prior_papers=prior_papers,
        )

        if not evaluation.get("has_prior_art"):
            logger.info(
                f"Novelty validation: LLM found no matching prior art for "
                f"'{title[:60]}', keeping {current_level}"
            )
            return

        # Step 3: Downgrade
        old_level = current_level
        if current_level == "pioneering":
            new_level = "high"
        else:  # high → medium
            new_level = "medium"

        novelty_assessment["novelty_level"] = new_level
        matching_ids = evaluation.get("matching_paper_ids", [])
        reasoning = evaluation.get("reasoning", "")

        logger.info(
            f"Novelty validation: downgraded '{title[:60]}' from {old_level} → "
            f"{new_level}. Prior art: {matching_ids}. Reason: {reasoning}"
        )

        # Store validation metadata for debugging
        novelty_assessment["_validation"] = {
            "original_level": old_level,
            "downgraded_to": new_level,
            "matching_prior_art": matching_ids,
            "reasoning": reasoning,
            "prior_art_count": len(prior_papers),
        }

    except Exception as e:
        # Fail-open: any error → keep original level
        logger.warning(
            f"Novelty validation failed for '{title[:60]}': {e}. "
            f"Keeping original level: {current_level}"
        )
