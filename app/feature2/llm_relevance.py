"""
Relevance scoring module for the ranking pipeline.

This module scores papers using LLM-based tier classification.
Papers are scored in batches for cost efficiency, with results cached by (paper_id, query_hash).
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .work_topic_store import WorkForMap
from .query_expansion import normalize_query_for_expansion

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

LLM_SCORING_HARD_CAP = 90  # Increased slightly to match 2-wave equivalent
BATCH_SIZE = 15
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
MAX_PARALLEL_BATCHES = 6  # 6 batches of 15 = 90 papers, all in parallel
# Model version for cache invalidation
# v12: LLM outputs paper TYPE (foundational/methodology/review/application/theoretical/other)
#      Python determines output CATEGORY using type + citations + year
# v13: Added strict domain matching rules - papers from target domain without target method → LOW
# v14: Clarified foundational vs methodology distinction
# v15: Simplified prompt, removed hardcoded examples, Python routing respects LLM type
# v16: Added back clear examples for paper types
# v17: Added modality boundaries rule (vision/NLP/audio are distinct domains)
# v18: Cache invalidation - ensure domain matching is applied to all papers
# v19: Intersection-aware scoring - compound queries require ALL concepts, not just one
# v20: Stricter intersection scoring - 2-of-3 concepts caps at MEDIUM, not HIGH
# v21: Single-topic vs intersection distinction, sharper MEDIUM/LOW boundary,
#      generic-tool and social-impact rules
# v22: Technique-name≠domain-match rule, sub-task boundaries, stronger modality enforcement
# v23: Research-contribution vs application-use rule, generic-theory cap for domain-specific queries,
#      strengthened sub-task boundary with explicit examples
# v24: Expanded modality boundaries (time-series, medical imaging explicit), human-concept vs ML-concept rule
# v25: Consolidated prompt - merged 9 rule sections into 4 clear sections for better LLM compliance
# v26: System-boundary principle (same concept in different scientific system ≠ same domain),
#      intersection decomposition (METHOD + DOMAIN, different method for same task → LOW)
# v27: Adjacent-phenomena specificity rule (dark matter ≠ dark energy, etc.),
#      cause-effect topic constraint for "[cause] [system]" queries
# v28: Continuous 0-10 scoring replaces 5 discrete tiers (ESSENTIAL/HIGH/MEDIUM/LOW/NONE).
#      Eliminates "60% of papers get 0.50" problem — papers get 5.2, 6.8, 7.3 etc.
#      Score is divided by 10 for 0-1 range. Used as one input to RRF ensemble.
# v29: Qualified-topic rule — "[QUALIFIER] [NOUN]" queries cap papers about
#      the NOUN without the QUALIFIER at max 3 (e.g., LLM alignment, federated learning).
MODEL_VERSION = "llm-type-v29"

# Legacy tier mapping kept for backwards compatibility with cached scores
TIER_SCORES = {
    "ESSENTIAL": 0.95,
    "HIGH": 0.75,
    "MEDIUM": 0.50,
    "LOW": 0.25,
    "NONE": 0.05,
}

# Valid paper types - LLM classifies paper TYPE, Python decides output category
PAPER_TYPES = {"foundational", "methodology", "review", "application", "theoretical", "other"}


def get_cached_scores(
    conn: Connection,
    query_hash: str,
    paper_ids: List[str],
    model_version: str = MODEL_VERSION,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve cached relevance scores and paper types for given papers and query."""
    if not paper_ids:
        return {}

    try:
        rows = conn.execute(
            text("""
                SELECT paper_id, relevance_score, paper_type
                FROM llm_relevance_cache
                WHERE query_hash = :query_hash
                AND paper_id = ANY(:paper_ids)
                AND model_version = :model_version
            """),
            {"query_hash": query_hash, "paper_ids": paper_ids, "model_version": model_version},
        ).mappings().all()

        return {
            row["paper_id"]: {
                "score": float(row["relevance_score"]),
                "paper_type": row.get("paper_type") or "other",
            }
            for row in rows
        }
    except Exception as e:
        logger.warning(f"Failed to retrieve cached scores: {e}")
        return {}


def cache_scores(
    conn: Connection,
    query_hash: str,
    scores: Dict[str, Dict[str, Any]],
    model_version: str,
) -> None:
    """Store relevance scores and paper types in cache."""
    if not scores:
        return

    try:
        for paper_id, data in scores.items():
            score = data.get("score", 0.0) if isinstance(data, dict) else data
            paper_type = data.get("paper_type", "other") if isinstance(data, dict) else "other"
            conn.execute(
                text("""
                    INSERT INTO llm_relevance_cache (paper_id, query_hash, relevance_score, paper_type, model_version)
                    VALUES (:paper_id, :query_hash, :relevance_score, :paper_type, :model_version)
                    ON CONFLICT (paper_id, query_hash) DO UPDATE SET
                        relevance_score = EXCLUDED.relevance_score,
                        paper_type = EXCLUDED.paper_type,
                        model_version = EXCLUDED.model_version,
                        created_at = now()
                """),
                {
                    "paper_id": paper_id,
                    "query_hash": query_hash,
                    "relevance_score": score,
                    "paper_type": paper_type,
                    "model_version": model_version,
                },
            )
    except Exception as e:
        logger.warning(f"Failed to cache scores: {e}")


def _truncate_text(text_str: str, max_chars: int = 1500) -> str:
    """Truncate text to max characters, preserving word boundaries."""
    if len(text_str) <= max_chars:
        return text_str
    truncated = text_str[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def prepare_scoring_input(
    works: Dict[str, WorkForMap],
    paper_ids: List[str],
) -> List[Dict[str, str]]:
    """Prepare (paper_id, title, abstract) dicts for scoring."""
    papers = []
    for pid in paper_ids:
        w = works.get(pid)
        if not w:
            continue
        title = w.title or "Untitled"
        abstract = getattr(w, 'abstract', None) or ""
        papers.append({
            "paper_id": pid,
            "title": title,
            "abstract": _truncate_text(abstract),
        })
    return papers


def score_batch(
    query_text: str,
    papers: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Score papers using LLM tier classification.

    Returns dict mapping paper_id to {"score": float, "paper_type": str}.
    Uses Groq/Llama for tier-based classification and paper type detection.
    """
    if not papers:
        return {}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, returning zero scores")
        return {p["paper_id"]: {"score": 0.0, "paper_type": "other"} for p in papers}

    normalized_query = normalize_query_for_expansion(query_text)
    client = Groq(api_key=api_key)

    # Build papers JSON for the prompt
    papers_for_prompt = []
    for p in papers:
        papers_for_prompt.append({
            "id": p["paper_id"],
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "")[:800],
        })
    papers_json = json.dumps(papers_for_prompt, indent=2)

    prompt = f"""Score papers by relevance on a continuous 0-10 scale AND classify type. Return ONLY a JSON object.

STEP 1 - DOMAIN CHECK (apply FIRST):
Identify the paper's PRIMARY research domain. If it does NOT match the query's domain → score 0-2.
Sharing a technique name (attention, transformer, BERT, RL) does NOT make domains match.
The paper must CONTRIBUTE TO the query's domain.

Domain mismatches → 0-2:
- Vision paper for NLP query, NLP paper for vision query
- Genomics paper using BERT for an NLP query
- Game RL (Atari) for robotics query
- Different scientific systems (bacterial vs tumor drug resistance)
- Adjacent phenomena (dark matter ≠ dark energy, Type 1 ≠ Type 2 diabetes)

STEP 2 - QUERY TYPE:
- SINGLE TOPIC: Papers about any core aspect can score 7-10.
  For "[cause] [system]" queries, paper must discuss the cause-effect, not just the system.
- INTERSECTION QUERY ("[METHOD] + [DOMAIN]"):
  Paper about METHOD only → max 5. DOMAIN only → max 5.
  Different method for same task → max 2. Must address THE INTERSECTION for 7+.
- QUALIFIED TOPIC ("[QUALIFIER] [NOUN]" like "federated learning", "LLM alignment", "graph neural networks"):
  Paper about the NOUN without the QUALIFIER → max 3.
  "LLM alignment" requires alignment techniques (RLHF, DPO, safety), not just LLM usage/applications.
  "Federated learning" requires federated protocols, not just distributed or general ML.
  "Reinforcement learning" requires RL algorithms (Q-learning, policy gradient), not just optimization.
  The qualifier is what makes the topic specific — without it, the paper is a different topic.

STEP 3 - CONTINUOUS SCORE (0-10):
- 9-10: Seminal/foundational work that defined this specific field
- 7-8: Directly addresses the query topic — paper is primarily ABOUT this
- 5-6: Same field, useful context, but not primarily about the query topic
- 3-4: Tangentially related, different sub-area or only mentions topic
- 1-2: Wrong domain, wrong modality, or barely related
- 0: Completely unrelated

IMPORTANT: Use the FULL range. Do NOT cluster scores. A 5.3 is different from a 6.7.

PAPER TYPE:
- foundational: Introduced a genuinely new paradigm (RARE)
- methodology: Tools, algorithms, frameworks, techniques (DEFAULT)
- review: Surveys, meta-analyses, systematic reviews
- application: Real-world implementations, clinical trials
- theoretical: Pure theory, proofs
- other: If unclear

QUERY: {normalized_query}

PAPERS:
{papers_json}

OUTPUT (JSON only):
{{"paper_id": {{"score": NUMBER, "type": "TYPE"}}, ...}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from anywhere in the response
            # Handle: plain JSON, ```json blocks, or JSON embedded in text
            start = content.find("{")
            if start == -1:
                logger.warning(f"LLM response has no JSON: {content[:200]}")
                continue

            # Find the matching closing brace (handle nested objects)
            depth = 0
            end = start
            for i, c in enumerate(content[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end <= start:
                logger.warning(f"LLM response has malformed JSON: {content[:200]}")
                continue

            json_str = content[start:end]
            response_map = json.loads(json_str)

            # Convert to result format with scores and paper types
            result = {}
            for p in papers:
                pid = p["paper_id"]
                entry = response_map.get(pid, {})

                if isinstance(entry, (int, float)):
                    # Raw number — treat as 0-10 score
                    raw_score = float(entry)
                    paper_type = "other"
                elif isinstance(entry, str):
                    # Legacy tier string — map to score
                    tier = entry.upper()
                    raw_score = TIER_SCORES.get(tier, 0.5) * 10.0
                    paper_type = "other"
                else:
                    # Dict with "score" and "type" (expected format)
                    raw_score = float(entry.get("score", 0))
                    paper_type = entry.get("type", "other").lower()
                    # Handle legacy "relevance" field (tier string)
                    if isinstance(raw_score, str) or raw_score in (0.05, 0.25, 0.50, 0.75, 0.95):
                        tier = entry.get("relevance", "NONE").upper()
                        if tier in TIER_SCORES:
                            raw_score = TIER_SCORES[tier] * 10.0

                # Clamp to [0, 10] and normalize to [0, 1]
                raw_score = max(0.0, min(10.0, raw_score))
                normalized_score = raw_score / 10.0

                if paper_type not in PAPER_TYPES:
                    paper_type = "other"

                result[pid] = {
                    "score": normalized_score,
                    "paper_type": paper_type,
                }

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        except Exception as e:
            logger.warning(f"LLM API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

    # All retries failed
    return {p["paper_id"]: {"score": 0.0, "paper_type": "other"} for p in papers}


def score_papers(
    conn: Connection,
    query_text: str,
    query_hash: str,
    paper_ids: List[str],
    works: Dict[str, WorkForMap],
    llm_scoring_cap: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Main entry point for relevance scoring.

    1. Check cache for existing scores
    2. Identify papers needing scoring
    3. Apply hard cap (default 88, or custom cap if provided)
    4. Score in batches (parallel)
    5. Cache results
    6. Return all scores and paper types

    Returns dict mapping paper_id to {"score": float, "paper_type": str}.
    """
    if not paper_ids:
        return {}

    # Apply hard cap
    effective_cap = llm_scoring_cap if llm_scoring_cap is not None else LLM_SCORING_HARD_CAP
    if len(paper_ids) > effective_cap:
        logger.info(f"Applying hard cap: {len(paper_ids)} -> {effective_cap}")
        paper_ids = paper_ids[:effective_cap]

    # Check cache
    cached_scores = get_cached_scores(conn, query_hash, paper_ids)
    logger.info(f"Cache hit: {len(cached_scores)}/{len(paper_ids)} papers")

    # Find papers needing scoring
    needs_scoring = [pid for pid in paper_ids if pid not in cached_scores]

    if not needs_scoring:
        return cached_scores

    # Prepare scoring input
    scoring_input = prepare_scoring_input(works, needs_scoring)
    logger.info(f"Scoring {len(scoring_input)} papers with LLM tier classification")

    # Split into batches
    batches = []
    for i in range(0, len(scoring_input), BATCH_SIZE):
        batches.append(scoring_input[i:i + BATCH_SIZE])

    # Score batches in parallel
    new_scores: Dict[str, Dict[str, Any]] = {}

    def score_single_batch(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        return score_batch(query_text, batch)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(score_single_batch, batch): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            try:
                batch_scores = future.result()
                new_scores.update(batch_scores)
            except Exception as e:
                logger.error(f"Batch scoring failed: {e}")

    logger.info(f"Scored {len(new_scores)} new papers")

    # Cache new scores
    if new_scores:
        cache_scores(conn, query_hash, new_scores, MODEL_VERSION)

    # Combine cached and new scores
    all_scores = {**cached_scores, **new_scores}

    # Ensure all requested papers have a score
    for pid in paper_ids:
        if pid not in all_scores:
            all_scores[pid] = {"score": 0.0, "paper_type": "other"}

    return all_scores


def score_wave(
    conn: Connection,
    query_text: str,
    query_hash: str,
    paper_ids: List[str],
    works: Dict[str, WorkForMap],
) -> Dict[str, Dict[str, Any]]:
    """Score a single wave of papers (no cap logic — caller manages wave sizes).

    Checks cache, scores uncached papers in parallel batches, caches results.
    Returns dict mapping paper_id to {"score": float, "paper_type": str}.
    """
    if not paper_ids:
        return {}

    cached_scores = get_cached_scores(conn, query_hash, paper_ids)
    needs_scoring = [pid for pid in paper_ids if pid not in cached_scores]

    if not needs_scoring:
        return cached_scores

    scoring_input = prepare_scoring_input(works, needs_scoring)
    logger.info(f"Wave scoring {len(scoring_input)} papers with LLM tier classification")

    batches = []
    for i in range(0, len(scoring_input), BATCH_SIZE):
        batches.append(scoring_input[i:i + BATCH_SIZE])

    new_scores: Dict[str, Dict[str, Any]] = {}

    def score_single_batch(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        return score_batch(query_text, batch)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(score_single_batch, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            try:
                batch_scores = future.result()
                new_scores.update(batch_scores)
            except Exception as e:
                logger.error(f"Wave batch scoring failed: {e}")

    if new_scores:
        cache_scores(conn, query_hash, new_scores, MODEL_VERSION)

    all_scores = {**cached_scores, **new_scores}
    for pid in paper_ids:
        if pid not in all_scores:
            all_scores[pid] = {"score": 0.0, "paper_type": "other"}

    return all_scores
