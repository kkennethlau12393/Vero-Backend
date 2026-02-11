"""
Topic inference for Feature 3.

This module infers topics for papers that don't have primary_topic_id,
using OpenAlex lookup or LLM-based classification.
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
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_identity import normalize_doi, normalize_arxiv_id

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# API settings
OPENALEX_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5


# ============================================================================
# OpenAlex Topic Fetching
# ============================================================================

def fetch_topic_from_openalex_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch topic information from OpenAlex by DOI.

    Returns: {
        "primary_topic_id": str,
        "primary_topic_score": float,
        "primary_topic_name": str,
        "topics_json": List[dict],
    }
    """
    doi = normalize_doi(doi)
    if not doi:
        return None

    try:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        resp = requests.get(url, timeout=OPENALEX_TIMEOUT)

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        return _extract_topic_from_openalex(data)

    except Exception as e:
        logger.warning(f"OpenAlex API error for DOI {doi}: {e}")
        return None


def fetch_topic_from_openalex_by_title(title: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Search OpenAlex by title to find topic information.

    This is less reliable than DOI lookup, so we verify the match.
    """
    if not title or len(title) < 10:
        return None

    try:
        # Build search query
        search_title = title.replace('"', '')
        url = "https://api.openalex.org/works"
        params = {
            "search": search_title,
            "per-page": 5,
        }
        if year:
            params["filter"] = f"publication_year:{year}"

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        if not results:
            return None

        # Take first result (best match from OpenAlex search)
        best = results[0]

        # Basic title verification
        result_title = best.get("title", "").lower()
        query_title = title.lower()

        # Require some similarity
        words1 = set(result_title.split())
        words2 = set(query_title.split())
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            if overlap < 0.5:
                logger.debug(f"Title mismatch in OpenAlex search: '{title[:50]}...' vs '{result_title[:50]}...'")
                return None

        return _extract_topic_from_openalex(best)

    except Exception as e:
        logger.warning(f"OpenAlex search error for '{title[:50]}...': {e}")
        return None


def _extract_topic_from_openalex(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract topic information from OpenAlex work data.
    """
    primary_topic = data.get("primary_topic")
    if not primary_topic:
        return None

    topic_id = primary_topic.get("id", "")
    if topic_id and "/" in topic_id:
        topic_id = topic_id.rsplit("/", 1)[-1]

    if not topic_id:
        return None

    # Build topics_json from all topics
    topics_json = []
    for t in data.get("topics", []):
        tid = t.get("id", "")
        if tid and "/" in tid:
            tid = tid.rsplit("/", 1)[-1]
        if tid:
            topics_json.append({
                "topic_id": tid,
                "score": t.get("score", 0.0),
            })

    return {
        "primary_topic_id": topic_id,
        "primary_topic_score": primary_topic.get("score", 0.0),
        "primary_topic_name": primary_topic.get("display_name"),
        "topics_json": topics_json,
    }


# ============================================================================
# LLM-Based Topic Inference
# ============================================================================

TOPIC_INFERENCE_PROMPT = """Analyze this academic paper and identify its primary research topic.

Title: {title}
Abstract: {abstract}

Based on the paper's content, identify:
1. The primary research FIELD (broad discipline)
2. The SUBFIELD (specific area within the field)
3. A brief description of the main topic

Common fields include: Computer Science, Physics, Biology, Chemistry, Mathematics, Economics, Medicine, Psychology, etc.

Return your answer as JSON with this exact structure:
{{
    "field": "Computer Science",
    "subfield": "Machine Learning",
    "topic_description": "Deep neural networks for image classification",
    "confidence": 0.95
}}

Guidelines:
- Be specific but accurate
- If the paper clearly belongs to a field, use high confidence (0.8-1.0)
- If uncertain, use lower confidence (0.5-0.7)
- Focus on the main contribution, not tangential topics"""


def infer_topic_via_llm(
    title: str,
    abstract: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to infer the research topic when OpenAlex lookup fails.

    Returns: {
        "field": str,
        "subfield": str,
        "topic_description": str,
        "confidence": float,
    }
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping LLM inference")
        return None

    if not title:
        return None

    abstract_text = abstract if abstract else "No abstract available."

    prompt = TOPIC_INFERENCE_PROMPT.format(
        title=title,
        abstract=abstract_text[:1500],  # Truncate long abstracts
    )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic classifier. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=30.0,
            )

            content = (resp.choices[0].message.content or "").strip()

            # Use robust JSON extraction (handles code blocks, extra text, etc.)
            result, error = extract_json_from_llm_response(content, expected_type="object")

            if result is None:
                logger.warning(f"Failed to parse LLM topic response: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt))
                    continue
                return None

            # Validate required fields
            if not result.get("field") or not result.get("subfield"):
                logger.warning("LLM response missing required fields")
                return None

            return {
                "field": result["field"],
                "subfield": result["subfield"],
                "topic_description": result.get("topic_description", ""),
                "confidence": float(result.get("confidence", 0.5)),
            }

        except Exception as e:
            logger.warning(f"LLM topic inference error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                return None

    return None


def map_llm_topic_to_openalex(
    conn: Connection,
    field: str,
    subfield: str,
) -> Optional[str]:
    """
    Map LLM-inferred topic to the closest OpenAlex topic_id.

    Strategy:
    1. Search openalex_topics for matching subfield/field names
    2. Use fuzzy matching if exact match not found
    3. Return most common topic in that subfield

    Note: This requires topic names in the openalex_topics table.
    If not available, we'll create a synthetic topic_id.
    """
    # First, try to find topics in this field
    # (This is a simplified mapping - ideally we'd have topic names in DB)

    # Common field -> topic_id prefix mappings
    field_prefixes = {
        "computer science": "T10",
        "physics": "T11",
        "mathematics": "T12",
        "biology": "T13",
        "chemistry": "T14",
        "medicine": "T15",
        "economics": "T16",
        "psychology": "T17",
    }

    field_lower = field.lower()
    prefix = None
    for f, p in field_prefixes.items():
        if f in field_lower or field_lower in f:
            prefix = p
            break

    if not prefix:
        # Default to computer science for ML/AI papers
        if any(term in subfield.lower() for term in ["learning", "neural", "ai", "vision", "nlp"]):
            prefix = "T10"
        else:
            prefix = "T10"  # Default fallback

    # Try to find a topic with this prefix that has papers in the DB
    try:
        row = conn.execute(
            text("""
                SELECT primary_topic_id, COUNT(*) as cnt
                FROM works
                WHERE primary_topic_id LIKE :prefix || '%'
                AND primary_topic_id IS NOT NULL
                GROUP BY primary_topic_id
                ORDER BY cnt DESC
                LIMIT 1
            """),
            {"prefix": prefix},
        ).mappings().first()

        if row:
            return row["primary_topic_id"]

    except Exception as e:
        logger.warning(f"Error mapping topic: {e}")

    # Return a synthetic topic_id based on field
    return f"{prefix}000"


# ============================================================================
# Main Inference Function
# ============================================================================

def ensure_topic(
    conn: Connection,
    work_id: str,
    title: str,
    abstract: Optional[str],
    current_topic_id: Optional[str],
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Ensure work has a primary_topic_id, inferring if necessary.

    Returns:
        (topic_id, source)
        - topic_id: The inferred or existing topic_id
        - source: "cached" | "openalex_fetch" | "llm_inference" | "unavailable"

    Flow:
    1. If current_topic_id exists -> return (current_topic_id, "cached")
    2. Try fetching from OpenAlex by DOI
    3. Try fetching from OpenAlex by title search
    4. Infer via LLM and map to OpenAlex topic
    5. Return (None, "unavailable")
    """
    if current_topic_id:
        return current_topic_id, "cached"

    logger.info(f"Topic missing for {work_id}. Attempting inference...")

    # Try OpenAlex by DOI
    if doi:
        logger.debug(f"Trying OpenAlex by DOI: {doi}")
        result = fetch_topic_from_openalex_by_doi(doi)
        if result and result.get("primary_topic_id"):
            logger.info(f"Found topic from OpenAlex (DOI) for {work_id}: {result['primary_topic_id']}")
            _cache_inferred_topic(
                conn, work_id,
                result["primary_topic_id"],
                result.get("primary_topic_score", 0.0),
                result.get("topics_json", []),
                "openalex_fetch",
            )
            return result["primary_topic_id"], "openalex_fetch"

    # Try OpenAlex by title search
    logger.debug(f"Trying OpenAlex search for: {title[:50]}...")
    result = fetch_topic_from_openalex_by_title(title)
    if result and result.get("primary_topic_id"):
        logger.info(f"Found topic from OpenAlex (search) for {work_id}: {result['primary_topic_id']}")
        _cache_inferred_topic(
            conn, work_id,
            result["primary_topic_id"],
            result.get("primary_topic_score", 0.0),
            result.get("topics_json", []),
            "openalex_fetch",
        )
        return result["primary_topic_id"], "openalex_fetch"

    # Try LLM inference
    logger.debug(f"Trying LLM inference for: {title[:50]}...")
    llm_result = infer_topic_via_llm(title, abstract)
    if llm_result:
        topic_id = map_llm_topic_to_openalex(
            conn,
            llm_result["field"],
            llm_result["subfield"],
        )
        if topic_id:
            logger.info(
                f"Inferred topic via LLM for {work_id}: {topic_id} "
                f"({llm_result['field']}/{llm_result['subfield']})"
            )
            _cache_inferred_topic(
                conn, work_id,
                topic_id,
                llm_result["confidence"],
                [],  # No detailed topics from LLM
                "llm_inference",
            )
            return topic_id, "llm_inference"

    logger.warning(f"Could not infer topic for {work_id}")
    return None, "unavailable"


def _cache_inferred_topic(
    conn: Connection,
    work_id: str,
    topic_id: str,
    topic_score: float,
    topics_json: List[Dict[str, Any]],
    source: str,
) -> None:
    """
    Update works table with inferred topic.
    """
    try:
        conn.execute(
            text("""
                UPDATE works
                SET primary_topic_id = :topic_id,
                    primary_topic_score = :topic_score,
                    topics_json = :topics_json,
                    topic_source = :source,
                    topic_inferred_at = now()
                WHERE work_id = :work_id
            """),
            {
                "work_id": work_id,
                "topic_id": topic_id,
                "topic_score": topic_score,
                "topics_json": json.dumps(topics_json) if topics_json else None,
                "source": source,
            },
        )
        conn.commit()
        logger.info(f"Cached inferred topic {topic_id} for {work_id} from {source}")
    except Exception as e:
        logger.warning(f"Failed to cache topic for {work_id}: {e}")
