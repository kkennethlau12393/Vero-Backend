"""
Query expansion module for the new ranking pipeline.

This module extracts key concepts from a user query and generates synonyms
and related terms for each concept, with importance scoring. Results are
cached by query hash to avoid redundant LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PROMPT_VERSION = "qe_structured_v2"

# Regex to strip question prefixes for consistent query processing
# Three groups: (1) question words, (2) optional auxiliary verbs, (3) optional articles
# Note: Articles require word boundary (\b) to avoid matching inside words like "algorithms"
QUESTION_PREFIXES = re.compile(
    r'^(what|how|why|when|where|which|who|can|could|should|would|does|do|is|are)\s+'
    r'(is|are|does|do|was|were|will|would|could|should|can)?\s*'
    r'(?:(the|a|an|some)\s+)?',
    re.IGNORECASE
)


def normalize_query_for_expansion(query_text: str) -> str:
    """Strip question prefixes to normalize queries for consistent processing.

    Pure regex - no LLM calls. Ensures question-style and topic-style queries
    produce identical results.

    Examples:
        "What are the mechanisms of X?" -> "mechanisms of X"
        "How does climate change affect Y?" -> "climate change affect Y"
        "mechanisms of X" -> "mechanisms of X" (unchanged)
    """
    # Remove leading question words
    normalized = QUESTION_PREFIXES.sub('', query_text.strip())
    # Remove trailing question mark
    normalized = normalized.rstrip('?').strip()
    return normalized if normalized else query_text.strip()


@dataclass
class ExpandedConcept:
    """A key concept from the query with its expansions."""
    term: str                              # e.g., "children"
    importance: float                      # 0.0-1.0, how critical to the query
    synonyms: List[str] = field(default_factory=list)    # direct synonyms
    related_terms: List[str] = field(default_factory=list)  # related but may drift
    foundational_works: List[str] = field(default_factory=list)  # seminal paper names/acronyms


@dataclass
class QueryExpansion:
    """Result of query expansion containing structured concepts."""
    original_query: str
    concepts: List[ExpandedConcept]
    query_hash: str
    # Legacy fields for backward compatibility
    original_terms: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)


def compute_query_hash(query_text: str) -> str:
    """Compute SHA-256 hash of normalized query text.

    Applies question prefix normalization so that question-style and
    topic-style queries produce the same hash.
    """
    normalized = normalize_query_for_expansion(query_text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tokenize_query(query_text: str) -> List[str]:
    """Extract individual terms from the query."""
    import re
    # Simple tokenization: lowercase, remove punctuation, split on whitespace
    cleaned = re.sub(r'[^\w\s]', ' ', query_text.lower())
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


def get_cached_expansion(conn: Connection, query_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached expansion data if available.

    Returns the raw JSON data (concepts list) or None if not cached.
    """
    try:
        row = conn.execute(
            text("""
                SELECT expansion_terms
                FROM query_expansion_cache
                WHERE query_hash = :query_hash
            """),
            {"query_hash": query_hash},
        ).mappings().first()
        if row:
            terms = row["expansion_terms"]
            if isinstance(terms, dict):
                return terms
            if isinstance(terms, str):
                return json.loads(terms)
            if isinstance(terms, list):
                # Legacy format - return as-is for backward compat
                return {"legacy_terms": terms}
        return None
    except Exception as e:
        logger.warning(f"Failed to retrieve cached expansion: {e}")
        return None


def cache_expansion(
    conn: Connection,
    query_hash: str,
    query_text: str,
    expansion_data: Dict[str, Any],
    model_version: str,
) -> None:
    """Store expansion data in cache."""
    try:
        conn.execute(
            text("""
                INSERT INTO query_expansion_cache (query_hash, query_text, expansion_terms, model_version)
                VALUES (:query_hash, :query_text, :expansion_terms, :model_version)
                ON CONFLICT (query_hash) DO UPDATE SET
                    expansion_terms = EXCLUDED.expansion_terms,
                    model_version = EXCLUDED.model_version,
                    created_at = now()
            """),
            {
                "query_hash": query_hash,
                "query_text": query_text,
                "expansion_terms": json.dumps(expansion_data),
                "model_version": model_version,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to cache expansion: {e}")


def generate_structured_expansion(query_text: str) -> Dict[str, Any]:
    """Call LLM to extract key concepts with synonyms and related terms.

    Returns a dict with 'concepts' list, or empty dict on failure.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping query expansion")
        return {}

    model = MODEL_VERSION
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    logger.info(f"Query expansion prompt: {PROMPT_VERSION} (model={model})")
    prompt = f"""Analyze this research query and extract KEY CONCEPTS with expansions.

CONCEPT EXTRACTION:
- Extract meaningful research concepts (single words OR multi-word phrases)
- Group related words into phrases (e.g., "air pollution" not "air" + "pollution")
- IGNORE stopwords (a, the, in, of, for, with, etc.) - do NOT list these as concepts
- Focus on domain-specific, searchable terms

For each key concept:
1. Rate its IMPORTANCE (0.0-1.0) - how critical is this concept to answering the research question?
2. List SYNONYMS (direct terminology alternatives, be generous - these are safe for search)
3. List RELATED TERMS (conceptually related but may drift topic, be conservative - max 2-3)
4. List FOUNDATIONAL_WORKS: acronyms, specific names, or exact titles of seminal/foundational papers that researchers commonly cite when discussing this concept. Include the full title if it's a well-known paper (e.g., "Attention Is All You Need" for transformers). This helps find original papers that started a research area.

Return JSON:
{{
  "concepts": [
    {{
      "term": "extracted concept",
      "importance": 0.9,
      "synonyms": ["syn1", "syn2", ...],
      "related": ["rel1", "rel2"],
      "foundational_works": ["DDPM", "Denoising Diffusion Probabilistic Models"]
    }}
  ]
}}

IMPORTANT for foundational_works:
- Do NOT include annotations like "(DDPM)" or "(Ho et al., 2020)" - just the paper title or acronym
- Each entry should be a CLEAN, SEARCHABLE string
- BAD: "Denoising Diffusion Probabilistic Models (DDPM)"
- GOOD: "Denoising Diffusion Probabilistic Models" (separate) and "DDPM" (separate)

Examples of foundational_works:
- "diffusion models" -> ["DDPM", "Denoising Diffusion Probabilistic Models", "score-based generative models"]
- "transformer architecture" -> ["Attention Is All You Need", "BERT", "GPT"]
- "reinforcement learning" -> ["Q-learning", "policy gradient", "DQN"]
- "word embeddings" -> ["Word2Vec", "GloVe"]

Query: {query_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research query analyzer. Return ONLY valid JSON, no explanation or text before/after. The JSON must have a 'concepts' array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for more consistent JSON output
                timeout=30.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle potential markdown code blocks (```json ... ``` or just ``` ... ```)
            if content.startswith("```"):
                lines = content.split("\n")
                # Skip first line (```json or ```) and last line if it's closing ```
                start_idx = 1
                end_idx = len(lines)
                if lines[-1].strip() in ("```", ""):
                    end_idx = -1
                content = "\n".join(lines[start_idx:end_idx])

            # Also handle case where content is wrapped in single backticks
            content = content.strip("`").strip()

            data = json.loads(content)
            if isinstance(data, dict) and "concepts" in data:
                return data
            return {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse expansion JSON: {e}")
            return {}
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"Query expansion failed: {e}")
            return {}

    return {}


def _parse_concepts(data: Dict[str, Any]) -> List[ExpandedConcept]:
    """Parse raw JSON data into ExpandedConcept objects."""
    concepts = []
    for c in data.get("concepts", []):
        if not isinstance(c, dict):
            continue
        term = c.get("term", "")
        if not term:
            continue
        concepts.append(ExpandedConcept(
            term=str(term).strip(),
            importance=float(c.get("importance", 0.5)),
            synonyms=[str(s).strip() for s in c.get("synonyms", []) if s],
            related_terms=[str(r).strip() for r in c.get("related", []) if r],
            foundational_works=[str(f).strip() for f in c.get("foundational_works", []) if f],
        ))
    return concepts


def _build_legacy_expansion_terms(concepts: List[ExpandedConcept]) -> List[str]:
    """Build flat list of expansion terms for backward compatibility."""
    terms = []
    for c in concepts:
        # Add foundational works first (highest priority - seminal paper names)
        terms.extend(c.foundational_works)
        # Add synonyms (high priority)
        terms.extend(c.synonyms)
        # Add related terms
        terms.extend(c.related_terms)
    # Dedupe while preserving order
    seen = set()
    result = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            result.append(t)
    return result


def expand_query(conn: Connection, query_text: str) -> QueryExpansion:
    """Main entry point for query expansion.

    Returns QueryExpansion with structured concepts and legacy compatibility fields.
    Uses caching to avoid redundant LLM calls.

    Note: Query is normalized (question prefixes stripped) before processing
    to ensure consistent results for question-style vs topic-style queries.
    """
    # Normalize query to strip question prefixes for consistent processing
    normalized_query = normalize_query_for_expansion(query_text)
    query_hash = compute_query_hash(query_text)  # Also uses normalization internally
    original_terms = _tokenize_query(normalized_query)

    # Check cache first
    cached = get_cached_expansion(conn, query_hash)
    if cached is not None:
        logger.info(f"Using cached expansion for query hash: {query_hash[:16]}...")

        # Handle legacy cache format
        if "legacy_terms" in cached:
            return QueryExpansion(
                original_query=normalized_query,
                concepts=[],
                query_hash=query_hash,
                original_terms=original_terms,
                expansion_terms=cached["legacy_terms"],
            )

        # Parse structured format
        concepts = _parse_concepts(cached)
        expansion_terms = _build_legacy_expansion_terms(concepts)
        return QueryExpansion(
            original_query=normalized_query,
            concepts=concepts,
            query_hash=query_hash,
            original_terms=original_terms,
            expansion_terms=expansion_terms,
        )

    # Generate new structured expansion using normalized query
    expansion_data = generate_structured_expansion(normalized_query)
    concepts = _parse_concepts(expansion_data)
    expansion_terms = _build_legacy_expansion_terms(concepts)

    logger.info(f"Generated {len(concepts)} concepts with {len(expansion_terms)} expansion terms")

    # Cache the result
    if expansion_data:
        cache_expansion(conn, query_hash, normalized_query, expansion_data, MODEL_VERSION)

    return QueryExpansion(
        original_query=normalized_query,
        concepts=concepts,
        query_hash=query_hash,
        original_terms=original_terms,
        expansion_terms=expansion_terms,
    )
