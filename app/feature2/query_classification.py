"""
Query classification module for methodological query-aware ranking.

This module classifies research queries into types (methodological, topical,
empirical, mixed) using LLM. Results are cached by query hash to avoid
redundant LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .query_expansion import compute_query_hash, normalize_query_for_expansion
from .domain_filters import detect_domain

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class QueryType(Enum):
    """Types of research queries."""
    METHODOLOGICAL = "methodological"  # Seeking methods, techniques, approaches
    TOPICAL = "topical"                # Seeking papers on a topic/domain
    EMPIRICAL = "empirical"            # Seeking specific findings or effects
    MIXED = "mixed"                    # Combination or unclear


class QuerySpecificity(Enum):
    """Specificity level of research queries."""
    BROAD = "broad"      # General research area (e.g., "machine learning", "psychology")
    SPECIFIC = "specific"  # Focused topic or technique (e.g., "dropout regularization", "BERT fine-tuning")


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_hash: str
    query_type: QueryType
    confidence: float
    methodological_indicators: List[str] = field(default_factory=list)
    domain_keywords: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # Detected domain (e.g., "civil_engineering")
    # Field maturity detection for dynamic cutoffs
    is_emerging_field: bool = False  # True if field is < 15 years old
    field_emergence_year: Optional[int] = None  # Approximate year field emerged (e.g., 2017 for GenAI)
    # Query specificity for temporal map flow
    query_specificity: QuerySpecificity = QuerySpecificity.BROAD
    detected_topic_id: Optional[str] = None  # Primary topic for specific queries


def get_cached_classification(conn: Connection, query_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached query classification if available."""
    try:
        row = conn.execute(
            text("""
                SELECT query_type, confidence, is_emerging_field, field_emergence_year,
                       query_specificity, detected_topic_id
                FROM query_classification_cache
                WHERE query_hash = :query_hash
            """),
            {"query_hash": query_hash},
        ).mappings().first()
        if row:
            return {
                "query_type": row["query_type"],
                "confidence": float(row["confidence"]),
                "is_emerging_field": bool(row.get("is_emerging_field", False)),
                "field_emergence_year": row.get("field_emergence_year"),
                "query_specificity": row.get("query_specificity", "broad"),
                "detected_topic_id": row.get("detected_topic_id"),
            }
        return None
    except Exception as e:
        logger.warning(f"Failed to retrieve cached classification: {e}")
        return None


def cache_classification(
    conn: Connection,
    query_hash: str,
    query_text: str,
    classification: QueryClassification,
    model_version: str,
) -> None:
    """Store query classification in cache.

    Uses a savepoint to ensure cache failures don't abort the parent transaction.
    """
    try:
        # Use savepoint for atomic cache operation - if it fails, we rollback
        # only this operation and continue with the parent transaction
        conn.execute(text("SAVEPOINT cache_classification_sp"))
        conn.execute(
            text("""
                INSERT INTO query_classification_cache
                    (query_hash, query_text, query_type, confidence, model_version,
                     is_emerging_field, field_emergence_year, query_specificity, detected_topic_id)
                VALUES (:query_hash, :query_text, :query_type, :confidence, :model_version,
                        :is_emerging_field, :field_emergence_year, :query_specificity, :detected_topic_id)
                ON CONFLICT (query_hash) DO UPDATE SET
                    query_type = EXCLUDED.query_type,
                    confidence = EXCLUDED.confidence,
                    model_version = EXCLUDED.model_version,
                    is_emerging_field = EXCLUDED.is_emerging_field,
                    field_emergence_year = EXCLUDED.field_emergence_year,
                    query_specificity = EXCLUDED.query_specificity,
                    detected_topic_id = EXCLUDED.detected_topic_id,
                    created_at = now()
            """),
            {
                "query_hash": query_hash,
                "query_text": query_text,
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "model_version": model_version,
                "is_emerging_field": classification.is_emerging_field,
                "field_emergence_year": classification.field_emergence_year,
                "query_specificity": classification.query_specificity.value,
                "detected_topic_id": classification.detected_topic_id,
            },
        )
        conn.execute(text("RELEASE SAVEPOINT cache_classification_sp"))
    except Exception as e:
        # Rollback to savepoint to keep parent transaction valid
        try:
            conn.execute(text("ROLLBACK TO SAVEPOINT cache_classification_sp"))
        except Exception:
            pass  # Savepoint might not exist if error was before it
        logger.warning(f"Failed to cache classification: {e}")


def generate_query_classification_llm(query_text: str) -> Dict[str, Any]:
    """Call LLM to classify the query type.

    Returns dict with query_type, confidence, and indicators.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, defaulting to MIXED")
        return {"query_type": "mixed", "confidence": 0.5}

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    prompt = f"""Classify this research query into ONE of four types:

1. METHODOLOGICAL: Seeking methods, techniques, or approaches for doing research
   - Indicators: "methods for", "how to measure", "approaches to", "techniques for", "estimation of", "methodology"
   - Example: "policy evaluation methods in economics", "causal inference techniques"

2. TOPICAL: Seeking papers on a subject or domain
   - Indicators: General subject exploration, domain names
   - Example: "climate change and agriculture", "machine learning in healthcare"

3. EMPIRICAL: Seeking specific findings, effects, or evidence
   - Indicators: "effect of", "impact of", "relationship between", "does X cause Y"
   - Example: "effect of minimum wage on employment", "impact of education on earnings"

4. MIXED: Combination of above or unclear intent

ALSO determine:

A) Is this an EMERGING FIELD (became a distinct research area within the last ~15 years)?
- EMERGING examples: "generative AI/LLMs" (emerged ~2017), "diffusion models" (emerged ~2020), "graph neural networks" (emerged ~2016)
- ESTABLISHED examples: "structural dynamics", "economics", "machine learning" (general), "biology"
- If emerging, provide the approximate year the field emerged

B) Query SPECIFICITY - is this query BROAD or SPECIFIC?

The key test: Does the query contain a NAMED THING (proper noun, acronym, specific algorithm/model name)?

SPECIFIC = Query contains a NAMED algorithm, model, tool, or technique:
  - Named algorithms/architectures: GAN, LSTM, CNN, ResNet, BERT, GPT, AlphaFold, YOLO, U-Net
  - Named tools/systems: CRISPR, channelrhodopsin, Kalman filter, PageRank
  - Named methods: dropout, backpropagation, Adam optimizer, k-means
  - If you can point to ONE paper/system that introduced it → SPECIFIC

BROAD = Query describes a FIELD or CATEGORY (no specific named technique):
  - Research fields: "machine learning", "computer vision", "drug discovery", "climate change"
  - Field intersections: "deep learning for medical imaging", "NLP for healthcare"
  - General approaches: "neural networks", "optimization", "image classification"
  - If many DIFFERENT named techniques could solve it → BROAD

Analyze the query and return JSON:
{{
  "query_type": "methodological|topical|empirical|mixed",
  "confidence": 0.0-1.0,
  "methodological_indicators": ["list", "of", "method-seeking", "words"],
  "domain_keywords": ["subject", "area", "terms"],
  "is_emerging_field": true/false,
  "field_emergence_year": 2017,
  "query_specificity": "broad|specific"
}}

Notes:
- field_emergence_year should only be set if is_emerging_field is true
- query_specificity helps determine if user needs subtopic suggestions (broad) or can go directly to temporal map (specific)

Query: {query_text}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You are a research query classifier. Return ONLY valid JSON, no explanation or text before/after."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent JSON output
                timeout=30.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks (```json ... ``` or just ``` ... ```)
            if content.startswith("```"):
                lines = content.split("\n")
                start_idx = 1
                end_idx = len(lines)
                if lines[-1].strip() in ("```", ""):
                    end_idx = -1
                content = "\n".join(lines[start_idx:end_idx])

            # Also handle case where content is wrapped in single backticks
            content = content.strip("`").strip()

            data = json.loads(content)
            if isinstance(data, dict) and "query_type" in data:
                # Validate query_type
                qt = data.get("query_type", "mixed").lower()
                if qt not in ("methodological", "topical", "empirical", "mixed"):
                    qt = "mixed"
                data["query_type"] = qt
                # Clamp confidence
                data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                # Validate emerging field data
                data["is_emerging_field"] = bool(data.get("is_emerging_field", False))
                emergence_year = data.get("field_emergence_year")
                if data["is_emerging_field"] and emergence_year is not None:
                    try:
                        data["field_emergence_year"] = int(emergence_year)
                    except (ValueError, TypeError):
                        data["field_emergence_year"] = None
                else:
                    data["field_emergence_year"] = None
                # Validate query specificity
                specificity = data.get("query_specificity", "broad").lower()
                if specificity not in ("broad", "specific"):
                    specificity = "broad"
                data["query_specificity"] = specificity
                return data
            return {"query_type": "mixed", "confidence": 0.5, "query_specificity": "broad"}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classification JSON: {e}")
            return {"query_type": "mixed", "confidence": 0.5}
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"Query classification failed: {e}")
            return {"query_type": "mixed", "confidence": 0.5}

    return {"query_type": "mixed", "confidence": 0.5}


def classify_query(conn: Connection, query_text: str) -> QueryClassification:
    """Main entry point for query classification.

    Classifies query as methodological, topical, empirical, or mixed.
    Also detects domain (e.g., civil_engineering) for domain-aware filtering.
    Uses caching to avoid redundant LLM calls.

    Note: Query is normalized (question prefixes stripped) before classification
    to ensure consistent results for question-style vs topic-style queries.
    """
    # Normalize query to strip question prefixes for consistent classification
    normalized_query = normalize_query_for_expansion(query_text)
    query_hash = compute_query_hash(query_text)  # Also uses normalization internally

    # Detect domain using vocabulary matching (fast, no LLM)
    detected_domain = detect_domain(normalized_query)
    if detected_domain:
        logger.info(f"Detected domain: {detected_domain}")

    # Check cache first
    cached = get_cached_classification(conn, query_hash)
    if cached is not None:
        is_emerging = cached.get("is_emerging_field", False)
        emergence_year = cached.get("field_emergence_year")
        specificity = cached.get("query_specificity", "broad")
        detected_topic = cached.get("detected_topic_id")
        logger.info(
            f"Using cached classification: {cached['query_type']} (conf={cached['confidence']:.2f}), "
            f"emerging={is_emerging}, specificity={specificity}"
        )
        return QueryClassification(
            query_hash=query_hash,
            query_type=QueryType(cached["query_type"]),
            confidence=cached["confidence"],
            domain=detected_domain,  # Always use fresh domain detection
            is_emerging_field=is_emerging,
            field_emergence_year=emergence_year,
            query_specificity=QuerySpecificity(specificity),
            detected_topic_id=detected_topic,
        )

    # Generate new classification using normalized query
    result = generate_query_classification_llm(normalized_query)

    classification = QueryClassification(
        query_hash=query_hash,
        query_type=QueryType(result["query_type"]),
        confidence=result.get("confidence", 0.5),
        methodological_indicators=result.get("methodological_indicators", []),
        domain_keywords=result.get("domain_keywords", []),
        domain=detected_domain,
        is_emerging_field=result.get("is_emerging_field", False),
        field_emergence_year=result.get("field_emergence_year"),
        query_specificity=QuerySpecificity(result.get("query_specificity", "broad")),
        detected_topic_id=None,  # Will be set post-ranking if needed
    )

    emergence_info = ""
    if classification.is_emerging_field:
        emergence_info = f", emerging_field=True, emergence_year={classification.field_emergence_year}"
    logger.info(
        f"Query classified as: {classification.query_type.value} (conf={classification.confidence:.2f}), "
        f"domain={detected_domain}, specificity={classification.query_specificity.value}{emergence_info}"
    )

    # Cache the result
    cache_classification(conn, query_hash, normalized_query, classification, MODEL_VERSION)

    return classification
