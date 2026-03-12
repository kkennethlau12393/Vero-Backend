"""
Query decomposition module for structured query analysis.

Decomposes natural language queries into structured components
(topic, domain, aspect) to enable intersection-aware retrieval
and scoring. Also supports decomposing paper metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
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
MODEL_VERSION = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DECOMPOSITION_VERSION = "decomp_v3.1"


DECOMPOSITION_PROMPT = """You are a research query analyzer. Decompose this query into structured components.

Query: "{user_query}"

Return JSON only:
{{
  "topic": "primary research field or method",
  "topic_aliases": ["2-4 common abbreviations, acronyms, or synonyms for the topic"],
  "domain": "application domain or null if purely theoretical",
  "domain_aliases": ["2-4 alternative names for the domain, or empty if null"],
  "aspect": "specific technique/subtopic or null if general",
  "aspect_aliases": ["alternative names for the aspect, or empty if null"],
  "suggested_specificity": "broad | balanced | specific",
  "intent": "method_in_domain | cross_domain | survey | single_topic",
  "reasoning": "one sentence explaining your decomposition"
}}

Intent types:
- method_in_domain: applying a method/technique to a specific domain (e.g., "GNN for drug discovery")
- cross_domain: comparing or combining two independent fields (e.g., "transfer learning between NLP and computer vision")
- survey: looking for overview/review of a topic (e.g., "survey of attention mechanisms")
- single_topic: focused on one topic without intersection (e.g., "transformer architectures")

CONSTRAINT: For cross_domain intent, topic and domain must BOTH be non-null. Assign the method/technique to topic and the application area to domain. If both are equal fields, assign the first mentioned to topic and the second to domain.

Examples:
- "NLP for law" → topic: "natural language processing", topic_aliases: ["NLP", "computational linguistics", "text mining"], domain: "law", domain_aliases: ["legal", "legal AI", "legaltech"], aspect: null, aspect_aliases: [], intent: "method_in_domain"
- "transformer architectures" → topic: "transformer architectures", topic_aliases: ["transformers", "attention models"], domain: null, domain_aliases: [], aspect: null, aspect_aliases: [], intent: "single_topic"
- "BERT fine-tuning for medical NER" → topic: "BERT fine-tuning", topic_aliases: ["BERT", "language model fine-tuning"], domain: "medical", domain_aliases: ["healthcare", "clinical", "biomedical"], aspect: "named entity recognition", aspect_aliases: ["NER", "entity extraction"], intent: "method_in_domain"
- "transfer learning between NLP and computer vision" → topic: "transfer learning", topic_aliases: ["TL", "domain adaptation", "knowledge transfer"], domain: "NLP and computer vision", domain_aliases: ["natural language processing and CV", "text and image models"], aspect: null, aspect_aliases: [], intent: "cross_domain"
- "bridging genomics and machine learning" → topic: "machine learning", topic_aliases: ["ML", "statistical learning"], domain: "genomics", domain_aliases: ["genome analysis", "genetic data analysis"], aspect: null, aspect_aliases: [], intent: "cross_domain"
- "influence of market sentiment on venture capital decision making" → topic: "venture capital decision making", topic_aliases: ["VC decision making", "VC investment decisions", "venture capital evaluation"], domain: "market sentiment and macroeconomics", domain_aliases: ["investor sentiment", "economic conditions", "market mood"], aspect: "causal influence", aspect_aliases: ["effect", "impact", "role of"], intent: "method_in_domain"
- "how climate change affects agricultural supply chains" → topic: "agricultural supply chains", topic_aliases: ["food supply chains", "agricultural logistics", "crop distribution"], domain: "climate change", domain_aliases: ["global warming", "climate impacts", "environmental change"], aspect: "effects and adaptation", aspect_aliases: ["impacts", "resilience", "vulnerability"], intent: "method_in_domain"
- "comparing deep learning and traditional methods for anomaly detection" → topic: "anomaly detection", topic_aliases: ["outlier detection", "anomaly identification"], domain: null, domain_aliases: [], aspect: "deep learning vs traditional methods", aspect_aliases: ["DL comparison", "neural vs classical"], intent: "single_topic"
- "ethical implications of facial recognition in law enforcement" → topic: "facial recognition", topic_aliases: ["face detection", "biometric identification"], domain: "law enforcement", domain_aliases: ["policing", "criminal justice", "public safety"], aspect: "ethical implications", aspect_aliases: ["ethics", "privacy concerns", "civil liberties"], intent: "method_in_domain"
"""


PDF_DECOMPOSITION_PROMPT = """Given this research paper's metadata, extract structured query components.

Title: "{title}"
Abstract: "{abstract}"
Authors: {authors}
Year: {year}
Venue: "{venue}"

Return JSON only:
{{
  "topic": "primary research field or method of this paper",
  "topic_aliases": ["2-4 common abbreviations, acronyms, or synonyms"],
  "domain": "application domain or null",
  "domain_aliases": ["alternative names for the domain, or empty if null"],
  "aspect": "specific technique/subtopic or null",
  "aspect_aliases": ["alternative names for the aspect, or empty if null"],
  "keywords": ["top 5 keywords for retrieval"],
  "suggested_specificity": "broad | balanced | specific"
}}
"""


def _compute_decomposition_hash(query_text: str) -> str:
    """Compute cache key for decomposition, including version for invalidation."""
    from app.feature2.query_expansion import normalize_query_for_expansion

    normalized = normalize_query_for_expansion(query_text).lower()
    key = f"{DECOMPOSITION_VERSION}:{normalized}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _get_cached_decomposition(conn: Connection, query_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached decomposition if available."""
    try:
        row = conn.execute(
            text("""
                SELECT decomposition
                FROM query_decomposition_cache
                WHERE query_hash = :query_hash
            """),
            {"query_hash": query_hash},
        ).mappings().first()
        if row:
            data = row["decomposition"]
            if isinstance(data, str):
                return json.loads(data)
            return data
        return None
    except Exception as e:
        logger.warning(f"Failed to retrieve cached decomposition: {e}")
        return None


def _cache_decomposition(
    conn: Connection,
    query_hash: str,
    query_text: str,
    decomposition: Dict[str, Any],
) -> None:
    """Store decomposition in cache."""
    try:
        conn.execute(
            text("""
                INSERT INTO query_decomposition_cache (query_hash, query_text, decomposition, model_version)
                VALUES (:query_hash, :query_text, :decomposition, :model_version)
                ON CONFLICT (query_hash) DO UPDATE SET
                    decomposition = EXCLUDED.decomposition,
                    model_version = EXCLUDED.model_version,
                    created_at = now()
            """),
            {
                "query_hash": query_hash,
                "query_text": query_text,
                "decomposition": json.dumps(decomposition),
                "model_version": DECOMPOSITION_VERSION,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to cache decomposition: {e}")


def _call_groq(prompt: str) -> Optional[Dict[str, Any]]:
    """Call Groq LLM and return parsed JSON response."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping decomposition")
        return None

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Extract JSON from possible markdown wrapping
            if content.startswith("```"):
                lines = content.split("\n")
                json_lines = []
                inside = False
                for line in lines:
                    if line.strip().startswith("```"):
                        inside = not inside
                        continue
                    if inside:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            # Find the JSON object
            start = content.find("{")
            if start == -1:
                logger.warning(f"Decomposition response has no JSON: {content[:200]}")
                return None
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
            content = content[start:end]

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse decomposition JSON: {e}")
            return None
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str or "validate" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"Decomposition LLM call failed: {e}")
            return None

    return None


def decompose_query(conn: Connection, query_text: str) -> Dict[str, Any]:
    """Auto-decompose a natural language query into structured components.

    Returns dict with keys: topic, domain, aspect, suggested_specificity, reasoning.
    Falls back to {"topic": query_text, "domain": None, ...} on failure.
    """
    query_hash = _compute_decomposition_hash(query_text)

    # Check cache
    cached = _get_cached_decomposition(conn, query_hash)
    if cached:
        logger.info(f"Decomposition cache hit for '{query_text[:50]}'")
        return cached

    # Call LLM
    prompt = DECOMPOSITION_PROMPT.format(user_query=query_text)
    result = _call_groq(prompt)

    if not result or "topic" not in result:
        # Fallback: treat entire query as topic
        result = {
            "topic": query_text,
            "domain": None,
            "aspect": None,
            "suggested_specificity": "broad",
            "reasoning": "Decomposition failed, using raw query as topic",
        }

    # Only cache successful decompositions (not fallbacks)
    if result.get("domain") is not None or result.get("reasoning", "") != "Decomposition failed, using raw query as topic":
        _cache_decomposition(conn, query_hash, query_text, result)
    logger.info(f"Decomposed '{query_text[:50]}' → topic='{result.get('topic')}', domain='{result.get('domain')}'")
    return result


def decompose_paper(
    conn: Connection,
    title: str,
    abstract: str = "",
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
    venue: str = "",
) -> Dict[str, Any]:
    """Auto-decompose a paper's metadata into structured components.

    Returns dict with keys: topic, domain, aspect, keywords, suggested_specificity.
    """
    # Use title as cache key for paper decomposition
    cache_key = f"paper:{title[:100]}"
    query_hash = hashlib.sha256(f"{DECOMPOSITION_VERSION}:{cache_key}".encode("utf-8")).hexdigest()

    cached = _get_cached_decomposition(conn, query_hash)
    if cached:
        return cached

    prompt = PDF_DECOMPOSITION_PROMPT.format(
        title=title,
        abstract=(abstract or "")[:1000],
        authors=(authors or [])[:5],
        year=year or "Unknown",
        venue=venue or "Unknown",
    )
    result = _call_groq(prompt)

    if not result or "topic" not in result:
        result = {
            "topic": title,
            "domain": None,
            "aspect": None,
            "keywords": [],
            "suggested_specificity": "balanced",
        }

    _cache_decomposition(conn, query_hash, title, result)
    return result
