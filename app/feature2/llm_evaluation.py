"""
LLM-generated query-relevance explanations for ranked papers.

Each paper in the ranked results gets a 1-2 sentence explanation of
WHY it appeared in results for this specific query — the topical
connection, not a novelty assessment or impact statement.

Uses Groq/Llama 3.3 70B with batched calls and DB caching.
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

from app.common.id_mapping import IdMapper

from .work_topic_store import WorkForMap

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

EVAL_MODEL_VERSION = "eval-v5"
EVAL_BATCH_SIZE = 8
MAX_PARALLEL_BATCHES = 4
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5


def get_cached_evaluations(
    conn: Connection,
    query_hash: str,
    paper_ids: List[str],
    model_version: str = EVAL_MODEL_VERSION,
) -> Dict[str, str]:
    """Retrieve cached evaluation paragraphs for given papers and query."""
    if not paper_ids:
        return {}

    try:
        rows = conn.execute(
            text("""
                SELECT paper_id, evaluation_text
                FROM llm_evaluation_cache
                WHERE query_hash = :query_hash
                AND paper_id = ANY(:paper_ids)
                AND model_version = :model_version
            """),
            {"query_hash": query_hash, "paper_ids": paper_ids, "model_version": model_version},
        ).mappings().all()

        return {row["paper_id"]: row["evaluation_text"] for row in rows}
    except Exception as e:
        logger.warning(f"Failed to retrieve cached evaluations: {e}")
        return {}


def cache_evaluations(
    conn: Connection,
    query_hash: str,
    evaluations: Dict[str, str],
    model_version: str = EVAL_MODEL_VERSION,
) -> None:
    """Store evaluation paragraphs in cache."""
    if not evaluations:
        return

    try:
        for paper_id, eval_text in evaluations.items():
            conn.execute(
                text("""
                    INSERT INTO llm_evaluation_cache (paper_id, query_hash, evaluation_text, model_version)
                    VALUES (:paper_id, :query_hash, :evaluation_text, :model_version)
                    ON CONFLICT (paper_id, query_hash) DO UPDATE SET
                        evaluation_text = EXCLUDED.evaluation_text,
                        model_version = EXCLUDED.model_version,
                        created_at = now()
                """),
                {
                    "paper_id": paper_id,
                    "query_hash": query_hash,
                    "evaluation_text": eval_text,
                    "model_version": model_version,
                },
            )
    except Exception as e:
        logger.warning(f"Failed to cache evaluations: {e}")


def _truncate_abstract(abstract: str, max_chars: int = 600) -> str:
    """Truncate abstract for prompt, preserving word boundaries."""
    if not abstract or len(abstract) <= max_chars:
        return abstract or ""
    truncated = abstract[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def _generate_evaluation_batch(
    query_text: str,
    batch: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Generate evaluation paragraphs for a batch of papers via single Groq call.

    Returns dict mapping paper_id to evaluation paragraph string.
    """
    if not batch:
        return {}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping evaluations")
        return {}

    client = Groq(api_key=api_key)

    mapper = IdMapper("P")
    papers_for_prompt = []
    for p in batch:
        papers_for_prompt.append({
            "id": mapper.add(p["paper_id"]),
            "title": p["title"],
            "year": p.get("year"),
            "cited_by_count": p.get("cited_by_count", 0),
            "abstract": _truncate_abstract(p.get("abstract", "")),
            "category": p.get("category", "methodology"),
        })
    papers_json = json.dumps(papers_for_prompt, indent=2)

    system_message = (
        "You are a search result annotator. For each paper, write exactly what "
        "topics it covers that match the user's query. Describe WHAT the paper "
        "is about, not what it achieved or why it matters. Use present tense only: "
        "'covers', 'addresses', 'focuses on', 'presents', 'proposes'. "
        "Never use past tense like 'proved', 'established', 'enabled', 'advanced'."
    )

    prompt = f"""For each paper below, write 1 sentence stating what it covers that matches the query. Return ONLY a JSON object mapping paper ID to explanation string.

QUERY: {query_text}

RULES (strict):
1. Describe WHAT the paper covers — its topic, method, or subject matter.
2. State which query term(s) it matches.
3. Use ONLY present tense: "covers", "addresses", "focuses on", "presents", "proposes", "applies", "uses", "models", "studies".
4. If the connection is indirect, say "indirectly related" or "partially relevant".
5. Maximum 1 sentence. No compound sentences joined by "and" or "which".

BANNED (any of these = failure):
- Past tense verbs: "proved", "showed", "established", "enabled", "advanced", "paved", "pioneered", "introduced", "demonstrated", "achieved", "revolutionized", "changed", "contributed", "shifted"
- Impact words: "groundbreaking", "seminal", "influential", "pivotal", "crucial", "profound", "significant", "important", "key contribution", "paradigm shift"
- Phrases: "paving the way", "building upon", "subsequent work", "widely adopted", "state-of-the-art results"

GOOD examples:
- "Covers self-attention for sequence modeling, addressing the 'attention mechanisms' query term."
- "Focuses on pre-trained language representations via masked token prediction, matching the 'transformers NLP' query terms."
- "Studies Li-ion cell aging processes, addressing the 'degradation mechanisms' query term."
- "Proposes a graph convolution method for node-level classification, matching the 'graph neural networks' query term."
- "Partially relevant — covers batch training optimization, not transformer architectures directly."

PAPERS:
{papers_json}

OUTPUT (JSON only — paper_id maps to 1 sentence):
{{"paper_id": "...", ...}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            start = content.find("{")
            if start == -1:
                logger.warning(f"Evaluation response has no JSON: {content[:200]}")
                continue

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
                logger.warning(f"Evaluation response has malformed JSON: {content[:200]}")
                continue

            json_str = content[start:end]
            response_map = json.loads(json_str)

            result = {}
            for p in batch:
                pid = p["paper_id"]
                short_key = mapper.get_short(pid, pid)
                eval_text = response_map.get(short_key, response_map.get(pid, ""))
                if isinstance(eval_text, str) and eval_text.strip():
                    result[pid] = eval_text.strip()

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Evaluation JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        except Exception as e:
            logger.warning(f"Evaluation API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))

    return {}


def generate_evaluations(
    conn: Connection,
    query_text: str,
    query_hash: str,
    paper_items: List[Dict[str, Any]],
    works: Dict[str, WorkForMap],
) -> Dict[str, str]:
    """Generate evaluation paragraphs for ranked papers.

    Checks cache first, then generates missing evaluations in parallel batches.
    Returns dict mapping work_id to evaluation paragraph string.
    """
    if not paper_items:
        return {}

    paper_ids = [item["work_id"] for item in paper_items]

    # Check cache
    cached = get_cached_evaluations(conn, query_hash, paper_ids)
    logger.info(f"Evaluation cache hit: {len(cached)}/{len(paper_ids)} papers")

    # Find papers needing evaluation
    needs_eval = []
    for item in paper_items:
        wid = item["work_id"]
        if wid in cached:
            continue
        w = works.get(wid)
        if not w:
            continue
        needs_eval.append({
            "paper_id": wid,
            "title": w.title or "Untitled",
            "year": w.year,
            "cited_by_count": w.cited_by_count or 0,
            "abstract": getattr(w, 'abstract', None) or "",
            "category": item.get("breakdown", {}).get("paper_type", "methodology"),
        })

    if not needs_eval:
        return cached

    logger.info(f"Generating evaluations for {len(needs_eval)} papers")

    # Split into batches
    batches = []
    for i in range(0, len(needs_eval), EVAL_BATCH_SIZE):
        batches.append(needs_eval[i:i + EVAL_BATCH_SIZE])

    # Generate in parallel
    new_evals: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(_generate_evaluation_batch, query_text, batch): batch
            for batch in batches
        }
        for future in as_completed(future_to_batch):
            try:
                batch_evals = future.result()
                new_evals.update(batch_evals)
            except Exception as e:
                logger.error(f"Evaluation batch failed: {e}")

    logger.info(f"Generated {len(new_evals)} new evaluations")

    # Cache new evaluations
    if new_evals:
        cache_evaluations(conn, query_hash, new_evals)

    # Combine cached and new
    return {**cached, **new_evals}
