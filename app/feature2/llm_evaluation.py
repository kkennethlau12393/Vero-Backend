"""
LLM-generated evaluation paragraphs for ranked papers.

Each paper in the ranked results gets a 3-5 sentence evaluative paragraph
explaining WHY it matters for the user's query — not a generic summary,
but a query-contextual assessment of significance, contribution, and role.

Uses Groq/Llama 4 Maverick with batched calls and DB caching.
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

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

EVAL_MODEL_VERSION = "eval-v3"
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

    papers_for_prompt = []
    for p in batch:
        papers_for_prompt.append({
            "id": p["paper_id"],
            "title": p["title"],
            "year": p.get("year"),
            "cited_by_count": p.get("cited_by_count", 0),
            "abstract": _truncate_abstract(p.get("abstract", "")),
            "category": p.get("category", "methodology"),
        })
    papers_json = json.dumps(papers_for_prompt, indent=2)

    system_message = (
        "You are an expert academic reviewer, not a summarizer. "
        "When evaluating a paper's significance, you write like a senior researcher "
        "explaining to a colleague why this paper matters for their specific research question. "
        "You are direct, opinionated, and specific. You never write generic descriptions."
    )

    prompt = f"""For each paper below, write an evaluative paragraph (3-5 sentences) explaining its significance FOR THIS SPECIFIC QUERY. Return ONLY a JSON object mapping paper ID to evaluation string.

QUERY: {query_text}

YOUR TASK: Evaluate — do NOT summarize.
- WHY does this paper matter for someone researching this exact query?
- What specific contribution did it make that the reader needs to know about?
- What did it enable or change in the field? What would be missing without it?
- If the paper is only tangentially related, say so directly — explain the connection honestly.

BAD (summary-style, generic — DO NOT write like this):
- "This foundational paper introduced the Transformer architecture, revolutionizing the field of deep learning by demonstrating that attention mechanisms alone can achieve state-of-the-art results."
- "This work proposed a novel approach that has been influential in the field."
- "This paper presents an important contribution to the area of neural networks."
Why bad: Just restates what the paper did. Says "revolutionizing" and "influential" without substance. Could describe any paper.

GOOD (evaluative, query-specific, opinionated — write like this):
- "This is the work that proved attention alone — without recurrence or convolution — is sufficient for sequence modeling. Every transformer variant in this query's scope descends from this architectural decision. The multi-head attention mechanism it introduced remains the core building block that subsequent work either refines or extends."
- "This paper's contribution to the query is indirect but important: it established that features learned by deep networks transfer across tasks, which is the theoretical basis for the pre-trained transformer paradigm. Without this insight, the fine-tuning approach that defines modern NLP would lack empirical justification."
Why good: States a clear judgment. Names the specific technical contribution. Connects it to the query. Has an opinion about the paper's role.

PAPERS:
{papers_json}

OUTPUT (JSON only — paper_id maps to evaluation string):
{{"paper_id": "evaluation paragraph", ...}}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=4096,
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
                eval_text = response_map.get(pid, "")
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
