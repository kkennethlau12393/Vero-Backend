"""
Subtopic generation service for Feature 2.

This module clusters ranked papers by topic and generates human-readable
subtopic labels for broad queries, allowing users to explore specific areas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MAX_SUBTOPICS = 5
MIN_PAPERS_PER_SUBTOPIC = 2


def get_ranked_papers_with_topics(
    conn: Connection,
    rank_job_id: UUID,
) -> List[Dict[str, Any]]:
    """
    Get ranked papers with their topic information.

    Returns list of dicts with: work_id, rank_index, title, year, primary_topic_id
    """
    rows = conn.execute(
        text("""
            SELECT
                rr.work_id,
                rr.rank_index,
                rr.work_preview_json,
                w.primary_topic_id,
                w.title,
                w.year
            FROM rank_results rr
            JOIN works w ON w.work_id = rr.work_id
            WHERE rr.rank_job_id = :rank_job_id
            ORDER BY rr.rank_index
        """),
        {"rank_job_id": rank_job_id},
    ).mappings().all()

    result = []
    for row in rows:
        result.append({
            "work_id": row["work_id"],
            "rank_index": row["rank_index"],
            "title": row["title"],
            "year": row["year"],
            "primary_topic_id": row["primary_topic_id"],
            "preview": row["work_preview_json"] or {},
        })

    return result


def cluster_by_topic(
    papers: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cluster papers by their primary_topic_id.

    Returns dict: topic_id -> list of papers
    """
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for paper in papers:
        topic_id = paper.get("primary_topic_id")
        if topic_id:
            clusters[topic_id].append(paper)
        else:
            clusters["_unknown"].append(paper)

    return dict(clusters)


def get_topic_display_names(
    conn: Connection,
    topic_ids: List[str],
) -> Dict[str, str]:
    """
    Get display names for OpenAlex topic IDs.

    Note: OpenAlex topics table may not have display names stored locally,
    so we return the topic_id as a fallback.
    """
    if not topic_ids:
        return {}

    # Try to get from openalex_topics table
    try:
        rows = conn.execute(
            text("""
                SELECT topic_id, topic_id as display_name
                FROM openalex_topics
                WHERE topic_id = ANY(:topic_ids)
            """),
            {"topic_ids": topic_ids},
        ).mappings().all()

        return {row["topic_id"]: row["display_name"] for row in rows}
    except Exception:
        return {tid: tid for tid in topic_ids}


def identify_subtopics_from_titles_llm(
    papers: List[Dict[str, Any]],
    query_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use LLM to identify distinct subtopics directly from paper titles.

    This approach doesn't rely on OpenAlex topic IDs - it analyzes the actual
    paper titles to find natural groupings.

    Args:
        papers: List of papers with title, work_id, etc.
        query_text: Original query for context

    Returns:
        List of subtopic dicts with label, description, and work_ids
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, cannot generate subtopics")
        return []

    # Get top papers by rank (most relevant first)
    sorted_papers = sorted(papers, key=lambda p: p.get("rank_index", 999))[:50]

    # Build list of titles with IDs for the LLM
    paper_list = []
    for i, p in enumerate(sorted_papers):
        paper_list.append({
            "id": i,
            "title": (p.get("title") or "Untitled")[:200],  # Truncate long titles
            "work_id": p.get("work_id"),
        })

    query_context = f'Research area: "{query_text}"' if query_text else ""

    # Extract query words to help LLM avoid repeating them
    query_words_list = query_text.split() if query_text else []
    query_words_str = ", ".join(f'"{w}"' for w in query_words_list) if query_words_list else "(none)"

    prompt = f"""Cluster these papers into exactly 4-5 distinct subtopics. You MUST produce at least 4.
{query_context}

Papers:
{json.dumps([{"id": p["id"], "title": p["title"]} for p in paper_list], indent=2)}

CRITICAL: STAY ON TOPIC
The query is: "{query_text or 'research papers'}"
ALL subtopics must be SPECIFIC ASPECTS of this query - not tangentially related topics.

TOPIC FOCUS TEST: For each subtopic, ask yourself:
- Does this subtopic represent a SPECIFIC ANGLE on the query's main focus?
- Or is this subtopic about the GENERAL FIELD that the query happens to be part of?

If the query has multiple keywords (e.g., "X Y Z"), subtopics should be about the INTERSECTION of those concepts, not just about X, Y, or Z separately.

If a paper is about the query's GENERAL field but not the query's SPECIFIC focus, either:
1. Exclude it from subtopics, OR
2. Place it in a subtopic that captures HOW it relates to the query's specific focus

LABEL RULES:
1. Labels must name SPECIFIC sub-aspects of the query — not generic categories or tangential topics
2. Extract NAMED ENTITIES from the paper titles: specific algorithm names, named methods, material names, acronyms, named frameworks, specific phenomena
3. NEVER use pattern "[Field] + [Generic Word]" where generic words include: methods, techniques, applications, approaches, systems, frameworks, models, mechanisms, learning, analysis
4. ABSOLUTE RULE: Labels must NOT contain ANY of these query words: {query_words_str}
   - If you find yourself writing a label that contains any query word, STOP and rewrite it
   - Instead, find specific named methods, techniques, or phenomena from the paper titles
   - Example: If query is "gene expression", WRONG: "Gene Expression Profiling", RIGHT: "RNA-Seq Differential Analysis"
5. Each subtopic must be ORTHOGONAL - covering a fundamentally different angle (e.g., different methods, different applications, different scales)
6. Subtopics should have <20% word overlap with each other

LABEL QUALITY TEST - Before using a label, ask:
- Does this label contain at least one NAMED ENTITY (specific algorithm, method, material, framework)?
- Would searching this label return focused results different from the original query?
- Is this label specific to THIS query, or could it apply to many different queries?

If a label is just a general concept (like "Optimization" or "Attention Mechanisms"), make it specific by adding:
- The specific variant/algorithm name from the papers
- The specific application domain
- The specific technique being used

Return JSON only:
{{"subtopics": [{{"label": "Specific Subarea Name", "description": "One sentence description", "paper_ids": [0,1,2]}}]}}

REQUIREMENTS:
- EXACTLY 4-5 subtopics (never fewer than 4)
- Minimum 2 papers per subtopic
- Every paper assigned to EXACTLY ONE subtopic (no duplicates)
- Labels must be specific enough that someone could search for them
- Labels must NOT repeat the original query words
- ALL subtopics must be specific aspects of the QUERY FOCUS, not tangential topics"""

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research librarian expert at categorizing academic papers. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=60.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                start_idx = 1
                end_idx = len(lines)
                if lines[-1].strip().startswith("```"):
                    end_idx = -1
                content = "\n".join(lines[start_idx:end_idx])

            # Extract just the JSON object (LLM sometimes adds text after)
            # Find the outermost { } pair
            start = content.find("{")
            if start >= 0:
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

            result = json.loads(content)
            subtopics_raw = result.get("subtopics", [])

            if not subtopics_raw:
                logger.warning("LLM returned no subtopics")
                return []

            # Convert paper IDs back to work_ids and filter banned labels
            subtopics = []
            banned_count = 0
            seen_work_ids = set()  # Track papers already assigned to prevent duplicates
            for st in subtopics_raw:
                paper_ids = st.get("paper_ids", [])
                work_ids = []
                for pid in paper_ids:
                    if isinstance(pid, int) and 0 <= pid < len(paper_list):
                        wid = paper_list[pid]["work_id"]
                        # Only add if not already assigned to another subtopic
                        if wid not in seen_work_ids:
                            work_ids.append(wid)
                            seen_work_ids.add(wid)

                if len(work_ids) >= MIN_PAPERS_PER_SUBTOPIC:
                    raw_label = st.get("label", "Subtopic")
                    # Check for banned patterns - SKIP subtopics with generic labels
                    if _has_banned_pattern(raw_label, query_text):
                        logger.info(f"SKIPPING subtopic with banned pattern: '{raw_label}'")
                        banned_count += 1
                        continue  # Skip this subtopic entirely
                    subtopics.append({
                        "label": raw_label,
                        "description": st.get("description", ""),
                        "work_ids": work_ids,
                    })

            # If too many banned (>50%), retry
            total_valid = len(subtopics) + banned_count
            if total_valid > 0 and banned_count / total_valid > 0.5:
                logger.warning(f"Too many banned labels ({banned_count}/{total_valid}), valid: {[s['label'] for s in subtopics]}, retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue

            # Need at least 2 good subtopics
            if len(subtopics) < 2 and attempt < MAX_RETRIES - 1:
                logger.warning(f"Only {len(subtopics)} valid subtopics, retrying...")
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue

            # Sort by paper count
            subtopics.sort(key=lambda s: -len(s["work_ids"]))
            return subtopics[:MAX_SUBTOPICS]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse subtopic JSON: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break
        except Exception as e:
            logger.warning(f"Subtopic identification failed: {e}")
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break

    return []


BANNED_LABEL_WORDS = {
    # Generic methodology words
    "algorithms", "techniques", "methods", "approaches", "architectures",
    "frameworks", "systems", "applications", "integration",
    "policies", "impacts", "optimization", "innovations",
    "studies", "research", "analysis", "fundamentals", "advances",
    "developments", "implementations", "effects", "solutions",
    # Additional generic words
    "technologies", "technology", "policy", "concepts", "principles",
    "strategies", "overview", "perspectives",
    "challenges", "opportunities", "trends", "future", "emerging",
    "oriented",
    # Data/platform generic words
    "databases", "datasets",
    "resources", "platforms",
    "infrastructure", "repositories",
    # Vague modifier words that create generic labels
    "foundational", "core", "general", "basic", "fundamental",
    "advanced", "novel", "modern", "classical", "traditional",
}


def _clean_label(label: str, query_text: Optional[str] = None) -> str:
    """Light cleaning - just log, don't aggressively modify."""
    # Don't aggressively clean - just return as-is for now
    # The prompt should do the heavy lifting
    return label


def _has_banned_pattern(label: str, query_text: Optional[str] = None) -> bool:
    """Check if label is a generic "[Query] + [Generic]" or "[Generic] of [Query]" pattern.

    Only flags labels where the MAJORITY of content words are generic/query words,
    meaning the label doesn't add specific information.

    Flags: "Deep Learning Applications", "Machine Learning Methods"
    Does NOT flag: "Convolutional Neural Networks", "Perovskite Solar Cells",
                   "Grid-Scale Battery Storage", "Fragment-Based Drug Design"
    """
    label_lower = label.lower()
    words = label_lower.replace("&", " ").replace("-", " ").split()
    if not words:
        return False

    connectors = {"of", "and", "for", "in", "on", "the", "a", "an", "to", "with"}
    content_words = [w for w in words if w not in connectors]
    if not content_words:
        return False

    # Count how many content words are banned or from the query
    query_words = set()
    if query_text:
        query_words = set(query_text.lower().replace("-", " ").split()) - connectors

    generic_or_query_count = 0
    for word in content_words:
        is_banned = word in BANNED_LABEL_WORDS or word.rstrip("s") in BANNED_LABEL_WORDS
        is_query = word in query_words
        if is_banned or is_query:
            generic_or_query_count += 1

    # Only flag if ALL or nearly all content words are generic/query
    # This means labels like "Perovskite Solar Cells" (1/3 generic) pass,
    # but "Deep Learning Applications" (2/3 generic+query, 1/3 banned) fails
    ratio = generic_or_query_count / len(content_words)
    return ratio >= 0.8


def _label_to_query(label: str, original_query: Optional[str] = None) -> str:
    """Convert a subtopic label to a search query.

    Takes a label like "BERT & Transformer Pre-training" and converts it
    to a search-friendly query like "BERT transformer pre-training".

    Always appends original_query context unless label is already long
    and contains query-specific terms.
    """
    # Replace & with space, normalize whitespace
    query = label.replace("&", " ")
    # Collapse multiple spaces to single space
    query = " ".join(query.split())

    if original_query:
        query_words = set(original_query.lower().split())
        label_words = set(query.lower().split())

        # Check if label already contains significant query context
        overlap = len(query_words & label_words)
        has_good_overlap = overlap >= min(2, len(query_words))

        # Append original query if label doesn't have enough context
        # This ensures drill-down is always at least as specific as original
        if not has_good_overlap:
            query = f"{query} {original_query}"

    return query


def _fallback_subtopics(
    clusters: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Generate fallback subtopics using topic IDs as labels."""
    result = []
    for topic_id, papers in clusters.items():
        if len(papers) < MIN_PAPERS_PER_SUBTOPIC:
            continue
        result.append({
            "topic_id": topic_id,
            "label": topic_id.replace("_", " ").title() if "_" in topic_id else topic_id,
            "description": f"Papers in topic cluster {topic_id}",
        })
    return sorted(result, key=lambda s: -len(clusters.get(s["topic_id"], [])))[:MAX_SUBTOPICS]


def generate_subtopics(
    engine: Engine,
    *,
    rank_job_id: UUID,
    query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate subtopics for a ranked query result.

    Uses LLM to identify distinct research themes directly from paper titles,
    rather than relying on OpenAlex topic IDs which can be too broad.

    Args:
        engine: Database engine
        rank_job_id: ID of the rank job to analyze
        query_text: Original query text for context

    Returns:
        Dict with rank_job_id, query_specificity, and subtopics list
    """
    with engine.connect() as conn:
        # Get ranked papers with topic info
        papers = get_ranked_papers_with_topics(conn, rank_job_id)

        if not papers:
            logger.warning(f"No papers found for rank_job_id {rank_job_id}")
            return {
                "rank_job_id": str(rank_job_id),
                "query_specificity": "broad",
                "subtopics": [],
            }

        logger.info(f"Identifying subtopics from {len(papers)} papers via LLM")

        # Use LLM to identify subtopics directly from paper titles
        llm_subtopics = identify_subtopics_from_titles_llm(papers, query_text)

        if not llm_subtopics:
            # Fallback to topic-based clustering if LLM fails
            logger.warning("LLM subtopic identification failed, using fallback")
            clusters = cluster_by_topic(papers)
            llm_subtopics = _fallback_subtopics(clusters)
            # Add work_ids from clusters
            for st in llm_subtopics:
                topic_id = st.get("topic_id")
                if topic_id and topic_id in clusters:
                    st["work_ids"] = [p["work_id"] for p in clusters[topic_id]]

        # Build response from LLM subtopics
        subtopics = []
        for i, st in enumerate(llm_subtopics):
            work_ids = st.get("work_ids", [])
            label = st.get("label", f"Subtopic {i+1}")
            # Generate drill-down query from label
            # Clean up the label for use as a search query:
            # - Replace & with space (so "BERT & GPT" becomes "BERT GPT")
            # - Keep hyphens (they're often part of terms like "pre-training")
            drill_down_query = _label_to_query(label, query_text)

            subtopics.append({
                "subtopic_id": f"subtopic_{i}",
                "label": label,
                "description": st.get("description", ""),
                "representative_work_ids": work_ids[:5],  # Top 5 as representatives
                "paper_count": len(work_ids),
                "drill_down_query": drill_down_query,
            })

        return {
            "rank_job_id": str(rank_job_id),
            "query_specificity": "broad",
            "subtopics": subtopics,
        }
