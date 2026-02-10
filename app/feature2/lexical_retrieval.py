"""
Lexical retrieval module for the new ranking pipeline.

This module implements Postgres full-text search (FTS) retrieval using the
existing search_tsv GIN index. It retrieves 1,000-2,000 papers using weighted
queries for original terms (high weight) and expansion terms (lower weight),
then partitions results into strong lexical matches (A) and expansion-only
matches (B).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .work_topic_store import WorkStore

logger = logging.getLogger(__name__)

DEFAULT_FTS_LIMIT = 2000
MIN_ABSTRACT_LENGTH = 100


@dataclass
class FTSCandidate:
    """A candidate paper from full-text search."""
    work_id: str
    orig_score: float
    exp_score: float
    combined_score: float
    matches_original: bool
    partition: Literal["A", "B"]


def _sanitize_term(term: str) -> str:
    """Sanitize a term for use in tsquery."""
    # Remove special characters that could break tsquery
    cleaned = re.sub(r'[^\w\s]', '', term)
    cleaned = cleaned.strip()
    # Replace spaces with & for phrase matching
    if ' ' in cleaned:
        parts = [p.strip() for p in cleaned.split() if p.strip()]
        return ' <-> '.join(parts) if parts else ''
    return cleaned


def build_tsquery(terms: List[str], operator: str = "|") -> str:
    """Build a Postgres tsquery from term list.

    Args:
        terms: List of search terms
        operator: Either "|" (OR) or "&" (AND)

    Returns:
        A tsquery string suitable for Postgres FTS
    """
    if not terms:
        return ""

    sanitized = []
    for term in terms:
        clean = _sanitize_term(term)
        if clean:
            # Wrap multi-word terms in parentheses
            if '<->' in clean:
                sanitized.append(f"({clean})")
            else:
                sanitized.append(clean)

    if not sanitized:
        return ""

    joiner = f" {operator} "
    return joiner.join(sanitized)


def execute_fts_search(
    conn: Connection,
    original_terms: List[str],
    expansion_terms: List[str],
    limit: int = DEFAULT_FTS_LIMIT,
) -> List[FTSCandidate]:
    """Execute weighted FTS query against works table.

    Original terms are weighted 2x higher than expansion terms.
    Returns candidates with scores and partition assignment.
    """
    if not original_terms and not expansion_terms:
        return []

    # Build tsqueries
    orig_query = build_tsquery(original_terms, "|")
    exp_query = build_tsquery(expansion_terms, "|")

    if not orig_query and not exp_query:
        return []

    # Build the SQL query with conditional scoring
    # We need to handle cases where one query might be empty
    sql_parts = []
    params: Dict[str, any] = {"limit": limit}

    if orig_query and exp_query:
        sql = """
            SELECT
                work_id,
                ts_rank(search_tsv, to_tsquery('english', :orig_query)) * 2.0 AS orig_score,
                ts_rank(search_tsv, to_tsquery('english', :exp_query)) AS exp_score,
                (search_tsv @@ to_tsquery('english', :orig_query)) AS matches_original
            FROM works
            WHERE search_tsv @@ (
                to_tsquery('english', :orig_query) || to_tsquery('english', :exp_query)
            )
            AND is_retracted = false
            ORDER BY (
                ts_rank(search_tsv, to_tsquery('english', :orig_query)) * 2.0 +
                ts_rank(search_tsv, to_tsquery('english', :exp_query))
            ) DESC
            LIMIT :limit
        """
        params["orig_query"] = orig_query
        params["exp_query"] = exp_query
    elif orig_query:
        sql = """
            SELECT
                work_id,
                ts_rank(search_tsv, to_tsquery('english', :orig_query)) * 2.0 AS orig_score,
                0.0 AS exp_score,
                true AS matches_original
            FROM works
            WHERE search_tsv @@ to_tsquery('english', :orig_query)
            AND is_retracted = false
            ORDER BY ts_rank(search_tsv, to_tsquery('english', :orig_query)) DESC
            LIMIT :limit
        """
        params["orig_query"] = orig_query
    else:
        sql = """
            SELECT
                work_id,
                0.0 AS orig_score,
                ts_rank(search_tsv, to_tsquery('english', :exp_query)) AS exp_score,
                false AS matches_original
            FROM works
            WHERE search_tsv @@ to_tsquery('english', :exp_query)
            AND is_retracted = false
            ORDER BY ts_rank(search_tsv, to_tsquery('english', :exp_query)) DESC
            LIMIT :limit
        """
        params["exp_query"] = exp_query

    try:
        rows = conn.execute(text(sql), params).mappings().all()
    except Exception as e:
        logger.error(f"FTS query failed: {e}")
        return []

    candidates = []
    for row in rows:
        orig_score = float(row["orig_score"] or 0.0)
        exp_score = float(row["exp_score"] or 0.0)
        matches_original = bool(row["matches_original"])

        candidates.append(FTSCandidate(
            work_id=row["work_id"],
            orig_score=orig_score,
            exp_score=exp_score,
            combined_score=orig_score + exp_score,
            matches_original=matches_original,
            partition="A" if matches_original else "B",
        ))

    logger.info(f"FTS retrieved {len(candidates)} candidates")
    return candidates


def partition_candidates(
    candidates: List[FTSCandidate],
) -> Tuple[List[FTSCandidate], List[FTSCandidate]]:
    """Split candidates into partition A (matches original) and B (expansion-only)."""
    partition_a = [c for c in candidates if c.partition == "A"]
    partition_b = [c for c in candidates if c.partition == "B"]
    logger.info(f"Partitioned: A={len(partition_a)}, B={len(partition_b)}")
    return partition_a, partition_b


def apply_pre_filters(
    conn: Connection,
    candidates: List[FTSCandidate],
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_abstract_length: int = MIN_ABSTRACT_LENGTH,
) -> List[FTSCandidate]:
    """Filter candidates by year bounds and minimum abstract length.

    Args:
        conn: Database connection
        candidates: List of FTS candidates
        year_min: Minimum publication year (inclusive)
        year_max: Maximum publication year (inclusive)
        min_abstract_length: Minimum abstract character length

    Returns:
        Filtered list of candidates
    """
    if not candidates:
        return []

    work_ids = [c.work_id for c in candidates]
    candidate_map = {c.work_id: c for c in candidates}

    # Load work metadata for filtering
    works = WorkStore.load_many(conn, work_ids)

    filtered = []
    for wid, candidate in candidate_map.items():
        w = works.get(wid)
        if not w:
            continue

        # Year filter
        if year_min is not None:
            if w.year is None or w.year < year_min:
                continue
        if year_max is not None:
            if w.year is None or w.year > year_max:
                continue

        # Abstract length filter
        abstract = getattr(w, 'abstract', None) or ""
        if len(abstract) < min_abstract_length:
            continue

        filtered.append(candidate)

    logger.info(f"Pre-filter: {len(candidates)} -> {len(filtered)} candidates")
    return filtered


def select_papers_for_scoring(
    partition_a: List[FTSCandidate],
    partition_b: List[FTSCandidate],
    cap: int = 68,
) -> List[str]:
    """Select up to `cap` papers for LLM scoring.

    Prioritizes partition A (strong lexical matches) by FTS score,
    then fills remaining slots with partition B.

    Args:
        partition_a: Strong lexical matches
        partition_b: Expansion-only matches
        cap: Maximum number of papers to select

    Returns:
        List of work_ids to score
    """
    # Sort each partition by combined_score descending
    sorted_a = sorted(partition_a, key=lambda c: -c.combined_score)
    sorted_b = sorted(partition_b, key=lambda c: -c.combined_score)

    selected: List[str] = []

    # Take from partition A first (aim for ~80% of cap)
    a_limit = int(cap * 0.8)
    for c in sorted_a[:a_limit]:
        selected.append(c.work_id)

    # Fill remaining from partition B
    remaining = cap - len(selected)
    for c in sorted_b[:remaining]:
        selected.append(c.work_id)

    # If we still have room and partition A has more, take them
    if len(selected) < cap and len(sorted_a) > a_limit:
        remaining = cap - len(selected)
        for c in sorted_a[a_limit:a_limit + remaining]:
            selected.append(c.work_id)

    logger.info(f"Selected {len(selected)} papers for LLM scoring (cap={cap})")
    return selected


def generate_candidates_lexical(
    conn: Connection,
    *,
    query_text: str,
    original_terms: List[str],
    expansion_terms: List[str],
    filters_json: Optional[Dict] = None,
    limit: int = DEFAULT_FTS_LIMIT,
) -> Tuple[List[FTSCandidate], List[FTSCandidate]]:
    """Main entry point for lexical candidate generation.

    Returns partitioned candidates (A, B) after FTS retrieval and pre-filtering.
    """
    filters_json = filters_json or {}

    # Execute FTS search
    candidates = execute_fts_search(
        conn,
        original_terms=original_terms,
        expansion_terms=expansion_terms,
        limit=limit,
    )

    if not candidates:
        return [], []

    # Apply pre-filters
    year_min = filters_json.get("year_min")
    year_max = filters_json.get("year_max")
    min_abstract = filters_json.get("min_abstract_length", MIN_ABSTRACT_LENGTH)

    filtered = apply_pre_filters(
        conn,
        candidates,
        year_min=int(year_min) if year_min is not None else None,
        year_max=int(year_max) if year_max is not None else None,
        min_abstract_length=int(min_abstract),
    )

    # Partition into A and B
    return partition_candidates(filtered)
