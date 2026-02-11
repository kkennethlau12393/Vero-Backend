"""
Citation lineage builder for Feature 4 (Methodology Comparison).

Builds citation relationships between selected papers:
- Direct citations between the selected papers
- Shared references (common methodological ancestors)
- Chronological evolution chain
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from app.feature4.schemas import CitationLineage, DirectCitation, SharedReference

logger = logging.getLogger(__name__)

# Max shared references to include (most-cited first)
MAX_SHARED_REFS = 10


def build_citation_lineage(
    conn: Connection,
    work_ids: list[str],
    map_id: Optional[str] = None,
) -> CitationLineage:
    """
    Build citation relationships between the selected papers.

    1. Find direct citations between selected papers (from map_edges or referenced_works_json)
    2. Find shared references (papers cited by 2+ selected papers)
    3. Build chronological evolution chain if applicable
    """
    direct = _find_direct_citations(conn, work_ids, map_id)
    shared = _find_shared_references(conn, work_ids)
    chain = _build_evolution_chain(conn, work_ids, direct)

    return CitationLineage(
        direct_citations=direct,
        shared_references=shared,
        evolution_chain=chain,
    )


def _find_direct_citations(
    conn: Connection,
    work_ids: list[str],
    map_id: Optional[str],
) -> list[DirectCitation]:
    """
    Find direct citations between the selected papers.

    First tries map_edges (if map_id provided), then falls back to
    referenced_works_json from the works table.
    """
    citations: list[DirectCitation] = []
    work_set = set(work_ids)

    # Try map_edges first (faster, already computed)
    if map_id:
        rows = conn.execute(
            sa_text("""
                SELECT from_work_id, to_work_id
                FROM map_edges
                WHERE map_id = :map_id
                  AND from_work_id = ANY(:ids)
                  AND to_work_id = ANY(:ids)
            """),
            {"map_id": map_id, "ids": work_ids},
        ).mappings().all()

        for row in rows:
            citations.append(DirectCitation(
                from_work_id=row["from_work_id"],
                to_work_id=row["to_work_id"],
            ))

        if citations:
            return citations

    # Fallback: check referenced_works_json
    rows = conn.execute(
        sa_text("""
            SELECT work_id, referenced_works_json
            FROM works
            WHERE work_id = ANY(:ids)
              AND referenced_works_json IS NOT NULL
        """),
        {"ids": work_ids},
    ).mappings().all()

    for row in rows:
        refs = row["referenced_works_json"]
        if not isinstance(refs, list):
            continue
        for ref_id in refs:
            if ref_id in work_set and ref_id != row["work_id"]:
                citations.append(DirectCitation(
                    from_work_id=row["work_id"],
                    to_work_id=ref_id,
                ))

    return citations


def _find_shared_references(
    conn: Connection,
    work_ids: list[str],
) -> list[SharedReference]:
    """
    Find papers referenced by 2+ of the selected papers.

    These are the "common methodological ancestors" — papers that
    multiple selected works build upon.
    """
    # Collect referenced_works_json for each selected paper
    rows = conn.execute(
        sa_text("""
            SELECT work_id, referenced_works_json
            FROM works
            WHERE work_id = ANY(:ids)
              AND referenced_works_json IS NOT NULL
        """),
        {"ids": work_ids},
    ).mappings().all()

    # Build ref_id -> set of citing selected papers
    ref_to_citers: dict[str, set[str]] = {}
    for row in rows:
        refs = row["referenced_works_json"]
        if not isinstance(refs, list):
            continue
        for ref_id in refs:
            if ref_id not in set(work_ids):  # Exclude self-references to selected set
                ref_to_citers.setdefault(ref_id, set()).add(row["work_id"])

    # Keep only refs cited by 2+ selected papers
    shared_ids = {
        ref_id: citers
        for ref_id, citers in ref_to_citers.items()
        if len(citers) >= 2
    }

    if not shared_ids:
        return []

    # Load metadata for shared references
    ref_id_list = list(shared_ids.keys())
    ref_rows = conn.execute(
        sa_text("""
            SELECT work_id, title, cited_by_count
            FROM works
            WHERE work_id = ANY(:ids)
        """),
        {"ids": ref_id_list},
    ).mappings().all()

    results = []
    for row in ref_rows:
        wid = row["work_id"]
        results.append(SharedReference(
            work_id=wid,
            title=row["title"] or "Unknown title",
            cited_by=sorted(shared_ids[wid]),
        ))

    # Sort by number of citers (descending), then by cited_by_count
    results.sort(key=lambda r: -len(r.cited_by))
    return results[:MAX_SHARED_REFS]


def _build_evolution_chain(
    conn: Connection,
    work_ids: list[str],
    direct_citations: list[DirectCitation],
) -> Optional[str]:
    """
    Build an evolution chain string if papers form a citation sequence.

    e.g. "W123 (2012) → W456 (2017) → W789 (2022)"
    Only produces a chain if there's a clear chronological citation path.
    """
    if len(work_ids) < 2:
        return None

    # Only build chain if there are actual citation links
    if not direct_citations:
        return None

    # Load years
    rows = conn.execute(
        sa_text("""
            SELECT work_id, title, year
            FROM works
            WHERE work_id = ANY(:ids)
        """),
        {"ids": work_ids},
    ).mappings().all()

    year_map = {row["work_id"]: row["year"] for row in rows}
    title_map = {row["work_id"]: row["title"] or row["work_id"] for row in rows}

    # Sort papers chronologically
    sorted_papers = sorted(
        work_ids,
        key=lambda wid: year_map.get(wid) or 9999,
    )

    # Check if there's a citation chain (each paper cites at least one earlier paper)
    citation_set = {(c.from_work_id, c.to_work_id) for c in direct_citations}

    chain_valid = True
    for i in range(1, len(sorted_papers)):
        current = sorted_papers[i]
        # Check if current cites any earlier paper
        cites_earlier = any(
            (current, sorted_papers[j]) in citation_set
            for j in range(i)
        )
        if not cites_earlier and direct_citations:
            chain_valid = False
            break

    if not chain_valid:
        return None

    # Build chain string
    parts = []
    for wid in sorted_papers:
        year = year_map.get(wid)
        title = title_map[wid]
        # Truncate long titles
        if len(title) > 60:
            title = title[:57] + "..."
        if year:
            parts.append(f"{title} ({year})")
        else:
            parts.append(title)

    return " → ".join(parts)
