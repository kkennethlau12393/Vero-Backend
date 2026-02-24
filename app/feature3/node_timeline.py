"""
Node timeline builder for Feature 3.

This module builds a temporal timeline around a target paper, showing:
- Backward: References + Landmarks (papers before the target)
- Forward: Citing papers (papers after the target)
- Narrative: Rich vertical-evolution story of the research lineage

Papers are grouped by era (decades) for visualization, with optional
per-era commentary from the LLM narrative.

Note: Paper fetching and caching is handled by paper_cache.py for sharing
between timeline and novelty assessment features.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Literal

import requests
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.paper_cache import get_citing_papers

logger = logging.getLogger(__name__)

MAX_CITING_PAPERS = 20  # Limit for timeline (more than grounding supplement)

SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")


# ============================================================================
# Era Grouping Utilities
# ============================================================================

def get_era_label(year: Optional[int]) -> str:
    """Convert a year to an era label (decade)."""
    if year is None:
        return "Unknown"
    decade = (year // 10) * 10
    return f"{decade}s"


def group_papers_by_era(
    papers: List[Dict[str, Any]],
    relationship: Literal["reference", "landmark", "citing"],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group papers by era (decade).

    Returns dict: era_label -> list of papers with relationship added.
    """
    era_map: Dict[str, List[Dict[str, Any]]] = {}

    for paper in papers:
        year = paper.get("year")
        era = get_era_label(year)

        paper_entry = {
            "work_id": paper.get("work_id"),
            "title": paper.get("title"),
            "year": year,
            "cited_by_count": paper.get("cited_by_count") or 0,
            "relationship": relationship,
        }

        if era not in era_map:
            era_map[era] = []
        era_map[era].append(paper_entry)

    return era_map


def merge_era_maps(*era_maps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Merge multiple era maps into one, deduplicating by work_id within each era."""
    merged: Dict[str, List[Dict[str, Any]]] = {}

    for era_map in era_maps:
        for era, papers in era_map.items():
            if era not in merged:
                merged[era] = []
            merged[era].extend(papers)

    # Deduplicate within each era by work_id (keep first occurrence)
    for era, papers in merged.items():
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for p in papers:
            wid = p.get("work_id")
            if wid and wid in seen:
                continue
            if wid:
                seen.add(wid)
            deduped.append(p)
        merged[era] = deduped

    return merged


def era_map_to_sections(
    era_map: Dict[str, List[Dict[str, Any]]],
    sort_ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert era map to sorted list of TimelineSection dicts.

    Args:
        era_map: Dict of era -> papers
        sort_ascending: If True, sort eras chronologically (oldest first)
    """
    sections = []

    for era, papers in era_map.items():
        # Sort papers within era by citation count (most cited first)
        sorted_papers = sorted(papers, key=lambda p: -(p.get("cited_by_count") or 0))
        sections.append({
            "era": era,
            "papers": sorted_papers,
        })

    # Sort sections by era (extract decade number for sorting)
    def era_sort_key(section: Dict) -> int:
        era = section["era"]
        if era == "Unknown":
            return 9999 if sort_ascending else -9999
        try:
            return int(era.replace("s", ""))
        except ValueError:
            return 9999 if sort_ascending else -9999

    sections.sort(key=era_sort_key, reverse=not sort_ascending)
    return sections


def _apply_era_commentaries(
    sections: List[Dict[str, Any]],
    era_commentaries: List[Dict[str, Any]],
) -> None:
    """Map LLM era commentaries onto timeline sections by matching era labels."""
    commentary_map = {
        ec.get("era"): ec
        for ec in era_commentaries
        if ec.get("era")
    }
    for section in sections:
        ec = commentary_map.get(section["era"])
        if ec:
            section["commentary"] = ec.get("narrative")


# ============================================================================
# Dual-Source Retrieval (OA + S2)
# ============================================================================

def _fetch_citing_papers_s2(doi: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch citing papers from Semantic Scholar via DOI."""
    if not doi:
        return []

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    clean_doi = doi.replace("https://doi.org/", "")
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}/citations"
        params = {
            "fields": "paperId,title,year,citationCount,externalIds,abstract",
            "limit": min(limit, 1000),
        }
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            return []

        result = []
        for item in (resp.json() or {}).get("data") or []:
            citing = (item or {}).get("citingPaper")
            if not citing or not isinstance(citing, dict):
                continue
            external_ids = citing.get("externalIds") or {}
            result.append({
                "doi": external_ids.get("DOI"),
                "s2_id": citing.get("paperId"),
                "title": citing.get("title"),
                "year": citing.get("year"),
                "cited_by_count": citing.get("citationCount") or 0,
                "abstract": citing.get("abstract"),
            })
        logger.info(f"S2 citing papers: {len(result)} for DOI:{clean_doi}")
        return result
    except Exception as e:
        logger.warning(f"S2 citing papers fetch failed: {e}")
        return []


def _resolve_s2_papers_to_openalex(
    s2_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve S2 papers to OpenAlex work_ids via DOI batch lookup."""
    papers_with_dois = [p for p in s2_papers if p.get("doi")]
    if not papers_with_dois:
        return []

    resolved = []
    dois = [p["doi"] for p in papers_with_dois]
    for i in range(0, len(dois), 50):
        batch = dois[i:i + 50]
        try:
            doi_filter = "|".join(batch)
            url = "https://api.openalex.org/works"
            params = {
                "filter": f"doi:{doi_filter}",
                "per-page": len(batch),
                "select": "id,title,publication_year,cited_by_count,abstract_inverted_index",
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY

            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                continue

            from app.feature3.landmark_retrieval import decode_openalex_abstract
            for w in resp.json().get("results", []):
                wid_full = w.get("id")
                if not wid_full or "/" not in wid_full:
                    continue
                wid = wid_full.rsplit("/", 1)[-1]
                resolved.append({
                    "work_id": wid,
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract": decode_openalex_abstract(w.get("abstract_inverted_index")),
                })
        except Exception as e:
            logger.warning(f"S2->OA resolution failed: {e}")

    logger.info(f"Resolved {len(resolved)}/{len(papers_with_dois)} S2 citing papers to OpenAlex")
    return resolved


def _get_doi_for_work(conn: Connection, work_id: str) -> Optional[str]:
    """Get DOI for a work_id from the works table."""
    row = conn.execute(
        text("SELECT doi FROM works WHERE work_id = :wid"),
        {"wid": work_id},
    ).mappings().first()
    return row["doi"] if row and row.get("doi") else None


def _get_citing_papers_dual_source(
    conn: Connection, work_id: str, limit: int = MAX_CITING_PAPERS,
) -> List[Dict[str, Any]]:
    """
    Get citing papers from both OpenAlex and Semantic Scholar.

    OA is queried via the existing paper_cache. S2 is queried via DOI.
    Results are merged and deduplicated by work_id, keeping OA as primary.
    """
    # OA citing papers (primary — already cached)
    oa_papers = get_citing_papers(conn, work_id, limit=limit)
    oa_work_ids = {p.get("work_id") for p in oa_papers if p.get("work_id")}

    # S2 citing papers (co-equal source — fetch in parallel)
    doi = _get_doi_for_work(conn, work_id)
    s2_raw = _fetch_citing_papers_s2(doi, limit=limit * 2)

    if not s2_raw:
        return oa_papers[:limit]

    # Resolve S2 papers to OA work_ids
    s2_resolved = _resolve_s2_papers_to_openalex(s2_raw)

    # Merge: add S2 papers not already in OA set
    new_from_s2 = [p for p in s2_resolved if p.get("work_id") not in oa_work_ids]
    if new_from_s2:
        logger.info(f"S2 contributed {len(new_from_s2)} new citing papers not in OpenAlex")

    merged = oa_papers + new_from_s2
    # Sort by citation count and return top N
    merged.sort(key=lambda p: -(p.get("cited_by_count") or 0))
    return merged[:limit]


# ============================================================================
# Timeline Builder
# ============================================================================

def build_node_timeline(
    conn: Connection,
    work_id: str,
    target_year: Optional[int],
    references: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    include_narrative: bool = True,
    target_title: Optional[str] = None,
    target_abstract: Optional[str] = None,
    target_cited_by_count: int = 0,
) -> Dict[str, Any]:
    """
    Build a complete timeline for a node.

    Args:
        conn: Database connection
        work_id: Target paper's OpenAlex ID
        target_year: Target paper's publication year
        references: Papers the target cites (from referenced_works_json)
        landmarks: Landmark papers for context (from landmark_works_json)
        include_narrative: If True, generate rich LLM narrative
        target_title: Target paper's title
        target_abstract: Target paper's abstract
        target_cited_by_count: Target paper's citation count

    Returns:
        Dict with target_work_id, target_year, backward, forward, narrative
    """
    # Group backward papers by era (dedup handled in merge)
    ref_eras = group_papers_by_era(references, "reference")
    landmark_eras = group_papers_by_era(landmarks, "landmark")
    backward_eras = merge_era_maps(ref_eras, landmark_eras)

    # Get citing papers (forward) - dual-source: OA + S2
    citing_papers = _get_citing_papers_dual_source(conn, work_id, limit=MAX_CITING_PAPERS)
    forward_eras = group_papers_by_era(citing_papers, "citing")

    # Convert to sections
    backward_sections = era_map_to_sections(backward_eras, sort_ascending=True)
    forward_sections = era_map_to_sections(forward_eras, sort_ascending=True)

    # Generate rich narrative
    narrative = None
    if include_narrative:
        from app.feature3.timeline_narrative import generate_timeline_narrative

        logger.info(f"Generating timeline narrative for {work_id}")
        narrative = generate_timeline_narrative(
            conn=conn,
            work_id=work_id,
            title=target_title or "",
            abstract=target_abstract,
            year=target_year,
            cited_by_count=target_cited_by_count,
            references=references,
            landmarks=landmarks,
            citing_papers=citing_papers,
        )

        # Map era commentaries onto sections
        if narrative and narrative.get("era_commentaries"):
            _apply_era_commentaries(backward_sections, narrative["era_commentaries"])
            _apply_era_commentaries(forward_sections, narrative["era_commentaries"])

    return {
        "target_work_id": work_id,
        "target_year": target_year,
        "backward": backward_sections,
        "forward": forward_sections,
        "narrative": narrative,
    }
