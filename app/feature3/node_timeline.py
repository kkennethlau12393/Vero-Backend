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
            "abstract": paper.get("abstract"),
            "authors": paper.get("authors", []),
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
            "fields": "paperId,title,year,citationCount,externalIds,abstract,authors",
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
            s2_authors = [a.get("name") for a in (citing.get("authors") or []) if a.get("name")]
            result.append({
                "doi": external_ids.get("DOI"),
                "s2_id": citing.get("paperId"),
                "title": citing.get("title"),
                "year": citing.get("year"),
                "cited_by_count": citing.get("citationCount") or 0,
                "abstract": citing.get("abstract"),
                "authors": s2_authors,
            })
        logger.info(f"S2 citing papers: {len(result)} for DOI:{clean_doi}")
        return result
    except Exception as e:
        logger.warning(f"S2 citing papers fetch failed: {e}")
        return []


def _resolve_s2_papers_to_openalex(
    s2_papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve S2 papers to OpenAlex work_ids via DOI batch lookup.

    Papers that resolve get W... work_ids.
    Papers that don't resolve keep S... work_ids (S + S2 paperId).
    """
    if not s2_papers:
        return []

    papers_with_dois = [p for p in s2_papers if p.get("doi")]
    papers_without_dois = [p for p in s2_papers if not p.get("doi")]

    # Build DOI -> S2 paper mapping for unresolved tracking
    doi_to_s2 = {p["doi"]: p for p in papers_with_dois}
    resolved_dois: set = set()

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
                "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index",
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
                oa_authors = [
                    au.get("author", {}).get("display_name") or au.get("display_name")
                    for au in w.get("authorships", [])
                    if au.get("author", {}).get("display_name") or au.get("display_name")
                ]
                resolved.append({
                    "work_id": wid,
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract": decode_openalex_abstract(w.get("abstract_inverted_index")),
                    "authors": oa_authors,
                })
                # Track which DOIs resolved
                w_doi = w.get("doi")
                if w_doi:
                    clean = w_doi.replace("https://doi.org/", "")
                    resolved_dois.add(clean)
        except Exception as e:
            logger.warning(f"S2->OA resolution failed: {e}")

    # Keep unresolved S2 papers with S prefix work_ids
    unresolved = []
    for p in papers_with_dois:
        if p["doi"] not in resolved_dois:
            unresolved.append(_s2_paper_to_dict(p))
    for p in papers_without_dois:
        unresolved.append(_s2_paper_to_dict(p))

    logger.info(
        f"S2 citing papers: {len(resolved)} resolved to OA, "
        f"{len(unresolved)} kept as S-prefixed"
    )
    return resolved + unresolved


def _s2_paper_to_dict(p: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an S2 paper to a standard dict with S-prefixed work_id."""
    s2_id = p.get("s2_id") or p.get("paperId") or ""
    return {
        "work_id": f"S{s2_id}",
        "title": p.get("title"),
        "year": p.get("year"),
        "cited_by_count": p.get("cited_by_count") or 0,
        "abstract": p.get("abstract"),
        "authors": p.get("authors", []),
    }


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
# S2 Batch Abstract Enrichment
# ============================================================================

def _get_dois_for_work_ids(
    conn: Connection, work_ids: List[str],
) -> Dict[str, str]:
    """Batch-fetch DOIs from the works table for W-prefixed work_ids."""
    w_ids = [wid for wid in work_ids if wid.startswith("W")]
    if not w_ids:
        return {}

    rows = conn.execute(
        text("SELECT work_id, doi FROM works WHERE work_id = ANY(:ids) AND doi IS NOT NULL"),
        {"ids": w_ids},
    ).mappings().all()
    return {row["work_id"]: row["doi"].replace("https://doi.org/", "") for row in rows if row["doi"]}


def _fetch_s2_abstracts_batch(
    dois: List[str],
) -> Dict[str, str]:
    """
    Fetch abstracts from Semantic Scholar batch API for a list of DOIs.

    Uses POST /paper/batch which accepts up to 500 IDs per call.
    Returns dict: DOI -> abstract.
    """
    if not dois:
        return {}

    headers = {"Content-Type": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    result: Dict[str, str] = {}

    # S2 batch API accepts up to 500 IDs per call
    for i in range(0, len(dois), 500):
        batch = dois[i:i + 500]
        try:
            resp = requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "title,abstract,externalIds"},
                json={"ids": [f"DOI:{d}" for d in batch]},
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(f"S2 batch API returned {resp.status_code}")
                continue

            for paper in resp.json():
                if not paper or not isinstance(paper, dict):
                    continue
                abstract = paper.get("abstract")
                if not abstract or len(abstract) < 50:
                    continue
                ext_ids = paper.get("externalIds") or {}
                doi = ext_ids.get("DOI")
                if doi:
                    result[doi.lower()] = abstract
        except Exception as e:
            logger.warning(f"S2 batch abstract fetch failed: {e}")

    logger.info(f"S2 batch abstract enrichment: {len(result)}/{len(dois)} DOIs returned abstracts")
    return result


def _fetch_s2_abstract_by_title(title: str) -> Optional[str]:
    """Search S2 by title and return abstract of the best match."""
    if not title or len(title) < 10:
        return None

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": 3, "fields": "title,abstract"},
            headers=headers,
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        for paper in (resp.json() or {}).get("data") or []:
            abstract = paper.get("abstract")
            if not abstract or len(abstract) < 50:
                continue
            # Verify title matches (simple word overlap)
            s2_title = (paper.get("title") or "").lower()
            query_title = title.lower()
            s2_words = set(s2_title.split())
            q_words = set(query_title.split())
            if len(s2_words & q_words) >= max(2, len(q_words) * 0.5):
                return abstract
    except Exception as e:
        logger.warning(f"S2 title search failed for '{title[:40]}': {e}")
    return None


def _enrich_timeline_abstracts(
    conn: Connection,
    references: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    citing_papers: List[Dict[str, Any]],
) -> int:
    """
    Enrich papers with missing abstracts via S2 batch API + title search.

    Strategy:
    1. Batch-fetch by DOI (fast, one API call for all)
    2. Title search for remaining (one call per paper, limited to 10)

    Modifies paper dicts in-place. Persists enriched abstracts to the works table.
    Returns the number of papers enriched.
    """
    # Collect papers missing abstracts
    all_papers = references + landmarks + citing_papers
    missing = [p for p in all_papers if not p.get("abstract") and p.get("work_id")]
    if not missing:
        return 0

    logger.info(f"Timeline abstract enrichment: {len(missing)} papers missing abstracts")

    # Get DOIs for W-prefixed papers from DB
    missing_wids = [p["work_id"] for p in missing]
    doi_map = _get_dois_for_work_ids(conn, missing_wids)

    # Split into papers with DOIs and without
    with_doi: List[Dict[str, Any]] = []
    without_doi: List[Dict[str, Any]] = []
    doi_to_papers: Dict[str, List[Dict[str, Any]]] = {}
    for p in missing:
        wid = p["work_id"]
        doi = doi_map.get(wid)
        if doi:
            doi_lower = doi.lower()
            if doi_lower not in doi_to_papers:
                doi_to_papers[doi_lower] = []
            doi_to_papers[doi_lower].append(p)
            with_doi.append(p)
        else:
            without_doi.append(p)

    enriched_count = 0

    # Strategy 1: Batch-fetch by DOI (fast)
    if doi_to_papers:
        s2_abstracts = _fetch_s2_abstracts_batch(list(doi_to_papers.keys()))
        for doi_lower, abstract in s2_abstracts.items():
            for p in doi_to_papers.get(doi_lower, []):
                p["abstract"] = abstract
                enriched_count += 1
                _persist_abstract(conn, p["work_id"], abstract)

    # Strategy 2: Title search for remaining (capped at 10 to avoid rate limits)
    still_missing = [p for p in missing if not p.get("abstract") and p.get("title")]
    for p in still_missing[:10]:
        abstract = _fetch_s2_abstract_by_title(p["title"])
        if abstract:
            p["abstract"] = abstract
            enriched_count += 1
            _persist_abstract(conn, p["work_id"], abstract)

    if enriched_count > 0:
        try:
            conn.commit()
        except Exception:
            pass

    logger.info(f"Timeline abstract enrichment: enriched {enriched_count}/{len(missing)} papers via S2")
    return enriched_count


def _persist_abstract(conn: Connection, work_id: str, abstract: str) -> None:
    """Persist enriched abstract to works table for future reuse."""
    if not work_id.startswith("W"):
        return
    try:
        conn.execute(
            text("""
                UPDATE works SET abstract = :abstract,
                    abstract_source = 'semantic_scholar',
                    abstract_validated_at = now()
                WHERE work_id = :wid AND (abstract IS NULL OR length(abstract) < 50)
            """),
            {"abstract": abstract, "wid": work_id},
        )
    except Exception as e:
        logger.warning(f"Failed to persist S2 abstract for {work_id}: {e}")


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

    # Enrich papers with missing abstracts via S2 batch API
    # This improves narrative quality by giving the LLM more context
    _enrich_timeline_abstracts(conn, references, landmarks, citing_papers)

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
