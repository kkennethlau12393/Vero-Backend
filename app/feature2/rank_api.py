"""
API router for Feature 2 (Direct Ranking).

This module exposes an endpoint to perform direct ranking using the
production‑grade pipeline defined in ``app.feature2.rank_service``.
It accepts a query string and optional context, filters and ranking
parameters, then delegates to ``direct_rank_prod``.  The endpoint
returns a ``DirectRankResponse`` with the ranked items and job
metadata.  Legacy v1 ranking is no longer exposed via this router.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.db import make_engine
from app.auth.tenant import get_tenant_id
from app.feature2.schemas import (
    BreakthroughAnalysis,
    BreakthroughPaper,
    Citation,
    DirectRankResponse,
    EvolutionTrend,
    RankFilters,
    RankParams,
    SubtopicsResponse,
    SubtopicSuggestion,
    TemporalEra,
    TemporalMapAnalytics,
    TemporalMapResponse,
    TemporalPaper,
)
from app.feature2.rank_service import direct_rank_prod
from app.feature2.subtopic_service import generate_subtopics
from app.feature2.temporal_map_service import build_temporal_map
from app.feature2.title_to_query import extract_display_title, generate_topic_query_from_title


router = APIRouter(prefix="/v1/rank", tags=["rank"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazily construct the database engine once per process."""
    return make_engine()


class DirectQueryRankRequest(BaseModel):
    """Request model for the production direct ranking endpoint.

    Parameters
    ----------
    query_text : str, optional
        The free‑text query used to retrieve and rank works.
        Exactly ONE of query_text, seed_doi, or seed_work_id must be provided.
    seed_doi : str, optional
        DOI of a paper to research related topics (e.g., "10.48550/arXiv.1706.03762").
        Backend extracts the title and generates a query. For "Research this topic" workflows.
    seed_work_id : str, optional
        OpenAlex/S2/ArXiv work ID (e.g., "W2963663278", "S2:204e3073...", "AX:1706.03762").
        Backend extracts the title and generates a query. For "Research this topic" workflows.
    seed_title : str, optional
        DEPRECATED - Use seed_doi or seed_work_id instead, or upload PDF via /v1/rank/pdf.
        Paper title to infer a topic query from. Kept for backward compatibility.
    context : dict[str, Any], optional
        Additional context, such as a target topic ID.  Defaults to an
        empty dict if not provided.
    filters : RankFilters, optional
        Hard filters to restrict the candidate set, e.g. ``year_min``
        and ``year_max``.  Defaults to None.
    params : RankParams, optional
        Ranking parameters controlling pool sizes, weights and decay
        rates.  Defaults to None.
    """

    query_text: Optional[str] = None
    seed_doi: Optional[str] = None      # NEW: DOI → title → query
    seed_work_id: Optional[str] = None  # NEW: work_id → title → query
    seed_title: Optional[str] = None    # DEPRECATED: kept for backward compatibility
    context: Optional[dict[str, Any]] = None
    filters: Optional[RankFilters] = None
    params: Optional[RankParams] = None
    # Backwards-compatible alias for clients using "rank_params"
    rank_params: Optional[RankParams] = None


@router.post("", response_model=DirectRankResponse)
def direct_rank_endpoint(
    req: DirectQueryRankRequest,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """Handle a direct ranking request using the production pipeline.

    This endpoint accepts a query string and optional context/filters
    and delegates to ``direct_rank_prod``.  It does not require
    pre‑computing a candidate set or passing a candidate_set_id.

    For "Research this topic" workflows:
    - Use seed_doi for DOI-based seed (e.g., from PDF metadata)
    - Use seed_work_id for work_id-based seed (e.g., from citation maps)
    - Use /v1/rank/pdf endpoint for PDF upload
    - seed_title is deprecated but still supported for backward compatibility
    """
    try:
        # Validation: exactly ONE of query_text, seed_doi, seed_work_id, seed_title
        provided_modes = sum([
            bool(req.query_text),
            bool(req.seed_doi),
            bool(req.seed_work_id),
            bool(req.seed_title),
        ])

        if provided_modes == 0:
            raise ValueError(
                "Must provide ONE of: query_text, seed_doi, seed_work_id, or seed_title"
            )
        elif provided_modes > 1:
            raise ValueError(
                "Provide only ONE input mode (query_text, seed_doi, seed_work_id, or seed_title)"
            )

        # Determine query_text based on input mode
        query_text = req.query_text
        display_title = extract_display_title(req.query_text) if req.query_text else None

        if req.seed_doi or req.seed_work_id:
            from app.feature2.seed_resolver import resolve_doi_to_title, resolve_work_id_to_title

            if req.seed_doi:
                title, metadata = resolve_doi_to_title(req.seed_doi)
                logger.info(f"Resolved DOI {req.seed_doi} to title: '{title[:50]}...'")
            else:  # seed_work_id
                title, metadata = resolve_work_id_to_title(req.seed_work_id)
                logger.info(f"Resolved work_id {req.seed_work_id} to title: '{title[:50]}...'")

            display_title = title

            # Generate query from extracted title
            query_text = generate_topic_query_from_title(title)
            if not query_text:
                raise ValueError("Failed to generate query from extracted title")
            logger.info(f"Generated query: '{query_text}'")

        elif req.seed_title:
            # Backward compatibility path
            logger.warning("seed_title is deprecated. Use seed_doi, seed_work_id, or /v1/rank/pdf")
            logger.info(f"Generating query from seed_title: '{req.seed_title[:50]}...'")
            display_title = req.seed_title
            query_text = generate_topic_query_from_title(req.seed_title)
            if not query_text:
                raise ValueError("Failed to generate query from seed_title")
            logger.info(f"Generated query: '{query_text}'")

        if not query_text:
            raise ValueError(
                "Must provide ONE of: query_text, seed_doi, seed_work_id, or seed_title"
            )

        context_json = req.context or {}
        filters_json = req.filters.model_dump() if req.filters else {}
        params_obj = req.params or req.rank_params
        rank_params_json = params_obj.model_dump() if params_obj else {}

        # Invoke the production ranking pipeline
        result = direct_rank_prod(
            engine,
            tenant_id=tenant_id,
            query_text=query_text,
            context_json=context_json,
            filters_json=filters_json,
            rank_params_json=rank_params_json,
        )

        if isinstance(result, dict):
            result["display_title"] = display_title

        status = (result.get("job") or {}).get("status")
        if status in ("pending", "running"):
            return JSONResponse(
                status_code=202,
                content={"rank_job_id": str(result["rank_job_id"]), "status": status},
            )
        return result

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        # Map domain errors to appropriate HTTP codes
        msg = str(e)
        if msg == "candidate_set_not_found":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in direct_rank_prod endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf", response_model=DirectRankResponse)
async def rank_from_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    context: Optional[str] = None,   # JSON string (multipart limitation)
    filters: Optional[str] = None,   # JSON string
    params: Optional[str] = None,    # JSON string
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Rank papers for a topic from an uploaded PDF.

    Workflow:
    1. Extract title from PDF (metadata or text)
    2. Generate topic query from title
    3. Run ranking pipeline

    Accepts same filters/params as /rank endpoint (as JSON strings).
    """
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")

    try:
        pdf_bytes = await pdf_file.read()

        from app.feature2.seed_resolver import resolve_pdf_to_title
        title, metadata = resolve_pdf_to_title(pdf_bytes)

        logger.info(f"Extracted title from PDF '{pdf_file.filename}': '{title[:50]}...'")
        query_text = generate_topic_query_from_title(title)

        if not query_text:
            raise ValueError("Failed to generate query from extracted title")

        logger.info(f"Generated query from PDF: '{query_text}'")

        # Parse JSON params from multipart form
        context_json = json.loads(context) if context else {}
        filters_json = json.loads(filters) if filters else {}
        params_json = json.loads(params) if params else {}

        result = direct_rank_prod(
            engine,
            tenant_id=tenant_id,
            query_text=query_text,
            context_json=context_json,
            filters_json=filters_json,
            rank_params_json=params_json,
        )

        # Add extraction metadata and display title to response
        if isinstance(result, dict):
            result["display_title"] = title
            result["seed_extraction"] = {
                "pdf_filename": pdf_file.filename,
                "extracted_title": title,
                "extraction_method": metadata.get("extraction_method"),
                "generated_query": query_text,
            }

        status = (result.get("job") or {}).get("status")
        if status in ("pending", "running"):
            return JSONResponse(
                status_code=202,
                content={"rank_job_id": str(result["rank_job_id"]), "status": status},
            )
        return result

    except ValueError as e:
        raise HTTPException(400, f"PDF processing failed: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error in PDF ranking endpoint")
        raise HTTPException(500, str(e))


class DrillDownRequest(BaseModel):
    """Request model for lightweight subtopic drill-down.

    Parameters
    ----------
    query_text : str
        The drill_down_query from a subtopic suggestion.
    top_k : int, optional
        Number of papers to return. Defaults to 10.
    """

    query_text: str
    top_k: int = 10


# Lightweight params for drill-down: fewer candidates = faster
DRILL_DOWN_MAX_CANDIDATES = 300  # Increased from 200 for better coverage
DRILL_DOWN_LLM_CAP = 30  # Balance speed and coverage


@router.post("/drill-down", response_model=DirectRankResponse)
def drill_down_endpoint(
    req: DrillDownRequest,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """Lightweight ranking for subtopic drill-down.

    This endpoint is optimized for fetching a small number of papers
    for a specific subtopic. It uses reduced candidate scoring and returns
    a flat ranked list (no categorization into fundamentals/core/etc).

    Use this when drilling down into a subtopic from generate-subtopics.
    For full research queries, use the main /rank endpoint instead.
    """
    try:
        # Use lightweight params: fewer candidates, small top_k, reduced LLM cap
        # Skip external sources (DBLP, PubMed, CrossRef) to avoid 15s+ timeouts
        # Skip categorization: return flat list of top papers (no fundamentals/core/etc)
        # Higher LLM relevance threshold (0.35 vs 0.25) to filter out tangentially related papers
        rank_params_json = {
            "top_k": req.top_k,
            "max_candidates_scored": DRILL_DOWN_MAX_CANDIDATES,
            "llm_scoring_cap": DRILL_DOWN_LLM_CAP,
            "skip_external_sources": True,
            "skip_categorization": True,
            "min_llm_relevance": 0.35,  # Higher threshold for drill-down precision
            # Drill-down weights: LLM relevance and impact are primary signals
            # LLM (0.50) is the main relevance signal
            # Impact (0.35) rewards foundational high-cite papers
            # Zero recency/topic to prevent recent/keyword-matched papers from dominating
            # Minimal other weights to reduce noise
            "w_recency": 0.0,
            "w_impact": 0.35,
            "w_llm_relevance": 0.50,
            "w_methodological_alignment": 0.05,
            "w_domain_alignment": 0.05,
            "w_topic": 0.0,
        }

        result = direct_rank_prod(
            engine,
            tenant_id=tenant_id,
            query_text=req.query_text,
            context_json={},
            filters_json={},
            rank_params_json=rank_params_json,
        )

        status = (result.get("job") or {}).get("status")
        if status in ("pending", "running"):
            return JSONResponse(
                status_code=202,
                content={"rank_job_id": str(result["rank_job_id"]), "status": status},
            )
        return result

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg == "candidate_set_not_found":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in drill_down endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{rank_job_id}/generate-subtopics", response_model=SubtopicsResponse)
def generate_subtopics_endpoint(
    rank_job_id: UUID,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Generate subtopics for a broad query's ranked results.

    This clusters the ranked papers by topic and generates human-readable
    labels for each cluster. Use this to offer specific focus areas when
    a user searches for a broad topic like "machine learning".

    Each subtopic includes:
    - A short label (e.g., "Transformers & Attention")
    - A description
    - Representative papers
    - Paper count
    - A drill_down_query for fetching dedicated papers via /rank

    To get dedicated papers for a subtopic, call:
    POST /v1/rank with {"query_text": subtopic.drill_down_query}
    """
    try:
        # Fetch the original query_text from the rank job's candidate set
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT cs.seed_json->>'query_text' as query_text
                    FROM rank_jobs rj
                    JOIN candidate_sets cs ON cs.candidate_set_id = rj.candidate_set_id
                    WHERE rj.rank_job_id = :rank_job_id
                """),
                {"rank_job_id": rank_job_id},
            ).mappings().first()
            query_text = row["query_text"] if row else None

        result = generate_subtopics(
            engine,
            rank_job_id=rank_job_id,
            query_text=query_text,
        )

        return SubtopicsResponse(
            rank_job_id=result["rank_job_id"],
            query_specificity=result["query_specificity"],
            subtopics=[
                SubtopicSuggestion(
                    subtopic_id=s["subtopic_id"],
                    label=s["label"],
                    description=s["description"],
                    representative_work_ids=s["representative_work_ids"],
                    paper_count=s["paper_count"],
                    drill_down_query=s.get("drill_down_query", s["label"]),
                )
                for s in result["subtopics"]
            ],
        )

    except ValueError as e:
        msg = str(e)
        if "not_found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in generate_subtopics endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{rank_job_id}/temporal-map", response_model=TemporalMapResponse)
def get_temporal_map_endpoint(
    rank_job_id: UUID,
    subtopic_id: Optional[str] = None,
    include_analytics: bool = False,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Get a temporal map of ranked results grouped by era (decade).

    This endpoint builds a temporal visualization showing how papers
    are distributed across time periods, identifying milestone papers
    and optionally including breakthrough/evolution analytics.

    Args:
        rank_job_id: ID of the rank job to visualize
        subtopic_id: Optional subtopic to filter results
        include_analytics: If true, includes breakthrough detection and
            evolution trend analysis (requires additional LLM calls)
    """
    try:
        result = build_temporal_map(
            engine,
            rank_job_id=rank_job_id,
            subtopic_id=subtopic_id,
            include_analytics=include_analytics,
        )

        # Convert nested dicts to Pydantic models
        eras = []
        for era_data in result.get("eras", []):
            papers = [
                TemporalPaper(
                    work_id=p["work_id"],
                    title=p.get("title"),
                    year=p.get("year"),
                    cited_by_count=p.get("cited_by_count"),
                    is_milestone=p.get("is_milestone", False),
                    rank_in_results=p.get("rank_in_results"),
                )
                for p in era_data.get("papers", [])
            ]
            eras.append(TemporalEra(
                label=era_data["label"],
                start_year=era_data["start_year"],
                end_year=era_data["end_year"],
                papers=papers,
                milestone_count=era_data.get("milestone_count", 0),
                is_breakthrough_era=era_data.get("is_breakthrough_era", False),
            ))

        analytics = None
        if result.get("analytics"):
            raw_analytics = result["analytics"]

            breakthrough = None
            if raw_analytics.get("breakthrough"):
                b = raw_analytics["breakthrough"]
                # Convert breakthrough_papers to BreakthroughPaper objects
                raw_papers = b.get("breakthrough_papers", [])
                breakthrough_papers = [
                    BreakthroughPaper(
                        work_id=p.get("work_id") if isinstance(p, dict) else p,
                        title=p.get("title") if isinstance(p, dict) else None,
                        cited_by_count=p.get("cited_by_count", 0) if isinstance(p, dict) else 0,
                    )
                    for p in raw_papers
                ]
                # Convert citation dicts to Citation objects
                shift_citations = [
                    Citation(claim=c.get("claim", ""), paper_title=c.get("paper_title", ""))
                    for c in b.get("shift_citations", [])
                    if isinstance(c, dict)
                ]
                rejection_citations = [
                    Citation(claim=c.get("claim", ""), paper_title=c.get("paper_title", ""))
                    for c in b.get("rejection_citations", [])
                    if isinstance(c, dict)
                ]
                breakthrough = BreakthroughAnalysis(
                    breakthrough_year=b.get("breakthrough_year"),
                    from_era=b.get("from_era"),
                    breakthrough_era=b.get("breakthrough_era"),
                    breakthrough_papers=breakthrough_papers,
                    pre_breakthrough_approach=b.get("pre_breakthrough_approach"),
                    post_breakthrough_approach=b.get("post_breakthrough_approach"),
                    shift_description=b.get("shift_description"),
                    shift_citations=shift_citations,
                    rejection_reason=b.get("rejection_reason"),
                    rejection_citations=rejection_citations,
                    confidence=b.get("confidence", 0.0),
                )

            evolution = [
                EvolutionTrend(
                    era=e.get("era", ""),
                    dominant_approach=e.get("dominant_approach", ""),
                    key_themes=e.get("key_themes", []),
                    representative_paper_id=e.get("representative_paper_id"),
                )
                for e in raw_analytics.get("evolution", [])
            ]

            analytics = TemporalMapAnalytics(
                breakthrough=breakthrough,
                evolution=evolution,
            )

        return TemporalMapResponse(
            rank_job_id=result["rank_job_id"],
            scope=result["scope"],
            scope_label=result["scope_label"],
            topic_id=result.get("topic_id"),
            subtopic_id=result.get("subtopic_id"),
            eras=eras,
            analytics=analytics,
        )

    except ValueError as e:
        msg = str(e)
        if "not_found" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.exception("Unexpected error in temporal_map endpoint")
        raise HTTPException(status_code=500, detail=str(e))
