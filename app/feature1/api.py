"""
API router for Feature 1: Citation Map Retrieval.
"""
from __future__ import annotations

from functools import lru_cache
import logging
import re
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.engine import Engine

from app.auth.tenant import get_tenant_id
from app.db import make_engine
from app.feature1.citation_map_service import build_citation_map
from app.feature1.pdf_parser import extract_metadata_from_pdf
from app.feature1.schemas import CitationMapRequest, CitationMapResponse

router = APIRouter(prefix="/v1", tags=["citation-map"])
logger = logging.getLogger(__name__)

_WORK_ID_LIKE_PATTERN = re.compile(r"^(W\d+|S2:[^/\s]+|AX:[^/\s]+)$", re.IGNORECASE)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


@router.post("/citation-map", response_model=CitationMapResponse)
def build_citation_map_endpoint(
    req: CitationMapRequest,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Build a citation graph around a seed paper.

    Multiple input modes supported:
    1. seed_work_id: Direct OpenAlex work ID (internal use)
    2. seed_doi: DOI from PDF metadata (primary - most PDFs have DOI)
    3. seed_title: Paper title (fallback - all PDFs have title)
    4. query_text: Natural language query to find best seed (exploratory search)

    Returns nodes and edges of the citation graph, plus optionally
    a graph_draft_id that can be passed to /v1/maps/build.
    """
    # Compatibility guard: allow older/newer clients that accidentally send work_id in seed_doi.
    if req.seed_doi and not req.seed_work_id and _WORK_ID_LIKE_PATTERN.match(req.seed_doi.strip()):
        req = req.model_copy(update={"seed_work_id": req.seed_doi.strip(), "seed_doi": None})

    # Validate input - exactly ONE seed identification mode required
    provided_modes = sum([
        bool(req.seed_work_id),
        bool(req.seed_doi),
        bool(req.seed_title),
        bool(req.query_text),
    ])

    if provided_modes == 0:
        raise HTTPException(
            status_code=400,
            detail="Must provide ONE of: seed_work_id, seed_doi, seed_title, or query_text",
        )
    elif provided_modes > 1:
        raise HTTPException(
            status_code=400,
            detail="Provide only ONE seed identification mode",
        )

    try:
        return build_citation_map(
            engine,
            tenant_id=tenant_id,
            request=req,
        )
    except ValueError as e:
        logger.warning("Citation map request validation/domain error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Citation map internal error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/citation-map/pdf", response_model=CitationMapResponse)
async def build_citation_map_from_pdf(
    pdf_file: UploadFile = File(...),
    citing_limit: int = 15,
    references_limit: int = 15,
    min_citations: int = 0,
    total_nodes: int | None = None,
    create_graph_draft: bool = True,
    engine: Engine = Depends(get_engine),
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Build a citation graph from an uploaded PDF file.

    Workflow:
    1. Extract title and year from PDF metadata
    2. Match PDF to OpenAlex work_id using title search
    3. Build citation map around the matched paper

    Returns nodes and edges of the citation graph, plus optionally
    a graph_draft_id that can be passed to /v1/maps/build.
    """
    # Validate file type
    if not pdf_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Read PDF bytes
        pdf_bytes = await pdf_file.read()

        # Extract metadata from PDF
        metadata = extract_metadata_from_pdf(pdf_bytes)
        title = metadata.get("title")
        arxiv_id = metadata.get("arxiv_id")

        if not title and not arxiv_id:
            raise HTTPException(
                status_code=400,
                detail="Could not extract title from PDF. Please ensure the PDF has readable text."
            )

        # Build request: prefer ArXiv DOI (most reliable), fall back to title
        if arxiv_id:
            try:
                req = CitationMapRequest(
                    seed_doi=f"10.48550/arXiv.{arxiv_id}",
                    citing_limit=citing_limit,
                    references_limit=references_limit,
                    min_citations=min_citations,
                    total_nodes=total_nodes,
                    create_graph_draft=create_graph_draft,
                )
                return build_citation_map(engine, tenant_id=tenant_id, request=req)
            except ValueError:
                # ArXiv DOI not found in OpenAlex/S2 — fall back to title
                if not title:
                    raise
                # Fall through to title-based search below

        if not title:
            raise HTTPException(
                status_code=400,
                detail="Could not extract title or ArXiv ID from PDF.",
            )

        req = CitationMapRequest(
            seed_title=title,
            citing_limit=citing_limit,
            references_limit=references_limit,
            min_citations=min_citations,
            total_nodes=total_nodes,
            create_graph_draft=create_graph_draft,
        )
        return build_citation_map(engine, tenant_id=tenant_id, request=req)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
