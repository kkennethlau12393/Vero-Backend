"""Paper tagging endpoints — user-defined tags on papers, workspace-scoped."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.db import make_engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Lazily construct the database engine once per process."""
    return make_engine()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/tags", tags=["tags"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateTagRequest(BaseModel):
    workspace_id: UUID
    paper_id: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1, max_length=100)
    color: Optional[str] = "#6b7280"
    title: Optional[str] = None


def _format_tag(row: dict) -> dict:
    """Map DB columns to frontend field names."""
    return {
        "id": str(row["id"]),
        "paper_id": row["work_id"],
        "label": row["tag"],
        "color": row.get("color") or "#6b7280",
        "title": row.get("title"),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
def list_user_tags(
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """List all unique tags for the current user (for autocomplete)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT tag, color
                FROM paper_tags
                WHERE user_id = :uid
                ORDER BY tag
            """),
            {"uid": user_id},
        ).mappings().all()

    return [dict(r) for r in rows]


@router.get("/all")
def list_all_user_tags(
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Get all tags for the current user across all workspaces."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, workspace_id, work_id, tag, color, title, created_at
                FROM paper_tags
                WHERE user_id = :uid
                ORDER BY tag, created_at DESC
            """),
            {"uid": user_id},
        ).mappings().all()

    return [
        {
            "id": str(r["id"]),
            "workspace_id": str(r["workspace_id"]),
            "paper_id": r["work_id"],
            "label": r["tag"],
            "color": r.get("color") or "#6b7280",
            "title": r.get("title"),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


@router.get("/workspace/{workspace_id}")
def list_workspace_tags(
    workspace_id: UUID,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Get all tags for papers in a workspace, for the current user."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, work_id, tag, color, title, created_at
                FROM paper_tags
                WHERE workspace_id = :ws AND user_id = :uid
                ORDER BY created_at DESC
            """),
            {"ws": workspace_id, "uid": user_id},
        ).mappings().all()

    return [_format_tag(dict(r)) for r in rows]


@router.post("/", status_code=201)
def create_tag(
    req: CreateTagRequest,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Add a tag to a paper. Returns the tag row (existing or new)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    tag_text = req.tag.strip()
    if not tag_text:
        raise HTTPException(status_code=400, detail="Tag cannot be empty")

    with engine.connect() as conn:
        # Insert, ignoring duplicate
        conn.execute(
            text("""
                INSERT INTO paper_tags (workspace_id, user_id, work_id, tag, color, title)
                VALUES (:ws, :uid, :wid, :tag, :color, :title)
                ON CONFLICT (workspace_id, user_id, work_id, tag)
                DO UPDATE SET title = COALESCE(EXCLUDED.title, paper_tags.title)
            """),
            {
                "ws": req.workspace_id,
                "uid": user_id,
                "wid": req.paper_id,
                "tag": tag_text,
                "color": req.color,
                "title": req.title,
            },
        )

        # Always fetch back the row (new or existing)
        row = conn.execute(
            text("""
                SELECT id, workspace_id, user_id, work_id, tag, color, title, created_at
                FROM paper_tags
                WHERE workspace_id = :ws AND user_id = :uid
                  AND work_id = :wid AND tag = :tag
            """),
            {
                "ws": req.workspace_id,
                "uid": user_id,
                "wid": req.paper_id,
                "tag": tag_text,
            },
        ).mappings().first()

        conn.commit()

    return _format_tag(dict(row))


@router.delete("/{tag_id}", status_code=204)
def delete_tag(
    tag_id: UUID,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Remove a tag. Only the owning user can delete."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM paper_tags WHERE id = :tid AND user_id = :uid"),
            {"tid": tag_id, "uid": user_id},
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Tag not found")

    return JSONResponse(status_code=204, content=None)
