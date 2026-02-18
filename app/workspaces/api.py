from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.tenant import get_workspace_id
from app.db import make_engine

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


def get_request_user_id(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> UUID:
    raw = x_user_id or os.getenv("DEV_USER_ID")
    if not raw:
        raise HTTPException(status_code=401, detail="Missing user context (X-User-Id or DEV_USER_ID)")
    try:
        return UUID(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-User-Id must be a valid UUID") from exc


class WorkspaceCreateRequest(BaseModel):
    workspace_name: str


class MemberCreateRequest(BaseModel):
    user_id: UUID
    role: Literal["owner", "admin", "member"] = "member"
    status: Literal["active", "invited", "suspended"] = "active"


class MemberUpdateRequest(BaseModel):
    role: Literal["owner", "admin", "member"] | None = None
    status: Literal["active", "invited", "suspended"] | None = None


def _require_admin_or_owner(conn, workspace_id: UUID, actor_user_id: UUID) -> None:
    row = conn.execute(
        text(
            """
            SELECT role, status
            FROM workspace_members
            WHERE workspace_id = :workspace_id
              AND user_id = :actor_user_id
            LIMIT 1
            """
        ),
        {"workspace_id": workspace_id, "actor_user_id": actor_user_id},
    ).mappings().first()
    if not row or row["status"] != "active" or row["role"] not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Workspace admin or owner role required")


@router.get("")
def list_workspaces(
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    w.workspace_id,
                    w.workspace_name,
                    w.owner_user_id,
                    wm.role,
                    wm.status
                FROM workspaces w
                JOIN workspace_members wm ON wm.workspace_id = w.workspace_id
                WHERE wm.user_id = :actor_user_id
                ORDER BY w.created_at DESC
                """
            ),
            {"actor_user_id": actor_user_id},
        ).mappings().all()
    return [dict(r) for r in rows]


@router.post("")
def create_workspace(
    req: WorkspaceCreateRequest,
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    workspace_name = req.workspace_name.strip()
    if not workspace_name:
        raise HTTPException(status_code=400, detail="workspace_name cannot be blank")

    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO workspaces (owner_user_id, workspace_name)
                VALUES (:actor_user_id, :workspace_name)
                RETURNING workspace_id, workspace_name, owner_user_id
                """
            ),
            {"actor_user_id": actor_user_id, "workspace_name": workspace_name},
        ).mappings().first()

        conn.execute(
            text(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, status)
                VALUES (:workspace_id, :actor_user_id, 'owner', 'active')
                ON CONFLICT (workspace_id, user_id) DO UPDATE
                SET role = 'owner', status = 'active'
                """
            ),
            {"workspace_id": row["workspace_id"], "actor_user_id": actor_user_id},
        )

    return dict(row)


@router.get("/{workspace_id}/members")
def list_workspace_members(
    workspace_id: UUID,
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    with engine.connect() as conn:
        membership = conn.execute(
            text(
                """
                SELECT 1
                FROM workspace_members
                WHERE workspace_id = :workspace_id
                  AND user_id = :actor_user_id
                  AND status = 'active'
                LIMIT 1
                """
            ),
            {"workspace_id": workspace_id, "actor_user_id": actor_user_id},
        ).first()
        if not membership:
            raise HTTPException(status_code=403, detail="Not a member of this workspace")

        rows = conn.execute(
            text(
                """
                SELECT user_id, workspace_id, role, status, created_at, updated_at
                FROM workspace_members
                WHERE workspace_id = :workspace_id
                ORDER BY created_at ASC
                """
            ),
            {"workspace_id": workspace_id},
        ).mappings().all()
    return [dict(r) for r in rows]


@router.post("/{workspace_id}/members")
def add_workspace_member(
    workspace_id: UUID,
    req: MemberCreateRequest,
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    with engine.begin() as conn:
        _require_admin_or_owner(conn, workspace_id, actor_user_id)
        conn.execute(
            text(
                """
                INSERT INTO workspace_members (workspace_id, user_id, role, status)
                VALUES (:workspace_id, :user_id, :role, :status)
                ON CONFLICT (workspace_id, user_id) DO UPDATE
                SET role = EXCLUDED.role,
                    status = EXCLUDED.status
                """
            ),
            {
                "workspace_id": workspace_id,
                "user_id": req.user_id,
                "role": req.role,
                "status": req.status,
            },
        )
    return {"status": "ok"}


@router.patch("/{workspace_id}/members/{user_id}")
def update_workspace_member(
    workspace_id: UUID,
    user_id: UUID,
    req: MemberUpdateRequest,
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    if req.role is None and req.status is None:
        raise HTTPException(status_code=400, detail="Provide role and/or status")

    with engine.begin() as conn:
        _require_admin_or_owner(conn, workspace_id, actor_user_id)

        current = conn.execute(
            text(
                """
                SELECT role
                FROM workspace_members
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                LIMIT 1
                """
            ),
            {"workspace_id": workspace_id, "user_id": user_id},
        ).mappings().first()
        if not current:
            raise HTTPException(status_code=404, detail="Member not found")

        if current["role"] == "owner" and req.role != "owner":
            owners = conn.execute(
                text(
                    """
                    SELECT count(*) AS owner_count
                    FROM workspace_members
                    WHERE workspace_id = :workspace_id
                      AND role = 'owner'
                    """
                ),
                {"workspace_id": workspace_id},
            ).mappings().first()
            if int(owners["owner_count"]) <= 1:
                raise HTTPException(status_code=400, detail="Cannot demote the last owner")

        conn.execute(
            text(
                """
                UPDATE workspace_members
                SET role = COALESCE(:role, role),
                    status = COALESCE(:status, status)
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                """
            ),
            {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "role": req.role,
                "status": req.status,
            },
        )

    return {"status": "ok"}


@router.delete("/{workspace_id}/members/{user_id}")
def remove_workspace_member(
    workspace_id: UUID,
    user_id: UUID,
    _workspace_context: UUID = Depends(get_workspace_id),
    actor_user_id: UUID = Depends(get_request_user_id),
    engine: Engine = Depends(get_engine),
):
    with engine.begin() as conn:
        _require_admin_or_owner(conn, workspace_id, actor_user_id)

        current = conn.execute(
            text(
                """
                SELECT role
                FROM workspace_members
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                LIMIT 1
                """
            ),
            {"workspace_id": workspace_id, "user_id": user_id},
        ).mappings().first()
        if not current:
            raise HTTPException(status_code=404, detail="Member not found")

        if current["role"] == "owner":
            owners = conn.execute(
                text(
                    """
                    SELECT count(*) AS owner_count
                    FROM workspace_members
                    WHERE workspace_id = :workspace_id
                      AND role = 'owner'
                    """
                ),
                {"workspace_id": workspace_id},
            ).mappings().first()
            if int(owners["owner_count"]) <= 1:
                raise HTTPException(status_code=400, detail="Cannot remove the last owner")

        conn.execute(
            text(
                """
                DELETE FROM workspace_members
                WHERE workspace_id = :workspace_id
                  AND user_id = :user_id
                """
            ),
            {"workspace_id": workspace_id, "user_id": user_id},
        )

    return {"status": "ok"}
