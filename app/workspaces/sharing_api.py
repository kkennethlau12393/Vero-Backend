"""Workspace sharing API — invite members, list, update roles, remove.

All endpoints require JWT auth to identify the requesting user.
Owner-only operations check workspaces.owner_user_id.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional
from uuid import UUID

import requests as http_requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.billing.credits import ensure_billing_tables
from app.db import make_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/workspaces", tags=["sharing"])


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InviteRequest(BaseModel):
    email: str
    role: str  # 'viewer' or 'editor'


class MemberResponse(BaseModel):
    user_id: str
    email: str
    role: str
    status: str


class UpdateRoleRequest(BaseModel):
    role: str  # 'viewer' or 'editor'


class CreateWorkspaceRequest(BaseModel):
    workspace_name: str
    input_mode: str | None = None
    input_query: str | None = None
    entry_type: str = "ranked"


class CreateWorkspaceResponse(BaseModel):
    workspace_id: str
    workspace_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_user(user_id: Optional[UUID]) -> UUID:
    """Raise 401 if no authenticated user."""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


def _get_workspace_owner(conn, workspace_id: UUID) -> UUID:
    """Return owner_user_id or raise 404."""
    row = conn.execute(
        text("SELECT owner_user_id FROM workspaces WHERE workspace_id = :ws"),
        {"ws": workspace_id},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return UUID(str(row["owner_user_id"]))


def _require_owner(conn, workspace_id: UUID, user_id: UUID) -> None:
    """Raise 403 if user is not the workspace owner."""
    owner_id = _get_workspace_owner(conn, workspace_id)
    if owner_id != user_id:
        raise HTTPException(status_code=403, detail="Only the workspace owner can perform this action")


def _is_owner_or_member(conn, workspace_id: UUID, user_id: UUID) -> bool:
    """Check if user is owner or active member."""
    owner_id = _get_workspace_owner(conn, workspace_id)
    if owner_id == user_id:
        return True
    member = conn.execute(
        text("""
            SELECT 1 FROM workspace_members
            WHERE workspace_id = :ws AND user_id = :uid AND status = 'active'
        """),
        {"ws": workspace_id, "uid": user_id},
    ).first()
    return member is not None


def _validate_role(role: str) -> None:
    if role not in ("viewer", "editor"):
        raise HTTPException(status_code=400, detail="Role must be 'viewer' or 'editor'")


def _send_invite_email(
    to_email: str,
    inviter_email: str,
    workspace_name: str,
    role: str,
) -> None:
    """Send workspace invite email via Resend. Best-effort — never raises."""
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        return

    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background-color:#f9fafb;font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;-webkit-font-smoothing:antialiased;">
  <div style="max-width:480px;margin:0 auto;padding:40px 24px;">

    <div style="background-color:#ffffff !important;border-radius:16px;box-shadow:0 20px 25px -5px rgba(0,0,0,0.1),0 8px 10px -6px rgba(0,0,0,0.1);padding:32px;text-align:center;">

      <img src="https://www.alexandrialabs.uk/Alexandria_logo_black_no_background.png" alt="Alexandria" height="36" style="margin:0 auto 28px;" />

      <h1 style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:1.875rem;font-weight:700;color:#111827 !important;margin:0 0 8px;letter-spacing:-0.025em;">
        You're invited
      </h1>

      <p style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:1rem;color:#4b5563 !important;line-height:1.6;margin:0 0 24px;">
        <strong style="color:#111827 !important;">{inviter_email}</strong> invited you to collaborate on a workspace.
      </p>

      <div style="background-color:#f9fafb !important;border-radius:12px;padding:20px;margin:0 0 24px;text-align:center;">
        <p style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:1.25rem;font-weight:700;color:#111827 !important;margin:0 0 4px;">
          {workspace_name}
        </p>
        <p style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:0.75rem;color:#9ca3af !important;margin:0;text-transform:uppercase;letter-spacing:0.05em;">
          Role: {role}
        </p>
      </div>

      <a href="https://www.alexandrialabs.uk/dashboard"
         style="display:inline-block;background-color:#000000 !important;color:#ffffff !important;font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:0.875rem;font-weight:700;text-decoration:none;padding:12px 40px;border-radius:8px;">
        Open Workspace
      </a>

      <p style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:0.75rem;color:#9ca3af !important;margin:28px 0 0;line-height:1.5;">
        Didn't expect this invite? You can safely ignore this email.
      </p>

    </div>

    <p style="font-family:'Playfair Display',Georgia,Cambria,'Times New Roman',serif;font-size:0.75rem;color:#9ca3af !important;text-align:center;margin:20px 0 0;">
      &copy; 2026 Alexandria Limited
    </p>

  </div>
</body>
</html>"""

    try:
        resp = http_requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": "Alexandria <noreply@alexandrialabs.uk>",
                "to": to_email,
                "subject": f"{inviter_email} invited you to a workspace on Alexandria",
                "html": html,
            },
            timeout=10,
        )
        if resp.status_code >= 400:
            logger.warning("Resend API error %s: %s", resp.status_code, resp.text)
    except Exception:
        logger.exception("Failed to send invite email to %s", to_email)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{workspace_id}/invite", response_model=MemberResponse)
def invite_member(
    workspace_id: UUID,
    req: InviteRequest,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Invite a user to a workspace by email. Owner only."""
    uid = _require_user(user_id)
    _validate_role(req.role)

    with engine.connect() as conn:
        _require_owner(conn, workspace_id, uid)

        # Look up email in auth.users
        target_row = conn.execute(
            text("SELECT id, email FROM auth.users WHERE email = :email"),
            {"email": req.email},
        ).mappings().first()
        if not target_row:
            raise HTTPException(status_code=404, detail="user_not_found")

        target_user_id = target_row["id"]

        # Check if already a member
        existing = conn.execute(
            text("""
                SELECT 1 FROM workspace_members
                WHERE workspace_id = :ws AND user_id = :uid
            """),
            {"ws": workspace_id, "uid": target_user_id},
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="already_member")

        # Insert member
        conn.execute(
            text("""
                INSERT INTO workspace_members (workspace_id, user_id, role, status)
                VALUES (:ws, :uid, :role, 'active')
            """),
            {"ws": workspace_id, "uid": target_user_id, "role": req.role},
        )
        conn.commit()

        # Fetch inviter email and workspace name for the invite email
        inviter_row = conn.execute(
            text("SELECT email FROM auth.users WHERE id = :uid"),
            {"uid": uid},
        ).mappings().first()
        ws_row = conn.execute(
            text("SELECT workspace_name FROM workspaces WHERE workspace_id = :ws"),
            {"ws": workspace_id},
        ).mappings().first()

        inviter_email = inviter_row["email"] if inviter_row else "A teammate"
        workspace_name = ws_row["workspace_name"] if ws_row else "Untitled Workspace"

        _send_invite_email(req.email, inviter_email, workspace_name, req.role)

        return MemberResponse(
            user_id=str(target_user_id),
            email=req.email,
            role=req.role,
            status="active",
        )


@router.get("/{workspace_id}/members", response_model=list[MemberResponse])
def list_members(
    workspace_id: UUID,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """List all members of a workspace. Owner or member access."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        if not _is_owner_or_member(conn, workspace_id, uid):
            raise HTTPException(status_code=403, detail="Not a member of this workspace")

        # Get owner
        owner_row = conn.execute(
            text("""
                SELECT w.owner_user_id, u.email
                FROM workspaces w
                JOIN auth.users u ON u.id = w.owner_user_id
                WHERE w.workspace_id = :ws
            """),
            {"ws": workspace_id},
        ).mappings().first()

        members = []
        if owner_row:
            members.append(MemberResponse(
                user_id=str(owner_row["owner_user_id"]),
                email=owner_row["email"],
                role="owner",
                status="active",
            ))

        # Get members
        rows = conn.execute(
            text("""
                SELECT wm.user_id, u.email, wm.role, wm.status
                FROM workspace_members wm
                JOIN auth.users u ON u.id = wm.user_id
                WHERE wm.workspace_id = :ws AND wm.status = 'active'
            """),
            {"ws": workspace_id},
        ).mappings().all()

        for row in rows:
            members.append(MemberResponse(
                user_id=str(row["user_id"]),
                email=row["email"],
                role=row["role"],
                status=row["status"],
            ))

        return members


@router.patch("/{workspace_id}/members/{member_user_id}", response_model=MemberResponse)
def update_member_role(
    workspace_id: UUID,
    member_user_id: UUID,
    req: UpdateRoleRequest,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Update a member's role. Owner only."""
    uid = _require_user(user_id)
    _validate_role(req.role)

    with engine.connect() as conn:
        _require_owner(conn, workspace_id, uid)

        # Verify member exists
        row = conn.execute(
            text("""
                SELECT wm.user_id, u.email, wm.status
                FROM workspace_members wm
                JOIN auth.users u ON u.id = wm.user_id
                WHERE wm.workspace_id = :ws AND wm.user_id = :mid
            """),
            {"ws": workspace_id, "mid": member_user_id},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Member not found")

        conn.execute(
            text("""
                UPDATE workspace_members SET role = :role
                WHERE workspace_id = :ws AND user_id = :mid
            """),
            {"ws": workspace_id, "mid": member_user_id, "role": req.role},
        )
        conn.commit()

        return MemberResponse(
            user_id=str(member_user_id),
            email=row["email"],
            role=req.role,
            status=row["status"],
        )


@router.delete("/{workspace_id}/members/{member_user_id}", status_code=204)
def remove_member(
    workspace_id: UUID,
    member_user_id: UUID,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Remove a member. Owner can remove anyone; members can remove themselves."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        owner_id = _get_workspace_owner(conn, workspace_id)

        # Owner can remove anyone, member can only remove themselves
        if uid != owner_id and uid != member_user_id:
            raise HTTPException(status_code=403, detail="Only the owner or the member themselves can remove")

        result = conn.execute(
            text("""
                DELETE FROM workspace_members
                WHERE workspace_id = :ws AND user_id = :mid
            """),
            {"ws": workspace_id, "mid": member_user_id},
        )
        conn.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Member not found")


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------

@router.post("/create", response_model=CreateWorkspaceResponse)
def create_workspace(
    req: CreateWorkspaceRequest,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Create a new workspace. Free plan limited to 1 workspace ever."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        ensure_billing_tables(conn)

        # Upsert billing row
        conn.execute(
            text("""
                INSERT INTO user_billing (user_id)
                VALUES (:uid)
                ON CONFLICT (user_id) DO NOTHING
            """),
            {"uid": uid},
        )

        row = conn.execute(
            text("SELECT plan, workspaces_created_count FROM user_billing WHERE user_id = :uid"),
            {"uid": uid},
        ).mappings().first()

        if row["plan"] == "free" and row["workspaces_created_count"] >= 1:
            raise HTTPException(
                status_code=403,
                detail="Free plan limited to 1 workspace. Upgrade to Alexandria X for unlimited workspaces.",
            )

        # Create workspace
        ws_row = conn.execute(
            text("""
                INSERT INTO workspaces (owner_user_id, workspace_name, input_mode, input_query, entry_type)
                VALUES (:uid, :name, :mode, :query, :entry)
                RETURNING workspace_id, workspace_name
            """),
            {
                "uid": uid,
                "name": req.workspace_name.strip(),
                "mode": req.input_mode,
                "query": req.input_query,
                "entry": req.entry_type,
            },
        ).mappings().first()

        # Increment counter (never decremented)
        conn.execute(
            text("""
                UPDATE user_billing
                SET workspaces_created_count = workspaces_created_count + 1, updated_at = now()
                WHERE user_id = :uid
            """),
            {"uid": uid},
        )
        conn.commit()

        logger.info("Created workspace %s for user %s", ws_row["workspace_id"], uid)
        return CreateWorkspaceResponse(
            workspace_id=str(ws_row["workspace_id"]),
            workspace_name=ws_row["workspace_name"],
        )


@router.delete("/{workspace_id}", status_code=200)
def delete_workspace(
    workspace_id: UUID,
    engine: Engine = Depends(get_engine),
    user_id: Optional[UUID] = Depends(get_current_user_id),
):
    """Delete a workspace and all related data. Owner only. Does NOT decrement creation counter."""
    uid = _require_user(user_id)

    with engine.connect() as conn:
        _require_owner(conn, workspace_id, uid)

        ws_id = str(workspace_id)

        # Cascade delete all related data.
        # Tables with ON DELETE CASCADE from parent FKs are handled automatically
        # when we delete the parent rows (maps, rank_jobs, candidate_sets).
        # Tables without FK cascades must be deleted explicitly.

        # 1. Saved papers (keyed by workspace_id)
        conn.execute(text("DELETE FROM saved_papers WHERE workspace_id = :ws"), {"ws": workspace_id})

        # 2. Methodology caches (tenant_id is text in these tables)
        conn.execute(text("DELETE FROM methodology_comparison_cache WHERE tenant_id = :tid"), {"tid": ws_id})
        conn.execute(text("DELETE FROM methodology_comparisons WHERE rank_job_id IN (SELECT rank_job_id::text FROM rank_jobs WHERE tenant_id = :tid)"), {"tid": workspace_id})

        # 3. Research activity log
        conn.execute(text("DELETE FROM research_activity_log WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 4. Citation maps
        conn.execute(text("DELETE FROM citation_maps WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 5. Gap analysis (FK cascades from maps/rank_jobs, but also has tenant_id)
        conn.execute(text("DELETE FROM gap_analysis_results WHERE tenant_id = :tid"), {"tid": workspace_id})
        conn.execute(text("DELETE FROM gap_analysis_jobs WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 6. Maps (cascades: map_nodes, map_edges, gap_feature_usage)
        conn.execute(text("DELETE FROM maps WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 7. Rank results first (FK references rank_jobs without CASCADE)
        conn.execute(text(
            "DELETE FROM rank_results WHERE rank_job_id IN "
            "(SELECT rank_job_id FROM rank_jobs WHERE tenant_id = :tid)"
        ), {"tid": workspace_id})

        # 8. Rank jobs (now safe to delete)
        conn.execute(text("DELETE FROM rank_jobs WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 9. Graph drafts & candidate sets (cascades: candidate_set_items)
        conn.execute(text("DELETE FROM graph_drafts WHERE tenant_id = :tid"), {"tid": workspace_id})
        conn.execute(text("DELETE FROM candidate_sets WHERE tenant_id = :tid"), {"tid": workspace_id})

        # 10. Workspace members
        conn.execute(text("DELETE FROM workspace_members WHERE workspace_id = :ws"), {"ws": workspace_id})

        # 10. Workspace itself
        conn.execute(text("DELETE FROM workspaces WHERE workspace_id = :ws"), {"ws": workspace_id})

        conn.commit()
        logger.info("Deleted workspace %s and all related data for user %s", workspace_id, uid)

    return {"status": "deleted", "workspace_id": ws_id}
