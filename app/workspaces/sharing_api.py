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
<body style="margin:0;padding:0;background-color:#f4f4f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f7;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="520" cellpadding="0" cellspacing="0" style="max-width:520px;width:100%;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
          <!-- Header -->
          <tr>
            <td style="background-color:#1a1a2e;padding:28px 32px;text-align:center;">
              <span style="color:#ffffff;font-size:22px;font-weight:700;letter-spacing:0.5px;">Alexandria</span>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="background-color:#ffffff;padding:36px 32px 28px;">
              <p style="margin:0 0 20px;font-size:16px;line-height:1.6;color:#333333;">
                <strong>{inviter_email}</strong> invited you to collaborate on a workspace:
              </p>
              <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;background-color:#f8f8fc;border-radius:6px;border-left:4px solid #1a1a2e;">
                <tr>
                  <td style="padding:16px 20px;">
                    <span style="font-size:18px;font-weight:600;color:#1a1a2e;">{workspace_name}</span>
                    <br>
                    <span style="font-size:13px;color:#888888;text-transform:uppercase;letter-spacing:0.5px;">Role: {role}</span>
                  </td>
                </tr>
              </table>
              <p style="margin:0 0 28px;font-size:14px;line-height:1.6;color:#555555;">
                Open Alexandria to view the shared workspace and start collaborating.
              </p>
              <!-- CTA Button -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center">
                    <a href="https://www.alexandrialabs.uk/dashboard"
                       style="display:inline-block;background-color:#1a1a2e;color:#ffffff;font-size:15px;font-weight:600;text-decoration:none;padding:14px 36px;border-radius:6px;">
                      Open Workspace
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="background-color:#fafafa;padding:20px 32px;text-align:center;border-top:1px solid #eeeef2;">
              <p style="margin:0;font-size:12px;color:#999999;">
                Alexandria Labs &middot; Academic research, mapped.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
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
