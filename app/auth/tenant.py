import os
from typing import Optional
from uuid import UUID

from fastapi import Header, HTTPException

from app.auth.jwt_user import get_current_user_id


def get_tenant_id(
    x_workspace_id: str | None = Header(None),
    authorization: Optional[str] = Header(None),
) -> UUID:
    """Resolve the workspace (tenant) ID from the request.

    Production: frontend sends X-Workspace-Id header with the workspace UUID
    and Authorization: Bearer <token> for user identity.

    When both are present, verifies the user is the owner or an active member
    of the workspace. When no Authorization header (dev mode), skips the check.
    """
    raw = x_workspace_id or os.getenv("DEV_TENANT_ID")
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Missing X-Workspace-Id header",
        )
    try:
        workspace_id = UUID(raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid X-Workspace-Id: must be a valid UUID",
        )

    # If JWT is present, verify the user has access to this workspace
    user_id = get_current_user_id(authorization)
    if user_id is not None:
        _verify_workspace_access(workspace_id, user_id)

    return workspace_id


def _verify_workspace_access(workspace_id: UUID, user_id: UUID) -> None:
    """Verify user is owner or active member. Raises 403 if not."""
    from app.db import make_engine
    from sqlalchemy import text

    engine = make_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT 1 FROM workspaces
                WHERE workspace_id = :ws AND owner_user_id = :uid
                UNION ALL
                SELECT 1 FROM workspace_members
                WHERE workspace_id = :ws AND user_id = :uid AND status = 'active'
                LIMIT 1
            """),
            {"ws": workspace_id, "uid": user_id},
        ).first()

    if not row:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this workspace",
        )
