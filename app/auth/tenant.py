import os
from uuid import UUID

from fastapi import Header, HTTPException


def get_tenant_id(
    x_workspace_id: str | None = Header(None),
) -> UUID:
    """Resolve the workspace (tenant) ID from the request.

    Production: frontend sends X-Workspace-Id header with the workspace UUID.
    Dev fallback: reads DEV_TENANT_ID from env if no header is provided.
    """
    raw = x_workspace_id or os.getenv("DEV_TENANT_ID")
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Missing X-Workspace-Id header",
        )
    try:
        return UUID(raw)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid X-Workspace-Id: must be a valid UUID",
        )
