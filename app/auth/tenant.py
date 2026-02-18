import os
from uuid import UUID

from fastapi import Header


def _parse_uuid(raw: str | None, field_name: str) -> UUID | None:
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a valid UUID") from exc


def get_workspace_id(
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> UUID:
    """Primary request scope identifier.

    Resolution order:
    1) X-Workspace-Id header
    2) X-Tenant-Id header (legacy compatibility)
    3) DEV_WORKSPACE_ID env var
    4) DEV_TENANT_ID env var (legacy compatibility)
    """
    for field_name, raw in (
        ("X-Workspace-Id", x_workspace_id),
        ("X-Tenant-Id", x_tenant_id),
        ("DEV_WORKSPACE_ID", os.getenv("DEV_WORKSPACE_ID")),
        ("DEV_TENANT_ID", os.getenv("DEV_TENANT_ID")),
    ):
        parsed = _parse_uuid(raw, field_name)
        if parsed:
            return parsed

    raise RuntimeError(
        "Workspace context is not set. Provide X-Workspace-Id header or set DEV_WORKSPACE_ID."
    )


def get_tenant_id(
    x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
) -> UUID:
    """Legacy alias for compatibility during workspace migration."""
    return get_workspace_id(
        x_workspace_id=x_workspace_id,
        x_tenant_id=x_tenant_id,
    )
