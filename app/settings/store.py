from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

_EMPTY_SETTINGS = {
    "institutional_proxy_prefix": None,
    "libkey_api_key": None,
    "libkey_library_id": None,
}


def _resolve_owner_user_id(conn: Connection, tenant_id: UUID) -> Optional[UUID]:
    """Look up the workspace owner's user_id from the workspaces table."""
    row = conn.execute(
        text("SELECT owner_user_id FROM workspaces WHERE workspace_id = :ws"),
        {"ws": str(tenant_id)},
    ).mappings().first()
    if row:
        return UUID(str(row["owner_user_id"]))
    return None


def load_tenant_settings(conn: Connection, tenant_id: UUID) -> dict:
    """Load settings via the workspace owner's user_settings row."""
    owner_id = _resolve_owner_user_id(conn, tenant_id)
    if owner_id is None:
        logger.warning("No workspace found for tenant_id=%s", tenant_id)
        return dict(_EMPTY_SETTINGS)

    row = conn.execute(
        text("""
            SELECT institutional_proxy_prefix, libkey_api_key, libkey_library_id
            FROM user_settings
            WHERE user_id = :user_id
        """),
        {"user_id": str(owner_id)},
    ).mappings().first()

    if not row:
        return dict(_EMPTY_SETTINGS)
    return dict(row)


def upsert_tenant_settings(
    conn: Connection,
    tenant_id: UUID,
    *,
    institutional_proxy_prefix: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    libkey_library_id: Optional[str] = None,
) -> None:
    """Upsert settings via the workspace owner's user_settings row."""
    owner_id = _resolve_owner_user_id(conn, tenant_id)
    if owner_id is None:
        logger.warning("No workspace found for tenant_id=%s; skipping upsert", tenant_id)
        return

    conn.execute(
        text("""
            INSERT INTO user_settings (user_id, institutional_proxy_prefix, libkey_api_key, libkey_library_id, updated_at)
            VALUES (:user_id, :proxy, :api_key, :library_id, now())
            ON CONFLICT (user_id) DO UPDATE SET
                institutional_proxy_prefix = COALESCE(:proxy, user_settings.institutional_proxy_prefix),
                libkey_api_key = COALESCE(:api_key, user_settings.libkey_api_key),
                libkey_library_id = COALESCE(:library_id, user_settings.libkey_library_id),
                updated_at = now()
        """),
        {
            "user_id": str(owner_id),
            "proxy": institutional_proxy_prefix,
            "api_key": libkey_api_key,
            "library_id": libkey_library_id,
        },
    )
    conn.commit()
