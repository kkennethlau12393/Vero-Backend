from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection


def load_workspace_settings(conn: Connection, workspace_id: UUID) -> dict:
    row = conn.execute(
        text("""
            SELECT institutional_proxy_prefix, libkey_api_key, libkey_library_id
            FROM workspace_settings
            WHERE workspace_id = :workspace_id
        """),
        {"workspace_id": str(workspace_id)},
    ).mappings().first()

    if not row:
        return {
            "institutional_proxy_prefix": None,
            "libkey_api_key": None,
            "libkey_library_id": None,
        }
    return dict(row)


def upsert_workspace_settings(
    conn: Connection,
    workspace_id: UUID,
    *,
    institutional_proxy_prefix: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    libkey_library_id: Optional[str] = None,
) -> None:
    conn.execute(
        text("""
            INSERT INTO workspace_settings (workspace_id, institutional_proxy_prefix, libkey_api_key, libkey_library_id, updated_at)
            VALUES (:workspace_id, :proxy, :api_key, :library_id, now())
            ON CONFLICT (workspace_id) DO UPDATE SET
                institutional_proxy_prefix = COALESCE(:proxy, workspace_settings.institutional_proxy_prefix),
                libkey_api_key = COALESCE(:api_key, workspace_settings.libkey_api_key),
                libkey_library_id = COALESCE(:library_id, workspace_settings.libkey_library_id),
                updated_at = now()
        """),
        {
            "workspace_id": str(workspace_id),
            "proxy": institutional_proxy_prefix,
            "api_key": libkey_api_key,
            "library_id": libkey_library_id,
        },
    )
    conn.commit()


def load_tenant_settings(conn: Connection, tenant_id: UUID) -> dict:
    """Legacy alias during workspace migration."""
    return load_workspace_settings(conn, tenant_id)


def upsert_tenant_settings(
    conn: Connection,
    tenant_id: UUID,
    *,
    institutional_proxy_prefix: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    libkey_library_id: Optional[str] = None,
) -> None:
    """Legacy alias during workspace migration."""
    upsert_workspace_settings(
        conn,
        tenant_id,
        institutional_proxy_prefix=institutional_proxy_prefix,
        libkey_api_key=libkey_api_key,
        libkey_library_id=libkey_library_id,
    )
