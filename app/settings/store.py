from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection


def load_tenant_settings(conn: Connection, tenant_id: UUID) -> dict:
    row = conn.execute(
        text("""
            SELECT institutional_proxy_prefix, libkey_api_key, libkey_library_id
            FROM tenant_settings
            WHERE tenant_id = :tenant_id
        """),
        {"tenant_id": str(tenant_id)},
    ).mappings().first()

    if not row:
        return {
            "institutional_proxy_prefix": None,
            "libkey_api_key": None,
            "libkey_library_id": None,
        }
    return dict(row)


def upsert_tenant_settings(
    conn: Connection,
    tenant_id: UUID,
    *,
    institutional_proxy_prefix: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    libkey_library_id: Optional[str] = None,
) -> None:
    conn.execute(
        text("""
            INSERT INTO tenant_settings (tenant_id, institutional_proxy_prefix, libkey_api_key, libkey_library_id, updated_at)
            VALUES (:tenant_id, :proxy, :api_key, :library_id, now())
            ON CONFLICT (tenant_id) DO UPDATE SET
                institutional_proxy_prefix = COALESCE(:proxy, tenant_settings.institutional_proxy_prefix),
                libkey_api_key = COALESCE(:api_key, tenant_settings.libkey_api_key),
                libkey_library_id = COALESCE(:library_id, tenant_settings.libkey_library_id),
                updated_at = now()
        """),
        {
            "tenant_id": str(tenant_id),
            "proxy": institutional_proxy_prefix,
            "api_key": libkey_api_key,
            "library_id": libkey_library_id,
        },
    )
    conn.commit()
