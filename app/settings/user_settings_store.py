from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection


def load_user_settings(conn: Connection, user_id: UUID) -> Optional[dict]:
    """Load user settings row. Returns dict or None if not found."""
    row = conn.execute(
        text("""
            SELECT libkey_library_id, libkey_api_key, institutional_proxy_prefix
            FROM user_settings
            WHERE user_id = :user_id
        """),
        {"user_id": str(user_id)},
    ).mappings().first()

    if not row:
        return None
    return dict(row)


def upsert_user_settings(
    conn: Connection,
    user_id: UUID,
    *,
    libkey_library_id: Optional[str] = None,
    libkey_api_key: Optional[str] = None,
    institutional_proxy_prefix: Optional[str] = None,
) -> None:
    """Insert or update user settings using COALESCE to preserve existing values."""
    conn.execute(
        text("""
            INSERT INTO user_settings (user_id, libkey_library_id, libkey_api_key, institutional_proxy_prefix, updated_at)
            VALUES (:user_id, :library_id, :api_key, :proxy, now())
            ON CONFLICT (user_id) DO UPDATE SET
                libkey_library_id = COALESCE(:library_id, user_settings.libkey_library_id),
                libkey_api_key = COALESCE(:api_key, user_settings.libkey_api_key),
                institutional_proxy_prefix = COALESCE(:proxy, user_settings.institutional_proxy_prefix),
                updated_at = now()
        """),
        {
            "user_id": str(user_id),
            "library_id": libkey_library_id,
            "api_key": libkey_api_key,
            "proxy": institutional_proxy_prefix,
        },
    )
    conn.commit()
