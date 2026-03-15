"""Admin authorization dependency."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.db import make_engine


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    return make_engine()


def require_admin(
    user_id: Optional[UUID] = Depends(get_current_user_id),
    engine: Engine = Depends(_get_engine),
) -> UUID:
    """Dependency that enforces admin access. Returns the admin user_id."""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT is_admin FROM public.users WHERE user_id = :uid"),
            {"uid": user_id},
        ).mappings().first()

    if not row or not row["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")

    return user_id
