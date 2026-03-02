"""JWT-based user identification for Supabase auth.

Decodes the Supabase JWT from the Authorization header to extract
the user_id (sub claim). Returns None in dev mode when no token is
provided, to maintain backward compatibility.
"""
from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import UUID

import jwt
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")


def get_current_user_id(
    authorization: Optional[str] = Header(None),
) -> Optional[UUID]:
    """Extract user_id from Supabase JWT Bearer token.

    Returns None if no Authorization header (dev mode).
    Raises 401 if token is present but invalid.
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization[7:]  # Strip "Bearer "

    if not SUPABASE_JWT_SECRET:
        logger.warning("SUPABASE_JWT_SECRET not set — skipping JWT verification")
        return None

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256", "HS384", "HS512"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing sub claim")
        return UUID(user_id)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user_id in token")
