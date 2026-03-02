"""JWT-based user identification for Supabase auth.

Decodes the Supabase JWT from the Authorization header to extract
the user_id (sub claim). Supports both asymmetric (RS256/ES256 via JWKS)
and legacy symmetric (HS256) verification.

Returns None in dev mode when no token is provided, to maintain
backward compatibility.
"""
from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import UUID

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")

# JWKS client for asymmetric key verification (RS256/ES256)
_jwks_client: Optional[PyJWKClient] = None
if SUPABASE_URL:
    _jwks_client = PyJWKClient(f"{SUPABASE_URL}/.well-known/jwks.json")


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

    if not SUPABASE_JWT_SECRET and not _jwks_client:
        logger.warning("No JWT verification configured — skipping")
        return None

    try:
        # Check token algorithm to decide verification method
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "")

        if alg in ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512"):
            # Asymmetric — verify with JWKS public key
            if not _jwks_client:
                raise HTTPException(status_code=401, detail="SUPABASE_URL not configured for JWKS verification")
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],
                audience="authenticated",
            )
        else:
            # Symmetric (HS256/384/512) — verify with shared secret
            if not SUPABASE_JWT_SECRET:
                raise HTTPException(status_code=401, detail="SUPABASE_JWT_SECRET not configured")
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
