"""
User-scoped settings and access resolution endpoints.

US-006: GET/PUT /v1/user/institutional-access
US-007: GET /v1/settings/institutions
US-008: POST /v1/user/resolve-access
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.engine import Engine

from app.auth.jwt_user import get_current_user_id
from app.db import make_engine
from app.settings.access_links import resolve_access_link
from app.settings.schemas import (
    AccessResult,
    InstitutionItem,
    InstitutionListResponse,
    ResolveDoisRequest,
    ResolveAccessResponse,
    UserInstitutionalAccessResponse,
    UserInstitutionalAccessUpdate,
)
from app.settings.user_settings_store import load_user_settings, upsert_user_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["user-settings"])

# ---------------------------------------------------------------------------
# Shared engine (lazy singleton)
# ---------------------------------------------------------------------------
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


# ---------------------------------------------------------------------------
# US-007: In-memory cache for LibKey library list (24h TTL)
# ---------------------------------------------------------------------------
_institutions_cache: Optional[List[dict]] = None
_institutions_cache_ts: float = 0.0
_INSTITUTIONS_CACHE_TTL = 86_400  # 24 hours in seconds

LIBKEY_UNIVERSAL_API_KEY = os.environ.get("LIBKEY_UNIVERSAL_API_KEY", "")
_LIBKEY_PUBLIC_API_BASE = "https://public-api.thirdiron.com/public/v1"


# ---------------------------------------------------------------------------
# US-006: GET /v1/user/institutional-access
# ---------------------------------------------------------------------------

@router.get("/v1/user/institutional-access", response_model=UserInstitutionalAccessResponse)
def get_user_institutional_access(
    user_id: Optional[UUID] = Depends(get_current_user_id),
    engine: Engine = Depends(_get_engine),
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        settings = load_user_settings(conn, user_id)

    if not settings:
        return UserInstitutionalAccessResponse(libkey_library_id=None, has_libkey=False)

    library_id = settings.get("libkey_library_id")
    return UserInstitutionalAccessResponse(
        libkey_library_id=library_id,
        has_libkey=bool(library_id),
    )


# ---------------------------------------------------------------------------
# US-006: PUT /v1/user/institutional-access
# ---------------------------------------------------------------------------

@router.put("/v1/user/institutional-access")
def update_user_institutional_access(
    body: UserInstitutionalAccessUpdate,
    user_id: Optional[UUID] = Depends(get_current_user_id),
    engine: Engine = Depends(_get_engine),
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    with engine.connect() as conn:
        upsert_user_settings(
            conn,
            user_id,
            libkey_library_id=body.libkey_library_id,
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# US-007: GET /v1/settings/institutions
# ---------------------------------------------------------------------------

@router.get("/v1/settings/institutions", response_model=InstitutionListResponse)
async def list_institutions(
    q: Optional[str] = Query(None, description="Case-insensitive name filter"),
):
    """Public endpoint — returns LibKey active libraries with optional name filter."""
    institutions = await _get_institutions()

    if q:
        q_lower = q.lower()
        institutions = [i for i in institutions if q_lower in i["name"].lower()]

    return InstitutionListResponse(
        institutions=[InstitutionItem(**i) for i in institutions]
    )


async def _get_institutions() -> List[dict]:
    """Fetch and cache the active LibKey library list (24h TTL)."""
    global _institutions_cache, _institutions_cache_ts

    now = time.monotonic()
    if _institutions_cache is not None and (now - _institutions_cache_ts) < _INSTITUTIONS_CACHE_TTL:
        return _institutions_cache

    if not LIBKEY_UNIVERSAL_API_KEY:
        logger.warning("LIBKEY_UNIVERSAL_API_KEY not set; returning empty institution list")
        return []

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{_LIBKEY_PUBLIC_API_BASE}/libraries",
                params={"filter[libkey_subscription_status]": "active"},
                headers={"Authorization": f"Bearer {LIBKEY_UNIVERSAL_API_KEY}"},
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch LibKey institution list")
        # Return stale cache if available, else empty list
        return _institutions_cache or []

    raw_items = data.get("data", [])
    institutions = []
    for item in raw_items:
        attrs = item.get("attributes", {})
        institutions.append({
            "library_id": str(item.get("id", "")),
            "name": attrs.get("name", ""),
            "homepage_url": attrs.get("homepageUrl"),
            "image_url": attrs.get("imageUrl"),
        })

    _institutions_cache = institutions
    _institutions_cache_ts = now
    return institutions


# ---------------------------------------------------------------------------
# US-008: POST /v1/user/resolve-access
# ---------------------------------------------------------------------------

@router.post("/v1/user/resolve-access", response_model=ResolveAccessResponse)
async def resolve_user_access(
    body: ResolveDoisRequest,
    user_id: Optional[UUID] = Depends(get_current_user_id),
    engine: Engine = Depends(_get_engine),
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    if len(body.dois) > 50:
        raise HTTPException(status_code=422, detail="Maximum 50 DOIs per request")

    with engine.connect() as conn:
        user_settings = load_user_settings(conn, user_id)

    libkey_library_id = None
    libkey_api_key = None
    proxy_prefix = None

    if user_settings:
        libkey_library_id = user_settings.get("libkey_library_id")
        libkey_api_key = user_settings.get("libkey_api_key")
        proxy_prefix = user_settings.get("institutional_proxy_prefix")

    results: List[AccessResult] = []

    for doi in body.dois:
        resolved = await _resolve_access_link_for_user(
            doi=doi,
            libkey_library_id=libkey_library_id,
            libkey_api_key=libkey_api_key,
            proxy_prefix=proxy_prefix,
        )
        results.append(AccessResult(
            doi=doi,
            access_status=resolved["access_status"],
            pdf_url=resolved.get("pdf_url"),
            source=resolved.get("source"),
        ))
        # Rate-limit LibKey calls
        await asyncio.sleep(0.2)

    return ResolveAccessResponse(results=results)


async def _resolve_access_link_for_user(
    *,
    doi: str,
    libkey_library_id: Optional[str],
    libkey_api_key: Optional[str],
    proxy_prefix: Optional[str],
) -> dict:
    """Resolve access for a single DOI using user-level settings.

    Wraps the synchronous resolve_access_link() helper. OA status/url are not
    stored per-user, so we omit the OA fast-path and rely on LibKey + proxy.
    """
    return resolve_access_link(
        doi=doi,
        is_open_access=None,
        oa_pdf_url=None,
        proxy_prefix=proxy_prefix,
        libkey_api_key=libkey_api_key,
        libkey_library_id=libkey_library_id,
    )
