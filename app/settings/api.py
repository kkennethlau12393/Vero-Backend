from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.auth.tenant import get_workspace_id
from app.db import make_engine
from app.settings.store import load_workspace_settings, upsert_workspace_settings

router = APIRouter(prefix="/v1/settings", tags=["settings"])

_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


class InstitutionalAccessSettings(BaseModel):
    institutional_proxy_prefix: Optional[str] = None
    libkey_api_key: Optional[str] = None
    libkey_library_id: Optional[str] = None


class InstitutionalAccessResponse(BaseModel):
    institutional_proxy_prefix: Optional[str] = None
    has_libkey: bool = False
    libkey_library_id: Optional[str] = None


@router.get("/institutional-access", response_model=InstitutionalAccessResponse)
def get_institutional_access(
    workspace_id: UUID = Depends(get_workspace_id),
    engine: Engine = Depends(_get_engine),
):
    with engine.connect() as conn:
        settings = load_workspace_settings(conn, workspace_id)
    return InstitutionalAccessResponse(
        institutional_proxy_prefix=settings["institutional_proxy_prefix"],
        has_libkey=bool(settings["libkey_api_key"]),
        libkey_library_id=settings["libkey_library_id"],
    )


@router.put("/institutional-access")
def update_institutional_access(
    body: InstitutionalAccessSettings,
    workspace_id: UUID = Depends(get_workspace_id),
    engine: Engine = Depends(_get_engine),
):
    with engine.connect() as conn:
        upsert_workspace_settings(
            conn,
            workspace_id,
            institutional_proxy_prefix=body.institutional_proxy_prefix,
            libkey_api_key=body.libkey_api_key,
            libkey_library_id=body.libkey_library_id,
        )
    return {"status": "ok"}
