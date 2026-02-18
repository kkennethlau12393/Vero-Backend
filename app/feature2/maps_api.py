from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from uuid import UUID

from app.db import make_engine
from app.feature2.schemas import BuildMapRequest, BuildMapResponse, MapRenderResponse
from app.feature2.maps_service import build_map, get_map_render_payload
from app.auth.tenant import get_workspace_id

from functools import lru_cache
from typing import Optional

from sqlalchemy.engine import Engine

router = APIRouter(prefix="/v1/maps", tags=["maps"])

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()

@router.post("/build", response_model=BuildMapResponse)
def build_map_endpoint(
    req: BuildMapRequest,
    engine: Engine = Depends(get_engine),
    workspace_id: UUID = Depends(get_workspace_id),
):
    try:
        result = build_map(
            engine,
            workspace_id=workspace_id,
            graph_draft_id=req.graph_draft_id,
            connector_score_mode=req.connector_score_mode,
            layout_mode=req.layout_mode,
            grouping_policy_version=req.grouping_policy_version,
        )
        stats = result["stats"]
        return BuildMapResponse(
            map_id=result["map_id"],
            default_grouping=result["default_grouping"],
            allowed_groupings=result["allowed_groupings"],
            node_count=int(stats.get("node_count", 0)),
            edge_count=int(stats.get("edge_count", 0)),
            stats=stats,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg == "graph_draft_not_found":
            raise HTTPException(status_code=404, detail=msg)
        if msg == "graph_draft_empty":
            raise HTTPException(status_code=400, detail=msg)
        if msg == "graph_too_large_for_sync":
            raise HTTPException(status_code=413, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

@router.get("/{map_id}", response_model=MapRenderResponse)
def get_map_endpoint(
    map_id: UUID,
    group_by: Optional[str] = None,
    engine: Engine = Depends(get_engine),
    workspace_id: UUID = Depends(get_workspace_id),
):
    try:
        return get_map_render_payload(
            engine,
            workspace_id=workspace_id,
            map_id=map_id,
            group_by=group_by,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        msg = str(e)
        if msg == "map_not_found":
            raise HTTPException(status_code=404, detail=msg)
        if msg == "invalid_group_by":
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
