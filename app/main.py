from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

# app/main.py -> repo root is ../
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.feature1.api import router as feature1_router
from app.feature2.maps_api import router as feature2_maps_router
from app.feature2.rank_api import router as feature2_rank_router
from app.feature3.node_details_api import router as feature3_node_details_router
from app.feature4.compare_api import router as feature4_compare_router
from app.feature5.gap_api import router as feature5_gap_router
from app.settings.api import router as settings_router
from app.settings.user_api import router as user_settings_router
from app.workspaces.status_api import router as workspaces_router
from app.workspaces.sharing_api import router as sharing_router
from app.billing.billing_api import router as billing_router
from app.common.decomposition_api import router as decomposition_router
from app.tags.tags_api import router as tags_router
from app.admin.admin_api import router as admin_router
from app.admin.analytics_api import router as analytics_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(feature1_router)
app.include_router(feature2_maps_router)
app.include_router(feature2_rank_router)
app.include_router(feature3_node_details_router)
app.include_router(feature4_compare_router)
app.include_router(feature5_gap_router)
app.include_router(settings_router)
app.include_router(user_settings_router)
app.include_router(workspaces_router)
app.include_router(sharing_router)
app.include_router(billing_router)
app.include_router(decomposition_router)
app.include_router(tags_router)
app.include_router(admin_router)
app.include_router(analytics_router)

@app.get("/test")
async def test():
    return {"message": "Test successful"}