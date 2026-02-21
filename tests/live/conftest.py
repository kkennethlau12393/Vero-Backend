"""
Live test conftest — real APIs, real DB, no mocks.

These tests burn API credits (Groq, OpenAlex, S2, ArXiv).
Only run when changing retrieval logic or LLM prompts.

Usage:
    pytest tests/live/ -v -s --timeout=300
    pytest tests/live/test_ranking_e2e.py -k "transformer" -v -s
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import load_dotenv

# Load real .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Marker: all tests in this directory are live
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(items):
    for item in items:
        if "live" in str(item.fspath):
            item.add_marker(pytest.mark.live)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"


@pytest.fixture(scope="session", autouse=True)
def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


@pytest.fixture(scope="session")
def dev_tenant_id() -> UUID:
    return UUID(os.getenv("DEV_TENANT_ID", "00000000-0000-0000-0000-000000000001"))


@pytest.fixture(scope="session")
def auth_headers(dev_tenant_id) -> dict[str, str]:
    return {"X-Workspace-Id": str(dev_tenant_id)}


@pytest.fixture(scope="session")
def db_engine():
    from app.db import make_engine
    engine = make_engine()
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def live_client():
    """Sync test client using the real FastAPI app (no mocks)."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client


def save_result(results_dir: Path, name: str, data: dict):
    """Save test result JSON for manual review + regression detection."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{name}_{ts}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    # Also write a "latest" symlink-style file for easy comparison
    latest = results_dir / f"{name}_latest.json"
    latest.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_previous_result(results_dir: Path, name: str) -> dict | None:
    """Load most recent previous result for regression comparison."""
    latest = results_dir / f"{name}_latest.json"
    if latest.exists():
        return json.loads(latest.read_text())
    return None
