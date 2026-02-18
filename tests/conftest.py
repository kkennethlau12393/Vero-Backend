"""
Root conftest.py — shared fixtures for the Alexandria test suite.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import load_dotenv

# Load .env from repo root so DATABASE_URL / DEV_WORKSPACE_ID are available
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: pure unit tests (no I/O)")
    config.addinivalue_line("markers", "integration: tests that hit the DB or network")
    config.addinivalue_line("markers", "quality: golden-set quality tests")
    config.addinivalue_line("markers", "slow: tests that take >5 s")
    config.addinivalue_line("markers", "live: live tests hitting real APIs (burns credits)")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Auto-deselect live tests unless explicitly requested via path or -m live."""
    # If user explicitly ran tests/live/ or used -m live, don't deselect
    args = config.invocation_params.args
    if any("live" in str(a) for a in args):
        return
    # Also check -m flag
    markexpr = config.getoption("-m", default="")
    if "live" in str(markexpr):
        return

    skip_live = pytest.mark.skip(reason="Live tests require explicit invocation: pytest tests/live/")
    for item in items:
        if "live" in str(item.fspath):
            item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def dev_workspace_id() -> UUID:
    val = os.getenv(
        "DEV_WORKSPACE_ID",
        os.getenv("DEV_TENANT_ID", "00000000-0000-0000-0000-000000000001"),
    )
    return UUID(val)


@pytest.fixture()
def dev_tenant_id(dev_workspace_id: UUID) -> UUID:
    """Legacy alias fixture for compatibility during migration."""
    return dev_workspace_id


@pytest.fixture()
def auth_headers(dev_workspace_id: UUID) -> dict[str, str]:
    """Headers that satisfy the dev auth stub."""
    return {"X-Workspace-Id": str(dev_workspace_id)}


# ---------------------------------------------------------------------------
# FastAPI async client
# ---------------------------------------------------------------------------
@pytest.fixture()
async def client():
    """Async test client wired to the real FastAPI app."""
    from httpx import ASGITransport, AsyncClient
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# DB engine / session (uses the dev DATABASE_URL from .env)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def db_engine():
    from app.db import make_engine
    engine = make_engine()
    yield engine
    engine.dispose()


@pytest.fixture()
def db_conn(db_engine):
    with db_engine.connect() as conn:
        yield conn


# ---------------------------------------------------------------------------
# External API mocks (using respx)
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_groq(respx_mock):
    """Intercept all calls to api.groq.com."""
    route = respx_mock.route(host="api.groq.com").respond(
        json={
            "choices": [{"message": {"content": "mocked groq response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    return route


@pytest.fixture()
def mock_openalex(respx_mock):
    """Intercept calls to api.openalex.org."""
    route = respx_mock.route(host="api.openalex.org").respond(json={"results": []})
    return route


@pytest.fixture()
def mock_semantic_scholar(respx_mock):
    """Intercept calls to api.semanticscholar.org."""
    route = respx_mock.route(host="api.semanticscholar.org").respond(json={"data": []})
    return route


@pytest.fixture()
def mock_arxiv(respx_mock):
    """Intercept calls to export.arxiv.org."""
    route = respx_mock.route(host="export.arxiv.org").respond(
        text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )
    return route
