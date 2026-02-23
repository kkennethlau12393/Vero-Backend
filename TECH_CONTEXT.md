# TECH_CONTEXT.md

Technical context for future sessions in `Vero-Backend` (`alexandria-backend`).

## What This Backend Is

- Python/FastAPI backend for research workflows around:
  - Feature 1: citation map generation
  - Feature 2: ranking and map building
  - Feature 3: node details + novelty assessment
  - Feature 4: methodology comparison
  - Feature 5: research gap analysis
- Multi-tenant by workspace (`X-Workspace-Id` header).

## Runtime Stack

- Python `>=3.11`
- FastAPI + Uvicorn
- SQLAlchemy Core (raw SQL), `psycopg2-binary` for Postgres
- Pydantic v2
- Requests + HTTPX (in tests/integration contexts)
- LLM/API clients: `openai`, `groq`
- Ranking/data libs: `numpy`, `scipy`, `rank-bm25`
- PDF tooling: `PyMuPDF`, `python-multipart`

See: `pyproject.toml`

## App Entry and Router Wiring

- Entry point: `app/main.py`
  - Loads `.env` from repo root
  - Configures permissive CORS
  - Mounts all feature routers
- Health/test route: `GET /test`

Routers currently included from `app/main.py`:

- `app/feature1/api.py` (prefix `/v1`)
- `app/feature2/maps_api.py` (prefix `/v1/maps`)
- `app/feature2/rank_api.py` (prefix `/v1/rank`)
- `app/feature3/node_details_api.py` (prefix `/v1`)
- `app/feature4/compare_api.py` (prefix `/v1`)
- `app/feature5/gap_api.py` (prefix `/v1`)
- `app/settings/api.py` (prefix `/v1/settings`)
- `app/workspaces/status_api.py` (prefix `/v1/workspaces`)

## High-Value Endpoints (Quick Map)

- Feature 1 citation map:
  - `POST /v1/citation-map`
  - `POST /v1/citation-map/pdf`
- Feature 2 map build/render:
  - `POST /v1/maps/build`
  - `GET /v1/maps/{map_id}`
- Feature 3 node details:
  - `GET /v1/maps/{map_id}/nodes/{work_id}/details`
- Feature 4 methodology comparison:
  - `POST /v1/maps/{map_id}/compare-methodologies`
- Workspaces status:
  - `GET /v1/workspaces/status`
  - `POST /v1/workspaces/status/batch`

For full request/response models, read each `*_api.py` and `schemas.py`.

## Data Layer and DB Conventions

- Engine factory: `app/db.py`
  - Reads `DATABASE_URL_DIRECT` first, then `DATABASE_URL_LOCAL`
  - Sets `pool_pre_ping=True`
  - Applies Postgres `statement_timeout = 300s` on connect
- SQL style:
  - Raw SQL via `sqlalchemy.text()`
  - No ORM model layer
  - Feature stores/repositories own SQL statements

Schema and bootstrap files:

- `bootstrap.sql` (core app tables + caches + vectors)
- `user-bootstrap.sql` (workspace/user/auth bootstrap tables)
- `migrate-multi-workspace.sql` (workspace migration support)

## Multi-Tenancy and Auth Context

- Tenant resolver: `app/auth/tenant.py`
  - Primary: `X-Workspace-Id` request header (UUID)
  - Dev fallback: `DEV_TENANT_ID` from env
- Most endpoints depend on `get_tenant_id()` and scope DB queries by tenant/workspace.
- Tenant settings storage:
  - `app/settings/store.py`
  - Table: `tenant_settings`
  - Includes institutional proxy and LibKey configuration fields

## Retrieval, LLM, and External Integrations

- Feature 2 retrieval (`app/feature2/retrieval.py`) runs multiple co-equal academic sources in parallel:
  - OpenAlex
  - Semantic Scholar
  - ArXiv
  - CrossRef / PubMed / DBLP
- Feature 1 citation map uses OpenAlex + S2 bridging paths.
- Feature 3 novelty/details uses LLM assessment plus grounding/reference pipelines.
- LLM config/versioning is explicit in service modules (example: `ASSESSMENT_VERSION` in `app/feature3/node_details_service.py`).

## Caching Strategy (Important)

- DB-backed caches are first-class and version-gated:
  - Query/LLM caches (Feature 2)
  - Node details/novelty cache (Feature 3)
  - Reference cache (Feature 3)
  - Paper full-text cache (Feature 4)
  - Gap analysis cache/status tables (Feature 5)
- In-process singleton cache pattern for DB engine:
  - `@lru_cache(maxsize=1)` in multiple API modules for `get_engine()`

When prompt/model behavior changes, bump relevant cache version constants to invalidate stale outputs.

## Testing and Quality Guardrails

Canonical source of testing policy: `CLAUDE.md`.

Short version:

- Unit: pure logic, no DB/network
- Integration: dev DB, mock all external APIs
- Quality: regression/golden behavior checks
- Live: real APIs and LLMs, explicit invocation only

Test entry settings are defined in `pyproject.toml` (`pytest` markers and defaults), and shared fixtures are in `tests/conftest.py`.

## Coding Conventions to Preserve

- Keep feature-based module boundaries (`featureN/api.py`, `featureN/*_service.py`).
- Prefer deterministic, general logic over hard-coded one-off fixes.
- Keep raw SQL approach; do not introduce ORM models unless architecture changes intentionally.
- Preserve cache-first behavior and versioned invalidation.
- Do not add duplicate test files where canonical tests already exist.

## Fast Orientation Checklist for Future Sessions

1. Read `CLAUDE.md` for strict testing and feature rules.
2. Read `app/main.py` to see currently mounted routers.
3. Read the relevant `featureN/*_api.py` for endpoint contracts.
4. Read corresponding `*_service.py` for business logic and cache/version knobs.
5. Validate schema assumptions in `bootstrap.sql` before changing SQL queries.
6. Prefer updating existing canonical tests over adding new parallel test files.
