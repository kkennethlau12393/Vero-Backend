# AI Assistant Context Document

This file provides structured context for AI coding assistants (GitHub Copilot, Claude, ChatGPT, etc.) working on this codebase.

## Project Identity

- **Name**: Vero Backend
- **Type**: FastAPI REST API
- **Architecture**: Modular router-based structure
- **Purpose**: Backend service for Vero application

## Critical Information

### 1. Project Structure Pattern
This project follows **FastAPI's official "Bigger Applications" pattern**:
- Main app in `app/main.py`
- **Shared models in `app/models/`** - Organized by category (work, topic, candidate, graph, map, ranking)
- Routers in `app/routers/` directory
- Dependencies in `app/dependencies.py`
- Internal/admin routes in `app/internal/`
- All directories have `__init__.py` to make them Python packages

### 2. Import Style
**Always use absolute imports from app package**:
```python
✅ CORRECT:
from app.routers import users
from app.dependencies import get_token_header
from app.models import WorkRef, TopicRef, CandidateSet

❌ INCORRECT:
from .routers import users
from ..dependencies import get_token_header
from .core import WorkRef
```

### 3. Router Pattern
All routers follow this template:
```python
from fastapi import APIRouter

router = APIRouter(
    prefix="/resource",
    tags=["resource"],
    # optional dependencies
)

@router.get("/")
async def list_resources():
    return []
```

### 4. Adding New Features
When adding new endpoints:
1. Create/modify file in `app/routers/`
2. Define router with prefix and tags
3. Add route handlers
4. Import and include in `app/main.py` using `app.include_router()`

### 5. Dependencies
Shared dependencies go in `app/dependencies.py`. Use FastAPI's `Depends()` for:
- Authentication
- Authorization
- Shared business logic
- Database sessions (future)

## File Purposes

| File/Directory | Purpose | Modify When |
|----------------|---------|-------------|
| `app/main.py` | FastAPI app instance, router inclusion | Adding new routers |
| `app/models/` | **Shared data models (organized by category)** | Adding/modifying domain models |
| `app/models/__init__.py` | Package exports all models for easy importing | Adding new model files |
| `app/models/work.py` | Work/paper models | Adding paper-related models |
| `app/models/topic.py` | Topic/subject models | Adding topic-related models |
| `app/models/candidate.py` | Discovery result models | Adding candidate models |
| `app/models/graph.py` | Citation graph models | Adding graph models |
| `app/models/map.py` | Finalized map models | Adding visualization models |
| `app/models/ranking.py` | Ranking/recommendation models | Adding ranking models |
| `app/dependencies.py` | Shared dependencies | Adding auth/validation logic |
| `app/routers/*.py` | Domain-specific endpoints | Adding/modifying API routes |
| `app/internal/*.py` | Admin/internal endpoints | Adding privileged operations |
| `requirements.txt` | Python dependencies | Adding new packages |
| `.env` | Environment variables | Adding config values |

## Current State

### Implemented
- ✅ Basic FastAPI structure
- ✅ **Core data models** (WorkRef, TopicRef, CandidateSet, GraphDraft, Map, RankedList)
- ✅ User endpoints (GET list, GET by username)
- ✅ Item endpoints (GET, PUT with auth)
- ✅ Admin endpoint (POST)
- ✅ Token-based authentication example
- ✅ In-memory data storage (fake_items_db)

### Not Implemented (TODO)
- ❌ Database integration
- ❌ Real authentication (currently uses fake tokens)
- ❌ Unit tests
- ❌ Environment configuration
- ❌ Logging
- ❌ CORS configuration
- ❌ Rate limiting

## Data Models (app/models/)

All domain objects are defined as Pydantic models organized by category in `app/models/`.

**Package Structure:**
```
app/models/
├── __init__.py       # Exports all models for easy importing
├── work.py           # WorkRef, WorkRefThin, Author, IngestState
├── topic.py          # TopicRef, TopicQueryRef, TopicHierarchy  
├── candidate.py      # CandidateSet, CandidateItem
├── graph.py          # GraphDraft, GraphStats, CitationEdge
├── map.py            # Map, MapNode, SubtopicDefinition, FieldContext, LayoutCoordinates
└── ranking.py        # RankedList, RankedItem, RankingContext, ScoreBreakdown
```

### Work Models
- **WorkRef**: Full paper metadata (DOI, title, abstract, authors, citations, topics)
- **WorkRefThin**: Lightweight version with minimal fields
- **Author**: Author information
- **IngestState**: Enum for data completeness (resolved/partially_resolved/unresolved)

### Topic Models
- **TopicRef**: Research topic with hierarchy and score
- **TopicQueryRef**: Topics extracted from text query
- **TopicHierarchy**: Domain → Field → Subfield → Topic structure

### Candidate Models
- **CandidateSet**: Pool of papers from discovery (Feature 1)
- **CandidateItem**: Individual paper with discovery metadata

### Graph Models
- **GraphDraft**: Temporary citation graph before finalization
- **GraphStats**: Coverage and quality metrics
- **CitationEdge**: Directed citation relationship

### Map Models
- **Map**: Final stored graph with visualization data
- **MapNode**: Node with full metadata + layout coordinates
- **SubtopicDefinition**: Node grouping strategy
- **FieldContext**: Research field context
- **LayoutCoordinates**: 2D position for visualization

### Ranking Models
- **RankedList**: Scored paper recommendations
- **RankedItem**: Single ranked paper with explanation
- **RankingContext**: What ranking is based on
- **ScoreBreakdown**: Transparent score calculation

**Usage:**
```python
# Import from package __init__.py (easiest)
from app.models import WorkRef, TopicRef, CandidateSet, IngestState

# Or import from specific module
from app.models.work import WorkRef, IngestState
from app.models.topic import TopicRef
from app.models.candidate import CandidateSet

# Create a work
work = WorkRef(
    work_id="W123",
    title="Paper Title",
    ingest_state=IngestState.RESOLVED
)

# Use in API responses
@router.get("/work/{work_id}")
async def get_work(work_id: str) -> WorkRef:
    return WorkRef(...)
```

## Common Tasks

### Task: Add a new API endpoint
1. Decide which router file it belongs to (or create new)
2. Add function with decorator: `@router.get("/path")`
3. Use async functions: `async def function_name()`
4. Return dict or Pydantic model
5. If new router file, include in `app/main.py`

### Task: Add authentication to endpoint
```python
from fastapi import Depends
from app.dependencies import get_token_header

@router.get("/", dependencies=[Depends(get_token_header)])
async def protected_endpoint():
    return {"data": "secret"}
```

### Task: Add query/path parameters
```python
@router.get("/{item_id}")
async def get_item(item_id: str, skip: int = 0, limit: int = 10):
    return {"item_id": item_id, "skip": skip, "limit": limit}
```

### Task: Add request body
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@router.post("/")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

### Task: Add new router
```python
# 1. Create app/routers/new_feature.py
from fastapi import APIRouter

router = APIRouter(prefix="/feature", tags=["feature"])

@router.get("/")
async def list_features():
    return []

# 2. Include in app/main.py
from app.routers import new_feature
app.include_router(new_feature.router)
```

## Code Style Guidelines

- Use `async def` for all route handlers
- Use type hints for parameters
- Use Pydantic models for request/response schemas
- Raise `HTTPException` for errors
- Use descriptive function and variable names
- Group related endpoints in same router file
- Use prefixes to organize URL structure

## Running & Testing

```bash
# Start development server
uvicorn app.main:app --reload

# Access interactive docs
open http://localhost:8000/docs

# Test endpoints
curl http://localhost:8000/
curl http://localhost:8000/users/
curl -H "X-Token: fake-super-secret-token" http://localhost:8000/items/
```

## Dependencies Information

Key packages:
- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `pydantic`: Data validation

## Known Issues / Limitations

1. **Fake Authentication**: Current token validation is placeholder only
2. **In-Memory Storage**: No persistence, data lost on restart
3. **No Database**: Not configured yet
4. **Minimal Error Handling**: Basic HTTPExceptions only
5. **No Tests**: Test suite not implemented

## Extension Points

Areas designed for future expansion:
- `app/models/` - Database models (create when needed)
- `app/schemas/` - Pydantic schemas (create when needed)
- `app/crud/` - Database operations (create when needed)
- `app/core/` - Configuration, security (create when needed)
- `tests/` - Test suite (create when needed)

## Questions to Ask Before Changing Code

1. Does this endpoint belong to an existing router or need a new one?
2. Should this functionality be shared (put in dependencies.py)?
3. Does this need authentication/authorization?
4. What HTTP status code should errors return?
5. Should this be in `app/routers/` (public) or `app/internal/` (admin)?

## References

- FastAPI Docs: https://fastapi.tiangolo.com
- Bigger Applications Guide: https://fastapi.tiangolo.com/tutorial/bigger-applications/
- This project's architecture: See ARCHITECTURE.md
