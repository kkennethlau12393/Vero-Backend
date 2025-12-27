# Vero-Backend

A FastAPI backend application following the recommended project structure.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # Main FastAPI application
│   ├── dependencies.py      # Shared dependencies
│   ├── models/              # Data models (organized by category)
│   │   ├── __init__.py
│   │   ├── work.py          # Work/paper models
│   │   ├── topic.py         # Topic/subject models
│   │   ├── candidate.py     # Candidate pool models
│   │   ├── graph.py         # Citation graph models
│   │   ├── map.py           # Finalized map models
│   │   └── ranking.py       # Ranking models
│   ├── routers/             # API route modules
│   │   ├── __init__.py
│   │   ├── users.py         # User-related endpoints
│   │   └── items.py         # Item-related endpoints
│   └── internal/            # Internal/admin modules
│       ├── __init__.py
│       └── admin.py         # Admin endpoints
├── requirements.txt         # Python dependencies
└── venv/                    # Virtual environment
```

## Data Models

All domain objects organized in `app/models/` package:
- **work.py**: WorkRef, WorkRefThin, Author, IngestState
- **topic.py**: TopicRef, TopicQueryRef, TopicHierarchy
- **candidate.py**: CandidateSet, CandidateItem
- **graph.py**: GraphDraft, GraphStats, CitationEdge
- **map.py**: Map, MapNode, SubtopicDefinition, FieldContext, LayoutCoordinates
- **ranking.py**: RankedList, RankedItem, RankingContext, ScoreBreakdown

Import any model from the package: `from app.models import WorkRef, TopicRef, ...`

See `ARCHITECTURE.md` and `DATA_MODELS.md` for detailed documentation.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

- Interactive API docs (Swagger UI): `http://localhost:8000/docs`
- Alternative API docs (ReDoc): `http://localhost:8000/redoc`

## Endpoints

- `GET /` - Root endpoint
- `GET /users/` - List users
- `GET /users/{username}` - Get specific user
- `GET /items/` - List items (requires X-Token header)
- `GET /items/{item_id}` - Get specific item
- `PUT /items/{item_id}` - Update item
- `POST /admin/` - Admin endpoint