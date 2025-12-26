# Vero Backend - Architecture Documentation

## Project Overview

This is a FastAPI-based backend application following the official FastAPI recommended structure for larger applications. The project uses modular routing with separated concerns for better maintainability and scalability.

## Technology Stack

- **Framework**: FastAPI 0.115.0+
- **Server**: Uvicorn with standard extras
- **Validation**: Pydantic 2.0+
- **Language**: Python 3.8+

## Project Structure

```
Vero-Backend/
├── app/                        # Main application package
│   ├── __init__.py            # Makes app a Python package
│   ├── main.py                # FastAPI application entry point
│   ├── dependencies.py        # Shared dependency injection functions
│   ├── routers/               # API route modules
│   │   ├── __init__.py
│   │   ├── users.py           # User management endpoints
│   │   └── items.py           # Item management endpoints
│   └── internal/              # Internal/admin functionality
│       ├── __init__.py
│       └── admin.py           # Administrative endpoints
├── requirements.txt           # Python dependencies
├── venv/                      # Virtual environment (not in git)
├── .env                       # Environment variables (not in git)
└── README.md                  # User-facing documentation
```

## Core Components

### 1. Main Application (`app/main.py`)

The central FastAPI application instance that:
- Creates the FastAPI app with metadata (title, version)
- Includes all routers from different modules
- Defines the root endpoint
- Serves as the entry point for the ASGI server

**Key Features:**
- Modular router inclusion using `app.include_router()`
- Prefix and tag organization for admin routes
- Root health check endpoint at `/`

### 2. Dependencies (`app/dependencies.py`)

Contains reusable dependency injection functions used across routes:

- `get_token_header(x_token: str)`: Validates X-Token header authentication
- `get_query_token(token: str)`: Validates query parameter tokens

**Purpose**: Centralize authentication and validation logic to avoid duplication.

### 3. Routers

#### Users Router (`app/routers/users.py`)
- **Prefix**: `/users`
- **Tags**: `["users"]`
- **Endpoints**:
  - `GET /users/` - List all users
  - `GET /users/{username}` - Retrieve specific user by username
- **Dependencies**: None (public endpoints)

#### Items Router (`app/routers/items.py`)
- **Prefix**: `/items`
- **Tags**: `["items"]`
- **Dependencies**: `get_token_header` (requires X-Token header)
- **Endpoints**:
  - `GET /items/` - List all items
  - `GET /items/{item_id}` - Retrieve specific item
  - `PUT /items/{item_id}` - Update specific item (tagged as "custom")
- **Data**: Uses in-memory `fake_items_db` dictionary

#### Admin Router (`app/internal/admin.py`)
- **Prefix**: `/admin` (set in main.py)
- **Tags**: `["admin"]`
- **Endpoints**:
  - `POST /admin/` - Administrative update operation
- **Purpose**: Internal administrative operations

## API Design Patterns

### 1. Router Organization
Each router is defined using `APIRouter()` with:
- Clear prefix for URL namespacing
- Descriptive tags for OpenAPI documentation
- Optional shared dependencies
- Custom response schemas for error handling

### 2. Dependency Injection
FastAPI's dependency injection system is used for:
- Authentication (token validation)
- Authorization checks
- Shared business logic

### 3. Error Handling
Standard HTTP exceptions are raised with appropriate status codes:
- `400`: Bad Request (invalid tokens)
- `403`: Forbidden (operation not allowed)
- `404`: Not Found (resource doesn't exist)

## Running the Application

### Development Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app.main:app --reload
```

### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Development Guidelines

### Adding New Endpoints

1. **Create or modify a router file** in `app/routers/`
2. **Define the router** with appropriate prefix and tags
3. **Add endpoint functions** using FastAPI decorators
4. **Include the router** in `app/main.py`

Example:
```python
# app/routers/new_feature.py
from fastapi import APIRouter

router = APIRouter(prefix="/feature", tags=["feature"])

@router.get("/")
async def list_features():
    return {"features": []}
```

```python
# app/main.py
from app.routers import new_feature
app.include_router(new_feature.router)
```

### Adding Dependencies

1. **Define dependency function** in `app/dependencies.py`
2. **Use `Depends()`** in route decorators or function parameters
3. **Share across routers** as needed

### Code Organization Principles

- **Single Responsibility**: Each router handles one domain/resource
- **DRY (Don't Repeat Yourself)**: Use dependencies for shared logic
- **Separation of Concerns**: Keep routing, business logic, and data separate
- **Explicit Imports**: Use absolute imports from `app.*`

## Security Considerations

Current implementation includes:
- Token-based authentication (header and query parameter)
- Route-level dependency enforcement
- Input validation via Pydantic models (to be expanded)

**TODO for Production**:
- Replace fake tokens with real authentication (JWT, OAuth2)
- Add database integration
- Implement proper user management
- Add rate limiting
- Enable CORS with specific origins
- Add logging and monitoring
- Implement environment-based configuration

## Database Integration (Future)

When adding database support:
1. Create `app/database.py` for connection configuration
2. Create `app/models/` for SQLAlchemy/Tortoise ORM models
3. Create `app/schemas/` for Pydantic request/response models
4. Create `app/crud/` for database operations

## Testing Structure (Future)

Recommended test organization:
```
tests/
├── __init__.py
├── test_main.py
├── test_users.py
├── test_items.py
└── test_admin.py
```

## Environment Variables

Store sensitive configuration in `.env`:
```
DATABASE_URL=postgresql://user:pass@localhost/dbname
SECRET_KEY=your-secret-key
API_TOKEN=your-api-token
```

Load using `python-dotenv` or similar libraries.

## Common Operations

### Adding a New Router
1. Create file: `app/routers/resource_name.py`
2. Define router with prefix and tags
3. Add endpoints
4. Include in `app/main.py`

### Adding Authentication
1. Create dependency in `app/dependencies.py`
2. Apply to router: `router = APIRouter(dependencies=[Depends(auth_dep)])`
3. Or per-endpoint: `@router.get("/", dependencies=[Depends(auth_dep)])`

### Organizing Related Endpoints
- Group by resource/domain in separate router files
- Use prefixes to namespace URLs
- Use tags to group in API documentation
- Use internal/ folder for admin/privileged operations

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [FastAPI Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [APIRouter Reference](https://fastapi.tiangolo.com/reference/apirouter/)
- [Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
