
build:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

db-bootstrap:
    psql "$DATABASE_URL_LOCAL" -v ON_ERROR_STOP=1 -f bootstrap.sql

db-bootstrap-users:
    psql "$DATABASE_URL_LOCAL" -v ON_ERROR_STOP=1 -f user-bootstrap.sql

db-bootstrap-all: db-bootstrap db-bootstrap-users