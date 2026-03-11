import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

_engine = None

def make_engine() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    url = os.getenv("DATABASE_URL_DIRECT") or os.getenv("DATABASE_URL_LOCAL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env)")
    _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10, future=True)

    @event.listens_for(_engine, "connect")
    def set_statement_timeout(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET statement_timeout = '300s'")
        cursor.close()

    return _engine
