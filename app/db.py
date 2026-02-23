import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

def make_engine() -> Engine:
    url = os.getenv("DATABASE_URL_DIRECT") or os.getenv("DATABASE_URL_LOCAL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env)")
    engine = create_engine(url, pool_pre_ping=True, future=True)

    @event.listens_for(engine, "connect")
    def set_statement_timeout(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET statement_timeout = '300s'")
        cursor.close()

    return engine
