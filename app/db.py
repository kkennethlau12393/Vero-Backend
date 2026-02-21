import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def make_engine() -> Engine:
    url = os.getenv("DATABASE_URL_DIRECT") or os.getenv("DATABASE_URL_LOCAL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env)")
    return create_engine(url, pool_pre_ping=True, future=True)
