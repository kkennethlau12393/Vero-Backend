import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool


def make_engine() -> Engine:
    url = os.getenv("DATABASE_URL_POOLER")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env)")
    return create_engine(url, poolclass=NullPool)
