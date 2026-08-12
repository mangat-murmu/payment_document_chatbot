from __future__ import annotations

import config
from sqlmodel import Session, create_engine


def database_url() -> str:
    url = config.DATABASE_URL
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def create_database_engine(url: str | None = None):
    return create_engine(url or database_url(), pool_pre_ping=True)


def session_scope(engine):
    return Session(engine)
