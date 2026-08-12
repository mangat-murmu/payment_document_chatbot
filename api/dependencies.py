from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fastapi import Request
from sqlmodel import Session


def database_session(request: Request) -> Generator[Session, None, None]:
    with Session(request.app.state.database_engine) as session:
        yield session


def knowledge_base(request: Request) -> Any:
    return request.app.state.knowledge_base
