from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel

import config
from api import chat_endpoints, document_intelligence, document_upload
from api.auth_middleware import AuthMiddleware
from database.session import create_database_engine
from vector_db.knowledge_base import KnowledgeBase

ROOT = Path(__file__).resolve().parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5.5s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.database_engine = create_database_engine(config.DATABASE_URL)
    SQLModel.metadata.create_all(app.state.database_engine)
    app.state.knowledge_base = KnowledgeBase()
    try:
        app.state.knowledge_base.ensure_indices()
    except Exception:
        logger.exception("OpenSearch indices could not be initialized")
    document_intelligence.preload_models()
    yield
    app.state.database_engine.dispose()


app = FastAPI(
    title="Payment Document Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(AuthMiddleware)
app.mount("/frontend", StaticFiles(directory=ROOT / "frontend"), name="frontend")
app.include_router(document_upload.router)
app.include_router(document_intelligence.router)
app.include_router(chat_endpoints.router)


@app.get("/", include_in_schema=False)
async def interface() -> FileResponse:
    return FileResponse(ROOT / "frontend/chat_interface.html")


@app.get("/chats/{chat_id}", include_in_schema=False)
async def chat_interface(chat_id: int) -> FileResponse:
    return FileResponse(ROOT / "frontend/chat_interface.html")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=config.APP_RELOAD,
    )
