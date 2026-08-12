from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel


class BaseTable(SQLModel):
    """Base model with common fields and methods."""

    id: int | None = Field(default=None, primary_key=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )


class Document(BaseTable, table=True):
    """A document uploaded by a user."""

    __tablename__ = "documents"

    filename: str = Field(max_length=255, nullable=False)
    doc_type: str = Field(max_length=100, nullable=False)
    indexing_status: str = Field(default="inprogress", max_length=20, nullable=False)
    indexing_progress: int = Field(default=0, ge=0, le=100, nullable=False)
    indexing_error: str | None = Field(default=None, nullable=True)
    byte_size: int = Field(ge=0, nullable=False)
    storage_path: str = Field(max_length=255, nullable=False)


class Chat(BaseTable, table=True):
    """A conversation containing an ordered collection of messages."""

    __tablename__ = "chats"

    title: str | None = Field(max_length=255, default=None)

    messages: list["Message"] = Relationship(
        back_populates="chat", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class Citation(BaseModel):
    """A citation for a document."""

    document_id: int
    index: int
    text: str


class Message(BaseTable, table=True):
    """A user, assistant, or system message in a chat."""

    __tablename__ = "messages"

    chat_id: int = Field(foreign_key="chats.id", index=True, ondelete="CASCADE")
    previous_id: int | None = Field(
        foreign_key="messages.id", index=True, default=None, ondelete="SET NULL"
    )
    role: str = Field(max_length=20, nullable=False)
    name: str | None = Field(max_length=100, default=None)
    content: str | dict[str, Any] = Field(sa_column=Column(JSON))
    meta_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    token_usage: int | None = Field(default=None, ge=0)

    chat: Chat = Relationship(back_populates="messages")
