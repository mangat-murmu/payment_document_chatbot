"""Pydantic request and response schemas for FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaginationParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)


class DocumentType(StrEnum):
    UPI_TRANSACTION = "upi_transaction"
    BANK_API_LOG = "bank_api_log"
    COMPLIANCE_AUDIT = "compliance_audit"
    PARTNERSHIP_SLA = "partnership_sla"


class IndexingStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    INPROGRESS = "inprogress"


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str = Field(min_length=1, max_length=255)
    doc_type: DocumentType
    indexing_status: IndexingStatus
    indexing_progress: int = Field(ge=0, le=100)
    indexing_error: str | None = None
    byte_size: int = Field(ge=0)
    storage_path: str = Field(min_length=1, max_length=255)
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class UploadDocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteDocumentsRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1)


class DeleteDocumentsResponse(BaseModel):
    deleted_document_ids: list[int]


class UpdateDocumentRequest(BaseModel):
    filename: str | None = Field(default=None, min_length=1, max_length=255)
    doc_type: DocumentType | None = None
    indexing_status: IndexingStatus | None = None
    indexing_progress: int | None = Field(default=None, ge=0, le=100)
    indexing_error: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    storage_path: str | None = Field(default=None, min_length=1, max_length=255)


class DocumentClassificationResponse(BaseModel):
    label: str
    confidence: float


class EntityExtractionResponse(BaseModel):
    entities: dict[str, list[dict[str, Any]]]


class ChatInput(BaseModel):
    text: str = Field(min_length=1)


class ChatGenerationRequest(BaseModel):
    chat_id: int | None = None
    input: ChatInput
    last_response_id: int | None = None
    stream: bool = False


class ChatGenerationResponse(BaseModel):
    chat: "ChatResponse"
    message: "MessageResponse"


class UpdateChatRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime


class ChatListResponse(BaseModel):
    chats: list[ChatResponse]


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    chat_id: int
    previous_id: int | None
    role: str
    name: str | None
    content: str | dict[str, Any]
    meta_data: dict[str, Any] | None
    token_usage: int | None
    created_at: datetime
    updated_at: datetime


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
