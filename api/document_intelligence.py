from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from api.schema import DocumentClassificationResponse, EntityExtractionResponse
from document_processor.document_classifier import DocumentClassifier
from document_processor.document_loader import DocumentLoader
from document_processor.entity_extractor import EntityExtractor

router = APIRouter(prefix="/api/document-intelligence", tags=["Document Intelligence"])


@lru_cache(maxsize=1)
def _classifier() -> DocumentClassifier:
    return DocumentClassifier()


@lru_cache(maxsize=1)
def _entity_extractor() -> EntityExtractor:
    return EntityExtractor()


def preload_models() -> None:
    pass
    # _classifier()
    # _entity_extractor()


async def _input_text(text: str | None, file: UploadFile | None) -> str:
    if bool(text and text.strip()) == bool(file):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="provide exactly one of text or file",
        )
    if text and text.strip():
        return text
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="provide exactly one of text or file",
        )

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="uploaded file is empty",
        )

    temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary_file.write(contents)
        temporary_file.close()
        try:
            documents = await run_in_threadpool(
                DocumentLoader(temporary_file.name).load
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        return "\n\n".join(document.page_content for document in documents)
    finally:
        Path(temporary_file.name).unlink(missing_ok=True)


@router.post(
    "/classify",
    response_model=DocumentClassificationResponse,
    summary="Classify a document",
)
async def classify_document(
    text: Annotated[str | None, Form(description="Raw document text")] = None,
    file: Annotated[UploadFile | None, File(description="Document file upload")] = None,
) -> DocumentClassificationResponse:
    input_text = await _input_text(text, file)
    result = await run_in_threadpool(_classifier().classify, input_text)
    return DocumentClassificationResponse(
        label=result.label,
        confidence=result.confidence,
    )


@router.post(
    "/extract-entities",
    response_model=EntityExtractionResponse,
    summary="Extract document entities",
)
async def extract_entities(
    text: Annotated[str | None, Form(description="Raw document text")] = None,
    file: Annotated[UploadFile | None, File(description="Document file upload")] = None,
) -> EntityExtractionResponse:
    input_text = await _input_text(text, file)
    entities = await run_in_threadpool(_entity_extractor().extract, input_text)
    return EntityExtractionResponse(entities=entities)
