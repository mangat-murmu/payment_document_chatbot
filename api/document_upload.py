from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlmodel import Session

from api.dependencies import database_session, knowledge_base
from api.schema import (
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentType,
    UpdateDocumentRequest,
    UploadDocumentsResponse,
)
from database.repository import DocumentRepository
from vector_db.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/api/documents", tags=["Documents"])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"


def _document_or_404(
    repository: DocumentRepository, document_id: int
) -> DocumentResponse:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    return DocumentResponse.model_validate(document)


def _storage_filename(filename: str) -> str:
    source = Path(filename)
    suffix = source.suffix.lower()
    max_stem_length = 255 - len(suffix) - 9
    if max_stem_length < 1:
        suffix = suffix[:245]
        max_stem_length = 1
    stem = (source.stem or "upload")[:max_stem_length]
    return f"{stem}_{uuid4().hex[:8]}{suffix}"


async def _store_upload(
    upload: UploadFile, doc_type: DocumentType
) -> tuple[dict[str, str | int], Path]:
    filename = Path(upload.filename or "upload").name
    if not filename or len(filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid filename"
        )

    contents = await upload.read()
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    storage_file = UPLOAD_DIRECTORY / _storage_filename(filename)
    storage_file.write_bytes(contents)
    return (
        {
            "filename": filename,
            "doc_type": doc_type.value,
            "indexing_status": "inprogress",
            "indexing_progress": 0,
            "indexing_error": None,
            "byte_size": len(contents),
            "storage_path": str(storage_file.relative_to(PROJECT_ROOT)),
        },
        storage_file,
    )


def _remove_files(files: list[Path]) -> None:
    for file_path in files:
        file_path.unlink(missing_ok=True)


def _storage_file(storage_path: str) -> Path | None:
    storage_file = (PROJECT_ROOT / storage_path).resolve()
    if not storage_file.is_relative_to(UPLOAD_DIRECTORY.resolve()):
        return None
    return storage_file


def _remove_document_files(documents: list[DocumentResponse]) -> None:
    files = []
    for document in documents:
        storage_file = _storage_file(document.storage_path)
        if storage_file is not None:
            files.append(storage_file)
    _remove_files(files)


def _index_document_background(
    *,
    engine,
    kb: KnowledgeBase,
    document_id: int,
    storage_file: Path,
) -> None:
    with Session(engine) as session:
        repository = DocumentRepository(session)
        document = repository.update(
            document_id,
            {
                "indexing_status": "inprogress",
                "indexing_progress": 10,
                "indexing_error": None,
            },
        )
        if document is None:
            return
        try:
            kb.insert_document(document, storage_file)
        except Exception as error:
            repository.update(
                document_id,
                {
                    "indexing_status": "failed",
                    "indexing_progress": 100,
                    "indexing_error": str(error)[:4000],
                },
            )
        else:
            repository.update(
                document_id,
                {
                    "indexing_status": "success",
                    "indexing_progress": 100,
                    "indexing_error": None,
                },
            )


@router.post(
    "", response_model=UploadDocumentsResponse, status_code=status.HTTP_201_CREATED
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    doc_type: Annotated[DocumentType, Form()],
    files: Annotated[list[UploadFile], File(...)],
    session: Annotated[Session, Depends(database_session)],
    kb: Annotated[KnowledgeBase, Depends(knowledge_base)],
) -> UploadDocumentsResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="at least one file is required",
        )

    payloads: list[dict[str, str | int]] = []
    storage_files: list[Path] = []
    try:
        for upload in files:
            payload, storage_file = await _store_upload(upload, doc_type)
            payloads.append(payload)
            storage_files.append(storage_file)
        repository = DocumentRepository(session)
        documents = repository.create_many(payloads)
        for document, storage_file in zip(documents, storage_files):
            background_tasks.add_task(
                _index_document_background,
                engine=session.get_bind(),
                kb=kb,
                document_id=document.id,
                storage_file=storage_file,
            )
    except Exception:
        _remove_files(storage_files)
        raise
    return UploadDocumentsResponse(
        documents=[
            DocumentResponse.model_validate(document)
            for document in documents
            if document is not None
        ]
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    session: Annotated[Session, Depends(database_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> DocumentListResponse:
    documents = DocumentRepository(session).list(offset=offset, limit=limit)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(document) for document in documents]
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    session: Annotated[Session, Depends(database_session)],
) -> DocumentResponse:
    return _document_or_404(DocumentRepository(session), document_id)


@router.get("/{document_id}/download", response_class=FileResponse)
def download_document(
    document_id: int,
    session: Annotated[Session, Depends(database_session)],
) -> FileResponse:
    document = _document_or_404(DocumentRepository(session), document_id)
    storage_file = _storage_file(document.storage_path)
    if storage_file is None or not storage_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document file not found"
        )
    return FileResponse(storage_file, filename=document.filename)


@router.put("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: int,
    payload: UpdateDocumentRequest,
    session: Annotated[Session, Depends(database_session)],
) -> DocumentResponse:
    document = DocumentRepository(session).update(document_id, payload)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    session: Annotated[Session, Depends(database_session)],
    kb: Annotated[KnowledgeBase, Depends(knowledge_base)],
) -> Response:
    repository = DocumentRepository(session)
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    kb.delete_document(document)
    repository.delete(document_id)
    _remove_document_files([DocumentResponse.model_validate(document)])
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("", response_model=DeleteDocumentsResponse)
def delete_documents(
    payload: DeleteDocumentsRequest,
    session: Annotated[Session, Depends(database_session)],
    kb: Annotated[KnowledgeBase, Depends(knowledge_base)],
) -> DeleteDocumentsResponse:
    repository = DocumentRepository(session)
    unique_ids = list(dict.fromkeys(payload.document_ids))
    documents = [repository.get(document_id) for document_id in unique_ids]
    missing_ids = [
        document_id
        for document_id, document in zip(unique_ids, documents)
        if document is None
    ]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"missing_document_ids": missing_ids},
        )

    existing_documents = [document for document in documents if document is not None]
    kb.delete_documents(existing_documents)
    deleted_documents = [
        DocumentResponse.model_validate(document) for document in existing_documents
    ]
    repository.delete_many(unique_ids)
    _remove_document_files(deleted_documents)
    return DeleteDocumentsResponse(
        deleted_document_ids=[document.id for document in deleted_documents]
    )
