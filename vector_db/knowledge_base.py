from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownTextSplitter

import config
from api.schema import DocumentType
from database.models import Document
from document_processor.document_loader import DocumentLoader
from vector_db.embedding_service import EmbeddingService
from vector_db.vector_search import SearchResult, hybrid_search

logger = logging.getLogger(__name__)

INDEX_PREFIX = config.OPENSEARCH_INDEX_PREFIX
VECTOR_DIMENSIONS = EmbeddingService.dimensions
HYBRID_SEARCH_PIPELINE = f"{INDEX_PREFIX}_hybrid_search_pipeline"


def keyword(ignore_above: int = 256) -> dict[str, Any]:
    return {"type": "keyword", "ignore_above": ignore_above}


def text_with_keyword(ignore_above: int = 256) -> dict[str, Any]:
    return {"type": "text", "fields": {"keyword": keyword(ignore_above)}}


BASE_INDEX_SETTINGS: dict[str, Any] = {
    "index": {
        "knn": True,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
}

BASE_PROPERTIES: dict[str, Any] = {
    "document_id": {"type": "integer"},
    "doc_type": keyword(),
    "filename": text_with_keyword(),
    "chunk_index": {"type": "integer"},
    "page_number": {"type": "integer"},
    "page_count": {"type": "integer"},
    "content": {"type": "text"},
    "content_vector": {
        "type": "knn_vector",
        "dimension": VECTOR_DIMENSIONS,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "lucene",
            "parameters": {
                "ef_construction": 128,
                "m": 24,
            },
        },
    },
    "created_at": {"type": "date"},
    "updated_at": {"type": "date"},
    "indexed_at": {"type": "date"},
}

UPI_TRANSACTION_PROPERTIES: dict[str, Any] = {
    "transaction_id": keyword(),
    "timestamp": {"type": "date"},
    "psp": keyword(),
    "transaction_type": keyword(),
    "payer_vpa": keyword(),
    "payee_vpa": keyword(),
    "payer_bank": text_with_keyword(),
    "payee_bank": text_with_keyword(),
    "merchant_id": keyword(),
    "merchant_name": text_with_keyword(),
    "merchant_category": text_with_keyword(),
    "mcc": keyword(),
    "amount": {"type": "scaled_float", "scaling_factor": 100},
    "currency": keyword(),
    "status": keyword(),
    "response_code": keyword(),
    "failure_reason": text_with_keyword(),
    "rrn": keyword(),
    "bank_reference_id": keyword(),
    "payment_mode": keyword(),
    "device_type": keyword(),
    "state": keyword(),
    "settlement_status": keyword(),
    "risk_score": {"type": "short"},
    "fraud_flag": {"type": "boolean"},
}

BANK_API_LOG_PROPERTIES: dict[str, Any] = {
    "timestamp": {"type": "date"},
    "psp": keyword(),
    "service_owner": keyword(),
    "integration_direction": keyword(),
    "bank": text_with_keyword(),
    "operation": keyword(),
    "request_id": keyword(),
    "trace_id": keyword(),
    "transaction_id": keyword(),
    "settlement_id": keyword(),
    "status": keyword(),
    "latency_ms": {"type": "integer"},
    "internal_amount": {"type": "scaled_float", "scaling_factor": 100},
    "bank_amount": {"type": "scaled_float", "scaling_factor": 100},
    "reconciliation_status": keyword(),
    "severity": keyword(),
    "transaction_count": {"type": "integer"},
    "gross_amount": {"type": "scaled_float", "scaling_factor": 100},
    "fees": {"type": "scaled_float", "scaling_factor": 100},
    "net_amount": {"type": "scaled_float", "scaling_factor": 100},
    "currency": keyword(),
}

COMPLIANCE_AUDIT_PROPERTIES: dict[str, Any] = {
    "circular_number": keyword(),
    "subject": text_with_keyword(512),
    "issue_date": {"type": "date"},
}

PARTNERSHIP_SLA_PROPERTIES: dict[str, Any] = {
    "agreement_id": keyword(),
    "title": text_with_keyword(512),
    "effective_date": {"type": "date"},
    "expiry_date": {"type": "date"},
}

DOCUMENT_TYPE_PROPERTIES: dict[DocumentType, dict[str, Any]] = {
    DocumentType.UPI_TRANSACTION: UPI_TRANSACTION_PROPERTIES,
    DocumentType.BANK_API_LOG: BANK_API_LOG_PROPERTIES,
    DocumentType.COMPLIANCE_AUDIT: COMPLIANCE_AUDIT_PROPERTIES,
    DocumentType.PARTNERSHIP_SLA: PARTNERSHIP_SLA_PROPERTIES,
}


def index_name(document_type: DocumentType | str) -> str:
    return f"{INDEX_PREFIX}_{DocumentType(document_type).value}"


def build_index_schema(document_type: DocumentType | str) -> dict[str, Any]:
    resolved_type = DocumentType(document_type)
    properties = deepcopy(BASE_PROPERTIES)
    properties.update(deepcopy(DOCUMENT_TYPE_PROPERTIES[resolved_type]))
    return {
        "settings": deepcopy(BASE_INDEX_SETTINGS),
        "mappings": {
            "dynamic": "false",
            "properties": properties,
        },
    }


OPENSEARCH_SCHEMAS: dict[str, dict[str, Any]] = {
    index_name(document_type): build_index_schema(document_type)
    for document_type in DocumentType
}


def create_indices(client: Any, *, recreate: bool = False) -> None:
    """Create all document indexes in OpenSearch if they do not already exist."""
    for name, schema in OPENSEARCH_SCHEMAS.items():
        if recreate and client.indices.exists(index=name):
            client.indices.delete(index=name)
        if not client.indices.exists(index=name):
            client.indices.create(index=name, body=schema)
    create_hybrid_search_pipeline(client)


def create_hybrid_search_pipeline(client: Any) -> None:
    """Create the search-time score normalization pipeline used by hybrid queries."""
    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{HYBRID_SEARCH_PIPELINE}",
        body={
            "description": "Normalize and combine keyword and vector search scores.",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {
                                "weights": [
                                    0.3,
                                    0.7,
                                ]
                            },
                        },
                    }
                }
            ],
        },
    )


def create_opensearch_client(url: str | None = None) -> Any:
    """Build an OpenSearch client from configuration."""
    try:
        from opensearchpy import OpenSearch
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies to use OpenSearch: pip install -r requirements.txt"
        ) from error

    return OpenSearch(hosts=[url or config.OPENSEARCH_URL])


@dataclass(frozen=True)
class IndexedDocument:
    """Summary of a document indexing operation."""

    document_id: int
    index: str
    chunks: int


@dataclass(frozen=True)
class SqlResult:
    """Tabular result returned from the OpenSearch SQL plugin."""

    columns: list[str]
    rows: list[list[Any]]


class KnowledgeBase:
    """OpenSearch-backed knowledge base for uploaded payment documents."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        embedding_service: EmbeddingService | None = None,
        chunk_size: int = config.KNOWLEDGE_BASE_CHUNK_SIZE,
        chunk_overlap: int = config.KNOWLEDGE_BASE_CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.client = client or create_opensearch_client()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ensure_indices(self, *, recreate: bool = False) -> None:
        create_indices(self.client, recreate=recreate)

    def insert_document(
        self, document: Document, file_path: str | Path
    ) -> IndexedDocument:
        """Load, chunk, embed, and index one uploaded document."""
        if document.id is None:
            raise ValueError("document must be persisted before indexing")

        doc_type = DocumentType(document.doc_type)
        index = index_name(doc_type)
        actions = []
        for chunk_index, chunk in enumerate(self._document_chunks(file_path)):
            body = {
                "document_id": document.id,
                "doc_type": doc_type.value,
                "filename": document.filename,
                "chunk_index": chunk_index,
                "page_number": chunk["page_number"],
                "page_count": chunk["page_count"],
                "content": chunk["content"],
                "content_vector": self.embedding_service.embed(chunk["content"]),
                "created_at": self._serialize(document.created_at),
                "updated_at": self._serialize(document.updated_at),
                "indexed_at": self._serialize(datetime.now(timezone.utc)),
            }
            body.update(
                self._type_specific_fields(doc_type, chunk.get("metadata") or {})
            )
            actions.append(
                {
                    "_op_type": "index",
                    "_index": index,
                    "_id": f"{document.id}:{chunk_index}",
                    "_source": body,
                }
            )

        if not actions:
            raise ValueError("document did not produce any indexable content")

        try:
            from opensearchpy.helpers import bulk
        except ImportError as error:
            raise RuntimeError(
                "Install project dependencies to use OpenSearch bulk indexing"
            ) from error

        bulk(self.client, actions, refresh=True)
        return IndexedDocument(
            document_id=document.id,
            index=index,
            chunks=len(actions),
        )

    def delete_document(self, document: Document) -> None:
        if document.id is None:
            return
        self.delete_document_id(document.id, DocumentType(document.doc_type))

    def delete_document_id(
        self, document_id: int, document_type: DocumentType | str | None = None
    ) -> None:
        indexes = (
            [index_name(document_type)]
            if document_type is not None
            else list(OPENSEARCH_SCHEMAS)
        )
        for index in indexes:
            if not self.client.indices.exists(index=index):
                continue
            self.client.delete_by_query(
                index=index,
                body={"query": {"term": {"document_id": document_id}}},
                conflicts="proceed",
                refresh=True,
            )

    def delete_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self.delete_document(document)

    def search(
        self,
        query: str,
        *,
        document_type: DocumentType | str,
        limit: int = 5,
        score_threshold: float = config.SCORE_THRESHOLD,
    ) -> list[SearchResult]:
        """Hybrid search one role-specific index using vector and keyword matches."""
        logger.info(
            "Searching for query=%r in document_type=%s with limit=%d score_threshold=%s",
            query,
            document_type,
            limit,
            score_threshold,
        )
        query = query.strip()
        if not query or limit <= 0:
            return []

        index = index_name(document_type)
        if not self.client.indices.exists(index=index):
            return []

        return hybrid_search(
            client=self.client,
            index=index,
            pipeline=HYBRID_SEARCH_PIPELINE,
            query=query,
            query_vector=self.embedding_service.embed(query),
            document_type=document_type,
            document_type_properties=DOCUMENT_TYPE_PROPERTIES,
            limit=limit,
            score_threshold=score_threshold,
        )

    def sql_query(
        self,
        query: str,
        *,
        document_type: DocumentType | str,
    ) -> SqlResult:
        """Run a read-only OpenSearch SQL query against one document index."""
        sql = self._validate_sql_query(query, document_type)
        response = self._sql_request(sql)
        schema = response.get("schema") or []
        columns = [str(column.get("alias") or column.get("name")) for column in schema]
        rows = response.get("datarows") or []
        return SqlResult(columns=columns, rows=rows)

    def _sql_request(self, query: str) -> dict[str, Any]:
        body = {"query": query}
        try:
            return self.client.transport.perform_request(
                "POST",
                "/_plugins/_sql",
                params={"format": "jdbc"},
                body=body,
            )
        except Exception as first_error:
            try:
                return self.client.transport.perform_request(
                    "POST",
                    "/_opendistro/_sql",
                    params={"format": "jdbc"},
                    body=body,
                )
            except Exception:
                raise first_error

    @staticmethod
    def _validate_sql_query(
        query: str,
        document_type: DocumentType | str,
    ) -> str:
        sql = query.strip().rstrip(";")
        if not sql:
            raise ValueError("SQL query cannot be empty")
        normalized = " ".join(sql.lower().split())
        if not normalized.startswith(("select ", "show ", "describe ", "desc ")):
            raise ValueError("Only read-only SELECT, SHOW, and DESCRIBE SQL is allowed")
        if ";" in sql:
            raise ValueError("Only one SQL statement is allowed")

        required_index = index_name(document_type)
        if (
            normalized.startswith("select ")
            and required_index.lower() not in normalized
        ):
            raise ValueError(
                f"SQL must query the role-specific index `{required_index}`"
            )
        return sql

    def _document_chunks(self, file_path: str | Path) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        loaded_documents = DocumentLoader(file_path).load()
        page_count = len(loaded_documents) or None
        for fallback_page_number, loaded_document in enumerate(
            loaded_documents, start=1
        ):
            content = loaded_document.page_content.strip()
            if not content:
                continue
            metadata = dict(loaded_document.metadata or {})
            page_number = self._page_number(metadata, fallback_page_number)
            for chunk_content in self._chunk_text(content):
                chunks.append(
                    {
                        "content": chunk_content,
                        "metadata": metadata,
                        "page_number": page_number,
                        "page_count": self._int_or_none(
                            metadata.get("page_count") or page_count
                        ),
                    }
                )
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        return [
            chunk.strip()
            for chunk in self.text_splitter.split_text(text)
            if chunk.strip()
        ]

    @staticmethod
    def _page_number(metadata: dict[str, Any], fallback: int) -> int:
        page = metadata.get("page_number", metadata.get("page", fallback))
        page_number = KnowledgeBase._int_or_none(page)
        if page_number is None:
            return fallback
        if metadata.get("page") == page and "page_number" not in metadata:
            return page_number + 1
        return page_number

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _type_specific_fields(
        doc_type: DocumentType, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        allowed_fields = DOCUMENT_TYPE_PROPERTIES[doc_type]
        fields: dict[str, Any] = {}
        for field, mapping in allowed_fields.items():
            value = KnowledgeBase._clean_typed_value(metadata.get(field), mapping)
            if value is not None:
                fields[field] = value
        return fields

    @staticmethod
    def _clean_typed_value(value: Any, mapping: dict[str, Any]) -> Any:
        value = KnowledgeBase._clean_value(value)
        if value is None:
            return None

        field_type = mapping.get("type")
        if field_type in {"integer", "short", "long"}:
            return int(value)
        if field_type in {"float", "double", "scaled_float"}:
            return float(value)
        if field_type == "boolean":
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "y"}
            return bool(value)
        if field_type == "date" and isinstance(value, datetime | date):
            return KnowledgeBase._serialize(value)
        return value

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, datetime | date):
            return KnowledgeBase._serialize(value)
        if hasattr(value, "item"):
            return KnowledgeBase._clean_value(value.item())
        if isinstance(value, str):
            return value.strip() or None
        return value

    @staticmethod
    def _serialize(value: datetime | date) -> str:
        if isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
