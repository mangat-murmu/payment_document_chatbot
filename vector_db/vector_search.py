from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.schema import DocumentType


@dataclass(frozen=True)
class SearchResult:
    """One matching chunk returned from OpenSearch."""

    document_id: int
    filename: str
    doc_type: str
    content: str
    score: float
    page_number: int | None = None
    chunk_index: int | None = None


def search_source_fields() -> list[str]:
    return [
        "document_id",
        "filename",
        "doc_type",
        "content",
        "page_number",
        "chunk_index",
    ]


def keyword_search_fields(
    document_type: DocumentType | str,
    document_type_properties: dict[DocumentType, dict[str, Any]],
) -> list[str]:
    fields = ["content^3", "filename^2"]
    resolved_type = DocumentType(document_type)
    for field, mapping in document_type_properties[resolved_type].items():
        if mapping.get("type") == "text":
            fields.append(field)
    return fields


def build_hybrid_query_body(
    *,
    query: str,
    query_vector: list[float],
    document_type: DocumentType | str,
    document_type_properties: dict[DocumentType, dict[str, Any]],
    limit: int,
    score_threshold: float,
) -> dict[str, Any]:
    """Build the OpenSearch hybrid keyword/vector query body."""
    return {
        "size": limit,
        "min_score": score_threshold,
        "_source": search_source_fields(),
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": keyword_search_fields(
                                document_type,
                                document_type_properties,
                            ),
                            "type": "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO",
                        }
                    },
                    {
                        "knn": {
                            "content_vector": {
                                "vector": query_vector,
                                "min_score": score_threshold,
                            }
                        }
                    },
                ]
            }
        },
    }


def hybrid_search(
    *,
    client: Any,
    index: str,
    pipeline: str,
    query: str,
    query_vector: list[float],
    document_type: DocumentType | str,
    document_type_properties: dict[DocumentType, dict[str, Any]],
    limit: int,
    score_threshold: float,
) -> list[SearchResult]:
    """Execute hybrid search against one OpenSearch vector index."""
    response = client.search(
        index=index,
        params={"search_pipeline": pipeline},
        body=build_hybrid_query_body(
            query=query,
            query_vector=query_vector,
            document_type=document_type,
            document_type_properties=document_type_properties,
            limit=limit,
            score_threshold=score_threshold,
        ),
    )
    hits = response.get("hits", {}).get("hits", [])
    return [search_result_from_hit(hit) for hit in hits]


def search_result_from_hit(hit: dict[str, Any]) -> SearchResult:
    source = hit["_source"]
    return SearchResult(
        document_id=int(source["document_id"]),
        filename=str(source["filename"]),
        doc_type=str(source["doc_type"]),
        content=str(source["content"]),
        score=float(hit.get("_score") or 0),
        page_number=_int_or_none(source.get("page_number")),
        chunk_index=_int_or_none(source.get("chunk_index")),
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
