from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

import config
from vector_db.knowledge_base import SqlResult
from vector_db.vector_search import SearchResult


@dataclass(frozen=True)
class Citation:
    document_id: int
    title: str
    chunk_indexes: list[int]
    page_numbers: list[int]
    score: float

    def as_dict(self) -> dict[str, Any]:
        first_chunk = self.chunk_indexes[0] if self.chunk_indexes else None
        first_page = self.page_numbers[0] if self.page_numbers else None
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunk_index": first_chunk,
            "index_number": first_chunk,
            "chunk_indexes": self.chunk_indexes,
            "index_numbers": self.chunk_indexes,
            "page_number": first_page,
            "page_numbers": self.page_numbers,
            "score": self.score,
        }


def build_llm() -> ChatOpenAI:
    """Create an OpenAI-compatible LangChain chat model for the local Qwen server."""
    return ChatOpenAI(
        model=config.LOCAL_LLM_MODEL,
        base_url=config.LOCAL_LLM_OPENAI_BASE_URL,
        api_key=config.LOCAL_LLM_API_KEY,
        temperature=config.LOCAL_LLM_TEMPERATURE,
        timeout=config.LOCAL_LLM_TIMEOUT_SECONDS,
        max_tokens=config.LOCAL_LLM_MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": config.LOCAL_LLM_ENABLE_THINKING
            }
        },
    )


def build_agent_prompt(
    *,
    agent_name: str,
    document_types: list[str],
    perspective: str,
    tool_names: list[str],
) -> str:
    readable_document_types = ", ".join(
        document_type.replace("_", " ") for document_type in document_types
    )
    readable_tool_names = ", ".join(f"`{tool_name}`" for tool_name in tool_names)
    prefix = (
        f"{config.LOCAL_LLM_PROMPT_PREFIX}\n"
        if config.LOCAL_LLM_PROMPT_PREFIX
        else ""
    )
    return (
        prefix
        + f"You are {agent_name}, a payment-document assistant for {perspective}. "
        f"Current datetime: {datetime.now().astimezone().isoformat()}. "
        f"Tools: {readable_tool_names}. For every payment, document, data, metric, "
        "compliance, bank, transaction, log, SLA, or agreement question, you must "
        "call at least one relevant tool before answering. You may skip tools only "
        "for greetings or generic small talk like hi, hello, thanks, who are you, "
        "or what can you do. Never answer domain questions from memory. "
        f"Answer only from retrieved {readable_document_types} evidence.\n"
        "Tool choice: use SQL for exact analytics: counts, totals, success/failure "
        "rates, percentages, sums, averages, min/max, GROUP BY, ORDER BY, filters, "
        "date windows such as today/this month, and status or risk breakdowns. "
        "Example: `What's the success rate of UPI transactions this month?` must "
        "use SQL, not vector search. Use vector search only for fuzzy/document-text "
        "questions: clauses, obligations, explanations, incident context, or finding "
        "relevant passages. If a question needs both metric and explanation, call "
        "SQL first, then vector search for supporting context.\n"
        "SQL: follow the SQL tool schema/rules exactly; CASE branches must have one "
        "type, so use 1.0/0.0 for rates. If SQL returns an error, revise and retry "
        "before answering. Keep answers concise, practical, markdown-formatted, and "
        "cite document ids, filenames, pages, or executed SQL."
    )


def build_user_message(query: str) -> str:
    if config.LOCAL_LLM_PROMPT_PREFIX:
        return f"{config.LOCAL_LLM_PROMPT_PREFIX}\n{query}"
    return query


def format_search_result(result: SearchResult) -> str:
    page = f", page {result.page_number}" if result.page_number else ""
    return (
        f"[doc:{result.document_id}{page}, score:{result.score:.3f}, "
        f"file:{result.filename}] {result.content}"
    )


def format_tool_results(results: list[SearchResult]) -> str:
    if not results:
        return "No matching chunks were retrieved from this role-specific index."
    return "\n\n".join(
        f"{index}. {format_search_result(result)}"
        for index, result in enumerate(results, 1)
    )


def format_sql_result(result: SqlResult) -> str:
    if not result.columns:
        return "The SQL query completed but returned no columns."
    if not result.rows:
        return "The SQL query returned no rows."

    header = "| " + " | ".join(result.columns) + " |"
    separator = "| " + " | ".join("---" for _ in result.columns) + " |"
    rows = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in result.rows[:50]
    ]
    suffix = ""
    if len(result.rows) > 50:
        suffix = f"\n\nShowing first 50 of {len(result.rows)} rows."
    return "\n".join([header, separator, *rows]) + suffix


def build_citations(results: list[SearchResult]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for result in results:
        citation = grouped.setdefault(
            result.document_id,
            {
                "title": result.filename,
                "chunk_indexes": [],
                "page_numbers": [],
                "score": result.score,
            },
        )
        citation["score"] = max(citation["score"], result.score)
        if (
            result.chunk_index is not None
            and result.chunk_index not in citation["chunk_indexes"]
        ):
            citation["chunk_indexes"].append(result.chunk_index)
        if (
            result.page_number is not None
            and result.page_number not in citation["page_numbers"]
        ):
            citation["page_numbers"].append(result.page_number)

    citations = [
        Citation(
            document_id=document_id,
            title=str(citation["title"]),
            chunk_indexes=sorted(citation["chunk_indexes"]),
            page_numbers=sorted(citation["page_numbers"]),
            score=float(citation["score"]),
        )
        for document_id, citation in grouped.items()
    ]
    return [citation.as_dict() for citation in citations]


def final_message_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type != "ai":
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type not in (None, "text"):
                        continue
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            text = "\n".join(part for part in parts if part.strip()).strip()
            if text:
                return text
    return "I could not generate a response from the selected agent."


def search_error_answer(
    *,
    agent_name: str,
    document_types: list[str],
    error: Exception,
) -> str:
    readable_document_type = ", ".join(
        document_type.replace("_", " ") for document_type in document_types
    )
    return (
        f"{agent_name}: I could not use the {readable_document_type} retrieval/query "
        f"tools right now: {error}"
    )
