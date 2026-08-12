from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import config as app_config
from api.schema import DocumentType
from chatbot.response_generator import (
    build_agent_prompt,
    build_llm,
    build_user_message,
    final_message_text,
    format_sql_result,
    format_tool_results,
    search_error_answer,
)
from vector_db.knowledge_base import (
    BASE_PROPERTIES,
    DOCUMENT_TYPE_PROPERTIES,
    KnowledgeBase,
    index_name,
)
from vector_db.vector_search import SearchResult

UserRole = Literal[
    "product_lead",
    "tech_lead",
    "compliance_lead",
    "bank_alliance_lead",
]


class ChatbotState(TypedDict, total=False):
    query: str
    history: list[BaseMessage]
    user_role: str
    routed_agent: str
    document_type: str
    search_results: list[SearchResult]
    query_citations: list[dict]
    answer: str


@dataclass(frozen=True)
class VectorRetriver:
    document_type: DocumentType
    tool_name: str
    description: str


@dataclass(frozen=True)
class SQLRetriever:
    document_type: DocumentType
    tool_name: str
    description: str


class VectorRetrieverInput(BaseModel):
    query: str = Field(
        description=(
            "The query to search for in the OpenSearch vector index. "
            "User may ask a vague or incomplete question, but the "
            "query should have full context and be specific enough to retrieve relevant results. "
            "For example, instead of 'transaction', use 'failed UPI transaction with error code 404'."
        )
    )


SQL_QUERY_RULES = (
    "Write one read-only OpenSearch SQL statement using the exact role-specific "
    "index. Use SQL for counts, totals, success/failure rates, percentages, sums, "
    "averages, min/max, GROUP BY, ORDER BY, filters, and date windows like today "
    "or this month. No semicolons. Prefer LIMIT 50 or less. For exact filters/"
    "grouping on text fields, use the `.keyword` subfield when available. CASE expressions "
    "must return the same type in every THEN/ELSE branch: use 1.0/0.0 for ratios "
    "or CAST all branches to DOUBLE/INTEGER consistently. Do not mix INTEGER and "
    "DOUBLE in CASE. Use AVG(CASE WHEN condition THEN 1.0 ELSE 0.0 END) for rates."
)

SQL_QUERY_EXAMPLES = (
    " Example queries: "
    "success rate: SELECT COUNT(*) AS total, "
    "SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS successful, "
    "AVG(CASE WHEN status = 'SUCCESS' THEN 1.0 ELSE 0.0 END) * 100 AS success_rate "
    "FROM {index} WHERE timestamp >= '2026-08-01' AND timestamp < '2026-09-01'; "
    "status breakdown: SELECT status, COUNT(*) AS count FROM {index} "
    "GROUP BY status ORDER BY count DESC LIMIT 20; "
    "daily trend: SELECT DATE(timestamp) AS day, COUNT(*) AS total FROM {index} "
    "GROUP BY DATE(timestamp) ORDER BY day"
)


class SQLRetrieverInput(BaseModel):
    query: str = Field(
        description=(f"The OpenSearch SQL query to execute. {SQL_QUERY_RULES}")
    )


@dataclass(frozen=True)
class RoleAgentConfig:
    name: str
    role: UserRole
    perspective: str
    tools: tuple[VectorRetriver | SQLRetriever, ...]

    @property
    def vector_retrievers(self) -> tuple[VectorRetriver, ...]:
        return tuple(tool for tool in self.tools if isinstance(tool, VectorRetriver))

    @property
    def sql_retrievers(self) -> tuple[SQLRetriever, ...]:
        return tuple(tool for tool in self.tools if isinstance(tool, SQLRetriever))

    @property
    def document_types(self) -> tuple[DocumentType, ...]:
        seen: set[DocumentType] = set()
        document_types: list[DocumentType] = []
        for tool in self.tools:
            if tool.document_type not in seen:
                seen.add(tool.document_type)
                document_types.append(tool.document_type)
        return tuple(document_types)


SQL_VALUE_EXAMPLES: dict[DocumentType, dict[str, list[str]]] = {
    DocumentType.UPI_TRANSACTION: {
        "status": ["SUCCESS", "FAILED", "PENDING", "REVERSED", "REFUNDED"],
        "response_code": ["00", "U16", "U30", "91", "96"],
        "settlement_status": ["SETTLED", "PENDING", "FAILED"],
        "payment_mode": ["UPI", "UPI_LITE", "AUTOPAY"],
        "fraud_flag": ["true", "false"],
    },
    DocumentType.BANK_API_LOG: {
        "status": ["SUCCESS", "FAILED", "TIMEOUT", "PENDING"],
        "severity": ["INFO", "WARN", "SEV1", "SEV2", "SEV3"],
        "reconciliation_status": ["MATCHED", "MISMATCH", "PENDING"],
        "operation": ["PAY", "REFUND", "STATUS_CHECK", "RECONCILIATION"],
        "integration_direction": ["INBOUND", "OUTBOUND"],
    },
}


def _mapping_type(mapping: dict[str, Any]) -> str:
    field_type = str(mapping.get("type") or "object")
    if field_type == "text" and "keyword" in mapping.get("fields", {}):
        return "text; use field.keyword for exact filters/grouping"
    if field_type == "scaled_float":
        return "DOUBLE/scaled_float"
    if field_type in {"integer", "short", "long"}:
        return "INTEGER"
    if field_type == "date":
        return "DATE"
    if field_type == "boolean":
        return "BOOLEAN"
    if field_type == "keyword":
        return "KEYWORD/string"
    return field_type.upper()


def sql_value_examples(document_type: DocumentType) -> str:
    examples = SQL_VALUE_EXAMPLES.get(document_type)
    if not examples:
        return ""
    values = "; ".join(
        f"{field}: {', '.join(items)}" for field, items in examples.items()
    )
    return f" Example filter values: {values}."


def sql_schema_description(document_type: DocumentType) -> str:
    properties = {**BASE_PROPERTIES, **DOCUMENT_TYPE_PROPERTIES[document_type]}
    fields = ", ".join(
        f"{field} ({_mapping_type(mapping)})"
        for field, mapping in properties.items()
        if field != "content_vector"
    )
    return (
        f"Index: `{index_name(document_type)}`. SQL rules: {SQL_QUERY_RULES} "
        f"{SQL_QUERY_EXAMPLES.format(index=index_name(document_type))} "
        f"Schema fields: {fields}.{sql_value_examples(document_type)}"
    )


ROLE_AGENTS: dict[str, RoleAgentConfig] = {
    "product_lead": RoleAgentConfig(
        name="Product Lead Agent",
        role="product_lead",
        perspective="business and product performance",
        tools=(
            SQLRetriever(
                document_type=DocumentType.UPI_TRANSACTION,
                tool_name="sql_upi_transactions",
                description=f"Use this for exact UPI transaction analytics. Required for success rate, failure rate, percentages, counts, filters, GROUP BY, ORDER BY, sums, averages, date ranges such as today/this month, status/response_code breakdowns, amount, bank, merchant, risk_score, and fraud_flag analysis. Do not use vector search for these calculations. {sql_schema_description(DocumentType.UPI_TRANSACTION)}",
            ),
            VectorRetriver(
                document_type=DocumentType.UPI_TRANSACTION,
                tool_name="search_upi_transactions",
                description="Use only for fuzzy semantic search over UPI transaction chunks: explanations, examples, suspicious-pattern context, or finding relevant narrative evidence. Do not use for exact counts, rates, percentages, filters, GROUP BY, sums, averages, date windows, or status breakdowns; use sql_upi_transactions for those.",
            ),
        ),
    ),
    "tech_lead": RoleAgentConfig(
        name="Tech Lead Agent",
        role="tech_lead",
        perspective="technical reliability, integrations, latency, and failures",
        tools=(
            SQLRetriever(
                document_type=DocumentType.BANK_API_LOG,
                tool_name="sql_bank_api_logs",
                description=f"Use this for exact bank integration log analytics. Required for counts, rates, percentages, filters, GROUP BY, ORDER BY, AVG(latency_ms), MAX(latency_ms), status/severity/reconciliation_status breakdowns, bank or operation comparisons, timestamps/date windows, amount mismatches, settlement_id, trace_id, and request_id lookups. Do not use vector search for these calculations. {sql_schema_description(DocumentType.BANK_API_LOG)}",
            ),
            VectorRetriver(
                document_type=DocumentType.BANK_API_LOG,
                tool_name="search_bank_api_logs",
                description="Use only for fuzzy semantic search over bank API log chunks: incident narratives, error context, operational evidence, or examples. Do not use for exact counts, rates, filters, GROUP BY, averages, latency stats, status/reconciliation breakdowns, or date windows; use sql_bank_api_logs for those.",
            ),
        ),
    ),
    "compliance_lead": RoleAgentConfig(
        name="Compliance Lead Agent",
        role="compliance_lead",
        perspective="regulatory compliance, circulars, controls, and audit impact",
        tools=(
            VectorRetriver(
                document_type=DocumentType.COMPLIANCE_AUDIT,
                tool_name="search_compliance_documents",
                description="Search compliance and audit documents for circulars, regulatory obligations, controls, risks, and deadlines.",
            ),
        ),
    ),
    "bank_alliance_lead": RoleAgentConfig(
        name="Bank Alliance Agent",
        role="bank_alliance_lead",
        perspective="bank partnerships, agreements, SLAs, and commercial terms",
        tools=(
            VectorRetriver(
                document_type=DocumentType.PARTNERSHIP_SLA,
                tool_name="search_partnership_slas",
                description="Use for semantic search over bank partnership agreements and SLA documents: uptime clauses, commercial terms, obligations, escalation, termination, and contract text.",
            ),
            SQLRetriever(
                document_type=DocumentType.BANK_API_LOG,
                tool_name="sql_bank_integration_logs",
                description=f"Use this for exact integration-log metrics supporting bank alliance questions. Required for counts, rates, percentages, SLA performance metrics, counts by bank/status/severity, AVG(latency_ms), MAX(latency_ms), reconciliation mismatches, operation-level failures, timestamp/date windows, gross_amount, fees, and net_amount. Do not use vector search for these calculations. {sql_schema_description(DocumentType.BANK_API_LOG)}",
            ),
            VectorRetriver(
                document_type=DocumentType.BANK_API_LOG,
                tool_name="search_bank_integration_logs",
                description="Use only for fuzzy semantic search over bank integration logs: API performance narratives, outages, failures, reconciliation context, and operational SLA evidence. Do not use for exact SLA metrics, latency averages, error counts, rates, grouped bank/operation analysis, or date windows; use sql_bank_integration_logs for those.",
            ),
        ),
    ),
}


class RoleAgent:
    """A role-scoped LangChain agent with retriever tools."""

    def __init__(self, config: RoleAgentConfig, knowledge_base: KnowledgeBase) -> None:
        self.config = config
        self.knowledge_base = knowledge_base
        self.last_results: list[SearchResult] = []
        self.last_query_citations: list[dict] = []
        self.tools = [self._build_tool(tool) for tool in config.tools]
        self.agent = create_agent(
            model=build_llm(),
            tools=self.tools,
            system_prompt=build_agent_prompt(
                agent_name=config.name,
                document_types=[
                    document_type.value for document_type in config.document_types
                ],
                perspective=config.perspective,
                tool_names=[tool.tool_name for tool in config.tools],
            ),
            name=config.role,
        )

    def _build_tool(self, tool: VectorRetriver | SQLRetriever) -> StructuredTool:
        if isinstance(tool, VectorRetriver):
            return self._build_retriever_tool(tool)
        return self._build_sql_tool(tool)

    def _build_retriever_tool(self, retriever: VectorRetriver) -> StructuredTool:
        def retrieve(query: str) -> str:
            results = self.knowledge_base.search(
                query,
                document_type=retriever.document_type,
                limit=app_config.RETRIEVER_RESULT_LIMIT,
                score_threshold=app_config.SCORE_THRESHOLD,
            )
            self.last_results.extend(results)
            return format_tool_results(results)

        return StructuredTool.from_function(
            name=retriever.tool_name,
            description=retriever.description,
            func=retrieve,
            args_schema=VectorRetrieverInput,
        )

    def _build_sql_tool(self, sql_tool: SQLRetriever) -> StructuredTool:
        def query_opensearch_sql(query: str) -> str:
            try:
                result = self.knowledge_base.sql_query(
                    query,
                    document_type=sql_tool.document_type,
                )
            except Exception as error:
                return (
                    "The SQL query failed. Revise and call this SQL tool again. "
                    f"{SQL_QUERY_RULES} Error: {error}"
                )
            self.last_query_citations.append(
                {
                    "citation_type": "query",
                    "tool_name": sql_tool.tool_name,
                    "document_type": sql_tool.document_type.value,
                    "title": query,
                    "query": query,
                }
            )
            return format_sql_result(result)

        return StructuredTool.from_function(
            name=sql_tool.tool_name,
            description=sql_tool.description,
            func=query_opensearch_sql,
            args_schema=SQLRetrieverInput,
        )

    def __call__(self, state: ChatbotState) -> ChatbotState:
        query = state["query"]
        self.last_results = []
        self.last_query_citations = []
        try:
            response = self.agent.invoke(
                {
                    "messages": [
                        *state.get("history", []),
                        HumanMessage(content=build_user_message(query)),
                    ]
                },
                config={"recursion_limit": 10},
            )
        except Exception as error:
            answer = search_error_answer(
                agent_name=self.config.name,
                document_types=[
                    document_type.value for document_type in self.config.document_types
                ],
                error=error,
            )
            return self._state(state, self.last_results, answer)

        return self._state(
            state,
            self.last_results,
            final_message_text(response.get("messages", [])),
        )

    def stream_answer(
        self,
        query: str,
        history: list[BaseMessage] | None = None,
    ) -> Iterator[str]:
        self.last_results = []
        self.last_query_citations = []
        try:
            events = self.agent.stream(
                {
                    "messages": [
                        *(history or []),
                        HumanMessage(content=build_user_message(query)),
                    ]
                },
                config={"recursion_limit": 10},
                stream_mode="messages",
            )
            for event in events:
                text = self._stream_text(event)
                if text:
                    yield text
        except Exception as error:
            yield search_error_answer(
                agent_name=self.config.name,
                document_types=[
                    document_type.value for document_type in self.config.document_types
                ],
                error=error,
            )

    def _state(
        self,
        state: ChatbotState,
        results: list[SearchResult],
        answer: str,
    ) -> ChatbotState:
        return {
            **state,
            "routed_agent": self.config.name,
            "document_type": ",".join(
                document_type.value for document_type in self.config.document_types
            ),
            "search_results": results,
            "query_citations": self.last_query_citations,
            "answer": answer,
        }

    @staticmethod
    def _stream_text(event: Any) -> str:
        message = event[0] if isinstance(event, tuple) else event
        if not RoleAgent._is_assistant_message(message):
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
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
            return "".join(parts)
        return ""

    @staticmethod
    def _is_assistant_message(message: Any) -> bool:
        message_type = getattr(message, "type", None)
        if message_type == "ai":
            return True
        return message.__class__.__name__.startswith("AIMessage")


def build_role_agents(knowledge_base: KnowledgeBase) -> dict[str, RoleAgent]:
    return {
        role: RoleAgent(config, knowledge_base) for role, config in ROLE_AGENTS.items()
    }
