from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from chatbot.response_generator import build_citations
from chatbot.stakeholder_handler import (
    ROLE_AGENTS,
    ChatbotState,
    build_role_agents,
)
from vector_db.knowledge_base import KnowledgeBase


class PaymentDocumentChatbot:
    """LangGraph router that dispatches each role to its document-search agent."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.agents = build_role_agents(knowledge_base)
        self.graph = self._build_graph()

    def route_query(
        self,
        query: str,
        user_role: str,
        history: list[BaseMessage] | None = None,
    ) -> str:
        state = self.graph.invoke(
            {"query": query, "user_role": user_role, "history": history or []}
        )
        return state["answer"]

    def route_with_metadata(
        self,
        query: str,
        user_role: str,
        history: list[BaseMessage] | None = None,
    ) -> dict[str, Any]:
        state = self.graph.invoke(
            {"query": query, "user_role": user_role, "history": history or []}
        )
        citations = [
            *build_citations(state.get("search_results", [])),
            *state.get("query_citations", []),
        ]
        return {
            "answer": state["answer"],
            "agent": state["routed_agent"],
            "document_type": state["document_type"],
            "citations": citations,
        }

    def stream_with_metadata(
        self,
        query: str,
        user_role: str,
        history: list[BaseMessage] | None = None,
    ) -> Iterator[dict[str, Any]]:
        role = user_role if user_role in ROLE_AGENTS else "product_lead"
        agent = self.agents[role]
        answer_parts = []
        for delta in agent.stream_answer(query, history or []):
            answer_parts.append(delta)
            yield {"delta": delta}

        yield {
            "done": True,
            "answer": "".join(answer_parts).strip(),
            "agent": agent.config.name,
            "document_type": ",".join(
                document_type.value for document_type in agent.config.document_types
            ),
            "citations": [
                *build_citations(agent.last_results),
                *agent.last_query_citations,
            ],
        }

    def _build_graph(self) -> Any:
        graph = StateGraph(ChatbotState)
        graph.add_node("router", self._router_node)
        for role, agent in self.agents.items():
            graph.add_node(role, agent)
            graph.add_edge(role, END)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            lambda state: state["user_role"],
            {role: role for role in self.agents},
        )
        return graph.compile()

    @staticmethod
    def _router_node(state: ChatbotState) -> ChatbotState:
        role = state.get("user_role") or "product_lead"
        if role not in ROLE_AGENTS:
            role = "product_lead"
        return {**state, "user_role": role}


_default_chatbot: PaymentDocumentChatbot | None = None


def _get_default_chatbot() -> PaymentDocumentChatbot:
    global _default_chatbot
    if _default_chatbot is None:
        _default_chatbot = PaymentDocumentChatbot(KnowledgeBase())
    return _default_chatbot


def process_business_query(query: str) -> str:
    return _get_default_chatbot().route_query(query, "product_lead")


def process_technical_query(query: str) -> str:
    return _get_default_chatbot().route_query(query, "tech_lead")


def process_compliance_query(query: str) -> str:
    return _get_default_chatbot().route_query(query, "compliance_lead")


def process_partnership_query(query: str) -> str:
    return _get_default_chatbot().route_query(query, "bank_alliance_lead")


def route_query(query: str, user_role: str) -> str:
    if user_role == "product_lead":
        return process_business_query(query)
    if user_role == "tech_lead":
        return process_technical_query(query)
    if user_role == "compliance_lead":
        return process_compliance_query(query)
    if user_role == "bank_alliance_lead":
        return process_partnership_query(query)
    return process_business_query(query)
