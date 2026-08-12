from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlmodel import Session
from starlette.concurrency import iterate_in_threadpool

from api.dependencies import database_session, knowledge_base
from api.schema import (
    ChatGenerationRequest,
    ChatGenerationResponse,
    ChatListResponse,
    ChatResponse,
    MessageListResponse,
    MessageResponse,
    UpdateChatRequest,
)
from chatbot.query_router import PaymentDocumentChatbot
from database.models import Chat, Message
from database.repository import ChatRepository, MessageRepository
from vector_db.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/api/chats", tags=["Chats"])


def _get_chat_or_404(repository: ChatRepository, chat_id: int) -> Chat:
    chat = repository.get(chat_id)
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="chat not found"
        )
    return chat


def _message_text(content: str | dict) -> str:
    if isinstance(content, str):
        return content
    text = content.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(content)


def _history_messages(
    message_repository: MessageRepository,
    chat_id: int,
) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for message in message_repository.list_for_chat(chat_id, limit=100):
        text = _message_text(message.content)
        if message.role == "user":
            history.append(HumanMessage(content=text))
        elif message.role == "assistant":
            history.append(AIMessage(content=text))
    return history


def _prepare_chat_request(
    payload: ChatGenerationRequest,
    session: Session,
) -> tuple[Chat, Message, list[BaseMessage]]:
    chat_repository = ChatRepository(session)
    if payload.chat_id is None:
        chat = chat_repository.create({"title": payload.input.text[:255]})
    else:
        chat = _get_chat_or_404(chat_repository, payload.chat_id)

    if payload.last_response_id is not None:
        previous_message = MessageRepository(session).get(payload.last_response_id)
        if previous_message is None or previous_message.chat_id != chat.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="last response not found in chat",
            )

    message_repository = MessageRepository(session)
    history = _history_messages(message_repository, chat.id)
    user_message = message_repository.create(
        {
            "chat_id": chat.id,
            "previous_id": payload.last_response_id,
            "role": "user",
            "content": {"text": payload.input.text},
        }
    )
    return chat, user_message, history


def _save_assistant_message(
    *,
    session: Session,
    chat: Chat,
    previous_id: int | None,
    answer: str,
    agent: str,
    document_type: str,
    citations: list[dict],
) -> tuple[Chat, MessageResponse]:
    message_repository = MessageRepository(session)
    assistant_message = message_repository.create(
        {
            "chat_id": chat.id,
            "previous_id": previous_id,
            "role": "assistant",
            "name": "PayDoc AI",
            "content": {"text": answer},
            "meta_data": {
                "generation": "langgraph_role_agent",
                "agent": agent,
                "document_type": document_type,
                "citations": citations,
            },
        }
    )
    chat = ChatRepository(session).touch(chat.id)
    assert chat is not None
    return chat, MessageResponse.model_validate(assistant_message)


async def _create_chat_response(
    payload: ChatGenerationRequest,
    session: Session,
    kb: KnowledgeBase,
    user_role: str,
) -> tuple[Chat, MessageResponse]:
    chat, user_message, history = _prepare_chat_request(payload, session)
    chatbot = PaymentDocumentChatbot(kb)
    result = await run_in_threadpool(
        chatbot.route_with_metadata,
        payload.input.text,
        user_role,
        history,
    )
    return _save_assistant_message(
        session=session,
        chat=chat,
        previous_id=user_message.id,
        answer=result["answer"],
        agent=result["agent"],
        document_type=result["document_type"],
        citations=result["citations"],
    )


async def _stream_chat_response(
    *,
    chat: Chat,
    user_message: Message,
    history: list[BaseMessage],
    query: str,
    user_role: str,
    session: Session,
    kb: KnowledgeBase,
):
    yield f"data: {json.dumps({'chat_id': chat.id})}\n\n"
    chatbot = PaymentDocumentChatbot(kb)
    final_event = None
    events = chatbot.stream_with_metadata(query, user_role, history)
    async for event in iterate_in_threadpool(events):
        if event.get("delta"):
            yield f"data: {json.dumps({'delta': event['delta']})}\n\n"
        if event.get("done"):
            final_event = event

    if final_event is None:
        final_event = {
            "answer": "",
            "agent": "PayDoc AI",
            "document_type": "",
            "citations": [],
        }
    _, message = _save_assistant_message(
        session=session,
        chat=chat,
        previous_id=user_message.id,
        answer=final_event["answer"],
        agent=final_event["agent"],
        document_type=final_event["document_type"],
        citations=final_event["citations"],
    )
    yield (
        "data: "
        + json.dumps(
            {
                "response_id": message.id,
                "done": True,
                "citations": final_event["citations"],
            }
        )
        + "\n\n"
    )


@router.post("", response_model=ChatGenerationResponse)
async def generate_chat_response(
    request: Request,
    payload: ChatGenerationRequest,
    session: Annotated[Session, Depends(database_session)],
    kb: Annotated[KnowledgeBase, Depends(knowledge_base)],
) -> ChatGenerationResponse | StreamingResponse:
    user_role = request.state.user_role
    if payload.stream:
        chat, user_message, history = _prepare_chat_request(payload, session)
        return StreamingResponse(
            _stream_chat_response(
                chat=chat,
                user_message=user_message,
                history=history,
                query=payload.input.text,
                user_role=user_role,
                session=session,
                kb=kb,
            ),
            media_type="text/event-stream",
        )
    chat, message = await _create_chat_response(payload, session, kb, user_role)
    return ChatGenerationResponse(
        chat=ChatResponse.model_validate(chat),
        message=message,
    )


@router.get("", response_model=ChatListResponse)
def list_chats(
    session: Annotated[Session, Depends(database_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ChatListResponse:
    chats = ChatRepository(session).list(offset=offset, limit=limit)
    return ChatListResponse(chats=[ChatResponse.model_validate(chat) for chat in chats])


@router.get("/{chat_id}", response_model=ChatResponse)
def get_chat(
    chat_id: int,
    session: Annotated[Session, Depends(database_session)],
) -> ChatResponse:
    return ChatResponse.model_validate(
        _get_chat_or_404(ChatRepository(session), chat_id)
    )


@router.put("/{chat_id}", response_model=ChatResponse)
def update_chat(
    chat_id: int,
    payload: UpdateChatRequest,
    session: Annotated[Session, Depends(database_session)],
) -> ChatResponse:
    chat = ChatRepository(session).update(chat_id, payload)
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="chat not found"
        )
    return ChatResponse.model_validate(chat)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: int,
    session: Annotated[Session, Depends(database_session)],
) -> Response:
    deleted = ChatRepository(session).delete(chat_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="chat not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{chat_id}/messages", response_model=MessageListResponse)
def list_messages(
    chat_id: int,
    session: Annotated[Session, Depends(database_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> MessageListResponse:
    _get_chat_or_404(ChatRepository(session), chat_id)
    messages = MessageRepository(session).list_for_chat(
        chat_id, offset=offset, limit=limit
    )
    return MessageListResponse(
        messages=[MessageResponse.model_validate(message) for message in messages]
    )
