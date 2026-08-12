"""Typed CRUD repositories for the application's SQLModel entities."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import Session, select

from database.models import BaseTable, Chat, Document, Message

ModelT = TypeVar("ModelT", bound=BaseTable)
ModelData = ModelT | BaseModel | Mapping[str, Any]


class BaseRepository(Generic[ModelT]):
    """Provide common persistence operations for one SQLModel model.

    A repository owns no session lifecycle; the caller supplies a session and
    remains responsible for closing it.  Mutating operations commit immediately
    and roll back the session when the database rejects a change.
    """

    def __init__(self, session: Session, model: type[ModelT]) -> None:
        self.session = session
        self.model = model

    def create(self, data: ModelData) -> ModelT:
        if isinstance(data, self.model):
            instance = data
        else:
            values = self._values(data)
            self._validate_fields(values)
            instance = self.model.model_validate(values)
        self.session.add(instance)
        self._commit()
        self.session.refresh(instance)
        return instance

    def get(self, object_id: int) -> ModelT | None:
        return self.session.get(self.model, object_id)

    def get_or_raise(self, object_id: int) -> ModelT:
        instance = self.get(object_id)
        if instance is None:
            raise LookupError(
                f"{self.model.__name__} with id {object_id} was not found"
            )
        return instance

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        **filters: Any,
    ) -> list[ModelT]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        statement = select(self.model)
        for field, value in filters.items():
            statement = statement.where(self._column(field) == value)
        statement = statement.order_by(self.model.id).offset(offset).limit(limit)
        return list(self.session.exec(statement).all())

    def update(
        self, object_id: int, data: BaseModel | Mapping[str, Any]
    ) -> ModelT | None:
        instance = self.get(object_id)
        if instance is None:
            return None

        values = self._values(data, exclude_unset=True)
        for field in ("id", "created_at", "updated_at"):
            values.pop(field, None)
        self._validate_fields(values)
        for field, value in values.items():
            setattr(instance, field, value)
        instance.updated_at = datetime.now(timezone.utc)

        self.session.add(instance)
        self._commit()
        self.session.refresh(instance)
        return instance

    def delete(self, object_id: int) -> bool:
        instance = self.get(object_id)
        if instance is None:
            return False
        self.session.delete(instance)
        self._commit()
        return True

    def delete_many(self, object_ids: list[int]) -> tuple[list[ModelT], list[int]]:
        unique_ids = list(dict.fromkeys(object_ids))
        statement = select(self.model).where(self.model.id.in_(unique_ids))
        instances = list(self.session.exec(statement).all())
        instance_by_id = {instance.id: instance for instance in instances}
        missing_ids = [object_id for object_id in unique_ids if object_id not in instance_by_id]
        if missing_ids:
            return [], missing_ids

        for object_id in unique_ids:
            self.session.delete(instance_by_id[object_id])
        self._commit()
        return [instance_by_id[object_id] for object_id in unique_ids], []

    def count(self, **filters: Any) -> int:
        statement = select(func.count()).select_from(self.model)
        for field, value in filters.items():
            statement = statement.where(self._column(field) == value)
        return int(self.session.exec(statement).one())

    def _column(self, field: str) -> Any:
        if field not in self.model.model_fields:
            raise ValueError(f"unknown {self.model.__name__} field: {field}")
        return getattr(self.model, field)

    def _validate_fields(self, values: Mapping[str, Any]) -> None:
        for field in values:
            self._column(field)

    @staticmethod
    def _values(
        data: BaseModel | Mapping[str, Any], *, exclude_unset: bool = False
    ) -> dict[str, Any]:
        if isinstance(data, Mapping):
            return dict(data)
        if isinstance(data, BaseModel):
            return data.model_dump(exclude_unset=exclude_unset)
        raise TypeError("data must be a Pydantic model or mapping")

    def _commit(self) -> None:
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise


CRUDRepository = BaseRepository


class DocumentRepository(BaseRepository[Document]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Document)

    def create_many(self, documents: list[Mapping[str, Any]]) -> list[Document]:
        instances = []
        for document in documents:
            values = self._values(document)
            self._validate_fields(values)
            instances.append(Document.model_validate(values))

        self.session.add_all(instances)
        self._commit()
        for instance in instances:
            self.session.refresh(instance)
        return instances


class ChatRepository(BaseRepository[Chat]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Chat)

    def list(self, *, offset: int = 0, limit: int = 100, **filters: Any) -> list[Chat]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        statement = select(Chat)
        for field, value in filters.items():
            statement = statement.where(self._column(field) == value)
        statement = statement.order_by(Chat.updated_at.desc(), Chat.id.desc())
        statement = statement.offset(offset).limit(limit)
        return list(self.session.exec(statement).all())

    def touch(self, object_id: int) -> Chat | None:
        chat = self.get(object_id)
        if chat is None:
            return None
        chat.updated_at = datetime.now(timezone.utc)
        self.session.add(chat)
        self._commit()
        self.session.refresh(chat)
        return chat


class MessageRepository(BaseRepository[Message]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Message)

    def list_for_chat(
        self, chat_id: int, *, offset: int = 0, limit: int = 100
    ) -> list[Message]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        statement = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at, Message.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.exec(statement).all())
