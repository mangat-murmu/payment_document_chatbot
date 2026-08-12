from __future__ import annotations

from functools import cached_property
from typing import Any

import config


class EmbeddingService:
    """BGE-small embedding service with a simple local interface."""

    model_name = config.EMBEDDING_MODEL_NAME
    dimensions = config.EMBEDDING_DIMENSIONS

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name

    @cached_property
    def model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install project dependencies to use BGE embeddings: "
                "pip install -r requirements.txt"
            ) from error

        return SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Cannot compare embeddings with different dimensions")

        return sum(a * b for a, b in zip(left, right))
