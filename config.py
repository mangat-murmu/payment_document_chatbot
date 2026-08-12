from __future__ import annotations

import os
from pathlib import Path

import environment  # noqa: F401

ROOT = Path(__file__).resolve().parent


def _env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str) -> int:
    return int(_env(name))


def _float_env(name: str) -> float:
    return float(_env(name))


def _bool_env(name: str) -> bool:
    return _env(name).strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str) -> Path:
    path = Path(_env(name))
    if path.is_absolute():
        return path
    return ROOT / path


APP_HOST = _env("APP_HOST")
APP_PORT = _int_env("APP_PORT")
APP_RELOAD = _bool_env("APP_RELOAD")

DATABASE_URL = _env("DATABASE_URL")

OPENSEARCH_URL = _env("OPENSEARCH_URL")
OPENSEARCH_INDEX_PREFIX = _env("OPENSEARCH_INDEX_PREFIX")

KNOWLEDGE_BASE_CHUNK_SIZE = _int_env("KNOWLEDGE_BASE_CHUNK_SIZE")
KNOWLEDGE_BASE_CHUNK_OVERLAP = _int_env("KNOWLEDGE_BASE_CHUNK_OVERLAP")

EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL_NAME")
EMBEDDING_DIMENSIONS = _int_env("EMBEDDING_DIMENSIONS")

LOCAL_LLM_MODEL = _env("LOCAL_LLM_MODEL")
LOCAL_LLM_OPENAI_BASE_URL = _env("LOCAL_LLM_OPENAI_BASE_URL")
LOCAL_LLM_API_KEY = _env("LOCAL_LLM_API_KEY")
LOCAL_LLM_TIMEOUT_SECONDS = _float_env("LOCAL_LLM_TIMEOUT_SECONDS")
LOCAL_LLM_TEMPERATURE = _float_env("LOCAL_LLM_TEMPERATURE")
LOCAL_LLM_MAX_TOKENS = _int_env("LOCAL_LLM_MAX_TOKENS")
LOCAL_LLM_ENABLE_THINKING = _bool_env("LOCAL_LLM_ENABLE_THINKING")
LOCAL_LLM_PROMPT_PREFIX = os.getenv("LOCAL_LLM_PROMPT_PREFIX", "")

DOCUMENT_CLASSIFIER_MODEL_PATH = _path_env("DOCUMENT_CLASSIFIER_MODEL_PATH")
ENTITY_EXTRACTOR_MODEL_PATH = _path_env("ENTITY_EXTRACTOR_MODEL_PATH")

RETRIEVER_RESULT_LIMIT = _int_env("RETRIEVER_RESULT_LIMIT")
SCORE_THRESHOLD = _float_env("SCORE_THRESHOLD")

PDF_MIN_TEXT_CHARS = _int_env("PDF_MIN_TEXT_CHARS")
PDF_USE_OCR = _bool_env("PDF_USE_OCR")
PDF_FORCE_OCR = _bool_env("PDF_FORCE_OCR")
PDF_OCR_DPI = _int_env("PDF_OCR_DPI")
PDF_OCR_LANG = _env("PDF_OCR_LANG")
