"""Load PDF, CSV, JSON, JSONL, and text documents."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

import pandas as pd
import pymupdf
import pymupdf4llm
import pytesseract
from langchain_community.document_loaders import DataFrameLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from PIL import Image

import config

logger = logging.getLogger(__name__)


class DocumentFormat(StrEnum):
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TEXT = "txt"


class PdfLoader(BaseLoader):
    """Custom PDF loader using PyMuPDF4LLM with OCR fallback."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        min_text_chars: int = config.PDF_MIN_TEXT_CHARS,
        use_ocr: bool = config.PDF_USE_OCR,
        force_ocr: bool = config.PDF_FORCE_OCR,
        ocr_dpi: int = config.PDF_OCR_DPI,
        ocr_lang: str = config.PDF_OCR_LANG,
    ) -> None:
        self.file_path = Path(file_path)
        self.min_text_chars = min_text_chars
        self.use_ocr = use_ocr
        self.force_ocr = force_ocr
        self.ocr_dpi = ocr_dpi
        self.ocr_lang = ocr_lang

    def lazy_load(self) -> Iterator[Document]:
        try:
            md_pages = pymupdf4llm.to_markdown(
                self.file_path,
                page_chunks=True,
                show_progress=False,
                use_ocr=self.use_ocr,
                force_ocr=self.force_ocr,
                ocr_dpi=self.ocr_dpi,
                ocr_language=self.ocr_lang,
            )
        except Exception:
            logger.exception("Failed to extract PDF text using PyMuPDF4LLM")
            md_pages = []

        if isinstance(md_pages, list):
            documents = []
            page_count = len(md_pages)
            for index, page in enumerate(md_pages, start=1):
                if not isinstance(page, dict):
                    continue
                metadata = page.get("metadata") or {}
                documents.append(
                    Document(
                        page.get("text", ""),
                        metadata={
                            **metadata,
                            "source": str(self.file_path),
                            "page_number": index,
                            "page_count": page_count,
                        },
                    )
                )
            if any(document.page_content.strip() for document in documents):
                with pymupdf.open(self.file_path) as pdf:
                    matrix = pymupdf.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
                    for index, document in enumerate(documents):
                        if len(document.page_content.strip()) >= self.min_text_chars:
                            yield document
                            continue
                        page = pdf[index]
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        yield Document(
                            pytesseract.image_to_string(
                                img, lang=self.ocr_lang
                            ).strip(),
                            metadata={
                                "source": str(self.file_path),
                                "page_number": page.number + 1,
                                "page_count": len(pdf),
                                "extraction_method": "ocr",
                            },
                        )
                return

        logger.warning("No text extracted from PDF. Falling back to page OCR.")
        matrix = pymupdf.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
        with pymupdf.open(self.file_path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                yield Document(
                    pytesseract.image_to_string(img, lang=self.ocr_lang).strip(),
                    metadata={
                        "source": str(self.file_path),
                        "page_number": page.number + 1,
                        "page_count": len(pdf),
                        "extraction_method": "ocr",
                    },
                )
            return


def detect_document_format(file_path: str | Path) -> DocumentFormat:
    """Detect the document format based on the file extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return DocumentFormat.PDF
    elif ext == ".csv":
        return DocumentFormat.CSV
    elif ext == ".json":
        return DocumentFormat.JSON
    elif ext == ".jsonl":
        return DocumentFormat.JSONL
    elif ext in {".txt", ".text"}:
        return DocumentFormat.TEXT
    else:
        raise ValueError(f"Unsupported document format: {ext}")


class DocumentLoader:
    """Load documents from various formats."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_format = detect_document_format(self.file_path)

    def load(self) -> list[Document]:
        """Load documents based on the specified format."""
        if self.file_format == DocumentFormat.PDF:
            loader = PdfLoader(self.file_path)
        elif self.file_format == DocumentFormat.CSV:
            df = pd.read_csv(self.file_path)
            df["text"] = df.apply(lambda row: str(row.to_dict()), axis=1)
            loader = DataFrameLoader(df)
        elif self.file_format == DocumentFormat.JSON:
            df = pd.read_json(self.file_path)
            df["text"] = df.apply(lambda row: str(row.to_dict()), axis=1)
            loader = DataFrameLoader(df)
        elif self.file_format == DocumentFormat.JSONL:
            df = pd.read_json(self.file_path, lines=True)
            df["text"] = df.apply(lambda row: str(row.to_dict()), axis=1)
            loader = DataFrameLoader(df)
        elif self.file_format == DocumentFormat.TEXT:
            loader = TextLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported document format: {self.file_format}")

        return loader.load()
