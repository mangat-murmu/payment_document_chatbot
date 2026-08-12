from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from document_processor.document_loader import DocumentLoader


DEFAULT_MODEL_PATH = config.DOCUMENT_CLASSIFIER_MODEL_PATH


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float


class DocumentClassifier:
    """Payment document classifier backed by the local fine-tuned model."""

    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_name = str(model_name)
        if not Path(self.model_name).exists():
            raise FileNotFoundError(f"classifier model not found: {self.model_name}")
        self.pipeline = self._load_pipeline(self.model_name)

    @staticmethod
    def _load_pipeline(model_name: str) -> Any:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("Install transformers to use the fine-tuned classifier model") from error

        return pipeline(
            "text-classification",
            model=str(model_name),
            tokenizer=str(model_name),
            truncation=True,
        )

    def classify(self, text: str) -> Classification:
        if not text.strip():
            raise ValueError("text cannot be empty")
        result = self.pipeline(text, truncation=True)[0]
        return Classification(str(result["label"]), round(float(result["score"]), 4))


def _read_text(arguments: argparse.Namespace) -> str:
    if arguments.text:
        return arguments.text
    documents = DocumentLoader(arguments.input_file).load()
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a payment document")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Document text to classify")
    source.add_argument("--input-file", help="UTF-8 text, CSV, or JSON file to classify")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Local fine-tuned classifier directory")
    arguments = parser.parse_args()
    result = DocumentClassifier(arguments.model).classify(_read_text(arguments))
    print(json.dumps({"document_type": result.label, "confidence": result.confidence}, indent=2))


if __name__ == "__main__":
    main()
