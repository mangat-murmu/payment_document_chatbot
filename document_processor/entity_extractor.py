from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import config
from document_processor.document_loader import DocumentLoader

DEFAULT_MODEL_PATH = config.ENTITY_EXTRACTOR_MODEL_PATH


class EntityExtractor:
    """Payment entity extractor backed by the local fine-tuned NER model."""

    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_name = str(model_name)
        if not Path(self.model_name).exists():
            raise FileNotFoundError(
                f"entity extraction model not found: {self.model_name}"
            )
        self.pipeline = self._load_pipeline(self.model_name)

    @staticmethod
    def _load_pipeline(model_name: str) -> Any:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "Install transformers to use the fine-tuned entity extraction model"
            ) from error

        return pipeline(
            "token-classification",
            model=str(model_name),
            tokenizer=str(model_name),
            aggregation_strategy="simple",
        )

    def extract(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("text cannot be empty")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entity in self.pipeline(text):
            entity_type = (
                str(entity.get("entity_group") or entity["entity"])
                .removeprefix("B-")
                .removeprefix("I-")
            )
            grouped.setdefault(entity_type, []).append(
                {
                    "text": entity["word"],
                    "score": round(float(entity["score"]), 4),
                    "start": entity["start"],
                    "end": entity["end"],
                }
            )
        return grouped


def _read_text(arguments: argparse.Namespace) -> str:
    if arguments.text:
        return arguments.text
    documents = DocumentLoader(arguments.input_file).load()
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract entities from a payment document"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Document text to process")
    source.add_argument("--input-file", help="UTF-8 text, CSV, or JSON file to process")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Local fine-tuned NER directory",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            EntityExtractor(arguments.model).extract(_read_text(arguments)), indent=2
        )
    )


if __name__ == "__main__":
    main()
