from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    load_dotenv(ROOT / ".env")


load_environment()
