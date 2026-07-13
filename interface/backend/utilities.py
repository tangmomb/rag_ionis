from __future__ import annotations

import json
import os
import unicodedata
from importlib import import_module
from typing import Any

from openai import OpenAI

from interface.backend.config import (
    DEFAULT_COHERE_RERANK_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_RERANK_MODEL,
)


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_cohere_client() -> Any | None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    try:
        cohere = import_module("cohere")
    except ImportError:
        return None
    return cohere.ClientV2(api_key)


def normalize_model_name(value: str, default: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return default
    lower = cleaned.lower()
    if lower in {"meme modele que la base", "même modèle que la base"}:
        return DEFAULT_EMBEDDING_MODEL
    if lower in {"cohere rerank", "cohere-rerank"}:
        return "cohere-rerank"
    if lower in {"5.6 luna", "gpt 5.6 luna", "gpt-5.6 luna"}:
        return DEFAULT_GENERATION_MODEL
    # Compatibilite avec d'anciens clients : le modele de generation officiel
    # reste toujours celui defini par DEFAULT_GENERATION_MODEL.
    if lower in {"gpt5.4nano", "gpt-5.4-nano"}:
        return DEFAULT_GENERATION_MODEL
    return cleaned


def resolve_cohere_rerank_model(value: str) -> str:
    normalized = normalize_model_name(value, DEFAULT_RERANK_MODEL)
    if normalized == "cohere-rerank":
        return DEFAULT_COHERE_RERANK_MODEL
    return normalized


def safe_json_loads(value: str) -> dict[str, Any]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(value[start : end + 1])
        raise


def format_sql_for_trace(sql: str | None) -> str | None:
    if not sql:
        return None
    return " ".join(sql.split())


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def serialize_openai_response(response: Any) -> str:
    """Serialize the complete SDK response while keeping it valid JSON."""
    to_json = getattr(response, "to_json", None)
    if callable(to_json):
        return str(to_json())

    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        return str(model_dump_json())

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(mode="json"), ensure_ascii=False)

    return json.dumps(response, default=str, ensure_ascii=False)
