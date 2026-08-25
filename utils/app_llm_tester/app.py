from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from pipeline.support.environment import load_project_env

from interface.backend.llm_providers import (
    LLM_MODEL_CATALOG,
    LLMProviderError,
    create_llm_response,
    provider_api_key,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

load_project_env(PROJECT_DIR)

app = FastAPI(title="RAG IONIS — LLM tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    provider: Literal["mistral"]
    model: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=200_000)
    max_output_tokens: int = Field(default=2048, ge=1, le=32_768)

    @field_validator("model", "message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ce champ ne peut pas être vide.")
        return cleaned


PROVIDERS = {
    "mistral": LLM_MODEL_CATALOG["mistral"],
}


def provider_config(provider: str) -> dict[str, Any]:
    definition = PROVIDERS[provider]
    configured_default = (
        os.getenv(definition["default_model_env"], "").strip()
        or definition["default_model"]
    )
    catalog_models = [
        model_id
        for _model_label, model_id in definition["models"]
    ]
    models = list(dict.fromkeys((configured_default, *catalog_models)))
    return {
        "id": provider,
        "label": definition["label"],
        "configured": provider_api_key(provider) is not None,
        "keyNames": list(definition["key_names"]),
        "defaultModel": configured_default,
        "models": models,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "providers": [
            provider_config(provider)
            for provider in PROVIDERS
        ]
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    if provider_api_key(request.provider) is None:
        key_names = " ou ".join(PROVIDERS[request.provider]["key_names"])
        raise HTTPException(
            status_code=400,
            detail={
                "provider": request.provider,
                "message": f"Clé absente : renseigne {key_names} dans .env.",
            },
        )

    started_at = time.perf_counter()
    try:
        response = create_llm_response(
            provider=request.provider,
            model=request.model,
            input=request.message,
            max_output_tokens=request.max_output_tokens,
            store=False,
        )
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "provider": request.provider,
                "message": str(exc),
                "statusCode": exc.status_code,
                "response": exc.payload,
            },
        ) from exc

    return {
        "provider": request.provider,
        "model": request.model,
        "durationMs": round((time.perf_counter() - started_at) * 1000),
        "text": response.output_text,
        "response": response.raw_payload,
    }
