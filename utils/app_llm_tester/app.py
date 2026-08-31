from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
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

MISTRAL_REGIONS = {
    "global": {"label": "Global", "base_url": None},
    "eu": {"label": "Union européenne", "base_url": "https://api.eu.mistral.ai/v1"},
    "us": {"label": "États-Unis", "base_url": "https://api.us.mistral.ai/v1"},
}
OPENAI_REGIONS = {
    "global": {"label": "Global", "base_url": None},
    "eu": {"label": "Union européenne", "base_url": "https://eu.api.openai.com/v1"},
    "us": {"label": "États-Unis", "base_url": "https://us.api.openai.com/v1"},
}

load_project_env(PROJECT_DIR)

app = FastAPI(title="RAG IONIS — LLM tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    provider: Literal["openai", "mistral", "google"]
    model: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=200_000)
    system_message: str | None = Field(default=None, max_length=200_000)
    max_output_tokens: int = Field(default=2048, ge=1, le=32_768)
    reasoning_effort: str | None = Field(default=None, max_length=10)
    verbosity: str | None = Field(default=None, max_length=10)
    thinking_budget: int | None = Field(default=None, ge=-1, le=32_768)
    mistral_region: Literal["global", "eu", "us"] = "global"
    openai_region: Literal["global", "eu", "us"] = "global"
    openai_service_tier: Literal["default", "fast"] | None = None

    @field_validator("model", "message")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ce champ ne peut pas être vide.")
        return cleaned

    @field_validator("system_message", mode="before")
    @classmethod
    def normalize_system_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("reasoning_effort", "verbosity", mode="before")
    @classmethod
    def normalize_optional_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        return cleaned or None

    @model_validator(mode="after")
    def validate_generation_options(self) -> "GenerateRequest":
        openai_reasoning = {"none", "minimal", "low", "medium", "high", "xhigh"}
        verbosity_levels = {"low", "medium", "high"}
        if self.provider == "openai":
            if self.reasoning_effort not in {None, *openai_reasoning}:
                raise ValueError("Niveau de reasoning OpenAI invalide.")
            if self.verbosity not in {None, *verbosity_levels}:
                raise ValueError("Niveau de verbosity OpenAI invalide.")
            if self.thinking_budget is not None:
                raise ValueError("Le budget de réflexion est réservé à Gemini.")
        elif self.provider == "google":
            if self.reasoning_effort is not None:
                raise ValueError("Gemini utilise un budget de réflexion, pas un niveau de reasoning.")
            if self.verbosity is not None:
                raise ValueError("Gemini ne propose pas de paramètre de verbosity dédié.")
            if self.thinking_budget is not None and not self.model.startswith("gemini-2.5-"):
                raise ValueError(
                    "Le budget de réflexion est pris en charge uniquement par les modèles Gemini 2.5 dans cette version de l'intégration."
                )
        else:
            if (
            self.reasoning_effort is not None
            or self.verbosity is not None
            or self.thinking_budget is not None
            ):
                raise ValueError("Mistral ne propose pas ces paramètres dans cet outil.")
        if self.provider != "mistral" and self.mistral_region != "global":
            raise ValueError("La région d'inférence est réservée à Mistral.")
        if self.provider != "openai" and self.openai_region != "global":
            raise ValueError("La région d'inférence est réservée à OpenAI.")
        if self.provider != "openai" and self.openai_service_tier is not None:
            raise ValueError("Le mode de latence est réservé à OpenAI.")
        return self


PROVIDERS = {
    "openai": LLM_MODEL_CATALOG["openai"],
    "mistral": LLM_MODEL_CATALOG["mistral"],
    "google": LLM_MODEL_CATALOG["google"],
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
        "label": "Gemini" if provider == "google" else definition["label"],
        "configured": provider_api_key(provider) is not None,
        "keyNames": list(definition["key_names"]),
        "defaultModel": configured_default,
        "models": models,
        "generationControls": generation_controls(provider),
        "regions": [
            {"id": region, "label": config["label"]}
            for region, config in (
                MISTRAL_REGIONS.items()
                if provider == "mistral"
                else OPENAI_REGIONS.items()
                if provider == "openai"
                else []
            )
        ],
    }


def generation_controls(provider: str) -> dict[str, Any]:
    if provider == "openai":
        return {
            "reasoning": ["none", "minimal", "low", "medium", "high", "xhigh"],
            "verbosity": ["low", "medium", "high"],
            "thinkingBudget": False,
            "serviceTiers": [
                {"id": "default", "label": "Standard"},
                {"id": "fast", "label": "Fast"},
            ],
        }
    if provider == "google":
        return {
            "reasoning": [],
            "verbosity": [],
            "thinkingBudgetModelPrefix": "gemini-2.5-",
        }
    return {"reasoning": [], "verbosity": []}


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
            input=(
                [
                    *(
                        [{"role": "system", "content": request.system_message}]
                        if request.system_message
                        else []
                    ),
                    {"role": "user", "content": request.message},
                ]
            ),
            max_output_tokens=request.max_output_tokens,
            store=False,
            reasoning_effort=request.reasoning_effort,
            verbosity=request.verbosity,
            thinking_budget=request.thinking_budget,
            mistral_base_url=MISTRAL_REGIONS[request.mistral_region]["base_url"],
            openai_base_url=OPENAI_REGIONS[request.openai_region]["base_url"],
            openai_service_tier=request.openai_service_tier,
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
