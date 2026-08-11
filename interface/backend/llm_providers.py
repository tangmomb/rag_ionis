from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import quote

import requests
from mistralai.client import Mistral
from openai import OpenAI

from interface.backend.telemetry import TraceOperation, trace_operation


LLMProvider = Literal["openai", "mistral", "google"]
REQUEST_TIMEOUT_SECONDS = 300
OPENAI_SERVICE_TIER_ENV = "OPENAI_SERVICE_TIER"
OPENAI_SERVICE_TIERS = frozenset(
    {"auto", "default", "flex", "scale", "priority", "fast"}
)

LLM_MODEL_CATALOG: dict[LLMProvider, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "key_names": ("OPENAI_API_KEY",),
        "default_model_env": "OPENAI_LLM_TEST_MODEL",
        "default_model": "gpt-5.6-sol",
        "models": (
            ("Sol", "gpt-5.6-sol"),
            ("Terra", "gpt-5.6-terra"),
            ("Luna", "gpt-5.6-luna"),
        ),
    },
    "mistral": {
        "label": "Mistral",
        "key_names": ("MISTRAL_API_KEY",),
        "default_model_env": "MISTRAL_LLM_TEST_MODEL",
        "default_model": "mistral-large-latest",
        "models": (
            ("Medium", "mistral-medium-latest"),
            ("Small", "mistral-small-latest"),
            ("Large", "mistral-large-latest"),
        ),
    },
    "google": {
        "label": "Google",
        "key_names": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "default_model_env": "GOOGLE_LLM_TEST_MODEL",
        "default_model": "gemini-3.6-flash",
        "models": (
            ("Gemini 3.1 Flash-Lite", "gemini-3.1-flash-lite"),
            ("Gemini 3.6 Flash", "gemini-3.6-flash"),
            ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite"),
        ),
    },
}


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.payload = payload


@dataclass(frozen=True)
class LLMResponse:
    provider: LLMProvider
    model: str
    output_text: str
    raw_payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.raw_payload, ensure_ascii=False)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return self.raw_payload

    def model_dump_json(self) -> str:
        return self.to_json()


class ResponsesClientProtocol(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class LLMClientProtocol(Protocol):
    responses: ResponsesClientProtocol


def provider_api_key(provider: LLMProvider) -> str | None:
    for name in LLM_MODEL_CATALOG[provider]["key_names"]:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def configured_openai_service_tier() -> str | None:
    service_tier = os.getenv(OPENAI_SERVICE_TIER_ENV, "").strip().lower()
    if not service_tier:
        return None
    if service_tier not in OPENAI_SERVICE_TIERS:
        choices = ", ".join(sorted(OPENAI_SERVICE_TIERS))
        raise LLMProviderError(
            "openai",
            (
                f"{OPENAI_SERVICE_TIER_ENV} invalide : {service_tier!r}. "
                f"Valeurs acceptees : {choices}."
            ),
        )
    return service_tier


def configured_llm_provider_exists() -> bool:
    return any(provider_api_key(provider) for provider in LLM_MODEL_CATALOG)


def provider_for_model(model: str) -> LLMProvider:
    normalized = str(model or "").strip().casefold()
    if normalized.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if normalized.startswith(
        ("mistral-", "ministral-", "codestral-", "pixtral-")
    ):
        return "mistral"
    if normalized.startswith("gemini-"):
        return "google"
    raise LLMProviderError(
        "unknown",
        (
            f"Fournisseur impossible à déduire du modèle {model!r}. "
            "Utilise un identifiant gpt-*, mistral-* ou gemini-*."
        ),
    )


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(
        content,
        (str, bytes, bytearray),
    ):
        return str(content or "")

    pieces = []
    for part in content:
        if isinstance(part, str):
            pieces.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text", part.get("content"))
        if isinstance(text, str):
            pieces.append(text)
    return "\n".join(pieces)


def normalize_messages(input_value: Any) -> list[dict[str, str]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, Sequence):
        return [{"role": "user", "content": str(input_value)}]

    messages = []
    for item in input_value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        messages.append(
            {
                "role": role,
                "content": content_text(item.get("content")),
            }
        )
    return messages or [{"role": "user", "content": ""}]


def extract_openai_text(payload: dict[str, Any]) -> str:
    top_level = payload.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    pieces = []
    for output in payload.get("output", []) or []:
        if not isinstance(output, dict) or output.get("type") != "message":
            continue
        for content in output.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str) and text:
                    pieces.append(text)
    return "\n".join(pieces).strip()


def extract_mistral_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", []) or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    pieces = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text", part.get("content"))
        if isinstance(text, str) and text:
            pieces.append(text)
    return "\n".join(pieces).strip()


def extract_google_text(payload: dict[str, Any]) -> str:
    pieces = []
    for candidate in payload.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
    return "\n".join(pieces).strip()


def json_response(
    response: requests.Response,
    provider: LLMProvider,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMProviderError(
            provider,
            "Le fournisseur a renvoyé une réponse non JSON.",
            status_code=response.status_code,
            payload={"response_preview": response.text[:2_000]},
        ) from exc
    if not response.ok:
        raise LLMProviderError(
            provider,
            "Le fournisseur a refusé la requête.",
            status_code=response.status_code,
            payload=payload,
        )
    if not isinstance(payload, dict):
        raise LLMProviderError(
            provider,
            "Le fournisseur a renvoyé un JSON inattendu.",
            status_code=response.status_code,
            payload=payload,
        )
    return payload


def add_llm_message_attributes(
    operation: TraceOperation,
    attribute_name: str,
    messages: list[dict[str, str]],
) -> None:
    for index, message in enumerate(messages):
        prefix = f"{attribute_name}.{index}.message"
        operation.set_attribute(f"{prefix}.role", message["role"])
        operation.set_attribute(f"{prefix}.content", message["content"])


def add_llm_usage_attributes(
    operation: TraceOperation,
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
) -> None:
    operation.set_attribute("llm.token_count.prompt", prompt_tokens)
    operation.set_attribute("llm.token_count.completion", completion_tokens)
    operation.set_attribute("llm.token_count.total", total_tokens)


def call_openai(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int | None,
    store: bool | None,
) -> LLMResponse:
    request: dict[str, Any] = {
        "model": model,
        "input": messages,
    }
    service_tier = configured_openai_service_tier()
    if service_tier is not None:
        request["service_tier"] = service_tier
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens
    if store is not None:
        request["store"] = store
    try:
        response = OpenAI(api_key=api_key).responses.create(**request)
    except Exception as exc:
        raise LLMProviderError("openai", str(exc)) from exc
    payload = response.model_dump(mode="json")
    return LLMResponse(
        provider="openai",
        model=model,
        output_text=(response.output_text or "").strip(),
        raw_payload=payload,
    )


def call_mistral(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int | None,
    response_schema: dict[str, Any] | None,
) -> LLMResponse:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if max_output_tokens is not None:
        request["max_tokens"] = max_output_tokens
    if response_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "schema": response_schema,
                "strict": True,
            },
        }
    try:
        response = Mistral(
            api_key=api_key,
            timeout_ms=REQUEST_TIMEOUT_SECONDS * 1_000,
        ).chat.complete(**request)
    except Exception as exc:
        raw_response = getattr(exc, "raw_response", None)
        status_code = getattr(raw_response, "status_code", None)
        payload = getattr(exc, "body", None)
        raise LLMProviderError(
            "mistral",
            str(exc),
            status_code=status_code,
            payload=payload,
        ) from exc

    payload = response.model_dump(mode="json")
    output_text = extract_mistral_text(payload)
    return LLMResponse(
        provider="mistral",
        model=model,
        output_text=output_text,
        raw_payload=payload,
    )


def call_google(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int | None,
) -> LLMResponse:
    system_parts = [
        {"text": message["content"]}
        for message in messages
        if message["role"] == "system" and message["content"]
    ]
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message["role"] != "system"
    ]
    request: dict[str, Any] = {"contents": contents}
    if system_parts:
        request["systemInstruction"] = {"parts": system_parts}
    if max_output_tokens is not None:
        request["generationConfig"] = {
            "maxOutputTokens": max_output_tokens,
        }
    model_id = model.removeprefix("models/")
    invocation_parameters = {
        key: value
        for key, value in request.items()
        if key not in {"contents", "systemInstruction"}
    }
    with trace_operation(
        "GoogleGenerateContent",
        kind="LLM",
        input_value=messages,
        attributes={
            "llm.model_name": model,
            "llm.provider": "google",
            "llm.system": "google",
            "llm.invocation_parameters": invocation_parameters,
        },
    ) as operation:
        add_llm_message_attributes(operation, "llm.input_messages", messages)
        response = requests.post(
            (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{quote(model_id, safe='-._')}:generateContent"
            ),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        payload = json_response(response, "google")
        output_text = extract_google_text(payload)
        operation.set_output(payload)
        add_llm_message_attributes(
            operation,
            "llm.output_messages",
            [{"role": "assistant", "content": output_text}],
        )
        usage = payload.get("usageMetadata", {})
        if isinstance(usage, dict):
            add_llm_usage_attributes(
                operation,
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount"),
            )
        return LLMResponse(
            provider="google",
            model=model,
            output_text=output_text,
            raw_payload=payload,
        )


def create_llm_response(
    *,
    model: str,
    input: Any,
    provider: LLMProvider | None = None,
    max_output_tokens: int | None = None,
    store: bool | None = None,
    response_schema: dict[str, Any] | None = None,
) -> LLMResponse:
    selected_provider = provider or provider_for_model(model)
    api_key = provider_api_key(selected_provider)
    if api_key is None:
        key_names = " ou ".join(
            LLM_MODEL_CATALOG[selected_provider]["key_names"]
        )
        raise LLMProviderError(
            selected_provider,
            f"Clé absente : renseigne {key_names} dans .env.",
        )
    messages = normalize_messages(input)
    try:
        if selected_provider == "openai":
            return call_openai(
                model,
                messages,
                api_key,
                max_output_tokens,
                store,
            )
        if selected_provider == "mistral":
            return call_mistral(
                model,
                messages,
                api_key,
                max_output_tokens,
                response_schema,
            )
        return call_google(
            model,
            messages,
            api_key,
            max_output_tokens,
        )
    except requests.RequestException as exc:
        raise LLMProviderError(
            selected_provider,
            f"Erreur réseau : {exc}",
        ) from exc


class RoutedResponses:
    def create(self, **kwargs: Any) -> LLMResponse:
        return create_llm_response(**kwargs)


class RoutedLLMClient:
    def __init__(self) -> None:
        self.responses = RoutedResponses()


def get_llm_client() -> RoutedLLMClient | None:
    if not configured_llm_provider_exists():
        return None
    return RoutedLLMClient()
