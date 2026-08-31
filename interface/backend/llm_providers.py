from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from interface.backend.telemetry import trace_operation



LLMProvider = Literal["openai", "mistral", "google"]
REQUEST_TIMEOUT_SECONDS = 300
LLM_REQUEST_TIMEOUT_ENV = "RAG_LLM_REQUEST_TIMEOUT_SECONDS"
LLM_MAX_RETRIES_ENV = "RAG_LLM_MAX_RETRIES"
MISTRAL_EU_BASE_URL = "https://api.eu.mistral.ai/v1"
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
            ("ZAI GLM 5.2", "zai-glm-5-2"),
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


def configured_request_timeout_seconds() -> float:
    raw_value = os.getenv(LLM_REQUEST_TIMEOUT_ENV, "").strip()
    if not raw_value:
        return float(REQUEST_TIMEOUT_SECONDS)
    try:
        value = float(raw_value)
    except ValueError:
        return float(REQUEST_TIMEOUT_SECONDS)
    return value if value > 0 else float(REQUEST_TIMEOUT_SECONDS)


def configured_max_retries() -> int | None:
    raw_value = os.getenv(LLM_MAX_RETRIES_ENV, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def add_runtime_limits(options: dict[str, Any]) -> dict[str, Any]:
    retries = configured_max_retries()
    if retries is not None:
        options["max_retries"] = retries
    return options


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
        ("mistral-", "ministral-", "codestral-", "pixtral-", "zai-")
    ):
        return "mistral"
    if normalized.startswith("gemini-"):
        return "google"
    raise LLMProviderError(
        "unknown",
        (
            f"Fournisseur impossible à déduire du modèle {model!r}. "
            "Utilise un identifiant gpt-*, mistral-*, zai-* ou gemini-*."
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


def langchain_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Give function-calling providers the name LangChain requires."""
    return {"title": "rag_response", **schema}


def invoke_langchain_model(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, str]],
    invoke: Callable[[], Any],
    *,
    invocation_parameters: dict[str, Any] | None = None,
) -> Any:
    """Create one provider-level Phoenix LLM span, without LangChain internals."""
    with trace_operation(
        model,
        kind="LLM",
        input_value=messages,
        attributes={
            "llm.provider": provider,
            "llm.system": provider,
            "llm.model_name": model,
            "llm.invocation_parameters": invocation_parameters or {"model": model},
        },
    ) as operation:
        add_llm_message_attributes(operation, "llm.input_messages", messages)
        result = invoke()
        operation.set_output(result)
        response = raw_langchain_response(result)
        output_text = langchain_result_text(result)
        add_llm_message_attributes(
            operation,
            "llm.output_messages",
            [{"role": "assistant", "content": output_text}],
        )
        add_llm_usage_attributes(operation, response)
        return result


def add_llm_message_attributes(
    operation: Any,
    attribute_name: str,
    messages: list[dict[str, str]],
) -> None:
    """Write the flattened OpenInference attributes Phoenix renders as message cards."""
    for index, message in enumerate(messages):
        prefix = f"{attribute_name}.{index}.message"
        operation.set_attribute(f"{prefix}.role", message["role"])
        operation.set_attribute(f"{prefix}.content", message["content"])


def raw_langchain_response(result: Any) -> Any:
    if isinstance(result, dict) and result.get("raw") is not None:
        return result["raw"]
    return result


def langchain_result_text(result: Any) -> str:
    if isinstance(result, dict) and result.get("parsed") is not None:
        return json.dumps(result["parsed"], ensure_ascii=False)
    return content_text(getattr(raw_langchain_response(result), "content", "")).strip()


def add_llm_usage_attributes(operation: Any, response: Any) -> None:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
    operation.set_attribute(
        "llm.token_count.prompt",
        usage.get("input_tokens", usage.get("prompt_tokens")),
    )
    operation.set_attribute(
        "llm.token_count.completion",
        usage.get("output_tokens", usage.get("completion_tokens")),
    )
    operation.set_attribute("llm.token_count.total", usage.get("total_tokens"))


def traced_invocation_parameters(options: dict[str, Any]) -> dict[str, Any]:
    """Keep useful request options in Phoenix without exposing credentials."""
    return {
        key: value
        for key, value in options.items()
        if key not in {"api_key", "timeout", "request_timeout"}
    }


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


def call_openai(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int | None,
    store: bool | None,
    response_schema: dict[str, Any] | None,
    reasoning_effort: str | None,
    verbosity: str | None,
    base_url: str | None,
    service_tier_override: str | None,
) -> LLMResponse:
    options: dict[str, Any] = add_runtime_limits(
        {
            "model": model,
            "api_key": api_key,
            "timeout": configured_request_timeout_seconds(),
        }
    )
    service_tier = service_tier_override or configured_openai_service_tier()
    if service_tier is not None:
        options["service_tier"] = service_tier
    if max_output_tokens is not None:
        options["max_completion_tokens"] = max_output_tokens
    if store is not None:
        options["store"] = store
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    if verbosity is not None:
        options["verbosity"] = verbosity
    if base_url is not None:
        options["base_url"] = base_url
    if reasoning_effort is not None or verbosity is not None:
        # These controls are Responses API fields: LangChain serializes them as
        # reasoning={"effort": ...} and text={"verbosity": ...}.
        options["use_responses_api"] = True
    try:
        chat = ChatOpenAI(**options)
        if response_schema is not None:
            structured_chat = chat.with_structured_output(
                langchain_response_schema(response_schema),
                method="json_schema",
                include_raw=True,
            )
            result = invoke_langchain_model(
                "openai",
                model,
                messages,
                lambda: structured_chat.invoke(messages),
                invocation_parameters={
                    **traced_invocation_parameters(options),
                    "response_format": "json_schema",
                },
            )
            parsed = result["parsed"]
            response = result["raw"]
            output_text = json.dumps(parsed, ensure_ascii=False)
        else:
            response = invoke_langchain_model(
                "openai",
                model,
                messages,
                lambda: chat.invoke(messages),
                invocation_parameters=traced_invocation_parameters(options),
            )
            output_text = content_text(response.content).strip()
    except Exception as exc:
        raise LLMProviderError("openai", str(exc)) from exc
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else {"content": output_text}
    return LLMResponse(
        provider="openai",
        model=model,
        output_text=output_text,
        raw_payload=payload,
    )


def call_mistral(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    max_output_tokens: int | None,
    response_schema: dict[str, Any] | None,
    base_url: str | None,
) -> LLMResponse:
    options: dict[str, Any] = add_runtime_limits(
        {
            "model_name": model,
            "api_key": api_key,
            "timeout": configured_request_timeout_seconds(),
        }
    )
    if max_output_tokens is not None:
        options["max_tokens"] = max_output_tokens
    # Le RAG est hébergé pour une inférence Mistral dans l'Union européenne.
    # Un appel explicite (notamment depuis le testeur de modèles) reste libre
    # de choisir un autre endpoint.
    options["base_url"] = base_url or MISTRAL_EU_BASE_URL
    try:
        chat = ChatMistralAI(**options)
        if response_schema is not None:
            structured_chat = chat.with_structured_output(
                langchain_response_schema(response_schema), include_raw=True
            )
            result = invoke_langchain_model(
                "mistral",
                model,
                messages,
                lambda: structured_chat.invoke(messages),
                invocation_parameters={
                    **traced_invocation_parameters(options),
                    "response_format": "json_schema",
                },
            )
            parsed = result["parsed"]
            response = result["raw"]
            output_text = json.dumps(parsed, ensure_ascii=False)
        else:
            response = invoke_langchain_model(
                "mistral",
                model,
                messages,
                lambda: chat.invoke(messages),
                invocation_parameters=traced_invocation_parameters(options),
            )
            output_text = content_text(response.content).strip()
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

    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else {"content": output_text}
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
    response_schema: dict[str, Any] | None,
    thinking_budget: int | None,
) -> LLMResponse:
    options: dict[str, Any] = add_runtime_limits(
        {
            "model": model.removeprefix("models/"),
            "api_key": api_key,
            "request_timeout": configured_request_timeout_seconds(),
        }
    )
    if max_output_tokens is not None:
        options["max_tokens"] = max_output_tokens
    if thinking_budget is not None:
        options["thinking_budget"] = thinking_budget
    try:
        chat = ChatGoogleGenerativeAI(**options)
        if response_schema is not None:
            structured_chat = chat.with_structured_output(
                langchain_response_schema(response_schema),
                method="json_schema",
                include_raw=True,
            )
            result = invoke_langchain_model(
                "google",
                model,
                messages,
                lambda: structured_chat.invoke(messages),
                invocation_parameters={
                    **traced_invocation_parameters(options),
                    "response_format": "json_schema",
                },
            )
            response = result["raw"]
            output_text = json.dumps(result["parsed"], ensure_ascii=False)
        else:
            response = invoke_langchain_model(
                "google",
                model,
                messages,
                lambda: chat.invoke(messages),
                invocation_parameters=traced_invocation_parameters(options),
            )
            output_text = content_text(response.content).strip()
    except Exception as exc:
        raise LLMProviderError("google", str(exc)) from exc
    payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else {"content": output_text}
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
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    thinking_budget: int | None = None,
    mistral_base_url: str | None = None,
    openai_base_url: str | None = None,
    openai_service_tier: str | None = None,
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
    if selected_provider == "openai":
        return call_openai(
            model,
            messages,
            api_key,
            max_output_tokens,
            store,
            response_schema,
            reasoning_effort,
            verbosity,
            openai_base_url,
            openai_service_tier,
        )
    if selected_provider == "mistral":
        return call_mistral(
            model,
            messages,
            api_key,
            max_output_tokens,
            response_schema,
            mistral_base_url,
        )
    return call_google(
        model,
        messages,
        api_key,
        max_output_tokens,
        response_schema,
        thinking_budget,
    )


class RoutedResponses:
    def create(self, **kwargs: Any) -> LLMResponse:
        return create_llm_response(**kwargs)


class RoutedLLMClient:
    def __init__(self) -> None:
        self.responses = RoutedResponses()


def get_llm_client() -> RoutedLLMClient | None:
    # Le client exposé au RAG ne doit être disponible qu'avec la clé du seul
    # fournisseur d'inférence autorisé.
    if not provider_api_key("openai"):
        return None
    return RoutedLLMClient()
