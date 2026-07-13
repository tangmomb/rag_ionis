from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator


_LOCK = threading.Lock()
_CONFIGURED = False
_ENABLED = False
_TRACER: Any = None
_TRACER_PROVIDER: Any = None
_INITIALIZATION_ERROR: str | None = None

# La télémétrie reste optionnelle : aucune erreur Phoenix ne doit bloquer le RAG.


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_telemetry() -> None:
    """Configure Phoenix tracing once without making it an API dependency."""
    global _CONFIGURED, _ENABLED, _TRACER, _TRACER_PROVIDER, _INITIALIZATION_ERROR

    if _CONFIGURED:
        return

    with _LOCK:
        if _CONFIGURED:
            return
        _CONFIGURED = True

        if not _env_flag("PHOENIX_ENABLED", True):
            return

        endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006/v1/traces",
        ).strip()
        project_name = os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis").strip() or "rag-ionis"

        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            from phoenix.otel import register

            _TRACER_PROVIDER = register(
                endpoint=endpoint,
                project_name=project_name,
                protocol="http/protobuf",
                batch=True,
                verbose=False,
            )
            _TRACER = _TRACER_PROVIDER.get_tracer("rag_ionis.interface")
            OpenAIInstrumentor().instrument(tracer_provider=_TRACER_PROVIDER)
            _ENABLED = True
            print(
                f"[telemetry] Phoenix actif: project={project_name} endpoint={endpoint}",
                flush=True,
            )
        except Exception as exc:  # L'observabilite ne doit jamais bloquer le RAG.
            _INITIALIZATION_ERROR = str(exc)
            print(f"[telemetry] Phoenix desactive: {exc}", flush=True)


def telemetry_status() -> dict[str, Any]:
    return {
        "configured": _CONFIGURED,
        "enabled": _ENABLED,
        "project": os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis"),
        "collector_endpoint": os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006/v1/traces",
        ),
        "initialization_error": _INITIALIZATION_ERROR,
    }


def shutdown_telemetry() -> None:
    provider = _TRACER_PROVIDER
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis=5_000)
        provider.shutdown()
    except Exception:
        pass


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


class TraceOperation:
    def __init__(self, span: Any = None) -> None:
        self._span = span

    def set_attribute(self, name: str, value: Any) -> None:
        if self._span is None or value is None:
            return
        if isinstance(value, (str, bool, int, float)):
            serialized = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        ):
            serialized = list(value)
        else:
            serialized = _json_value(value)
        self._span.set_attribute(name, serialized)

    def set_output(self, value: Any) -> None:
        if self._span is None:
            return
        try:
            from phoenix.otel import OpenInferenceMimeTypeValues, SpanAttributes

            self._span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json_value(value))
            self._span.set_attribute(
                SpanAttributes.OUTPUT_MIME_TYPE,
                OpenInferenceMimeTypeValues.JSON.value,
            )
        except Exception:
            pass

    def set_output_text(self, value: str | None) -> None:
        if self._span is None or value is None:
            return
        try:
            from phoenix.otel import OpenInferenceMimeTypeValues, SpanAttributes

            self._span.set_attribute(SpanAttributes.OUTPUT_VALUE, value)
            self._span.set_attribute(
                SpanAttributes.OUTPUT_MIME_TYPE,
                OpenInferenceMimeTypeValues.TEXT.value,
            )
        except Exception:
            pass

    def set_documents(self, attribute_name: str, documents: list[dict[str, Any]]) -> None:
        """Expose documents using the flattened OpenInference document convention."""
        if self._span is None:
            return
        for index, document in enumerate(documents):
            prefix = f"{attribute_name}.{index}.document"
            if document.get("chunk_id") is not None:
                self.set_attribute(f"{prefix}.id", str(document["chunk_id"]))
            if document.get("text") is not None:
                self.set_attribute(f"{prefix}.content", str(document["text"]))
            for score_name in (
                "cohere_relevance_score",
                "rrf_score",
                "bm25_score",
                "vector_score",
            ):
                score = document.get(score_name)
                if score is not None:
                    self.set_attribute(f"{prefix}.score", float(score))
                    break

    def set_session_id(self, conversation_id: int | str | None) -> None:
        if self._span is None or conversation_id is None:
            return
        try:
            from phoenix.otel import SpanAttributes

            self._span.set_attribute(SpanAttributes.SESSION_ID, str(conversation_id))
        except Exception:
            pass


@contextmanager
def trace_operation(
    name: str,
    *,
    kind: str = "CHAIN",
    input_value: Any = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[TraceOperation]:
    if not _ENABLED or _TRACER is None:
        yield TraceOperation()
        return

    from phoenix.otel import (
        OpenInferenceMimeTypeValues,
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    resolved_kind = getattr(OpenInferenceSpanKindValues, kind.upper(), OpenInferenceSpanKindValues.CHAIN)
    with _TRACER.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, resolved_kind.value)
        if input_value is not None:
            span.set_attribute(SpanAttributes.INPUT_VALUE, _json_value(input_value))
            span.set_attribute(
                SpanAttributes.INPUT_MIME_TYPE,
                OpenInferenceMimeTypeValues.JSON.value,
            )
        operation = TraceOperation(span)
        for attribute_name, attribute_value in (attributes or {}).items():
            operation.set_attribute(attribute_name, attribute_value)
        yield operation


def current_trace_id() -> str | None:
    if not _ENABLED:
        return None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return f"{context.trace_id:032x}"
    except Exception:
        return None
