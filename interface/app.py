from __future__ import annotations

import os
import json
import re
from datetime import datetime
from typing import Any

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interface import rate_limit
from interface.backend.request_context import current_request_details, current_request_id
from interface.monitoring.history import router as monitoring_history_router
from interface.request_logging import configure_json_logging, should_log
from interface.backend.api import execute_rag, router
from interface.backend.config import INTERFACE_DIR
from interface.backend.database import ensure_chat_schema
from interface.backend.orchestration import orchestrate_request
from interface.backend.schemas import ChunkSource, ExecutionPlan, PlannerPlan, RagRequest, RagResponse
from interface.backend.telemetry import (
    configure_telemetry,
    shutdown_telemetry,
    telemetry_status,
)


_STARTED_AT: datetime | None = None
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)

app = FastAPI(title="RAG IONIS API")
request_logger = configure_json_logging()


@app.middleware("http")
async def protect_and_log(request: Request, call_next):
    """Rate-limit paid routes and write one JSON log line per API call."""
    request_id = request.headers.get("x-request-id", "")[:64] or uuid4().hex
    # Read and filled by run_rag (trace_id, conversation, message) for this log line.
    current_request_id.set(request_id)
    details: dict[str, Any] = {}
    current_request_details.set(details)
    started_at = perf_counter()
    path = request.url.path
    client_ip = request.client.host if request.client else None
    rejection = None
    if (
        request.method == "POST"
        and rate_limit.PAID_PATH_PATTERN.match(path)
        and rate_limit.enabled()
    ):
        rejection = await run_in_threadpool(rate_limit.check_request, client_ip)
    if rejection is not None:
        response = JSONResponse(
            {"detail": rejection.detail},
            status_code=rejection.status_code,
            headers={"Retry-After": str(rejection.retry_after_seconds)},
        )
    else:
        try:
            response = await call_next(request)
        except Exception:
            request_logger.exception(
                "http_request_failed",
                extra={"method": request.method, "path": path, "request_id": request_id},
            )
            raise
    response.headers["X-Request-ID"] = request_id
    if not should_log(path):
        return response

    def write_log_line() -> None:
        request_logger.info(
            "http_request",
            extra={
                **details,
                "method": request.method,
                "path": path,
                "status": response.status_code,
                # Measured once the whole body is sent: for a streamed answer,
                # the real duration, not the time until the first byte.
                "duration_ms": round((perf_counter() - started_at) * 1_000),
                "client": rate_limit.hashed_client(client_ip)[:12],
                "request_id": request_id,
                "rate_limited": rejection is not None,
            },
        )

    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        write_log_line()
        return response

    async def body_then_log():
        try:
            async for chunk in body_iterator:
                yield chunk
        finally:
            write_log_line()

    response.body_iterator = body_then_log()
    return response


public_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("PUBLIC_ORIGINS", "").split(",")
    if origin.strip()
]
if public_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=public_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(router, prefix="/api")
app.include_router(monitoring_history_router)
app.mount(
    "/pipeline_reponse",
    StaticFiles(directory=INTERFACE_DIR / "pipeline_reponse", html=True),
    name="pipeline_reponse",
)
app.mount(
    "/pipeline_preparation_videos",
    StaticFiles(directory=INTERFACE_DIR / "pipeline_preparation_videos", html=True),
    name="pipeline_preparation_videos",
)
app.mount(
    "/assets",
    StaticFiles(directory=INTERFACE_DIR / "assets"),
    name="assets",
)


@app.on_event("startup")
def startup() -> None:
    global _STARTED_AT
    _STARTED_AT = datetime.now().astimezone()
    print(f"[backend] started_at={_STARTED_AT.isoformat(timespec='seconds')}", flush=True)
    configure_telemetry()
    ensure_chat_schema()


@app.on_event("shutdown")
def shutdown() -> None:
    shutdown_telemetry()


@app.get("/health")
def health() -> dict[str, str]:
    ensure_chat_schema()
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, Any]:
    return {
        "started_at": _STARTED_AT.isoformat(timespec="seconds") if _STARTED_AT else None,
        "telemetry": telemetry_status(),
    }


@app.get("/api/pipeline-reponse/traces/{trace_id}")
def read_pipeline_response_trace(trace_id: str) -> dict[str, Any]:
    """Expose les sorties déjà enregistrées par Phoenix pour le visualiseur."""
    if not _TRACE_ID_PATTERN.fullmatch(trace_id):
        return {"trace_id": trace_id, "spans": []}

    collector_endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
    ).rstrip("/")
    base_url = collector_endpoint.removesuffix("/v1/traces")
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis")

    try:
        from phoenix.client import Client

        raw_spans = Client(base_url=base_url).spans.get_spans(
            project_identifier=project_name,
            trace_ids=[trace_id],
            limit=1_000,
            timeout=10,
        )
    except Exception as exc:
        return {"trace_id": trace_id, "spans": [], "error": str(exc)}

    spans: list[dict[str, Any]] = []
    for span in raw_spans:
        attributes = span.get("attributes") or {}
        output = attributes.get("output.value")
        if output is None:
            continue
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass
        spans.append(
            {
                "name": span.get("name"),
                "output": output,
                "start_time": span.get("start_time"),
            }
        )
    return {"trace_id": trace_id, "spans": spans}


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(
        INTERFACE_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/faq")
def read_faq() -> FileResponse:
    return FileResponse(
        INTERFACE_DIR / "faq.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/styles.css")
def read_styles() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "styles.css", media_type="text/css")


@app.get("/favicon.svg")
def read_favicon() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "favicon.svg", media_type="image/svg+xml")


__all__ = [
    "ChunkSource",
    "ExecutionPlan",
    "PlannerPlan",
    "RagRequest",
    "RagResponse",
    "app",
    "execute_rag",
    "orchestrate_request",
]
