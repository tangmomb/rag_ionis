from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

app = FastAPI(title="RAG IONIS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


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


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/styles.css")
def read_styles() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "styles.css", media_type="text/css")


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
