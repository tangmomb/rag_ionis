from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from interface.backend.telemetry import shutdown_telemetry
from utils.replay_phoenix_conversation import run_replay


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="RAG IONIS — Phoenix Replay")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ReplayRequest(BaseModel):
    trace_id: str = Field(min_length=32, max_length=32)
    mode: Literal["exact-context"] = "exact-context"

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if len(cleaned) != 32 or any(
            character not in "0123456789abcdef" for character in cleaned
        ):
            raise ValueError("L'identifiant doit contenir exactement 32 caractères hexadécimaux.")
        return cleaned


def replay_arguments(request: ReplayRequest, *, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        trace_id=request.trace_id,
        mode=request.mode,
        phoenix_base_url=os.getenv("PHOENIX_BASE_URL", "http://localhost:6006"),
        project=os.getenv("PHOENIX_PROJECT_NAME", "rag-ionis"),
        timeout=10,
        dry_run=dry_run,
    )


def execute(request: ReplayRequest, *, dry_run: bool) -> dict[str, Any]:
    try:
        return run_replay(
            replay_arguments(request, dry_run=dry_run),
            shutdown_after=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/inspect")
def inspect_context(request: ReplayRequest) -> dict[str, Any]:
    return execute(request, dry_run=True)


@app.post("/api/replay")
def replay_context(request: ReplayRequest) -> dict[str, Any]:
    return execute(request, dry_run=False)


@app.on_event("shutdown")
def shutdown() -> None:
    shutdown_telemetry()
