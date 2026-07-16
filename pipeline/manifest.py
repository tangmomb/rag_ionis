from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .context import LONG_VIDEO_THRESHOLD_SECONDS, VideoContext
from .options import PipelineOptions
from .planner import PlannedTask


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_manifest(
    context: VideoContext,
    options: PipelineOptions,
    tasks: Iterable[PlannedTask],
    *,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_list = list(tasks)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "video": {
            "id": context.video_id,
            **context.media,
        },
        "questions": {
            "longer_than_10_minutes": {
                "answer": context.is_long_video,
                "duration_seconds": context.duration_seconds,
                "threshold_seconds": LONG_VIDEO_THRESHOLD_SECONDS,
            },
            "has_embedded_subtitles": {
                "answer": context.has_subtitles,
                "details": context.analysis.get("has_subtitles_details"),
            },
            "is_motion_design": {
                "answer": (
                    context.video_type == "motion_design"
                    if context.video_type is not None
                    else None
                ),
            },
            "is_interview": {
                "answer": (
                    context.video_type == "interview"
                    if context.video_type is not None
                    else None
                ),
            },
        },
        "features": {
            "duration": {
                "seconds": context.duration_seconds,
                "threshold_seconds": LONG_VIDEO_THRESHOLD_SECONDS,
                "longer_than_10_minutes": context.is_long_video,
            },
            "has_subtitles": {
                "value": context.has_subtitles,
                "details": context.analysis.get("has_subtitles_details"),
            },
            "video_type": context.video_type,
            "motion_design": (
                context.video_type == "motion_design"
                if context.video_type is not None
                else None
            ),
            "interview": (
                context.video_type == "interview"
                if context.video_type is not None
                else None
            ),
        },
        "routing": context.routing(),
        "options": options.to_dict(),
        "plan": {
            "task_count": len(task_list),
            "tasks": [
                task.to_dict(context, options)
                for task in task_list
            ],
        },
        "execution": execution or {"status": "not_started", "tasks": {}},
    }


def write_manifest(
    context: VideoContext,
    options: PipelineOptions,
    tasks: Iterable[PlannedTask],
    *,
    execution: dict[str, Any] | None = None,
) -> Path:
    context.metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(
        context,
        options,
        tasks,
        execution=execution,
    )
    context.manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return context.manifest_path


def update_execution(
    manifest_path: str | Path,
    *,
    pipeline_status: str | None = None,
    task_id: str | None = None,
    task_status: str | None = None,
    error: str | None = None,
) -> None:
    target = Path(manifest_path)
    payload = read_manifest(target)
    execution = payload.setdefault("execution", {"status": "not_started", "tasks": {}})
    if pipeline_status is not None:
        execution["status"] = pipeline_status
    if task_id is not None and task_status is not None:
        tasks = execution.setdefault("tasks", {})
        task = tasks.setdefault(task_id, {})
        task["status"] = task_status
        if task_status == "running":
            task["started_at"] = utc_now()
        else:
            task["finished_at"] = utc_now()
        if error:
            task["error"] = error
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
