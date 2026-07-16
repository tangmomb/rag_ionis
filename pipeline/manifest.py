from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .context import (
    LONG_VIDEO_THRESHOLD_SECONDS,
    PipelineContext,
    PipelineExecution,
    utc_now,
)
from .options import PipelineOptions
from .planner import PlannedTask


SCHEMA_VERSION = 3


def read_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _plan_payload(
    context: PipelineContext,
    tasks: Iterable[PlannedTask] | None,
) -> list[dict[str, Any]]:
    if tasks is not None:
        return [task.to_dict(context) for task in tasks]
    return list(context.plan)


def build_manifest(
    context: PipelineContext,
    options: PipelineOptions | None = None,
    tasks: Iterable[PlannedTask] | None = None,
    *,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_options = options or context.options
    task_list = _plan_payload(context, tasks)
    execution_payload = (
        execution
        if execution is not None
        else context.execution.to_dict()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "video": {
            "id": context.video_id,
            "url": context.video_url,
            "title": context.title,
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
        "options": selected_options.to_dict(),
        "artifacts": context.artifacts.to_dict(),
        "plan": {
            "task_count": len(task_list),
            "tasks": task_list,
        },
        "execution": execution_payload,
    }


def write_manifest(
    context: PipelineContext,
    options: PipelineOptions | None = None,
    tasks: Iterable[PlannedTask] | None = None,
    *,
    execution: dict[str, Any] | None = None,
) -> Path:
    if options is not None:
        context.options = options
    if tasks is not None:
        context.set_plan([task.to_dict(context) for task in tasks])
    if execution is not None:
        context.execution = PipelineExecution.from_dict(execution)
        context.execution.ensure_tasks(
            [str(task["id"]) for task in context.plan if task.get("id")]
        )

    context.metadata_dir.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(context)
    temporary_path = context.manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(context.manifest_path)
    return context.manifest_path
