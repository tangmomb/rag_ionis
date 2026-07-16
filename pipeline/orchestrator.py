from __future__ import annotations

from pathlib import Path

from .context import VideoContext
from .executor import execute_tasks
from .manifest import read_manifest, write_manifest
from .options import PipelineOptions
from .planner import inspection_plan, processing_plan


def plan_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    include_inspection: bool = False,
) -> VideoContext:
    context = VideoContext.inspect(video_path)
    tasks = inspection_plan() if include_inspection else (
        processing_plan(context) if context.routing_ready else inspection_plan()
    )
    previous = read_manifest(context.manifest_path)
    write_manifest(
        context,
        options,
        tasks,
        execution=previous.get("execution"),
    )
    return context


def inspect_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    probe_only: bool = False,
    dry_run: bool = False,
) -> VideoContext:
    context = VideoContext.inspect(video_path)
    tasks = inspection_plan()
    write_manifest(context, options, tasks)
    print(f"[manifest] {context.manifest_path}", flush=True)
    if probe_only:
        return context

    execute_tasks(context, options, tasks, dry_run=dry_run)
    if dry_run:
        return context

    refreshed = VideoContext.inspect(video_path)
    downstream = processing_plan(refreshed)
    inspection_execution = read_manifest(context.manifest_path).get("execution")
    write_manifest(
        refreshed,
        options,
        downstream,
        execution=inspection_execution,
    )
    print(
        f"[route] {refreshed.routing()['pipeline_id']} -> "
        f"{len(downstream)} utilitaire(s)",
        flush=True,
    )
    return refreshed


def run_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    skip_inspection: bool = False,
    dry_run: bool = False,
) -> VideoContext:
    context = VideoContext.inspect(video_path)
    if not skip_inspection or not context.routing_ready:
        context = inspect_video(
            video_path,
            options,
            probe_only=False,
            dry_run=dry_run,
        )
        if dry_run:
            return context

    tasks = processing_plan(context)
    previous_execution = read_manifest(context.manifest_path).get("execution")
    write_manifest(
        context,
        options,
        tasks,
        execution=previous_execution,
    )
    execute_tasks(context, options, tasks, dry_run=dry_run)
    return context
