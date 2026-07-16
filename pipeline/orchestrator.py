from __future__ import annotations

from pathlib import Path

from .context import PipelineContext
from .executor import execute_tasks
from .manifest import write_manifest
from .options import PipelineOptions
from .planner import inspection_plan, processing_plan


def plan_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    include_inspection: bool = False,
) -> PipelineContext:
    context = PipelineContext.inspect(video_path, options)
    tasks = inspection_plan() if include_inspection else (
        processing_plan(context) if context.routing_ready else inspection_plan()
    )
    write_manifest(context, tasks=tasks)
    return context


def inspect_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    probe_only: bool = False,
    dry_run: bool = False,
) -> PipelineContext:
    context = PipelineContext.inspect(video_path, options)
    tasks = inspection_plan()
    write_manifest(context, tasks=tasks)
    print(f"[manifest] {context.manifest_path}", flush=True)
    if probe_only:
        return context

    execute_tasks(context, dry_run=dry_run)
    if dry_run:
        return context

    downstream = processing_plan(context)
    write_manifest(context, tasks=downstream)
    print(
        f"[route] {context.routing()['pipeline_id']} -> "
        f"{len(downstream)} utilitaire(s)",
        flush=True,
    )
    return context


def run_video(
    video_path: str | Path,
    options: PipelineOptions,
    *,
    skip_inspection: bool = False,
    dry_run: bool = False,
) -> PipelineContext:
    context = PipelineContext.inspect(video_path, options)
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
    write_manifest(context, tasks=tasks)
    execute_tasks(context, dry_run=dry_run)
    return context
