from __future__ import annotations

import shutil
from pathlib import Path

from .context import PipelineContext
from .executor import execute_tasks
from .manifest import write_manifest
from .options import PipelineOptions
from .planner import inspection_plan, processing_plan
from .support.paths import OUTPUTS_DIR_NAME, outputs_dir


def clean_outputs_for_force_run(video_path: str | Path) -> Path:
    video_path = Path(video_path)
    video_directory = (
        video_path.resolve()
        if video_path.is_dir()
        else video_path.parent.resolve()
    )
    target = outputs_dir(video_path).resolve()
    expected = video_directory / OUTPUTS_DIR_NAME
    if target != expected or target.name != OUTPUTS_DIR_NAME:
        raise RuntimeError(
            f"Refus de supprimer un dossier outputs inattendu: {target}"
        )
    if target.exists():
        shutil.rmtree(target)
        print(f"[clean] {target}", flush=True)
    return target


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
    context.set_plan(tasks)
    write_manifest(context)
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
    context.set_plan(tasks)
    write_manifest(context)
    print(f"[manifest] {context.manifest_path}", flush=True)
    if probe_only:
        return context

    execute_tasks(context, dry_run=dry_run)
    if dry_run:
        return context

    downstream = processing_plan(context)
    context.set_plan(downstream)
    write_manifest(context)
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
    if options.force and not dry_run:
        clean_outputs_for_force_run(video_path)
    context = PipelineContext.inspect(video_path, options)
    if not skip_inspection or not context.routing_ready:
        inspection_tasks = inspection_plan()
        context.set_plan(inspection_tasks)
        write_manifest(context)
        print(f"[manifest] {context.manifest_path}", flush=True)
        execute_tasks(context, dry_run=dry_run)
        if dry_run:
            return context

    tasks = processing_plan(context)
    context.set_plan(tasks)
    write_manifest(context)
    execute_tasks(context, dry_run=dry_run)
    return context
