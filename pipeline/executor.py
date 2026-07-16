from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from .catalog import PROJECT_ROOT, task_command, task_environment
from .context import VideoContext
from .manifest import update_execution
from .options import PipelineOptions
from .planner import PlannedTask


def execute_tasks(
    context: VideoContext,
    options: PipelineOptions,
    tasks: Iterable[PlannedTask],
    *,
    dry_run: bool = False,
) -> None:
    task_list = list(tasks)
    if dry_run:
        for task in task_list:
            print("[dry-run] " + " ".join(task_command(task.id, context, options)))
        return

    update_execution(context.manifest_path, pipeline_status="running")
    env = os.environ.copy()
    env.update(task_environment(context, options))
    for index, task in enumerate(task_list, start=1):
        command = task_command(task.id, context, options)
        print(
            f"\n[{index}/{len(task_list)}] {task.id} - "
            + " ".join(command),
            flush=True,
        )
        update_execution(
            context.manifest_path,
            task_id=task.id,
            task_status="running",
        )
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
        except Exception as error:
            update_execution(
                context.manifest_path,
                pipeline_status="failed",
                task_id=task.id,
                task_status="failed",
                error=str(error),
            )
            raise
        update_execution(
            context.manifest_path,
            task_id=task.id,
            task_status="completed",
        )
    update_execution(context.manifest_path, pipeline_status="completed")
