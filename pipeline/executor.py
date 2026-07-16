from __future__ import annotations

from .catalog import TASKS
from .context import PipelineContext
from .manifest import write_manifest


def execute_tasks(
    context: PipelineContext,
    *,
    dry_run: bool = False,
) -> None:
    task_ids = [
        str(task["id"])
        for task in context.plan
        if task.get("id")
    ]
    if dry_run:
        for task_id in task_ids:
            spec = TASKS[task_id]
            print(f"[dry-run] {task_id} -> {spec.entrypoint}")
        return

    context.execution.start_pipeline()
    write_manifest(context)
    for index, task_id in enumerate(task_ids, start=1):
        spec = TASKS[task_id]
        print(
            f"\n[{index}/{len(task_ids)}] {task_id} -> {spec.entrypoint}",
            flush=True,
        )
        context.execution.start_task(task_id)
        write_manifest(context)
        try:
            with context.runtime_environment():
                spec.handler(context)
        except Exception as error:
            context.execution.fail_task(task_id, error)
            context.execution.fail_pipeline()
            write_manifest(context)
            raise
        context.execution.complete_task(task_id)
        write_manifest(context)

    context.execution.complete_pipeline()
    write_manifest(context)
