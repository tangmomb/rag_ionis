from __future__ import annotations

from .catalog import TASKS, TaskSpec
from .context import PipelineContext
from .contracts import TaskResult, TaskStatus
from .manifest import write_manifest


class PipelineBlockedError(RuntimeError):
    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"Pipeline bloque par {task_id}: {reason}")


def _postcondition_satisfied(
    spec: TaskSpec,
    context: PipelineContext,
) -> bool:
    try:
        return spec.postcondition_satisfied(context)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _completion_satisfied(
    spec: TaskSpec,
    context: PipelineContext,
    task_id: str,
) -> bool:
    if spec.postcondition is not None:
        return _postcondition_satisfied(spec, context)
    return context.artifacts_valid(task_id)


def _validate_handler_result(
    result: object,
    *,
    context: PipelineContext,
    spec: TaskSpec,
    task_id: str,
) -> TaskResult:
    if not isinstance(result, TaskResult):
        raise TypeError(
            f"Le handler {task_id!r} doit retourner TaskResult, "
            f"pas {type(result).__name__}."
        )

    context.apply_task_result(task_id, result)

    if result.status is TaskStatus.CACHED and not _completion_satisfied(
        spec,
        context,
        task_id,
    ):
        return TaskResult.blocked(
            result.reason
            or "Le cache annonce par la tache ne satisfait pas sa postcondition."
        )
    if (
        result.status is TaskStatus.SUCCEEDED
        and spec.postcondition is not None
        and not _postcondition_satisfied(spec, context)
    ):
        return TaskResult.blocked(
            result.reason
            or "La postcondition de la tache n'est pas satisfaite."
        )
    if result.artifacts and not context.artifacts_valid(task_id):
        return TaskResult.blocked(
            result.reason
            or "Un ou plusieurs artefacts annonces sont introuvables."
        )
    return result


def _resume_from_checkpoint(
    context: PipelineContext,
    spec: TaskSpec,
    task_id: str,
) -> bool:
    if context.options.force:
        return False
    execution = context.execution.tasks.get(task_id)
    if execution is None or not execution.status.is_successful:
        return False
    if not _completion_satisfied(spec, context, task_id):
        return False
    current_fingerprint = context.artifact_fingerprint(task_id)
    return (
        execution.artifact_fingerprint is None
        or execution.artifact_fingerprint == current_fingerprint
    )


def execute_tasks(
    context: PipelineContext,
    *,
    dry_run: bool = False,
) -> None:
    planned_tasks = context._coerce_plan(context.plan, validate=True)
    if planned_tasks != context.plan:
        context.set_plan(planned_tasks)
    task_ids = [task.id for task in planned_tasks]
    specs = [TASKS[task_id] for task_id in task_ids]

    if context.execution.plan_hash != context.compute_plan_hash(planned_tasks):
        context.set_plan(planned_tasks)

    if dry_run:
        checkpoint_chain_valid = True
        for task_id, spec in zip(task_ids, specs):
            resumable = (
                checkpoint_chain_valid
                and _resume_from_checkpoint(context, spec, task_id)
            )
            suffix = (
                " [checkpoint valide]"
                if resumable
                else ""
            )
            print(
                f"[dry-run] {task_id} -> {spec.entrypoint}{suffix}",
                flush=True,
            )
            checkpoint_chain_valid = resumable
        return

    context.execution.start_pipeline()
    write_manifest(context)
    checkpoint_chain_valid = True
    for index, (task_id, spec) in enumerate(
        zip(task_ids, specs),
        start=1,
    ):
        if (
            checkpoint_chain_valid
            and _resume_from_checkpoint(context, spec, task_id)
        ):
            print(
                f"\n[{index}/{len(task_ids)}] {task_id} -> cache valide",
                flush=True,
            )
            context.execution.finish_task(
                task_id,
                TaskStatus.CACHED,
                reason="Repris depuis le checkpoint; artefacts valides.",
                artifact_fingerprint=context.artifact_fingerprint(task_id),
            )
            write_manifest(context)
            continue

        checkpoint_chain_valid = False
        print(
            f"\n[{index}/{len(task_ids)}] {task_id} -> {spec.entrypoint}",
            flush=True,
        )
        context.execution.start_task(task_id)
        write_manifest(context)
        try:
            context._task_force = True
            try:
                raw_result = spec.handler(context)
            finally:
                context._task_force = False
            result = _validate_handler_result(
                raw_result,
                context=context,
                spec=spec,
                task_id=task_id,
            )
        except Exception as error:
            context.execution.fail_task(task_id, error)
            context.execution.fail_pipeline()
            write_manifest(context)
            raise

        if result.status is TaskStatus.FAILED:
            error = RuntimeError(result.reason or "Echec explicite de la tache.")
            context.execution.fail_task(task_id, error)
            context.execution.fail_pipeline()
            write_manifest(context)
            raise error

        if result.status is TaskStatus.BLOCKED:
            reason = result.reason or "Postcondition non satisfaite."
            context.execution.finish_task(
                task_id,
                TaskStatus.BLOCKED,
                reason=reason,
            )
            context.execution.skip_remaining(
                task_ids[index:],
                reason=f"Tache amont bloquee: {task_id}.",
            )
            context.execution.block_pipeline()
            write_manifest(context)
            raise PipelineBlockedError(task_id, reason)

        context.execution.finish_task(
            task_id,
            result.status,
            reason=result.reason,
            artifact_fingerprint=context.artifact_fingerprint(task_id),
        )
        write_manifest(context)

    context.execution.complete_pipeline()
    write_manifest(context)
