from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .context import PipelineContext
from .contracts import RoutingFacts, RunExecution, utc_now
from .options import PipelineOptions
from .support.json_io import read_json, write_json
from .support.paths import analysed_infos_path


SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = {3, SCHEMA_VERSION}


class ManifestValidationError(ValueError):
    pass


def validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ManifestValidationError(
            "Le manifeste doit contenir un objet JSON."
        )

    version = payload.get("schema_version")
    if not isinstance(version, int):
        raise ManifestValidationError(
            "Le manifeste exige un schema_version entier."
        )
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(
            str(item) for item in sorted(SUPPORTED_SCHEMA_VERSIONS)
        )
        raise ManifestValidationError(
            f"schema_version={version} non supporte; versions: {supported}."
        )

    plan = payload.get("plan")
    if plan is not None:
        if not isinstance(plan, Mapping):
            raise ManifestValidationError("plan doit etre un objet JSON.")
        tasks = plan.get("tasks", [])
        if not isinstance(tasks, list):
            raise ManifestValidationError("plan.tasks doit etre une liste.")
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                raise ManifestValidationError(
                    f"plan.tasks[{index}] doit etre un objet."
                )
            if not str(task.get("id") or "").strip():
                raise ManifestValidationError(
                    f"plan.tasks[{index}].id est obligatoire."
                )

    facts = payload.get("routing_facts")
    if facts is not None:
        try:
            RoutingFacts.from_dict(facts)
        except ValueError as error:
            raise ManifestValidationError(str(error)) from error

    route = payload.get("route")
    if route is not None:
        if not isinstance(route, Mapping):
            raise ManifestValidationError("route doit etre un objet JSON.")
        missing_facts = route.get("missing_facts", [])
        if not isinstance(missing_facts, list) or not all(
            isinstance(item, str)
            for item in missing_facts
        ):
            raise ManifestValidationError(
                "route.missing_facts doit etre une liste de chaines."
            )

    artifacts = payload.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, Mapping):
            raise ManifestValidationError("artifacts doit etre un objet JSON.")
        by_task = artifacts.get("by_task", {})
        if not isinstance(by_task, Mapping) or not all(
            isinstance(paths, list)
            and all(isinstance(path, str) for path in paths)
            for paths in by_task.values()
        ):
            raise ManifestValidationError(
                "artifacts.by_task doit associer chaque tache a une liste de chemins."
            )

    options = payload.get("options")
    if options is not None:
        if not isinstance(options, Mapping):
            raise ManifestValidationError("options doit etre un objet JSON.")
        if "force" in options and not isinstance(options["force"], bool):
            raise ManifestValidationError(
                "options.force doit etre un booleen."
            )
        try:
            PipelineOptions.from_dict(options)
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"Options du manifeste invalides: {error}"
            ) from error

    execution = payload.get("execution")
    if execution is not None:
        if not isinstance(execution, Mapping):
            raise ManifestValidationError("execution doit etre un objet JSON.")
        execution_tasks = execution.get("tasks", {})
        if not isinstance(execution_tasks, Mapping):
            raise ManifestValidationError(
                "execution.tasks doit etre un objet JSON."
            )
        try:
            RunExecution.from_dict(execution)
        except ValueError as error:
            raise ManifestValidationError(
                f"Execution du manifeste invalide: {error}"
            ) from error

    history = payload.get("execution_history")
    if history is not None:
        if not isinstance(history, list):
            raise ManifestValidationError(
                "execution_history doit etre une liste."
            )
        try:
            for item in history:
                RunExecution.from_dict(item)
        except ValueError as error:
            raise ManifestValidationError(
                f"Historique d'execution invalide: {error}"
            ) from error

    if version == SCHEMA_VERSION:
        if not isinstance(facts, Mapping):
            raise ManifestValidationError(
                "routing_facts est obligatoire dans un manifeste v4."
            )
        if not isinstance(route, Mapping):
            raise ManifestValidationError(
                "route est obligatoire dans un manifeste v4."
            )
        if not isinstance(artifacts, Mapping):
            raise ManifestValidationError(
                "artifacts est obligatoire dans un manifeste v4."
            )
        if not isinstance(plan, Mapping):
            raise ManifestValidationError(
                "plan est obligatoire dans un manifeste v4."
            )
        if not isinstance(execution, Mapping):
            raise ManifestValidationError(
                "execution est obligatoire dans un manifeste v4."
            )
        run_id = execution.get("run_id")
        execution_plan_hash = execution.get("plan_hash")
        plan_hash = plan.get("hash")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ManifestValidationError(
                "execution.run_id est obligatoire dans un manifeste v4."
            )
        if not isinstance(plan_hash, str):
            raise ManifestValidationError(
                "plan.hash doit etre une chaine dans un manifeste v4."
            )
        if execution_plan_hash != plan_hash:
            raise ManifestValidationError(
                "execution.plan_hash doit correspondre a plan.hash."
            )

    return payload


def read_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = read_json(target)
    except OSError as error:
        raise ManifestValidationError(
            f"Manifeste illisible: {target}"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"Manifeste JSON invalide: {target}: {error}"
        ) from error
    return validate_manifest(payload)


def build_manifest(context: PipelineContext) -> dict[str, Any]:
    """Construit le manifeste depuis l'unique état du contexte."""

    for task_id, paths in list(context.artifacts.by_task.items()):
        retained = [
            path
            for path in paths
            if Path(path).name != "pipeline_analysis.json"
        ]
        if retained:
            context.artifacts.by_task[task_id] = retained
        else:
            context.artifacts.by_task.pop(task_id, None)

    selected_options = context.options
    planned_tasks = list(context.plan)
    task_list = [task.to_dict() for task in planned_tasks]
    plan_hash = context._plan_hash(
        planned_tasks,
        options_payload=selected_options.to_dict(),
        routing_facts=context.routing_facts,
        source_fingerprint=context.source_fingerprint,
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
        "routing_facts": context.routing_facts.to_dict(),
        "route": context.routing(),
        "options": selected_options.to_dict(),
        "artifacts": context.artifacts.to_dict(),
        "plan": {
            "hash": plan_hash,
            "task_count": len(task_list),
            "tasks": task_list,
        },
        "execution": context.execution.to_dict(),
        "execution_history": [
            item.to_dict()
            for item in context.execution_history
        ],
    }


def write_manifest(context: PipelineContext) -> Path:
    payload = build_manifest(context)
    validate_manifest(payload)
    target = write_json(context.manifest_path, payload)
    legacy_analysis = analysed_infos_path(context.video_path)
    if legacy_analysis.exists() and legacy_analysis.resolve() != target.resolve():
        legacy_analysis.unlink()
    return target
