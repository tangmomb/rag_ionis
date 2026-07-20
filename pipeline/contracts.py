from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4


LONG_VIDEO_THRESHOLD_SECONDS = 600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoType(str, Enum):
    INTERVIEW = "interview"
    LONG_VIDEO = "long_video"
    MOTION_DESIGN = "motion_design"
    VIDEO_RECORDING = "video_recording"

    @classmethod
    def from_value(cls, value: Any) -> "VideoType | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        try:
            return cls(normalized)
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"video_type invalide: {value!r}; valeurs attendues: {allowed}."
            ) from error


@dataclass(frozen=True)
class RoutingFacts:
    """Faits metier persistants utilises pour choisir la route."""

    has_subtitles: bool | None = None
    video_type: VideoType | None = None
    has_subtitles_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "RoutingFacts":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("Les faits de routage doivent etre un objet JSON.")

        has_subtitles = payload.get("has_subtitles")
        if has_subtitles is not None and not isinstance(has_subtitles, bool):
            raise ValueError("has_subtitles doit etre un booleen ou null.")

        details = payload.get("has_subtitles_details")
        if details is not None and not isinstance(details, Mapping):
            raise ValueError(
                "has_subtitles_details doit etre un objet JSON ou null."
            )

        return cls(
            has_subtitles=has_subtitles,
            video_type=VideoType.from_value(payload.get("video_type")),
            has_subtitles_details=dict(details) if details is not None else None,
        )

    def to_dict(self, *, include_none: bool = True) -> dict[str, Any]:
        payload = {
            "has_subtitles": self.has_subtitles,
            "video_type": self.video_type.value if self.video_type else None,
            "has_subtitles_details": self.has_subtitles_details,
        }
        if include_none:
            return payload
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }


class TaskStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    # La valeur JSON historique reste "completed". Le nom SUCCEEDED porte
    # desormais la semantique du contrat sans casser les lecteurs existants.
    SUCCEEDED = "completed"
    COMPLETED = "completed"
    CACHED = "cached"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: Any) -> "TaskStatus":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.NOT_STARTED.value).strip().lower()
        if normalized == "succeeded":
            return cls.SUCCEEDED
        try:
            return cls(normalized)
        except ValueError as error:
            allowed = ", ".join(dict.fromkeys(item.value for item in cls))
            raise ValueError(
                f"Statut de tache invalide: {value!r}; valeurs attendues: {allowed}."
            ) from error

    @property
    def is_successful(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.CACHED,
            TaskStatus.SKIPPED,
        }

    @property
    def is_terminal(self) -> bool:
        return self not in {TaskStatus.NOT_STARTED, TaskStatus.RUNNING}


class RunStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: Any) -> "RunStatus":
        if isinstance(value, cls):
            return value
        normalized = str(value or cls.NOT_STARTED.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Statut d'execution invalide: {value!r}; valeurs attendues: {allowed}."
            ) from error


@dataclass(frozen=True)
class TaskResult:
    """Resultat explicite retourne par une etape au nouvel executant."""

    status: TaskStatus = TaskStatus.SUCCEEDED
    artifacts: tuple[str | Path, ...] = ()
    reason: str | None = None
    value: Any = None

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, TaskStatus)
            else TaskStatus.from_value(self.status)
        )
        if status in {TaskStatus.NOT_STARTED, TaskStatus.RUNNING}:
            raise ValueError(
                "TaskResult exige un statut terminal."
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))

    @classmethod
    def succeeded(
        cls,
        value: Any = None,
        *,
        artifacts: Iterable[str | Path] = (),
        reason: str | None = None,
    ) -> "TaskResult":
        return cls(
            TaskStatus.SUCCEEDED,
            tuple(artifacts),
            reason,
            value,
        )

    @classmethod
    def cached(
        cls,
        *,
        artifacts: Iterable[str | Path] = (),
        reason: str | None = None,
        value: Any = None,
    ) -> "TaskResult":
        return cls(TaskStatus.CACHED, tuple(artifacts), reason, value)

    @classmethod
    def skipped(cls, reason: str) -> "TaskResult":
        return cls(TaskStatus.SKIPPED, reason=reason)

    @classmethod
    def blocked(cls, reason: str) -> "TaskResult":
        return cls(TaskStatus.BLOCKED, reason=reason)

    @classmethod
    def failed(cls, reason: str) -> "TaskResult":
        return cls(TaskStatus.FAILED, reason=reason)


@dataclass(frozen=True)
class PlannedTask:
    id: str
    reason: str

    @classmethod
    def from_dict(cls, payload: Any) -> "PlannedTask":
        if not isinstance(payload, Mapping):
            raise ValueError("Une tache planifiee doit etre un objet.")
        task_id = str(payload.get("id") or "").strip()
        if not task_id:
            raise ValueError("Une tache planifiee exige un identifiant.")
        reason = str(payload.get("reason") or "unspecified").strip()
        return cls(task_id, reason or "unspecified")

    def to_dict(self) -> dict[str, object]:
        # Import local pour eviter le cycle contracts -> catalog -> context.
        from .catalog import TASKS

        try:
            spec = TASKS[self.id]
        except KeyError as error:
            raise ValueError(
                f"Tache inconnue dans le plan: {self.id!r}."
            ) from error
        return {
            "id": self.id,
            "phase": spec.phase,
            "title": spec.title,
            "reason": self.reason,
            "handler": spec.entrypoint,
            "version": spec.version,
        }


@dataclass
class TaskExecution:
    status: TaskStatus = TaskStatus.NOT_STARTED
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    reason: str | None = None
    attempts: int = 0
    artifact_fingerprint: str | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "TaskExecution":
        if not isinstance(payload, Mapping):
            return cls()
        attempts = payload.get("attempts", 0)
        try:
            normalized_attempts = max(0, int(attempts))
        except (TypeError, ValueError) as error:
            raise ValueError("execution.tasks.*.attempts doit etre un entier.") from error
        return cls(
            status=TaskStatus.from_value(payload.get("status")),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            error=payload.get("error"),
            reason=payload.get("reason"),
            attempts=normalized_attempts,
            artifact_fingerprint=(
                str(payload["artifact_fingerprint"])
                if payload.get("artifact_fingerprint") is not None
                else None
            ),
        )

    def start(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = utc_now()
        self.finished_at = None
        self.error = None
        self.reason = None
        self.artifact_fingerprint = None
        self.attempts += 1

    def finish(
        self,
        status: TaskStatus,
        *,
        reason: str | None = None,
        error: Exception | str | None = None,
        artifact_fingerprint: str | None = None,
    ) -> None:
        normalized = TaskStatus.from_value(status)
        if not normalized.is_terminal:
            raise ValueError("Une tache ne peut finir avec un statut non terminal.")
        self.status = normalized
        self.finished_at = utc_now()
        self.reason = reason
        self.error = str(error) if error is not None else None
        self.artifact_fingerprint = artifact_fingerprint

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }


@dataclass
class RunExecution:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    plan_hash: str = ""
    status: RunStatus = RunStatus.NOT_STARTED
    tasks: dict[str, TaskExecution] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def for_plan(
        cls,
        plan_hash: str,
        task_ids: Iterable[str],
    ) -> "RunExecution":
        execution = cls(plan_hash=plan_hash)
        execution.ensure_tasks(task_ids)
        return execution

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        fallback_plan_hash: str = "",
    ) -> "RunExecution":
        if not isinstance(payload, Mapping):
            return cls(plan_hash=fallback_plan_hash)
        tasks = payload.get("tasks")
        run_id = str(payload.get("run_id") or uuid4().hex)
        plan_hash = str(payload.get("plan_hash") or fallback_plan_hash)
        return cls(
            run_id=run_id,
            plan_hash=plan_hash,
            status=RunStatus.from_value(payload.get("status")),
            tasks={
                str(task_id): TaskExecution.from_dict(task_payload)
                for task_id, task_payload in (
                    tasks.items() if isinstance(tasks, Mapping) else []
                )
            },
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
        )

    @property
    def has_activity(self) -> bool:
        return (
            self.status is not RunStatus.NOT_STARTED
            or any(
                task.status is not TaskStatus.NOT_STARTED
                for task in self.tasks.values()
            )
        )

    def ensure_tasks(self, task_ids: Iterable[str]) -> None:
        for task_id in task_ids:
            self.tasks.setdefault(str(task_id), TaskExecution())

    def start_pipeline(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = self.started_at or utc_now()
        self.finished_at = None

    def complete_pipeline(self) -> None:
        self.status = RunStatus.COMPLETED
        self.finished_at = utc_now()

    def block_pipeline(self) -> None:
        self.status = RunStatus.BLOCKED
        self.finished_at = utc_now()

    def fail_pipeline(self) -> None:
        self.status = RunStatus.FAILED
        self.finished_at = utc_now()

    def start_task(self, task_id: str) -> None:
        self.tasks.setdefault(task_id, TaskExecution()).start()

    def finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None = None,
        artifact_fingerprint: str | None = None,
    ) -> None:
        self.tasks.setdefault(task_id, TaskExecution()).finish(
            status,
            reason=reason,
            artifact_fingerprint=artifact_fingerprint,
        )

    def complete_task(self, task_id: str) -> None:
        self.finish_task(task_id, TaskStatus.SUCCEEDED)

    def fail_task(self, task_id: str, error: Exception | str) -> None:
        self.tasks.setdefault(task_id, TaskExecution()).finish(
            TaskStatus.FAILED,
            error=error,
        )

    def skip_remaining(
        self,
        task_ids: Iterable[str],
        *,
        reason: str,
    ) -> None:
        for task_id in task_ids:
            task = self.tasks.setdefault(task_id, TaskExecution())
            if task.status in {TaskStatus.NOT_STARTED, TaskStatus.RUNNING}:
                task.finish(TaskStatus.SKIPPED, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "status": self.status.value,
            **(
                {"started_at": self.started_at}
                if self.started_at is not None
                else {}
            ),
            **(
                {"finished_at": self.finished_at}
                if self.finished_at is not None
                else {}
            ),
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
        }
