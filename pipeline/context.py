from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .options import PipelineOptions
from .probe import probe_video


LONG_VIDEO_THRESHOLD_SECONDS = 600
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
ANALYSIS_NAME = "pipeline_analysis.json"
MANIFEST_NAME = "video_manifest.json"
YOUTUBE_METADATA_NAME = "youtube_video_metadata.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def video_in_directory(directory: str | Path) -> Path:
    root = Path(directory)
    videos = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) != 1:
        raise RuntimeError(
            f"{root} doit contenir exactement une video; trouve: {len(videos)}."
        )
    return videos[0]


@dataclass
class PipelineArtifacts:
    """References legeres vers les sorties produites par les etapes."""

    directories: dict[str, str] = field(default_factory=dict)
    by_task: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "PipelineArtifacts":
        if not isinstance(payload, dict):
            return cls()
        directories = payload.get("directories")
        by_task = payload.get("by_task")
        return cls(
            directories={
                str(key): str(value)
                for key, value in (directories.items() if isinstance(directories, dict) else [])
            },
            by_task={
                str(key): [str(item) for item in value if str(item).strip()]
                for key, value in (by_task.items() if isinstance(by_task, dict) else [])
                if isinstance(value, list)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "directories": dict(self.directories),
            "by_task": {
                task_id: list(paths)
                for task_id, paths in self.by_task.items()
            },
        }


@dataclass
class TaskExecution:
    status: str = "not_started"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "TaskExecution":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            status=str(payload.get("status") or "not_started"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass
class PipelineExecution:
    status: str = "not_started"
    tasks: dict[str, TaskExecution] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "PipelineExecution":
        if not isinstance(payload, dict):
            return cls()
        tasks = payload.get("tasks")
        return cls(
            status=str(payload.get("status") or "not_started"),
            tasks={
                str(task_id): TaskExecution.from_dict(task_payload)
                for task_id, task_payload in (
                    tasks.items() if isinstance(tasks, dict) else []
                )
            },
        )

    def ensure_tasks(self, task_ids: list[str]) -> None:
        for task_id in task_ids:
            self.tasks.setdefault(task_id, TaskExecution())

    def start_pipeline(self) -> None:
        self.status = "running"

    def complete_pipeline(self) -> None:
        self.status = "completed"

    def fail_pipeline(self) -> None:
        self.status = "failed"

    def start_task(self, task_id: str) -> None:
        task = self.tasks.setdefault(task_id, TaskExecution())
        task.status = "running"
        task.started_at = utc_now()
        task.finished_at = None
        task.error = None

    def complete_task(self, task_id: str) -> None:
        task = self.tasks.setdefault(task_id, TaskExecution())
        task.status = "completed"
        task.finished_at = utc_now()
        task.error = None

    def fail_task(self, task_id: str, error: Exception | str) -> None:
        task = self.tasks.setdefault(task_id, TaskExecution())
        task.status = "failed"
        task.finished_at = utc_now()
        task.error = str(error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tasks": {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            },
        }


@dataclass
class PipelineContext:
    """Etat mutable partage par l'orchestrateur et toutes les etapes."""

    video_path: Path
    options: PipelineOptions = field(default_factory=PipelineOptions)
    media: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    execution: PipelineExecution = field(default_factory=PipelineExecution)
    plan: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def inspect(
        cls,
        video_path: str | Path,
        options: PipelineOptions | None = None,
    ) -> "PipelineContext":
        candidate = Path(video_path)
        video = video_in_directory(candidate) if candidate.is_dir() else candidate
        metadata_dir = video.parent / "metadata"
        previous_manifest = load_json(metadata_dir / MANIFEST_NAME)
        context = cls(
            video_path=video,
            options=options or PipelineOptions(),
            media=probe_video(video),
            source_metadata=load_json(metadata_dir / YOUTUBE_METADATA_NAME),
            analysis=load_json(metadata_dir / ANALYSIS_NAME),
            artifacts=PipelineArtifacts.from_dict(previous_manifest.get("artifacts")),
            execution=PipelineExecution.from_dict(previous_manifest.get("execution")),
        )
        context.refresh_artifact_directories()
        return context

    @property
    def video_dir(self) -> Path:
        return self.video_path.parent

    @property
    def video_id(self) -> str:
        return self.video_path.stem

    @property
    def video_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def title(self) -> str | None:
        snippet = self.source_metadata.get("snippet")
        value = snippet.get("title") if isinstance(snippet, dict) else None
        normalized = str(value).strip() if value is not None else ""
        return normalized or None

    @property
    def metadata_dir(self) -> Path:
        return self.video_dir / "metadata"

    @property
    def outputs_dir(self) -> Path:
        return self.video_dir / "outputs"

    @property
    def manifest_path(self) -> Path:
        return self.metadata_dir / MANIFEST_NAME

    @property
    def duration_seconds(self) -> float:
        value = self.media.get("duration_seconds")
        if value is None:
            raise RuntimeError(f"Duree video introuvable: {self.video_path}")
        return float(value)

    @property
    def is_long_video(self) -> bool:
        return self.duration_seconds > LONG_VIDEO_THRESHOLD_SECONDS

    @property
    def has_subtitles(self) -> bool | None:
        value = self.analysis.get("has_subtitles")
        return value if isinstance(value, bool) else None

    @property
    def subtitle_available(self) -> bool:
        return self.has_subtitles is True

    @property
    def video_type(self) -> str | None:
        value = self.analysis.get("video_type")
        normalized = str(value).strip() if value is not None else ""
        return normalized or None

    @property
    def visual_strategy(self) -> str | None:
        return self.video_type

    @property
    def transcript_strategy(self) -> str | None:
        if self.has_subtitles is True:
            return "ocr"
        if self.has_subtitles is False:
            return "whisper"
        return None

    @property
    def transcripts_dir_name(self) -> str:
        if self.transcript_strategy == "ocr":
            return "transcripts_ocr"
        if self.transcript_strategy == "whisper":
            return "transcripts_whisper"
        return "transcripts"

    @property
    def chunk_strategy(self) -> str:
        return "long" if self.is_long_video else "short"

    @property
    def routing_ready(self) -> bool:
        return self.transcript_strategy is not None and self.visual_strategy is not None

    def routing(self) -> dict[str, Any]:
        missing = []
        if self.has_subtitles is None:
            missing.append("has_subtitles")
        if self.visual_strategy is None:
            missing.append("video_type")
        parts = [
            self.chunk_strategy,
            self.transcript_strategy,
            self.visual_strategy,
        ]
        return {
            "status": "ready" if not missing else "needs_content_inspection",
            "pipeline_id": ".".join(part for part in parts if part),
            "transcript_strategy": self.transcript_strategy,
            "chunk_strategy": self.chunk_strategy,
            "visual_strategy": self.visual_strategy,
            "missing_features": missing,
        }

    def set_plan(self, tasks: list[dict[str, Any]]) -> None:
        self.plan = list(tasks)
        self.execution.ensure_tasks(
            [str(task["id"]) for task in self.plan if task.get("id")]
        )

    def refresh_analysis(self) -> None:
        self.analysis = load_json(self.metadata_dir / ANALYSIS_NAME)
        self.refresh_artifact_directories()

    def refresh_artifact_directories(self) -> None:
        self.artifacts.directories = {
            "video": ".",
            "metadata": "metadata",
            "outputs": "outputs",
            "images": "outputs/images",
            "ocr": "outputs/ocr",
            "transcripts": f"outputs/{self.transcripts_dir_name}",
            "speakers": "outputs/speakers",
            "chunks": "outputs/chunks",
        }

    def record_artifacts(self, task_id: str, result: Any) -> None:
        paths: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, Path):
                if not value.exists():
                    return
                try:
                    relative = value.resolve().relative_to(self.video_dir.resolve())
                    paths.append(relative.as_posix())
                except ValueError:
                    paths.append(value.resolve().as_posix())
                return
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        collect(result)
        if paths:
            self.artifacts.by_task[task_id] = list(dict.fromkeys(paths))
        self.refresh_analysis()

    @contextmanager
    def runtime_environment(self) -> Iterator[None]:
        """Compatibilite temporaire pour les utilitaires qui lisent encore os.environ."""

        updates = {
            "PIPELINE_OPENAI_MODE": self.options.openai_mode,
            "PIPELINE_TRANSCRIPTS_DIR_NAME": self.transcripts_dir_name,
            "PYTHONUTF8": "1",
        }
        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
