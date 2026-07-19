from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import (
    PlannedTask,
    RoutingFacts,
    RunExecution,
    TaskExecution,
    TaskResult,
    TaskStatus,
)
from .options import PipelineOptions
from .probe import probe_video
from .support.json_io import read_json
from .support.youtube_metadata import (
    load_youtube_metadata,
    youtube_duration_seconds,
)


LONG_VIDEO_THRESHOLD_SECONDS = 600
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
ANALYSIS_NAME = "pipeline_analysis.json"
MANIFEST_NAME = "video_manifest.json"

def load_json(
    path: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        payload = read_json(path) if strict else read_json(path, default={})
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if strict:
            raise ValueError(f"JSON invalide ou illisible: {path}") from error
        return {}
    if not isinstance(payload, dict):
        if strict:
            raise ValueError(f"Un objet JSON est attendu dans: {path}")
        return {}
    return payload


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

    by_task: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "PipelineArtifacts":
        if not isinstance(payload, Mapping):
            return cls()
        by_task = payload.get("by_task")
        return cls(
            by_task={
                str(key): [str(item) for item in value if str(item).strip()]
                for key, value in (
                    by_task.items()
                    if isinstance(by_task, Mapping)
                    else []
                )
                if isinstance(value, list)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_task": {
                task_id: list(paths)
                for task_id, paths in self.by_task.items()
            },
        }


@dataclass
class PipelineContext:
    """Etat mutable partage par l'orchestrateur et toutes les etapes."""

    video_path: Path
    options: PipelineOptions = field(default_factory=PipelineOptions)
    media: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    routing_facts: RoutingFacts = field(default_factory=RoutingFacts)
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    execution: RunExecution = field(default_factory=RunExecution)
    execution_history: list[RunExecution] = field(default_factory=list)
    plan: list[PlannedTask] = field(default_factory=list)
    _task_force: bool = field(default=False, init=False, repr=False)

    @classmethod
    def inspect(
        cls,
        video_path: str | Path,
        options: PipelineOptions | None = None,
    ) -> "PipelineContext":
        candidate = Path(video_path)
        video = video_in_directory(candidate) if candidate.is_dir() else candidate
        metadata_dir = video.parent / "metadata"

        # Import local: manifest importe PipelineContext pour la serialisation.
        from .manifest import read_manifest

        previous_manifest = read_manifest(metadata_dir / MANIFEST_NAME)
        analysis_payload = cls._manifest_routing_facts(previous_manifest)
        if not analysis_payload:
            # Compatibilite de migration uniquement. Toute nouvelle ecriture
            # persiste ces faits dans video_manifest.json.
            analysis_payload = load_json(
                metadata_dir / ANALYSIS_NAME,
                strict=True,
            )

        source_metadata = load_youtube_metadata(video)
        media = probe_video(video)
        media["duration_seconds"] = youtube_duration_seconds(source_metadata)

        manifest_options = previous_manifest.get("options")
        selected_options = (
            options
            if options is not None
            else (
                PipelineOptions.from_dict(manifest_options)
                if isinstance(manifest_options, Mapping)
                else PipelineOptions()
            )
        )
        previous_plan = cls._coerce_plan(
            cls._manifest_plan(previous_manifest),
            validate=False,
        )
        previous_options = manifest_options
        if not isinstance(previous_options, Mapping):
            previous_options = selected_options.to_dict()
        previous_facts = RoutingFacts.from_dict(analysis_payload)
        fallback_plan_hash = cls._plan_hash(
            previous_plan,
            options_payload=dict(previous_options),
            routing_facts=previous_facts,
            source_fingerprint=cls._source_fingerprint(video),
        )

        history_payload = previous_manifest.get("execution_history")
        context = cls(
            video_path=video,
            options=selected_options,
            media=media,
            source_metadata=source_metadata,
            routing_facts=previous_facts,
            artifacts=PipelineArtifacts.from_dict(
                previous_manifest.get("artifacts")
            ),
            execution=RunExecution.from_dict(
                previous_manifest.get("execution"),
                fallback_plan_hash=fallback_plan_hash,
            ),
            execution_history=[
                RunExecution.from_dict(item)
                for item in (
                    history_payload
                    if isinstance(history_payload, list)
                    else []
                )
            ],
            plan=previous_plan,
        )
        context.execution.ensure_tasks(task.id for task in previous_plan)
        return context

    @staticmethod
    def _manifest_plan(payload: Mapping[str, Any]) -> list[Any]:
        plan = payload.get("plan")
        if not isinstance(plan, Mapping):
            return []
        tasks = plan.get("tasks")
        return list(tasks) if isinstance(tasks, list) else []

    @staticmethod
    def _manifest_routing_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
        facts = payload.get("routing_facts")
        if isinstance(facts, Mapping):
            return dict(facts)

        # Migration des manifestes v3, qui stockaient ces valeurs dans une
        # vue ``features`` au lieu d'un contrat de faits dédié.
        features = payload.get("features")
        if not isinstance(features, Mapping):
            return {}
        subtitle_feature = features.get("has_subtitles")
        if isinstance(subtitle_feature, Mapping):
            has_subtitles = subtitle_feature.get("value")
            details = subtitle_feature.get("details")
        else:
            has_subtitles = subtitle_feature
            details = None
        return {
            "has_subtitles": has_subtitles,
            "video_type": features.get("video_type"),
            "has_subtitles_details": details,
        }

    @staticmethod
    def _coerce_plan(
        tasks: Iterable[PlannedTask | Mapping[str, Any]],
        *,
        validate: bool,
    ) -> list[PlannedTask]:
        normalized = [
            task
            if isinstance(task, PlannedTask)
            else PlannedTask.from_dict(task)
            for task in tasks
        ]
        if validate:
            from .catalog import TASKS

            unknown = [task.id for task in normalized if task.id not in TASKS]
            if unknown:
                identifiers = ", ".join(dict.fromkeys(unknown))
                raise ValueError(
                    f"Tache(s) inconnue(s) dans le plan: {identifiers}."
                )
        seen: set[str] = set()
        duplicates: list[str] = []
        for task in normalized:
            if task.id in seen:
                duplicates.append(task.id)
            seen.add(task.id)
        if duplicates:
            identifiers = ", ".join(dict.fromkeys(duplicates))
            raise ValueError(
                f"Tache(s) dupliquee(s) dans le plan: {identifiers}."
            )
        return normalized

    @staticmethod
    def _plan_hash(
        tasks: Iterable[PlannedTask],
        *,
        options_payload: Mapping[str, Any],
        routing_facts: RoutingFacts,
        source_fingerprint: Mapping[str, Any] | None = None,
    ) -> str:
        task_list = list(tasks)
        if not task_list:
            return ""

        from .catalog import TASKS

        task_payload = []
        phases: set[str] = set()
        for task in task_list:
            spec = TASKS.get(task.id)
            version = spec.version if spec is not None else "legacy"
            phase = spec.phase if spec is not None else "unknown"
            phases.add(phase)
            task_payload.append(
                {
                    "id": task.id,
                    "reason": task.reason,
                    "version": version,
                    "handler": spec.entrypoint if spec is not None else None,
                }
            )

        plan_options = dict(options_payload)
        # ``force`` pilote seulement cette invocation; il ne change pas le
        # contenu attendu du plan ni l'identité de ses checkpoints.
        plan_options.pop("force", None)
        payload: dict[str, Any] = {
            "tasks": task_payload,
            "options": plan_options,
            "source": dict(source_fingerprint or {}),
        }
        # Les faits evoluent pendant l'inspection; ils ne doivent donc pas
        # invalider sa reprise. Ils font en revanche partie du contrat d'un
        # plan de traitement.
        if phases != {"inspection"}:
            payload["routing_facts"] = {
                "has_subtitles": routing_facts.has_subtitles,
                "video_type": (
                    routing_facts.video_type.value
                    if routing_facts.video_type is not None
                    else None
                ),
            }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def compute_plan_hash(
        self,
        tasks: Iterable[PlannedTask] | None = None,
    ) -> str:
        return self._plan_hash(
            self.plan if tasks is None else tasks,
            options_payload=self.options.to_dict(),
            routing_facts=self.routing_facts,
            source_fingerprint=self.source_fingerprint,
        )

    @staticmethod
    def _source_fingerprint(path: Path) -> dict[str, int]:
        try:
            stat = path.stat()
        except OSError:
            return {}
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    @property
    def source_fingerprint(self) -> dict[str, int]:
        return self._source_fingerprint(self.video_path)

    @property
    def force_rebuild(self) -> bool:
        """Vrai quand l'exécuteur exige une reconstruction réelle de la tâche."""

        return self.options.force or self._task_force

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
        return self.routing_facts.has_subtitles

    @property
    def video_type(self) -> str | None:
        value = self.routing_facts.video_type
        return value.value if value is not None else None

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
        return (
            self.transcript_strategy is not None
            and self.video_type is not None
        )

    def routing(self) -> dict[str, Any]:
        missing = []
        if self.has_subtitles is None:
            missing.append("has_subtitles")
        if self.video_type is None:
            missing.append("video_type")
        parts = [
            self.chunk_strategy,
            self.transcript_strategy,
            self.video_type,
        ]
        return {
            "status": "ready" if not missing else "needs_content_inspection",
            "pipeline_id": ".".join(part for part in parts if part),
            "transcript_strategy": self.transcript_strategy,
            "chunk_strategy": self.chunk_strategy,
            "visual_strategy": self.video_type,
            "missing_facts": missing,
        }

    def set_plan(
        self,
        tasks: Iterable[PlannedTask | Mapping[str, Any]],
    ) -> None:
        # La validation contre le registre a lieu juste avant execution.
        # Cela permet aussi aux tests/extensions d'injecter temporairement une
        # TaskSpec dans TASKS apres avoir construit leur contexte.
        normalized = self._coerce_plan(tasks, validate=False)
        plan_hash = self.compute_plan_hash(normalized)
        task_ids = [task.id for task in normalized]

        if self.execution.plan_hash != plan_hash:
            if self.execution.has_activity and all(
                item.run_id != self.execution.run_id
                for item in self.execution_history
            ):
                self.execution_history.append(self.execution)
            matching_index = next(
                (
                    index
                    for index in range(len(self.execution_history) - 1, -1, -1)
                    if self.execution_history[index].plan_hash == plan_hash
                ),
                None,
            )
            if matching_index is None:
                self.execution = RunExecution.for_plan(plan_hash, task_ids)
            else:
                self.execution = self.execution_history.pop(matching_index)
                self.execution.ensure_tasks(task_ids)
        else:
            self.execution.ensure_tasks(task_ids)
        self.plan = normalized

    def _artifact_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.video_dir / candidate
        return candidate

    def artifact_paths(self, task_id: str) -> list[Path]:
        return [
            self._artifact_path(value)
            for value in self.artifacts.by_task.get(task_id, [])
        ]

    def artifacts_valid(self, task_id: str) -> bool:
        paths = self.artifact_paths(task_id)

        def is_valid(path: Path) -> bool:
            try:
                if path.is_file():
                    return True
                if path.is_dir():
                    return any(
                        candidate.is_file()
                        for candidate in path.rglob("*")
                    )
                return False
            except OSError:
                return False

        return bool(paths) and all(is_valid(path) for path in paths)

    def artifact_fingerprint(self, task_id: str) -> str | None:
        """Empreinte les sorties afin de détecter un checkpoint écrasé."""

        try:
            return self._compute_artifact_fingerprint(task_id)
        except OSError:
            return None

    def _compute_artifact_fingerprint(self, task_id: str) -> str | None:
        paths = self.artifact_paths(task_id)
        if not paths or not self.artifacts_valid(task_id):
            return None

        media_suffixes = {
            *VIDEO_EXTENSIONS,
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".wav",
            ".mp3",
            ".m4a",
        }
        entries: list[dict[str, Any]] = []
        for path in sorted(paths, key=lambda item: item.as_posix()):
            resolved = path.resolve()
            if resolved.is_file():
                digest = hashlib.sha256()
                with resolved.open("rb") as handle:
                    for chunk in iter(
                        lambda: handle.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)
                stat = resolved.stat()
                entries.append(
                    {
                        "path": resolved.as_posix(),
                        "size": stat.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
                continue

            children = sorted(
                (
                    candidate
                    for candidate in resolved.rglob("*")
                    if candidate.is_file()
                ),
                key=lambda item: item.relative_to(resolved).as_posix(),
            )
            media_children = [
                child
                for child in children
                if child.suffix.lower() in media_suffixes
            ]
            for child in media_children or children:
                stat = child.stat()
                entries.append(
                    {
                        "path": (
                            resolved.as_posix()
                            + "/"
                            + child.relative_to(resolved).as_posix()
                        ),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )

        canonical = json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def record_artifacts(
        self,
        task_id: str,
        result: Any,
        *,
        replace: bool = False,
    ) -> list[str]:
        paths: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, TaskResult):
                collect(value.artifacts)
                if not value.artifacts:
                    collect(value.value)
                return
            if isinstance(value, Path):
                candidate = value
            elif isinstance(value, str):
                candidate = self._artifact_path(value)
            else:
                candidate = None

            if candidate is not None:
                if not candidate.exists():
                    return
                try:
                    relative = candidate.resolve().relative_to(
                        self.video_dir.resolve()
                    )
                    paths.append(relative.as_posix())
                except ValueError:
                    paths.append(candidate.resolve().as_posix())
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        collect(result)
        normalized = list(dict.fromkeys(paths))
        if normalized:
            self.artifacts.by_task[task_id] = normalized
        elif replace:
            self.artifacts.by_task.pop(task_id, None)
        return normalized

    def apply_task_result(self, task_id: str, result: TaskResult) -> None:
        explicit_source: Any = result.artifacts or result.value
        if explicit_source is not None:
            recorded = self.record_artifacts(
                task_id,
                explicit_source,
                replace=bool(result.artifacts),
            )
            if result.artifacts and not recorded:
                self.artifacts.by_task.pop(task_id, None)
        if result.status in {TaskStatus.BLOCKED, TaskStatus.FAILED}:
            if not self.artifacts_valid(task_id):
                self.artifacts.by_task.pop(task_id, None)
