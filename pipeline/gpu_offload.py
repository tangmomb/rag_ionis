"""Execution des taches GPU sur Modal, le reste du pipeline restant local.

Avec ``PIPELINE_EXECUTION_BACKEND=modal``, l'executeur tourne sur la machine
courante (le VPS) et seules les taches ``requires_gpu`` partent sur Modal :
leurs fichiers d'entree sont envoyes dans une archive, le handler habituel s'y
execute sur un GPU, et ses fichiers de sortie reviennent dans le dossier video.

La fonction Modal ne lit ni S3, ni YouTube, ni OpenAI : elle ne recoit que des
fichiers et n'a besoin d'aucun secret.
"""

from __future__ import annotations

import io
import os
import shutil
import tarfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import TaskResult, TaskStatus


DEFAULT_APP_NAME = "rag-ionis-gpu"
FUNCTION_NAME = "run_gpu_task"


class GpuOffloadError(RuntimeError):
    """Raised when a GPU task cannot run on Modal."""


EXECUTION_BACKENDS = ("local", "modal")


def execution_backend() -> str:
    """``local``: everything here; ``modal``: GPU tasks on Modal, the rest here."""
    backend = os.getenv("PIPELINE_EXECUTION_BACKEND", "local").strip().lower() or "local"
    if backend not in EXECUTION_BACKENDS:
        raise RuntimeError("PIPELINE_EXECUTION_BACKEND doit valoir 'local' ou 'modal'.")
    return backend


def gpu_offload_enabled() -> bool:
    if os.getenv("PIPELINE_GPU_WORKER", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return execution_backend() == "modal"


# --------------------------------------------------------------------------
# Files exchanged per task, relative to the video directory.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuTaskFiles:
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    excluded_outputs: tuple[Path, ...] = ()


def _images_dir(video_path: Path) -> Path:
    from .support.paths import existing_images_dir

    return existing_images_dir(video_path)


def task_files(task_id: str, context) -> GpuTaskFiles:
    """Absolute input and output paths of one GPU task for this video."""
    video_path = Path(context.video_path)
    if task_id == "frames.classify":
        images = _images_dir(video_path)
        # Classification moves every frame into a class folder.
        return GpuTaskFiles(inputs=(images,), outputs=(images,))
    if task_id == "ocr.extract_raw":
        from .support.paths import ocr_raw_dir

        return GpuTaskFiles(inputs=(_images_dir(video_path),), outputs=(ocr_raw_dir(video_path),))
    if task_id == "transcript.whisper":
        from .support.paths import transcripts_dir

        directory = transcripts_dir(video_path, name=context.transcripts_dir_name)
        audio = directory / "audio"
        return GpuTaskFiles(inputs=(audio,), outputs=(directory,), excluded_outputs=(audio,))
    raise GpuOffloadError(f"Tache GPU sans echange de fichiers defini: {task_id!r}")


def prepare_inputs(task_id: str, context) -> None:
    """Build locally whatever input the GPU does not need to compute."""
    if task_id != "transcript.whisper":
        return
    from .steps.transcripts.transcribe_with_whisper import extract_audio

    audio_dir = task_files(task_id, context).inputs[0]
    audio_dir.mkdir(parents=True, exist_ok=True)
    # Sending a 16 kHz mono MP3 instead of the video keeps the upload small.
    extract_audio(Path(context.video_path), audio_dir)


# --------------------------------------------------------------------------
# Archives and serialisation, shared by both sides.
# --------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def pack(root: Path, paths: Iterable[Path], *, excluded: Iterable[Path] = ()) -> bytes:
    root = Path(root)
    excluded_set = {Path(path).resolve() for path in excluded}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            candidates = [path] if path.is_file() else sorted(path.rglob("*"))
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if any(resolved == item or item in resolved.parents for item in excluded_set):
                    continue
                archive.add(candidate, arcname=_relative(candidate, root), recursive=False)
    return buffer.getvalue()


def unpack(data: bytes, root: Path) -> int:
    root = Path(root).resolve()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if (
                not member.isfile()
                or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                continue
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            count += 1
    return count


def encode_value(value: Any, root: Path) -> Any:
    """JSON-friendly copy of a handler value; paths become relative markers."""
    if isinstance(value, Path):
        try:
            return {"__path__": _relative(value, root)}
        except ValueError:
            return {"__path__": None}
    if isinstance(value, dict):
        return {str(key): encode_value(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [encode_value(item, root) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def decode_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__path__"}:
            relative = value["__path__"]
            return root.joinpath(*PurePosixPath(relative).parts) if relative else None
        return {key: decode_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item, root) for item in value]
    return value


# --------------------------------------------------------------------------
# Remote side: runs inside the Modal container.
# --------------------------------------------------------------------------


@dataclass
class RemoteTaskContext:
    """The few context attributes the GPU handlers read."""

    video_path: Path
    force_rebuild: bool
    transcripts_dir_name: str
    video_type: str | None


def run_task_in_directory(
    task_id: str,
    work_dir: Path,
    video_name: str,
    archive: bytes,
    task_context: dict,
) -> dict:
    """Unpack the inputs, run the task handler, return its result and outputs."""
    from .catalog import TASKS

    spec = TASKS[task_id]
    if not spec.requires_gpu:
        raise GpuOffloadError(f"{task_id!r} n'est pas une tache GPU.")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    unpack(archive, work_dir)
    video_path = work_dir / video_name
    if not video_path.exists():
        # The GPU tasks only use the video path to locate their folders.
        video_path.touch()
    context = RemoteTaskContext(
        video_path=video_path,
        force_rebuild=bool(task_context.get("force_rebuild", True)),
        transcripts_dir_name=str(task_context["transcripts_dir_name"]),
        video_type=task_context.get("video_type"),
    )
    result = spec.handler(context)
    files = task_files(task_id, context)
    return {
        "status": result.status.value,
        "reason": result.reason,
        "artifacts": [
            _relative(Path(path), work_dir)
            for path in result.artifacts
            if Path(path).resolve().is_relative_to(work_dir.resolve())
        ],
        "value": encode_value(result.value, work_dir),
        "archive": pack(work_dir, files.outputs, excluded=files.excluded_outputs),
    }


# --------------------------------------------------------------------------
# Local side: called by the executor instead of the handler.
# --------------------------------------------------------------------------


def _modal_function():
    try:
        import modal
    except ImportError as error:
        raise GpuOffloadError(
            "Le paquet 'modal' est requis pour PIPELINE_EXECUTION_BACKEND=modal."
        ) from error
    options = {}
    environment = os.getenv("MODAL_ENVIRONMENT", "").strip()
    if environment:
        options["environment_name"] = environment
    app_name = os.getenv("MODAL_APP_NAME", DEFAULT_APP_NAME).strip() or DEFAULT_APP_NAME
    return modal.Function.from_name(app_name, FUNCTION_NAME, **options)


def _is_wait_timeout(error: BaseException) -> bool:
    if type(error).__name__ == "FunctionTimeoutError":
        return False
    if isinstance(error, TimeoutError):
        return True
    try:
        from modal.exception import TimeoutError as ModalTimeoutError
    except ImportError:
        return False
    return isinstance(error, ModalTimeoutError)


def call_remote(function, arguments: tuple, *, label: str, clock=time.monotonic) -> dict:
    """Spawn the Modal call and wait for it, cancelling it on any failure."""
    poll_seconds = max(1.0, float(os.getenv("MODAL_POLL_SECONDS", "60")))
    timeout_seconds = float(os.getenv("MODAL_TASK_TIMEOUT_SECONDS", "10800"))
    call = function.spawn(*arguments)
    deadline = clock() + timeout_seconds
    try:
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise GpuOffloadError(
                    f"Delai depasse pour {label} sur Modal ({timeout_seconds:g} s)."
                )
            wait_seconds = min(poll_seconds, remaining)
            started = clock()
            try:
                output = call.get(timeout=wait_seconds)
            except Exception as error:
                # A remote TimeoutError is re-raised at once; a wait timeout
                # uses the whole delay.
                if _is_wait_timeout(error) and clock() - started >= wait_seconds * 0.5:
                    print(f"[modal] {label} toujours en cours", flush=True)
                    continue
                raise GpuOffloadError(
                    f"{label} en echec sur Modal: {type(error).__name__}: {error}"
                ) from error
            break
    except BaseException:
        try:
            call.cancel()
        except Exception as cancel_error:
            print(f"[modal] avertissement annulation {label}: {cancel_error}", flush=True)
        raise
    if not isinstance(output, dict) or "status" not in output:
        raise GpuOffloadError(f"{label}: reponse Modal invalide.")
    return output


def _replace_outputs(files: GpuTaskFiles, video_dir: Path, archive: bytes) -> int:
    for output in files.outputs:
        if not output.exists():
            continue
        if output.is_file():
            output.unlink()
            continue
        kept = [item for item in files.excluded_outputs if output in item.parents]
        for child in output.iterdir():
            if child in kept:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    return unpack(archive, video_dir)


def run_task_remotely(spec, context, *, function=None) -> TaskResult:
    task_id = spec.id
    video_path = Path(context.video_path)
    video_dir = video_path.parent
    prepare_inputs(task_id, context)
    files = task_files(task_id, context)
    archive = pack(video_dir, files.inputs)
    video_type = getattr(context, "video_type", None)
    task_context = {
        "force_rebuild": bool(context.force_rebuild),
        "transcripts_dir_name": context.transcripts_dir_name,
        "video_type": getattr(video_type, "value", video_type),
    }
    label = f"{task_id} ({video_dir.name})"
    print(
        f"[modal] {label}: envoi de {len(archive) / 1e6:.1f} Mo",
        flush=True,
    )
    started = time.monotonic()
    output = call_remote(
        function or _modal_function(),
        (task_id, video_dir.name, video_path.name, archive, task_context),
        label=label,
    )
    status = TaskStatus.from_value(output["status"])
    returned = output.get("archive") or b""
    if status.is_successful and returned:
        restored = _replace_outputs(files, video_dir, returned)
    else:
        restored = 0
    print(
        f"[modal] {label}: {status.value} en {time.monotonic() - started:.0f} s, "
        f"{restored} fichier(s) recu(s)",
        flush=True,
    )
    reason = output.get("reason")
    artifacts = tuple(
        video_dir.joinpath(*PurePosixPath(item).parts) for item in output.get("artifacts") or ()
    )
    value = decode_value(output.get("value"), video_dir)
    if status is TaskStatus.CACHED:
        return TaskResult.cached(artifacts=artifacts, reason=reason, value=value)
    if status is TaskStatus.SUCCEEDED:
        return TaskResult.succeeded(value, artifacts=artifacts, reason=reason)
    return TaskResult(status, artifacts, reason, value)


def run_handler(spec, context) -> TaskResult:
    """Executor entry point: GPU tasks go to Modal when it is enabled."""
    if spec.requires_gpu and gpu_offload_enabled():
        return run_task_remotely(spec, context)
    return spec.handler(context)
