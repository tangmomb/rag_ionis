from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipeline.catalog import TASKS
from pipeline.options import PipelineOptions
from pipeline.publish.upload_outputs_to_s3 import delete_prefix
from pipeline.support.runpod_jobs import (
    RunpodClient,
    RunpodConfig,
    RunpodJobError,
    download_s3_prefix,
    s3_client,
    upload_s3_directory,
)


GPU_TASK_IDS = frozenset(
    task_id for task_id, spec in TASKS.items() if spec.requires_gpu
)
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class RemoteCommand:
    command: str
    task_id: str | None = None
    skip_inspection: bool = False
    include_embeddings: bool = False


def execution_backend() -> str:
    if os.getenv("RUNPOD_WORKER", "").strip().lower() in {"1", "true", "yes"}:
        return "local"
    backend = os.getenv("PIPELINE_EXECUTION_BACKEND", "runpod").strip().lower()
    if backend not in {"local", "runpod"}:
        raise RuntimeError(
            "PIPELINE_EXECUTION_BACKEND doit valoir 'local' ou 'runpod'."
        )
    return backend


def should_delegate_to_runpod(
    command: str,
    *,
    task_id: str | None = None,
    dry_run: bool = False,
    probe_only: bool = False,
) -> bool:
    if execution_backend() != "runpod" or dry_run:
        return False
    if command == "run":
        return True
    if command == "inspect":
        return not probe_only
    if command == "task":
        return task_id in GPU_TASK_IDS
    return False


def positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} doit etre un entier positif.")
    return value


def remote_video_id(video_path: Path) -> str:
    candidate = video_path.parent.name or video_path.stem
    if SAFE_NAME_PATTERN.fullmatch(candidate):
        return candidate
    digest = hashlib.sha256(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"local-{digest}"


def _copy_remote_artifacts(downloaded_dir: Path, local_video_dir: Path) -> None:
    remote_outputs = downloaded_dir / "outputs"
    local_outputs = local_video_dir / "outputs"
    if remote_outputs.is_dir():
        if local_outputs.exists():
            shutil.rmtree(local_outputs)
        shutil.copytree(remote_outputs, local_outputs)

    remote_metadata = downloaded_dir / "metadata"
    local_metadata = local_video_dir / "metadata"
    if remote_metadata.is_dir():
        shutil.copytree(remote_metadata, local_metadata, dirs_exist_ok=True)


def run_video_on_runpod(
    video_path: Path,
    options: PipelineOptions,
    remote_command: RemoteCommand,
) -> dict[str, object]:
    video_path = Path(video_path).resolve()
    video_dir = video_path.parent
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("S3_REGION", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME est requis pour executer le pipeline sur Runpod.")

    request_id = uuid.uuid4().hex
    root_prefix = os.getenv("RUNPOD_S3_JOB_PREFIX", "runpod/jobs").strip("/")
    if not root_prefix:
        raise RuntimeError("RUNPOD_S3_JOB_PREFIX ne peut pas etre vide.")
    input_prefix = f"{root_prefix}/{request_id}/input"
    result_prefix = f"{root_prefix}/{request_id}/result"
    selected_video_id = remote_video_id(video_path)
    object_store = s3_client(region)

    uploaded = upload_s3_directory(
        object_store,
        bucket,
        video_dir,
        input_prefix,
        included_roots={video_path.name, "metadata", "outputs"},
    )
    print(
        f"[runpod] {uploaded} fichier(s) source envoye(s) pour {video_path.name}",
        flush=True,
    )
    job_input = {
        "schema_version": 2,
        "operation": "pipeline_command",
        "command": remote_command.command,
        "task_id": remote_command.task_id,
        "skip_inspection": remote_command.skip_inspection,
        "include_embeddings": remote_command.include_embeddings,
        "video_id": selected_video_id,
        "archive_name": f"cli-{request_id}",
        "options": options.to_dict(),
        "source": {
            "type": "s3",
            "bucket": bucket,
            "prefix": input_prefix,
        },
        "result": {
            "bucket": bucket,
            "prefix": result_prefix,
        },
    }

    succeeded = False
    try:
        client = RunpodClient(RunpodConfig.from_env())
        job_id, output = client.run(job_input)
        if str(output.get("video_id") or "") != selected_video_id:
            raise RunpodJobError(
                f"Le job Runpod {job_id} a renvoye un autre identifiant video."
            )
        if str(output.get("bucket") or "") != bucket:
            raise RunpodJobError(f"Le job Runpod {job_id} a renvoye un bucket invalide.")
        if str(output.get("prefix") or "").strip("/") != result_prefix:
            raise RunpodJobError(f"Le job Runpod {job_id} a renvoye un prefixe invalide.")

        with tempfile.TemporaryDirectory(prefix="rag-ionis-result-") as temporary_dir:
            downloaded_dir = Path(temporary_dir)
            downloaded = download_s3_prefix(
                object_store,
                bucket,
                result_prefix,
                downloaded_dir,
            )
            _copy_remote_artifacts(downloaded_dir, video_dir)
        succeeded = True
        print(
            f"[runpod] {downloaded} artefact(s) rapatrie(s) pour {video_path.name}",
            flush=True,
        )
        return {
            "backend": "runpod",
            "job_id": job_id,
            "video": str(video_path),
            "s3_uri": f"s3://{bucket}/{result_prefix}",
        }
    finally:
        keep = os.getenv("RUNPOD_KEEP_JOB_ARTIFACTS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if succeeded and not keep:
            delete_prefix(object_store, bucket, input_prefix)
            delete_prefix(object_store, bucket, result_prefix)


def run_videos_on_runpod(
    videos: list[Path],
    options: PipelineOptions,
    remote_command: RemoteCommand,
) -> list[dict[str, object]]:
    if not videos:
        return []
    max_workers = min(
        len(videos),
        positive_int_env("RUNPOD_MAX_CONCURRENT_JOBS", 4),
    )
    if max_workers == 1:
        return [run_video_on_runpod(videos[0], options, remote_command)]

    results: list[dict[str, object] | None] = [None] * len(videos)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="runpod-job") as pool:
        futures = {
            pool.submit(run_video_on_runpod, video, options, remote_command): index
            for index, video in enumerate(videos)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]
