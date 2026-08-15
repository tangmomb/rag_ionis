from __future__ import annotations

import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path

from pipeline.ingest.download_videos import download_video
from pipeline.ingest.fetch_youtube_metadata import video_info_payload
from pipeline.publish.upload_outputs_to_s3 import (
    delete_prefix,
    s3_client,
)
from pipeline.support.json_io import write_json
from pipeline.support.paths import youtube_api_infos_path, youtube_comments_path
from pipeline.support.runpod_jobs import download_s3_prefix, upload_s3_directory


SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".mov", ".m4v"})
_GPU_VALIDATED = False


def required_text(payload: dict, key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Champ Runpod requis manquant: {key}")
    return value


def safe_s3_prefix(value: object, label: str) -> str:
    prefix = str(value or "").strip().strip("/")
    if not prefix or "\\" in prefix or any(
        part in {"", ".", ".."} for part in prefix.split("/")
    ):
        raise ValueError(f"Prefixe S3 Runpod invalide pour {label}.")
    return prefix


def validate_worker_gpu() -> None:
    global _GPU_VALIDATED
    if _GPU_VALIDATED:
        return
    enforce = os.getenv("RUNPOD_ENFORCE_GPU", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not enforce:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Le worker Runpod ne voit aucun GPU CUDA.")
    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / (1024**3)
    required_name = os.getenv("RUNPOD_REQUIRED_GPU_NAME", "4090").strip().lower()
    minimum_memory_gb = float(os.getenv("RUNPOD_MIN_GPU_MEMORY_GB", "20"))
    if required_name and required_name not in properties.name.lower():
        raise RuntimeError(
            f"GPU Runpod inattendu: {properties.name}; {required_name!r} requis."
        )
    if memory_gb < minimum_memory_gb:
        raise RuntimeError(
            f"VRAM Runpod insuffisante: {memory_gb:.1f} Go; "
            f"minimum {minimum_memory_gb:g} Go."
        )
    print(
        f"[runpod] GPU valide: {properties.name} ({memory_gb:.1f} Go)",
        flush=True,
    )
    _GPU_VALIDATED = True


def run_pipeline(
    video_id: str,
    archive_dir: Path,
    *,
    command: str = "run",
    task_id: str | None = None,
    options_payload: dict | None = None,
    skip_inspection: bool = False,
    include_embeddings: bool = False,
) -> None:
    from pipeline.catalog import TASKS
    from pipeline.context import PipelineContext
    from pipeline.contracts import PlannedTask
    from pipeline.discovery import select_videos
    from pipeline.executor import execute_tasks
    from pipeline.manifest import write_manifest
    from pipeline.options import PipelineOptions
    from pipeline.orchestrator import inspect_video, run_video

    base_options = PipelineOptions.from_dict(options_payload or {})
    options = replace(
        base_options,
        force=bool((options_payload or {}).get("force", False)),
    )
    videos = select_videos(archive_dir, video_id)
    if len(videos) != 1:
        raise RuntimeError(f"Video Runpod introuvable: {video_id}")
    video_path = videos[0]

    if command == "run":
        run_video(
            video_path,
            options,
            skip_inspection=skip_inspection,
        )
    elif command == "inspect":
        inspect_video(video_path, options)
    elif command == "task":
        if task_id not in TASKS:
            raise ValueError(f"Tache Runpod inconnue: {task_id!r}")
        context = PipelineContext.inspect(video_path, options)
        context.set_plan([PlannedTask(task_id, "runpod_manual")])
        write_manifest(context)
        execute_tasks(context)
    else:
        raise ValueError(f"Commande pipeline Runpod non supportee: {command!r}")

    if include_embeddings and not (
        command == "task" and task_id == "embeddings.create"
    ):
        context = PipelineContext.inspect(video_path, options)
        context.set_plan([PlannedTask("embeddings.create", "runpod_finalize")])
        write_manifest(context)
        execute_tasks(context)


def process_job(job_input: dict, *, object_store_client=None) -> dict:
    validate_worker_gpu()
    operation = str(job_input.get("operation") or "pipeline_command").strip()
    if operation != "pipeline_command":
        raise ValueError(f"Operation Runpod non supportee: {operation!r}")
    source = job_input.get("source")
    if not isinstance(source, dict):
        # Compatibilite avec les premiers jobs daily sync.
        source = {"type": "youtube", "video": job_input.get("video")}
    source_type = str(source.get("type") or "").strip().lower()
    video = source.get("video") or job_input.get("video")
    if source_type == "youtube" and not isinstance(video, dict):
        raise ValueError("Le payload Runpod doit contenir l'objet video YouTube.")
    video_id = required_text(
        video if isinstance(video, dict) else job_input,
        "id" if isinstance(video, dict) else "video_id",
    )
    archive_name = required_text(job_input, "archive_name")
    if not SAFE_NAME_PATTERN.fullmatch(video_id):
        raise ValueError("Identifiant video Runpod invalide.")
    if not SAFE_NAME_PATTERN.fullmatch(archive_name):
        raise ValueError("Nom d'archive Runpod invalide.")

    configured_bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("S3_REGION", "").strip()
    if not configured_bucket:
        raise RuntimeError("S3_BUCKET_NAME manquant sur le worker Runpod.")
    result = job_input.get("result")
    if not isinstance(result, dict):
        result = {
            "bucket": configured_bucket,
            "prefix": f"youtube/{archive_name}/{video_id}",
        }
    bucket = str(result.get("bucket") or configured_bucket).strip()
    if bucket != configured_bucket:
        raise ValueError("Le bucket resultat Runpod ne correspond pas au bucket configure.")
    prefix = safe_s3_prefix(result.get("prefix"), "resultat")
    client = object_store_client or s3_client(region)

    with tempfile.TemporaryDirectory(prefix="rag-ionis-runpod-") as temporary_dir:
        download_root = Path(temporary_dir)
        archive_dir = download_root / archive_name
        video_dir = archive_dir / video_id
        video_dir.mkdir(parents=True)
        if source_type == "youtube":
            metadata = video_info_payload(video)
            write_json(youtube_api_infos_path(video_dir), metadata)
            comments = source.get("comments") or job_input.get("comments")
            if isinstance(comments, dict):
                write_json(youtube_comments_path(video_dir), comments)
            download_video(
                (video_id, metadata["title"], metadata["url"], metadata),
                archive_dir,
                download_root,
                reuse_previous=False,
                sync_cached_metadata=False,
            )
        elif source_type == "s3":
            source_bucket = str(source.get("bucket") or "").strip()
            if source_bucket != configured_bucket:
                raise ValueError("Le bucket source Runpod ne correspond pas au bucket configure.")
            source_prefix = safe_s3_prefix(source.get("prefix"), "source")
            download_s3_prefix(
                client,
                source_bucket,
                source_prefix,
                video_dir,
            )
        else:
            raise ValueError(f"Source Runpod non supportee: {source_type!r}")

        run_pipeline(
            video_id,
            archive_dir.resolve(),
            command=str(job_input.get("command") or "run"),
            task_id=(str(job_input.get("task_id")) if job_input.get("task_id") else None),
            options_payload=(
                job_input.get("options")
                if isinstance(job_input.get("options"), dict)
                else None
            ),
            skip_inspection=bool(job_input.get("skip_inspection", False)),
            include_embeddings=bool(job_input.get("include_embeddings", True)),
        )

        delete_prefix(client, bucket, prefix)
        upload_s3_directory(
            client,
            bucket,
            video_dir,
            prefix,
            excluded_suffixes=VIDEO_SUFFIXES,
        )

    return {
        "video_id": video_id,
        "archive_name": archive_name,
        "bucket": bucket,
        "prefix": prefix,
        "status": "completed",
    }


def handler(job: dict) -> dict:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        raise ValueError("Entree de job Runpod invalide.")
    return process_job(job_input)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()

