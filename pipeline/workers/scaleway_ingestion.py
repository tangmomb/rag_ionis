from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

from pipeline.ingest.download_videos import download_video
from pipeline.ingest.fetch_youtube_metadata import video_info_payload
from pipeline.publish.upload_outputs_to_s3 import (
    delete_prefix,
)
from pipeline.support.json_io import write_json
from pipeline.support.paths import youtube_api_infos_path, youtube_comments_path
from pipeline.support.scaleway_jobs import (
    download_s3_prefix,
    s3_client,
    upload_s3_directory,
)
from botocore.exceptions import ClientError


SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".mov", ".m4v"})
_GPU_VALIDATED = False


def required_text(payload: dict, key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Champ Scaleway requis manquant: {key}")
    return value


def safe_s3_prefix(value: object, label: str) -> str:
    prefix = str(value or "").strip().strip("/")
    if not prefix or "\\" in prefix or any(
        part in {"", ".", ".."} for part in prefix.split("/")
    ):
        raise ValueError(f"Prefixe S3 Scaleway invalide pour {label}.")
    return prefix


def validate_worker_gpu() -> None:
    global _GPU_VALIDATED
    if _GPU_VALIDATED:
        return
    enforce = os.getenv("SCALEWAY_ENFORCE_GPU", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not enforce:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Le worker Scaleway ne voit aucun GPU CUDA.")
    properties = torch.cuda.get_device_properties(0)
    memory_gb = properties.total_memory / (1024**3)
    required_name = os.getenv("SCALEWAY_REQUIRED_GPU_NAME", "L40S").strip().lower()
    minimum_memory_gb = float(os.getenv("SCALEWAY_MIN_GPU_MEMORY_GB", "44"))
    if required_name and required_name not in properties.name.lower():
        raise RuntimeError(
            f"GPU Scaleway inattendu: {properties.name}; {required_name!r} requis."
        )
    if memory_gb < minimum_memory_gb:
        raise RuntimeError(
            f"VRAM Scaleway insuffisante: {memory_gb:.1f} Go; "
            f"minimum {minimum_memory_gb:g} Go."
        )
    print(
        f"[scaleway] GPU valide: {properties.name} ({memory_gb:.1f} Go)",
        flush=True,
    )
    _GPU_VALIDATED = True


def run_gpu_healthcheck() -> dict:
    import paddle
    import torch
    import whisperx  # noqa: F401

    properties = torch.cuda.get_device_properties(0)
    torch_input = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    torch_result = torch_input @ torch_input
    torch.cuda.synchronize()

    paddle.set_device("gpu:0")
    paddle_input = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
    paddle_result = float((paddle_input * paddle_input).sum().numpy())
    return {
        "operation": "healthcheck",
        "status": "completed",
        "gpu_name": properties.name,
        "gpu_memory_gib": round(properties.total_memory / (1024**3), 1),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_result_finite": bool(torch.isfinite(torch_result).all().item()),
        "paddle_version": paddle.__version__,
        "paddle_cuda": bool(paddle.device.is_compiled_with_cuda()),
        "paddle_result": paddle_result,
        "whisperx_imported": True,
    }


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
        raise RuntimeError(f"Video Scaleway introuvable: {video_id}")
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
            raise ValueError(f"Tache Scaleway inconnue: {task_id!r}")
        context = PipelineContext.inspect(video_path, options)
        context.set_plan([PlannedTask(task_id, "scaleway_manual")])
        write_manifest(context)
        execute_tasks(context)
    else:
        raise ValueError(f"Commande pipeline Scaleway non supportee: {command!r}")

    if include_embeddings and not (
        command == "task" and task_id == "embeddings.create"
    ):
        context = PipelineContext.inspect(video_path, options)
        context.set_plan([PlannedTask("embeddings.create", "scaleway_finalize")])
        write_manifest(context)
        execute_tasks(context)


def process_job(job_input: dict, *, object_store_client=None) -> dict:
    validate_worker_gpu()
    operation = str(job_input.get("operation") or "pipeline_command").strip()
    if operation == "healthcheck":
        return run_gpu_healthcheck()
    if operation != "pipeline_command":
        raise ValueError(f"Operation Scaleway non supportee: {operation!r}")
    source = job_input.get("source")
    if not isinstance(source, dict):
        source = {"type": "youtube", "video": job_input.get("video")}
    source_type = str(source.get("type") or "").strip().lower()
    video = source.get("video") or job_input.get("video")
    if source_type == "youtube" and not isinstance(video, dict):
        raise ValueError("Le payload Scaleway doit contenir l'objet video YouTube.")
    video_id = required_text(
        video if isinstance(video, dict) else job_input,
        "id" if isinstance(video, dict) else "video_id",
    )
    archive_name = required_text(job_input, "archive_name")
    if not SAFE_NAME_PATTERN.fullmatch(video_id):
        raise ValueError("Identifiant video Scaleway invalide.")
    if not SAFE_NAME_PATTERN.fullmatch(archive_name):
        raise ValueError("Nom d'archive Scaleway invalide.")

    configured_bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("S3_REGION", "").strip()
    if not configured_bucket:
        raise RuntimeError("S3_BUCKET_NAME manquant sur le worker Scaleway.")
    result = job_input.get("result")
    if not isinstance(result, dict):
        result = {
            "bucket": configured_bucket,
            "prefix": f"youtube/{archive_name}/{video_id}",
        }
    bucket = str(result.get("bucket") or configured_bucket).strip()
    if bucket != configured_bucket:
        raise ValueError("Le bucket resultat Scaleway ne correspond pas au bucket configure.")
    prefix = safe_s3_prefix(result.get("prefix"), "resultat")
    client = object_store_client or s3_client(region)

    with tempfile.TemporaryDirectory(prefix="rag-ionis-scaleway-") as temporary_dir:
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
                raise ValueError("Le bucket source Scaleway ne correspond pas au bucket configure.")
            source_prefix = safe_s3_prefix(source.get("prefix"), "source")
            download_s3_prefix(
                client,
                source_bucket,
                source_prefix,
                video_dir,
            )
        else:
            raise ValueError(f"Source Scaleway non supportee: {source_type!r}")

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


def write_job_status(job_input: dict, payload: dict, *, object_store_client=None) -> None:
    control = job_input.get("control")
    if not isinstance(control, dict):
        raise ValueError("Canal de controle Scaleway manquant.")
    bucket = required_text(control, "bucket")
    configured_bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if configured_bucket and bucket != configured_bucket:
        raise ValueError("Le bucket de controle Scaleway est invalide.")
    status_key = safe_s3_prefix(control.get("status_key"), "statut")
    client = object_store_client or s3_client(os.getenv("S3_REGION", "").strip())
    client.put_object(
        Bucket=bucket,
        Key=status_key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def _read_job_status(client, bucket: str, status_key: str) -> dict | None:
    try:
        response = client.get_object(Bucket=bucket, Key=status_key)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def find_pending_job(client, bucket: str, root_prefix: str) -> dict | None:
    """Return the oldest queued job found below the worker queue prefix."""
    normalized_root = safe_s3_prefix(root_prefix, "file d'attente")
    paginator = client.get_paginator("list_objects_v2")
    candidates: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized_root}/"):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if key.endswith("/job.json"):
                candidates.append(key)

    for job_key in sorted(candidates):
        response = client.get_object(Bucket=bucket, Key=job_key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        if not isinstance(payload, dict):
            continue
        control = payload.get("control")
        if not isinstance(control, dict):
            continue
        status_key = safe_s3_prefix(control.get("status_key"), "statut")
        status = _read_job_status(client, bucket, status_key)
        status_name = str((status or {}).get("status") or "queued").lower()
        if status_name in {"completed", "failed"}:
            continue
        return payload
    return None


def poll_jobs() -> None:
    """Process S3 jobs until the dedicated VM has been idle long enough."""
    validate_worker_gpu()
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME manquant sur le worker Scaleway.")
    root_prefix = os.getenv("SCALEWAY_S3_JOB_PREFIX", "scaleway/jobs").strip()
    poll_seconds = max(
        0.5, float(os.getenv("SCALEWAY_WORKER_POLL_SECONDS", "5"))
    )
    idle_seconds = max(
        poll_seconds, float(os.getenv("SCALEWAY_WORKER_IDLE_SECONDS", "30"))
    )
    client = s3_client(os.getenv("S3_REGION", "").strip())
    last_activity = time.monotonic()

    print(
        f"[scaleway] worker persistant en attente dans s3://{bucket}/{root_prefix}/",
        flush=True,
    )
    while True:
        job_input = find_pending_job(client, bucket, root_prefix)
        if job_input is not None:
            control = job_input.get("control")
            job_id = str((control or {}).get("job_id") or "inconnu")
            write_job_status(
                job_input,
                {"status": "running", "job_id": job_id},
                object_store_client=client,
            )
            try:
                output = process_job(job_input, object_store_client=client)
            except Exception as error:
                write_job_status(
                    job_input,
                    {
                        "status": "failed",
                        "job_id": job_id,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                    object_store_client=client,
                )
                print(f"[scaleway] job en echec: {job_id}", flush=True)
            else:
                write_job_status(
                    job_input,
                    {"status": "completed", "job_id": job_id, "output": output},
                    object_store_client=client,
                )
                print(f"[scaleway] job termine: {job_id}", flush=True)
            last_activity = time.monotonic()
            continue

        if time.monotonic() - last_activity >= idle_seconds:
            print("[scaleway] file d'attente vide, arret du worker", flush=True)
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker GPU Scaleway")
    parser.add_argument("--job-file", type=Path)
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Surveille la file S3 et traite les jobs de la VM dediee.",
    )
    args = parser.parse_args()
    if args.poll:
        poll_jobs()
        return
    if args.job_file is None:
        parser.error("--job-file est requis sauf avec --poll")
    job_input = json.loads(args.job_file.read_text(encoding="utf-8"))
    if not isinstance(job_input, dict):
        raise ValueError("Entree de job Scaleway invalide.")
    try:
        output = process_job(job_input)
    except Exception as error:
        write_job_status(
            job_input,
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    write_job_status(job_input, {"status": "completed", "output": output})


if __name__ == "__main__":
    main()

