from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.ingest.download_videos import download_video
from pipeline.ingest.fetch_youtube_metadata import video_info_payload
from pipeline.publish.upload_outputs_to_s3 import (
    delete_prefix,
    s3_client,
    upload_directory,
)
from pipeline.support.json_io import write_json
from pipeline.support.paths import youtube_api_infos_path, youtube_comments_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def required_text(payload: dict, key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Champ Runpod requis manquant: {key}")
    return value


def run_pipeline(video_id: str, archive_dir: Path) -> None:
    commands = (
        [sys.executable, "-m", "pipeline", "run", video_id, "--root", str(archive_dir)],
        [
            sys.executable,
            "-m",
            "pipeline",
            "task",
            "embeddings.create",
            video_id,
            "--root",
            str(archive_dir),
        ],
    )
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def process_job(job_input: dict, *, object_store_client=None) -> dict:
    video = job_input.get("video")
    if not isinstance(video, dict):
        raise ValueError("Le payload Runpod doit contenir l'objet video.")
    video_id = required_text(video, "id")
    archive_name = required_text(job_input, "archive_name")
    if not SAFE_NAME_PATTERN.fullmatch(video_id):
        raise ValueError("Identifiant video Runpod invalide.")
    if not SAFE_NAME_PATTERN.fullmatch(archive_name):
        raise ValueError("Nom d'archive Runpod invalide.")

    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("S3_REGION", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME manquant sur le worker Runpod.")
    prefix = f"youtube/{archive_name}/{video_id}"

    with tempfile.TemporaryDirectory(prefix="rag-ionis-runpod-") as temporary_dir:
        download_root = Path(temporary_dir)
        archive_dir = download_root / archive_name
        video_dir = archive_dir / video_id
        video_dir.mkdir(parents=True)
        write_json(youtube_api_infos_path(video_dir), video_info_payload(video))
        comments = job_input.get("comments")
        if isinstance(comments, dict):
            write_json(youtube_comments_path(video_dir), comments)

        metadata = video_info_payload(video)
        download_video(
            (video_id, metadata["title"], metadata["url"], metadata),
            archive_dir,
            download_root,
            reuse_previous=False,
            sync_cached_metadata=False,
        )
        run_pipeline(video_id, archive_dir.resolve())

        client = object_store_client or s3_client(region)
        # The archive name is unique per daily run. Clearing this one-video prefix
        # makes an explicit retry deterministic without touching other archives.
        delete_prefix(client, bucket, prefix)
        upload_directory(
            client,
            bucket,
            archive_dir,
            prefix=f"youtube/{archive_name}",
            force=True,
            video_ids=[video_id],
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

