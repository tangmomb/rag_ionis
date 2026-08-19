from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import psycopg
from pipeline.ingest.fetch_youtube_metadata import (
    CHANNEL,
    fetch_comments,
    fetch_videos,
    video_info_payload,
)
from pipeline.ingest.download_videos import download_video
from pipeline.publish.sync_database import (
    ensure_schema,
    sync_video_comments_incremental,
    upsert_video_stats,
)
from pipeline.support.json_io import read_json, write_json
from pipeline.support.environment import load_project_env
from pipeline.support.paths import youtube_api_infos_path, youtube_comments_path
from pipeline.support.scaleway_jobs import s3_client, upload_s3_directory


LOCK_NAME = "rag_ionis.update_runs"
DEFAULT_DOWNLOAD_DIR = Path("downloads/youtube")
DAILY_SYNC_LOG_NAME = "daily_sync_log.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIRECTORY_PATTERN = re.compile(r"^\d{8}_\d{4}(?:_\d{2})?$")
NON_VIDEO_DIRECTORY_NAMES = {"_00_info_videos", "_00_info_comments"}
UPDATE_S3_ROOT_PREFIX = "youtube"


def update_s3_archive_prefix(archive_name: str) -> str:
    return f"{UPDATE_S3_ROOT_PREFIX}/{archive_name}"


def required_update_s3() -> tuple[object, str]:
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME est requis pour les updates YouTube.")
    return s3_client(os.getenv("S3_REGION", "").strip()), bucket


def s3_prefix_exists(client, bucket: str, prefix: str) -> bool:
    response = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=1)
    return bool(response.get("Contents"))


def update_archive_name(client, bucket: str, started_at: datetime, suffix: str) -> str:
    base_name = f"{started_at.astimezone().strftime('%Y%m%d_%H%M')}_{suffix}"
    candidate = base_name
    counter = 2
    while s3_prefix_exists(client, bucket, update_s3_archive_prefix(candidate)):
        candidate = f"{base_name}_{counter:02d}"
        counter += 1
    return candidate


def upload_archive_file(client, bucket: str, archive_name: str, path: Path) -> None:
    client.upload_file(
        str(path),
        bucket,
        f"{update_s3_archive_prefix(archive_name)}/{path.name}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronise les statistiques ou traite les nouvelles videos "
            "de la chaine YouTube."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("all", "stats", "videos"),
        default="all",
        help=(
            "stats: statistiques uniquement; videos: nouvelles videos absentes "
            "de SQL; all: execute les deux updates (defaut)."
        ),
    )
    return parser.parse_args(argv)


def collect_videos() -> list[dict]:
    return fetch_videos(CHANNEL)


def database_videos(cursor) -> dict[str, int]:
    cursor.execute(
        """
        SELECT youtube_video_id, id
        FROM videos
        WHERE youtube_video_id IS NOT NULL
          AND btrim(youtube_video_id) <> ''
        ORDER BY id
        """
    )
    return {str(row[0]): int(row[1]) for row in cursor.fetchall()}


def videos_missing_from_database(
    api_videos: list[dict],
    existing_videos: dict[str, int] | set[str],
) -> list[dict]:
    existing_ids = set(existing_videos)
    return [
        video
        for video in api_videos
        if str(video.get("id") or "").strip() not in existing_ids
    ]


def acquire_lock(cursor) -> bool:
    cursor.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,))
    return bool(cursor.fetchone()[0])


def release_lock(cursor) -> None:
    cursor.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,))


def create_run(cursor, archive_path: str | Path | None) -> int:
    cursor.execute(
        """
        INSERT INTO update_runs (status, archive_path)
        VALUES ('running', %s)
        RETURNING id
        """,
        (str(archive_path),),
    )
    return int(cursor.fetchone()[0])


def finish_run(cursor, run_id: int, status: str, metrics: dict, errors: list[dict]) -> None:
    cursor.execute(
        """
        UPDATE update_runs
        SET finished_at = now(),
            status = %s,
            videos_discovered = %s,
            videos_updated = %s,
            videos_skipped = %s,
            stats_snapshots = %s,
            comments_seen = %s,
            comments_new = %s,
            comments_refreshed = %s,
            comments_deleted = %s,
            new_videos = %s,
            new_video_ids = %s::jsonb,
            previous_archive_path = %s,
            new_since_previous = %s,
            new_since_previous_ids = %s::jsonb,
            pipeline_videos_started = %s,
            pipeline_videos_completed = %s,
            videos_with_new_comments = %s,
            new_comments_detected = %s,
            errors = %s::jsonb
        WHERE id = %s
        """,
        (
            status,
            metrics["videos_discovered"],
            metrics["videos_updated"],
            metrics["videos_skipped"],
            metrics["stats_snapshots"],
            metrics["comments_seen"],
            metrics["comments_new"],
            metrics["comments_refreshed"],
            metrics["comments_deleted"],
            metrics["new_videos"],
            json.dumps(metrics["new_video_ids"], ensure_ascii=False),
            metrics["previous_archive_path"],
            metrics["new_since_previous"],
            json.dumps(metrics["new_since_previous_ids"], ensure_ascii=False),
            metrics["pipeline_videos_started"],
            metrics["pipeline_videos_completed"],
            metrics["videos_with_new_comments"],
            metrics["new_comments_detected"],
            json.dumps(errors, ensure_ascii=False),
            run_id,
        ),
    )


def create_archive_directory(
    download_dir: Path,
    started_at: datetime,
    suffix: str | None = None,
) -> Path:
    base_name = started_at.astimezone().strftime("%Y%m%d_%H%M")
    if suffix:
        base_name = f"{base_name}_{suffix}"
    archive_dir = download_dir / base_name
    suffix = 2
    while archive_dir.exists():
        archive_dir = download_dir / f"{base_name}_{suffix:02d}"
        suffix += 1
    archive_dir.mkdir(parents=True)
    return archive_dir


def video_directory_ids(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name not in NON_VIDEO_DIRECTORY_NAMES
    }


def new_video_ids_from_archive(archive_dir: Path, download_dir: Path) -> list[str]:
    archived_ids = video_directory_ids(archive_dir)
    init_ids = video_directory_ids(download_dir / "init")
    return sorted(archived_ids - init_ids)


def previous_archive_directory(
    download_dir: Path,
    current_archive_dir: Path,
) -> Path | None:
    candidates = sorted(
        path
        for path in download_dir.iterdir()
        if path.is_dir()
        and path != current_archive_dir
        and ARCHIVE_DIRECTORY_PATTERN.fullmatch(path.name)
        and path.name < current_archive_dir.name
    )
    return candidates[-1] if candidates else None


def new_video_ids_since_previous(
    current_archive_dir: Path,
    previous_archive_dir: Path,
) -> list[str]:
    current_ids = video_directory_ids(current_archive_dir)
    previous_ids = video_directory_ids(previous_archive_dir)
    return sorted(current_ids - previous_ids)


def pipeline_video_ids_from_archives(
    new_video_ids: list[str],
    new_since_previous_ids: list[str],
    *,
    has_previous_archive: bool,
) -> list[str]:
    detected_ids = new_since_previous_ids if has_previous_archive else new_video_ids
    return sorted(set(detected_ids))


def comment_ids_from_json(path: Path) -> set[str]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return set()
    return {
        str(item.get("youtube_comment_id") or "").strip()
        for item in payload.get("comments", [])
        if isinstance(item, dict)
        and str(item.get("youtube_comment_id") or "").strip()
    }


def new_comment_ids_by_video(
    current_archive_dir: Path,
    baseline_dir: Path,
) -> dict[str, list[str]]:
    result = {}
    for youtube_video_id in video_directory_ids(current_archive_dir):
        current_path = (
            current_archive_dir
            / youtube_video_id
            / "metadata"
            / "youtube_comments.json"
        )
        if not current_path.exists():
            continue
        baseline_path = (
            baseline_dir
            / youtube_video_id
            / "metadata"
            / "youtube_comments.json"
        )
        new_ids = sorted(
            comment_ids_from_json(current_path)
            - comment_ids_from_json(baseline_path)
        )
        if new_ids:
            result[youtube_video_id] = new_ids
    return result


def write_archive(
    video: dict,
    comments_payload: dict | None,
    archive_dir: Path,
) -> None:
    archive_video_dir = archive_dir / video["id"]
    archive_video_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        youtube_api_infos_path(archive_video_dir),
        video_info_payload(video),
    )
    if comments_payload is not None:
        write_json(
            youtube_comments_path(archive_video_dir),
            comments_payload,
        )


def optional_count(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def archive_log_payload(
    *,
    started_at: datetime,
    status: str,
    videos: list[dict],
    new_video_ids: list[str],
    pipeline_results: dict[str, dict],
    comparison_directory: Path,
    errors: list[dict],
    finished_at: datetime | None = None,
) -> dict:
    videos_by_id = {str(video.get("id") or ""): video for video in videos}
    pipelines = []
    for youtube_video_id in new_video_ids:
        video = videos_by_id.get(youtube_video_id, {})
        result = pipeline_results.get(youtube_video_id, {})
        pipelines.append(
            {
                "youtube_video_id": youtube_video_id,
                "title": (video.get("snippet") or {}).get("title"),
                "status": result.get("status", "pending"),
                "succeeded": result.get("succeeded"),
                "error": result.get("error"),
                "backend": result.get("backend"),
                "job_id": result.get("job_id"),
                "s3_uri": result.get("s3_uri"),
            }
        )

    video_snapshots = []
    for video in videos:
        statistics = video.get("statistics") or {}
        video_snapshots.append(
            {
                "youtube_video_id": video.get("id"),
                "title": (video.get("snippet") or {}).get("title"),
                "view_count": optional_count(statistics.get("viewCount")),
                "like_count": optional_count(statistics.get("likeCount")),
                "comment_count": optional_count(statistics.get("commentCount")),
            }
        )

    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at else None,
        "snapshot_date": started_at.date().isoformat(),
        "status": status,
        "comparison_directory": str(comparison_directory),
        "new_videos_detected": len(new_video_ids),
        "new_video_ids": new_video_ids,
        "pipelines": pipelines,
        "videos": video_snapshots,
        "errors": errors,
    }


def write_archive_log(
    archive_dir: Path,
    **payload_kwargs,
) -> Path:
    return write_json(
        archive_dir / DAILY_SYNC_LOG_NAME,
        archive_log_payload(**payload_kwargs),
    )


def download_and_run_pipeline(
    video: dict,
    archive_dir: Path,
    download_dir: Path,
    snapshot_date: date,
    comments_payload: dict | None = None,
) -> dict:
    backend = os.getenv("PIPELINE_EXECUTION_BACKEND", "scaleway").strip().lower()
    if backend == "scaleway":
        return run_pipeline_on_scaleway(
            video,
            archive_dir,
            snapshot_date,
            comments_payload=comments_payload,
        )
    if backend != "local":
        raise RuntimeError(
            "PIPELINE_EXECUTION_BACKEND doit valoir 'local' ou 'scaleway'."
        )

    youtube_video_id = video["id"]
    payload = video_info_payload(video)
    download_video(
        (
            youtube_video_id,
            payload["title"],
            payload["url"],
            payload,
        ),
        archive_dir,
        download_dir,
        reuse_previous=False,
        sync_cached_metadata=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline",
            "run",
            youtube_video_id,
            "--root",
            str(archive_dir.resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline",
            "task",
            "embeddings.create",
            youtube_video_id,
            "--root",
            str(archive_dir.resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    sync_processed_video(youtube_video_id, archive_dir, snapshot_date)
    return {"backend": "local"}


def sync_processed_video(
    youtube_video_id: str,
    archive_dir: Path,
    snapshot_date: date,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.publish.sync_database",
            "--video-dir",
            str(archive_dir.resolve()),
            "--video-id",
            youtube_video_id,
            "--snapshot-date",
            snapshot_date.isoformat(),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_pipeline_on_scaleway(
    video: dict,
    archive_dir: Path,
    snapshot_date: date,
    *,
    comments_payload: dict | None = None,
) -> dict:
    from pipeline.support.scaleway_jobs import (
        ScalewayClient,
        ScalewayConfig,
        ScalewayJobError,
        download_s3_prefix,
        s3_client,
    )

    youtube_video_id = str(video["id"])
    video_dir = archive_dir / youtube_video_id
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME est requis pour la daily sync Scaleway.")
    control_root = os.getenv(
        "SCALEWAY_S3_JOB_PREFIX", "scaleway/jobs"
    ).strip().strip("/")
    if not control_root:
        raise RuntimeError("SCALEWAY_S3_JOB_PREFIX ne peut pas etre vide.")
    control_prefix = f"{control_root}/{uuid.uuid4().hex}"
    job_id = control_prefix.rsplit("/", 1)[-1]
    job_input = {
        "schema_version": 2,
        "operation": "pipeline_command",
        "command": "run",
        "include_embeddings": True,
        "video_id": youtube_video_id,
        "archive_name": archive_dir.name,
        "snapshot_date": snapshot_date.isoformat(),
        "options": {
            "force": False,
            "openai_mode": os.getenv("DAILY_SYNC_OPENAI_MODE", "normal"),
            "correction_mode": os.getenv("DAILY_SYNC_CORRECTION_MODE", "balanced"),
            "frame_interval_seconds": float(
                os.getenv("DAILY_SYNC_FRAME_INTERVAL_SECONDS", "0.5")
            ),
            "details_per_section": int(
                os.getenv("DAILY_SYNC_DETAILS_PER_SECTION", "6")
            ),
        },
        "source": {
            "type": "youtube",
            "video": video,
        },
        "result": {
            "bucket": bucket,
            "prefix": f"youtube/{archive_dir.name}/{youtube_video_id}",
        },
        "control": {
            "bucket": bucket,
            "job_id": job_id,
            "job_key": f"{control_prefix}/job.json",
            "status_key": f"{control_prefix}/status.json",
        },
    }
    if comments_payload is not None:
        job_input["source"]["comments"] = comments_payload

    region = os.getenv("S3_REGION", "").strip()
    object_store = s3_client(region)
    client = ScalewayClient(ScalewayConfig.from_env())
    job_id, output = client.run(job_input, object_store=object_store, bucket=bucket)
    if str(output.get("video_id") or "") != youtube_video_id:
        raise ScalewayJobError(
            f"Le job Scaleway {job_id} a renvoye un autre identifiant video."
        )
    bucket = str(output.get("bucket") or "").strip()
    prefix = str(output.get("prefix") or "").strip().strip("/")
    expected_prefix = f"youtube/{archive_dir.name}/{youtube_video_id}"
    configured_bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if (
        not bucket
        or (configured_bucket and bucket != configured_bucket)
        or prefix != expected_prefix
    ):
        raise ScalewayJobError(
            f"Le job Scaleway {job_id} a renvoye une destination S3 invalide."
        )

    downloaded = download_s3_prefix(
        object_store,
        bucket,
        prefix,
        video_dir,
    )
    print(
        f"[scaleway] {downloaded} artefact(s) rapatrie(s) depuis "
        f"s3://{bucket}/{prefix}/",
        flush=True,
    )
    sync_processed_video(youtube_video_id, archive_dir, snapshot_date)
    keep_job_artifacts = os.getenv(
        "SCALEWAY_KEEP_JOB_ARTIFACTS", "0"
    ).strip().lower() in {"1", "true", "yes"}
    if not keep_job_artifacts:
        from pipeline.publish.upload_outputs_to_s3 import delete_prefix

        delete_prefix(object_store, bucket, control_prefix)
    return {
        "backend": "scaleway",
        "job_id": job_id,
        "s3_uri": f"s3://{bucket}/{prefix}",
    }


def write_new_videos_manifest(
    archive_dir: Path,
    videos: list[dict],
    *,
    generated_at: datetime | None = None,
) -> Path:
    payload = {
        "generated_at": (
            generated_at or datetime.now(timezone.utc)
        ).isoformat(),
        "count": len(videos),
        "videos": [video_info_payload(video) for video in videos],
    }
    return write_json(archive_dir / "new_videos.json", payload)


def write_stats_update_log(
    archive_dir: Path,
    metrics: dict,
    errors: list[dict],
    *,
    status: str,
    generated_at: datetime,
) -> Path:
    return write_json(
        archive_dir / "update_stats_log.json",
        {
            "generated_at": generated_at.isoformat(),
            "status": status,
            "videos_discovered": metrics["videos_discovered"],
            "videos_updated": metrics["videos_updated"],
            "videos_skipped": metrics["videos_skipped"],
            "stats_snapshots": metrics["stats_snapshots"],
            "errors": errors,
        },
    )


def run_stats_only() -> dict:
    """Refresh SQL stats for videos already present in the database."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")

    metrics = empty_metrics()
    errors: list[dict] = []
    run_id = None
    lock_acquired = False
    started_at = datetime.now().astimezone()
    archive_dir: Path | None = None
    temp_dir: Path | None = None
    object_store, bucket = required_update_s3()

    with psycopg.connect(
        database_url,
        options="-c search_path=data,public",
    ) as connection:
        try:
            with connection.cursor() as cursor:
                ensure_schema(cursor)
                lock_acquired = acquire_lock(cursor)
                if not lock_acquired:
                    raise RuntimeError(
                        "Une synchronisation YouTube est deja en cours."
                    )
                archive_name = update_archive_name(
                    object_store,
                    bucket,
                    started_at,
                    "update_stats",
                )
                temp_dir = Path(tempfile.mkdtemp(prefix="rag-ionis-update-stats-"))
                archive_dir = temp_dir / archive_name
                archive_dir.mkdir()
                run_id = create_run(
                    cursor,
                    f"s3://{bucket}/{update_s3_archive_prefix(archive_name)}",
                )
            connection.commit()

            api_videos = collect_videos()
            metrics["videos_discovered"] = len(api_videos)
            with connection.cursor() as cursor:
                existing_videos = database_videos(cursor)
            connection.rollback()

            metrics["videos_skipped"] = sum(
                1 for video in api_videos if video["id"] not in existing_videos
            )
            for video in api_videos:
                youtube_video_id = str(video.get("id") or "")
                video_db_id = existing_videos.get(youtube_video_id)
                if video_db_id is None:
                    continue
                try:
                    with connection.transaction():
                        with connection.cursor() as cursor:
                            if upsert_video_stats(
                                cursor,
                                video_db_id,
                                video_info_payload(video),
                                snapshot_date=started_at.date(),
                            ):
                                metrics["stats_snapshots"] += 1
                    metrics["videos_updated"] += 1
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "stats_sql",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )

            status = "completed_with_errors" if errors else "completed"
            with connection.cursor() as cursor:
                finish_run(cursor, run_id, status, metrics, errors)
            connection.commit()
            metrics["errors_count"] = len(errors)
            write_stats_update_log(
                archive_dir,
                metrics,
                errors,
                status=status,
                generated_at=started_at,
            )
            upload_archive_file(
                object_store,
                bucket,
                archive_dir.name,
                archive_dir / "update_stats_log.json",
            )
            return metrics
        except Exception as error:
            connection.rollback()
            errors.append({"type": type(error).__name__, "message": str(error)})
            if run_id is not None:
                with connection.cursor() as cursor:
                    finish_run(cursor, run_id, "failed", metrics, errors)
                connection.commit()
            if archive_dir is not None:
                try:
                    metrics["errors_count"] = len(errors)
                    write_stats_update_log(
                        archive_dir,
                        metrics,
                        errors,
                        status="failed",
                        generated_at=started_at,
                    )
                    upload_archive_file(
                        object_store,
                        bucket,
                        archive_dir.name,
                        archive_dir / "update_stats_log.json",
                    )
                except Exception:
                    pass
            raise
        finally:
            if lock_acquired:
                connection.rollback()
                with connection.cursor() as cursor:
                    release_lock(cursor)
                connection.commit()
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


def run_videos_only() -> dict:
    """Process YouTube videos whose IDs are absent from SQL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")

    metrics = empty_metrics()
    errors: list[dict] = []
    run_id = None
    lock_acquired = False
    download_dir = DEFAULT_DOWNLOAD_DIR
    started_at = datetime.now().astimezone()
    object_store, bucket = required_update_s3()
    archive_dir: Path | None = None
    temp_dir: Path | None = None
    archive_name: str | None = None
    pipeline_results: dict[str, dict] = {}
    api_videos: list[dict] = []
    new_video_ids: list[str] = []

    def update_archive_log(status: str, *, finished: bool = False) -> None:
        write_archive_log(
            archive_dir,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc) if finished else None,
            status=status,
            videos=api_videos,
            new_video_ids=new_video_ids,
            pipeline_results=pipeline_results,
            comparison_directory=Path("data.videos"),
            errors=errors,
        )
        upload_archive_file(
            object_store,
            bucket,
            archive_name,
            archive_dir / DAILY_SYNC_LOG_NAME,
        )

    with psycopg.connect(
        database_url,
        options="-c search_path=data,public",
    ) as connection:
        try:
            with connection.cursor() as cursor:
                ensure_schema(cursor)
                lock_acquired = acquire_lock(cursor)
                if not lock_acquired:
                    raise RuntimeError(
                        "Une synchronisation YouTube est deja en cours."
                    )
                archive_name = update_archive_name(
                    object_store,
                    bucket,
                    started_at,
                    "update_videos",
                )
                temp_dir = Path(tempfile.mkdtemp(prefix="rag-ionis-update-videos-"))
                archive_dir = temp_dir / archive_name
                archive_dir.mkdir()
                run_id = create_run(
                    cursor,
                    f"s3://{bucket}/{update_s3_archive_prefix(archive_name)}",
                )
            connection.commit()

            api_videos = collect_videos()
            metrics["videos_discovered"] = len(api_videos)
            with connection.cursor() as cursor:
                existing_videos = database_videos(cursor)
            connection.rollback()

            new_videos = videos_missing_from_database(api_videos, existing_videos)
            new_video_ids = [str(video["id"]) for video in new_videos]
            metrics["new_videos"] = len(new_videos)
            metrics["new_video_ids"] = new_video_ids
            metrics["videos_skipped"] = len(api_videos) - len(new_videos)
            write_new_videos_manifest(
                archive_dir,
                new_videos,
                generated_at=started_at,
            )
            upload_archive_file(
                object_store,
                bucket,
                archive_name,
                archive_dir / "new_videos.json",
            )
            update_archive_log("running")

            for video in new_videos:
                youtube_video_id = str(video["id"])
                comments_payload = None
                try:
                    comments_payload = fetch_comments(youtube_video_id)
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "comments_api",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                try:
                    write_archive(video, comments_payload, archive_dir)
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "cache",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    pipeline_results[youtube_video_id] = {
                        "status": "failed",
                        "succeeded": False,
                        "error": str(error),
                    }
                    update_archive_log("running")
                    continue

                metrics["pipeline_videos_started"] += 1
                pipeline_results[youtube_video_id] = {
                    "status": "running",
                    "succeeded": None,
                    "error": None,
                }
                update_archive_log("running")
                try:
                    result = download_and_run_pipeline(
                        video,
                        archive_dir,
                        download_dir,
                        started_at.date(),
                        comments_payload=comments_payload,
                    )
                    if result.get("backend") == "local":
                        upload_s3_directory(
                            object_store,
                            bucket,
                            archive_dir / youtube_video_id,
                            f"{update_s3_archive_prefix(archive_name)}/{youtube_video_id}",
                            excluded_suffixes={".mp4", ".mkv", ".webm", ".mov", ".m4v"},
                        )
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "pipeline_run",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    pipeline_results[youtube_video_id] = {
                        "status": "failed",
                        "succeeded": False,
                        "error": str(error),
                    }
                    update_archive_log("running")
                    continue

                metrics["pipeline_videos_completed"] += 1
                metrics["videos_updated"] += 1
                metrics["stats_snapshots"] += 1
                pipeline_results[youtube_video_id] = {
                    "status": "completed",
                    "succeeded": True,
                    "error": None,
                    **result,
                }
                update_archive_log("running")

            status = "completed_with_errors" if errors else "completed"
            with connection.cursor() as cursor:
                finish_run(cursor, run_id, status, metrics, errors)
            connection.commit()
            metrics["errors_count"] = len(errors)
            update_archive_log(status, finished=True)
            return metrics
        except Exception as error:
            connection.rollback()
            errors.append({"type": type(error).__name__, "message": str(error)})
            if run_id is not None:
                with connection.cursor() as cursor:
                    finish_run(cursor, run_id, "failed", metrics, errors)
                connection.commit()
            try:
                update_archive_log("failed", finished=True)
            except Exception:
                pass
            raise
        finally:
            if lock_acquired:
                connection.rollback()
                with connection.cursor() as cursor:
                    release_lock(cursor)
                connection.commit()
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


def empty_metrics() -> dict:
    return {
        "videos_discovered": 0,
        "videos_updated": 0,
        "videos_skipped": 0,
        "stats_snapshots": 0,
        "comments_seen": 0,
        "comments_new": 0,
        "comments_refreshed": 0,
        "comments_deleted": 0,
        "new_videos": 0,
        "new_video_ids": [],
        "previous_archive_path": None,
        "new_since_previous": 0,
        "new_since_previous_ids": [],
        "pipeline_videos_started": 0,
        "pipeline_videos_completed": 0,
        "videos_with_new_comments": 0,
        "new_comments_detected": 0,
        "errors_count": 0,
    }


def run() -> dict:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL manquant dans l'environnement.")

    metrics = empty_metrics()
    errors: list[dict] = []
    run_id = None
    lock_acquired = False
    download_dir = DEFAULT_DOWNLOAD_DIR
    started_at = datetime.now().astimezone()
    archive_dir = None
    api_videos: list[dict] = []
    pipeline_video_ids: list[str] = []
    pipeline_results: dict[str, dict] = {}
    comparison_directory = download_dir / "init"

    def update_archive_log(status: str, *, finished: bool = False) -> None:
        if archive_dir is None:
            return
        write_archive_log(
            archive_dir,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc) if finished else None,
            status=status,
            videos=api_videos,
            new_video_ids=pipeline_video_ids,
            pipeline_results=pipeline_results,
            comparison_directory=comparison_directory,
            errors=errors,
        )

    with psycopg.connect(
        database_url,
        options="-c search_path=data,public",
    ) as connection:
        try:
            with connection.cursor() as cursor:
                ensure_schema(cursor)
                lock_acquired = acquire_lock(cursor)
                if not lock_acquired:
                    raise RuntimeError("Une synchronisation YouTube quotidienne est deja en cours.")
                archive_dir = create_archive_directory(download_dir, started_at)
                run_id = create_run(cursor, archive_dir)
            connection.commit()

            with connection.cursor() as cursor:
                videos_by_youtube_id = database_videos(cursor)
            connection.rollback()

            api_videos = collect_videos()
            metrics["videos_discovered"] = len(api_videos)
            sql_missing_ids = [
                video["id"]
                for video in api_videos
                if video["id"] not in videos_by_youtube_id
            ]
            metrics["videos_skipped"] = len(sql_missing_ids)

            collected = []
            for video in api_videos:
                youtube_video_id = video["id"]
                comments_payload = None
                try:
                    comments_payload = fetch_comments(youtube_video_id)
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "comments_api",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                collected.append((video, comments_payload))

            # La phase de collecte se termine avant toute publication locale ou SQL.
            cache_ready = []
            for video, comments_payload in collected:
                try:
                    write_archive(
                        video,
                        comments_payload,
                        archive_dir,
                    )
                except Exception as error:
                    errors.append(
                        {
                            "video_id": video["id"],
                            "phase": "cache",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    continue
                cache_ready.append((video, comments_payload))

            new_video_ids = new_video_ids_from_archive(archive_dir, download_dir)
            metrics["new_video_ids"] = new_video_ids
            metrics["new_videos"] = len(new_video_ids)
            print(
                f"[daily-sync] {metrics['new_videos']} nouvelle(s) video(s) "
                f"par rapport a {download_dir / 'init'}."
            )

            previous_archive = previous_archive_directory(download_dir, archive_dir)
            if previous_archive is None:
                print("[daily-sync] aucune archive horodatee precedente a comparer.")
                comments_baseline_dir = download_dir / "init"
            else:
                comparison_directory = previous_archive
                new_since_previous_ids = new_video_ids_since_previous(
                    archive_dir,
                    previous_archive,
                )
                metrics["previous_archive_path"] = str(previous_archive)
                metrics["new_since_previous_ids"] = new_since_previous_ids
                metrics["new_since_previous"] = len(new_since_previous_ids)
                print(
                    f"[daily-sync] {metrics['new_since_previous']} nouvelle(s) "
                    f"video(s) depuis {previous_archive.name}."
                )
                comments_baseline_dir = previous_archive

            new_comments_by_video = new_comment_ids_by_video(
                archive_dir,
                comments_baseline_dir,
            )
            metrics["videos_with_new_comments"] = len(new_comments_by_video)
            metrics["new_comments_detected"] = sum(
                len(comment_ids) for comment_ids in new_comments_by_video.values()
            )
            print(
                f"[daily-sync] {metrics['new_comments_detected']} nouveau(x) "
                f"commentaire(s) sur {metrics['videos_with_new_comments']} video(s)."
            )
            for youtube_video_id, comment_ids in sorted(new_comments_by_video.items()):
                print(
                    f"[daily-sync] {youtube_video_id}: "
                    f"{len(comment_ids)} nouveau(x) commentaire(s)."
                )

            snapshot_date = started_at.date()
            for video, comments_payload in cache_ready:
                youtube_video_id = video["id"]
                if youtube_video_id not in videos_by_youtube_id:
                    continue
                video_db_id = videos_by_youtube_id[youtube_video_id]
                stats_written = False
                comment_result = None
                try:
                    with connection.transaction():
                        with connection.cursor() as cursor:
                            payload = video_info_payload(video)
                            stats_written = upsert_video_stats(
                                cursor,
                                video_db_id,
                                payload,
                                snapshot_date=snapshot_date,
                            )
                            if (
                                comments_payload is not None
                                and youtube_video_id in new_comments_by_video
                            ):
                                comment_result = sync_video_comments_incremental(
                                    cursor,
                                    video_db_id,
                                    comments_payload,
                                    collected_at=datetime.now(timezone.utc),
                                )
                except Exception as error:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "sql",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    continue
                if stats_written:
                    metrics["stats_snapshots"] += 1
                if comment_result is not None:
                    metrics["comments_seen"] += comment_result.seen
                    metrics["comments_new"] += comment_result.created
                    metrics["comments_refreshed"] += comment_result.refreshed
                    metrics["comments_deleted"] += comment_result.deleted
                metrics["videos_updated"] += 1

            pipeline_video_ids = pipeline_video_ids_from_archives(
                new_video_ids,
                metrics["new_since_previous_ids"],
                has_previous_archive=previous_archive is not None,
            )
            pipeline_results = {
                youtube_video_id: {
                    "status": "pending",
                    "succeeded": None,
                    "error": None,
                }
                for youtube_video_id in pipeline_video_ids
            }
            update_archive_log("running")
            archived_videos_by_id = {
                video["id"]: video for video, _comments in cache_ready
            }
            runnable_videos = []
            for youtube_video_id in pipeline_video_ids:
                video = archived_videos_by_id.get(youtube_video_id)
                if video is None:
                    pipeline_results[youtube_video_id] = {
                        "status": "failed",
                        "succeeded": False,
                        "error": "Archive locale indisponible pour cette vidéo.",
                    }
                    update_archive_log("running")
                    continue
                metrics["pipeline_videos_started"] += 1
                pipeline_results[youtube_video_id]["status"] = "running"
                runnable_videos.append((youtube_video_id, video))

            update_archive_log("running")

            def execute_remote_video(item):
                youtube_video_id, video = item
                print(
                    f"[daily-sync] lancement du pipeline pour {youtube_video_id}."
                )
                try:
                    result = download_and_run_pipeline(
                        video,
                        archive_dir,
                        download_dir,
                        snapshot_date,
                    )
                    return youtube_video_id, result, None
                except Exception as error:
                    return youtube_video_id, None, error

            completed_runs = [
                execute_remote_video(item) for item in runnable_videos
            ]

            for youtube_video_id, execution_result, error in completed_runs:
                if error is not None:
                    errors.append(
                        {
                            "video_id": youtube_video_id,
                            "phase": "pipeline_run",
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    pipeline_results[youtube_video_id] = {
                        "status": "failed",
                        "succeeded": False,
                        "error": str(error),
                    }
                    update_archive_log("running")
                    continue
                metrics["pipeline_videos_completed"] += 1
                pipeline_results[youtube_video_id] = {
                    "status": "completed",
                    "succeeded": True,
                    "error": None,
                    **execution_result,
                }
                update_archive_log("running")
                if youtube_video_id in sql_missing_ids:
                    metrics["videos_updated"] += 1
                    metrics["stats_snapshots"] += 1

            with connection.cursor() as cursor:
                status = "completed_with_errors" if errors else "completed"
                finish_run(cursor, run_id, status, metrics, errors)
            connection.commit()
            metrics["errors_count"] = len(errors)
            update_archive_log(status, finished=True)
            return metrics
        except Exception as error:
            connection.rollback()
            errors.append({"type": type(error).__name__, "message": str(error)})
            if run_id is not None:
                with connection.cursor() as cursor:
                    finish_run(cursor, run_id, "failed", metrics, errors)
                connection.commit()
            if archive_dir is not None:
                try:
                    update_archive_log("failed", finished=True)
                except Exception as log_error:
                    print(
                        f"[daily-sync] impossible d'ecrire {DAILY_SYNC_LOG_NAME}: "
                        f"{log_error}",
                        flush=True,
                    )
            raise
        finally:
            if lock_acquired:
                connection.rollback()
                with connection.cursor() as cursor:
                    release_lock(cursor)
                connection.commit()


def main() -> None:
    load_project_env(PROJECT_ROOT)
    args = parse_args()
    if args.mode == "stats":
        metrics = run_stats_only()
    elif args.mode == "videos":
        metrics = run_videos_only()
    else:
        stats_metrics = run_stats_only()
        metrics = run_videos_only()
        print(
            "Updates YouTube termines: "
            f"{stats_metrics['stats_snapshots']} snapshot(s) stats, "
            f"{metrics['pipeline_videos_completed']} pipeline(s) videos, "
            f"{stats_metrics['errors_count'] + metrics['errors_count']} erreur(s)."
        )
        return
    print(
        "Synchronisation YouTube terminee: "
        f"{metrics['videos_updated']} video(s), "
        f"{metrics['stats_snapshots']} snapshot(s), "
        f"{metrics['comments_seen']} commentaire(s) vu(s), "
        f"{metrics['comments_new']} nouveau(x), "
        f"{metrics['comments_deleted']} supprime(s), "
        f"{metrics['new_videos']} nouvelle(s) video(s), "
        f"{metrics['pipeline_videos_completed']} nouveau(x) pipeline(s) termine(s), "
        f"{metrics['errors_count']} erreur(s)."
    )
    if metrics["errors_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
