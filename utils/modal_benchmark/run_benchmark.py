"""Compare the Modal and Scaleway GPU backends on a few archived videos.

Each video is sent as a daily-sync job (YouTube source, so yt-dlp runs on the
worker). Results go under a dedicated S3 prefix that is deleted afterwards;
nothing is written to PostgreSQL.

Usage:
    python utils/modal_benchmark/run_benchmark.py --backend modal L0ipiwb9jS8 U-1vz4l4908
    python utils/modal_benchmark/run_benchmark.py --backend scaleway L0ipiwb9jS8

Videos are looked up in downloads/youtube/init/<id>/metadata. Each run appends
one line per video to utils/modal_benchmark/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.support.environment import load_project_env  # noqa: E402
from pipeline.publish.upload_outputs_to_s3 import delete_prefix  # noqa: E402
from pipeline.support.remote_backend import remote_job_client  # noqa: E402
from pipeline.support.scaleway_jobs import s3_client  # noqa: E402


ARCHIVE_DIR = PROJECT_ROOT / "downloads" / "youtube" / "init"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"
# EUR or USD per GPU hour, only used for the cost estimate printed per video.
HOURLY_RATE = {"modal": 0.80, "scaleway": 0.79}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_processing_seconds(manifest: dict) -> float | None:
    """Wall time of the first local run, from the archived manifest."""
    executions = list(manifest.get("execution_history") or [])
    if manifest.get("execution"):
        executions.append(manifest["execution"])
    spans = []
    for execution in executions:
        try:
            started = datetime.fromisoformat(execution["started_at"])
            finished = datetime.fromisoformat(execution["finished_at"])
        except (KeyError, TypeError, ValueError):
            continue
        spans.append((started, (finished - started).total_seconds()))
    if not spans:
        return None
    first_day = min(started for started, _ in spans).date()
    return sum(
        seconds
        for started, seconds in spans
        if started.date() == first_day and seconds < 4 * 3600
    )


def build_job(video_id: str, bucket: str, run_id: str) -> tuple[dict, str, str]:
    metadata_dir = ARCHIVE_DIR / video_id / "metadata"
    video = read_json(metadata_dir / "youtube_video_metadata.json")
    comments_path = metadata_dir / "youtube_comments.json"
    control_root = os.getenv("SCALEWAY_S3_JOB_PREFIX", "scaleway/jobs").strip("/")
    control_prefix = f"{control_root}/{uuid.uuid4().hex}"
    result_prefix = f"benchmarks/{run_id}/{video_id}"
    job_input = {
        "schema_version": 2,
        "operation": "pipeline_command",
        "command": "run",
        "include_embeddings": True,
        "video_id": video_id,
        "archive_name": f"benchmark-{run_id}",
        "options": {
            "force": False,
            "openai_mode": "normal",
            "correction_mode": "balanced",
            "frame_interval_seconds": 0.5,
            "details_per_section": 6,
        },
        "source": {"type": "youtube", "video": video},
        "result": {"bucket": bucket, "prefix": result_prefix},
        "control": {
            "bucket": bucket,
            "job_id": control_prefix.rsplit("/", 1)[-1],
            "job_key": f"{control_prefix}/job.json",
            "status_key": f"{control_prefix}/status.json",
        },
    }
    if comments_path.exists():
        job_input["source"]["comments"] = read_json(comments_path)
    return job_input, result_prefix, control_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=sorted(HOURLY_RATE), required=True)
    parser.add_argument("--keep", action="store_true", help="Keep S3 results.")
    parser.add_argument("video_ids", nargs="+")
    args = parser.parse_args()

    load_project_env(PROJECT_ROOT)
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME est requis.")
    object_store = s3_client(os.getenv("S3_REGION", "").strip())
    client = remote_job_client(args.backend)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for video_id in args.video_ids:
        manifest = read_json(ARCHIVE_DIR / video_id / "metadata" / "video_manifest.json")
        duration = float(manifest["video"]["duration_seconds"])
        job_input, result_prefix, control_prefix = build_job(video_id, bucket, run_id)
        print(f"\n=== {args.backend} {video_id} ({duration:.0f} s de video) ===", flush=True)
        started = time.monotonic()
        error = None
        try:
            client.run(job_input, object_store=object_store, bucket=bucket)
        except Exception as exc:  # recorded, then the next video runs
            error = f"{type(exc).__name__}: {exc}"
            print(f"[benchmark] echec: {error}", flush=True)
        wall = time.monotonic() - started
        local = local_processing_seconds(manifest)
        row = {
            "run_id": run_id,
            "backend": args.backend,
            "video_id": video_id,
            "video_seconds": duration,
            "wall_seconds": round(wall, 1),
            "local_pipeline_seconds": round(local, 1) if local else None,
            "estimated_gpu_cost": round(wall / 3600 * HOURLY_RATE[args.backend], 3),
            "error": error,
        }
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[benchmark] {json.dumps(row)}", flush=True)
        if not args.keep:
            delete_prefix(object_store, bucket, result_prefix)
        delete_prefix(object_store, bucket, control_prefix)


if __name__ == "__main__":
    main()
