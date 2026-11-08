"""Time the pipeline with its GPU tasks on Modal, or on the local GPU.

Each archived video is copied (video file and YouTube metadata only) into a
temporary archive, then ``python -m pipeline run`` processes it from scratch.
Nothing is written to S3 or PostgreSQL; OpenAI is called as in production.

Usage:
    python utils/modal_benchmark/run_benchmark.py --backend modal L0ipiwb9jS8 U-1vz4l4908
    python utils/modal_benchmark/run_benchmark.py --backend local L0ipiwb9jS8

One line per video is appended to utils/modal_benchmark/results.jsonl, with the
duration of each GPU task read from the pipeline manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "downloads" / "youtube" / "init"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"
GPU_TASKS = ("frames.classify", "ocr.extract_raw", "transcript.whisper")
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
METADATA_FILES = ("youtube_video_metadata.json", "youtube_comments.json")
# ``python -m pipeline run``, except that the backend chosen here survives the
# CLI loading .env.local, which overrides the process environment.
PIPELINE_RUNNER = """
import os, sys
from pipeline.support import environment

backend = os.environ["PIPELINE_EXECUTION_BACKEND"]
load = environment.load_project_env

def load_keeping_backend(*args, **kwargs):
    path = load(*args, **kwargs)
    os.environ["PIPELINE_EXECUTION_BACKEND"] = backend
    return path

environment.load_project_env = load_keeping_backend
from pipeline import cli

cli.load_project_env = load_keeping_backend
sys.argv = ["pipeline", "run", sys.argv[1], "--root", sys.argv[2]]
cli.main()
"""


def copy_source(video_id: str, destination: Path) -> Path:
    source = ARCHIVE_DIR / video_id
    videos = [path for path in source.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise FileNotFoundError(f"Aucune video locale pour {video_id} dans {source}")
    target = destination / video_id
    (target / "metadata").mkdir(parents=True)
    shutil.copy2(videos[0], target / videos[0].name)
    for name in METADATA_FILES:
        if (source / "metadata" / name).exists():
            shutil.copy2(source / "metadata" / name, target / "metadata" / name)
    return target


def task_seconds(manifest: dict) -> dict[str, float]:
    durations: dict[str, float] = {}
    executions = list(manifest.get("execution_history") or [])
    if manifest.get("execution"):
        executions.append(manifest["execution"])
    for execution in executions:
        for task_id, task in (execution.get("tasks") or {}).items():
            try:
                started = datetime.fromisoformat(task["started_at"])
                finished = datetime.fromisoformat(task["finished_at"])
            except (KeyError, TypeError, ValueError):
                continue
            durations[task_id] = durations.get(task_id, 0.0) + (
                finished - started
            ).total_seconds()
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("modal", "local"), required=True)
    parser.add_argument("video_ids", nargs="+")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    environment = {**os.environ, "PIPELINE_EXECUTION_BACKEND": args.backend}
    for video_id in args.video_ids:
        with tempfile.TemporaryDirectory(prefix="rag-ionis-bench-") as directory:
            archive = Path(directory) / f"benchmark-{run_id}"
            video_dir = copy_source(video_id, archive)
            print(f"\n=== {args.backend} {video_id} ===", flush=True)
            started = time.monotonic()
            completed = subprocess.run(
                [sys.executable, "-c", PIPELINE_RUNNER, video_id, str(archive)],
                cwd=PROJECT_ROOT,
                env=environment,
            )
            wall = time.monotonic() - started
            manifest_path = video_dir / "metadata" / "video_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {}
            )
        durations = task_seconds(manifest)
        row = {
            "run_id": run_id,
            "backend": args.backend,
            "video_id": video_id,
            "video_seconds": (manifest.get("video") or {}).get("duration_seconds"),
            "route": (manifest.get("route") or {}).get("pipeline_id"),
            "wall_seconds": round(wall, 1),
            "gpu_task_seconds": {
                task: round(durations[task], 1) for task in GPU_TASKS if task in durations
            },
            "succeeded": completed.returncode == 0,
        }
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"[benchmark] {json.dumps(row)}", flush=True)


if __name__ == "__main__":
    main()
