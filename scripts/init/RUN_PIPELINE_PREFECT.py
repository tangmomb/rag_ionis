from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import (
    create_markdown_artifact,
    create_progress_artifact,
    update_progress_artifact,
)

try:
    import RUN_PIPELINE_INIT as pipeline
except ModuleNotFoundError:  # Le depot historique utilise une casse differente selon l'OS.
    import run_pipeline_init as pipeline
from common.prefect_artifacts import artifact_records, command_argument, slugify


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
SAFE_ENV_OVERRIDES = {"PIPELINE_OPENAI_MODE", "PYTHONUTF8"}
PROGRESS_ARTIFACT_IDS: dict[str, Any] = {}

os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")
os.environ.setdefault("PIPELINE_PYTHON", str(DEFAULT_PIPELINE_PYTHON))


def video_id_from_command(command: list[str]) -> str:
    video_dir = command_argument(command, "--video-dir")
    if video_dir:
        return Path(video_dir).name
    video_url = command_argument(command, "--video-url")
    if video_url:
        return pipeline.extract_youtube_video_id(video_url) or "video-url"
    return "pipeline"


def retry_count_for_label(label: str) -> int:
    normalized = label.lower()
    if any(token in normalized for token in ("get data", "download", "openai", "upload", "update sql")):
        return 2
    if any(token in normalized for token in ("ocr", "whisper", "embedding", "classify")):
        return 1
    return 0


@task(log_prints=False, timeout_seconds=6 * 60 * 60)
def execute_pipeline_step(
    label: str,
    command: list[str],
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Commande: %s", " ".join(map(str, command)))
    started_at = time.perf_counter()

    environment = os.environ.copy()
    environment.update(env_overrides)
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        if cleaned:
            logger.info(cleaned)
    return_code = process.wait()
    elapsed_seconds = time.perf_counter() - started_at
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    published_artifacts = 0
    try:
        for record in artifact_records(label, command, elapsed_seconds):
            if record["progress"] is not None:
                progress_key = record["progress_key"]
                artifact_id = PROGRESS_ARTIFACT_IDS.get(progress_key)
                if artifact_id is None:
                    artifact_id = create_progress_artifact(
                        key=progress_key,
                        progress=record["progress"],
                        description=record["progress_description"],
                    )
                    PROGRESS_ARTIFACT_IDS[progress_key] = artifact_id
                else:
                    update_progress_artifact(
                        artifact_id=artifact_id,
                        progress=record["progress"],
                        description=record["progress_description"],
                    )
                published_artifacts += 1
            if record["publish_report"]:
                create_markdown_artifact(
                    key=record["key"],
                    markdown=record["markdown"],
                    description=record["description"],
                )
                published_artifacts += 1
    except Exception as exc:  # Les apercus ne doivent pas invalider un traitement reussi.
        logger.warning("Impossible de publier les artifacts Prefect: %s", exc)

    result = {
        "label": label,
        "return_code": return_code,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "published_artifacts": published_artifacts,
    }
    logger.info("Etape terminee en %.1f s; artifacts=%s", elapsed_seconds, published_artifacts)
    return result


def prefect_run_step(label: str, command: list[str], env: dict[str, str]) -> None:
    video_id = video_id_from_command(command)
    retries = retry_count_for_label(label)
    safe_overrides = {
        key: value
        for key, value in env.items()
        if key in SAFE_ENV_OVERRIDES and os.environ.get(key) != value
    }
    configured_task = execute_pipeline_step.with_options(
        name=slugify(label),
        task_run_name=f"{video_id} — {label}",
        retries=retries,
        retry_delay_seconds=30,
    )
    configured_task(label, [str(part) for part in command], safe_overrides)


@flow(name="RAG IONIS - Pipeline initial", log_prints=True)
def prefect_pipeline_flow() -> None:
    PROGRESS_ARTIFACT_IDS.clear()
    pipeline_python = Path(os.environ["PIPELINE_PYTHON"])
    if not pipeline_python.exists():
        raise FileNotFoundError(
            f"Python du pipeline introuvable: {pipeline_python}. "
            "Configure PIPELINE_PYTHON vers le venv GPU."
        )

    original_run_step = pipeline.run_step
    pipeline.run_step = prefect_run_step
    try:
        pipeline.main()
    finally:
        pipeline.run_step = original_run_step


@task(name="prefect-smoke-test")
def publish_smoke_artifact() -> str:
    artifact_key = "rag-ionis-prefect-smoke-test"
    create_markdown_artifact(
        key=artifact_key,
        markdown=(
            "# Prefect est connecté\n\n"
            "Le serveur, l'API, l'exécution locale et la publication d'artifacts fonctionnent."
        ),
        description="Test non destructif de l'integration Prefect",
    )
    return artifact_key


@flow(name="RAG IONIS - Test Prefect", log_prints=True)
def smoke_test_flow() -> str:
    return publish_smoke_artifact()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "smoke":
        smoke_test_flow()
        return
    prefect_pipeline_flow()


if __name__ == "__main__":
    main()
