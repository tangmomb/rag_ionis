from __future__ import annotations

import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
GPU_DOCKERFILE_PATH = ROOT_DIR / "Dockerfile.scaleway"
GPU_REQUIREMENTS_PATH = ROOT_DIR / "requirements-gpu.txt"
YTDLP_REQUIREMENTS_PATH = ROOT_DIR / "requirements-ytdlp.txt"
WORKER_SCRIPT_PATH = ROOT_DIR / "deploy" / "rag-ionis-scaleway-worker.sh"
PUBLISH_SCRIPT_PATH = ROOT_DIR / "deploy" / "publish-scaleway-image.sh"
SHARED_PUBLISH_SCRIPT_PATH = ROOT_DIR / "deploy" / "publish-image.sh"
SCALEWAY_DEPLOY_SCRIPT_PATH = ROOT_DIR / "deploy" / "deploy-scaleway-worker.sh"
SCALEWAY_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "deploy-scaleway.yml"
WORKER_ENV_EXAMPLE_PATH = ROOT_DIR / "deploy" / "scaleway-worker.env.example"


class ScalewayConfigurationTests(unittest.TestCase):
    def test_gpu_worker_has_supported_node_and_persistent_model_cache(self) -> None:
        dockerfile = GPU_DOCKERFILE_PATH.read_text(encoding="utf-8")
        worker_script = WORKER_SCRIPT_PATH.read_text(encoding="utf-8")
        worker_env = WORKER_ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

        self.assertIn("node:22.18.0-bullseye-slim", dockerfile)
        self.assertNotIn("ffmpeg nodejs ca-certificates", dockerfile)
        self.assertIn("dst=/models", worker_script)
        self.assertIn("SCALEWAY_MODEL_CACHE_DIR=/var/lib/rag-ionis/models", worker_env)

    def test_image_publisher_uses_remote_build_cache(self) -> None:
        publisher = PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")
        shared_publisher = SHARED_PUBLISH_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("rev-parse --short=7 HEAD", publisher)
        self.assertIn('publish-image.sh" scaleway', publisher)
        self.assertIn("docker buildx build", shared_publisher)
        self.assertIn('--tag "${commit_image}"', shared_publisher)
        self.assertIn('--tag "${latest_image}"', shared_publisher)
        self.assertIn('--cache-from "type=registry,ref=${cache_image}"', shared_publisher)
        self.assertIn('--cache-to "type=registry,ref=${cache_image},mode=max"', shared_publisher)
        self.assertIn("--push", shared_publisher)

    def test_deployment_script_supports_healthcheck_and_rollback(self) -> None:
        deployment = SCALEWAY_DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("docker pull", deployment)
        self.assertIn("--gpus all", deployment)
        self.assertIn("rolling back", deployment)
        self.assertIn("Waiting for the current GPU worker", deployment)

    def test_workflow_uses_registry_cache(self) -> None:
        workflow = SCALEWAY_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("Dockerfile.scaleway", workflow)
        self.assertIn("cache-to: type=registry", workflow)

    def test_gpu_requirements_do_not_include_api_or_database_stack(self) -> None:
        requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertNotIn("fastapi", requirements)
        self.assertNotIn("uvicorn", requirements)
        self.assertNotIn("psycopg", requirements)
        self.assertNotIn("pgvector", requirements)
        self.assertNotIn("cohere", requirements)

    def test_ytdlp_is_installed_after_expensive_gpu_dependencies(self) -> None:
        dockerfile = GPU_DOCKERFILE_PATH.read_text(encoding="utf-8")
        gpu_requirements = GPU_REQUIREMENTS_PATH.read_text(encoding="utf-8")
        ytdlp_requirements = YTDLP_REQUIREMENTS_PATH.read_text(encoding="utf-8")

        self.assertNotIn("yt-dlp", gpu_requirements)
        self.assertIn("yt-dlp==", ytdlp_requirements)
        self.assertLess(
            dockerfile.index("COPY --from=paddle-builder /paddle-parts/libphi_gpu.so"),
            dockerfile.index("COPY requirements-ytdlp.txt ./"),
        )
        self.assertLess(
            dockerfile.index("COPY requirements-ytdlp.txt ./"),
            dockerfile.index("COPY pipeline ./pipeline"),
        )


if __name__ == "__main__":
    unittest.main()
